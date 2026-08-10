# CPT Router 性能优化与注意力上下文探索记录

本文记录 blockwise CPT Router 的三轮性能优化(数学等价重构 → torch.compile → 预测-校正器)、一次被量化否决的方向(复用注意力上下文),以及过程中的方法论沉淀。

所有速率为单层 router、b=4、seq=2048、proj_dim=128、CUDA 实测;精度为相对 strict v1(chunk=1)的路由概率误差,proj_dim=16、seq=64、5 随机种子统计。

## 一、起点:瓶颈定位

blockwise 路由的 chunk 循环是 Python 串行循环,每 chunk 约 20 个 kernel launch,CPU 发射延迟主导(非 FLOPs):

| 方案 | 耗时 |
|---|---|
| main 的 Linear gate | 0.17 ms |
| CPT chunk=1(严格 v1) | 2493 ms |
| CPT chunk=32(当时默认) | **82.8 ms** |
| CPT chunk=128 | 20.5 ms |

耗时与 chunk 数成正比——优化方向就是减少循环迭代数、压缩每迭代的开销。

## 二、优化一:数学等价重构(~1.2×)

严格保持数学语义的等价改写,全部通过**位级一致**验证(重构前后捕获输出/梯度逐 bit 对比):

| 改动 | 原实现 | 新实现 |
|---|---|---|
| 收缩运算 | `einsum("btd,bdk->btk")` | `torch.bmm` |
| 后缀有效计数 | `flip→cumsum→flip→sub`(6 kernel) | `count - cumsum_inclusive`(3 kernel) |
| 无效 token 置零 | `where + zeros_like` | 乘 bool mask |
| 半径裁剪 | `maximum(ones_like, norm/r)` | `clamp_min(norm/r, 1.0)` |
| 结构 | 循环体内联 | 抽出 `_route_chunk` 方法 |

82.8 → 66.6 ms。**经验:重构前先捕获参考输出,位级一致是数学等价重构的硬验收标准。**

## 三、优化二:torch.compile(5-6×)

对整个 `_forward_impl` 做 `torch.compile`,inductor 将 chunk 循环静态展开为单张融合图,消除全部循环/发射开销。

**门控设计**(`forward` 包装层):

| 条件 | 行为 | 原因 |
|---|---|---|
| CUDA + 训练模式 + chunk 数 ≤ 128 | 编译 | 收益区间 |
| eval 模式 | eager | eval 输入长度可变,避免重编译风暴 |
| CPU | eager | 测试路径确定性 |
| chunk 数 > `_MAX_COMPILED_CHUNKS=128` | eager | 编译时间 ~0.7s/chunk 线性增长,chunk=1+长序列会挂死 |

**关键事实**:
- 编译缓存跨 router 实例共享(第二个实例 0.1s),全模型只付一次 warmup(64 chunks ≈ 46s,训练首步摊销)
- `_apply`(设备迁移)清空编译缓存
- 编译引入的数值偏差 ≤ 3.4e-7(kernel 融合改变归约顺序),远低于 2e-6 验收容差

66.6 → 11.4 ms(chunk=32)。

## 四、优化三:预测-校正器(精度 ~5.5×,进一步提速)

**动机**:blockwise 的近似误差来自 chunk 内冻结入口状态(原型不随 token 演化)。

**机制**(两遍结构,全并行):
1. **Pass 1**:入口状态路由 → q₁(现有行为)
2. **轨迹估计**(前缀和闭式,无串行):
   - 状态漂移:`S_t = S₀ - η·Σ_{s<t} g_s`(exclusive cumsum)
   - 责任度:`ν_t = ρ^{p_t}·(ν₀ + Σ_{s<t} q_s·ρ^{-(p_s+1)})`(衰减前缀和闭式)
3. **Pass 2**:逐位置原型 `M_t = normalize(anchors·(1-β_t) + S_t·β_t)`,重新路由

**不变量**:`chunk_size=1` 时前缀和为空,严格退化 strict v1(测试位级验证)。这是所有变体必须保持的退化锚点。

配置:`cpt_state_corrector: bool = True`(默认开)。误差 8.2e-6 → **1.5e-6**。

校正器与更大 chunk 协同后反而更快(迭代更少):corrector + chunk=128 = **5.9 ms**,比优化前的默认配置又快又准。

## 五、最终成绩

| 方案 | 耗时 | vs strict v1 误差 |
|---|---|---|
| 起点(chunk=32) | 82.8 ms | 8.2e-6 |
| 重构 + compile(chunk=32) | 11.4 ms | 8.2e-6 |
| corrector + compile(chunk=32) | 21.7 ms | 1.5e-6 |
| **corrector + compile(chunk=128)** | **5.9 ms** | **1.5e-6** |

**14× 提速 + ~5.5× 精度提升**。部署建议:`cpt_state_chunk_size=128` + `cpt_state_corrector=True`(默认)。

## 六、方向 C 探索:复用注意力上下文(否决)

**动机**:router 的串行状态是瓶颈,且与 attention 部分冗余——router 输入本就是 post-attention 表征,上下文已被 attention 并行聚合过。

**关键约束**:`F.scaled_dot_product_attention` 不返回注意力权重(flash 后端永不物化 S×S)。字面复用 attention 权重 = 手算 `softmax(QKᵀ/√d)` + 放弃 flash,不可行。可行形态是 **router 在投影空间跑自己的 SDPA**(Q=z, K=z, V_k=q_k·z,k 个原型当 k 个头),复用 attention 富集表征 + 注意力机制,完全并行。

**设计**:γ∈[0,1] 混合递推轨迹与注意力上下文:`S_blend = (1-γ)·S_轨迹 + γ·C_注意力`。

**量化结果**:

| 变体 | mean err | compiled ms |
|---|---|---|
| corrector(γ=0) | **1.5e-6** | **5.9**(chunk=128) |
| AC γ=0.25 | 2.5e-5 | — |
| AC γ=0.5 | 5.1e-5 | — |
| AC γ=1.0 | 1.0e-4 | 9.4 |
| AC-par γ=1(全并行) | 1.0e-4 | 7.8 |

**结论:corrector 在精度和速度两个轴上全面支配,方向 C 否决。**

**机制解释**:strict v1 的状态是**梯度下降轨迹**;corrector 是它的一阶泰勒展开,结构忠实。注意力上下文是"按与当前 token 的相似度加权的历史平均"——与梯度轨迹是**不同的对象**,不構成逼近,误差随 γ 线性恶化。

**自校验方法**:否定性结论必须先排除实现 bug——γ=0 必须退化回 corrector(实测残差 4e-7,来自二阶差异),确认退化成立后,γ>0 的恶化才是真实的机制性结论。

## 七、方法论沉淀

1. **位级一致是数学等价重构的验收标准**:捕获参考输出 → 重构 → 逐 bit 对比,再谈性能。
2. **原型先行,量化后集成**:corrector 与注意力上下文都先在 /tmp 独立原型测精度/速度,达标才进仓库;不达标留下量化记录。
3. **负结果也是结果**:方向 C 的量化否决避免了复杂度增长,机制解释(梯度轨迹 vs 相似度加权)可复用于后续设计判断。
4. **退化不变量是设计锚点**:chunk=1 ≡ strict v1 贯穿所有变体,既保正确性又简化测试(退化测试位级断言)。
5. **快速路径要有持久测试**:torch.compile 路径曾零覆盖(CPU 测试永不触发),已补 `tests/test_cpt_router_cuda.py`(CUDA 门控:编译/eager 等价、门控规则、bf16 精度岛、checkpoint 交互、事务提交、设备迁移)。
6. **编译门控三要素**:模式门控(训练)、预算门控(chunk 数上限)、缓存失效(设备迁移)。

## 八、遗留与后续

- 以上验证全部针对**机制保真度与速度**,端到端 loss 收益待训练实验确认(见 `next-experiments.md`)
- 注意力上下文若仍要探索,候选形式:不做状态混合,改为 logits 修正项或调制责任度衰减——需新原型先行量化
- HF 导出(`hf/modeling_tinymixtral.py`)仍是旧 Linear gate,发布 CPT 模型前需同步
