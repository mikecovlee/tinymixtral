# CPT v1.3：仅长期状态的 Token-Parallel Router

## 版本身份

- 发布名：`cpt_v1.3`
- 本地分支：`cpt_v1.3`
- 直接父版本：`cpt_v1`
- 父版本基线 SHA：`a68b8651f09d277eb1ff2604a7ec6f09f4d66dae`
- 版本关系：`cpt_v1.3` 与 `cpt_v1.2` 平级；两者分别从 `cpt_v1` 派生，`cpt_v1.3` 不包含 `cpt_v1.2` 的性能改造。
- Router 配置版本：`cpt_router_version = 3`
- Router 算法版本：`CPT_ROUTER_ALGORITHM_VERSION = 3`
- 实现范围：仅 Native TinyMixtral；本版本没有 HF Router 实现或 Native/HF 等价性声明。
- 交付状态：只在本地 worktree 生成，未提交，未推送，未创建 Pull Request。

## 数学定义

全部数学采用列向量约定。对一个 batch 中所有有效 token，将其列向量组成
`X ∈ R^{d×T}`：

\[
\mathbf Z=\operatorname{ColNorm}_{\epsilon_z}(\mathbf P\mathbf X),
\qquad
\overline{\mathbf A}=\operatorname{ColNorm}_{\epsilon_m}(\mathbf A),
\]

\[
\mathbf Q=
\operatorname{softmax}_{\mathrm{col}}
\left(\frac{\overline{\mathbf A}^{\mathsf T}\mathbf Z}{\tau_p}\right),
\]

\[
\mathbf B=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{
\boldsymbol\Theta_C\mathbf H_N-
\mathbf 1_K\operatorname{sg}(\boldsymbol\lambda)^{\mathsf T}
}{\tau_e}
\right),
\qquad
\boldsymbol\Pi=\mathbf B^{\mathsf T}\mathbf Q.
\]

代码按 token 行主序存储，因此最后一步等价实现为
`Pi.T = Q.T @ B`。`Pi` 之后不再施加 softmax。

## 删除内容

本版本只删除服务于序列内短期状态递推的内容：

- `short_state` / \(\mathbf S\)；
- `responsibility` / \(\boldsymbol\nu\)；
- `beta`、状态梯度、状态投影和责任质量更新；
- 按 `position` 扫描 `sequence_length` 的 Router 串行循环；
- 仅服务上述递推的配置项：`cpt_rho_beta`、`cpt_beta_max`、`cpt_kappa_beta`、`cpt_lambda_sa`、`cpt_state_step_size`、`cpt_state_radius`。

因此 Router 前向对 token 并行，当前 token 的 Router 概率不依赖同一序列的历史 token；前向中不存在序列局部可变状态。

## 保留内容

- strict-v1 的 `P X -> ColNorm` 投影路径；没有引入 projection-softmax；
- 长期可学习量 `projection`、`anchors`、`energy`（`energy` 对应 \(\boldsymbol\Theta_C\)）；
- 前向中的 `ColNorm_eps_m(A)`，以及 optimizer 成功后提交时的 anchor 列归一化；
- 长期拥塞价格 `congestion_price`（\(\boldsymbol\lambda\)）及其 soft-load proposal 更新；
- FP32 Router 精度岛；
- `Pi = B.T @ Q`、唯一的 `Q.T @ B` 核心乘法、无 post-`Pi` softmax；
- 原有 Top-2 选择与选中权重归一化；
- proposal、事务提交、回滚、`state_version`、`optimizer_step`；
- mask、all-padding、padding NaN 隔离；
- checkpoint/resume 的严格校验与 fail-closed 行为。

## Checkpoint 边界

`cpt_v1.3` 只接受算法版本缓冲区为 `3`、配置版本为 `3` 且 CPT 配置 schema 完整匹配的 checkpoint。以下 checkpoint 必须明确拒绝，不能静默迁移或部分加载：

- `cpt_v1`：算法版本 `1`，包含旧的短期状态配置语义；
- `cpt_v1.2`：仍为算法版本 `1`，保留短期状态串行；
- projection-softmax `cpt_v2`：算法版本 `2`，投影概率语义不同；
- 任何含未知旧 CPT 字段、缺少 v1.3 CPT 字段、Router 状态键不完整或算法版本非 `3` 的 checkpoint。

旧 checkpoint 如需用于 v1.3，只能进行显式、独立、可审计的转换；本版本不提供隐式兼容层。

## 验证命令

以下命令从
`D:\Codex\CPT-MoE Experiment\version_worktrees\cpt_v1.3`
运行：

```powershell
$python = 'D:\Anaconda\envs\sage-moe-py310\python.exe'

& $python scripts/audit_projection_l2_v1.py --device cpu
& $python scripts/audit_projection_l2_v1.py --device cuda

$env:CUDA_VISIBLE_DEVICES = '-1'
try {
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "CPU-only pytest failed with exit code $LASTEXITCODE"
    }
} finally {
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "CUDA-visible pytest failed with exit code $LASTEXITCODE"
}

git -c safe.directory='D:/Codex/CPT-MoE Experiment/version_worktrees/cpt_v1.3' diff --check
git -c safe.directory='D:/Codex/CPT-MoE Experiment/version_worktrees/cpt_v1.3' status --short --branch
git -c safe.directory='D:/Codex/CPT-MoE Experiment/version_worktrees/cpt_v1.3' diff --stat a68b8651f09d277eb1ff2604a7ec6f09f4d66dae
```

审计脚本必须输出 `status = "passed"`，并显式报告：

```json
{
  "sequence_local_state": false,
  "token_parallel": true
}
```

这些检查证明本地工程实现、数值公式和版本边界符合本文件；它们不构成收敛、质量、专家专门化或性能提升结论。
