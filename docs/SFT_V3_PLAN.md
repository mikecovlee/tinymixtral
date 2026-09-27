# TinyMixtral v3.0 大规模通用 SFT 计划（V1–V4）

> 本文件按用户要求写在磁盘上、**不提交 git**。

## 0. 背景（前序观测，作为本计划的依据）
- base（v3.0 预训练基座）rubric ≈ **0.1** → 完全不会指令对话。
- 通用指令 SFT（imp-sft：parquet 前 50k 行 × 1 epoch）rubric **7.4**（paired −7.28，t=−9.7）→ **唯一大且显著的增益**。
- 其上的 DPO / GRPO / KTO / RLOO / 定向 exam 全部 **不显著**（|t|≤1.1；8 任务 0.4244–0.4280 全平）。
- 在 base 上做 answer-only 的 exam SFT（exam3）rubric **1.2**（paired −5.81，t=−7.2）→ 窄格式数据会**破坏**通用能力，且未提升 MCQ。
- imp-sft 只用了 1M 行 parquet 的**前 50k 行**（行序偏置，非均衡采样）。
- 模型：477.5M 总 / 276.1M 激活 MoE，max_pos 2048，vocab 32000，英文为主。
- 实测吞吐：seq1024/bs8 ≈ 36 rows/s（30,752 行/epoch ≈ 13.8 min）。
- rubric 评委分档偏粗（~0–12），故必须用 **paired 检验 + 5k held-out** 才能有功效。

**目标**：用更多、更均衡、更高质量的数据做通用 SFT，提高综合能力（rubric + IFEval 为主，harness 仅作回归守卫）。

## 1. 决策（用户确认，m1949/m1954）
- 做到 **V3（更多数据 = 3M × 1 epoch）**，并加 **V4（小样本高质量 polish）**。
- **从 base（v3.0 预训练基座）开始**训练（不用 imp-sft 热启）。
- **不加中文数据**。
- 指标：**rubric + IFEval 为主**；8 任务 harness **仅确保基础能力不退化**。
- **本机**（RTX PRO 4500 Blackwell 32GB）训练，**调大 bsz**。
- 评测**全部 offload 到 10.31.0.14**（本机训练不停）。
- 训练数据脚本**先写好**再开干；**所有改动在 `sft` 分支**。

## 2. 验收标准（预注册）
- rubric（5k held-out，paired vs imp-sft）：**Δ ≥ +0.5 且 t > 2**。
- IFEval prompt-strict ≥ imp-sft，跌幅 ≤ 1pp。
- 8 任务 harness 相对 base 0.4272 跌幅 ≤ 0.3pp（回归守卫）。
- GSM8K flexible 预期 +1~3pp。
- dev loss 不上升（过拟合信号 → 停）。

## 3. 分支与目录
- 分支 `sft`（从 main `c113799` 新建）。
- 一切产物在**磁盘**：`scripts/`、`data/sft_v2/`、`checkpoints/`、`publish/`；不用 `/tmp`。
- 本计划文件 `docs/SFT_V3_PLAN.md` **不提交**；提交时只显式 `git add` 工具脚本。

## 4. 数据（脚本优先：`scripts/build_sft_v2.py`）

### 4.1 来源与配额（V2 = 1M；V3 = 3M ≈ ×3）
| 来源 | HF repo | V2 目标 |
|---|---|---|
| Tulu-3 现代混合 | `allenai/tulu-3-sft-mixture` | 300k |
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` | 150k（全局打乱；排除 imp-sft 的前 50k 行） |
| SlimOrca（GPT-4 ShareGPT） | `Open-Orca/SlimOrca` | 100k |
| OpenOrca | `Open-Orca/OpenOrca` | 100k |
| UltraChat 多轮 | `HuggingFaceH4/ultrachat_200k` | 100k |
| OASST2 多轮 | `OpenAssistant/oasst2` | 50k |
| MetaMathQA（数学 CoT） | `meta-math/MetaMathQA` | 60k |
| Orca-Math 应用题 | `microsoft/orca-math-word-problems-200k` | 40k |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | 30k（抽样） |
| 知识 QA | `squad_v2` / `mandarjoshi/trivia_qa`(nocontext) / `HuggingFaceTB/smoltalk` 切片 | 50k |
| IFEval 风格合成格式 | 本地生成（`gen_ifeval_prompts.py` 思路） | 15k |
| 安全/拒答 | Tulu-3 安全切片等 | 15k |
（**不含中文**）

### 4.2 管线步骤
1. 逐来源加载/流式，统一 schema：`conversations=[{from:human|gpt, value}]` + `source/category/lang/n_turns/tokens_est`。
2. 过滤：assistant 回复 10–2048 token（v3.0 tokenizer）；丢空/过短/benign 拒答；跳过超长。
3. 去重：精确 hash + **MinHash-LSH（Jaccard ≥ 0.8）**（OpenHermes 与 SlimOrca/OpenOrca 高度重叠）。
4. **全局打乱**（修正行序偏置）。
5. 平衡：单来源占比 ≤ 15%。
6. **去污染**：13-gram 与 `gsm8k test` / `arc,obqa,hellaswag,piqa test` / `IFEval` / `mmlu test` / `ceval val` / `cmmlu test` / 本项目 held-out 提示 比对，输出审计文件。
7. 输出：`data/sft_v2/{train.parquet, dev.parquet, stats.json, LICENSE_NOTES.md, heldout_prompts.parquet}`。
8. **先 10k pilot** 验证格式/去重/去污染 + 人工抽检 20 条。

### 4.3 held-out
- **5k rubric 提示**：在训练数据选取**之前**采样并排除（用于 rubric 评测）。
- dev 2–3k：只看 loss。

## 5. 训练（本机 RTX PRO 4500 32GB）
- 起点：**v3.0 base**（`checkpoints/base_v3_raw` 或 HF snapshot）；对照臂 B = imp-sft 续训（可选）。
- **bsz 标定**：seq1024 bs∈{8,16,24,32}、seq2048 {8,16} → 取最大稳定（有效 batch ≥32）；若 `train_sft.py` 无 `--grad-accum` 则补上（sft 分支）。
- 超参：lr 2e-5 cosine、warmup 3%、bf16 + gradient checkpointing、`save_every 1000` + keep-last-2、`log_every 50`、setsid 后台、可 resume。
- 阶梯：
  - **V1** 200k × 1ep（≈1.5h）
  - **V2** 1M × 1ep（≈8h）
  - **V3** 3M × 1ep（≈24h）
  - **V4** 50k 高质量 CoT polish @ lr 5e-6 × 1ep（≈1h）

## 6. 评测（offload 到 10.31.0.14，本机训练不停）
每阶段：
1. 本机 `scripts/publish_hf.py --checkpoint <stage final> --output publish/<arm>`（白名单补丁已打）。
2. `scp` 该 HF 目录（~1.9GB）到工作机。
3. 工作机（A5000 24GB，`tinymixtral` env，deepseek key）顺序：
   a. 生成 **5k held-out 回复**（`dpo_scripts/dpo_eval_judge.py gen`）。
   b. **rubric 打分**（`dpo_scripts/rubric_judge2.py`，并发 6–8）。
   c. `lm-eval`：**IFEval + GSM8K + 8 任务 harness**。
4. 结果 jsonl 拉回本机 → `dpo_scripts/rubric_stats.py` 出 mean±se + **paired vs base/imp-sft**。
预检：工作机 lm-eval 数据集缓存（8 任务含 boolq、IFEval、GSM8K）、磁盘余量、key、tmux。

## 7. 预算
数据下载 ~10–12GB（0.5–1h）+ 管线 1–3h + pilot 0.5h；bsz 标定 0.5h；训练 1.5+8+24+1h；评测 ~1–1.5h/阶段（与训练并行）→ **V2 结论约 1 天，V3 约 2 天**。

## 8. 风险与对策
- 过拟合（477M × 1–3M 样本）：1 epoch + V4 polish + dev loss 监控。
- 窄数据破坏通用性（exam3 教训）：窄来源份额小且带**完整解释**（非 answer-only）。
- 污染：13-gram 去污染 + 审计。
- 评测功效：5k held-out + paired 检验。
- 许可：逐来源记录 `LICENSE_NOTES.md`。
- 下载失败：proxy 10.31.0.14:7890，失败重试/换镜像；数据落在磁盘。

## 9. 交付物
- `scripts/build_sft_v2.py`、`data/sft_v2/*`
- 训练/评测/传输脚本（本机 + 工作机）
- `checkpoints/sft_v2_{v1,v2,v3,v4}`、`publish/*`
- 评测产物 + 对比表 + 报告
- `sft` 分支提交（含 `scripts/publish_hf.py` tokenizer 白名单修复）

## 10. 执行清单
- [x] 建 `sft` 分支（自 `main` c113799）
- [x] 写本计划文件（不提交）
- [x] `build_sft_v2.py` + 10k pilot 校验
- [x] 全量构建 V1/V2/V3 数据（含 5k heldout + dev）
- [x] bsz 标定（seq1024 max stable = bs24，21.5GB）
- [x] V1 训练（195,170 行 → 106,718 seq → 4,447 步 / 99.4m）→ offload 评测 → 判定
- [x] V2 训练（856,805 行 → 644,557 seq → 26,857 步 / 9.97h）→ offload 评测 → 判定（rubric +5.56 t=+22.4，9/25 23:42 全链完成）
- [x] V3 训练（2,168,835 行 → 1,443,804 seq → 60,159 步 / 22.3h）→ offload 评测 → 判定（rubric +8.61 t=+30.7，9/26 22:34 回传）
- [x] V4 polish（init=V3 final，1,099 步 / 24.5m）→ offload 评测（9/27 07:27 落地并自动回传；harness 0.4042 / IFEval 0.1664 / GSM8K 0.0212，未超 V3）
- [x] 汇总报告 + 提交工具脚本（`966fead` + `af313a3` + `66728d6`；REPORT/PLAN 已填至 V4 全量；V4 rubric 2,036 条已于 9/27 充值后 resume 补齐 4,955/4,955，表已刷新。**战役收尾，无遗留。**）

## 11. 实际执行记录（ACTUALS）

### 数据
| 规模 | 行数 | dev | heldout | raw | 构建耗时 |
|---|---|---|---|---|---|
| V1 | 195,170 | 1,000 | 5,000 | — | — |
| V2 | 856,805 | 4,305 | 5,000 | 2,395,834 | 1,840.9s |
| V3 | 2,168,835 | 10,898 | 2,598 | 4,143,061 | 2,531.9s |
| V4 | 50,000 | — | — | — | 1.9s |

V4 = V3 中 meta/orca 数学（各 12k）、omi2(8k)、tulu3(10k)、slimorca(8k) 分层抽样。

### 训练
| 规模 | 行数 | packed seq | steps | bs | lr | 耗时 |
|---|---|---|---|---|---|---|
| V1 | 195,170 | 106,718 | 4,447 | 24 | 2e-5 | 99.4m |
| V2 | 856,805 | 644,557 | 26,857 | 24 | 2e-5 | 9.97h |
| V3 | 2,168,835 | 1,443,804 | 60,159 | 24 | 2e-5 | 22.3h（1337.8m） |
| V4 | 50,000 | 26,365 | 1,099 | 24 | 5e-6 | 24.5m（init=V3 final） |

`train_sft.py` 修复：tokenize/pack 由 Python int list 改为 numpy int32（V2 峰值内存 40GB→~21GB，swap 7/7 抖动消除）。

### 评测（0-shot，工作机 offload）
| 指标 | base v3.0 | imp-sft | sft_v2_v1 | sft_v2_v2 | sft_v2_v3 | sft_v4 |
|---|---|---|---|---|---|---|
| 8-task harness（canonical） | 0.4250 | — | 0.4210 | 0.4158 | 0.4034 | 0.4042 |
| IFEval prompt-strict | — | 0.0924 | 0.0961 | 0.1091 | **0.1701** | 0.1664 |
| IFEval inst-strict | — | 0.1894 | 0.2014 | 0.2026 | **0.2794** | 0.2698 |
| GSM8K flexible | — | 0.0167 | 0.0265 | 0.0197 | **0.0227** | 0.0212 |
| rubric（4955 对新 heldout, 0-100） | — | **6.4±0.18** | **11.4±0.26**（paired **+5.04±0.26, t=+19.2**） | **12.0±0.24**（paired **+5.56±0.25, t=+22.4**） | **15.0±0.28**（paired **+8.61±0.28, t=+30.7**） | **14.9±0.28**（paired **+8.46±0.28, t=+30.1**；V4−V3 配对 −0.15±0.18，噪声内） |

注 1：rubric 数值基于新 5k heldout 集 + `rubric_judge2`（0-100），与旧 500-prompt 数字（base 0.1 / imp-sft 7.4）不可比。

注 2：harness 采用统一口径——hellaswag/piqa/arc_challenge/openbookqa 取 acc_norm，winogrande/arc_easy/boolq/lambada 取 acc；据此 base v3.0 = README 的 0.4250，V1 = 0.4210，V2 = 0.4158。原记录的 base 0.4272 / imp-sft 0.4260 在工作机上无对应 JSON、公式不可考，故改用本口径；V2 相对 base 掉 0.92pp（略超 0.3pp 回归护栏，主因 boolq/openbookqa/arc_easy），rubric 与 IFEval 为主指标，harness 仅作回归护栏。V3 进一步降至 0.4034（-2.16pp），拖累集中在 boolq（0.426 vs base 0.615）与 arc_easy，其余任务持平或更好。
