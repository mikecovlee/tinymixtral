# SFT V3 完整复现指南（base → sft_v2_v3）

本文档给出从零复现 **sft_v2_v3**（SFT V3/V4 战役交付臂）的全部命令与参数。所有脚本均在
分支 `sft` 上（commits 966fead / af313a3 / 66728d6 及本次文档提交）。

## 0. 环境

- **训练机**：Linux，单卡 ≥24GB（实测 RTX PRO 4500 32GB），60GB RAM。
  conda env `tinymixtral-cudev`：torch 2.14.0+cu130、transformers 4.57.6、pyarrow、numpy、safetensors 0.8.0、lm_eval 0.4.12。
- **评测机（offload）**：Windows，24GB 卡（实测 A5000），conda env `tinymixtral`，
  DeepSeek API key 位于 `%USERPROFILE%\.local\share\opencode\auth.json`（`deepseek` 项）。
  仓库路径 `C:\Users\<user>\tinymixtral-improve`，长任务一律 tmux + 落地 .ps1（不要内联嵌套引号）。
- **基座**：v3.0 base。两种来源任选：
  - HF：`~/.cache/huggingface/hub/models--mikecovlee--tinymixtral/snapshots/6e0792c1781d3c704c9f9a3844662998795b306c/`
    （即公开仓库 `mikecovlee/tinymixtral` main revision；tokenizer 也取自这里，脚本内 SNAP 变量即此路径）
  - 原始：`checkpoints/base_v3_raw/`（含 `config.json` + `pytorch_model.bin`）
- HF 下载走代理：`export HTTPS_PROXY=http://10.31.0.14:7890`（评测机上的 7890 端口）。

## 1. 数据构建（scripts/，训练机上跑）

```bash
python scripts/prefetch_sft_v2.py --out data/sft_v2_src          # 10 个源全量落盘（HF 下载，量大）
python scripts/build_sft_v2.py --out-dir data/sft_v2_v1 --scale v1   # 目标 200k -> 实得 195,170 行
python scripts/build_sft_v2.py --out-dir data/sft_v2_v2 --scale v2   # 目标 1M   -> 实得 856,805 行
python scripts/build_sft_v2.py --out-dir data/sft_v2_v3 --scale v3   # 目标 3M   -> 实得 2,168,835 行
```

要点（`build_sft_v2.py`）：
- 配额份额 SHARES：tulu3 .32 / openhermes .16 / slimorca .12 / openorca .12 / ultrachat .11 /
  metamath .06 / orcamath .04 / omi2 .03 / squad2 .02 / trivia .02；seed 42；无中文数据。
- 过滤：assistant 回复 40–12,000 字符（近似 10–2048 token）；每源占比 ≤15%。
- 去重：精确哈希 + MinHash-LSH（Jaccard ≥ 0.8）。
- 去污染：对 gsm8k / arc / openbookqa / hellaswag / piqa / ifeval / mmlu / ceval(cmmlu) 及
  held-out 集做 n-gram 去污染（`--decontam` 默认开）。
- 产出：`train.parquet / dev.parquet / heldout_prompts.parquet / stats.json / LICENSE_NOTES`。
- 运行时长参考：v2 约 40 分钟、v3 约 42 分钟（stats.json 内有 elapsed_s）。
- 评测用的固定 prompt 集已随仓库提供：**`eval_prompts/heldout_prompts_id_1k5.parquet`**
  （4,955 条，列 `id/prompt/ntok`，sha256 `6f8e48e67c592f830615ecc942b46aee3090eec3f827838227db0a5c12b2fdc8`）。
  clone 后 scp 到评测机 `data/sft_v2_v1/` 下（战役各臂共用同一份，保证 paired 可比）。

## 2. 训练（dpo_scripts/run_sftvN.sh，从 base 起训，非 warm-start）

```bash
bash dpo_scripts/run_sftv1.sh   # 195,170 行 -> 106,718 条 1024-seq -> 4,447 steps，~99 分钟
bash dpo_scripts/run_sftv2.sh   # 856,805 行 -> 644,557 seqs -> 26,857 steps，~10.0 h
bash dpo_scripts/run_sftv3.sh   # 2,168,835 行 -> 1,443,804 seqs -> 60,159 steps，~22.3 h
```

统一超参（train_sft.py）：`--epochs 1 --seq-len 1024 --lr 2e-5`（cosine + 3% warmup）
`--batch-size 24 --wd 0.1`、bf16 autocast、梯度检查点、`--save-every 1000 --log-every 100`，
输出 `checkpoints/sft_v2_v{1,2,3}/step_*_final`。

注意（经验教训 8.2）：
- `scripts/train_sft.py` 已含 **numpy int32 内存补丁**（tokenize/pack 不再是 Python int list）；
  否则 V2/V3 会在 `Packing...` 处内存爆到 40GB+ 触发 swap 抖动。补丁前 RSS ~40GB → 补丁后 ~21GB。
- 训练后期进程 RSS 仍有 ~42GB 是 glibc arena 滞留，属正常，勿误杀。
- 实测吞吐 ~0.745 steps/s（seq1024, bs24）。长跑建议外层 runner 打印 `SFTV<N>_DONE` 标记，
  配合 queue_*.sh 等待器可实现无人值守级联（战役实际就是这么跑的）。

## 3. 评测链（发布 → 传输 → 评测机 gen/rubric/lm-eval → 回传汇总）

```bash
bash dpo_scripts/offload_arm.sh imp-sft-v2-v3 sftv2v3 checkpoints/sft_v2_v3/step_0060159_final
#   -> publish/imp-sft-v2-v3（pytorch_model.bin + 白名单 tokenizer 文件，断言无 model.safetensors）
#   -> scp 到评测机 publish/，并打印 tmux 启动命令
tmux new-session -d -s sftv2v3 "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\run_offload_arm.ps1 -Arm imp-sft-v2-v3 -Tag sftv2v3 *> <repo>\logs\offload_sftv2v3.log"
#   评测机依次执行：GEN3 生成（dpo_eval_judge.py gen，4,955 条 held-out）-> RUBRIC3
#   （rubric_judge2.py，deepseek-flash 0-100，4 维，concurrency 8）-> HARNESS / IFEVAL / GSM8K（lm_eval）
#   完成标记 OFFLOAD_ARM_DONE；本地可用 queue_v3_pull.sh 式等待器自动 scp 回 data/dpo/
python dpo_scripts/final_table.py --dir data/dpo          # 汇总 markdown 对比表（rubric paired t 检验 + 三项 lm-eval + canonical harness）
python dpo_scripts/summarize_evals.py --dir data/dpo --detailed   # 分任务明细
```

口径（预注册，方法学 8.3）：
- **rubric**：同一 4,955 条 held-out prompt、同一裁判（rubric_judge2 + deepseek-flash），
  与对照臂做 **per-item paired** t 检验。
- **harness canonical 公式**：hellaswag/piqa/arc_challenge/openbookqa 取 acc_norm，
  winogrande/arc_easy/boolq/lambada 取 acc，8 项简单平均（base v3.0 = 0.4250）。
- rubric_judge2 失败重试 4 次后**静默跳过**——跑完必须核对行数 = prompt 数（V4 曾缺 2,036 行，
  根因 API 402；修复方式 `--resume` 只补缺失 id）。

## 4. 预期结果（v3.0 base → sft_v2_v3）

| 指标 | base v3.0 | imp-sft | sft_v2_v1 | sft_v2_v2 | **sft_v2_v3** |
|---|---|---|---|---|---|
| rubric（4955 paired vs imp-sft） | 未测（新集） | 6.40±0.18 | 11.44±0.26（+5.04, t=+19.2） | 11.97±0.24（+5.56, t=+22.4） | **15.01±0.28（+8.61, t=+30.7）** |
| IFEval prompt/inst-strict | — | 0.0924/0.1894 | 0.0961/0.2014 | 0.1091/0.2026 | **0.1701/0.2794** |
| GSM8K strict/flexible | — | —/0.0167 | 0.0159/0.0265 | 0.0174/0.0197 | 0.0205/**0.0227** |
| 8-task harness（canonical） | 0.4250 | 0.4260* | 0.4210 | 0.4158 | 0.4034（boolq 0.4263 为主要拖累） |

\* imp-sft 0.4260 为战役前记录值，口径不可考，仅供参考。

已知代价：数据规模化显著提升指令跟随/开放质量（rubric、IFEval），但基础判别任务回退，
boolq 最敏感（0.615→0.426）。V4 polish（50k、lr 5e-6、init=V3）经验证无增益（paired −0.15,
t=−0.8），复现 V3 无需执行。详见 `docs/SFT_V3_REPORT.md`（含 §8 经验教训）。

## 5. 一键级联参考

战役实际以 marker+waiter 无人值守运行：run_sftv2.sh →（SFTV2_DONE）→ queue_sftv3.sh →
run_sftv3.sh →（SFTV3_DONE）→ queue_v3_eval.sh（publish+scp+tmux）→ queue_v3_pull.sh
（回传 + final_table）。复现时可逐级手动执行，或仿写 queue 脚本串联。
