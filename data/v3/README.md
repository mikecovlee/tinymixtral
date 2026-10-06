# Data v3 — 六源四段时代

v3.0 系全部模型的数据配比。**完整复现**所需配方如下（tokenized 数据在 NAS / 远端，不入库）。

一键重建：`python data/v3/prepare_pretrain.py` / `python data/v3/prepare_sft.py`（subprocess 驱动 data/pipeline/ 工具，--dry-run 预览命令链）。

## 预训练：main_s1..s4（8.05B，4 段）

6 源混合（FineWeb-Edu / Cosmopedia-v2 / DCLM / OpenCodeInstruct / OpenWebMath / Wikipedia），逐段配方与构建命令见 `versions/v3.0/README.md`（`data/pipeline/make_blend_shards.py`）。

| 段 | 累计 token | LR | 数据池 |
|---|---|---|---|
| S1 | 2.00B | 5e-4 | main_s1 |
| S2 | 3.94B | 5e-4 | main_s2 |
| S3 | 6.14B | 4e-4 | main_s3 |
| S4 | 8.05B | 3e-4 | main_s4 |

## SFT：三档 + polish（`sft/`）

构建脚本在 `sft/`（build_dataset / prefetch_sources / sample_subset），数据集配方：

| 档 | 规模 | 配置 |
|---|---|---|
| 200k | 195k 对话 | `config/v3.0-it/200k.json` |
| 1M | 857k 对话 | `config/v3.0-it/1m.json` |
| 3M | 2.17M 对话 | `config/v3.0-it/3m.json` |
| polish | 50k 轻量微调 | `config/v3.0-it/polish.json` |

## 用它的模型

- `config/v3.0/`、`config/v3.0-it/`（4 专家 top-2）
- `config/v3.0-dense-276m/`、`config/v3.0-dense-477m/`（无路由稠密）
