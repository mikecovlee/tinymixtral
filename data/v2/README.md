# Data v2 — SmolLM 混合时代

v1.1 起引入 SmolLM 系语料后的数据配比。**完整复现**所需配方如下（tokenized 数据在 NAS / 远端，不入库）。

一键重建：`python data/v2/prepare_pretrain.py` / `python data/v2/prepare_posttrain.py`（subprocess 驱动 data/pipeline/ 工具，--dry-run 预览命令链）。

## 配比

| 阶段 | 内容 | 规模 | LR | 构建 |
|---|---|---|---|---|
| 预训练 | smollm_blend：FineWeb-Edu-10B + Cosmopedia-v2（36:4，`--weights 36 4`，`smollm-cosmo-tiny`） | 4.00B | 2e-3 cosine | `prepare_data_local.py` |
| 后训练 | knowledge_blend：Wiki + FineWeb-Edu 抽样 | 1.03B | 1e-3 cosine | 同 v1 后训练 |

注：v2.0-beta 记录的预训练比例为 89:11（`smollm-cosmo-tiny-2`，88.8%/11.2%）——与 36:4 同源配方的另一取样档，复现时以版本卡为准。

## 用它的模型

- `config/v1.1/`、`config/v1.1-1b/`（v1 结构 / 放大版）
- `config/v2.0-beta/`（v1 结构 + 常驻共享专家）
