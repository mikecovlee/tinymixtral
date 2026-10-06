# Data v1 — C4 时代

v1.0 首发使用的数据配比。**完整复现**所需配方如下（tokenized 数据在 NAS / 远端，不入库）。

一键重建：`python data/v1/prepare_pretrain.py` / `python data/v1/prepare_posttrain.py`（subprocess 驱动 data/pipeline/ 工具，--dry-run 预览命令链）。

## 配比

| 阶段 | 内容 | 规模 | LR | 构建 |
|---|---|---|---|---|
| 预训练 | C4-en（`c4en`） | 4.00B | 2e-3 cosine | `zst_jsonl_to_parquet.py` → `prepare_data.py`（40M/大块） |
| 后训练 | Wiki + 抽样 FineWeb-Edu | 1.03B | 1e-3 cosine | `prepare_data_local.py --pools wiki,webtext-sampled` |

配比数字见 `docs/DATA_LICENSES.md`「v1.0 post-train」行（Wiki 1.67% / FineWeb-Edu 98.33%）。

## 用它的模型

- `config/v1.0/`（6 专家 top-2，896/10 层，432M/176M）
