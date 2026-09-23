# TinyMixtral v1.0 (legacy — C4 training)

The first TinyMixtral release, trained on C4-en. Kept for historical reference and
reproducibility; **superseded by v3.0**.

- HF: [`mikecovlee/tinymixtral-v1.0`](https://huggingface.co/mikecovlee/tinymixtral-v1.0) (~432M)

## Data preparation

```bash
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

python scripts/prepare_data.py --dataset allenai/c4 --subset en \
  --tokenizer tokenizer/ --output data/c4/tokenized \
  --max-tokens 4000000000 --force
```

## Pretrain (4B tokens)

Same trunk architecture as v1.1 (the data-quality ablation changed only the data). Config:
[`configs/config.json`](configs/config.json).

```bash
python scripts/train.py --config versions/v1.0/configs/config.json \
  --cache-dir data/c4/tokenized \
  --batch-size 22 --max-tokens 4000000000 --lr 3e-4 --warmup-steps 2000 \
  --keep-last-checkpoints 5 2>&1 | tee train.log
```

| Parameter | Value |
|-----------|-------|
| Data | C4-en |
| Batch size | 22 |
| Sequence length | 1,024 |
| Tokens/step | 22,528 |
| Steps | 177,557 |
| Learning rate | 3e-4 |
| Warmup steps | 2,000 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Time | ~77 h |

## Post-train (1B tokens)

Continue from the C4 checkpoint on higher-quality data (FineWeb-Edu + Cosmopedia v2, 50:50):

| Parameter | Value |
|-----------|-------|
| Data | FineWeb-Edu + Cosmopedia v2 (50:50) |
| Tokens | 1B |
| Steps | 44,390 |
| Learning rate | 5e-5 |
| Warmup steps | 300 |
| Time | ~20.8 h |

## Results (lm-eval-harness, 0-shot)

| Task | Metric | v1.0 (C4 4B) |
|------|--------|:---:|
| HellaSwag | acc_norm | 0.310 |
| PIQA | acc | 0.613 |
| WinoGrande | acc | 0.508 |
| ARC-Easy | acc | 0.422 |
| ARC-Challenge | acc_norm | 0.247 |
| OpenBookQA | acc_norm | 0.308 |
| BoolQ | acc | 0.579 |
| LAMBADA | acc | 0.240 |

These serve as the (weak) baseline for the data-quality ablation that produced v1.1.
