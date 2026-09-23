# TinyMixtral v1.1

Dense-ish small MoE (~432M) trained on the SmolLM-inspired blend — the "data quality" ablation
that replaced C4 with FineWeb-Edu + Cosmopedia v2.

- HF: [`mikecovlee/tinymixtral-v1.1-0.5b`](https://huggingface.co/mikecovlee/tinymixtral-v1.1-0.5b)

## Recipe

- Data: FineWeb-Edu + Cosmopedia v2 (≈89:11; mixed 36:4), 4B tokens
- LR: 7e-4 (swept over {1e-4, 3e-4, 5e-4, 7e-4} on 100M-token runs; 7e-4 best)
- Batch: 24 × 1024; cosine with 2,000-step warmup
- Config: [`versions/v1.1/configs/config.json`](configs/config.json)

## Reproduce

```bash
# tokenizer
python scripts/prepare_tokenizer.py --from-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tokenizer/

# data: FineWeb-Edu (sample-10BT) + Cosmopedia v2, then mixed 36:4 (89:11, 4B tokens)
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --tokenizer tokenizer/ --output data/pretrain/fineweb --max-tokens 3560000000 --force
python scripts/prepare_data.py --dataset HuggingFaceTB/cosmopedia-v2 --subset cosmopedia-v2 \
  --tokenizer tokenizer/ --output data/pretrain/cosmopedia --max-tokens 440000000 --force
python scripts/mix_data.py data/pretrain/fineweb data/pretrain/cosmopedia \
  --output data/pretrain/smollm_blend --weights 36 4

# train (4B tokens)
python scripts/train.py --config versions/v1.1/configs/config.json \
  --cache-dir data/pretrain/smollm_blend \
  --batch-size 24 --max-tokens 4000000000 --lr 7e-4 --warmup-steps 2000
```

## Results (lm-eval-harness, 0-shot)

| Task | Metric | v1.1 (432M) |
|------|--------|:---:|
| HellaSwag | acc_norm | 0.308 |
| PIQA | acc | 0.616 |
| WinoGrande | acc | 0.524 |
| ARC-Easy | acc | 0.456 |
| ARC-Challenge | acc_norm | 0.247 |
| OpenBookQA | acc_norm | 0.288 |
| BoolQ | acc | 0.606 |
| LAMBADA | acc | 0.227 |

### Comparison with similar models

Same suite and settings, measured locally (lm-evaluation-harness v0.4.12, 0-shot, cuda, bf16):

| Task | Metric | v1.1 (432M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.308 | 0.563 | 0.473 |
| PIQA | acc | 0.616 | 0.719 | 0.673 |
| WinoGrande | acc | 0.524 | 0.587 | 0.563 |
| ARC-Easy | acc | 0.456 | 0.705 | 0.609 |
| ARC-Challenge | acc_norm | 0.247 | 0.383 | 0.340 |
| OpenBookQA | acc_norm | 0.288 | 0.372 | 0.316 |
| BoolQ | acc | 0.606 | 0.620 | 0.643 |
| LAMBADA | acc | 0.227 | 0.532 | 0.401 |

SmolLM2-360M was trained on 4T tokens and Qwen3-0.6B on 36T tokens, versus 4B tokens for
v1.1 (~1000× less) on a single consumer GPU — the gap is primarily a data-budget difference.

The C4 → SmolLM data-quality switch mainly improved ARC-Easy (+3.4pp) and BoolQ
(+2.7pp) versus the legacy v1.0 C4 model (mean 0.403 → 0.409); ARC-Challenge was
unchanged (0.247).
