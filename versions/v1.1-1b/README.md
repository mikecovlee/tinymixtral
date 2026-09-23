# TinyMixtral v1.1-1b (1B MoE)

Previous flagship (before v3.0). A wider sparse MoE: **~1.18B total / ~352M active** params
(top-2 of 8 routed experts), trained on a SmolLM-inspired blend.

- HF: [`mikecovlee/tinymixtral-v1.1-1b`](https://huggingface.co/mikecovlee/tinymixtral-v1.1-1b)

## Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_layers | 16 |
| Attention | GQA (16 heads / 4 KV heads) |
| Head dim | 64 |
| RoPE theta | 1,000,000 |
| Norm | RMSNorm |
| Experts | 8 routed (top-2) |
| Expert FFN | SwiGLU, intermediate = 2816 |
| Vocab size | 32,000 |
| Max position | 2,048 |
| **Total params** | **~1.18B** |
| **Active params** | **~352M** |

## Training

| Phase | Data | LR | Tokens | Steps | Time | End loss |
|-------|------|----|:------:|:-----:|:----:|:--------:|
| Pretrain (`v1b_moe`) | SmolLM blend (89:11) | 7e-4 | 4B | 244,141 | ~102.5 h | 1.9* |
| Continuation (`v1b_moe_cont`) | non-overlapping new 4B, same 89:11 | 4.2e-4 | 4B | 121,921 | 60.4 h | 1.80 |

\* Decay-phase loss plateaued at ~1.9 then spiked to 3.4 in the final ~300 steps as the data
stream wrapped back to the start of the corpus (shard 0).

Data: FineWeb-Edu + Cosmopedia v2 (89:11), pre-tokenized to 100M-token `.pt` shards.

```bash
python scripts/train.py --config versions/v1.1-1b/configs/v1b_moe.json \
  --cache-dir data/pretrain/smollm_blend \
  --batch-size 16 --max-tokens 4000000000 --lr 7e-4 --warmup-steps 2000
```

(The continuation run reuses the same config on a non-overlapping 4B shard set via `scripts/resume.py`.)

## Results (lm-eval-harness v0.4.12, 0-shot)

| Task | Metric | 1B MoE 4B | 1B MoE 8B |
|------|--------|:---:|:---:|
| HellaSwag | acc_norm | 0.311 | **0.329** |
| PIQA | acc | 0.620 | 0.630 |
| WinoGrande | acc | 0.510 | **0.523** |
| ARC-Easy | acc | 0.463 | **0.479** |
| ARC-Challenge | acc_norm | 0.273 | **0.279** |
| OpenBookQA | acc_norm | 0.288 | **0.306** |
| BoolQ | acc | 0.548 | **0.620** |
| LAMBADA | acc | 0.200 | **0.234** |
| **Mean** | — | 0.402 | **0.425** |

Doubling pretrain tokens improved **every** task (BoolQ +7.2pp, LAMBADA +3.5pp, mean +2.3pp).

**Few-shot:** HellaSwag 10-shot acc_norm 0.312 · WinoGrande 5-shot acc 0.524 ·
ARC-Easy 25-shot acc_norm 0.468 · ARC-Challenge 25-shot acc_norm 0.262.

## Instruction following (IFEval)

IFEval (instruction-level loose accuracy), `--apply_chat_template`:

| Model | IFEval inst-level loose |
|-------|:---:|
| 1B MoE 4B | 0.222 |
| 1B MoE 5B (post-train) | 0.234 |
| 1B MoE 8B (continuation) | 0.221 |

Instruction-following stays flat (~0.22) — more pretraining tokens do not improve it.

## Post-training (negative result)

Post-trained from the lowest-loss pretrain checkpoint (step 243,038) with the v1.1 recipe
(Wiki + Cosmopedia v2 50:50, 1B tokens, lr 2e-5, warmup 300, WSD, 60,975 steps, ~26 h).
Loss fell 2.98 → 1.8 but **downstream metrics were unchanged**:

| Task | Pretrain (4B) | Post-train (5B) |
|------|:---:|:---:|
| HellaSwag (acc_norm) | 0.311 | 0.313 |
| PIQA (acc) | 0.620 | 0.623 |
| WinoGrande (acc) | 0.510 | 0.505 |
| ARC-Easy (acc) | 0.463 | 0.465 |
| ARC-Challenge (acc_norm) | 0.273 | 0.272 |
| OpenBookQA (acc_norm) | 0.288 | 0.290 |
| BoolQ (acc) | 0.548 | 0.528 |
| LAMBADA (acc) | 0.200 | 0.195 |

At this scale, low-LR post-training on a knowledge blend lowers the loss but does not transfer
to task ability.
