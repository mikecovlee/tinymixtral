# TinyMixtral v2.0 beta — Shared Expert

Architecture-ablation release: adds a DeepSeek-style always-on **shared expert** to the MoE layer.
Code lives in [`shared_expert/`](../../shared_expert/).

> **Frozen historical code.** `shared_expert/` is a self-contained snapshot of the training stack as
> of this experiment; it is kept for reproducibility and is not maintained alongside the mainline
> `model/`, `hf/` and `scripts/`. Config: [`configs/config.json`](configs/config.json).

- HF: [`mikecovlee/tinymixtral-v2.0-beta`](https://huggingface.co/mikecovlee/tinymixtral-v2.0-beta)

## Architecture changes (vs v1.1)

| Parameter | Baseline (v1.1) | Shared Expert (v2) |
|-----------|:---:|:---:|
| Attention | GQA 14Q / 2KV (7:1) | GQA 16Q / 4KV (4:1) |
| Head dim | 64 | 56 |
| Experts | 6 routed (top-2) | 1 shared + 6 routed (top-2) |
| LR schedule | Cosine | WSD (Warmup-Stable-Decay) |
| Batch size | 24 | 22 |
| **Total params** | **~432M** | **~498M** |
| **Active params** | **~176M** | **~241M** |

The shared expert is always active (no routing), providing general-purpose features; routed
experts specialize via top-2 gating. Output is the sum of both.

## Training

```bash
python shared_expert/scripts/train.py --config versions/v2.0-beta/configs/config.json \
  --cache-dir data/pretrain/smollm_blend \
  --output-dir checkpoints/v2 --batch-size 22 \
  --max-tokens 4000000000 --lr 7e-4 --schedule wsd \
  --warmup-steps 2000 --save-every-min 120

python shared_expert/scripts/resume.py --checkpoint-dir checkpoints/v2 \
  --output-dir checkpoints/v2_posttrain \
  --cache-dir data/posttrain2/knowledge_blend \
  --max-tokens 1000000000 --lr 2e-5 --warmup-steps 300 \
  --batch-size 22 --schedule wsd --save-every-min 60
```

Data sources: pretrain blend `smollm_blend` = FineWeb-Edu
([`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT`) + Cosmopedia v2
([`HuggingFaceTB/cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2)) (mixed 36:4);
post-train blend `knowledge_blend` = Wikipedia ([`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia),
`20231101.en`) + Cosmopedia v2 (50:50).

## Results (lm-eval-harness, 0-shot)

| Task | Metric | v1.1 (432M) | v2.0 (498M) |
|------|--------|:---:|:---:|
| HellaSwag | acc_norm | 0.308 | **0.326** |
| PIQA | acc | 0.616 | **0.631** |
| WinoGrande | acc | **0.524** | 0.506 |
| ARC-Easy | acc | 0.456 | **0.474** |
| ARC-Challenge | acc_norm | 0.247 | **0.272** |
| OpenBookQA | acc_norm | 0.288 | **0.290** |
| BoolQ | acc | **0.606** | 0.455 |
| LAMBADA | acc | **0.227** | 0.224 |

### Comparison with similar models

Same suite and settings, measured locally (lm-evaluation-harness v0.4.12, 0-shot, cuda, bf16):

| Task | Metric | v2.0 (498M) | SmolLM2-360M | Qwen3-0.6B |
|------|--------|:---:|:---:|:---:|
| HellaSwag | acc_norm | 0.326 | 0.563 | 0.473 |
| PIQA | acc | 0.631 | 0.719 | 0.673 |
| WinoGrande | acc | 0.506 | 0.587 | 0.563 |
| ARC-Easy | acc | 0.474 | 0.705 | 0.609 |
| ARC-Challenge | acc_norm | 0.272 | 0.383 | 0.340 |
| OpenBookQA | acc_norm | 0.290 | 0.372 | 0.316 |
| BoolQ | acc | 0.455 | 0.620 | 0.643 |
| LAMBADA | acc | 0.224 | 0.532 | 0.401 |

SmolLM2-360M was trained on 4T tokens and Qwen3-0.6B on 36T tokens, versus 4B tokens for
v2.0 beta on a single consumer GPU; the gap is primarily a data-budget difference.

## Conclusion

The shared expert does **not** provide substantial improvement at this scale: mixed harness
results (HellaSwag/ARC +1.5–2.5pp, BoolQ −15pp) for 15% more total params. At ~241M active
params the routing overhead and representational fragmentation outweigh the capacity benefit.
The design is more likely to pay off at 1B+ active params (cf. DeepSeek, Mixtral).
