# TinyMixtral v3.0 Dense-Total-477M

Dense ablation of [TinyMixtral v3.0](../v3.0/README.md): same data mix, tokenizer,
training recipe and total parameter count, with the routed MoE replaced by a single
dense FFN per layer. Measures what the MoE routing contributes at fixed size and data.

**Status: training complete** (2026-10-05; 4 segments, 8.05B tokens, same step counts
as v3.0). Final verdict: at fixed data, recipe and total size, removing MoE routing
does **not** hurt — v3.0-dense-477m matches or slightly beats v3.0 on the canonical suite.

## Config (vs v3.0)

| | v3.0 (MoE) | v3.0 Dense |
|---|---|---|
| Architecture | 4 routed experts, top-2 | dense FFN (no router) |
| Total params | 477.5M | 477.4M |
| Active params / token | 276.1M | 477.4M |
| Hidden / layers / heads | 1024 / 16 / 16 (GQA 4 KV) | same |
| FFN intermediate | 2048 per expert | **8192** (= 4 experts merged) |
| head_dim / max_pos | 64 / 2048 | same |
| qk-norm / RMS eps / rope | true / 1e-6 / 1e6 | same |
| Embedding tying | yes | yes |
| Vocab | 32000 | same |
| Config file | [`../v3.0/configs/improve_v05b.json`](../v3.0/configs/improve_v05b.json) | [`configs/v3.0-dense-477m.json`](configs/v3.0-dense-477m.json) |

Dense FFN = router-free SwiGLU: `down(silu(gate(x)) * up(x))`, intermediate 8192 —
parametrically identical to the four expert FFNs merged (3 x 1024 x 8192 / layer).

## Training recipe (identical to v3.0)

4 segments over 8.05B tokens (WSD schedule, warmup 700, tail 10% decay; AdamW momentum
carried across segments), effective batch 48 x 1024, seed 42:

| Segment | Tokens | LR |
|---|---|---|
| S1 | 0 -> 2.00B | 5e-4 |
| S2 | -> 3.94B | 5e-4 |
| S3 | -> 6.14B | 4e-4 |
| S4 | -> 8.05B | 3e-4 |

Data pools `main_s1..s4` are rebuilt with the exact shard recipe of the v3.0 card
(same sources: FineWeb-Edu / Cosmopedia v2 / DCLM / OpenCodeInstruct / OpenWebMath /
Wikipedia; same `--start/--take` per shard) so the two runs see identical data.

## Results

Protocol: 7-task harness (canonical, no BoolQ) + MMLU (5-shot) + TruthfulQA MC1/MC2
(0-shot), lm-eval v0.4.12, no chat template. Canonical metric per task (the same
formula used for v3.0): HellaSwag / PIQA / ARC-Challenge / OpenBookQA = `acc_norm`;
WinoGrande / ARC-Easy / LAMBADA = `acc`. Sources: `evals/results/<seg>/` on the
training box.

### Final: v3.0-dense-477m (S4) vs v3.0 (S4) — both trained on 8.05B tokens

| Task | v3.0-dense-477m | v3.0 (MoE) | delta |
|---|---|---|---|
| HellaSwag (acc_norm) | **34.1** | 33.5 | +0.6 |
| PIQA (acc_norm) | 63.5 | **63.8** | −0.3 |
| WinoGrande (acc) | 51.1 | **51.5** | −0.4 |
| ARC-Easy (acc) | **49.1** | 47.8 | +1.3 |
| ARC-Challenge (acc_norm) | **25.7** | 25.5 | +0.2 |
| OpenBookQA (acc_norm) | **30.2** | 29.6 | +0.6 |
| LAMBADA (acc) | **27.1** | 26.8 | +0.3 |
| **7-task mean** | **40.1** | 39.8 | +0.3 |
| MMLU (5-shot, acc) | **24.0** | 23.4 | +0.6 |
| TruthfulQA MC1 | 23.6 | **23.7** | −0.1 |
| TruthfulQA MC2 | 40.9 | **41.7** | −0.8 |

### Per-segment trajectory (7-task mean / MMLU)

| Segment | tokens | 7-task mean | MMLU |
|---|---|---|---|
| S1 | 2.00B | 39.3 | 22.9 |
| S2 | 3.94B | 39.2 | 23.9 |
| S3 | 6.14B | **40.2** | 23.4 |
| S4 | 8.05B | 40.1 | **24.0** |

Reading: dense wins 6 of 11 rows and the 7-task mean (+0.3pp) with routing as the
only variable — the MoE router contributes no measurable gain at this scale/recipe.
Losses (TruthfulQA MC2 −0.8, WinoGrande −0.4) sit within single-run noise for this
model size. MMLU peaks at S4 (24.0, best of the whole v3.0 lineage). Caveat: dense
spends ~1.7x FLOPs per token (477M active vs 276M active for MoE) — this is a
parametric control, not a compute-matched one.

## Layout

```
versions/v3.0-dense-477m/
├── README.md            # this card
└── configs/
    └── v3.0-dense-477m.json  # dense config (experts 0, FFN inter 8192)
```

Training/eval tooling is shared with v3.0: `scripts/train.py`, `scripts/resume.py`,
`versions/v3.0/scripts/run_segment.ps1` (pass `-Cfg` pointing here).
