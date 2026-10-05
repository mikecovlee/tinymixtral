# TinyMixtral v3.0 Dense-Active-276M

Dense ablation of [TinyMixtral v3.0](../v3.0/README.md) at **matched active parameters and
FLOPs**. Where the earlier v3.0-dense experiment matched *total*
parameters (477.4M dense vs 477.5M MoE), this run matches *active* parameters: 276M dense vs
276.1M active MoE — the same compute budget per token. The goal is to isolate what MoE routing
buys when both models see identical data, identical training recipe, and identical per-token
FLOPs.

## Config (vs v3.0 MoE)

| | v3.0 (MoE) | v3.0 Dense-Active-276M |
|---|---|---|
| Architecture | 4 routed experts, top-2 | dense FFN (no router) |
| Total params | 477,465,600 | **276,073,472** |
| Active params / token | 276,139,008 | **276,073,472** |
| Hidden / layers / heads | 1024 / 16 / 16 (GQA 4 KV) | same |
| FFN intermediate | 2048 per expert (4 experts) | **4096** (single SwiGLU) |
| FFN MACs per token | 3 × 1024 × 2048 × 2 (top-2) | 3 × 1024 × 4096 |
| head_dim / max_pos | 64 / 2048 | same |
| QK-Norm / RMS eps / RoPE | true / 1e-6 / 1e6 | same |
| Embedding tying | yes | yes |
| Vocab | 32,000 | same |
| Config file | [`../v3.0/configs/improve_v05b.json`](../v3.0/configs/improve_v05b.json) | [`configs/v3.0-dense-276m.json`](configs/v3.0-dense-276m.json) |

Dense FFN = router-free SwiGLU (`down(silu(gate(x)) * up(x))`, intermediate 4096).
The two architectures have **identical FFN MACs per token** (the dense FFN width doubles
to compensate for top-2 routing activating only half the expert capacity). The only
parametric difference is 16 router matrices (65,536 params, 0.024% of total).

## Training recipe (identical to v3.0)

4 segments over 8.05B tokens (WSD schedule, warmup 700, tail 10% decay; AdamW momentum
carried across segments), effective batch 48 × 1024 = 49,152 tokens/step, seed 42,
bf16 compute + bf16 optimizer state, chunked cross-entropy, grad clip 1.0:

| Segment | Cumulative tokens | Steps | LR | Val PPL (MoE) | Val PPL (dense) |
|---|---|---|---|---|---|
| S1 | 0 → 2.00B | 40,640 | 5e-4 | 17.16 | 17.79 |
| S2 | → 3.94B | 39,421 | 5e-4 | 16.22 | 16.65 |
| S3 | → 6.14B | 44,704 | 4e-4 | 15.81 | 16.51 |
| S4 | → 8.05B | 38,811 | 3e-4 | **15.59** | **16.33** |

Throughput: ~23k tokens/s on a single RTX PRO 4500 32 GB (vs ~14.3k tokens/s for the MoE
run — the dense model has lower routing overhead and simpler kernels).

Data pools `main_s1..s4` are identical to the v3.0 run (same sources: FineWeb-Edu 44% /
DCLM 20% / Cosmopedia 12.5% / OpenCodeInstruct 12.5% / OpenWebMath 6% / Wikipedia 6%;
same shard recipe). Validation uses `pilot_blend30_val` (2 held-out shards).

## Evaluation

Measured with `lm-evaluation-harness` v0.4.12, 0-shot (cuda, bf16), no chat template.

| Metric | v3.0 (MoE) | Dense-Active-276M | Δ |
|---|---|---|---|
| Val PPL (final) | 15.59 | 16.33 | +4.7% |
| 7-task harness mean (excl BoolQ) | **0.3992** | 0.3904 | −0.88 pp |
| MMLU (5-shot) | 0.2480 | **0.2590** | +1.1 pp |
| TruthfulQA MC1 / MC2 | 0.2375 / 0.4169 | **0.2411** / **0.4304** | +0.4 / +1.4 pp |
| GSM8K strict / flex | 0.0000 / 0.0159 | 0.0000 / 0.0136 | — / −0.2 pp |

**Key finding (Rule 2):** at iso-active parameters (276M), iso-FLOPs, and iso-data (8.05B
tokens), the MoE model beats the dense model on LM harness (+0.88 pp) and validation PPL
(−4.7%). The dense model is slightly better on MMLU and TruthfulQA but both are near
chance at this scale. This confirms that MoE routing provides a real quality gain at
fixed compute, not just a parameter-count advantage.

> **Harness note.** The 7-task mean excludes BoolQ (not run in the dense/MoE comparison).
> The 8-task mean (used in the v1.x cards) is not directly comparable.

## Layout

```
versions/v3.0-dense-276m/
├── README.md                         # this card
└── configs/
    └── v3.0-dense-276m.json          # dense config (experts 0, FFN inter 4096)
```

Training/eval tooling is shared with v3.0: `scripts/train.py`, `scripts/resume.py`
(both accept `--no-grad-ckpt` to disable activation checkpointing for speed).
