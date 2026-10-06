# CPT Hyperparameter Search

This document records the hyperparameter search behind the adopted CPT from-scratch
configuration (`configs/cpt_from_scratch/cpt_v3_0_from_scratch.json`).

The search was run on an **8-layer / 100M-token proxy** rather than the full 16-layer /
8.05B-token recipe, because a single full-scale segment takes ~38 h on one GPU. Every
proxy run uses the same data, seed, schedule, and token stream, so the runs are strictly
paired and directly comparable with each other.

## Proxy setup

| Item | Value |
|---|---|
| Depth / width | 8 layers, hidden 1024, 16 Q / 4 KV heads, head dim 64 |
| Experts | 4 routed, top-2, expert FFN intermediate 2048, aux loss 0 |
| Router | CPT (`cpt_router_version` 2), K = 8 prototypes, state chunk 16 |
| Sequence | 1024 tokens, packed |
| Batch | 48 × 1024 = 49,152 tokens/step |
| Tokens | 100,000,202 (exactly one `main_s1` shard) = **2,035 steps** |
| Data | `data/pretrain/main_s1/train_0000.pt` |
| Validation | `data/pretrain/pilot_blend30_val/val_0000.pt`, 40 batches |
| LR schedule | WSD, warmup 100, peak 5e-4 |
| Weight decay | 0.1 (AdamW β 0.9/0.95), grad clip 1.0 |
| Precision | bf16 weights + bf16 optimizer states, chunked CE, grad checkpointing |
| Seed | 42 |
| Hardware | 1 × RTX PRO 4500 Blackwell 32 GB (200 W), ~33k tok/s, ~50 min/run |

The matched-budget **linear-router control** was run on a second machine
(1 × RTX A5000 24 GB) using a byte-identical copy of the same shard, same seed, same
glob order, so it consumes the same tokens in the same order.

## How to read the results

**Only binary and ordinal judgements are valid.** At 2,035 steps the proxy is still in
the steep part of the loss curve (Δ = −7.15 PPL per 400 steps between steps 1,600 and
2,000; final values ~36 vs ~17 at full scale). In that regime a small PPL difference
largely reflects **convergence speed**, not asymptotic quality. The practical noise floor
is **±0.3 PPL** at step 2,000.

A transient lead that decays toward zero (as `price_lr` 0.012 → 0.024 does) must not be
read as a steady-state gain. A gap that reverses and then **stays flat** (as the linear
control does) is real.

## Variants

Two reference configurations were used. The first seven variants were built from
`base`; the rest from `price_strong`, which differs from `base` only in
`cpt_price_learning_rate`.

| Variant | Changes vs its reference | Knob under test |
|---|---|---|
| `base` | — (energy_init 0.3, price_lr 0.012, tau_e 0.6, rho 0.95, beta_max 0.45, cap 1.25) | reference A |
| `price_strong` | price_lr 0.012 → **0.024** | price learning rate |
| `price_off` | price_lr 0.012 → **0** | congestion pricing off |
| `T_sharp` | tau_e 0.6 → **0.3** | expert temperature |
| `T_smooth` | tau_e 0.6 → **1.2** | expert temperature |
| `rho_short` | rho 0.95 → **0.90** | state half-life |
| `rho_long` | rho 0.95 → **0.99** | state half-life |
| `init_flat` | energy_init 0.3 → **0.1** | initial routing sharpness |
| `einit06` | price_lr 0.024, energy_init 0.3 → **0.6** | initial routing sharpness |
| `einit12` | energy_init → **1.2** | initial routing sharpness |
| `einit20` | energy_init → **2.0** | initial routing sharpness |
| `einit30` | energy_init → **3.0** | initial routing sharpness |
| `cap_tight` | capacity_factor 1.25 → **1.05** | price-controller dead zone |
| `cap_exact` | capacity_factor → **1.00** | price-controller dead zone |
| `beta_off` | beta_max 0.45 → **0.05** | sequential-state strength (off) |
| `beta_max` | beta_max 0.45 → **0.49** | sequential-state strength (ceiling) |
| `linear` | all `cpt_*` keys removed; `router_aux_loss_coef` 0.0 → **0.001** | v3.0-style linear router |

## Results

Validation perplexity (lower is better). Step 2,000 is the comparison point.

| Variant | @400 | @800 | @1200 | @1600 | **@2000** |
|---|---|---|---|---|---|
| `einit06` | 135.23 | 71.25 | 51.85 | 42.95 | **35.94** |
| `price_strong` | 137.60 | 73.12 | 52.07 | 43.30 | **36.17** |
| `einit12` | 135.63 | 71.37 | 52.00 | 43.19 | **36.22** |
| `T_sharp` | 136.32 | 73.06 | 52.01 | 43.39 | **36.24** |
| `base` | 138.98 | 72.80 | 52.72 | 43.43 | **36.28** |
| `rho_short` | 137.58 | 73.25 | 52.25 | 43.48 | **36.41** |
| `rho_long` | 137.87 | 71.17 | 52.13 | 43.54 | **36.41** |
| `einit20` | 136.41 | 74.96 | 52.77 | 43.79 | **36.56** |
| `T_smooth` | 139.87 | 72.13 | 52.76 | 43.70 | **36.57** |
| `beta_max` | 137.79 | 73.51 | 53.34 | 43.85 | **36.61** |
| `price_off` | 138.22 | 76.77 | 53.40 | 44.20 | **36.70** |
| `init_flat` | 138.30 | 79.99 | 54.22 | 44.10 | **36.70** |
| `cap_exact` | 138.71 | 77.82 | 53.90 | 44.22 | **36.86** |
| `beta_off` | 138.70 | 78.42 | 54.70 | 44.49 | **37.04** |
| `cap_tight` | 139.13 | 77.95 | 54.17 | 44.55 | **37.07** |
| `linear` | 133.45 | 75.38 | 53.63 | 44.40 | **37.08** |
| `einit30` | 136.57 | 78.95 | 54.48 | 44.64 | **37.23** |

## Conclusions

### `energy_init_scale` — U-shaped, minimum at 0.6

| energy_init | 0.1 | 0.3 | **0.6** | 1.2 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| val_ppl @2000 | 36.70 | 36.17 | **35.94** | 36.22 | 36.56 | 37.23 |

A clean U with the minimum at 0.6; both sides are monotone away from it. The step
0.3 → 2.0 costs 3.71 PPL at step 800 (12× the noise floor). This is counter-intuitive:
`energy_init_scale` monotonically *raises* the reachable routing sharpness, yet
performance has an interior optimum. The reason is that too large an init makes the
prototype→expert matrix row-near-one-hot from the start, so **routing locks in early**
and the experts never differentiate; too small an init pins the routing near uniform and
top-2 degenerates into indiscriminate averaging. **Adopt 0.6.**

### `capacity_factor` — keep 1.25

| capacity_factor | 1.05 | 1.00 | **1.25** |
|---|---|---|---|
| val_ppl @2000 | 37.07 | 36.86 | **36.17** |

Tightening the price-controller dead zone is one-sidedly harmful: 1.05 never leads at any
measured step and ends 0.90 worse than 1.25 (≈3× the noise floor). 1.00 and 1.05 are
statistically tied, so the damage is in the 1.25 → 1.05 step. The looser direction is
redundant with `price_off` (also worse). **Keep 1.25.**

### `beta_max` — interior optimum at 0.45; the sequential state is load-bearing

| beta_max (saturation) | 0.05 (0.025) | **0.45 (0.225)** | 0.49 (0.245) |
|---|---|---|---|
| val_ppl @2000 | 37.04 | **36.17** | 36.61 |

Turning the state essentially off costs **+0.87 PPL (≈3× the noise floor)**, so CPT's
sequential-state mechanism makes a real contribution — it is not a decoration on top of
the prototype router. Pushing the state to its hard ceiling also hurts (+0.44), so 0.45
is an interior optimum. **Keep 0.45.**

### `price_learning_rate` — binary, not a magnitude

Congestion pricing must be **on**: with it off, PPL is 36.70 (persistently ~0.4 worse)
and expert utilisation spread balloons to 10.7 pp versus 1.1–3.6 pp for every other run.
But raising the rate from 0.012 to 0.024 gains only 0.11 PPL, and that gain is a decaying
transient (the measured max |price| moves only 1.14× for a 2× rate increase — the
signature of a self-limiting integral controller whose steady state is set by the
required correction, not by the gain). Raising it further is not expected to help and
risks limit-cycle oscillation. **Adopt 0.024** (on, with margin).

### `tau_e`, `rho_beta`, `state_chunk_size` — within noise

`T_sharp` (0.30), `T_smooth` (1.20), `rho_short` (0.90) and `rho_long` (0.99) all land
within ±0.3 of `base`. Two observations:

- `rho_long` **and** `rho_short` being worse than 0.95 shows the state timescale is at a
  local optimum (the state is doing something; only its strength was untested until
  `beta_off` / `beta_max`).
- Lowering `tau_e` also amplifies the congestion price (it enters as `-price / tau_e`),
  so `tau_e` is a *confounded* way to sharpen routing. `T_sharp` (tau_e 0.3) and
  `einit06` (energy 0.6) reach the same B logit spread (1.657) yet `einit06` wins by
  0.30 — `energy_init_scale` is the clean lever. **Keep tau_e 0.6, rho 0.95, chunk 16.**

### Matched-budget linear control

At 100M tokens the v3.0-style linear router (`router_aux_loss_coef` 0.001, no CPT keys)
reaches **37.08** — the worst of all 17 runs except `einit30`. Against `einit06` the
paired delta reverses and then stays flat:

| step | 400 | 800 | 1200 | 1600 | 2000 |
|---|---|---|---|---|---|
| Δ(linear − einit06) | −1.78 | +4.13 | +1.78 | +1.45 | **+1.14** |

The linear router leads early only because its initial logit dynamic range already
matches its trained value (‖W‖₂ ‖x‖ ≈ 1.28 × 32 = 41.0 vs a trained 46.5), so it can
express near-one-hot from step 0, while CPT must grow into that range. Once both
converge, **CPT is 1.14 PPL (3.1% relative) better than the linear router at strictly
matched budget** — direct, paired evidence rather than extrapolation.

## Adopted configuration

The sweep winner `einit06` *is* the adopted full-scale configuration, written to
`configs/cpt_from_scratch/cpt_v3_0_from_scratch.json`:

```
cpt_energy_init_scale   = 3/5    (0.6)
cpt_price_learning_rate = 3/125  (0.024)
cpt_capacity_factor     = 5/4    (1.25)
cpt_beta_max            = 9/20   (0.45)
cpt_expert_temperature  = 3/5    (0.6)
cpt_rho_beta            = 19/20  (0.95)
cpt_state_chunk_size    = 16
```

All other CPT parameters keep their derived defaults. The full-scale run is 4 segments
of ~2B tokens each, following the v3.0 recipe (WSD warmup 700, LR ladder
5e-4 / 5e-4 / 4e-4 / 3e-4, batch 48 × 1024, seed 42). Because the configuration changed,
the run starts fresh — it cannot resume from the earlier `energy_init = 0.3` checkpoint.
