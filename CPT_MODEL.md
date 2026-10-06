# TinyMixtral v3.0 adaptive CPT model

This branch adds the CPT router as a **pure increment** on the upstream v3.0
native code on `main`. The two router modes coexist:

- **Linear mode (default)** — `model/` is byte-identical to main:
  `TinyMixtralConfig` + the native linear router with auxiliary router loss and
  jitter. Main-branch configs, checkpoints, scripts and tests run unchanged.
- **CPT mode** — everything CPT lives in the top-level `cpt_model/` package,
  which subclasses the mainline model: `CPTConfig`, `CPTRouter` and
  `CPTForCausalLM` retain the v3.0 backbone, QK-Norm, expert computation and
  host-controlled Top-k dispatch.

It contains model source, not trained weights. The prior upstream benchmark
results do not describe the CPT variant; the CPT numbers available so far come
from the proxy hyperparameter search recorded under
[Hyperparameter search](#hyperparameter-search) below — the full-scale run has
not been done yet.

## Package layout and entry points

```
model/        # upstream, zero diff vs main
cpt_model/    # the CPT increment
  config.py     # CPTConfig(TinyMixtralConfig) + config dispatch
  router.py     # CPTRouter, transaction dataclasses
  protocol.py   # CPTModelMixin: validate/commit transaction protocol
  numerics.py   # StableL2 with bounded analytic reverse derivative
  constants.py  # exact scalar arithmetic, correctly rounded FP32
  modeling.py   # thin subclasses: CPTSparseMoE, CPTBlock, CPTForCausalLM
```

Use `cpt_model.model_for_config(config)` or `cpt_model.load_model(path)` to
build the right model automatically. `scripts/train.py` uses the factory.

## Configuration and mode selection

`CPTConfig` subclasses `TinyMixtralConfig` and only adds `cpt_*` fields; it pins
`router_aux_loss_coef` and `router_jitter_noise` to `0` at initialization (CPT
has no auxiliary router loss or jitter). A saved `config.json` that contains
`cpt_router_version` loads as `CPTConfig`; anything else loads as
`TinyMixtralConfig`. `cpt_model.config_from_json_file` dispatches on this rule,
and `CPTForCausalLM.from_pretrained` rejects linear-router checkpoints
explicitly (load those with `model.modeling.TinyMixtralForCausalLM`).

Use `configs/tinymixtral_v3_0_downdp_adaptive.json`: 16 layers, hidden size 1024,
4 experts, default Top-2, K=8, d_p=7, tau_p=1/sqrt(7). `CPTConfig` asserts the
K = 2N and d_p = K-1 relations. The adopted from-scratch configuration is
`configs/cpt_from_scratch/cpt_v3_0_from_scratch.json`: relative to the design
default it changes only `cpt_energy_init_scale` (3/5) and
`cpt_price_learning_rate` (3/125) and keeps every other CPT derived default
(see [Hyperparameter search](#hyperparameter-search)). CPT emits expert
probabilities; the host selects experts and CPT updates prices from actual
assignment counts. Fixed constants derive from exact fractions/roots and round
once to FP32, with nearest/ties-to-even semantics. Normalization preserves the
clamped-L2 forward and uses its analytic bounded reverse derivative. Compiled
CUDA routing disables outer autocast at the call boundary so forward/backward
retain CPT's FP32 contract. The blockwise predictor-corrector calculation and
state update order are unchanged.

Native training uses the upstream entry points, for example:

```sh
# design default
python scripts/train.py --config configs/tinymixtral_v3_0_downdp_adaptive.json --cache-dir /path/to/tokenized --output-dir /path/to/checkpoints
# adopted from-scratch configuration (4-segment launcher: scripts/run_cpt_from_scratch.sh)
python scripts/train.py --config configs/cpt_from_scratch/cpt_v3_0_from_scratch.json --cache-dir /path/to/tokenized --output-dir /path/to/checkpoints
```

## CPT transaction protocol

Every CPT-mode forward returns `output["cpt_transaction"]` (linear mode returns
`None`). The shared native training loop validates the transaction before an
optimizer update and commits it once after the update succeeds. Custom loops
must do the same:

```python
output = model(input_ids, labels=labels)
output["loss"].backward()
model.validate_cpt_transaction(output["cpt_transaction"])
optimizer.step()
model.commit_cpt_transaction(
    output["cpt_transaction"], optimizer_step=model.get_cpt_optimizer_step() + 1,
)
```

**Gradient accumulation.** Commit once per optimizer update, passing every
transaction produced since the last commit; the actual dispatched loads merge
into a single price update and one state-version advance:

```python
transactions = []
for micro_batch in micro_batches:
    out = model(micro_batch, labels=labels)
    (out["loss"] / len(micro_batches)).backward()
    transactions.append(out["cpt_transaction"])
optimizer.step()
model.commit_cpt_transaction(transactions, optimizer_step=model.get_cpt_optimizer_step() + 1)
```

All transactions of one commit must share the same state version (i.e. no
commit happened in between) and each commit attempt is single-use: any later
preparation, write or validation failure discards the proposals even when
rollback succeeds. Commits are atomic across layers (all routers update or none).

## Hyperparameter search

The adopted from-scratch configuration was selected by a 17-run sweep on an
**8-layer / 100M-token proxy**, because one full-scale segment costs ~38 h on a
single GPU. Every run uses the same data, seed, schedule and token stream, so the
runs are strictly paired and directly comparable.

| Item | Value |
|---|---|
| Depth / width | 8 layers, hidden 1024, 16 Q / 4 KV heads, head dim 64 |
| Experts | 4 routed, top-2, expert FFN intermediate 2048, aux loss 0 |
| Router | CPT (`cpt_router_version` 2), K = 8 prototypes, state chunk 16 |
| Sequence | 1024 tokens, packed |
| Batch | 48 × 1024 = 49,152 tokens/step |
| Tokens | 100,000,202 (exactly one `main_s1` shard) = 2,035 steps |
| Data / validation | `main_s1/train_0000.pt` / `pilot_blend30_val/val_0000.pt`, 40 batches |
| LR schedule | WSD, warmup 100, peak 5e-4; weight decay 0.1 (AdamW β 0.9/0.95), grad clip 1.0 |
| Precision | bf16 weights + bf16 optimizer states, chunked CE, gradient checkpointing |
| Seed | 42 |
| Hardware | 1 × RTX PRO 4500 Blackwell 32 GB (200 W), ~33k tok/s, ~50 min/run |

The matched-budget **linear-router control** was run on a second machine
(1 × RTX A5000 24 GB) on a byte-identical copy of the same shard, with the same
seed and glob order, so it consumes the same tokens in the same order. The sweep
configs live in `configs/cpt_hp_sweep/`.

**How to read the results.** Only binary and ordinal judgements are valid. At
2,035 steps the proxy is still on the steep part of the loss curve (Δ = −7.15 PPL
per 400 steps between steps 1,600 and 2,000; final values ~36 here vs ~17 at full
scale), where a small PPL difference largely reflects convergence speed rather
than asymptotic quality. The practical noise floor is ±0.3 PPL at step 2,000. A
transient lead that decays toward zero (as `price_lr` 0.012 → 0.024 does) is not a
steady-state gain; a gap that reverses and then stays flat (as the linear control
does) is real.

### Variants

Two references were used: the first seven variants are built from `base`, the
rest from `price_strong`, which differs from `base` only in
`cpt_price_learning_rate`.

| Variant | Change vs its reference | Knob |
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

### Results

Validation perplexity, lower is better; step 2,000 is the comparison point.

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

### Per-knob conclusions

**`energy_init_scale` — U-shaped, minimum at 0.6.**

| energy_init | 0.1 | 0.3 | **0.6** | 1.2 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| val_ppl @2000 | 36.70 | 36.17 | **35.94** | 36.22 | 36.56 | 37.23 |

A clean U with the minimum at 0.6; both sides are monotone away from it. The step
0.3 → 2.0 costs 3.71 PPL at step 800 (12× the noise floor). This is
counter-intuitive: `energy_init_scale` monotonically *raises* the reachable
routing sharpness, yet performance has an interior optimum. The reason is that too
large an init makes the prototype→expert matrix `B` row-near-one-hot from the
start, so **routing locks in early** and the experts never differentiate; too
small an init pins the routing near uniform and top-2 degenerates into
indiscriminate averaging. 0.6 equals `cpt_expert_temperature`, i.e. the design's
unit ratio. **Adopt 0.6.**

**`capacity_factor` — keep 1.25.**

| capacity_factor | 1.05 | 1.00 | **1.25** |
|---|---|---|---|
| val_ppl @2000 | 37.07 | 36.86 | **36.17** |

Tightening the price-controller dead zone is one-sidedly harmful: 1.05 never leads
at any measured step and ends 0.90 worse than 1.25 (≈3× the noise floor). 1.00 and
1.05 are statistically tied, so the damage is in the 1.25 → 1.05 step. The looser
direction is redundant with `price_off` (also worse). **Keep 1.25.**

**`beta_max` — interior optimum at 0.45; the sequential state is load-bearing.**

| beta_max (saturation) | 0.05 (0.025) | **0.45 (0.225)** | 0.49 (0.245) |
|---|---|---|---|
| val_ppl @2000 | 37.04 | **36.17** | 36.61 |

Turning the state essentially off costs **+0.87 PPL (≈3× the noise floor)**, so
CPT's sequential-state mechanism makes a real contribution — it is not a
decoration on top of the prototype router. Pushing the state to its hard ceiling
also hurts (+0.44), so 0.45 is an interior optimum. **Keep 0.45.**

**`price_learning_rate` — binary, not a magnitude.**

Congestion pricing must be **on**: with it off, PPL is 36.70 (persistently ~0.4
worse) and the expert utilisation spread balloons to 10.7 pp versus 1.1–3.6 pp for
every other run. But raising the rate from 0.012 to 0.024 gains only 0.11 PPL, and
that gain is a decaying transient (the measured max `|price|` moves only 1.14× for
a 2× rate increase — the signature of a self-limiting integral controller whose
steady state is set by the required correction, not the gain). Raising it further
is not expected to help and risks limit-cycle oscillation. **Adopt 0.024** (on,
with margin).

**`expert_temperature`, `rho_beta`, `state_chunk_size` — within noise.**

`T_sharp` (0.30), `T_smooth` (1.20), `rho_short` (0.90) and `rho_long` (0.99) all
land within ±0.3 of `base`. Two observations: `rho_long` **and** `rho_short` being
worse than 0.95 shows the state timescale is at a local optimum (the state is
doing something; only its strength was untested until `beta_off` / `beta_max`);
and lowering `tau_e` also amplifies the congestion price (it enters as
`-price / tau_e`), so `tau_e` is a *confounded* way to sharpen routing — `T_sharp`
(tau_e 0.3) and `einit06` (energy 0.6) reach the same `B` logit spread (1.657) yet
`einit06` wins by 0.30, so `energy_init_scale` is the clean lever. **Keep tau_e
0.6, rho 0.95, chunk 16.**

### Adopted configuration

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

All other CPT parameters keep their derived defaults. The full-scale run is 4
segments of ~2B tokens each, following the v3.0 recipe (WSD warmup 700, LR ladder
5e-4 / 5e-4 / 4e-4 / 3e-4, batch 48 × 1024, seed 42). Because the configuration
changed, the run starts fresh — it cannot resume from the earlier
`energy_init = 0.3` checkpoint.

## Routing expressiveness

CPT routing is a two-stage design: `q = softmax(Aᵀ z / tau_p)` is the probability
over prototypes and `B = row_softmax((center(energy) - price) / tau_e)` is the
prototype→expert matrix; the expert probabilities are the total-probability
combination `pi = q B` with **no final softmax** (`router.py` carries this by
design). Two consequences matter for reading the numbers above:

- **Exact bound.** Because `pi` is a convex combination of the rows of `B`, for
  every token `max_q pi[j] = max_k B[k,j]`: the prototype side can only *select* a
  row of `B`, never exceed it. The reachable sharpness is therefore set entirely
  by `B`, i.e. by `|energy| / cpt_expert_temperature`.
- **Design intent.** Sharpness is not a goal. The design wants *staggered,
  overlapping* soft prototype→expert associations — e.g. prototypes 1,2 matching
  expert 1 while prototypes 2,3 also match expert 2 — with each prototype's token
  share and the expert loads taken into account. The constraint that keeps `B`
  from collapsing to a hard assignment is applied at **initialization only**;
  experts are expected to specialise gradually during training. `K = 2N` and
  `d_p = K - 1` are fixed by design.

Both points are visible in the sweep. Raising `energy_init_scale` sharpens `B`
monotonically — across the `energy_init` 0.6 / 1.2 / 2.0 / 3.0 checkpoints the mean
L1 radius of `B`'s rows grows 0.90 → 1.20 → 1.39 → 1.46 (the full simplex radius is
1.5) and the largest `B` entry reaches 0.997 (near one-hot) — yet PPL gets worse
(35.94 → 36.22 → 36.56 → 37.23). At the far end `B` is effectively one-hot from the
start, exactly the early hard assignment the init constraint exists to prevent.

**Matched-budget control (`linear` vs `einit06`).** Same 8-layer backbone, same
100M tokens, same seed/data/schedule — only the router differs. The v3.0-style
linear router is far sharper, yet scores worse:

| Metric | `linear` (100M / 8L) | CPT `einit06` (100M / 8L) | v3.0 linear (8.05B / 16L) |
|---|---|---|---|
| val PPL @2000 | 37.08 | **35.94** | — |
| mean renormalised top-2 weight | 0.807 | 0.591 | 0.623 |
| fraction with top-2 weight > 0.80 | 0.589 | 0.006 | 0.054 |

The paired delta against `einit06` reverses and then stays flat:

| step | 400 | 800 | 1200 | 1600 | 2000 |
|---|---|---|---|---|---|
| Δ(`linear` − `einit06`) | −1.78 | +4.13 | +1.78 | +1.45 | **+1.14** |

The linear router leads early only because its initial logit dynamic range already
matches its trained value (‖W‖₂ ‖x‖ ≈ 1.28 × 32 = 41.0 vs a trained 46.5), so it
can express near-one-hot from step 0, while CPT must grow into that range. Once
both converge, **CPT is 1.14 PPL (3.1 % relative) better than the linear router at
strictly matched budget**. Because the sharper router loses, routing sharpness is
**anti-correlated with quality at this scale**, and CPT's bounded `B` — the cap on
`w_max` — is a benefit here, not a limitation.

## Scope and limitations

- **Single-process training only.** Congestion prices, the CPT state version and
  the optimizer step counter are process-local buffers updated out of band from
  `optimizer.step()`. DDP/FSDP are unsupported; `commit_cpt_transaction` raises
  when `torch.distributed` is initialized.
- **Eval forwards never commit.** The transaction records the mode of the
  forward that produced it; committing a transaction from a non-training forward
  raises. The commit is governed by the forward's mode, not the mode at commit.
- `CPTForCausalLM.save_pretrained`/`from_pretrained` persist CPT router state
  (prices/anchors/versions); CPT checkpoints are incompatible with linear-router
  weights, which the CPT loader rejects explicitly.
- The inherited `hf/` implementation and HF publication scripts describe
  upstream linear models and are not CPT implementations.
- **Anchors** are constrained to the unit sphere and renormalized at every
  commit; they are excluded from AdamW weight decay via
  `no_weight_decay_parameters()`. Changing optimizer grouping means
  `training_state.pt` produced by earlier revisions of this code cannot resume;
  re-run from a model checkpoint.
- **Sequence state.** The state `(S, nu)` advances once per chunk as one
  aggregated step; `cpt_state_step_size` is bounded by
  `2 / (cpt_state_chunk_size * (1 + cpt_lambda_sa))` so the aggregated step
  cannot overshoot the state ball (violating it distorts routing at the
  percent level). The default chunk is 16, which tracks the strict sequential
  semantics closely; raising the chunk requires lowering the step
  proportionally. The responsibility trajectory is computed in bounded
  sub-blocks, so arbitrarily long chunks stay FP32-finite. The mixing weight
  `beta` saturates at `beta_max * nu/(nu+kappa)` ≈ 0.225 under uniform
  routing, so the sequence state modulates prototypes modestly; the learned
  anchors/energy carry most of the routing.
- **Price dynamics.** The dead-zone controller moves congestion prices by
  `price_learning_rate * control` per commit, where `control` is how far the
  utilisation sits outside the dead zone (0 inside it). With the adopted
  `cpt_price_learning_rate` = 3/125 and `cpt_expert_temperature` = 3/5, a
  utilisation excess of 0.02 moves a price by about 8e-4 logit units per
  commit, so cancelling a persistent expert advantage of order 1 takes hundreds
  of commits. Prices integrate (no decay), so imbalance is always corrected
  eventually; watch `util%` in training logs for early collapse.
- **Performance.** The compiled CUDA path unrolls the chunk loop (at most 128
  chunks; raise `cpt_state_chunk_size` for long sequences instead of falling
  back to eager). The predictor-corrector pass costs roughly 2-3x the routing
  compute of a single pass; routing is small relative to expert FFNs. The eager
  path computes the valid-token index twice (router and dispatch) by design, to
  keep the router self-contained.
- Optional local telemetry, experiment logs, verification archives, old
  evaluations, data tooling and checkpoint-retention extensions are excluded
  from this change.

## Testing

The default suite covers both modes and stays green:

```sh
python -m pytest tests/ -q
```

CPT-focused checks:

```sh
python -m pytest tests/test_adaptive.py tests/test_cpt_constants.py tests/test_cpt_numerics.py tests/test_codeupdate.py tests/test_cpt_training.py tests/test_cpt_protocol.py
```

The upstream linear-mode tests (`test_aux`, `test_router`, `test_hf_parity`, ...)
are kept unchanged as guards of main-branch compatibility. The only removed test
compared against a pre-CPT historical baseline commit, which this branch no
longer supports.

Set `CPT_RUN_CUDA_REGRESSION=1` to include real CUDA Inductor forward/backward
and activation-checkpoint tests under outer BF16 autocast. The default CPU suite
also checks mathematical derivatives, exact constants, model serialization,
transaction protocol guards and a short training-loop/strict-resume integration
check using synthetic data.
