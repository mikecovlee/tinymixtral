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

It contains model source, not trained weights. Prior upstream benchmark results
do not describe the untrained CPT variant.

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
4 experts, default Top-2, K=8, d_p=7, tau_p=1/sqrt(7). CPT emits expert
probabilities; the host selects experts and CPT updates prices from actual
assignment counts. Fixed constants derive from exact fractions/roots and round
once to FP32, with nearest/ties-to-even semantics. Normalization preserves the
clamped-L2 forward and uses its analytic bounded reverse derivative. Compiled
CUDA routing disables outer autocast at the call boundary so forward/backward
retain CPT's FP32 contract. The blockwise predictor-corrector calculation and
state update order are unchanged.

Native training uses the upstream entry points, for example:

```sh
python scripts/train.py --config configs/tinymixtral_v3_0_downdp_adaptive.json --cache-dir /path/to/tokenized --output-dir /path/to/checkpoints
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
