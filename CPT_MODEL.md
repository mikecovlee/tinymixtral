# TinyMixtral v3.0 adaptive CPT model

This independent branch derives from upstream v3.0 native code on `main`.
It replaces the native linear router with CPT while retaining the v3.0 backbone,
QK-Norm, expert computation and host-controlled Top-k dispatch. It contains model
source, not trained weights. Prior upstream benchmark results do not describe
this untrained CPT variant.

Use `configs/tinymixtral_v3_0_downdp_adaptive.json`: 16 layers, hidden size 1024,
4 experts, default Top-2, K=8, d_p=7, tau_p=1/sqrt(7). CPT emits expert probabilities;
the host selects experts and CPT updates prices from actual assignment counts.
There is no auxiliary router loss or jitter. Fixed constants derive from exact
fractions/roots and round once to FP32, with nearest/ties-to-even semantics.
Normalization preserves the clamped-L2 forward and uses its analytic bounded
reverse derivative. Compiled CUDA routing disables outer autocast at the call
boundary so forward/backward retain CPT's FP32 contract. The existing blockwise
predictor-corrector calculation and state update order are unchanged.

Native training uses the upstream entry points, for example:

```sh
python scripts/train.py --config configs/tinymixtral_v3_0_downdp_adaptive.json --cache-dir /path/to/tokenized --output-dir /path/to/checkpoints
```

The shared native training loop validates the CPT transaction before an optimizer
update and commits it once after the update succeeds. Custom loops must do the same:

```python
output = model(input_ids, labels=labels)
output["loss"].backward()
model.validate_cpt_transaction(output["cpt_transaction"])
optimizer.step()
model.commit_cpt_transaction(
    output["cpt_transaction"], optimizer_step=model.get_cpt_optimizer_step() + 1,
)
```

Use native `save_pretrained`/`from_pretrained` for CPT state. Existing linear-router
weights are incompatible. The inherited `hf/` implementation and HF publication
scripts describe upstream models and are not CPT implementations. Inherited
upstream tests/configurations for linear routing, auxiliary losses and HF parity
are not the test suite or configurations for this independent CPT model.
Optional local telemetry, experiment logs, verification archives, old evaluations,
data tooling and checkpoint-retention extensions are excluded from this change.

Focused checks:

```sh
python -m pytest tests/test_adaptive.py tests/test_cpt_constants.py tests/test_cpt_numerics.py tests/test_codeupdate.py tests/test_cpt_training.py
```

Set `CPT_RUN_CUDA_REGRESSION=1` to include real CUDA Inductor forward/backward and
activation-checkpoint tests under outer BF16 autocast. The default CPU suite also
checks mathematical derivatives, exact constants, model serialization and a short
training-loop/strict-resume integration check using synthetic data.
