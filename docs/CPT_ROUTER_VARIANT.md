# CPT Router v1 — TeX-strict variant

## Identity

- Git branch: `cpt_v1`.
- Router algorithm version: `1`.
- Authoritative references in the project handoff root:
  - `CPT_MoE方案.tex`, SHA-256
    `9DEBFEC5399353D5EBB2473D5FA6D11652EFA2683252F44CD22E41FD08225F06`;
  - `CPT_MoE方案源代码.txt`, SHA-256
    `DCD02EF46CB8144D48E33E54B03158C897A830136F3A3C6BDABF7F4272FA2`.
- The two references have the same 1,668 UTF-8 logical lines. Their byte hashes
  differ only because the `.tex` file uses LF and the `.txt` file uses CRLF.

The implementation binds its semantics to the source constant
`CPT_ROUTER_ALGORITHM_VERSION = 1`. Each Router layer also persists the int64
scalar `router_algorithm_version=1` in its `state_dict`. Missing, non-int64,
non-scalar, or cross-version values are rejected before checkpoint dtype
conversion. Old CPT model-only checkpoints that do not contain this new key are
intentionally incompatible with this hardened dual-version delivery.

## Mathematical contract

All theoretical vectors are columns. For each token,

\[
\widetilde{\boldsymbol z}_t=P\boldsymbol x_t,
\qquad
\boldsymbol z_t=
\frac{\widetilde{\boldsymbol z}_t}
{\max\{\lVert\widetilde{\boldsymbol z}_t\rVert_2,\epsilon_z\}}.
\]

There is no projection-coordinate softmax between
\(P\boldsymbol x_t\) and \(\boldsymbol z_t\). The two softmax operations
required by the specification remain:

1. prototype probabilities \(\boldsymbol q_t\);
2. row-wise prototype-to-expert kernel \(\boldsymbol B\).

The final expert probability matrix is

\[
\boldsymbol\Pi_{n,\mu}=\boldsymbol B_n^\top\boldsymbol Q_\mu.
\]

Token-major code evaluates the equivalent
\(\boldsymbol\Pi_{n,\mu}^{\top}=\boldsymbol Q_\mu^{\top}\boldsymbol B_n\).
No softmax is applied after \(\boldsymbol\Pi\).

## Shared model behavior

This variant keeps the hardened shared implementation for Top-2 dispatch,
selected-weight renormalization, experts, attention, continuation state,
padding/segment isolation, Router recompute `global|on|off`, strict training
transactions, exact-rollback/fail-stop policies, schema-v4 checkpoints,
Native/Hugging Face parity, and CE-only language-model loss.

## Rebased local verification snapshot

The current worktree rebases CPT Router v1 onto `upstream/main` commit
`f4108be038f1e89d9bd7e891863e77d22c33d2d2`.

- CUDA-available full regression: `686 passed`.
- True CPU-only full regression: `667 passed, 19 skipped`; the run first
  asserted `torch.cuda.is_available() == False` with
  `CUDA_VISIBLE_DEVICES=-1`.
- Schema-v5 optimizer/scheduler transaction subset: `166 passed`, including
  `AdamW|BF16AdamW` by `cosine|WSD` save/resume combinations, WSD decay-boundary
  continuation, mixed optimizer-moment dtypes, and live model/optimizer/scheduler
  identity gates.
- Strict projection-L2 numerical audit: passed with stable source hashes,
  including column-vector versus token-major equivalence, scale/sign behavior,
  both epsilon branches, Jacobians, zero-projection behavior, and CUDA BF16
  gradient direction.

The following deeper acceptance results belong to the pre-rebase v1 commit
`a9d9f25bea6e323c4b23a9aa7d0d681f369e4026`; they are historical evidence and
are not presented as re-run results for the rebased tree:

- Randomized Router audit: 200 cases plus a 4096-token recurrence, passed.
- CPU strict overfit/transaction audit: 40/40 training steps, final CPT version
  40, checkpoint/reload and continuation exact, passed.
- CUDA deep numerical/recompute audit: passed.
- CUDA tiny BF16 transaction: 15/15 steps under both failure policies.
- Default 433.5M-parameter model: 2/2 strict GPU steps under exact rollback and
  2/2 strict GPU steps under fail-stop.
- Offline Native-to-Hugging-Face publish/reload: passed, including generation,
  packed/masked forward, continuation, and exact persistent-state round-trip.
- Bidirectional v1/v2 compatibility audit: 122/122 checks passed; raw state,
  config, `.bin`, `.safetensors`, sharded weights, config hash, and training
  source-manifest cross-loads were all rejected before silent reinterpretation.

These are engineering acceptance results, not claims of long-run convergence,
quality improvement, expert specialization, throughput superiority, or
statistical significance.

## Known mathematical and execution boundaries

- The TeX formula permits `cpt_projection_dim=1`, but the one-dimensional
  unit sphere contains only the two distinct anchors `-1` and `+1`.
  Consequently this implementation accepts `d_p=1` only when `K=2`; Native,
  Hugging Face, and Router-level validation all reject `d_p=1, K>2`.
- In the branch \(\lVert P\boldsymbol x_t\rVert_2<\epsilon_z\), the local
  Jacobian is \(I/\epsilon_z\). With the default \(\epsilon_z=10^{-6}\), this
  is a real near-zero gradient-amplification risk of the TeX formula; gradient
  clipping remains required.
- The repository training entrypoints remain single-process. Distributed Router
  primitives are tested independently, but `train.py`/`resume.py` reject a
  multi-process launch until data sharding and distributed optimizer ownership
  are completed.
- Legacy Linear-router checkpoints and older CPT checkpoints without the
  algorithm-version buffer cannot be resumed as strict training checkpoints.
- Formal pretraining, matched v1/v2 comparison, multi-seed evaluation, and
  publication-quality research claims remain outside this delivery.
