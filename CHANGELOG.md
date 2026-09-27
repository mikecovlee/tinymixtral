# Changelog

## tinymixtral-it (2026-09-27) — instruction-tuned release

Shipped model: `mikecovlee/tinymixtral-it` (HF Hub), trained arm `sft_v2_v3`.

- SFT campaign V1→V4: from the v3.0 base, 1 epoch on progressively larger
  decontaminated English mixtures (195k / 857k / 2.17M rows), V4 = 50k-row
  low-lr polish from the best arm.
- Headline vs the prior 50k-row SFT (paired, n=4,955 held-out prompts):
  LLM rubric +8.61 (t=+30.7), IFEval prompt-strict 0.0924→0.1701,
  inst-strict 0.1894→0.2794, GSM8K flexible 0.0167→0.0227.
- Known trade-off: canonical 8-task harness 0.4250→0.4034, concentrated in
  boolq; documented honestly in `docs/SFT_V3_REPORT.md`.
- Tooling: `scripts/build_sft_v2.py` (quota/dedup/decontam pipeline),
  `scripts/train_sft.py` + `--seed/--resume/--grad-accum/--keep-last`,
  `tools/{sft,eval,judge}` campaign chain, `docs/DATA_LICENSES.md`,
  `eval_prompts/` pinned held-out set, numerical parity tests
  `model/` vs `hf/`, English docs (`docs/SFT_V3_{PLAN,REPORT,REPRODUCE}.md`).

## v3.0 (2026-09) — flagship MoE

- 477.5M total / 276.1M active (top-2 of 4 experts), 8.05B tokens, 6-source
  blend (FineWeb-Edu, Cosmopedia-v2, DCLM, OpenCodeInstruct, OpenWebMath,
  Wikipedia). Canonical 8-task harness 0.4250.

## v2.0-beta (2026) — shared-expert ablation

- 498M/241M with one always-on shared expert; harness 0.397.
  Frozen code snapshot kept in `shared_expert/`.

## v1.1-1b (2026)

- 1,182M/352M MoE; harness 0.425.

## v1.1 (2026)

- 432M/176M MoE; harness 0.409.

## v1.0 (2026)

- First MoE run on C4-en; harness 0.403.
