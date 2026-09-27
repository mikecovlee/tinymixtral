# Changelog

## tinymixtral-it (2026-09-27) — instruction-tuned release

Shipped model: `mikecovlee/tinymixtral-it` (HF Hub), shipped model `v3.0-it` (3M-row SFT).

- SFT campaign 200k→polish: from the v3.0 base, 1 epoch on progressively larger
  decontaminated English mixtures (195k / 857k / 2.17M rows), polish = 50k-row
  low-lr polish from the best arm.
- Headline vs the prior 50k-row SFT (paired, n=4,955 held-out prompts):
  LLM rubric +8.61 (t=+30.7), IFEval prompt-strict 0.0924→0.1701,
  inst-strict 0.1894→0.2794, GSM8K flexible 0.0167→0.0227.
- Harness (canonical 7-task suite): 0.3979→0.4002 (+0.23pp; only ARC-Easy dips).
  BoolQ is excluded from the suite as an unstable sentinel (see
  `versions/v3.0-it/REPORT.md` §8.5).
- Tooling: `versions/v3.0-it/data/build_dataset.py` (quota/dedup/decontam pipeline),
  `versions/v3.0-it/train_sft.py` + `--seed/--resume/--grad-accum/--keep-last`,
  `versions/v3.0-it/{eval,judge}` evaluation chain, `docs/DATA_LICENSES.md`,
  `versions/v3.0-it/eval_prompts/` pinned held-out set, numerical parity tests
  `model/` vs `hf/`, English docs (`versions/v3.0-it/{REPORT,REPRODUCE}.md`).

## v3.0 (2026-09) — flagship MoE

- 477.5M total / 276.1M active (top-2 of 4 experts), 8.05B tokens, 6-source
  blend (FineWeb-Edu, Cosmopedia-v2, DCLM, OpenCodeInstruct, OpenWebMath,
  Wikipedia). Canonical 7-task harness 0.3979.

## v2.0-beta (2026) — shared-expert ablation

- 498M/241M with one always-on shared expert; harness 0.389.
  Frozen code snapshot kept in `shared_expert/`.

## v1.1-1b (2026)

- 1,182M/352M MoE; harness 0.397.

## v1.1 (2026)

- 432M/176M MoE; harness 0.381.

## v1.0 (2026)

- First MoE run on C4-en; harness 0.378.
