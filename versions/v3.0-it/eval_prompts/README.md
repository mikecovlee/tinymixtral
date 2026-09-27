# Held-out Evaluation Prompts

`heldout_prompts_id_1k5.parquet` — the rubric evaluation prompt set used by the
SFT campaign.

| field | detail |
|---|---|
| rows | 4,955 |
| columns | `id`, `prompt`, `ntok` |
| sha256 | `6f8e48e67c592f830615ecc942b46aee3090eec3f827838227db0a5c12b2fdc8` |

Provenance and usage:

- Sampled as a held-out prompt set **before** training-set selection in
  `scripts/data/build_dataset.py`, so no model in this repo was ever trained on any
  of these prompts.
- Shared by all arms (base, prior SFT, v1/v2/v3/v4) so rubric comparisons are
  **paired per prompt id** (see `versions/v3.0-it/judge/rubric_stats.py` and
  `versions/v3.0-it/eval/final_table.py`).
- Consumed by the eval chain: `versions/v3.0-it/eval/run_offload_arm.sh` (Linux) or
  `versions/v3.0-it/eval/run_offload_arm.ps1` (Windows) generate responses for every id,
  then `versions/v3.0-it/judge/rubric_judge2.py` scores them 0-100 on four dimensions.
- The per-scale builds emit `data/sft_<scale>/heldout_prompts.parquet`; this
  directory ships the exact `_id_1k5` variant used by the campaign because no
  tracked script regenerates that slice byte-identically.
