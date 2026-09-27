# TinyMixtral v3.0-it

**Instruction-tuned release.** One-epoch supervised fine-tune of the v3.0 base on
2.17M decontaminated English conversations (10 public sources), produced entirely on
a single GPU. Shipped as `mikecovlee/tinymixtral-it` on the HF Hub.

- HF repo: `mikecovlee/tinymixtral-it`
- Base model: `mikecovlee/tinymixtral` (v3.0, see `versions/v3.0/`)
- Training config: `versions/v3.0-it/configs/3m.json` (the shipped run)

## What's in this bundle

| Path | Purpose |
|------|---------|
| `REPRODUCE.md` | Full runbook: data build → training → evaluation chain |
| `REPORT.md` | Report: methodology, results, lessons learned |
| `configs/` | Training configs per scale tier (200k / 1m / 3m / polish) |
| `data/` | Dataset tooling: `prefetch_sources.py`, `build_dataset.py`, `sample_subset.py` |
| `train_sft.py` | SFT trainer (packing, label masking, `--seed/--resume/--grad-accum/--keep-last`) |
| `run_sft.sh` | Parameterized training entry (`200k|1m|3m|polish`) |
| `eval/` | Evaluation chain: publish → generate → score → lm-eval → summary |
| `judge/` | LLM rubric scoring + paired statistics |
| `eval_prompts/` | Held-out rubric prompt set (4,955 rows, sha256-pinned) |

## Data

`data/build_dataset.py` blends 10 public instruction sources (per-source licenses in
`docs/DATA_LICENSES.md`), with exact-hash + MinHash-LSH (Jaccard ≥ 0.8) dedup, global
shuffle, per-source share caps, and 10-gram decontamination against 6 eval sets plus
the held-out prompt set. No Chinese data. Scale ladder:

| Tier | Rows | Packed seqs | Steps | Wall time |
|------|------|-------------|-------|-----------|
| 200k | 195,170 | 106,718 | 4,447 | 99 min |
| 1m | 856,805 | 644,557 | 26,857 | 10.0 h |
| **3m (shipped)** | **2,168,835** | **1,443,804** | **60,159** | **22.3 h** |
| polish | 50,000 | 26,365 | 1,099 | 24.5 min |

Training: from the v3.0 base (no warm start), lr 2e-5 cosine + 100-step warmup, bf16 +
gradient checkpointing, seq 1024, batch 24 (the max stable at 21.5 GB on a 32 GB
RTX PRO 4500). The polish tier is a low-lr (5e-6) stratified subset run evaluated but
**not shipped** (no gain over the 3M run).

## Results

| Metric | base v3.0 | 3M SFT (v3.0-it) |
|--------|-----------|------------------|
| IFEval prompt-strict | — | **0.1701** |
| IFEval inst-strict | — | **0.2794** |
| GSM8K flexible | — | 0.0227 |
| 8-task harness (canonical) | **0.4250** | 0.4034 |
| LLM rubric (0-100, n=4,955) | 6.4 ± 0.18* | **15.0 ± 0.28** |

*baseline SFT arm on the same held-out prompts; paired delta +8.61 ± 0.28 (t = +30.7).
Honest trade-off: instruction-following and answer quality are sharply better, while
multiple-choice common-sense accuracy drops ~2.2pp (mostly boolq 0.615 → 0.426).
Canonical harness mean = acc_norm for hellaswag/piqa/arc_challenge/openbookqa, acc for
winogrande/arc_easy/boolq/lambada. Full tables and per-task numbers in `REPORT.md`.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("mikecovlee/tinymixtral-it",
                                        trust_remote_code=True).to("cuda").to_bfloat16()
t = AutoTokenizer.from_pretrained("mikecovlee/tinymixtral-it")
```

To retrain from scratch see `REPRODUCE.md`; to reproduce the evaluation see
§3 of `REPRODUCE.md` (`eval/response_eval.py` + `judge/rubric_judge.py` + lm-eval).
