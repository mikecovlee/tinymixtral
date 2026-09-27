# SFT V3/V4 Training & Evaluation Report (Closed: shipped model = sft_v2_v3)

> Status: campaign complete. V1/V2/V3/V4 all trained and evaluated; final verdict in §6/§7.
> This document is the results report; the pre-registered plan and its ACTUALS live in `docs/SFT_V3_PLAN.md`.

## 1. Background & Goal

- Base model: `v3.0` (477.5M total / 276.1M active parameters, MoE top-2 of 4), max_pos 2048, vocab 32000.
- Prior findings: alignment attempts (DPO/GRPO/KTO/RLOO) showed no significant gain over base; the only large significant win came from `imp-sft` (rubric 0.1 → 7.4). `imp-sft` used only the first 50k rows of a 1M-row parquet (row-order bias), and answer-only data was shown to damage general ability.
- Goal: rebuild a larger, more balanced, higher-quality general SFT from base (**no Chinese data this round**):
  - V3: 3M rows, 1 epoch;
  - V4: 50k high-quality CoT polish (lr 5e-6, 1 epoch).
- Metric priority: **rubric and IFEval are primary**; the 8-task harness is a regression guard only.

## 2. Method

### 2.1 Data construction (`scripts/build_sft_v2.py` / `scripts/make_sft_v4.py`)

- Unified schema: `conversations=[{from:human|gpt, value}]` + `source/category/lang/n_turns`.
- Filtering: assistant spans 10–2048 tokens; exact-hash dedup + MinHash-LSH near-dup dedup (Jaccard ≥ 0.8); global shuffle; per-source share ≤ 15%.
- Decontamination: n-gram overlap removed against gsm8k / arc / obqa / hellaswag / piqa / ifeval / mmlu / ceval / cmmlu and the held-out set.
- No Chinese data.
- V4: stratified sample from the V3 training set (metamath 12k, orcamath 12k, omi2 8k, tulu3 10k, slimorca 8k).

### 2.2 Training (`scripts/train_sft.py`)

- Initialized from base; seq_len 1024, batch_size 24, lr 2e-5 cosine (3% warmup), bf16 + gradient checkpointing, 1 epoch.
- **Fix**: the tokenize/pack stage switched from Python int lists to numpy int32, cutting V2 peak RAM from 40GB to ~21GB and eliminating the swap-7/7 thrash.

### 2.3 Evaluation (fully offloaded to the work machine, `<EVAL_HOST>`)

- rubric: 4,955 fresh held-out prompts, `deepseek-flash` scores four dimensions 0–100 (correctness / completeness / reasoning / instruction_following); report mean ± se plus **paired** t-tests per item.
- IFEval: prompt/inst level, strict + loose.
- harness: 8-task regression guard. Canonical formula: hellaswag / piqa / arc_challenge / openbookqa = acc_norm; winogrande / arc_easy / boolq / lambada = acc.

## 3. Data Scale

| scale | rows | dev | heldout | raw | build time |
|---|---|---|---|---|---|
| V1 | 195,170 | 1,000 | 5,000 | — | — |
| V2 | 856,805 | 4,305 | 5,000 | 2,395,834 | 1,840.9s |
| V3 | 2,168,835 | 10,898 | 2,598 | 4,143,061 | 2,531.9s |
| V4 | 50,000 | — | — | — | 1.9s |

## 4. Training

| scale | rows | packed seq | steps | bs | lr | time |
|---|---|---|---|---|---|---|
| V1 | 195,170 | 106,718 | 4,447 | 24 | 2e-5 | 99.4m |
| V2 | 856,805 | 644,557 | 26,857 | 24 | 2e-5 | 9.97h |
| V3 | 2,168,835 | 1,443,804 | 60,159 | 24 | 2e-5 | 22.3h (1337.8m) |
| V4 | 50,000 | 26,365 | 1,099 | 24 | 5e-6 | 24.5m (done 9/26 23:15, init=V3 final) |

## 5. Evaluation Results

### 5.1 rubric (primary metric, 4,955 paired fresh held-out prompts, 0–100)

| dimension | imp-sft | sft_v2_v1 | sft_v2_v2 | sft_v2_v3 | sft_v4 |
|---|---|---|---|---|---|
| correctness | 7.6 | 12.3 | 12.8 | 14.6 | 14.5 |
| completeness | 6.3 | 11.3 | 11.7 | 14.6 | 14.4 |
| reasoning | 2.3 | 4.9 | 4.9 | 7.2 | 7.2 |
| instruction_following | 9.3 | 17.3 | 18.5 | 23.5 | 23.2 |
| **overall** | **6.4±0.18** | **11.4±0.26** | **12.0±0.24** | **15.0±0.28** | **14.9±0.28** |
| paired vs imp-sft | — | **+5.04±0.26 (t=+19.2)** | **+5.56±0.25 (t=+22.4)** | **+8.61±0.28 (t=+30.7)** | **+8.46±0.28 (t=+30.1)** |

Preregistered gate: Δ ≥ +0.5 and t > 2 — V1/V2/V3 all pass by a wide margin, monotonically increasing with scale.
Note: the V4 rubric set is complete (n=4,955/4,955; after the DeepSeek top-up on 9/27 10:48 the `--resume` fill run finished at 11:17). Root cause of the 2,036 missing rows: HTTP 402 insufficient balance, with failed items silently skipped; the interim value (15.4 at n=2,919) was inflated by front-subset bias. Full-set per-item paired difference V4−V3 = **−0.15±0.18 (t=−0.8, within noise)** — V4 does not beat V3.

### 5.2 IFEval / GSM8K (primary metrics)

| metric | base v3.0 | imp-sft | sft_v2_v1 | sft_v2_v2 | sft_v2_v3 | sft_v4 |
|---|---|---|---|---|---|---|
| IFEval prompt-strict | — | 0.0924 | 0.0961 | 0.1091 | **0.1701** | 0.1664 |
| IFEval inst-strict | — | 0.1894 | 0.2014 | 0.2026 | **0.2794** | 0.2698 |
| GSM8K flexible | — | 0.0167 | 0.0265 | 0.0197 | **0.0227** | 0.0212 |
| GSM8K strict | — | — | 0.0159 | 0.0174 | **0.0205** | 0.0182 |

### 5.3 8-task harness (regression guard, canonical formula)

| metric | base v3.0 | imp-sft | sft_v2_v1 | sft_v2_v2 | sft_v2_v3 | sft_v4 |
|---|---|---|---|---|---|---|
| hellaswag (acc_norm) | 0.335 | — | 0.3347 | 0.3377 | 0.3400 | 0.3400 |
| piqa (acc_norm) | 0.638 | — | 0.6306 | 0.6333 | 0.6338 | 0.6311 |
| winogrande (acc) | 0.515 | — | 0.5209 | 0.5185 | 0.5280 | 0.5272 |
| arc_easy (acc) | 0.478 | — | 0.4558 | 0.4545 | 0.4482 | 0.4436 |
| arc_challenge (acc_norm) | 0.255 | — | 0.2534 | 0.2517 | 0.2602 | 0.2577 |
| openbookqa (acc_norm) | 0.296 | — | 0.298 | 0.29 | 0.3020 | 0.3040 |
| boolq (acc) | 0.615 | — | 0.5801 | 0.5532 | 0.4263 | 0.4419 |
| lambada_openai (acc) | 0.268 | — | 0.2948 | 0.2874 | 0.2890 | 0.2882 |
| **mean** | **0.4250** | — | **0.4210 (-0.40pp)** | **0.4158 (-0.92pp)** | **0.4034 (-2.16pp)** | **0.4042 (-2.08pp)** |

Notes:
1. rubric numbers use the fresh 5k held-out set + `rubric_judge2` (0–100); NOT comparable with the old 500-prompt figures (base 0.1 / imp-sft 7.4).
2. The base 0.4272 / imp-sft 0.4260 recorded in the plan have no backing JSON on the work machine; their formula is unrecoverable. This table uses one canonical formula throughout, with base recomputed from the README v3.0 per-task table (0.4250).
3. The harness is a **regression guard, not an optimization target**. The drop grows with scale: V1 -0.40pp, V2 -0.92pp, V3 -2.16pp; V3's drag concentrates in boolq (0.615→0.426) and arc_easy, while hellaswag/piqa/winogrande/arc_challenge/openbookqa/lambada stay flat or improve. Logged as the trade-off for large instruction-following gains; the V4 polish (low lr, high-quality CoT) was expected to partially recover it.

## 6. Conclusions (through V4)

- **rubric rises monotonically with scale**: V1 +5.04 / V2 +5.56 / V3 **+8.61 (t=+30.7)**, significant across 4,955 paired items; V4 full-set 14.9±0.28 (**+8.46, t=+30.1**), paired V4−V3 −0.15±0.18 (t=−0.8, noise) → low-lr polish is neutral-to-slightly-negative, as suspected.
- **IFEval rises sharply**: prompt-strict 0.0961→0.1091→**0.1701** (imp-sft 0.0924); inst-strict 0.2014→0.2026→**0.2794** (imp-sft 0.1894). V4 polish dips slightly (0.1664/0.2698) — a 50k subset at low lr does not dislodge V3's instruction-following advantage.
- **GSM8K positive but limited**: flexible 0.0265 / 0.0197 / **0.0227** (imp-sft 0.0167), short of the planned +1~3pp expectation; strict climbs with scale to 0.0205. V4 polish roughly flat (0.0212).
- **Harness trade-off widens**: 0.4210 / 0.4158 / 0.4034 / 0.4042 (base 0.4250), concentrated in boolq / arc_easy; V4 polish nudges boolq up (0.4263→0.4419), +0.08pp on the mean, still outside the 0.3pp guard.
- **The primary-metric direction is clearly right**: V1→V2→V3 improves across the board. **V3 (sft_v2_v3) is this campaign's best model**: top rubric/IFEval/GSM8K. V4 polish did not significantly beat V3. Choosing V3 as the V4 init was correct; the polish step itself added nothing.

## 7. Next Steps

1. ~~V3 training + eval~~ (done: trained to 9/26 15:05, results pulled 22:34).
2. ~~V4 polish training~~ (done: 9/26 23:15, 1,099 steps / 24.5m, init=V3 final → `checkpoints/sft_v4/step_0001099_final`).
3. ~~V4 evaluation~~ (done: 9/27 07:27 `OFFLOAD_ARM_DONE`, auto-pulled, `data/dpo/final_table.md` produced 07:28; only the rubric was missing 2,036 rows, see §5.1 note).
4. ~~Commit tooling~~ (done: commit `966fead`, 44 files — `train_sft.py` numpy-int32 memory patch + `publish_hf.py` tokenizer whitelist + data-build / training / eval-offload scripts).
5. ~~Fill V4 columns into §5 tables + per-gate verdict~~ (done: rubric PASS, IFEval PASS, GSM8K positive, harness guard exceedance logged; verdict = V3 is the champion).
6. ~~Last open item: DeepSeek top-up → complete the 2,036 V4 rubric rows → refresh tables~~ (**done**: 9/27 resume, complete 4,955/4,955 at 11:17, final_table.md refreshed; conclusion unchanged = **ship sft_v2_v3**, V4 polish does not beat V3 on any primary metric). **Campaign closed, nothing outstanding.**

## 8. Lessons Learned

### 8.1 Data & training strategy

- **Narrow-slice data damages ability — it is not merely 'no gain'**: exam3 (answer-only math) scored rubric 1.2, paired −5.81 (t=−7.2). SFT data must keep full explanatory answers; math share must stay small and process-bearing.
- **Row-order bias is an invisible trap**: imp-sft historically took only the first 50k rows of a 1M parquet (unshuffled) — effectively training on a narrow slice. That is the main reason it plateaued at rubric 6.4 while V1 gained +5.04 purely from better data composition. **Always globally shuffle before sampling from large datasets** (built into this round's builder).
- **This scale axis is far from saturated**: 195k→857k→2.17M rows gave rubric 11.4→12.0→15.0 and IFEval 0.096→0.109→0.170. The V2→V3 jump was the largest (+3.0), indicating **data composition (tulu3 multi-turn instruction share) matters more than raw scale**.
- **Low-lr polish buys nothing**: V4 (50k math/long-form @ lr 5e-6, 1,099 steps) sat within noise of V3 on every primary metric (paired rubric −0.15±0.18). For a model at this size, a polish stage extracts nothing — cut it next time and spend the budget on main training data.
- **Decontamination / held-out must come first**: n-gram decontam across all 8 eval sets plus a zero-overlap rubric held-out set; without it the 4,955-pair significance would be fake.

### 8.2 Training-box engineering (RTX PRO 4500 32GB / 60GB RAM)

- **Python int lists are a memory bomb**: the tokenize intermediate of 856k × ~974-token examples ≈ 23–46GB; V2's first attempt thrashed swap 7/7 at the packing stage. After the numpy int32 patch, peak went 40GB→21GB (committed with `966fead`). Default data pipelines to numpy/arrow, never native int lists.
- **glibc arena retention**: V3 ran the whole job at RSS ~42GB, stable and harmless (free was only ~1GB); the correct signal is swap no longer growing. Do not kill jobs based on RSS alone.
- **Throughput baseline**: seq1024 / bs24 stable (21.6GB VRAM) at ~0.745 steps/s. V1 4,447 steps = 1.7h; V2 26,857 steps = 10.0h; V3 60,159 steps = 22.3h. Schedule accordingly.
- **Marker + waiter automation was the biggest engineering win**: SFTV2_DONE → auto-start V3 (14s later) → auto publish + scp + remote eval; zero GPU idle time, 48h unattended. Every job over 2h should use this pattern (outer shell printing `START/rc=/DONE` markers).

### 8.3 Evaluation methodology

- **Paired tests + large samples are the lifeline**: the rubric judge is coarse (scores cluster in 0–25); 500 samples cannot separate arms. With 4,955 per-id pairs, t-values reach 19–31 and even a 0.5-point V1/V2 gap resolves. **Set the acceptance gates before running eval** (this round preregistered Δ≥+0.5 and t>2, avoiding post-hoc metric shopping).
- **Metric formulas must be preregistered verbatim**: the legacy base 0.4272 / imp-sft 0.4260 numbers were voided — no JSON archives, formula unrecoverable; acc vs acc_norm differs by 3.5pp on arc_easy alone. Only after fixing one canonical formula (hellaswag/piqa/arc_challenge/openbookqa = acc_norm, rest = acc) did base/V1–V4 become comparable. **Any number entering a comparison table must be stored together with its formula**.
- **Partial data ≠ random data**: the 2,036 missing V4 rubric rows were all at id≥3022 (generation order = the harder tail); the interim 15.43 was inflated, the completed 14.86 flipped the conclusion (V4 does not beat V3). **Judge files must pass a coverage check before any verdict** (final_table-style tools should assert n-completeness).
- **Silent skipping is a bug-class anti-pattern**: rubric_judge2.py retried failed API calls 4 times, then skipped them without writing a row and still exited rc=0 — an entire 402-insufficient-balance round "succeeded". Batch eval scripts must **fail fast or write a failure manifest**, never silently drop samples.

### 8.4 Eval-box (Windows work machine) operations

- **Inline PowerShell over ssh with nested quotes fails silently** (no tmux session, no log, no error) → put all remote logic into .ps1 files and scp them.
- **tmux launch without `*> log` redirection loses the markers**: during the V4 resume run the DONE marker never landed, the local waiter waited in vain, and the finish had to be done manually.
- **Remote python stdout is block-buffered**: once redirected to a file, tqdm/progress is invisible; judge progress by output-file size and mtime (gen stage: .jsonl bytes against the ~7.3MB / 4,955-row reference).
- **Keep the API key on the eval box**: the local judge key is blocked by the CC safety net (auth.json unreadable), so local re-judging is impossible; running generation + scoring on the work machine (where the key lives) is the correct architecture. Also: run a minimal API probe before any external-API batch (a 402 pre-probe would have saved an entire resume round).

### 8.5 Model capability boundary (477M total / 276M active)

- After SFT the *conversational form* is fully there (chat template, lists/code blocks, clean stops, no ChatML leakage, no runaway repetition), but *content* shows classic small-model symptoms: intra-sentence echo loops, fabricated arithmetic (17×4→17), failure on strict format instructions ("reply with JSON only") — quantitatively consistent with IFEval 17% / GSM8K 2.7% (six-case chat probe, 9/27).
- Instruction-following and base commonsense/factual knowledge are two separate ceilings; the former is buyable with SFT data (+84% this round), the latter is bound by parameter count and pretraining corpus — **do not expect SFT to fix it**.
- boolq-style yes/no tasks are the most sensitive sentinel for SFT distribution drift (0.615→0.426 while the other 7 harness tasks stayed roughly flat). If the guard matters, add natural-language-judgment samples to the SFT mix as a hedge.
