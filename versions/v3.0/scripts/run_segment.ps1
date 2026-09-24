# run_segment.ps1 - 4-segment 8B main-run launcher: train -> publish -> harness eval.
#
# 4-segment recipe (WSD; user-approved 2026-09-15):
#   S1: 0 -> 2.00B tokens, lr 5e-4, pool main_s1, fresh start            (train.py)
#   S2: -> 3.94B tokens, lr 5e-4, pool main_s2, resume from S1 final  (resume.py)
#   S3: -> 6.14B tokens, lr 4e-4, pool main_s3, resume from S2 final  (resume.py)
#   S4: -> 8.05B tokens, lr 3e-4, pool main_s4, resume from S3 final  (resume.py)
#   (Ladder set 2026-09-16 by the P4b matrix: lr 5e-4 beat 3e-4 by 5.3% val PPL
#    at 400M-token cells; decaying ladder across segments per the 4B-run lesson)
#   - each segment gets its own 2B pool (main_s1..main_s4, built by
#     make_blend_shards.py, strictly non-overlapping) and its own full WSD
#     schedule (warmup 700, tail 10% decay), so LR anneals to the valley at
#     the pause point before eval; AdamW momentum carries over between
#     segments (resume.py post-train mode, --max-tokens)
#   - val is always pilot_blend30_val (2 held-out shards, comparable to the
#     pilot/matrix PPL curves)
#   - bs 48, chunked-ce, seed 42, keep-last 2 identical across all segments
#   - after each segment the final ckpt is published and the repo's 8-task
#     0-shot lm-eval-harness suite runs (canonical form: README "Evaluation";
#     scripted form: scripts/run_ifeval_all.sh)
#
#   -MaxTokens = actual pool size, one full pass per pool (docs P5a):
#               main_s1 2.00B, main_s2 1.94B, main_s3 2.20B, main_s4 1.91B (8.05B total)
#   NOTE: the v3.0 run uses chunked cross-entropy; -ChunkedCe is off by default, so pass
#         it explicitly (as below) to match the recipe.
#   pwsh versions\v3.0\scripts\run_segment.ps1 -Tag s1 -Pool main_s1 -Lr 5e-4 -MaxTokens 2000000000 -ChunkedCe
#   pwsh versions\v3.0\scripts\run_segment.ps1 -Tag s2 -Pool main_s2 -Lr 5e-4 -MaxTokens 1940000000 -ResumeFrom s1 -ChunkedCe
#   pwsh versions\v3.0\scripts\run_segment.ps1 -Tag s3 -Pool main_s3 -Lr 4e-4 -MaxTokens 2200000000 -ResumeFrom s2 -ChunkedCe
#   pwsh versions\v3.0\scripts\run_segment.ps1 -Tag s4 -Pool main_s4 -Lr 3e-4 -MaxTokens 1910000000 -ResumeFrom s3 -ChunkedCe
#
# Progress markers (one per line, grep for these):
#   SEGMENT_DONE <Tag>   training finished, final checkpoint found
#   EVAL_DONE <Tag>      harness eval finished, results under evals\results\<Tag>\
#   EVAL_SKIPPED <Tag>   eval could not run; manual command printed on stdout
#
# ASCII only: this file is parsed on a Windows GBK console; no non-ASCII chars.

param(
    [string]$Cfg = "improve_v05b",
    [string]$Pool = "",
    [string]$Val = "pilot_blend30_val",
    [string]$Lr = "",
    [string]$ResumeFrom = "",
    [string]$Tag = "",
    [int]$Bs = 48,
    [int]$Warmup = 700,
    [int]$EvalEvery = 500,
    [long]$MaxTokens = 2000000000,
    [switch]$ChunkedCe,
    [switch]$DryRun
)

# --- fixed environment (override with env vars if your paths differ) ---
$Repo       = if ($env:TINYMIXTRAL_REPO)    { $env:TINYMIXTRAL_REPO }    else { "C:\path\to\tinymixtral-improve" }
$DataRoot   = if ($env:TINYMIXTRAL_DATA)    { $env:TINYMIXTRAL_DATA }    else { "C:\path\to\tinymixtral" }
$Py         = if ($env:TINYMIXTRAL_PY)      { $env:TINYMIXTRAL_PY }      else { "python" }
$EnvScripts = if ($env:TINYMIXTRAL_SCRIPTS) { $env:TINYMIXTRAL_SCRIPTS } else { "C:\path\to\conda\envs\tinymixtral\Scripts" }

# --- required args ---
foreach ($kv in @(@("Tag", $Tag), @("Pool", $Pool), @("Lr", $Lr))) {
    if ([string]::IsNullOrWhiteSpace($kv[1])) {
        Write-Output ("ERROR: missing required param -" + $kv[0])
        exit 1
    }
}

Set-Location $Repo

# --- resolved paths / constants ---
$CacheDir   = "$DataRoot\data\pretrain\$Pool"
$ValDir     = "$DataRoot\data\pretrain\$Val"
$OutDir     = "checkpoints\$Tag"
$EvalDir    = "evals\results\$Tag"
$PublishDir = "evals\results\$Tag\publish"
$Tokenizer  = "$DataRoot\tokenizer"
$EvalTasks  = "hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai"

# --- training command: fresh (train.py) vs resumed (resume.py) ---
if ([string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $trainScript = "scripts\train.py"
    $trainArgs = @(
        "--config", "versions\v3.0\configs\$Cfg.json",
        "--cache-dir", $CacheDir,
        "--val-dir", $ValDir,
        "--output-dir", $OutDir,
        "--batch-size", "$Bs",
        "--max-tokens", "$MaxTokens",
        "--lr", $Lr,
        "--schedule", "wsd",
        "--warmup-steps", "$Warmup",
        "--seed", "42",
        "--bf16-optim",
        "--save-every-min", "60",
        "--log-every", "100",
        "--eval-every-steps", "$EvalEvery",
        "--keep-last-checkpoints", "2"
    )
    if ($ChunkedCe) { $trainArgs += "--chunked-ce" }
}
else {
    $resumeDir = "checkpoints\$ResumeFrom"
    $trainScript = "scripts\resume.py"
    # --max-tokens puts resume.py in post-train mode: step counter and data
    # position reset to 0, a FRESH full schedule is sized from
    # max-tokens / (batch-size * seq-len), --lr overrides the saved LR, and
    # optimizer momentum carries over (see scripts/resume.py main()).
    $trainArgs = @(
        "--checkpoint-dir", $resumeDir,
        "--cache-dir", $CacheDir,
        "--val-dir", $ValDir,
        "--output-dir", $OutDir,
        "--batch-size", "$Bs",
        "--max-tokens", "$MaxTokens",
        "--lr", $Lr,
        "--schedule", "wsd",
        "--warmup-steps", "$Warmup",
        "--save-every-min", "60",
        "--log-every", "100",
        "--eval-every-steps", "$EvalEvery",
        "--keep-last-checkpoints", "2",
        "--bf16-optim"
    )
    if ($ChunkedCe) { $trainArgs += "--chunked-ce" }
}

# --- eval command: publish the final ckpt, then 8-task 0-shot harness ---
$evalModelArgs = "pretrained=$PublishDir\,trust_remote_code=True,dtype=bfloat16"
$evalArgs = @(
    "--model", "hf",
    "--model_args", $evalModelArgs,
    "--tasks", $EvalTasks,
    "--batch_size", "16",
    "--device", "cuda",
    "--output_path", "evals\results\$Tag"
)

function Format-Command {
    param([string]$Bin, [string[]]$CArgs)
    $parts = @($Bin) + $CArgs
    $quoted = foreach ($a in $parts) {
        if ($a -match '\s' -or $a -eq '') { '"' + $a + '"' } else { $a }
    }
    $quoted -join " "
}

$publishCmd = "scripts\publish_hf.py --checkpoint $OutDir\step_<NNNNNNN>_final --output $PublishDir --tokenizer $Tokenizer"

# --- DryRun: print resolved commands, exit ---
if ($DryRun) {
    Write-Output ("== DryRun Tag=$Tag Pool=$Pool Lr=$Lr ResumeFrom=$($ResumeFrom) " +
        "Cfg=$Cfg Bs=$Bs Warmup=$Warmup EvalEvery=$EvalEvery ChunkedCe=$($ChunkedCe.IsPresent) ==")
    Write-Output ("TRAIN:   " + (Format-Command $Py (@($trainScript) + $trainArgs)))
    Write-Output ("PUBLISH: " + (Format-Command $Py ($publishCmd -split ' ')))
    Write-Output ("EVAL:    " + (Format-Command ("$EnvScripts\lm_eval") $evalArgs))
    exit 0
}

# --- preflight ---
New-Item -ItemType Directory -Force -Path "logs", $OutDir, $EvalDir | Out-Null

$shards = Get-ChildItem -Path $CacheDir -Filter "train_*.pt" -File -ErrorAction SilentlyContinue
if (-not $shards) {
    Write-Output ("ERROR: no train_*.pt shards in pool: " + $CacheDir)
    exit 1
}
if (-not (Test-Path $ValDir)) {
    Write-Output ("ERROR: val dir missing: " + $ValDir)
    exit 1
}
$existing = Get-ChildItem -Path $OutDir -Filter "step_*" -Directory -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output ("ERROR: output dir already contains checkpoints: " + $OutDir + " (delete it or pick a new -Tag)")
    exit 1
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $resumeCkpts = Get-ChildItem -Path $resumeDir -Filter "step_*" -Directory -ErrorAction SilentlyContinue
    if (-not $resumeCkpts) {
        Write-Output ("ERROR: no step_* checkpoints to resume from: " + $resumeDir)
        exit 1
    }
}

# --- train (foreground; pane keeps streaming for psmux parsing) ---
Write-Output ("NAME=$Tag TRAIN=$trainScript LR=$Lr BS=$Bs WARMUP=$Warmup " +
    "EVAL=$EvalEvery CHUNKED=$($ChunkedCe.IsPresent) POOL=$Pool LOG=logs\$Tag.train.log")
& $Py $trainScript @trainArgs 2>&1 | Tee-Object -FilePath "logs\$Tag.train.log"
$trainExit = $LASTEXITCODE
if ($trainExit -ne 0) {
    Write-Output ("ERROR: training failed with exit code " + $trainExit + " (log: logs\$Tag.train.log)")
    exit $trainExit
}

# --- locate final checkpoint ---
$final = Get-ChildItem -Path $OutDir -Filter "step_*_final" -Directory |
    Sort-Object Name | Select-Object -Last 1
if (-not $final) {
    Write-Output ("ERROR: no step_*_final checkpoint found under " + $OutDir)
    exit 1
}
Write-Output ("SEGMENT_DONE " + $Tag)
Write-Output ("FINAL_CKPT " + $final.FullName)

# --- eval: publish + 8-task 0-shot harness ---
function Print-Manual-Commands {
    param([string]$Why)
    Write-Output ("EVAL_SKIPPED " + $Tag)
    Write-Output ("eval could not be fully automated: " + $Why + " ; run manually from " + $Repo + ":")
    Write-Output ("  " + $Py + " " + $publishCmd.Replace(
        "$OutDir\step_<NNNNNNN>_final", $final.FullName))
    Write-Output ("  " + (Format-Command "lm_eval" $evalArgs))
}

$env:PATH = "$EnvScripts;$env:PATH"
$lmEval = Get-Command "lm_eval" -ErrorAction SilentlyContinue
if (-not $lmEval) {
    Print-Manual-Commands ("lm_eval not found in " + $EnvScripts)
    exit 1
}

& $Py "scripts\publish_hf.py" "--checkpoint" $final.FullName "--output" $PublishDir "--tokenizer" $Tokenizer 2>&1 |
    Tee-Object -FilePath "logs\$Tag.publish.log"
if ($LASTEXITCODE -ne 0) {
    Print-Manual-Commands "publish step failed (log: logs\$Tag.publish.log)"
    exit 1
}

& $lmEval.Source @evalArgs 2>&1 | Tee-Object -FilePath "logs\$Tag.eval.log"
if ($LASTEXITCODE -ne 0) {
    Print-Manual-Commands "lm_eval failed (log: logs\$Tag.eval.log)"
    exit 1
}

Write-Output ("EVAL_DONE " + $Tag)
