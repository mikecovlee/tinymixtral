param(
  [Parameter(Mandatory=$true)][string]$Arm,
  [Parameter(Mandatory=$true)][string]$Tag
)
$ErrorActionPreference = 'Continue'
# Repo root defaults to the checkout containing this script; override with
# $env:EVAL_REPO. Python defaults to PATH; override with $env:EVAL_PY.
if ($env:EVAL_REPO) { $repo = $env:EVAL_REPO }
elseif ($PSScriptRoot -match '[\\/]tools[\\/]eval$') { $repo = (Resolve-Path "$PSScriptRoot\..\..").Path }
else { $repo = (Resolve-Path "$PSScriptRoot\..").Path }
if ($env:EVAL_PY) { $py = $env:EVAL_PY } else { $py = (Get-Command python).Source }
if ($env:PROXY_URL) {
  $env:HTTP_PROXY  = $env:PROXY_URL
  $env:HTTPS_PROXY = $env:PROXY_URL
}
Set-Location $repo
$log = "$repo\logs\offload_$Tag.log"
function Log($m) { $line = "$(Get-Date -Format o) $m"; Add-Content -Path $log -Value $line; Write-Host $line }

Log "OFFLOAD ARM $Arm ($Tag) START"
$mdir = "$repo\publish\$Arm"
if (!(Test-Path $mdir)) { Log "FATAL missing $mdir"; exit 1 }

Log "GEN3 $Arm START"
& $py tools\eval\dpo_eval_judge.py gen --model $mdir --prompts eval_prompts\heldout_prompts_id_1k5.parquet --out "data\dpo\evalv3b_$Tag.jsonl" --batch-size 8 --max-new-tokens 448 1>> $log 2>&1
Log "GEN3 $Arm rc=$LASTEXITCODE"

Log "RUBRIC3 $Arm START"
& $py tools\judge\rubric_judge2.py --responses "data\dpo\evalv3b_$Tag.jsonl" --out "data\dpo\rubric2v2_$Tag.jsonl" --limit 5000 --concurrency 8 1>> $log 2>&1
Log "RUBRIC3 $Arm rc=$LASTEXITCODE"

$ma = "pretrained=$mdir,tokenizer=$mdir,trust_remote_code=True,dtype=bfloat16"

Log "HARNESS $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai --batch_size 16 --device cuda --output_path "$repo\evals\harness\$Arm" 1>> $log 2>&1
Log "HARNESS $Arm rc=$LASTEXITCODE"

Log "IFEVAL $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks ifeval --apply_chat_template --batch_size 8 --device cuda --output_path "$repo\evals\ifeval\$Arm" 1>> $log 2>&1
Log "IFEVAL $Arm rc=$LASTEXITCODE"

Log "GSM8K $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks gsm8k --batch_size 8 --device cuda --output_path "$repo\evals\gsm8k\$Arm" 1>> $log 2>&1
Log "GSM8K $Arm rc=$LASTEXITCODE"

Log "OFFLOAD_ARM_DONE $Arm"
