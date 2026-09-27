param(
  [Parameter(Mandatory=$true)][string]$Arm,
  [Parameter(Mandatory=$true)][string]$Tag
)
$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\mikecovlee\tinymixtral-improve'
$py   = 'C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$env:HTTP_PROXY  = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
Set-Location $repo
$log = "$repo\logs\offload_$Tag.log"
function Log($m) { $line = "$(Get-Date -Format o) $m"; Add-Content -Path $log -Value $line; Write-Host $line }

Log "OFFLOAD ARM $Arm ($Tag) START"
$mdir = "$repo\publish\$Arm"
if (!(Test-Path $mdir)) { Log "FATAL missing $mdir"; exit 1 }

Log "GEN3 $Arm START"
& $py scripts\dpo_eval_judge.py gen --model $mdir --prompts data/sft_v2_v1/heldout_prompts_id_1k5.parquet --out "data/dpo/evalv3b_$Tag.jsonl" --batch-size 8 --max-new-tokens 448
Log "GEN3 $Arm rc=$LASTEXITCODE"

Log "RUBRIC3 $Arm START"
& $py scripts\rubric_judge2.py --responses "data/dpo/evalv3b_$Tag.jsonl" --out "data/dpo/rubric2v2_$Tag.jsonl" --limit 5000 --concurrency 8
Log "RUBRIC3 $Arm rc=$LASTEXITCODE"

$ma = "pretrained=$mdir,tokenizer=$mdir,trust_remote_code=True,dtype=bfloat16"

Log "HARNESS $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai --batch_size 16 --device cuda --output_path "$repo\evals\harness\$Arm"
Log "HARNESS $Arm rc=$LASTEXITCODE"

Log "IFEVAL $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks ifeval --apply_chat_template --batch_size 8 --device cuda --output_path "$repo\evals\ifeval\$Arm"
Log "IFEVAL $Arm rc=$LASTEXITCODE"

Log "GSM8K $Arm START"
& $py -m lm_eval --model hf --model_args $ma --tasks gsm8k --batch_size 8 --device cuda --output_path "$repo\evals\gsm8k\$Arm"
Log "GSM8K $Arm rc=$LASTEXITCODE"

Log "OFFLOAD_ARM_DONE $Arm"
