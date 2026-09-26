$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\offload_v1.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log

foreach($pair in @(@('imp-sft-v2-v1','data/dpo/evalv3_sftv2v1.jsonl'), @('imp-sft','data/dpo/evalv3_sft.jsonl'))){
  $m=$pair[0]; $o=$pair[1]
  "=== GEN $m START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\dpo_eval_judge.py gen --model "publish/$m" --prompts data/sft_v2_v1/heldout_prompts_id.parquet --out $o --batch-size 8 --max-new-tokens 448 *>> $log
  "=== GEN $m rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}

foreach($pair in @(@('sftv2v1','data/dpo/evalv3_sftv2v1.jsonl'), @('sft','data/dpo/evalv3_sft.jsonl'))){
  $n=$pair[0]; $r=$pair[1]
  "=== RUBRIC $n START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\rubric_judge2.py --responses $r --out "data/dpo/rubric2_$n.jsonl" --limit 5000 --concurrency 8 *>> $log
  "=== RUBRIC $n rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}

$d="$repo\publish\imp-sft-v2-v1"
$ma="pretrained=$d,tokenizer=$d,trust_remote_code=True,dtype=bfloat16"
"=== HARNESS START $(Get-Date -Format o) ===" | Add-Content -Path $log
& $py -m lm_eval --model hf --model_args $ma --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai --batch_size 16 --device cuda --output_path "$repo\evals\harness\imp-sft-v2-v1" *>> $log
"HARNESS rc=$LASTEXITCODE" | Add-Content -Path $log
& $py -m lm_eval --model hf --model_args $ma --tasks ifeval --apply_chat_template --batch_size 8 --device cuda --output_path "$repo\evals\ifeval\imp-sft-v2-v1" *>> $log
"IFEVAL rc=$LASTEXITCODE" | Add-Content -Path $log
& $py -m lm_eval --model hf --model_args $ma --tasks gsm8k --batch_size 8 --device cuda --output_path "$repo\evals\gsm8k\imp-sft-v2-v1" *>> $log
"GSM8K rc=$LASTEXITCODE" | Add-Content -Path $log
"OFFLOAD_V1_DONE $(Get-Date -Format o)" | Add-Content -Path $log
