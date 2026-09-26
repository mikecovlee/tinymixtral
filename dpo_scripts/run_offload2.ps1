$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\offload2.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log

foreach($pair in @(@('imp-sft-v2-v1','data/dpo/evalv3b_sftv2v1.jsonl'), @('imp-sft','data/dpo/evalv3b_sft.jsonl'))){
  $m=$pair[0]; $o=$pair[1]
  "=== GEN2 $m START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\dpo_eval_judge.py gen --model "publish/$m" --prompts data/sft_v2_v1/heldout_prompts_id_1k5.parquet --out $o --batch-size 8 --max-new-tokens 448 *>> $log
  "=== GEN2 $m rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}

foreach($pair in @(@('sftv2v1','data/dpo/evalv3b_sftv2v1.jsonl'), @('sft','data/dpo/evalv3b_sft.jsonl'))){
  $n=$pair[0]; $r=$pair[1]
  "=== RUBRIC2 $n START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\rubric_judge2.py --responses $r --out "data/dpo/rubric2v2_$n.jsonl" --limit 5000 --concurrency 8 *>> $log
  "=== RUBRIC2 $n rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}
"OFFLOAD2_DONE $(Get-Date -Format o)" | Add-Content -Path $log
