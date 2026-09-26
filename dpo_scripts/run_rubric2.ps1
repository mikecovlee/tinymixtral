$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\rubric2.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log
foreach($a in @('sft','kto','dpo','sweepbest','grpoifeval','grpogsm','rloo')){
  "=== RUBRIC2 $a START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\rubric_judge2.py --responses "data/dpo/evalv2_$a.jsonl" --out "data/dpo/rubric2_$a.jsonl" --limit 500 --concurrency 6 *>> $log
  "=== RUBRIC2 $a rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}
"RUBRIC2_7_DONE $(Get-Date -Format o)" | Add-Content -Path $log
