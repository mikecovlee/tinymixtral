$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\rubric.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log
foreach($a in @('sft','kto','dpo','sweepbest','grpoifeval','grpogsm','rloo')){
  $in = "data/dpo/evalv2_$a.jsonl"
  $out = "data/dpo/rubric_$a.jsonl"
  if(Test-Path $in){
    "=== RUBRIC $a START $(Get-Date -Format o) ===" | Add-Content -Path $log
    & $py scripts\rubric_judge.py --responses $in --out $out --limit 200 --concurrency 6 *>> $log
    "=== RUBRIC $a rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
  } else {
    "=== SKIP $a (no $in) ===" | Add-Content -Path $log
  }
}
"RUBRIC_DONE $(Get-Date -Format o)" | Add-Content -Path $log
