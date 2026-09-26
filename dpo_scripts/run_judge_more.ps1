$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\judgev4.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log
foreach($c in @('sweepbest','grpoifeval','grpogsm')){
  "=== JUDGE $c START $(Get-Date -Format o) ===" | Add-Content -Path $log
  & $py scripts\dpo_eval_judge.py judge --a data/dpo/evalv2_sft.jsonl --b "data/dpo/evalv2_$c.jsonl" --out "data/dpo/evalv2_${c}_judgments.jsonl" *>> $log
  "=== JUDGE $c rc=$LASTEXITCODE END $(Get-Date -Format o) ===" | Add-Content -Path $log
}
"JUDGEV4_DONE $(Get-Date -Format o)" | Add-Content -Path $log
