$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\judgev5.log"
"START $(Get-Date -Format o)" | Set-Content -Path $log
Remove-Item "$repo\data\dpo\evalv2_grpogsm_judgments.jsonl" -ErrorAction SilentlyContinue
& $py scripts\dpo_eval_judge.py judge --a data/dpo/evalv2_sft.jsonl --b data/dpo/evalv2_grpogsm.jsonl --out data/dpo/evalv2_grpogsm_judgments.jsonl *>> $log
"DONE rc=$LASTEXITCODE $(Get-Date -Format o)" | Add-Content -Path $log
