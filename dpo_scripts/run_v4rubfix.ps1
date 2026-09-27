$ErrorActionPreference = "Continue"
$repo = "C:\Users\mikecovlee\tinymixtral-improve"
$py = "C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe"
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
Set-Location $repo
Write-Output ("V4RUB START " + (Get-Date -Format o))
& $py scripts\rubric_judge2.py --responses data\dpo\evalv3b_sftv4.jsonl --out data\dpo\rubric2v2_sftv4.jsonl --limit 5000 --concurrency 4 --resume
Write-Output ("V4RUB rc=" + $LASTEXITCODE + " " + (Get-Date -Format o))
Write-Output "V4RUB_DONE"
