$repo='C:\Users\mikecovlee\tinymixtral-improve'
Set-Location $repo
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
$py='C:\Users\mikecovlee\miniconda3\envs\tinymixtral\python.exe'
$log="$repo\logs\rloo_eval.log"
$d="$repo\checkpoints\rloo_ifeval\final"
"START $(Get-Date -Format o)" | Set-Content -Path $log
if(Test-Path "$d\config.json"){
  & $py -m lm_eval --model hf --model_args "pretrained=$d,tokenizer=$d,trust_remote_code=True,dtype=bfloat16" --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai --batch_size 16 --device cuda --output_path "$repo\evals\harness\rloo_ifeval" *>> $log
  "HARNESS rc=$LASTEXITCODE $(Get-Date -Format o)" | Add-Content -Path $log
} else {
  "NO FINAL at $d" | Add-Content -Path $log
}
"RLOO_EVAL_DONE $(Get-Date -Format o)" | Add-Content -Path $log
