param(
    [string]$Cfg = "pilot_top1",
    [string]$Lr = "3e-4",
    [string]$MaxTokens = "400000000",
    [string]$Tag = "",
    [int]$Bs = 16,
    [string]$Warmup = "2000",
    [string]$EvalEvery = "2000",
    [switch]$ChunkedCe,
    [switch]$Foreground
)

# --- fixed environment (override with env vars if your paths differ) ---
$Repo     = if ($env:TINYMIXTRAL_REPO) { $env:TINYMIXTRAL_REPO } else { "C:\path\to\tinymixtral-improve" }
$DataRoot = if ($env:TINYMIXTRAL_DATA) { $env:TINYMIXTRAL_DATA } else { "C:\path\to\tinymixtral" }

Set-Location $Repo
$name = if ($Tag -eq "") { $Cfg }
        elseif ($Tag.StartsWith("_")) { "$Cfg$Tag" }
        else { "${Cfg}_$Tag" }
New-Item -ItemType Directory -Force -Path logs, "checkpoints\$name" | Out-Null
$py = if ($env:TINYMIXTRAL_PY) { $env:TINYMIXTRAL_PY } else { "python" }
$pyArgs = @(
    "scripts\train.py",
    "--config", "versions\v3.0\configs\$Cfg.json",
    "--cache-dir", "$DataRoot\data\pretrain\pilot_blend30",
    "--val-dir", "$DataRoot\data\pretrain\pilot_blend30_val",
    "--output-dir", "checkpoints\$name",
    "--batch-size", "$Bs",
    "--max-tokens", $MaxTokens,
    "--lr", $Lr,
    "--schedule", "wsd",
    "--warmup-steps", $Warmup,
    "--seed", "42",
    "--bf16-optim",
    "--save-every-min", "60",
    "--log-every", "100",
    "--eval-every-steps", $EvalEvery,
    "--keep-last-checkpoints", "2"
)
if ($ChunkedCe) { $pyArgs += "--chunked-ce" }
if ($Foreground) {
    Write-Output "NAME=$name LR=$Lr BS=$Bs WARMUP=$Warmup EVAL=$EvalEvery CHUNKED=$($ChunkedCe.IsPresent)"
    & $py @pyArgs
    exit $LASTEXITCODE
}
$p = Start-Process -FilePath $py -ArgumentList $pyArgs -PassThru `
    -RedirectStandardOutput "logs\$name.out.log" `
    -RedirectStandardError "logs\$name.err.log" `
    -WindowStyle Hidden
Write-Output "PILOT_PID=$($p.Id) NAME=$name LR=$Lr BS=$Bs WARMUP=$Warmup EVAL=$EvalEvery CHUNKED=$($ChunkedCe.IsPresent) LOG=logs\$name.out.log"
