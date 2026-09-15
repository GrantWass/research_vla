# One-command TurboVLA policy server on Windows (grant-pc).
# Usage (PowerShell):
#   C:\Users\grant\vla\serve_turbovla.ps1 -Task stack_bowls -Ckpt <RUN_DIR> -Port 5999
# The sim client (Linux GPU box) then dials this machine:
#   bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode client \
#     --policy-host <THIS_PC_TAILNET_IP> --policy-port 5999
param(
  [string]$Task = "stack_bowls",
  [string]$Ckpt = "demo",
  [string]$EnvCfg = "arx_x5",
  [string]$ActionType = "joint",
  [int]$Seed = 0,
  [string]$PolicyGpu = "0",
  [int]$Port = 5999
)
$ErrorActionPreference = "Stop"

$VlaRoot = "C:\Users\grant\vla"
$PolicyDir = "$VlaRoot\RoboDojo\XPolicyLab\policy\TurboVLA"
$GitBash = "C:\Program Files\Git\bin\bash.exe"
$CondaScripts = "C:\Users\grant\miniconda3\Scripts"
$CondaBin = "C:\Users\grant\miniconda3\condabin"

# pip breaks when PATH contains the Codex junction (WinError 448); drop it.
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -notmatch 'Codex' }) -join ';'

# Persistent python3 shim for Git Bash (XPolicyLab launchers call python3).
$ShimDir = "$VlaRoot\bin"
New-Item -ItemType Directory -Force $ShimDir | Out-Null
$Shim = Join-Path $ShimDir "python3"
$ShimContent = '#!/bin/sh' + "`n" + 'exec /c/Users/grant/miniconda3/envs/turbovla-robodojo/python.exe "$@"' + "`n"
Set-Content -Path $Shim -Value $ShimContent -NoNewline

# Model assets (override per run: $env:DINOV3_PATH=...; .\serve_turbovla.ps1 ...)
if (-not $env:DINOV3_PATH) { $env:DINOV3_PATH = "C:/Users/grant/vla/models/dinov3-vitb" }
if (-not $env:BERT_PATH) { $env:BERT_PATH = "C:/Users/grant/vla/models/bert-base-uncased" }

$bashPath = (@($CondaScripts, $CondaBin, $ShimDir) | ForEach-Object {
  ($_ -replace '^C:', '/c') -replace '\\', '/'
}) -join ':'
$cmd = "export PATH=${bashPath}:`$PATH; cd C:/Users/grant/vla/RoboDojo/XPolicyLab/policy/TurboVLA && ./setup_eval_policy_server.sh RoboDojo $Task $Ckpt $EnvCfg $ActionType $Seed $PolicyGpu turbovla-robodojo $Port localhost"
& $GitBash -c $cmd
