<#
.SYNOPSIS
    Push the runtime half of the repo to the board and verify it landed.

.DESCRIPTION
    Only what the board needs is copied: the pipeline, the benchmark scripts,
    the configs and the model. The dataset, the venv, and the host-only export
    tooling stay on the laptop.

.EXAMPLE
    .\0-setup\deploy.ps1 -DeviceIp 192.168.1.77
    .\0-setup\deploy.ps1 -DeviceIp 192.168.1.77 -Adb        # push over USB instead
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)][string]$DeviceIp = "",
    [string]$User = "root",
    [string]$RemoteDir = "/opt/skysentry",
    [switch]$Adb,
    [string]$AdbPath = "$env:USERPROFILE\tools\platform-tools\adb.exe",
    [switch]$IncludeSampleClip
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$items = @(
    "3-pipeline",
    "4-bench",
    "2-augment",
    "5-server",
    "configs",
    "models",
    "requirements-device.txt",
    "0-setup/device_setup.sh"
)
if ($IncludeSampleClip) { $items += "assets" }

Write-Host "Repo root : $RepoRoot"
Write-Host "Payload   : $($items -join ', ')"

# --------------------------------------------------------------------------- #
if ($Adb) {
    if (-not (Test-Path $AdbPath)) { throw "adb not found at $AdbPath" }
    $dev = & $AdbPath devices | Select-Object -Skip 1 | Where-Object { $_ -match '\S' }
    if (-not $dev) { throw "no adb device attached" }

    Write-Host "`nPushing over adb..." -ForegroundColor Cyan
    & $AdbPath shell "mkdir -p $RemoteDir"
    foreach ($it in $items) {
        $src = Join-Path $RepoRoot $it
        if (-not (Test-Path $src)) { Write-Host "  skip (missing): $it"; continue }
        Write-Host "  push $it"
        & $AdbPath push $src "$RemoteDir/" | Out-Null
    }
    Write-Host "`nVerify:" -ForegroundColor Green
    & $AdbPath shell "ls -la $RemoteDir"
    exit 0
}

# --------------------------------------------------------------------------- #
if (-not $DeviceIp) { throw "Provide -DeviceIp, or use -Adb. Run 0-setup\find_device.ps1 first." }

Write-Host "`nTarget: $User@$DeviceIp`:$RemoteDir" -ForegroundColor Cyan
Write-Host "(password for the HSPTEK / RB3 images is usually 'oelinux123')" -ForegroundColor DarkGray

ssh "$User@$DeviceIp" "mkdir -p $RemoteDir"
if ($LASTEXITCODE -ne 0) { throw "ssh failed - is the IP right and the board up?" }

foreach ($it in $items) {
    $src = Join-Path $RepoRoot $it
    if (-not (Test-Path $src)) { Write-Host "  skip (missing): $it" -ForegroundColor DarkYellow; continue }
    Write-Host "  scp $it"
    scp -r -q $src "$User@$DeviceIp`:$RemoteDir/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed on $it" }
}

Write-Host "`nRemote listing:" -ForegroundColor Green
ssh "$User@$DeviceIp" "ls -la $RemoteDir && echo '--- python ---' && (python3 -V || echo 'python3 MISSING')"

Write-Host @"

Next on the board:
  ssh $User@$DeviceIp
  cd $RemoteDir
  bash device_setup.sh                 # installs deps, probes the runtime
  python3 4-bench/probe_power.py --discover
  python3 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx --iters 50
"@ -ForegroundColor Green
