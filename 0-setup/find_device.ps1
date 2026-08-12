<#
.SYNOPSIS
    Locate a Qualcomm edge board on the local network or over USB.

.DESCRIPTION
    The board can appear three ways, and this script checks all three in the
    order that is most likely to work:

      1. RJ45 / Wi-Fi  -> an SSH server on port 22 somewhere in the local /24
      2. USB Type-C    -> an adb device
      3. USB RNDIS     -> a link-local 169.254.x.x peer

    Prints every candidate rather than guessing, because on a venue network
    there will be other Linux hosts answering on 22.

.EXAMPLE
    .\0-setup\find_device.ps1
    .\0-setup\find_device.ps1 -Subnet 192.168.1 -TimeoutMs 300
#>
[CmdletBinding()]
param(
    [string]$Subnet = "",
    [int]$TimeoutMs = 250,
    [string]$AdbPath = "$env:USERPROFILE\tools\platform-tools\adb.exe"
)

$ErrorActionPreference = "Continue"

function Write-Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# --------------------------------------------------------------------------- #
Write-Section "Local interfaces"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch '^127\.' } |
    Select-Object InterfaceAlias, IPAddress, PrefixLength |
    Format-Table -AutoSize

if (-not $Subnet) {
    $primary = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -match '^(192\.168|10\.|172\.(1[6-9]|2\d|3[01]))' } |
        Sort-Object -Property SkipAsSource |
        Select-Object -First 1
    if ($primary) {
        $Subnet = ($primary.IPAddress -split '\.')[0..2] -join '.'
        Write-Host "Auto-detected subnet: $Subnet.0/24" -ForegroundColor Yellow
    }
}

# --------------------------------------------------------------------------- #
Write-Section "ADB (USB Type-C)"
if (Test-Path $AdbPath) {
    & $AdbPath start-server 2>&1 | Out-Null
    $devices = & $AdbPath devices -l | Select-Object -Skip 1 |
        Where-Object { $_ -match '\S' }
    if ($devices) {
        Write-Host "adb devices found:" -ForegroundColor Green
        $devices | ForEach-Object { Write-Host "  $_" }
        Write-Host "`n  Open a shell with:  $AdbPath shell" -ForegroundColor Green
    } else {
        Write-Host "  no adb devices." -ForegroundColor DarkYellow
        Write-Host "  If the board is plugged in over Type-C, the usual causes are:"
        Write-Host "    - the cable is charge-only (no data pair) -- try another cable"
        Write-Host "    - the Qualcomm Userspace Driver is missing"
        Write-Host "    - the cable is in the power port, not the data port"
    }
} else {
    Write-Host "  adb not found at $AdbPath" -ForegroundColor DarkYellow
    Write-Host "  Install: https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
}

# --------------------------------------------------------------------------- #
if ($Subnet) {
    Write-Section "Scanning $Subnet.0/24 for SSH (port 22)"
    Write-Host "  (this takes ~15-30 s)" -ForegroundColor DarkGray

    $jobs = 1..254 | ForEach-Object {
        $ip = "$Subnet.$_"
        [pscustomobject]@{
            IP     = $ip
            Client = New-Object System.Net.Sockets.TcpClient
            Async  = $null
        }
    }
    foreach ($j in $jobs) {
        try { $j.Async = $j.Client.BeginConnect($j.IP, 22, $null, $null) } catch {}
    }
    Start-Sleep -Milliseconds $TimeoutMs

    $hits = @()
    foreach ($j in $jobs) {
        try {
            if ($j.Async -and $j.Async.IsCompleted -and $j.Client.Connected) {
                $hits += $j.IP
            }
        } catch {}
        try { $j.Client.Close() } catch {}
    }

    if ($hits) {
        Write-Host "`n  Hosts with SSH open:" -ForegroundColor Green
        foreach ($h in $hits) {
            $mac = (Get-NetNeighbor -IPAddress $h -ErrorAction SilentlyContinue |
                    Select-Object -First 1).LinkLayerAddress
            $name = try { [System.Net.Dns]::GetHostEntry($h).HostName } catch { "" }
            Write-Host ("    {0,-16} mac={1,-18} {2}" -f $h, ($mac ?? "?"), $name)
        }
        Write-Host "`n  Match the MAC against the label on the device box, then:" -ForegroundColor Green
        Write-Host "    ssh root@<ip>        # password: oelinux123"
    } else {
        Write-Host "  no SSH hosts found on $Subnet.0/24" -ForegroundColor DarkYellow
        Write-Host "  Check that the board's Ethernet cable is in the same router/switch."
    }
}

# --------------------------------------------------------------------------- #
Write-Section "Link-local peers (USB RNDIS)"
$ll = Get-NetNeighbor -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -match '^169\.254\.' -and $_.State -match 'Reachable|Stale' }
if ($ll) { $ll | Select-Object IPAddress, LinkLayerAddress, InterfaceAlias | Format-Table -AutoSize }
else { Write-Host "  none" -ForegroundColor DarkYellow }

Write-Host ""
