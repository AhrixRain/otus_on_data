# Wait for Run_H_D3b to reach global epoch 20, then run the stage-1 mechanism readout.
# Read-only with respect to the training run: it only reads history.json and checkpoints
# and writes into outputs/cms_Joint/d3b_readout_probe/stage1_g20.
#
#   pwsh -NoProfile -File scripts_joint/d3b_epoch20_monitor.ps1
#
# Calibration (2026-10-04, artifact-measured; see
# outputs/cms_Joint/d3b_readout_probe/REPORT.md):
#   A2frozen / D3zcycle stage-1 (channel off) : 29.2 MeV spread, 29.3 MeV robust
#   D3b stage-1 (channel on from step 1)      : 18.5 MeV spread, 10.5 MeV robust
#   prior width (std / robust)                : 14.9 / 0.65 MeV
#   CMS width  (std / robust)                 : 28.0 / 30.0 MeV
# PASS requires the mean map to stay near the PRIOR width, not merely below CMS:
#   zero-noise spread < 21 MeV  AND  zero-noise robust width < 18 MeV,
#   and no checkpoint may exceed 24 MeV spread (the channel-off baseline sits at 29).
$maxStdMeV = 21.0
$maxRobMeV = 18.0
$hardFailMeV = 24.0

# Exit codes: 0 = training already finished before epoch 20 / monitor loop ended,
# 2 = probe ran and the mechanism check FAILED (stop training),
# 3 = probe ran and the mechanism check PASSED (continue training),
# 4 = probe ran but the checkpoint or the JSON was unreadable.

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repo

$runDir = Join-Path $repo 'outputs\cms_Joint\Run_H_D3b'
$history = Join-Path $runDir 'history.json'
$probeDir = Join-Path $repo 'outputs\cms_Joint\d3b_readout_probe\stage1_g20'
$monitorLog = Join-Path $probeDir 'monitor.log'
$verdictPath = Join-Path $probeDir 'VERDICT.txt'
$ckpt = Join-Path $runDir 'best_RunHD3b_stage1_deterministic_warmup.pt'
$ckptRel = 'outputs/cms_Joint/Run_H_D3b/best_RunHD3b_stage1_deterministic_warmup.pt'

New-Item -ItemType Directory -Force -Path $probeDir | Out-Null

function Log([string]$message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $message
    Write-Host $line
    Add-Content -Path $monitorLog -Value $line -Encoding utf8
}

function Get-CompletedEpoch {
    if (-not (Test-Path $history)) { return -1 }
    try {
        $rows = Get-Content $history -Raw | ConvertFrom-Json
    } catch {
        return -1   # partially written file, retry next poll
    }
    if ($null -eq $rows) { return -1 }
    if ($rows -isnot [System.Collections.IEnumerable] -or $rows -is [string]) { $rows = @($rows) }
    $epochs = @($rows | ForEach-Object { [int]$_.global_epoch })
    if ($epochs.Count -eq 0) { return -1 }
    return ($epochs | Measure-Object -Maximum).Maximum
}

Log "monitor started; waiting for global epoch 20 in $history"

$deadline = (Get-Date).AddHours(3)
$epoch = -1
while ((Get-Date) -lt $deadline) {
    $epoch = Get-CompletedEpoch
    if ($epoch -ge 20) { break }
    $training = Get-Process python* -ErrorAction SilentlyContinue
    if ($null -eq $training -and $epoch -lt 20) {
        Log "no python process and epoch=$epoch -> training ended early; exiting"
        exit 0
    }
    Start-Sleep -Seconds 60
}

if ($epoch -lt 20) {
    Log "timed out waiting for epoch 20 (last epoch=$epoch); exiting"
    exit 0
}

Log "epoch $epoch reached; probing $ckptRel"

$pyArgs = @(
    'scripts_joint/d3b_readout_probe.py',
    '--device', 'cuda',
    '--events', '24000',
    '--draws', '24',
    '--output-dir', 'outputs/cms_Joint/d3b_readout_probe/stage1_g20',
    '--checkpoint', "D3b_stage1_g$($epoch)=$ckptRel"
)
$stdout = & python @pyArgs 2>&1
$stdout | ForEach-Object { Add-Content -Path $monitorLog -Value $_ -Encoding utf8 }
if ($LASTEXITCODE -ne 0) { Log "probe exited $LASTEXITCODE; falling back to CPU" }
if ($LASTEXITCODE -ne 0) {
    $pyArgs[3] = 'cpu'
    $stdout = & python @pyArgs 2>&1
    $stdout | ForEach-Object { Add-Content -Path $monitorLog -Value $_ -Encoding utf8 }
}

$jsonPath = Join-Path $probeDir 'd3b_readout.json'
if (-not (Test-Path $jsonPath)) {
    Log "probe produced no JSON; exiting"
    exit 4
}

$payload = Get-Content $jsonPath -Raw | ConvertFrom-Json
$entry = $payload.checkpoints.PSObject.Properties | Select-Object -First 1
$zeroStd = [double]$entry.Value.zero_noise.ensemble_std_gev * 1000.0
$zeroRob = [double]$entry.Value.zero_noise.ensemble_robust_half_width_gev * 1000.0
$nativeStd = [double]$entry.Value.native_noise.ensemble_std_gev * 1000.0
$cmsStd = [double]$payload.reference.cms_mass_std_gev * 1000.0

$pass = ($zeroStd -lt $maxStdMeV) -and ($zeroRob -lt $maxRobMeV) -and ($zeroStd -lt $hardFailMeV)
$verdict = if ($pass) { 'PASS' } else { 'FAIL' }

$text = @(
    "Run_H_D3b stage-1 mechanism readout at global epoch $epoch",
    "checkpoint : $ckptRel",
    "CMS x mass std (reference)      : {0:N2} MeV" -f $cmsStd,
    "zero-noise decoded spread       : {0:N2} MeV  (mean-map contribution)" -f $zeroStd,
    "zero-noise decoded robust width : {0:N2} MeV" -f $zeroRob,
    "native-noise decoded spread     : {0:N2} MeV" -f $nativeStd,
    "",
    "criterion: zero-noise spread < $maxStdMeV MeV and robust width < $maxRobMeV MeV",
    "(PASS = the mean map stays near the prior width, FAIL = it absorbed the resolution).",
    "baseline for comparison (see ../../d3b_readout_probe/REPORT.md):",
    "   A2frozen / D3zcycle stage-1 (channel off) : 29.2 MeV spread, 29.3 MeV robust",
    "   D3b stage-1 (channel on from step 1)      : 18.5 MeV spread, 10.5 MeV robust",
    "   initial state, epoch 1                    : 15.5 MeV spread",
    "",
    "VERDICT: $verdict"
)
$text | Set-Content -Path $verdictPath -Encoding utf8
$text | ForEach-Object { Log $_ }

if ($pass) { exit 3 } else { exit 2 }
