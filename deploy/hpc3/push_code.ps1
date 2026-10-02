<#
.SYNOPSIS
    Copy the OTUS checkout to UCI HPC3 with scp.

.DESCRIPTION
    There is no rsync on this workstation (checked: not on PATH and not in the
    Git for Windows install), so this uses the portable three-step route:

        1. tar  - build one archive locally, with an exclude list
        2. scp  - move that single file to the data-transfer host
        3. ssh  - extract it on the cluster

    A single archive is also the reason excludes work at all: `scp -r` has no
    exclude option, and copying 3,000+ files one by one is both slow and
    non-idempotent.

    The exclude list exists so that a code push never drags along things the
    cluster either must not have in $HOME-shaped storage or does not need:

        .git/        1.34 GB of history      (use -WithGit when you want git here)
        data/        4.6 GB of open data     (stage it with stage_data.sh instead)
        outputs/     10 GB of run artifacts  (these flow back, not forward)
        *.pt *.hdf5 *.npz *.npy *.root       heavy artifacts
        __pycache__ .venv .region_cache      caches

    Repeat pushes are cheap: the archive is source, configs, tests, notebooks
    and the tracked PNG plots, roughly 30-60 MB.

    This script does not run git. To keep a real repository on the cluster
    (needed if you want to commit run artifacts there yourself), either pass
    -WithGit, or -- better -- clone once and rsync/tar the code over that
    checkout afterwards.

.EXAMPLE
    # see the archive size and file count without transferring anything
    .\push_code.ps1 -DryRun

.EXAMPLE
    # push the code to /pub/$USER/otus
    .\push_code.ps1

.EXAMPLE
    # also carry .git so you can run git on the cluster (one-time, slow)
    .\push_code.ps1 -WithGit
#>
[CmdletBinding()]
param(
    [string]$User = $env:USERNAME,
    [string]$HpcHost = 'access-hpc3.rcic.uci.edu',
    [string]$RemoteParent = '',
    [string]$LocalRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$RemoteDirName = 'otus',
    [switch]$WithGit,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RemoteParent)) {
    $RemoteParent = "/pub/$User"
}
$RemoteDir = "$RemoteParent/$RemoteDirName"

# tar patterns, relative to the repository root.
$excludes = @(
    '--exclude=./outputs'
    '--exclude=./data'
    '--exclude=./logs'
    '--exclude=./__pycache__'
    '--exclude=*/__pycache__'
    '--exclude=*.pyc'
    '--exclude=./.venv'
    '--exclude=./.region_cache'
    '--exclude=*/.region_cache'
    '--exclude=./.ipynb_checkpoints'
    '--exclude=*/.ipynb_checkpoints'
    '--exclude=./.mypy_cache'
    '--exclude=./.pytest_cache'
    '--exclude=*.pt'
    '--exclude=*.pth'
    '--exclude=*.ckpt'
    '--exclude=*.safetensors'
    '--exclude=*.hdf5'
    '--exclude=*.h5'
    '--exclude=*.npz'
    '--exclude=*.npy'
    '--exclude=*.root'
    '--exclude=*.zip'
)
if (-not $WithGit) {
    $excludes += '--exclude=./.git'
}

function Invoke-Checked {
    param([string]$What, [string[]]$Command)
    Write-Host "==> $What" -ForegroundColor Cyan
    Write-Host "    $($Command -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $Command[0] @($Command[1..($Command.Count - 1)])
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath (Join-Path $LocalRepo '.git'))) {
    Write-Warning "$LocalRepo has no .git; this will still copy the working tree."
}

$archive = Join-Path $env:TEMP "otus-code-$(Get-Date -Format yyyyMMdd-HHmmss).tgz"

# `tar -C <parent> <dirname>` keeps the archive rooted at the checkout name, so
# extraction lands in one predictable directory.
$parent = Split-Path -Parent $LocalRepo
$leaf = Split-Path -Leaf $LocalRepo

Invoke-Checked "building the archive" (@(
    'tar', '-czf', $archive, '-C', $parent
) + $excludes + @($leaf))

if (Test-Path -LiteralPath $archive) {
    $mb = [math]::Round((Get-Item -LiteralPath $archive).Length / 1MB, 1)
    $count = (tar -tzf $archive | Measure-Object -Line).Lines
    Write-Host "==> archive: $mb MB, $count entries" -ForegroundColor Green
    Write-Host "    $archive"
    if ($WithGit) {
        Write-Host "    (-WithGit: .git is included, so expect this to be ~1 GB and slow)" -ForegroundColor Yellow
    }
}

if ($DryRun) {
    Write-Host "dry-run: nothing transferred, archive kept at $archive" -ForegroundColor Yellow
    return
}

$target = "${User}@${HpcHost}"
$leafName = Split-Path -Leaf $archive
Invoke-Checked "creating $RemoteParent on the cluster" @('ssh', $target, "mkdir -p '$RemoteParent'")
Invoke-Checked "scp the archive" @('scp', $archive, "${target}:/pub/$User/")
Invoke-Checked "extracting into $RemoteParent" @(
    'ssh', $target, "cd '$RemoteParent' && tar -xzf '$leafName' && rm -f '$leafName'"
)

Remove-Item -LiteralPath $archive -Force
Write-Host ""
Write-Host "==> done. Next, on the cluster:" -ForegroundColor Green
Write-Host "    ssh $target"
Write-Host "    cd $RemoteDir"
Write-Host "    python deploy/hpc3/to_lf.py --apply        # scp copies CRLF verbatim; git would not"
Write-Host "    python deploy/hpc3/check_linux_compat.py   # verifies the checkout before you spend a GPU-hour"
