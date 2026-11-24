# Recreates the Deadtree environment on a Windows machine with an NVIDIA GPU.
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$TorchIndex = "https://download.pytorch.org/whl/cu126"
$LockFile = "requirements.lock.txt"
$Weights = "data\segformer_b5_full_epoch_100.safetensors"
$WeightsUrl = "https://drive.google.com/file/d/1ZvUkBNVtXFujHo_amGyhSXWnPiAc_VES/view"
$Python = ".venv\Scripts\python.exe"

function Assert-Success($What) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $What (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# PyTorch contains deeply nested files that exceed Windows' 260 character path
# limit when the project folder itself is too deep.
$LongPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -ErrorAction SilentlyContinue).LongPathsEnabled
if ($PSScriptRoot.Length -gt 90 -and $LongPaths -ne 1) {
    Write-Host "This folder's path is too long ($($PSScriptRoot.Length) characters, limit 90). Move the project somewhere shorter, such as C:\Projects\Deadtree." -ForegroundColor Red
    exit 1
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python launcher 'py' not found. Install Python 3.12 from https://www.python.org/downloads/ and run this again." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    py -3.12 -m venv .venv
    Assert-Success "creating .venv (is Python 3.12 installed?)"
}

& $Python -m pip install --upgrade pip
Assert-Success "upgrading pip"

# The CUDA build is only on the PyTorch index; PyPI has the CPU build on Windows.
Write-Host "Installing PyTorch with CUDA (several GB)..."
$TorchPackages = @(Get-Content $LockFile | Where-Object { $_ -match '^(torch|torchvision)==' })
& $Python -m pip install $TorchPackages --index-url $TorchIndex
Assert-Success "installing PyTorch"

Write-Host "Installing the remaining packages..."
& $Python -m pip install -r $LockFile
Assert-Success "installing $LockFile"

# web\uploads and web\outputs are gitignored, so a fresh clone lacks them
New-Item -ItemType Directory -Force data, web\uploads, web\outputs | Out-Null

Write-Host ""
& $Python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Assert-Success "importing PyTorch"

if (-not (Test-Path $Weights)) {
    Write-Host "Deadwood weights are missing. Restore $Weights from backup or download it from:" -ForegroundColor Yellow
    Write-Host "  $WeightsUrl"
}

Write-Host ""
Write-Host "Setup finished. Start the app with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m flask --app web.app run --host 127.0.0.1 --port 5000"
Write-Host "Then open http://localhost:5000"
