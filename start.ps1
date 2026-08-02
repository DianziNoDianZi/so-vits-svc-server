$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Prefer project venv, otherwise use system python
$python = $null
foreach ($candidate in @('server\venv\Scripts\python.exe', 'venv\Scripts\python.exe', '..\so-vits-svc\venv\Scripts\python.exe')) {
    if (Test-Path -LiteralPath $candidate) {
        $python = (Resolve-Path -LiteralPath $candidate).Path
        break
    }
}
if (-not $python) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $python) {
    Write-Error 'Python not found. Install Python 3.9+ or create a venv.'
    exit 1
}

Write-Host 'Starting So-VITS-SVC server...'
Write-Host 'URL: http://localhost:5000'
Write-Host 'Press Ctrl+C to stop.'
& $python 'server\app.py'
