$ErrorActionPreference = 'Stop'
$flightlyRoot = Split-Path -Parent $PSScriptRoot
$flightlyData = Join-Path $flightlyRoot 'work\flightly-mlflow\shared'
$flightlyUv = Join-Path $env:APPDATA 'Python\Python312\Scripts\uv.exe'
$flightlyConfig = Join-Path $flightlyRoot 'work\flightly-feedback-sync.json'
if (-not (Test-Path -LiteralPath $flightlyConfig)) { throw 'Missing private feedback sync configuration' }
New-Item -Path $flightlyData -ItemType Directory -Force | Out-Null
$flightlyProcess = Start-Process -FilePath $flightlyUv `
    -ArgumentList @('run', '--script', 'scripts/shared_learning.py', 'sync', '--config', $flightlyConfig, '--watch') `
    -WorkingDirectory $flightlyRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $flightlyData 'sync.stdout.log') `
    -RedirectStandardError (Join-Path $flightlyData 'sync.stderr.log') `
    -Wait -PassThru
exit $flightlyProcess.ExitCode
