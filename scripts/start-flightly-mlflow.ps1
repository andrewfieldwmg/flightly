$ErrorActionPreference = 'Stop'

$flightlyRoot = Split-Path -Parent $PSScriptRoot
$flightlyData = Join-Path $flightlyRoot 'work\flightly-mlflow'
$flightlyUv = Join-Path $env:APPDATA 'Python\Python312\Scripts\uv.exe'

if (-not (Test-Path -LiteralPath $flightlyUv)) {
    throw "Cannot find uv at $flightlyUv"
}

New-Item -Path $flightlyData -ItemType Directory -Force | Out-Null
$env:MLFLOW_SERVER_ALLOWED_HOSTS = '127.0.0.1,127.0.0.1:5505,localhost,localhost:5505,desktop-u6i86iq.tail53b14b.ts.net'
$env:MLFLOW_SERVER_CORS_ALLOWED_ORIGINS = 'http://127.0.0.1:5505,http://localhost:5505,https://desktop-u6i86iq.tail53b14b.ts.net'
$flightlyProcess = Start-Process -FilePath $flightlyUv `
    -ArgumentList @('run', '--script', 'scripts/flightly_mlflow.py', 'serve') `
    -WorkingDirectory $flightlyRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $flightlyData 'server.stdout.log') `
    -RedirectStandardError (Join-Path $flightlyData 'server.stderr.log') `
    -Wait -PassThru

exit $flightlyProcess.ExitCode
