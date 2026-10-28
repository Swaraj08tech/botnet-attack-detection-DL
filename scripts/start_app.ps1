# Starts the backend (http://localhost:8000) and the frontend (http://localhost:5173) in two windows and opens the browser.
# Close the two windows to stop the application. Run setup-app.cmd once before the first start.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'

foreach ($need in @($Python, 'frontend\node_modules', 'serving\bundle\manifest.json', 'backend\.env')) {
    if (-not (Test-Path $need)) { throw "Missing $need - run setup-app.cmd first." }
}

Start-Process powershell -WorkingDirectory $Root -ArgumentList '-NoExit', '-Command',
    "`$Host.UI.RawUI.WindowTitle = 'Attack detection - backend'; & '$Python' -m uvicorn app.main:app --app-dir backend --port 8000"
Start-Process powershell -WorkingDirectory $Root -ArgumentList '-NoExit', '-Command',
    "`$Host.UI.RawUI.WindowTitle = 'Attack detection - frontend'; npm --prefix frontend run dev"

Write-Host 'Waiting for the backend to load the model...'
$ready = $false
foreach ($i in 1..120) {
    try { Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 10 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw 'The backend did not start within 2 minutes: check the backend window for the error.' }

Start-Process 'http://localhost:5173'
Write-Host 'Application running: http://localhost:5173   (API documentation: http://localhost:8000/docs)' -ForegroundColor Green
