# One-time setup for running the web application locally on Windows.
# Safe to run again: every step is skipped when it is already done.
#   1. Python environment (.venv) with the backend packages
#   2. Frontend packages (frontend/node_modules)
#   3. Model bundle (serving/bundle): unzip serving/serving-bundle.zip, or build it from models/v3
#   4. backend/.env with a random JWT secret and the first administrator account
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }

# 1. Python environment
Step 'Python environment'
if (-not (Test-Path $Python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Python 3.11 is required: install it from https://www.python.org/downloads/ and run this script again.' }
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv with Python 3.11.' }
}
& $Python -c "import fastapi, uvicorn, sqlalchemy, jwt, bcrypt, multipart, lightgbm, pandas, email_validator" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install -r backend/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installing the backend packages failed.' }
}
Ok 'backend packages installed'

# 2. Frontend packages
Step 'Frontend packages'
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'Node.js 22 is required: install it from https://nodejs.org/ and run this script again.' }
if (-not (Test-Path 'frontend\node_modules')) {
    npm --prefix frontend ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
}
Ok 'frontend packages installed'

# 3. Model bundle
Step 'Model bundle'
if (-not (Test-Path 'serving\bundle\manifest.json')) {
    if (Test-Path 'serving\serving-bundle.zip') {
        Expand-Archive 'serving\serving-bundle.zip' -DestinationPath 'serving\bundle' -Force
    } elseif (Test-Path 'models\v3\final_detector_v2.json') {
        & $Python scripts/build_serving_bundle.py
        if ($LASTEXITCODE -ne 0) { throw 'Building the model bundle failed.' }
    } else {
        throw 'No model bundle: download serving-bundle.zip from the GitHub release into the serving folder and run this script again.'
    }
}
# Per-class reference used by the report reasoning (optional: the app works without it).
if (-not (Test-Path 'serving\bundle\class_reference.json') -and (Test-Path 'data\raw')) {
    & $Python scripts/build_class_reference.py
    if ($LASTEXITCODE -ne 0) { Write-Host '  class_reference.json could not be built; reports will skip the typical-value point.' -ForegroundColor Yellow }
}
Ok 'serving/bundle is ready'

# 4. Local settings and first administrator
Step 'Local settings (backend/.env)'
if (-not (Test-Path 'backend\.env')) {
    $email = Read-Host 'Administrator e-mail [admin@example.com]'
    if (-not $email) { $email = 'admin@example.com' }
    do {
        $secure = Read-Host 'Administrator password (at least 10 characters)' -AsSecureString
        $password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    } while ($password.Length -lt 10)
    $secret = & $Python -c "import secrets; print(secrets.token_urlsafe(48))"
    $values = @{ 'JWT_SECRET' = $secret; 'ADMIN_EMAIL' = $email; 'ADMIN_PASSWORD' = $password }
    $lines = foreach ($line in Get-Content 'backend\.env.example') {
        $key = ($line -split '=', 2)[0]
        if ($values.ContainsKey($key)) { "$key=$($values[$key])" } else { $line }
    }
    [IO.File]::WriteAllLines((Join-Path $Root 'backend\.env'), [string[]]$lines)
    Ok "created backend/.env (sign in as $email)"
} else {
    Ok 'backend/.env already exists'
}

# Optional: local LLM for incident reports
Step 'Local LLM for reports (optional)'
try {
    $tags = Invoke-RestMethod 'http://localhost:11434/api/tags' -TimeoutSec 3
    if ($tags.models.name -contains 'qwen2.5:7b') { Ok 'Ollama is running with qwen2.5:7b' }
    else { Write-Host '   Ollama is running; download the model once with: ollama pull qwen2.5:7b' -ForegroundColor Yellow }
} catch {
    Write-Host '   Ollama is not running: reports will be evidence-only. Install it from https://ollama.com and run: ollama pull qwen2.5:7b' -ForegroundColor Yellow
}

Write-Host "`nSetup complete. Start the application with: start-app.cmd" -ForegroundColor Green
