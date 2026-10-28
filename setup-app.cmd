@echo off
rem One-time setup of the web application (see docs/DEPLOYMENT.md).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_app.ps1"
pause
