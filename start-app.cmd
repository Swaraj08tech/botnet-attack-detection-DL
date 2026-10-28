@echo off
rem Starts the backend and frontend and opens the browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_app.ps1"
pause
