@echo off
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

set "HOST=127.0.0.1"
set "PORT=8000"
if not "%~1"=="" set "HOST=%~1"
if not "%~2"=="" set "PORT=%~2"

echo Finance Text-to-SQL API
echo   Base URL:   http://%HOST%:%PORT%
echo   Dashboard:  http://%HOST%:%PORT%/
echo   Vercel:     https://YOUR-PROJECT.vercel.app/?api=http://127.0.0.1:%PORT%
echo.
echo Loading the Qwen adapter; the first request can take a few seconds.
echo Press Ctrl+C to stop.
echo.

"%PY%" -m uvicorn api.main:app --host %HOST% --port %PORT%
