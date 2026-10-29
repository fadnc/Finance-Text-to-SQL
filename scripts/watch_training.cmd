@echo off
REM Live progress dashboard for the Qwen2.5-Coder + WikiSQL run.
REM Recommended: open it in its own terminal window:
REM     start "" scripts\watch_training.cmd
cd /d "%~dp0.."
title Qwen2.5-Coder + WikiSQL - live training progress
".venv\Scripts\python.exe" scripts\watch_training.py %*
echo.
pause
