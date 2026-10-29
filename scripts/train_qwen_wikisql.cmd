@echo off
REM Start (or restart) the Qwen2.5-Coder + WikiSQL QLoRA run.
REM Progress goes to logs\train_qwen_wikisql.log; watch it with
REM scripts\watch_training.cmd in a separate terminal.
setlocal
cd /d "%~dp0.."

set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set TQDM_MININTERVAL=2
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set HF_HOME=D:\HuggingFaceCache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set MLFLOW_TRACKING_URI=file:///%CD:\=/%/mlruns

if not exist logs mkdir logs
echo. >> logs\train_qwen_wikisql.log
echo ==== training start %DATE% %TIME% ==== >> logs\train_qwen_wikisql.log

".venv\Scripts\python.exe" -m src.training.train ^
  --output-dir models\qwen2.5-coder-1.5b-wikisql-sql %* >> logs\train_qwen_wikisql.log 2>&1

echo EXITCODE=%ERRORLEVEL% >> logs\train_qwen_wikisql.log
endlocal
