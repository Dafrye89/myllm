@echo off
setlocal enabledelayedexpansion

set VENV_DIR=.venv
if not exist "!VENV_DIR!" (
    python -m venv "!VENV_DIR!"
)
call "!VENV_DIR!\Scripts\activate.bat"

python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt >nul

python scripts\generate_synthetic.py %*
