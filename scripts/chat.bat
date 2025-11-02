@echo off
setlocal enabledelayedexpansion

set VENV_DIR=.venv
if not exist "!VENV_DIR!" (
    python -m venv "!VENV_DIR!"
)
call "!VENV_DIR!\Scripts\activate.bat"

if "%~1"=="" (
    set "CHECKPOINT=outputs\gpt1\latest.pt"
) else (
    set "CHECKPOINT=%~1"
    shift
)

python scripts\chat.py --preset gpt1 --checkpoint "!CHECKPOINT!" %*
