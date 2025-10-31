@echo off
setlocal enabledelayedexpansion
set PRESET=%1
if "!PRESET!"=="" set PRESET=gpt1
set VENV_DIR=.venv
if not exist "!VENV_DIR!" (
    python -m venv "!VENV_DIR!"
)
call "!VENV_DIR!\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m myllm.cli prepare-data --preset !PRESET!
python -m myllm.cli train --preset !PRESET!
