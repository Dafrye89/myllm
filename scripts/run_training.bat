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
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    where nvidia-smi >nul 2>&1
    set "GPU_STATUS=!errorlevel!"
    if "!GPU_STATUS!"=="0" (
        echo Installing CUDA-enabled PyTorch build from the official wheel index...
        python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch
    ) else (
        echo NVIDIA GPU not detected; keeping CPU-only PyTorch.
    )
)
python -m pip install -e .
python -m myllm.cli prepare-data --preset !PRESET!
python -m myllm.cli train --preset !PRESET!
