@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%" >nul

set "VENV_DIR=%PROJECT_ROOT%\.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Could not find virtual environment at %VENV_DIR%
    popd >nul
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --quiet datasets tqdm
python scripts\download_fineweb.py
set "ERR=%ERRORLEVEL%"

call deactivate >nul 2>&1
popd >nul
exit /b %ERR%
