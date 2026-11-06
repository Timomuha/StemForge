@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

for /d %%P in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared*") do for /d %%D in ("%%~P\ffmpeg-*-full_build-shared") do if exist "%%~D\bin\ffprobe.exe" set "PATH=%%~D\bin;%PATH%"

"%PYTHON%" -c "import fastapi, uvicorn, multipart, torch, demucs, numpy" >nul 2>nul
if errorlevel 1 (
    echo Backend dependencies are missing.
    echo Install them with:
    echo   "%PYTHON%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting StemForge API at http://127.0.0.1:8000
"%PYTHON%" -m uvicorn main:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
