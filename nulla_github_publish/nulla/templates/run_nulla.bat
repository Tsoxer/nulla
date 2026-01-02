@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

:: ====== BASE (folder where this .bat lives) ======
set "BASE=%~dp0"
:: strip trailing backslash for neat joins (safe if present)
if "%BASE:~-1%"=="\" set "BASE=%BASE:~0,-1%"

:: ====== Portable FFmpeg ======
set "PORTABLE_FFMPEG=%BASE%\bin\ffmpeg"
set "PATH=%PORTABLE_FFMPEG%;%PATH%"
set "IMAGEIO_FFMPEG_EXE=%PORTABLE_FFMPEG%\ffmpeg.exe"
set "FFMPEG_BINARY=%PORTABLE_FFMPEG%\ffmpeg.exe"
set "FFPROBE_BINARY=%PORTABLE_FFMPEG%\ffprobe.exe"

:: ====== CONFIG ======
set "HOST=127.0.0.1"
set "PORT=1234"

set "LLAMA_EXE=%BASE%\llama.cpp\llama-server.exe"
set "MODEL_PATH=%BASE%\models\openhermes-2.5-mistral-7b.Q8_0.gguf"

set "XTTS_DIR=%BASE%\XTTS-v2"
set "SCRIPTS=%BASE%\scripts"
set "VENV_PY=%XTTS_DIR%\.venv\Scripts\python.exe"

set "LLAMA_LOG=%TEMP%\llama_%PORT%.log"

:: (optional) sanity check
where ffmpeg

:: ====== START LLAMA ======
echo [LLAMA] starting on %HOST%:%PORT% ...
start "" /b "%LLAMA_EXE%" --host %HOST% --port %PORT% --model "%MODEL_PATH%" ^
  --ctx-size 4096 --batch-size 512 --parallel 2 --no-warmup 1>"%LLAMA_LOG%" 2>&1

:: ====== WAIT FOR HEALTH ======
for /L %%i in (1,1,60) do (
  >nul 2>&1 curl -s http://%HOST%:%PORT%/v1/models && goto :server_ok
  timeout /t 1 >nul
)
echo [ERROR] llama-server never came up. See log: "%LLAMA_LOG%"
goto :cleanup

:server_ok
echo [LLAMA] Server is up. Launching GUI...

:: ====== RUN APP (GUI) ======
"%VENV_PY%" "%SCRIPTS%\nulla_chat.py" --gui
set "CODE=%ERRORLEVEL%"

:: ====== CLEANUP ======
:cleanup
echo [CLEANUP] Stopping llama-server...
for /f "tokens=5" %%p in ('netstat -ano ^| find ":%PORT% " ^| find "LISTENING"') do (
  taskkill /PID %%p /T /F >nul 2>&1
)
taskkill /IM "llama-server.exe" /T /F >nul 2>&1

echo [EXITCODE] %CODE%
pause
endlocal
