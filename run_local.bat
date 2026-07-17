@echo off
REM Run the Rainbow Paint WhatsApp bot LOCALLY on this desktop with Ollama (free, on-PC AI).
REM Requires: Ollama installed + qwen2.5:0.5b pulled, and Python venv with flask/requests/openai.
cd /d "%~dp0"
set VENV_PY=%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe
if not exist "%VENV_PY%" set VENV_PY=python
REM Load local env (Ollama primary, Meta token, etc.)
set DOTENV=%~dp0.env.local
for /f "usebackq tokens=1,* delims==" %%A in ("%DOTENV%") do (
  if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
)
echo [run_local] BOT_MODEL=%BOT_MODEL%  PORT=%PORT%
"%VENV_PY%" whatsapp_bot.py
