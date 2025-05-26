@echo off
REM =========================================================
REM Multi-Model Agent Setup
REM
REM This script installs required dependencies and prepares
REM the environment for running the multi-model agent.
REM =========================================================

echo Multi-Model Agent Setup
echo ======================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.9+ and try again.
    goto :end
)

echo Installing required Python packages...
pip install -r requirements.txt

REM Offer to download a Mistral model if needed
if not exist models\*.gguf (
    echo.
    echo No Mistral model found in the models directory.
    choice /M "Would you like to download a Mistral model now"
    if errorlevel 2 goto :skip_download
    
    echo.
    echo Running model downloader...
    python src/download_mistral_model.py
) else (
    echo.
    echo Mistral model found in models directory.
)

:skip_download
echo.
echo Setup completed!
echo.
echo To run the multi-model agent:
echo 1. Run run_multi_model_agent.bat
echo 2. Enter your Claude API key when prompted (or set ANTHROPIC_API_KEY environment variable)
echo.
echo For more information, see MULTI_MODEL_AGENT_GUIDE.md
echo.

:end
pause
