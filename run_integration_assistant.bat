@echo off
echo Starting Integration Assistant API...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q fastapi uvicorn[standard] websockets pydantic

REM Set Python path
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM Run the FastAPI app
echo.
echo Starting FastAPI server on http://localhost:8000
echo API Documentation available at http://localhost:8000/docs
echo WebSocket endpoint: ws://localhost:8000/ws
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.api.integration_assistant.app:app --reload --host 0.0.0.0 --port 8000
