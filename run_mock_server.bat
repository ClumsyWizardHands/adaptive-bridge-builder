@echo off
echo Starting Mock Bridge Server...

REM Install dependencies
echo Installing required dependencies...
pip install flask

REM Kill any existing server processes on port 8080
echo Checking for existing processes on port 8080...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
    echo Found process: %%a
    taskkill /F /PID %%a 2>nul
)

echo.
echo Starting HTTP Server on port 8080...
cd src
python mock_server.py

pause
