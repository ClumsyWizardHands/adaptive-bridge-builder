@echo off
:: =========================================================
:: GitHub Upload Script for Windows
::
:: This script commits and pushes the A2A server and related changes
:: to GitHub repository
:: =========================================================

echo GitHub Upload Script
echo ===================
echo.

:: Check if git is installed
git --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Git not found. Please install Git and try again.
    exit /b 1
)

echo Checking repository status...
git status

echo.
echo Adding new files...
git add src/http_server.py
git add src/mock_server.py
git add src/test_endpoints.py
git add src/agent_card.json
git add run_server.bat
git add run_mock_server.bat
git add requirements.txt

echo.
set commit_msg="Add A2A Protocol server implementation and mock server"

echo.
echo Committing changes with message: %commit_msg%
git commit -m %commit_msg%

echo.
echo Pushing changes to GitHub...
git push

echo.
echo Upload completed! Check GitHub repository at:
echo https://github.com/ClumsyWizardHands/adaptive-bridge-builder
echo.

pause
