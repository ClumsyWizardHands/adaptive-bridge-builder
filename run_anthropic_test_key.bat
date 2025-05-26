@echo off
echo Anthropic + Mistral Multi-Model Test Runner
echo ==========================================
echo.

REM Get the Anthropic API key from the user if not set
if "%ANTHROPIC_API_KEY%"=="" (
  set /p ANTHROPIC_API_KEY="Enter your Anthropic API key: "
)

REM Check the key format
if "%ANTHROPIC_API_KEY:~0,2%" NEQ "sk" (
  echo.
  echo ⚠️ Warning: The key you entered doesn't start with "sk", which is unusual for Anthropic keys.
  echo   Current key: %ANTHROPIC_API_KEY%
  echo.
  set /p CONTINUE="Continue anyway? (y/n): "
  if /i "%CONTINUE%" NEQ "y" (
    echo Exiting.
    exit /b 1
  )
)

echo.
echo Running the Anthropic API integration test...
echo.

REM Install required packages if needed
pip install -q aiohttp

REM Run the real test
python src/run_real_test.py

echo.
echo Done.
pause
