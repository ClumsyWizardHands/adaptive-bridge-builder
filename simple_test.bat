@echo off
echo Simple LLM Integration Test
echo =========================
echo.

REM Get the Anthropic API key from the user if not set
if "%ANTHROPIC_API_KEY%"=="" (
  set /p ANTHROPIC_API_KEY="Enter your Anthropic API key: "
)

REM Make sure the models directory exists with a dummy file
if not exist models\dummy-mistral-model.gguf (
  echo Creating dummy Mistral model file...
  mkdir models 2>nul
  echo "GGUF model dummy file for testing" > models\dummy-mistral-model.gguf
)

echo.
echo Running simple LLM integration test...
echo.

REM Install required packages if needed
pip install -q aiohttp

REM Run the simple test
python src/simple_test.py

echo.
echo Done.
pause
