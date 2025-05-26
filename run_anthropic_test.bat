@echo off
echo Setting up test environment...

REM Set your Anthropic API key here or enter it when prompted
if "%ANTHROPIC_API_KEY%"=="" (
  set /p ANTHROPIC_API_KEY="Enter your Anthropic API key: "
)

echo Testing Anthropic and Mistral integration...
python src/test_anthropic_mistral.py

echo Done.
pause
