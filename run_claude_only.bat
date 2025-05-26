@echo off
REM =========================================================
REM Claude-Only Agent Runner
REM
REM This script runs the multi-model agent using only Claude,
REM without requiring any local model dependencies.
REM =========================================================

echo Claude-Only Agent
echo ================
echo.

REM Uncomment and set your Claude API key here if you don't want to be prompted each time
REM set ANTHROPIC_API_KEY=your_api_key_here

REM Run the agent but skip Mistral setup
set SKIP_LOCAL_MODELS=true
python src/run_multi_model_agent.py

echo.
echo Agent session ended.
pause
