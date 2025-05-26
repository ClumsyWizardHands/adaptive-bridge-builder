@echo off
REM =========================================================
REM Multi-Model Agent Runner
REM
REM This script runs the multi-model agent with:
REM - Local Mistral LLM (for privacy-sensitive operations)
REM - Anthropic Claude (for complex reasoning tasks)
REM =========================================================

echo Multi-Model Agent (Mistral + Claude)
echo ====================================
echo.

REM Uncomment and set your Claude API key here if you don't want to be prompted each time
REM set ANTHROPIC_API_KEY=your_api_key_here

REM Uncomment and set your local model path if it's not in the standard location
REM set LOCAL_MODEL_PATH=path\to\your\model.gguf

REM Run the multi-model agent
python src/run_multi_model_agent.py

echo.
echo Agent session ended.
pause
