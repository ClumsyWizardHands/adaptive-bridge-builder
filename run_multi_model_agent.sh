#!/bin/bash
# =========================================================
# Multi-Model Agent Runner
#
# This script runs the multi-model agent with:
# - Local Mistral LLM (for privacy-sensitive operations)
# - Anthropic Claude (for complex reasoning tasks)
# =========================================================

echo "Multi-Model Agent (Mistral + Claude)"
echo "===================================="
echo

# Uncomment and set your Claude API key here if you don't want to be prompted each time
# export ANTHROPIC_API_KEY=your_api_key_here

# Uncomment and set your local model path if it's not in the standard location
# export LOCAL_MODEL_PATH=/path/to/your/model.gguf

# Run the multi-model agent
python src/run_multi_model_agent.py

echo
echo "Agent session ended."
read -p "Press Enter to continue..."
