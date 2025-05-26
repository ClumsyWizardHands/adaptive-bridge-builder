#!/bin/bash
# =========================================================
# Claude-Only Agent Runner
#
# This script runs the multi-model agent using only Claude,
# without requiring any local model dependencies.
# =========================================================

echo "Claude-Only Agent"
echo "================"
echo

# Uncomment and set your Claude API key here if you don't want to be prompted each time
# export ANTHROPIC_API_KEY=your_api_key_here

# Run the agent but skip Mistral setup
export SKIP_LOCAL_MODELS=true
python3 src/run_multi_model_agent.py

echo
echo "Agent session ended."
read -p "Press Enter to continue..."
