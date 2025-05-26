#!/bin/bash
# =========================================================
# Multi-Model Agent Setup
#
# This script installs required dependencies and prepares
# the environment for running the multi-model agent.
# =========================================================

echo "Multi-Model Agent Setup"
echo "======================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.9+ and try again."
    exit 1
fi

echo "Installing required Python packages..."
pip3 install -r requirements.txt

# Offer to download a Mistral model if needed
if ! ls models/*.gguf 1> /dev/null 2>&1; then
    echo
    echo "No Mistral model found in the models directory."
    read -p "Would you like to download a Mistral model now? (y/n): " choice
    
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo
        echo "Running model downloader..."
        python3 src/download_mistral_model.py
    fi
else
    echo
    echo "Mistral model found in models directory."
fi

echo
echo "Setup completed!"
echo
echo "To run the multi-model agent:"
echo "1. Make the run script executable: chmod +x run_multi_model_agent.sh"
echo "2. Run: ./run_multi_model_agent.sh"
echo "3. Enter your Claude API key when prompted (or set ANTHROPIC_API_KEY environment variable)"
echo
echo "For more information, see MULTI_MODEL_AGENT_GUIDE.md"
echo

read -p "Press Enter to continue..."
