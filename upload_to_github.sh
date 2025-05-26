#!/bin/bash
# =========================================================
# GitHub Upload Script for Linux/macOS
#
# This script commits and pushes the multi-model agent changes
# to GitHub repository
# =========================================================

echo "GitHub Upload Script"
echo "==================="
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found. Please install Git and try again."
    exit 1
fi

echo "Checking repository status..."
git status

echo
echo "Adding new files..."
git add run_claude_only.bat
git add run_claude_only.sh
git add setup.bat
git add setup.sh
git add src/run_multi_model_agent.py
git add src/download_mistral_model.py
git add requirements.txt
git add MULTI_MODEL_AGENT_GUIDE.md
git add MULTI_MODEL_README.md

echo
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then 
    commit_msg="Add multi-model agent with Claude and Mistral support"
fi

echo
echo "Committing changes with message: \"$commit_msg\""
git commit -m "$commit_msg"

echo
echo "Pushing changes to GitHub..."
git push

echo
echo "Upload completed! Check GitHub repository at:"
echo "https://github.com/ClumsyWizardHands/adaptive-bridge-builder"
echo

read -p "Press Enter to continue..."
