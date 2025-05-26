# Multi-Model Agent (Claude + Mistral)

A flexible agent system that intelligently routes tasks between local Mistral LLM and cloud-based Anthropic Claude API.

## Features

- **Dual-model architecture** combining cloud and local LLMs
- **Intelligent task routing** based on privacy needs, complexity, and other factors
- **Privacy-first design** with option to use local models for sensitive data
- **Easy setup** with simple scripts for both Windows and Linux/macOS
- **Claude-only mode** that works without any local model setup

## Quick Start

### Option 1: Claude-Only Mode (Fastest)

If you want to get started immediately using only Claude:

1. Run the Claude-only script for your platform:
   - Windows: `run_claude_only.bat`
   - Linux/macOS: `chmod +x run_claude_only.sh && ./run_claude_only.sh`

2. Enter your Claude API key when prompted.

### Option 2: Full Setup (Both Models)

To use both Claude and a local Mistral model:

1. Run the setup script for your platform:
   - Windows: `setup.bat`
   - Linux/macOS: `chmod +x setup.sh && ./setup.sh`

2. Run the agent:
   - Windows: `run_multi_model_agent.bat`
   - Linux/macOS: `./run_multi_model_agent.sh`

3. Follow the prompts to enter your Claude API key.

## Usage Commands

Once the agent is running:

- `/privacy` - Use maximum privacy (prioritizes local model)
- `/balanced` - Use balanced privacy (prefers local but may use cloud)
- `/cloud` - Use standard privacy (selects best model for the task)
- `/simple` - Set task complexity to simple
- `/moderate` - Set task complexity to moderate
- `/complex` - Set task complexity to complex
- `/help` - Show available commands
- `/exit` or `/quit` - Exit the program

## Requirements

- Python 3.9+
- For Claude: An Anthropic API key
- For Mistral: A GGUF format model and the llama-cpp-python package

## Detailed Documentation

For complete installation and usage details, see [MULTI_MODEL_AGENT_GUIDE.md](MULTI_MODEL_AGENT_GUIDE.md).

## Architecture

The agent uses an adapter-based architecture that makes it easy to:

1. Add new LLM providers
2. Create sophisticated routing logic
3. Implement specialized models for different tasks
4. Extend with embeddings, RAG, and other advanced features

The core design principle is modularity - each component has a clear, focused responsibility, making the system easy to understand and extend.
