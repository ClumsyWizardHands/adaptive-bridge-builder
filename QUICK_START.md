# Adaptive Bridge Builder - Quick Start Guide

This guide provides simple instructions to get the Adaptive Bridge Builder agent running on your local machine.

## Windows Users

1. **Double-click** on `run_agent.bat`

This will start the setup process, which will:
- Check your Python version
- Set up a virtual environment
- Install required dependencies
- Verify agent files
- Present you with options to run the agent

## macOS/Linux Users

1. Make the shell script executable (one-time setup):
   ```bash
   chmod +x run_agent.sh
   ```

2. Run the script:
   ```bash
   ./run_agent.sh
   ```

## Manual Setup (All Platforms)

If you prefer to run the setup script directly:

```bash
# Windows
python setup_and_run.py

# macOS/Linux
python3 setup_and_run.py
```

## What to Expect

The setup script will guide you through the process and offer three ways to interact with the agent:

1. **Interactive Bridge Terminal** (Recommended for beginners)
   - Direct interaction with the Bridge agent
   - Simple command set for basic operations

2. **Dual-Agent Interactive Terminal**
   - Simulates communication between the Bridge agent and an External agent
   - More complex but shows the full messaging capabilities

3. **Demonstration Script**
   - Non-interactive showcase of the agent's capabilities
   - Good for understanding how the agent works

## Troubleshooting

- **Python not found**: Make sure Python 3.9 or higher is installed and in your PATH
- **Missing files**: Verify that all files in the `src` directory are present
- **Dependency errors**: Check your internet connection and try running the setup again

## No API Keys Required

The Adaptive Bridge Builder operates as a standalone system and does not require any API keys or external service connections to function.
