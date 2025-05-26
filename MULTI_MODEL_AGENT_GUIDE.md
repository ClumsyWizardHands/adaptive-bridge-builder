# Multi-Model Agent Guide

This guide explains how to run and use the Multi-Model Agent that combines:
- **Local Mistral LLM** for privacy-sensitive operations
- **Anthropic Claude API** for complex reasoning tasks

The agent intelligently routes tasks to the appropriate model based on privacy requirements, task complexity, and other factors.

## Prerequisites

1. **Python 3.9+** with the following packages:
   - aiohttp
   - asyncio
   - argparse

2. **For Claude integration**:
   - An Anthropic API key
   - Internet connection

3. **For local Mistral integration**:
   - A GGUF format Mistral model
   - One of these backends:
     - llama-cpp-python (`pip install llama-cpp-python`)
     - ctransformers (`pip install ctransformers`)

## Quick Start (Recommended)

### Option 1: Claude-Only Mode (Fastest)

If you want to get started immediately using only Claude without needing to set up a local model:

1. Run the Claude-only script for your platform:
   - Windows: `run_claude_only.bat`
   - Linux/macOS: `chmod +x run_claude_only.sh && ./run_claude_only.sh`

2. Enter your Claude API key when prompted (or set the ANTHROPIC_API_KEY environment variable)

This option skips all local model setup and runs with Claude only.

### Option 2: Full Setup (Both Models)

If you want to use both Claude and a local Mistral model:

1. Run the setup script for your platform:
   - Windows: `setup.bat`
   - Linux/macOS: `chmod +x setup.sh && ./setup.sh`

2. Run the agent:
   - Windows: `run_multi_model_agent.bat`
   - Linux/macOS: `./run_multi_model_agent.sh`

3. Follow the prompts to enter your Claude API key (if not already set).

## Manual Setting Up

### 1. Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

This is crucial as it installs the necessary backends for the local Mistral model. Without this step, you'll see an error like:

```
Failed to initialize local model: No local model backend available. Please install either llama-cpp-python or ctransformers.
```

### 2. Claude API Key

You have three options to provide your Claude API key:

- Set the `ANTHROPIC_API_KEY` environment variable
- Edit the batch/shell script to include your key
- Enter your key when prompted during startup

### 3. Local Mistral Model

Place your Mistral model file (`.gguf`) in the `models/` directory, or specify a custom path using the `LOCAL_MODEL_PATH` environment variable or command-line argument.

If you don't have a model yet, you can download one:

```bash
python src/download_mistral_model.py
```

Common model files:
- mistral-7b-instruct-v0.2.Q4_K_M.gguf (4.1 GB, recommended)
- mistral-7b-instruct.gguf
- mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf (26 GB, more powerful)

## Running the Agent

### On Windows

```
run_multi_model_agent.bat
```

Or with explicit arguments:

```
python src/run_multi_model_agent.py --claude-key YOUR_KEY --model-path path/to/model.gguf
```

### On Linux/macOS

Make the script executable first:

```
chmod +x run_multi_model_agent.sh
./run_multi_model_agent.sh
```

Or directly:

```
python src/run_multi_model_agent.py --claude-key YOUR_KEY --model-path /path/to/model.gguf
```

### Using Claude Only

If you get errors with the local Mistral model or simply want to use Claude only, you can still use the agent. It will fall back to Claude for all requests. You'll see a warning message:

```
⚠️ No local model found. Only Claude will be available.
```

The agent will still work perfectly fine with Claude, though privacy-sensitive operations will also use the cloud API.

## Using the Agent

Once running, you can interact with the agent through the command line interface.

### Commands

- `/privacy` - Use maximum privacy (forces local model use)
- `/balanced` - Use balanced privacy (prefers local but may use cloud)
- `/cloud` - Use standard privacy (selects best model)
- `/simple` - Set task complexity to simple
- `/moderate` - Set task complexity to moderate
- `/complex` - Set task complexity to complex
- `/help` - Show available commands
- `/exit` or `/quit` - Exit the program

### Example Usage

1. Start with a simple question that uses Claude by default:
   ```
   You: What is the capital of France?
   ```

2. Ask a privacy-sensitive question using the local model:
   ```
   You: /privacy
   You: Analyze this proprietary code: [code snippet]
   ```

3. Ask a complex reasoning question:
   ```
   You: /complex
   You: Explain the ethical implications of autonomous vehicles.
   ```

## How Model Selection Works

The agent uses these criteria to select between Claude and Mistral:

1. **Privacy requirements**: 
   - MAXIMUM: Uses local Mistral only
   - HIGH: Prefers local Mistral but will fall back to Claude
   - STANDARD: Selects best model for the task

2. **Task complexity**:
   - SIMPLE: Favors faster, cheaper models (often Mistral)
   - MODERATE: Balanced selection
   - COMPLEX: Favors powerful models (often Claude)

3. **Cost preference**:
   - Default is BALANCED (good performance without excessive cost)

4. **Latency requirements**:
   - Default is MEDIUM (balanced response time)

## Troubleshooting

### "No local model backend available" Error

If you see this error:
```
Failed to initialize local model: No local model backend available. Please install either llama-cpp-python or ctransformers.
```

Run the setup script first:
- Windows: `setup.bat`
- Linux/macOS: `chmod +x setup.sh && ./setup.sh`

Or manually install the required packages:
```
pip install -r requirements.txt
```

### No Claude Responses

- Verify your Anthropic API key is correct
- Check your internet connection
- Ensure you're not in MAXIMUM privacy mode

### No Mistral Responses

- Verify your model file exists at the specified path
- Check that you have llama-cpp-python or ctransformers installed
- Look for error messages about model loading failures

### Poor Performance

- For local models, a GPU can significantly improve performance
- For complex tasks, try using `/complex` to prioritize Claude
- For faster responses, try using `/simple` to prioritize speed

### Models Directory Missing

If the models directory doesn't exist:
```
mkdir models
```

Then either download a model or copy your existing model into the directory.

## Advanced Configuration

Edit the `src/run_multi_model_agent.py` file to:

- Change default Claude model (line ~190)
- Adjust model selection criteria
- Modify response formatting
- Change token limits and model parameters

## Extending the Agent

This agent demonstrates the integration of local and cloud LLMs, but can be extended to:

1. Add more LLM providers (OpenAI, Google, etc.)
2. Implement more complex routing logic
3. Add specialized models for domain-specific tasks
4. Incorporate embeddings, RAG, and other advanced features

The modular adapter architecture makes it easy to add new models without changing the core agent code.
