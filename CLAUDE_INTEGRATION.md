# Claude Integration for Adaptive Bridge Builder

This extension allows you to connect the Adaptive Bridge Builder agent to Claude AI by Anthropic. The integration creates a bridge between the A2A Protocol and Claude's API, enabling natural language interaction through the structured A2A Protocol framework.

## Features

- 🌉 **Bridge Architecture**: Connects the Adaptive Bridge Builder with Claude AI
- 🔐 **Secure API Key Storage**: Safely stores your Anthropic API key locally
- 💬 **Interactive Interface**: Simple command-line interface for communication
- 🧠 **A2A Protocol**: Leverages the Agent-to-Agent protocol for structured communication

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An Anthropic API key (starts with `sk-ant-`)
- Adaptive Bridge Builder project

### Installation

1. Make sure you've already set up the main Adaptive Bridge Builder project
2. Install the required packages:
   ```
   pip install anthropic
   ```

### Setup

1. Run the setup script to securely store your API key:
   ```
   python src/setup_claude_key.py
   ```
   This will create a `.claude_config.json` file to store your key safely.

2. Start the Bridge to Claude interface:
   ```
   python src/bridge_to_claude.py
   ```

## Using the Interface

The interactive terminal provides several commands:

- `bridge_card` - View the Bridge agent's capabilities
- `claude_card` - View Claude's capabilities
- `ask <message>` - Send a message to Claude through the Bridge
- `help` - Show available commands
- `exit` - Exit the terminal

Example:
```
(bridge-claude) ask What can you tell me about the A2A Protocol?
```

## How It Works

1. When you send a message, it's routed through the Adaptive Bridge Builder
2. The Bridge formats the message according to the A2A Protocol
3. The message is passed to the Claude agent
4. Claude processes the message and returns a response
5. The response is displayed in the terminal

## Files

- `src/anthropic_agent.py` - Claude agent implementation
- `src/bridge_to_claude.py` - Interactive terminal for communication
- `src/setup_claude_key.py` - Utility to securely store API key

## Security

- Your API key is stored locally in `.claude_config.json`
- The file is only readable by your user account
- The file is added to `.gitignore` to prevent accidental commits

## Troubleshooting

- If you encounter `ImportError: No module named 'anthropic'`, run: `pip install anthropic`
- If you get an authentication error, run `setup_claude_key.py` again to ensure your API key is correct
- For other issues, check the Claude API status or your account quota

## License

This integration follows the same license as the main Adaptive Bridge Builder project.
