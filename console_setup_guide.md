# Running Adaptive Bridge Builder on Your Local Machine

The Adaptive Bridge Builder agent can be easily run on your local console or terminal. This guide provides step-by-step instructions for setting up and running the agent on your machine.

## Prerequisites

1. **Python 3.9 or higher** installed on your system
2. **Git** for cloning the repository (optional)

## Setup Instructions

### 1. Clone or Download the Repository

```bash
# Option 1: Clone the repository
git clone https://github.com/YourUsername/adaptive-bridge-builder.git
cd adaptive-bridge-builder

# Option 2: Download and extract the ZIP file from GitHub
# Then navigate to the extracted directory
cd path/to/extracted/adaptive-bridge-builder
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Running the Agent

The repository provides multiple ways to interact with the agent:

### Option 1: Interactive Bridge Terminal

This provides a simple, command-based interface for direct interaction with the Bridge agent:

```bash
# Navigate to the src directory
cd src

# Run the interactive bridge terminal
python interactive_bridge.py
```

You'll see a terminal interface with the following commands:
- `card` - View the Bridge agent's card
- `send <message>` - Send a message to the Bridge agent
- `route <destination> <message>` - Route a message through the Bridge
- `translate <source_protocol> <target_protocol> <message>` - Translate between protocols
- `new` - Start a new conversation
- `exit` - Exit the terminal

### Option 2: Dual-Agent Interactive Terminal

This simulates communication between the Bridge agent and an External agent:

```bash
# Navigate to the src directory
cd src

# Run the dual-agent interactive terminal
python interactive_agents.py
```

Available commands:
- `bridge_to_external <message>` - Send a message from Bridge to External agent
- `external_to_bridge <message>` - Send a message from External to Bridge agent
- `get_bridge_card` - Get the agent card from the Bridge agent
- `get_external_card` - Get the agent card from the External agent
- `new_conversation` - Start a new conversation
- `exit` - Exit the terminal

### Option 3: Run the Demonstration Script

This runs a non-interactive demonstration of all main agent capabilities:

```bash
# Navigate to the src directory
cd src

# Run the demonstration script
python demo_agent_system.py
```

## Console Example

Here's an example console session using the interactive bridge terminal:

```
$ cd src
$ python interactive_bridge.py

Initializing Adaptive Bridge Builder...
Bridge Agent initialized with ID: bridge-agent-001
Active conversation ID: convo-68f9d5b8-dd3d-4e7c-934a-5df528976b99

╔══════════════════════════════════════════════════════╗
║ Adaptive Bridge Builder Interactive Terminal         ║
║                                                      ║
║ Type:                                                ║
║   'card'    - View the Bridge agent's card           ║
║   'send'    - Send a message to the Bridge agent     ║
║   'route'   - Route a message through the Bridge     ║
║   'translate' - Translate between protocols          ║
║   'help'    - Show available commands                ║
║   'exit'    - Exit the terminal                      ║
╚══════════════════════════════════════════════════════╝

(bridge) card

Retrieving Bridge Agent card...

Bridge Agent Card:
{
  "agent_id": "bridge-agent-001",
  "name": "Adaptive Bridge Builder",
  ...
}

(bridge) send Hello, bridge agent!

Sending message to Bridge: Hello, bridge agent!

Response from Bridge Agent:
{
  "jsonrpc": "2.0",
  "id": "msg-1-c5f3e26c-9b7d-4a29-b3e8-f41d0e8420f1",
  "result": {
    "conversation_id": "convo-68f9d5b8-dd3d-4e7c-934a-5df528976b99",
    "content": "Hello, bridge agent!",
    "timestamp": "2025-05-17T12:45:23.123456"
  }
}

(bridge) exit

Exiting Adaptive Bridge Builder terminal. Goodbye!
```

## Integration with Your Own Agents

To use the Adaptive Bridge Builder with your own agents:

1. Open `src/interactive_agents.py` in your editor
2. Replace the `ExternalAIAgent` class with your actual agent implementation
3. Ensure your agent implements the `process_message` method with the expected signature
4. Run the interactive terminal again

## Troubleshooting

- **File not found errors**: Make sure you're running the scripts from the correct directory
- **ModuleNotFoundError**: Verify all dependencies are installed via `pip install -r requirements.txt`
- **Permission errors**: You may need to make the Python scripts executable on Unix-based systems: `chmod +x src/*.py`

For additional help or questions, please refer to the project documentation or open an issue on GitHub.
