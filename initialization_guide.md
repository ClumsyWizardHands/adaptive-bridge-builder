# Adaptive Bridge Builder Initialization Guide

## Overview
The Adaptive Bridge Builder agent has been successfully initialized and is ready for use. This guide provides instructions on how to run and interact with the agent system.

## Available Interaction Methods

### 1. Interactive Bridge Terminal
A simplified command-line interface for direct interaction with the Bridge agent.

```bash
cd src
python interactive_bridge.py
```

This provides an easy-to-use terminal with the following commands:
- `card` - View the Bridge agent's card
- `send <message>` - Send a message to the Bridge agent
- `route <destination> <message>` - Route a message through the Bridge
- `translate <source_protocol> <target_protocol> <message>` - Translate between protocols
- `new` - Start a new conversation
- `exit` - Exit the terminal

### 2. Dual-Agent Interactive Terminal
A more advanced terminal that simulates communication between the Bridge agent and an External agent.

```bash
cd src
python interactive_agents.py
```

This terminal provides commands for bidirectional communication:
- `bridge_to_external <message>` - Send a message from the Bridge agent to the External agent
- `external_to_bridge <message>` - Send a message from the External agent to the Bridge agent
- `get_bridge_card` - Get the agent card from the Bridge agent
- `get_external_card` - Get the agent card from the External agent
- `new_conversation` - Start a new conversation with a new ID
- `exit` - Exit the terminal

### 3. Demonstration Script
A non-interactive script that demonstrates the key capabilities of the Bridge agent.

```bash
cd src
python demo_agent_system.py
```

This script showcases:
1. Agent Card retrieval
2. Echo message processing
3. Message routing capability
4. Protocol translation capability

## Integration with External Agents

To integrate your own agent with the Adaptive Bridge Builder:

1. Open `src/interactive_agents.py` in your editor
2. Replace the `ExternalAIAgent` class with your actual agent implementation
3. Ensure your agent implements the `process_message` method with the expected signature
4. Run the interactive terminal again with `cd src && python interactive_agents.py`

## Core Principles

The Adaptive Bridge Builder operates based on three core principles:

1. **Fairness as Truth**: Equal treatment of all messages and agents regardless of source
2. **Harmony Through Presence**: Maintaining clear communication and acknowledgment of all interactions
3. **Adaptability as Strength**: Ability to evolve and respond to changing communication needs

These principles guide all agent interactions and ensure reliable, transparent communication between diverse agent systems.

## Next Steps

With the agent system initialized, you can:

1. Use the interactive terminals to test custom message exchanges
2. Integrate with external A2A Protocol-compatible agent systems
3. Extend the bridge functionality by implementing new methods in the AdaptiveBridgeBuilder class
4. Develop custom adapters for specific communication protocols
