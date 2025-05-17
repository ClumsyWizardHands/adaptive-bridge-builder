# Interactive A2A Agent Terminal Guide

This guide explains how to use the interactive terminal to communicate between your Adaptive Bridge Builder agent and an external AI agent using the A2A Protocol.

## Getting Started

The interactive terminal is already running in your console. You'll see a prompt like this:

```
(a2a)
```

This indicates that the terminal is ready to accept commands.

## Available Commands

Here are the commands you can use in the interactive terminal:

### 1. Send Messages Between Agents

- **`bridge_to_external <message>`**: Send a message from the Bridge agent to the External agent
  
  Example:
  ```
  (a2a) bridge_to_external Hello from the Bridge agent!
  ```

- **`external_to_bridge <message>`**: Send a message from the External agent to the Bridge agent
  
  Example:
  ```
  (a2a) external_to_bridge Hello from the External agent!
  ```

### 2. Get Agent Information

- **`get_bridge_card`**: Get the agent card from the Bridge agent
  
  Example:
  ```
  (a2a) get_bridge_card
  ```

- **`get_external_card`**: Get the agent card from the External agent
  
  Example:
  ```
  (a2a) get_external_card
  ```

### 3. Conversation Management

- **`new_conversation`**: Start a new conversation with a new ID
  
  Example:
  ```
  (a2a) new_conversation
  ```

### 4. Terminal Control

- **`help`** or **`?`**: Display help information about available commands
  
  Example:
  ```
  (a2a) help
  ```

- **`exit`** or **Ctrl-D**: Exit the interactive terminal
  
  Example:
  ```
  (a2a) exit
  ```

## Integrating Your External Agent

Currently, the terminal is using a placeholder External AI Agent. To integrate your actual external agent:

1. Stop the current interactive terminal (use `exit` or Ctrl-C)
2. Open `src/interactive_agents.py` in your editor
3. Replace the `ExternalAIAgent` class with your actual agent class
4. Run the terminal again with `cd src; python interactive_agents.py`

## Example Session

Here's an example session using the interactive terminal:

1. Get the Bridge agent's card:
   ```
   (a2a) get_bridge_card
   ```

2. Send a message from the Bridge to the External agent:
   ```
   (a2a) bridge_to_external What capabilities do you have?
   ```

3. Send a message from the External agent to the Bridge:
   ```
   (a2a) external_to_bridge I need to route a message to another agent
   ```

4. Start a new conversation:
   ```
   (a2a) new_conversation
   ```

5. Exit the terminal:
   ```
   (a2a) exit
   ```

## Understanding A2A Protocol Messages

Behind the scenes, these commands are generating A2A Protocol messages in JSON-RPC 2.0 format:

```json
{
  "jsonrpc": "2.0",
  "method": "echo",
  "params": {
    "conversation_id": "unique-conversation-id",
    "content": "Your message here",
    "timestamp": "ISO timestamp"
  },
  "id": "unique-message-id"
}
```

The interactive terminal handles all the formatting for you, allowing you to focus on the communication between the agents.
