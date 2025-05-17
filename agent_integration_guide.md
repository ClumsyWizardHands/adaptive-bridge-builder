# Agent Integration Guide

This guide explains how to integrate your external AI agent with the Adaptive Bridge Builder agent using the A2A Protocol.

## Getting Started

### Prerequisites

- Python 3.9+
- Adaptive Bridge Builder project installed
- Your external AI agent code

### Running the Integration Script

1. Navigate to the project directory
2. Run the integration script:

```bash
python src/agent_integration.py
```

## Understanding the Integration

The `agent_integration.py` script demonstrates:

1. Initializing the Adaptive Bridge Builder agent
2. Setting up a placeholder for your external AI agent
3. Establishing basic A2A Protocol communication between the agents

## Customizing for Your Agent

To integrate your actual external AI agent:

1. Replace the `ExternalAIAgent` class with an import of your actual agent class
2. Update the initialization parameters in the `main()` function to match your agent's requirements
3. Ensure your agent can process or is adapted to handle A2A Protocol messages

## A2A Protocol Message Format

All communications follow this JSON-RPC 2.0 structure:

```json
{
  "jsonrpc": "2.0",
  "method": "methodName",
  "params": {
    "conversation_id": "unique-conversation-id",
    "content": "Message content or structured data",
    "timestamp": "ISO timestamp"
  },
  "id": "unique-message-id"
}
```

## Supported Methods

The Adaptive Bridge Builder supports these methods:

1. `getAgentCard` - Get information about the agent's capabilities and principles
2. `echo` - Simple echo response for testing
3. `route` - Route messages to other agents
4. `translateProtocol` - Translate between different protocols

## Implementing Ongoing Communication

For continuous communication between agents:

1. Maintain the conversation_id across related messages
2. Use appropriate methods based on the communication needs
3. Handle both success and error responses

## Error Handling

Always check for error responses:

```python
if "error" in response:
    # Handle error response
    print(f"Error: {response['error']['message']}")
else:
    # Process successful response
    result = response.get("result", {})
```

## Advanced Integration Options

For more advanced integration requirements:

1. **Universal Agent Connector**: For comprehensive third-party agent integration
2. **API Gateway System**: For integration via REST APIs
3. **A2A Task Handler**: For task-based interactions following the A2A protocol

## Example: Multi-turn Conversation

To implement a multi-turn conversation:

```python
# Start a conversation
conversation_id = f"convo-{str(uuid.uuid4())}"

# First message
response1 = send_a2a_message(
    sender=agent1,
    receiver=agent2,
    content="Initial message",
    method="echo",
    conversation_id=conversation_id
)

# Second message in same conversation
response2 = send_a2a_message(
    sender=agent2,
    receiver=agent1,
    content="Follow-up message",
    method="echo",
    conversation_id=conversation_id  # Same conversation_id
)
```

## Troubleshooting

Common issues:

1. **Import errors**: Ensure the path to modules is correct
2. **Message format errors**: Verify all messages follow JSON-RPC 2.0 structure
3. **Method not found**: Check that you're using supported methods
4. **Parameter errors**: Ensure required parameters are included

For more assistance, refer to the A2A Protocol documentation and the Adaptive Bridge Builder implementation files.
