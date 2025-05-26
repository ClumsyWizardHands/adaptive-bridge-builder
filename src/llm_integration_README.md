# LLM Integration System

This set of modules implements a comprehensive integration system for Large Language Models (LLMs) in the Adaptive Bridge Builder. It allows seamless interaction with various LLM providers through a unified interface, secure API key management, and integration with the existing agent registry system.

## Core Components

### 1. LLM Key Manager (`llm_key_manager.py`)

A secure and flexible system for managing API keys for different LLM providers:

- Supports multiple storage options (environment variables, file-based)
- Validates keys against known formats
- Secures stored keys with proper permissions
- Provides a user-friendly interactive setup

### 2. LLM Adapter Interface (`llm_adapter_interface.py`)

The foundation of our LLM integration, defining a consistent API for interacting with any LLM:

- Abstract base class (`BaseLLMAdapter`) that all LLM adapters must implement
- Common methods for sending requests, processing responses, streaming, etc.
- Standardized error handling with specific exception types
- Registry system for managing and retrieving adapters

### 3. Concrete LLM Adapters

Implementation of adapters for specific LLM providers:

- **OpenAI GPT Adapter** (`openai_llm_adapter.py`) - Supports GPT-3.5, GPT-4, etc.
- **Anthropic Claude Adapter** (`anthropic_llm_adapter.py`) - Supports Claude models

Each adapter handles provider-specific details while exposing the standardized interface.

### 4. Agent Registry Integration (`agent_registry_llm_integration.py`)

Extension of the existing agent registry to work with LLM adapters:

- `LLMAgentRegistry` class that extends `AgentRegistry`
- Methods for registering LLM adapters as agents
- Mapping of LLM capabilities to agent capabilities
- Helper methods for retrieving and managing LLM adapters

### 5. Universal Agent Connector Integration (`universal_agent_connector_llm.py`)

Bridge between the LLM adapters and the A2A protocol:

- `LLMConnector` class that extends `UniversalAgentConnector`
- Translates between A2A messages and LLM API calls
- Manages conversation contexts for multi-turn interactions
- Exposes LLM capabilities in the A2A protocol format

### 6. Usage Example (`llm_interaction_example.py`)

Demonstration of how to use the LLM adapters:

- Setting up the adapter registry
- Sending requests to different providers
- Processing responses
- Comparing results from different models
- Streaming responses

## Setup and Usage

### Installing Dependencies

The integration requires the following packages:

```bash
pip install openai anthropic tiktoken
```

### API Key Setup

Before using the integration, you need to set up API keys for the LLM providers:

1. Run the interactive setup:
   ```python
   from llm_key_manager import LLMKeyManager
   key_manager = LLMKeyManager()
   key_manager.setup_key_interactive("openai")  # For OpenAI
   key_manager.setup_key_interactive("anthropic")  # For Anthropic
   ```

2. Or configure keys using environment variables:
   ```
   LLM_API_KEY_OPENAI=sk-...
   LLM_API_KEY_ANTHROPIC=sk-ant-...
   ```

### Basic Usage

```python
import asyncio
from llm_key_manager import LLMKeyManager
from llm_adapter_interface import LLMAdapterRegistry
from openai_llm_adapter import OpenAIGPTAdapter
from anthropic_llm_adapter import AnthropicClaudeAdapter

async def main():
    # Set up the registry
    key_manager = LLMKeyManager()
    registry = LLMAdapterRegistry()
    
    # Register adapters
    openai_adapter = OpenAIGPTAdapter(key_manager=key_manager)
    registry.register_adapter(openai_adapter)
    
    # Get an adapter
    adapter = registry.get_adapter("openai")
    
    # Send a request
    response = await adapter.send_request(
        prompt="What is a language model?",
        temperature=0.7
    )
    
    # Process the response
    result = adapter.process_response(response)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent Registry Integration

```python
from agent_registry_llm_integration import LLMAgentRegistry, setup_llm_registry

# Set up the registry with available LLM adapters
registry = setup_llm_registry()

# Find agents with text generation capability
agents = registry.find_agents_with_capability(
    capability_name="text_generation",
    task_type=TaskType.GENERATION
)

# Get an LLM adapter by provider name
adapter = registry.get_adapter_by_provider("openai")
```

### Universal Agent Connector

```python
from universal_agent_connector_llm import LLMConnector

# Create an LLM connector
connector = LLMConnector(
    llm_registry=registry,
    adapter_provider="openai",
    system_message="You are a helpful AI assistant."
)

# Connect to the LLM
await connector.connect()

# Create and send a message
message = AgentMessage(
    id=str(uuid.uuid4()),
    type=MessageType.TASK_REQUEST,
    content={"text": "Explain what an LLM adapter is."},
    sender_id="user",
    recipient_id="openai",
    timestamp=datetime.now()
)
await connector.send_message(message)

# Receive the response
response = await connector.receive_message()
print(response.content.get("text", ""))
```

## Extending the System

### Adding New LLM Providers

To add support for a new LLM provider:

1. Create a new adapter class that extends `BaseLLMAdapter`
2. Implement all required methods
3. Register the adapter with the LLM registry

Example:
```python
class NewProviderAdapter(BaseLLMAdapter):
    # Implement required methods
    ...

# Register the adapter
new_adapter = NewProviderAdapter(key_manager=key_manager)
registry.register_adapter(new_adapter)
```

### Adding New Capabilities

To add new capabilities to existing adapters:

1. Update the adapter's `provider_models` property
2. If needed, implement new methods in the adapter
3. Update the agent registry integration to map new capabilities

## Best Practices

1. **Security**: Always use the LLMKeyManager for managing API keys
2. **Error Handling**: Catch and handle specific LLMAdapterError exceptions
3. **Token Management**: Monitor token usage to control costs
4. **Streaming**: Use streaming for long responses to improve user experience
5. **System Messages**: Customize system messages for different use cases

## Future Enhancements

Potential areas for improvement:

1. Support for more LLM providers
2. Better token counting and cost estimation
3. Enhanced error recovery and retry mechanisms
4. Caching mechanisms for frequent requests
5. Advanced conversation management

## Conclusion

This LLM Integration System provides a flexible and secure foundation for interacting with different LLM providers through a unified interface. It seamlessly integrates with the existing Adaptive Bridge Builder components, allowing LLMs to be used as agents in the ecosystem.
