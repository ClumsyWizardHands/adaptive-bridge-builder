# Empire Framework A2A Protocol Integration

This document explains how the Empire Framework has been adapted to be compatible with the Agent-to-Agent (A2A) Protocol, enabling seamless exchange of Empire components between agents.

## Overview

The Empire Framework components can now be exchanged between agents using the A2A Protocol through several integration points:

1. **JSON-RPC Message Structures**: Standardized message formats for component exchange
2. **Server-Sent Events (SSE) Streaming**: Real-time streaming of component updates
3. **Asynchronous Task Handling**: Long-running component operations
4. **Agent Card Capability Extensions**: Advertising Empire Framework capabilities

## Component Exchange via JSON-RPC

Empire components can be exchanged using standard JSON-RPC 2.0 messages. The `message_structures.py` module provides helper functions to create properly formatted messages:

```python
from empire_framework.a2a.message_structures import create_get_component_message

# Create a message to request a principle component
message = create_get_component_message(component_id="principle-001")

# The message is formatted as JSON-RPC 2.0
# {
#   "jsonrpc": "2.0",
#   "method": "empire.getComponent",
#   "params": {
#     "component_id": "principle-001",
#     "include_metadata": true
#   },
#   "id": "550e8400-e29b-41d4-a716-446655440000"
# }
```

### Available Message Methods

- `empire.getComponent`: Get a specific component by ID
- `empire.getComponents`: Get multiple components with optional filters
- `empire.updateComponent`: Update an existing component
- `empire.createComponent`: Create a new component
- `empire.deleteComponent`: Delete a component
- `empire.validateComponent`: Validate a component against a schema
- `empire.getComponentRelations`: Get components related to a specific component
- `empire.modifyComponentRelations`: Modify relationships between components

## Component Streaming via Server-Sent Events

Empire components can be streamed in real-time using Server-Sent Events (SSE). The `streaming_adapter.py` module provides the necessary functionality:

```python
from empire_framework.a2a.streaming_adapter import StreamingA2AAdapter

# Create a streaming adapter
adapter = StreamingA2AAdapter()

# Stream component updates
async for event in adapter.stream_component_updates(
    component_id="principle-001",
    update_callback=get_principle_update,
    interval=1.0
):
    # Process event (send to client, etc.)
    await send_sse_event(event)
```

### Streaming Features

- **Component Updates**: Stream updates to a component over time
- **Component Batches**: Stream large component sets in manageable chunks
- **Component Changes**: Stream only the changes to components, reducing bandwidth

## Asynchronous Task Handling

Empire components can be processed asynchronously using the task system. The `component_task_handler.py` module provides task queue processing:

```python
from empire_framework.a2a.component_task_handler import ComponentTaskHandler, ComponentTaskTypes

# Create a task handler
task_handler = ComponentTaskHandler()

# Start the task processors
await task_handler.start_processing()

# Create a task to create a new component
task_id = await task_handler.create_task(
    task_type=ComponentTaskTypes.CREATION,
    component_ids=[],
    task_data={
        "component_type": "principle",
        "component_data": {
            "name": "Fairness as Truth",
            "description": "Equal treatment of all agents"
        }
    },
    priority="medium"
)

# Check task status
task_status = task_handler.get_task(task_id)
```

### Task Types

- **Component Creation**: Creating new components
- **Component Update**: Updating existing components
- **Component Deletion**: Deleting components
- **Component Validation**: Validating components against schemas
- **Component Transformation**: Transforming components into different formats
- **Relationship Update**: Managing relationships between components
- **Batch Operations**: Performing operations on multiple components
- **Export/Import**: Exporting and importing components
- **Analysis**: Analyzing components for dependencies, complexity, etc.

## Agent Card Extensions

Agents can advertise their Empire Framework capabilities using the agent card extensions. The `agent_card_extensions.py` module provides utilities to extend agent cards:

```python
from empire_framework.a2a.agent_card_extensions import extend_agent_card_with_empire_capabilities

# Load an existing agent card
agent_card = load_agent_card("agent_card.json")

# Extend with Empire Framework capabilities
extended_card = extend_agent_card_with_empire_capabilities(agent_card)

# Save the extended card
save_agent_card(extended_card, "extended_agent_card.json")
```

### Added Capabilities

- **Component Exchange**: Methods for exchanging Empire components
- **Component Streaming**: Methods for streaming components in real-time
- **Component Relationships**: Methods for managing component relationships
- **Component Tasks**: Methods for creating and managing asynchronous tasks
- **Principle Engine**: Methods for evaluating against principles
- **Emotional Intelligence**: Methods for emotional analysis and response
- **Fairness Evaluation**: Methods for evaluating fairness

## Integration Examples

### Example 1: Requesting a Component

```python
# Create a request message
request = create_get_component_message("principle-001")

# Send the request to an agent
response = await send_to_agent(request)

# Process the response
if "result" in response:
    component = response["result"]
    print(f"Received component: {component['component_data']['name']}")
else:
    print(f"Error: {response['error']['message']}")
```

### Example 2: Streaming Component Updates

```python
# Set up SSE endpoint
@app.route("/stream/component/<component_id>")
async def stream_component(component_id):
    streaming_adapter = StreamingA2AAdapter()
    
    return Response(
        stream_with_context(streaming_adapter.stream_component_updates(
            component_id=component_id,
            update_callback=lambda: get_component(component_id),
            interval=1.0
        )),
        mimetype="text/event-stream"
    )
```

### Example 3: Creating an Asynchronous Task

```python
# Create a component creation task
task_id = await task_handler.create_task(
    task_type=ComponentTaskTypes.CREATION,
    component_ids=[],
    task_data={
        "component_type": "principle",
        "component_data": {
            "name": "Adaptability as Strength",
            "description": "Ability to evolve and respond to changing needs"
        }
    }
)

# Monitor the task status
while True:
    status = task_handler.get_task(task_id)
    
    if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        break
        
    print(f"Task progress: {status['progress'] * 100:.0f}%")
    await asyncio.sleep(0.5)
    
if status["status"] == TaskStatus.COMPLETED:
    print(f"Task completed successfully: {status['result']}")
else:
    print(f"Task failed: {status['error']['message']}")
```

## Implementation Considerations

When implementing agents that work with Empire Framework components:

1. **Message Validation**: Validate received messages against the A2A schema
2. **Error Handling**: Handle error responses appropriately
3. **Streaming Support**: Implement SSE handling for streaming operations
4. **Task Monitoring**: Monitor long-running tasks and handle timeouts
5. **Schema Validation**: Validate components before accepting them
6. **Capability Negotiation**: Check agent capabilities before attempting operations

## Best Practices

1. **Use Streaming for Large Data Sets**: When transferring multiple components or large components, use the streaming capabilities to avoid timeouts
2. **Prefer Asynchronous Tasks for Heavy Operations**: Use tasks for operations that may take time, such as component transformations or analysis
3. **Check Component Compatibility**: Validate components against schemas before exchange
4. **Include Appropriate Metadata**: Provide source, version, and other metadata when exchanging components
5. **Keep Components Slim**: Only include necessary fields to reduce payload size
6. **Cache Component Information**: Cache frequently accessed components to reduce traffic
