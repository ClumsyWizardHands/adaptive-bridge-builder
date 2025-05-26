# Empire Framework Integration Guide

This guide explains how the Empire Framework has been adapted to integrate with both the Agent-to-Agent (A2A) Protocol and the Agent Development Kit (ADK). These integrations enable Empire Framework components to be exchanged between agents and utilized in ADK-powered workflows.

## Overview

The Empire Framework now supports two complementary integration patterns:

1. **A2A Protocol Integration**: Enables standardized exchange of Empire components between agents through JSON-RPC messages and Server-Sent Events (SSE)
2. **ADK Integration**: Provides tools, event handling, state management, and workflow capabilities for using Empire components within the Agent Development Kit

These integrations can be used independently or together, depending on your needs.

## A2A Protocol Integration

The A2A Protocol integration enables agents to exchange Empire Framework components through standardized messages and streaming capabilities.

### Key Components

- `message_structures.py`: Defines JSON-RPC message structures for component operations
- `streaming_adapter.py`: Enables real-time streaming of component updates via SSE
- `component_task_handler.py`: Provides asynchronous processing of component operations
- `agent_card_extensions.py`: Extends agent cards to advertise Empire Framework capabilities

### Usage Examples

```python
# Create a message to request a principle component
from empire_framework.a2a.message_structures import create_get_component_message

message = create_get_component_message(component_id="principle-001")

# The message is formatted as JSON-RPC 2.0:
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

See the [Empire A2A Protocol Integration](empire_a2a_protocol_integration.md) document for detailed examples and usage patterns.

## ADK Integration

The ADK integration provides tools, event handling, and workflow capabilities for using Empire Framework components in ADK-powered environments.

### Key Components

- `empire_adk_adapter.py`: Core adapter integrating Empire components with ADK patterns
  - `EmpireADKAdapter`: Main class providing component operations and event handling
  - `EmpireEventType`: Constants for Empire-specific events
  - `EmpireADKStateManager`: Manages component and workflow state
  - `EmpireWorkflowTools`: Tools for workflow agents using Empire components
  - Tool definitions for ADK integration

### Key Features

1. **Event-Driven Operations**: All component operations emit events that can be captured and handled by ADK
2. **ADK Function Calls**: Empire operations wrapped as ADK-compatible function calls
3. **State Management**: Persistent tracking of component and workflow state
4. **Artifact Management**: Registration and tracking of Empire Framework outputs
5. **Workflow Tools**: Special tools designed for workflow agents
6. **Tool Context**: Integration with ADK tool context system

### Usage Examples

#### Basic Component Operations

```python
from empire_framework.adk.empire_adk_adapter import EmpireADKAdapter

# Create the adapter
adapter = EmpireADKAdapter()

# Create a component
principle = await adapter.create_component(
    component_data={
        "name": "Fairness as Truth",
        "description": "Equal treatment of all agents"
    },
    component_type="principle"
)

# Access a component
component = await adapter.get_component(principle["id"])

# Update a component
updated = await adapter.update_component(
    component_id=principle["id"],
    updates={"importance": "high"}
)
```

#### Event Handling

```python
async def event_handler(event_type, data):
    """Handle Empire Framework events."""
    if event_type == EmpireEventType.COMPONENT_CREATED:
        print(f"New component created: {data['component_id']}")
    elif event_type == EmpireEventType.PRINCIPLE_APPLIED:
        print(f"Principle applied with score: {data['alignment_score']}")

# Create adapter with event handler
adapter = EmpireADKAdapter(event_callback=event_handler)
```

#### Getting ADK Tools

```python
from empire_framework.adk.empire_adk_adapter import get_empire_adk_tools

# Get ADK tools from adapter
adapter = EmpireADKAdapter()
tools = get_empire_adk_tools(adapter)

# Register tools with ADK
adk.register_tools(tools)
```

#### Using Workflow Tools

```python
from empire_framework.adk.empire_adk_adapter import get_empire_workflow_tool_context

# Get workflow tool context
adapter = EmpireADKAdapter()
context = get_empire_workflow_tool_context(adapter)

# Evaluate content against all principles
workflow_tools = context["empire_framework"]["principles"]
evaluation = await workflow_tools["evaluate_with_principles"](
    content="The system should adapt to user needs."
)
print(f"Overall alignment: {evaluation['overall_alignment']}")
```

## Combining A2A and ADK Integration

The A2A Protocol and ADK integrations can be used together to enable powerful workflows:

1. **Component Exchange via A2A**: Use A2A Protocol to exchange components between agents
2. **Component Processing with ADK**: Use ADK tools to process and utilize components
3. **Stream Component Updates**: Stream real-time updates from components in workflows
4. **Task Management**: Handle long-running tasks with A2A task system and ADK state management

### Combined Example

```python
# Import from both integrations
from empire_framework.a2a.message_structures import ComponentMethods
from empire_framework.a2a.component_task_handler import ComponentTaskTypes
from empire_framework.adk.empire_adk_adapter import EmpireADKAdapter

# Create adapter
adapter = EmpireADKAdapter()

# Create a principle via ADK
principle = await adapter.create_component(
    component_data={"name": "Adaptability", "description": "..."},
    component_type="principle"
)

# Create a message to share the principle via A2A
from empire_framework.a2a.message_structures import create_custom_message
message = create_custom_message(
    method=ComponentMethods.CREATE_COMPONENT,
    params={
        "component_data": principle,
        "component_type": "principle"
    }
)

# Send message to another agent
response = await send_to_agent(message)

# Process response with ADK
if "result" in response:
    # Register the response as an artifact
    await adapter.register_artifact(
        artifact_type="shared_component_response",
        content=response
    )
```

## Implementation Considerations

When implementing agents that use both A2A Protocol and ADK with Empire Framework:

1. **Component Schemas**: Ensure components conform to schemas in both integrations
2. **Event Handling**: Set up appropriate event handlers for ADK integration
3. **State Persistence**: Manage component state appropriately across sessions
4. **Security**: Consider security implications of component exchange
5. **Error Handling**: Implement robust error handling for both integrations

## Best Practices

1. **Use ADK for Local Workflows**: When working within an agent's internal workflow, prefer the ADK integration for its state management and tool context
2. **Use A2A for Inter-Agent Exchange**: When exchanging components between different agents, use the A2A Protocol integration
3. **Combined Approach for Complex Systems**: For systems with multiple agents working on shared components, use both integrations together
4. **Event-Driven Architecture**: Design systems around the event types provided by the Empire ADK adapter
5. **Tool Context for Workflows**: Use the workflow tool context system for agent workflow scenarios
6. **State Management**: Use the state management capabilities to track component usage and history

## Reference

### A2A Protocol Methods

| Method | Description |
| ------ | ----------- |
| `empire.getComponent` | Get a specific component by ID |
| `empire.getComponents` | Get multiple components with optional filters |
| `empire.updateComponent` | Update an existing component |
| `empire.createComponent` | Create a new component |
| `empire.deleteComponent` | Delete a component |
| `empire.validateComponent` | Validate a component against a schema |
| `empire.getComponentRelations` | Get components related to a specific component |
| `empire.modifyComponentRelations` | Modify relationships between components |

### ADK Tool List

| Tool | Description |
| ---- | ----------- |
| `empire_get_component` | Get an Empire Framework component by ID |
| `empire_get_components` | Get Empire Framework components matching filters |
| `empire_create_component` | Create a new Empire Framework component |
| `empire_update_component` | Update an existing Empire Framework component |
| `empire_delete_component` | Delete an Empire Framework component |
| `empire_validate_component` | Validate an Empire Framework component against a schema |
| `empire_get_related_components` | Get components related to a specific component |
| `empire_add_relationship` | Add a relationship between Empire Framework components |
| `empire_remove_relationship` | Remove a relationship between Empire Framework components |
| `empire_create_task` | Create an asynchronous task for Empire Framework operations |
| `empire_get_task_status` | Get the status of an asynchronous Empire Framework task |
| `empire_cancel_task` | Cancel an asynchronous Empire Framework task |
| `empire_apply_principle` | Apply an Empire Framework principle to evaluate data |
| `empire_register_artifact` | Register an artifact from Empire Framework processing |
| `empire_get_artifact` | Get a registered Empire Framework artifact |
| `empire_list_artifacts` | List registered Empire Framework artifacts |

### Workflow Tool Context

| Category | Function | Description |
| -------- | -------- | ----------- |
| principles | get_principles | Get all principle components |
| principles | evaluate_with_principles | Evaluate content against all principles |
| components | create_from_template | Create a component from a template |
| state | persist_workflow_state | Persist workflow state |
| state | get_workflow_state | Get workflow state |
| state | get_component_state | Get component state |
| state | get_artifact_state | Get artifact state |

### Event Types

| Event Type | Description |
| ---------- | ----------- |
| `empire.component.created` | Component created |
| `empire.component.updated` | Component updated |
| `empire.component.deleted` | Component deleted |
| `empire.component.validated` | Component validated |
| `empire.relationship.created` | Relationship created |
| `empire.relationship.updated` | Relationship updated |
| `empire.relationship.deleted` | Relationship deleted |
| `empire.principle.applied` | Principle applied |
| `empire.principle.conflict` | Principle conflict detected |
| `empire.task.created` | Task created |
| `empire.task.completed` | Task completed |
| `empire.task.failed` | Task failed |
