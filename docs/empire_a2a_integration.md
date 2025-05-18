# Empire Framework A2A Integration

This document describes how Empire Framework components can be exposed and exchanged using the Agent-to-Agent (A2A) Protocol.

## Overview

The Empire Framework A2A integration enables the seamless exchange of Empire components (Principles, Means, Ends, etc.) between agents that implement the A2A Protocol. This integration provides a standardized way for agents to discover, retrieve, and interact with Empire components, facilitating interoperability in multi-agent systems.

## Architecture

The integration consists of three main parts:

1. **A2A Adapter**: Converts between Empire components and A2A message formats
2. **A2A API Handlers**: Exposes Empire components through standardized A2A endpoints
3. **Integration Tests**: Ensures the correct functioning of the A2A integration

```
┌─────────────────────┐     ┌──────────────────┐     ┌────────────────┐
│                     │     │                  │     │                │
│  Empire Components  │◄───►│    A2A Adapter   │◄───►│   A2A Messages │
│                     │     │                  │     │                │
└─────────────────────┘     └──────────────────┘     └────────────────┘
                                     ▲
                                     │
                                     ▼
                             ┌──────────────────┐
                             │                  │
                             │   A2A Handlers   │
                             │                  │
                             └──────────────────┘
                                     ▲
                                     │
                                     ▼
                             ┌──────────────────┐
                             │                  │
                             │ External Agents  │
                             │                  │
                             └──────────────────┘
```

## A2A Adapter

The A2A Adapter (`empire_framework.a2a.a2a_adapter.A2AAdapter`) provides bidirectional conversion between Empire components and A2A message formats. It handles:

- Converting Empire components to A2A messages
- Converting A2A messages back to Empire components
- Breaking down components into A2A parts for granular access
- Batch operations for multiple components

### Key Features

- **Validation**: Optional validation during conversion to ensure component integrity
- **Metadata Handling**: Preserves component metadata during conversion
- **Type-Specific Processing**: Specialized handling for different component types (Principles, Means, Ends, etc.)
- **Relationship Preservation**: Maintains relationship information between components

## A2A API Endpoints

The A2A integration exposes the following endpoints through the `EmpireA2AHandlers` class:

| Method | Description | Parameters |
|--------|-------------|------------|
| `empire.getComponents` | Returns components matching specified filters | `filters`: Optional component filters |
| `empire.getComponentById` | Returns a specific component by ID | `component_id`: ID of the component to retrieve |
| `empire.getRelatedComponents` | Returns components related to a specific component | `component_id`: ID of the root component<br>`relationship_types`: Optional types to filter by |
| `empire.getComponentParts` | Returns specific parts of a component | `component_id`: ID of the component<br>`parts`: Optional specific parts to retrieve |

### Message Format

All A2A messages follow this general structure:

```json
{
  "a2a_version": "1.0",
  "message_type": "empire.component",
  "content": {
    "component_id": "principle-001",
    "component_type": "principle",
    "component_data": {
      "name": "Fairness as Truth",
      "description": "Equal treatment of all agents",
      ...
    }
  },
  "metadata": {
    "source": "empire_framework",
    "component_version": "1.0"
  }
}
```

## Usage Examples

### Getting Components by Filter

**Request:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.getComponents",
  "content": {
    "filters": {
      "type": "principle",
      "tags": ["fairness"]
    }
  }
}
```

**Response:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.component_batch",
  "content": {
    "component_count": 1,
    "components": [
      {
        "component_id": "principle-001",
        "component_type": "principle",
        "component_data": {
          "name": "Fairness as Truth",
          "description": "Equal treatment of all agents",
          "importance": "high",
          "tags": ["fairness", "core_principle"]
        }
      }
    ]
  },
  "metadata": {
    "source": "empire_framework",
    "filter_applied": {
      "type": "principle",
      "tags": ["fairness"]
    }
  }
}
```

### Getting a Component by ID

**Request:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.getComponentById",
  "content": {
    "component_id": "means-001"
  }
}
```

**Response:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.component",
  "content": {
    "component_id": "means-001",
    "component_type": "means",
    "component_data": {
      "name": "Adaptive Communication",
      "capabilities": [
        "Protocol translation",
        "Message routing",
        "Format adaptation"
      ],
      "limitations": [
        "Requires clear message structure"
      ]
    }
  },
  "metadata": {
    "source": "empire_framework",
    "component_version": "1.0"
  }
}
```

### Getting Related Components

**Request:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.getRelatedComponents",
  "content": {
    "component_id": "means-001",
    "relationship_types": ["implements"]
  }
}
```

**Response:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.component_batch",
  "content": {
    "component_count": 1,
    "components": [
      {
        "component_id": "principle-001",
        "component_type": "principle",
        "component_data": {
          "name": "Fairness as Truth",
          "description": "Equal treatment of all agents",
          "importance": "high"
        }
      }
    ]
  },
  "metadata": {
    "source": "empire_framework",
    "root_component_id": "means-001",
    "relationship_types": ["implements"],
    "relationships": [
      {
        "component_id": "principle-001",
        "relationship_type": "implements",
        "relationship_strength": 0.8
      }
    ]
  }
}
```

### Getting Component Parts

**Request:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.getComponentParts",
  "content": {
    "component_id": "principle-001",
    "parts": ["identity", "principle"]
  }
}
```

**Response:**
```json
{
  "a2a_version": "1.0",
  "message_type": "empire.component_parts",
  "content": {
    "component_id": "principle-001",
    "component_type": "principle",
    "parts": {
      "identity": {
        "id": "principle-001",
        "type": "principle",
        "version": "1.0",
        "name": "Fairness as Truth"
      },
      "principle": {
        "name": "Fairness as Truth",
        "description": "Equal treatment of all agents",
        "importance": "high"
      }
    }
  },
  "metadata": {
    "source": "empire_framework",
    "component_version": "1.0",
    "requested_parts": ["identity", "principle"]
  }
}
```

## Error Handling

The A2A integration uses standardized error responses:

```json
{
  "a2a_version": "1.0",
  "message_type": "empire.error",
  "content": {
    "error": {
      "code": -32001,
      "message": "Component not found",
      "data": {
        "component_id": "nonexistent-id"
      }
    }
  },
  "metadata": {
    "source": "empire_framework"
  }
}
```

Common error codes:
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid parameters
- `-32001`: Component not found
- `-32000`: General error

## Integration Testing

The integration is thoroughly tested in `src/tests/test_a2a_integration.py`, which covers:

1. Basic adapter functionality
2. Component conversion (both directions)
3. API handler functionality for all endpoints
4. Error handling
5. Batch operations

## Best Practices

### When Using the A2A Adapter Directly

1. Enable validation during development to catch issues early
2. Use batch operations when working with multiple components to reduce overhead
3. Use component parts for granular access to specific aspects of a component

### When Interacting with A2A Endpoints

1. Use filters to limit the number of components returned by `getComponents`
2. Specify specific parts when using `getComponentParts` to reduce payload size
3. Use appropriate relationship types when querying related components

## Implementation Considerations

When implementing agents that interact with Empire components via A2A:

1. Handle error responses appropriately
2. Implement proper validation for incoming components
3. Check for required fields in component data
4. Respect the A2A Protocol version requirements

---

## Appendix: Component Types and Parts

### Principle Components

- **Identity**: id, type, version, name
- **Principle**: name, description, importance, example, rationale, evaluation_criteria

### Means Components

- **Identity**: id, type, version, name
- **Means**: name, capabilities, limitations, resources, efficiency, adaptability

### Ends Components

- **Identity**: id, type, version, name
- **Ends**: goal, success_criteria, priority, timeline, dependencies, metrics

### Resentment Components

- **Identity**: id, type, version, name
- **Resentment**: trigger, response, intensity, justification, resolution_path

### Emotion Components

- **Identity**: id, type, version, name
- **Emotion**: emotion_type, intensity, trigger, duration, associated_principle, physical_manifestation
