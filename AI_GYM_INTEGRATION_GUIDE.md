# AI Principles Gym Integration Guide

## Overview

The Adaptive Bridge Builder now fully supports AI Principles Gym scenarios! The integration leverages the Bridge Builder's advanced capabilities including:

- **Principle Engine**: Evaluates decisions against core principles
- **Emotional Intelligence**: Considers emotional context and stress levels
- **Ethical Dilemma Resolver**: Specialized handling for moral choices
- **Fairness Evaluator**: Ensures equitable outcomes
- **Conflict Resolution**: Mediates between competing interests
- **Learning System**: Improves decisions over time

## 🚀 Quick Start

### 1. Start the Bridge Builder Server

```bash
# Windows
run_server.bat

# Linux/Mac
./run_server.sh
```

The server will start on `http://localhost:8080`

### 2. Verify Gym Support

Check that the server has Gym support enabled:

```bash
curl http://localhost:8080/health
```

You should see:
```json
{
  "status": "healthy",
  "agent": "available",
  "gym_adapter": "available"
}
```

### 3. Send Your First Scenario

The Bridge Builder accepts scenarios at the `/process` endpoint:

```python
import requests

scenario = {
    "scenario": {
        "execution_id": "unique-id",
        "description": "Description of the situation",
        "actors": ["Actor1", "Actor2"],
        "resources": ["Resource1", "Resource2"],
        "constraints": ["Constraint1", "Constraint2"],
        "choice_options": [
            {
                "id": "option1",
                "name": "Option Name",
                "description": "What this option does"
            }
        ],
        "time_limit": 30,
        "archetype": "ETHICAL_DILEMMA",
        "stress_level": 0.5
    },
    "history": [],
    "metadata": {
        "framework": "principles_gym",
        "version": "1.0.0",
        "request_id": "request-id"
    }
}

response = requests.post(
    "http://localhost:8080/process",
    json=scenario
)

result = response.json()
print(f"Decision: {result['action']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Confidence: {result['confidence']}")
```

## 📋 Integration Details

### Endpoint

- **URL**: `http://localhost:8080/process`
- **Method**: POST
- **Content-Type**: application/json

### Request Format

Your AI Gym must send requests in this format:

```json
{
  "scenario": {
    "execution_id": "string",
    "description": "string",
    "actors": ["string"],
    "resources": ["string"],
    "constraints": ["string"],
    "choice_options": [
      {
        "id": "string",
        "name": "string",
        "description": "string"
      }
    ],
    "time_limit": 30,
    "archetype": "ETHICAL_DILEMMA",
    "stress_level": 0.0-1.0
  },
  "history": [],
  "metadata": {
    "framework": "principles_gym",
    "version": "1.0.0",
    "request_id": "string"
  }
}
```

### Response Format

The Bridge Builder returns:

```json
{
  "action": "option_id",
  "reasoning": "Detailed explanation of the decision",
  "confidence": 0.85,
  "target": "Optional specific target" 
}
```

## 🎯 Supported Scenario Types

### 1. Ethical Dilemmas
- Trolley problems
- Medical ethics
- AI safety decisions
- Privacy vs security

### 2. Resource Allocation
- Limited resources distribution
- Prioritization decisions
- Fairness in allocation
- Efficiency vs equity

### 3. Conflict Resolution
- Interpersonal conflicts
- Team disagreements
- Competing interests
- Negotiation scenarios

### 4. Trust Building
- Establishing credibility
- Repairing relationships
- Building consensus

### 5. Crisis Management
- Emergency responses
- High-stress decisions
- Time-critical choices

## 🧠 How Decisions Are Made

1. **Multi-Criteria Evaluation**: Each option is scored on:
   - Principle alignment (fairness, harmony, adaptability)
   - Fairness to all parties
   - Archetype-specific criteria
   - Emotional appropriateness

2. **Weighted Scoring**: Different criteria are weighted based on the scenario archetype

3. **Reasoning Generation**: Comprehensive explanations include:
   - Primary rationale
   - Principle considerations
   - Constraint compliance
   - Historical learning

4. **Confidence Calculation**: Based on:
   - Score differentials
   - Stress level impact
   - Principle alignment strength

## 🔧 Testing Your Integration

Use the provided test client:

```bash
python src/test_gym_integration.py
```

This runs through multiple scenario types and shows how the Bridge Builder responds.

## 📊 Advanced Features

### Learning from History

The Bridge Builder learns from past scenarios:

```json
"history": [
  {
    "action": "previous_action",
    "timestamp": "2025-01-06T12:00:00Z",
    "reasoning": "why that action was taken"
  }
]
```

### Stress Level Impact

Higher stress levels (0.7-1.0) lead to:
- Preference for cautious approaches
- Lower confidence scores
- More detailed reasoning

### Custom Principles

You can add domain-specific principles by modifying the principle engine configuration.

## 🛠️ Troubleshooting

### Connection Issues

1. Verify server is running: `curl http://localhost:8080/health`
2. Check firewall settings
3. Ensure correct JSON format

### Unexpected Decisions

1. Review principle weights in the adapter
2. Check scenario archetype assignment
3. Examine confidence scores

### Performance

1. The Bridge Builder caches similar scenarios
2. Batch requests when possible
3. Monitor server resources

## 📚 Example Scenarios

See `src/test_gym_integration.py` for complete examples including:
- Self-driving car dilemmas
- Medical resource allocation
- Workplace conflict resolution

## 🤝 Support

For issues or questions:
1. Check server logs for detailed error messages
2. Use the test client to verify your scenario format
3. Review the adapter code for customization options

The Adaptive Bridge Builder is now ready to help your AI Gym agents make principled, intelligent decisions!
