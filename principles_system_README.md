# Principles-Based Agent System

This system allows you to incorporate moral and ethical principles into your agent's decision-making process. It provides tools for converting plain text principles into a structured format, evaluating actions against these principles, and integrating principle-based guidance into agent behaviors.

## Overview

The Principles System consists of several components:

1. **Principles Converter**: Transforms plain text principles into a structured JSON format.
2. **Principle Engine**: Core engine for storing principles and evaluating items against them.
3. **Principle Action Evaluator**: Extends the engine to evaluate general actions against principles.
4. **Principles Integration**: Provides a simple API for incorporating principles into agent systems.

This system allows your agent to:
- Check if actions align with defined principles
- Offer explanations when actions conflict with principles
- Suggest alternative approaches that would respect principles
- Negotiate to find solutions that meet user goals without violating principles

## Quick Start

Here's how to get started with principle-based guidance in your agent:

```python
from principles_integration import PrinciplesIntegration

# Define your principles as plain text
my_principles = """
1. Respect Privacy
   Never collect or use personal information without explicit permission.

2. Be Truthful
   Always provide accurate information and never intentionally mislead.

3. Protect Security
   Safeguard systems and data against unauthorized access or misuse.
"""

# Create the integration
principles = PrinciplesIntegration(principles_text=my_principles)

# Check if an action aligns with principles
action = "Collect user browsing data without their knowledge"
result = principles.check_action(action)

if result["complies"]:
    # Perform the action
    print(f"Action complies with principles: {action}")
else:
    # Explain why action violates principles and suggest alternatives
    print(f"Action violates principles: {result['explanation']}")
    print("Suggested alternatives:")
    for alt in principles.get_alternatives(action):
        print(f"- {alt}")
```

## Components

### 1. Principles Converter (`principles_converter.py`)

This tool converts plain text principles into a structured JSON format that the Principle Engine can use.

```python
from principles_converter import convert_principles

# Define principles as plain text
principles_text = """
1. First Principle
   Detailed description of the first principle.

2. Second Principle
   Detailed description of the second principle.
"""

# Convert to structured format
structured_principles = convert_principles(principles_text)

# Save to file
import json
with open("my_principles.json", "w") as f:
    json.dump(structured_principles, f, indent=2)
```

### 2. Principle Engine Action Evaluator (`principle_engine_action_evaluator.py`)

This component evaluates general actions against principles and generates explanations and alternatives.

```python
from principle_engine import PrincipleEngine
from principle_engine_action_evaluator import PrincipleActionEvaluator

# Create evaluator with principles file
evaluator = PrincipleActionEvaluator(principles_file="my_principles.json")

# Check action compliance
result = evaluator.check_action_compliance("Add user tracking without consent")

# Generate explanation
explanation = evaluator.generate_explanation(result)

# Get alternative suggestions
alternatives = evaluator.suggest_alternatives("Add user tracking without consent")
```

### 3. Principles Integration (`principles_integration.py`)

This provides a simple API for incorporating principles into agent systems.

```python
from principles_integration import PrinciplesIntegration, PrincipledAgent

# Create a principled agent with your principles
agent = PrincipledAgent(principles_text=my_principles)

# Perform actions with principle checks
result = agent.perform_action("Send an email to all users")

if result["action_performed"]:
    print("Action performed successfully")
else:
    print(f"Action rejected: {result['reason']}")
    print("Alternatives:", result["alternatives"])
```

## Adding Custom Principles

### Format

Principles can be provided in plain text format with each principle starting with a number or bullet point:

```
1. Principle Name
   Detailed description of the principle that explains its meaning and importance.

2. Another Principle
   Description of another principle. The description can span multiple lines.

* Third Principle
  Another description format with bullet points instead of numbers.
```

### Example Principles

```
1. Respect User Privacy
   Privacy is a fundamental right. Systems must collect only necessary data,
   be transparent about collection and usage, and give users control over their information.

2. Ensure Security
   Protect user data and system integrity through appropriate security measures,
   regular updates, and prompt addressing of vulnerabilities.

3. Be Transparent
   Be open and honest about system capabilities, limitations, and how user data is used.
   Avoid deception or misleading information.
```

## Running Examples

The package includes several example scripts that demonstrate the principles system:

1. `principle_action_evaluator_example.py`: Demonstrates evaluating actions against principles.

```bash
# Run with example principles
python src/principle_action_evaluator_example.py

# Run with your own principles file
python src/principle_action_evaluator_example.py --principles my_principles.txt --interactive
```

2. `principles_integration.py`: Shows how to integrate principles into an agent.

```bash
# Run the example agent
python src/principles_integration.py
```

## Customizing Evaluation Logic

The default evaluation logic uses keyword matching and pattern detection to identify potential principle violations. You can customize this logic by extending the `_evaluate_action_against_principle` method in the `PrincipleActionEvaluator` class:

```python
from principle_engine_action_evaluator import PrincipleActionEvaluator

class CustomPrincipleEvaluator(PrincipleActionEvaluator):
    def _evaluate_action_against_principle(self, action, principle, context):
        # Get the base evaluation
        score, violations, recommendations = super()._evaluate_action_against_principle(action, principle, context)
        
        # Add custom evaluation logic
        principle_id = principle["id"]
        
        # Example: Custom logic for privacy principle
        if principle_id == "respect_user_privacy":
            if "tracking" in action.lower() and "consent" not in action.lower():
                score -= 40
                violations.append("Action includes tracking without explicit consent")
                recommendations.append("Add explicit consent mechanisms for any tracking")
        
        return score, violations, recommendations
```

## Integration with LLMs

For more sophisticated principle evaluations, you can integrate with language models by using the `principle_engine_llm.py` module, which extends the base `PrincipleEngine` to leverage LLM capabilities for nuanced reasoning about principles.

## Best Practices

1. **Keep principles clear and specific**: Vague principles are harder to evaluate meaningfully.
2. **Use plain language**: Avoid jargon or overly technical terms in principle descriptions.
3. **Set reasonable thresholds**: The default compliance threshold is 70/100. Adjust this based on your use case.
4. **Provide context**: When evaluating actions, include relevant context for more accurate assessments.
5. **Offer alternatives**: Always suggest constructive alternatives when rejecting an action.

## License

This project is licensed under the terms included in the LICENSE file.
