#!/usr/bin/env python3
"""
Fairness Evaluator Example

This module demonstrates the usage of the FairnessEvaluator class to assess
fairness and generate alternatives for potentially biased content.
"""

import json
from fairness_evaluator import FairnessEvaluator, FairnessMetric, FairnessFlag
from principle_engine import PrincipleEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FairnessEvaluatorExample")

def print_metrics(metrics) -> None:
    """Print fairness metrics in a readable format."""
    print("\n=== Fairness Metrics ===")
    for metric in metrics:
        print(f"Dimension: {metric.dimension}")
        print(f"  Score: {metric.score:.2f}")
        print(f"  Confidence: {metric.confidence:.2f}")
        print(f"  Details: {json.dumps(metric.details, indent=2)}")
        print()

def print_flags(flags) -> None:
    """Print fairness flags in a readable format."""
    print("\n=== Fairness Issues Detected ===")
    if not flags:
        print("No fairness issues detected.")
        return
    
    for flag in flags:
        print(f"Issue Type: {flag.type}")
        print(f"Description: {flag.description}")
        print(f"Severity: {flag.severity:.2f}")
        print(f"Affected Groups: {', '.join(flag.affected_groups)}")
        print(f"Problematic Section: {flag.content_reference.get('problematic_section', 'N/A')}")
        print()

def print_alternatives(alternatives) -> None:
    """Print fairness alternatives in a readable format."""
    print("\n=== Suggested Alternatives ===")
    if not alternatives:
        print("No alternatives generated.")
        return
    
    for alt in alternatives:
        print(f"Alternative for Issue: {alt.flag_id}")
        print(f"Description: {alt.description}")
        print(f"Fairness Improvement: {alt.fairness_improvement:.2f}")
        print(f"Impact Assessment: {json.dumps(alt.impact_assessment, indent=2)}")
        if alt.replacement_content:
            print(f"Replacement Content: {alt.replacement_content[:100]}..." if len(alt.replacement_content) > 100 else alt.replacement_content)
        print()

def main() -> None:
    # Example 1: Evaluate message with gender-biased language
    print("\n\nEXAMPLE 1: EVALUATING MESSAGE WITH GENDER-BIASED LANGUAGE")
    message1 = {
        "method": "send_message",
        "params": {
            "text": "When a programmer starts his day, he should check his emails first. Every developer must use his own tools to be productive. If he has any questions, he should ask his team leader."
        }
    }
    
    # Create a FairnessEvaluator instance
    evaluator = FairnessEvaluator()
    
    # Evaluate the message for fairness
    metrics, flags = evaluator.evaluate_message(message1)
    
    # Print metrics and flags
    print_metrics(metrics)
    print_flags(flags)
    
    # Generate and print alternatives
    alternatives = evaluator.generate_alternatives(flags)
    print_alternatives(alternatives)
    
    # Example 2: Evaluate message with assumption biases
    print("\n\n\nEXAMPLE 2: EVALUATING MESSAGE WITH ASSUMPTION BIASES")
    message2 = {
        "method": "send_instruction",
        "params": {
            "text": "Obviously, everyone can easily understand this code. Just quickly implement the feature, it's simply a matter of adding a few lines. Of course, you already know how REST APIs work, so I won't explain that."
        }
    }
    
    # Evaluate the message for fairness
    metrics, flags = evaluator.evaluate_message(message2)
    
    # Print metrics and flags
    print_metrics(metrics)
    print_flags(flags)
    
    # Generate and print alternatives
    alternatives = evaluator.generate_alternatives(flags)
    print_alternatives(alternatives)
    
    # Example 3: Evaluate message with perspective diversity issues
    print("\n\n\nEXAMPLE 3: EVALUATING MESSAGE WITH PERSPECTIVE DIVERSITY ISSUES")
    message3 = {
        "method": "recommend_approach",
        "params": {
            "text": "You must use React for this project. It's the only way to build modern web applications. There is no reason to consider any other framework or library."
        }
    }
    
    # Evaluate the message for fairness
    metrics, flags = evaluator.evaluate_message(message3)
    
    # Print metrics and flags
    print_metrics(metrics)
    print_flags(flags)
    
    # Generate and print alternatives
    alternatives = evaluator.generate_alternatives(flags)
    print_alternatives(alternatives)

if __name__ == "__main__":
    main()
