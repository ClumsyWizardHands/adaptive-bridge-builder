#!/usr/bin/env python3
"""
Principle Engine Fairness Extension Example

This module demonstrates how to use the fairness extension with PrincipleEngine,
providing concrete examples of fairness evaluation in action.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from principle_engine import PrincipleEngine
from fairness_evaluator import FairnessEvaluator, FairnessFlag, FairnessMetric, FairnessAlternative
from principle_engine_fairness import evaluate_fairness, integrate_with_principle_engine, compare_fairness_across_interactions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineFairnessExtension")

def print_fairness_evaluation(evaluation_result: Dict[str, Any]) -> None:
    """
    Print a fairness evaluation result in a readable format.
    
    Args:
        evaluation_result: The fairness evaluation result to print
    """
    print("\n=== Fairness Evaluation Result ===")
    print(f"Overall Score: {evaluation_result['score']:.2f}")
    print(f"Passed Fairness Check: {'Yes' if evaluation_result['is_fair'] else 'No'}")
    print(f"Reason: {evaluation_result['reason']}")
    
    print("\nDetailed Metrics:")
    for metric in evaluation_result.get("metrics", []):
        print(f"  - {metric.dimension}: {metric.score:.2f} (confidence: {metric.confidence:.2f})")
    
    if evaluation_result.get("flags"):
        print("\nFairness Issues:")
        for i, flag in enumerate(evaluation_result["flags"]):
            print(f"  {i+1}. {flag.type}: {flag.description}")
            print(f"     Severity: {flag.severity:.2f}")
            print(f"     Affected Groups: {', '.join(flag.affected_groups)}")
    
    if evaluation_result.get("alternatives"):
        print("\nSuggested Alternatives:")
        for i, alt in enumerate(evaluation_result["alternatives"]):
            print(f"  {i+1}. {alt.description}")
            print(f"     Improvement: {alt.fairness_improvement:.2f}")

def extend_principle_engine_with_fairness() -> PrincipleEngine:
    """
    Create a PrincipleEngine instance and extend it with fairness capabilities.
    
    Returns:
        PrincipleEngine instance with fairness capabilities
    """
    # Create a principle engine with some basic principles
    principles = [
        {
            "id": "user_autonomy",
            "description": "Respect the user's ability to make their own choices.",
            "weight": 0.75
        },
        {
            "id": "transparency",
            "description": "Be clear and transparent about system capabilities and limitations.",
            "weight": 0.8
        },
        {
            "id": "helpfulness",
            "description": "Provide helpful and relevant information to the user.",
            "weight": 0.9
        }
    ]
    
    principle_engine = PrincipleEngine(principles)
    
    # Integrate fairness evaluation capabilities
    integrate_with_principle_engine(principle_engine)
    
    return principle_engine

def demonstrate_fairness_evaluation() -> None:
    """
    Demonstrate fairness evaluation with different types of messages.
    """
    # Create and extend a principle engine
    principle_engine = extend_principle_engine_with_fairness()
    
    # Example messages to evaluate
    messages = [
        {
            "id": "msg1",
            "method": "respond",
            "params": {
                "text": "This advanced feature requires technical expertise. Only experienced users should attempt to use it without guidance."
            }
        },
        {
            "id": "msg2",
            "method": "respond",
            "params": {
                "text": "All users will find this interface intuitive. Everyone can easily complete this process in just a few clicks."
            }
        },
        {
            "id": "msg3",
            "method": "respond",
            "params": {
                "text": "The only way to achieve this goal is to follow our recommended approach. Alternative methods are not supported."
            }
        },
        {
            "id": "msg4",
            "method": "decide",
            "params": {
                "decision": "approve",
                "reason": "The request meets our guidelines for approval."
            }
        }
    ]
    
    # Add historical action for consistency checks
    principle_engine.fairness_evaluator.historical_actions.append({
        "id": "hist1",
        "method": "decide",
        "params": {
            "decision": "deny",
            "reason": "The request does not meet our guidelines for approval."
        }
    })
    
    # Evaluate each message
    evaluation_results = []
    
    for i, message in enumerate(messages):
        print(f"\n\n=== Evaluating Message {i+1} ===")
        print(f"Content: {message['params'].get('text', message['params'])}")
        
        # Evaluate fairness
        result = principle_engine.evaluate_fairness(message)
        evaluation_results.append(result)
        
        # Print results
        print_fairness_evaluation(result)
    
    # Compare fairness across all messages
    print("\n\n=== Fairness Comparison Across Messages ===")
    comparison = compare_fairness_across_interactions(messages, principle_engine)
    
    print(f"Overall Fairness Score: {comparison['overall_score']:.2f}")
    print(f"Consistency Score: {comparison['consistency_score']:.2f}")
    print(f"Messages with Issues: {comparison['interactions_with_issues']} out of {comparison['interaction_count']}")
    
    if comparison.get("bias_patterns"):
        print("\nBias Patterns Detected:")
        for bias_type, info in comparison["bias_patterns"].items():
            print(f"  - {bias_type}: Found in {info['count']} messages (avg severity: {info['avg_severity']:.2f})")
            print(f"    Affected Groups: {', '.join(info['affected_groups'])}")
    
    if comparison.get("recommendations"):
        print("\nRecommendations:")
        for i, rec in enumerate(comparison["recommendations"]):
            print(f"  {i+1}. {rec['description']}")

def demonstrate_fairness_in_decision_making() -> None:
    """
    Demonstrate how fairness evaluation can be used in decision-making processes.
    """
    # Create and extend a principle engine
    principle_engine = extend_principle_engine_with_fairness()
    
    # Example decision to evaluate
    decision = {
        "id": "decision1",
        "method": "make_decision",
        "params": {
            "request_type": "resource_access",
            "user_id": "user123",
            "resource_id": "res456",
            "justification": "The user requires access to complete their assigned task."
        },
        "result": {
            "decision": "approved",
            "explanation": "Request approved based on business need. This type of access is typically granted to senior team members."
        }
    }
    
    # Add historical decisions for consistency checks
    historical_decisions = [
        {
            "id": "hist_decision1",
            "method": "make_decision",
            "params": {
                "request_type": "resource_access",
                "user_id": "user789",
                "resource_id": "res456",
                "justification": "The user requires access to complete their assigned task."
            },
            "result": {
                "decision": "denied",
                "explanation": "Request denied as this access level is restricted. User does not have sufficient privileges."
            }
        }
    ]
    
    for hist_decision in historical_decisions:
        principle_engine.fairness_evaluator.historical_actions.append(hist_decision)
    
    # Evaluate the decision for fairness
    print("\n\n=== Evaluating Decision for Fairness ===")
    print(f"Decision: {decision['result']['decision']}")
    print(f"Explanation: {decision['result']['explanation']}")
    
    # Add context about the decision
    context = {
        "user_metadata": {
            "role": "developer",
            "department": "engineering",
            "tenure": "6 months"
        },
        "resource_metadata": {
            "type": "database",
            "sensitivity": "medium",
            "access_policy": "need-to-know basis"
        }
    }
    
    result = principle_engine.evaluate_fairness(decision, context)
    
    # Print results
    print_fairness_evaluation(result)
    
    # If fairness issues are detected, demonstrate remediation
    if not result["is_fair"] and result.get("alternatives"):
        print("\n=== Remediating Fairness Issues ===")
        
        # Choose the first alternative
        alternative = result["alternatives"][0]
        print(f"Applying Alternative: {alternative.description}")
        
        # Create a remediated decision based on the alternative
        remediated_decision = decision.copy()
        if alternative.replacement_content:
            remediated_decision["result"]["explanation"] = alternative.replacement_content
        
        # Evaluate the remediated decision
        print("\nRe-evaluating Remediated Decision:")
        remediated_result = principle_engine.evaluate_fairness(remediated_decision, context)
        print_fairness_evaluation(remediated_result)

if __name__ == "__main__":
    print("=== Demonstrating Fairness Evaluation with PrincipleEngine ===")
    demonstrate_fairness_evaluation()
    
    print("\n\n" + "="*60)
    print("=== Demonstrating Fairness in Decision Making ===")
    demonstrate_fairness_in_decision_making()