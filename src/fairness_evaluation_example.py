#!/usr/bin/env python3
"""
Fairness Evaluation Example

This module provides a practical example of using the fairness evaluation
functionality with PrincipleEngine in real-world scenarios.
"""

import json
import logging
from principle_engine import PrincipleEngine
from fairness_evaluation_integrator import extend_principle_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FairnessExample")

def run_message_routing_example() -> None:
    """Run an example of fairness evaluation in message routing."""
    logger.info("Running message routing fairness example")
    
    # Extend the PrincipleEngine with fairness evaluation
    extend_principle_engine()
    
    # Create a PrincipleEngine instance
    engine = PrincipleEngine()
    
    # Example: A set of historical message routing actions with different priorities
    historical_actions = [
        # Standard priority messages
        {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "conversation_id": "conv-123",
                "destination": "agent-001",
                "message": {
                    "content": "Standard request for agent 1",
                    "type": "inquiry"
                },
                "priority": 0  # Standard priority
            },
            "id": "hist-1"
        },
        {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "conversation_id": "conv-124",
                "destination": "agent-002",
                "message": {
                    "content": "Standard request for agent 2",
                    "type": "inquiry"
                },
                "priority": 0  # Standard priority
            },
            "id": "hist-2"
        },
        # One emergency message with justified high priority
        {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "conversation_id": "conv-125",
                "destination": "emergency-response",
                "message": {
                    "content": "Emergency alert: System failure detected",
                    "type": "alert"
                },
                "priority": 5,  # High priority
                "emergency": True,
                "justification": "Critical system failure"
            },
            "id": "hist-3"
        }
    ]
    
    # Example 1: A standard message with unjustified high priority
    unfair_action = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": "conv-126",
            "destination": "agent-001",
            "message": {
                "content": "Standard request with unjustified high priority",
                "type": "inquiry"
            },
            "priority": 5  # High priority without justification
        },
        "id": "test-1"
    }
    
    # Example 2: A standard message with proper priority
    fair_action = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": "conv-127",
            "destination": "agent-001",
            "message": {
                "content": "Standard request with appropriate priority",
                "type": "inquiry"
            },
            "priority": 0  # Standard priority
        },
        "id": "test-2"
    }
    
    # Example 3: A justified high priority message
    justified_action = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": "conv-128",
            "destination": "emergency-response",
            "message": {
                "content": "Emergency alert: Database connection lost",
                "type": "alert"
            },
            "priority": 5,  # High priority
            "emergency": True,
            "justification": "Critical database failure"
        },
        "id": "test-3"
    }
    
    # Evaluate and display results for each example
    print("\n===== FAIRNESS EVALUATION EXAMPLES =====\n")
    
    print("Example 1: Unjustified High Priority Message")
    result1 = engine.evaluate_fairness(unfair_action, historical_actions, "test-agent")
    print_evaluation_results(result1)
    
    print("\nExample 2: Fair Standard Priority Message")
    result2 = engine.evaluate_fairness(fair_action, historical_actions, "test-agent")
    print_evaluation_results(result2)
    
    print("\nExample 3: Justified High Priority Message")
    result3 = engine.evaluate_fairness(justified_action, historical_actions, "test-agent")
    print_evaluation_results(result3)

def run_resource_allocation_example() -> None:
    """Run an example of fairness evaluation in resource allocation."""
    logger.info("Running resource allocation fairness example")
    
    # Historical resource allocation actions
    historical_actions = [
        # Equal resource allocations to different groups
        {
            "type": "resource_allocation",
            "resource": "compute_time",
            "recipient_id": "group_A",
            "amount": 100,
            "timestamp": "2025-05-15T10:00:00Z"
        },
        {
            "type": "resource_allocation",
            "resource": "compute_time",
            "recipient_id": "group_B",
            "amount": 100,
            "timestamp": "2025-05-15T10:05:00Z"
        },
        {
            "type": "resource_allocation",
            "resource": "compute_time",
            "recipient_id": "group_C",
            "amount": 100,
            "timestamp": "2025-05-15T10:10:00Z"
        }
    ]
    
    # Example 1: Biased allocation giving more resources to one group
    biased_allocation = {
        "type": "resource_allocation",
        "resource": "compute_time",
        "recipient_id": "group_A",  # Same group getting different treatment
        "amount": 250,  # Much higher allocation
        "timestamp": "2025-05-18T14:30:00Z"
    }
    
    # Example 2: Fair allocation consistent with historical patterns
    fair_allocation = {
        "type": "resource_allocation",
        "resource": "compute_time",
        "recipient_id": "group_B",
        "amount": 100,  # Consistent with historical allocations
        "timestamp": "2025-05-18T14:35:00Z"
    }
    
    # Evaluate and display results
    print("\n===== RESOURCE ALLOCATION EXAMPLES =====\n")
    
    print("Example 1: Biased Resource Allocation")
    result1 = engine.evaluate_fairness(biased_allocation, historical_actions, "resource-allocator")
    print_evaluation_results(result1)
    
    print("\nExample 2: Fair Resource Allocation")
    result2 = engine.evaluate_fairness(fair_allocation, historical_actions, "resource-allocator")
    print_evaluation_results(result2)

def print_evaluation_results(result) -> None:
    """Print evaluation results in a readable format."""
    print(f"Fairness Score: {result['fairness_score']:.2f}")
    
    if result["bias_flags"]:
        print("\nBias Flags:")
        for flag in result["bias_flags"]:
            print(f"- {flag['description']} ({flag['severity']} severity)")
    else:
        print("\nBias Flags: None detected")
    
    if result["alternative_suggestions"]:
        print("\nAlternative Suggestions:")
        for suggestion in result["alternative_suggestions"]:
            print(f"- {suggestion['description']}")
    
    print(f"\nSummary: {result['summary']}")

def practical_usage_guide() -> None:
    """Print a practical guide on integrating fairness evaluation."""
    print("\n===== PRACTICAL USAGE GUIDE =====\n")
    print("How to Integrate Fairness Evaluation in Your System:\n")
    
    print("1. Initialization:")
    print("   # Import the necessary modules")
    print("   from principle_engine import PrincipleEngine")
    print("   from fairness_evaluation_integrator import extend_principle_engine")
    print("")
    print("   # Extend the PrincipleEngine with fairness evaluation (do this once at startup)")
    print("   extend_principle_engine()")
    print("   engine = PrincipleEngine()")
    print("")
    
    print("2. Collecting Historical Actions:")
    print("   # Maintain a database or cache of historical actions")
    print("   # Retrieve relevant historical actions when needed")
    print("   historical_actions = retrieve_historical_actions(action_type='route')")
    print("")
    
    print("3. Pre-execution Fairness Check:")
    print("   def process_action(action, agent_id):")
    print("       # Evaluate fairness before executing")
    print("       fairness_result = engine.evaluate_fairness(action, historical_actions, agent_id)")
    print("       ")
    print("       # Check if action has significant fairness issues")
    print("       if fairness_result['fairness_score'] < 0.7:")
    print("           # Log the fairness concerns")
    print("           log_fairness_issues(fairness_result)")
    print("           ")
    print("           # Option 1: Automatically use a suggested alternative")
    print("           if fairness_result['alternative_suggestions']:")
    print("               action = fairness_result['alternative_suggestions'][0]['action']")
    print("               log_action_modification('Using more fair alternative')")
    print("           ")
    print("           # Option 2: Request human review for low fairness scores")
    print("           if fairness_result['fairness_score'] < 0.5:")
    print("               await_human_approval(action, fairness_result)")
    print("       ")
    print("       # Execute the (potentially modified) action")
    print("       execute_action(action)")
    print("")
    
    print("4. Learning from Evaluations:")
    print("   # Periodically analyze fairness evaluations to identify systemic issues")
    print("   # Update rules and policies based on fairness findings")
    print("   scheduled_fairness_review(last_week_evaluations)")
    print("")
    
    print("5. Continuous Improvement:")
    print("   # Track fairness scores over time and set improvement goals")
    print("   # Regularly update your historical action dataset")
    print("   # Create specialized fairness tests for critical operations")

if __name__ == "__main__":
    # Extend the PrincipleEngine with fairness evaluation
    extend_principle_engine()
    
    # Run examples
    run_message_routing_example()
    run_resource_allocation_example()
    
    # Print practical usage guide
    practical_usage_guide()
