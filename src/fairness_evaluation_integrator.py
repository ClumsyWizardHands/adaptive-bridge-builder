"""
Fairness Evaluation Integrator

This module integrates the fairness_evaluation.py module with the PrincipleEngine
to extend it with comprehensive fairness evaluation capabilities.
"""

import logging
from principle_engine import PrincipleEngine
from fairness_evaluation import evaluate_fairness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FairnessEvaluationIntegrator")

def extend_principle_engine() -> int:
    """
    Extends the PrincipleEngine class with the evaluate_fairness method.
    
    This function adds comprehensive fairness evaluation capabilities to the 
    PrincipleEngine, enabling it to:
    - Ensure consistent application of rules across interactions
    - Check for bias in proposed actions or messages
    - Compare current actions against historical patterns
    - Generate fairness scores and flag potentially biased actions
    - Suggest alternative unbiased approaches when needed
    
    Call this function once at application startup to integrate the fairness
    evaluation functionality into the PrincipleEngine.
    
    Returns:
        bool: True if the extension was successful, False otherwise
    """
    # Check if already extended
    if hasattr(PrincipleEngine, 'evaluate_fairness'):
        logger.info("PrincipleEngine already has fairness evaluation capabilities")
        return False
    
    # Add the evaluate_fairness method to PrincipleEngine
    PrincipleEngine.evaluate_fairness = evaluate_fairness
    
    # Log success
    logger.info("PrincipleEngine successfully extended with fairness evaluation capabilities")
    return True


# Example usage
if __name__ == "__main__":
    # Extend the PrincipleEngine with fairness evaluation
    extend_principle_engine()
    
    # Create a PrincipleEngine instance
    engine = PrincipleEngine()
    
    # Example action to evaluate
    example_action = {
        "method": "route",
        "params": {
            "destination": "agent-001",
            "message": "Hello world",
            "priority": 3  # High priority that might not be justified
        },
        "id": "test-1"
    }
    
    # Example historical actions
    historical_actions = [
        {
            "method": "route",
            "params": {
                "destination": "agent-001",
                "message": "Regular message",
                "priority": 0
            },
            "id": "hist-1"
        },
        {
            "method": "route",
            "params": {
                "destination": "agent-002",
                "message": "Another message",
                "priority": 0
            },
            "id": "hist-2"
        }
    ]
    
    # Evaluate fairness
    result = engine.evaluate_fairness(example_action, historical_actions, "test-agent")
    
    # Print results
    print(f"Fairness Score: {result['fairness_score']:.2f}")
    print("\nBias Flags:")
    for flag in result["bias_flags"]:
        print(f"- {flag['description']} ({flag['severity']} severity)")
    
    print("\nAlternative Suggestions:")
    for suggestion in result["alternative_suggestions"]:
        print(f"- {suggestion['description']}")
    
    print(f"\nSummary: {result['summary']}")