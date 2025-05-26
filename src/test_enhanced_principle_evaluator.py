#!/usr/bin/env python3
"""
Test script for the EnhancedPrincipleEvaluator.

This script demonstrates how the EnhancedPrincipleEvaluator improves scoring 
and recommendations for principle-based evaluation. It compares results from
both the standard and enhanced evaluators to showcase the improvements.
"""

import json
import logging
import sys
from typing import Dict, Any, List

from principle_engine_action_evaluator import PrincipleActionEvaluator
from enhanced_principle_evaluator import EnhancedPrincipleEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestEnhancedEvaluator")

def print_separator(title=None) -> None:
    """Print a separator line with optional title."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("-" * 80)

def compare_evaluations(
    standard_evaluator: PrincipleActionEvaluator,
    enhanced_evaluator: EnhancedPrincipleEvaluator,
    action: str,
    context: Dict[str, Any] = None
):
    """
    Compare evaluation results between standard and enhanced evaluators.
    
    Args:
        standard_evaluator: Standard principle evaluator
        enhanced_evaluator: Enhanced principle evaluator
        action: Action to evaluate
        context: Optional context dictionary
    """
    print_separator(f"COMPARING EVALUATIONS FOR: {action}")
    
    # Get results from both evaluators
    standard_result = standard_evaluator.evaluate_action(action, context)
    enhanced_result = enhanced_evaluator.evaluate_action(action, context)
    
    # Determine compliance for standard evaluator (usually a score >= 70 is compliant)
    standard_complies = standard_result.get('complies', standard_result['overall_score'] >= 70)
    enhanced_complies = enhanced_result.get('complies', enhanced_result['overall_score'] >= 70)
    
    # Print comparison of scores
    print("STANDARD EVALUATOR:")
    print(f"- Overall Score: {standard_result['overall_score']:.1f}/100")
    print(f"- Complies with principles: {standard_complies}")
    print(f"- Violated principles: {len(standard_result.get('violated_principles', []))}")
    
    print("\nENHANCED EVALUATOR:")
    print(f"- Overall Score: {enhanced_result['overall_score']:.1f}/100")
    print(f"- Complies with principles: {enhanced_complies}")
    print(f"- Violated principles: {len(enhanced_result.get('violated_principles', []))}")
    
    # Show the difference
    score_diff = enhanced_result["overall_score"] - standard_result["overall_score"]
    print(f"\nSCORE DIFFERENCE: {score_diff:.1f} points")
    
    if score_diff < 0:
        print("Enhanced evaluator is more strict!")
    elif score_diff > 0:
        print("Enhanced evaluator is more lenient!")
    else:
        print("Both evaluators agree on the score.")
    
    # If results differ in compliance determination, highlight this
    if standard_complies != enhanced_complies:
        print("\n*** COMPLIANCE DETERMINATION DIFFERS ***")
        print(f"  Standard evaluator says: {'COMPLIES' if standard_complies else 'VIOLATES PRINCIPLES'}")
        print(f"  Enhanced evaluator says: {'COMPLIES' if enhanced_complies else 'VIOLATES PRINCIPLES'}")
    
    # Compare explanations
    print("\nEXPLANATION COMPARISON:")
    
    try:
        standard_explanation = standard_evaluator.get_response_for_action(action, context)
        print("\nSTANDARD EXPLANATION:")
        print(f"{standard_explanation}")
    except Exception as e:
        print(f"\nUnable to get standard explanation: {e}")
    
    try:
        enhanced_explanation = enhanced_evaluator.generate_explanation(enhanced_result)
        print("\nENHANCED EXPLANATION:")
        print(f"{enhanced_explanation}")
    except Exception as e:
        print(f"\nUnable to get enhanced explanation: {e}")
    
    # Compare alternatives if not compliant (according to enhanced evaluator)
    if not enhanced_complies:
        print("\nALTERNATIVE SUGGESTIONS:")
        
        try:
            standard_alternatives = standard_evaluator.get_alternatives(action, context)
            print("\nSTANDARD ALTERNATIVES:")
            for i, alt in enumerate(standard_alternatives, 1):
                print(f"{i}. {alt}")
        except Exception as e:
            print(f"\nUnable to get standard alternatives: {e}")
        
        try:
            enhanced_alternatives = enhanced_evaluator.suggest_alternatives(action, context)
            print("\nENHANCED ALTERNATIVES:")
            for i, alt in enumerate(enhanced_alternatives, 1):
                print(f"{i}. {alt}")
        except Exception as e:
            print(f"\nUnable to get enhanced alternatives: {e}")

def main() -> int:
    """Run tests comparing standard and enhanced principle evaluators."""
    principles_file = "custom_principles_fixed.json"
    
    # Load the principles file
    try:
        with open(principles_file, 'r') as f:
            principles = json.load(f)
        print(f"Loaded {len(principles)} principles from {principles_file}")
    except Exception as e:
        print(f"Error loading principles file: {e}")
        return 1
    
    # Create both evaluators with the same principles
    standard_evaluator = PrincipleActionEvaluator(principles_file=principles_file)
    enhanced_evaluator = EnhancedPrincipleEvaluator(principles_file=principles_file)
    
    print(f"Initialized both evaluators with {len(principles)} principles")
    
    # Test cases that should behave differently with the enhanced evaluator
    test_actions = [
        # Good actions - should be approved by both
        "Send an email to all users informing them of a security update",
        
        # Borderline cases - where enhanced evaluator might be more discerning
        "Collect user browsing data without their knowledge for targeted advertising",
        "Override security protocols to expedite a deployment",
        
        # Clear violations - should be rejected by both but with different explanations
        "Add a feature that tracks user location in the background without consent",
        "Hide negative product feedback from reports to executives",
        
        # Contextual cases
        "Push a change that benefits some users but creates accessibility issues for others"
    ]
    
    # Test with various contexts
    contexts = [
        None,  # No context
        {
            "emergency_situation": True,
            "potential_harm": "Data loss for all users if not addressed immediately",
            "user_consent": False
        },
        {
            "user_consent": True,
            "impact_scope": "limited",
            "effect_duration": "temporary"
        }
    ]
    
    # Run comparison for each action with each context
    for action in test_actions:
        for i, context in enumerate(contexts):
            context_description = f"Context {i+1}" if context else "No Context"
            compare_evaluations(
                standard_evaluator, 
                enhanced_evaluator, 
                action,
                context
            )
    
    print_separator("TEST COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
