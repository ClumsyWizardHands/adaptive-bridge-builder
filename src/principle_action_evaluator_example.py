#!/usr/bin/env python3
"""
Example demonstrating how to use custom principles with PrincipleActionEvaluator

This script shows how to:
1. Convert a set of plain text principles to JSON format
2. Initialize the PrincipleActionEvaluator with custom principles
3. Evaluate actions against these principles
4. Generate explanations and alternatives for non-compliant actions
"""

import json
import os
import argparse
import textwrap
from typing import List, Dict, Any

from principles_converter import convert_principles
from principle_engine import PrincipleEngine
from principle_engine_action_evaluator import PrincipleActionEvaluator


def print_separator(title=None) -> None:
    """Print a separator line with optional title."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("-" * 80)


def create_custom_principles_file(principles_text: str, output_file: str = "custom_principles.json") -> str:
    """
    Create a JSON file from custom principles text.
    
    Args:
        principles_text: The text containing principles
        output_file: Path to output JSON file
        
    Returns:
        Path to the created file
    """
    # Convert principles to structured format
    principles = convert_principles(principles_text)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(principles, f, indent=2)
    
    print(f"Created principles file: {output_file} with {len(principles)} principles")
    return output_file


def demonstrate_evaluator(evaluator: PrincipleActionEvaluator) -> None:
    """
    Demonstrate the PrincipleActionEvaluator with some example actions.
    
    Args:
        evaluator: PrincipleActionEvaluator instance
    """
    print_separator("PRINCIPLE ACTION EVALUATOR DEMONSTRATION")
    
    # Show principles summary
    print("Loaded Principles:")
    print(evaluator.get_principles_summary())
    
    print_separator("EXAMPLE ACTIONS EVALUATION")
    
    # Example actions to evaluate (some good, some questionable)
    test_actions = [
        "Send an email to all users informing them of a security update",
        "Collect user browsing data without their knowledge for targeted advertising",
        "Delete a user's account immediately upon their request",
        "Override security protocols to expedite a critical deployment",
        "Add a feature that tracks user location in the background",
    ]
    
    # Evaluate each action
    for action in test_actions:
        print_separator(f"Evaluating: {action}")
        
        result = evaluator.check_action_compliance(action)
        explanation = evaluator.generate_explanation(result)
        
        print(f"Compliance Score: {result['overall_score']:.1f}/100")
        print(f"Complies with principles: {result['complies']}")
        print("\nExplanation:")
        for line in explanation.split('\n'):
            print(textwrap.fill(line, width=78) or line)
        
        if not result['complies']:
            print("\nSuggested Alternatives:")
            alternatives = evaluator.suggest_alternatives(action)
            for i, alt in enumerate(alternatives, 1):
                print(f"{i}. {textwrap.fill(alt, width=74, initial_indent='   ', subsequent_indent='      ')}")
        
        input("\nPress Enter to continue...")


def interactive_demo(evaluator: PrincipleActionEvaluator) -> None:
    """
    Run an interactive demo where the user can input actions to evaluate.
    
    Args:
        evaluator: PrincipleActionEvaluator instance
    """
    print_separator("INTERACTIVE PRINCIPLE EVALUATION")
    print("Enter actions to check against principles. Type 'exit' to quit.")
    
    while True:
        print("\n" + "-" * 80)
        action = input("Enter an action to evaluate: ")
        if action.lower() in ('exit', 'quit', 'q'):
            break
        
        if not action.strip():
            continue
        
        result = evaluator.check_action_compliance(action)
        explanation = evaluator.generate_explanation(result)
        
        print(f"\nCompliance Score: {result['overall_score']:.1f}/100")
        print(f"Complies with principles: {result['complies']}")
        print("\nExplanation:")
        for line in explanation.split('\n'):
            print(textwrap.fill(line, width=78) or line)
        
        if not result['complies']:
            print("\nSuggested Alternatives:")
            alternatives = evaluator.suggest_alternatives(action)
            for i, alt in enumerate(alternatives, 1):
                print(f"{i}. {textwrap.fill(alt, width=74, initial_indent='   ', subsequent_indent='      ')}")


def main() -> int:
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate PrincipleActionEvaluator with custom principles.")
    parser.add_argument("--principles", "-p", help="Text file containing principles")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode after examples")
    parser.add_argument("--output", "-o", default="custom_principles.json", help="Output JSON file for principles")
    
    args = parser.parse_args()
    
    # Check if principles file was provided
    if args.principles:
        with open(args.principles, 'r') as f:
            principles_text = f.read()
        
        # Create principles file
        principles_file = create_custom_principles_file(principles_text, args.output)
    else:
        # Use default principles from PrincipleEngine
        principles_file = None
        print("No custom principles provided. Using default principles.")
    
    # Create evaluator
    evaluator = PrincipleActionEvaluator(principles_file=principles_file)
    
    # Run the demonstrations
    demonstrate_evaluator(evaluator)
    
    # Run interactive mode if requested
    if args.interactive:
        interactive_demo(evaluator)
    
    print_separator("DEMONSTRATION COMPLETE")
    return 0


# Example principles for testing (if no file is provided)
EXAMPLE_PRINCIPLES = """
1. Respect User Privacy
   Privacy is a fundamental right. Systems must collect only necessary data,
   be transparent about collection and usage, and give users control over their information.

2. Ensure Security
   Protect user data and system integrity through appropriate security measures,
   regular updates, and prompt addressing of vulnerabilities.

3. Be Transparent
   Be open and honest about system capabilities, limitations, and how user data is used.
   Avoid deception or misleading information.

4. Prioritize User Control
   Users should maintain meaningful control over systems that affect them,
   with the ability to override automated decisions where appropriate.

5. Design for All
   Create systems that are accessible and usable by people of diverse abilities,
   backgrounds, and resources.

6. Prevent Harm
   Anticipate and prevent potential harms from system use, including both
   direct and indirect impacts on individuals and society.

7. Be Accountable
   Take responsibility for system outcomes, provide clear paths for redress
   when issues occur, and learn from mistakes.
"""


if __name__ == "__main__":
    # If no input file is provided, create one with example principles for demo purposes
    if len(os.sys.argv) == 1:
        print("No parameters provided. Creating example principles file for demonstration.")
        create_custom_principles_file(EXAMPLE_PRINCIPLES, "example_principles.json")
        os.sys.argv.extend(["--principles", "example_principles.json", "--interactive"])
    
    main()
