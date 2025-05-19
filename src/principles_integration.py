#!/usr/bin/env python3
"""
Principles Integration System

This module provides an easy way to integrate custom principles into an agent.
It combines the principle conversion, evaluation, and action checking processes
into a single interface that can be easily incorporated into any agent system.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from principles_converter import convert_principles
from principle_engine import PrincipleEngine
from principle_engine_action_evaluator import PrincipleActionEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrinciplesIntegration")


class PrinciplesIntegration:
    """
    Central integration point for principle-based guidance in agents.
    
    This class simplifies the process of loading principles, evaluating actions,
    and incorporating principle-based decision-making into agent behaviors.
    """
    
    def __init__(
        self, 
        principles_text: Optional[str] = None, 
        principles_file: Optional[str] = None,
        output_file: str = "custom_principles.json",
        threshold: float = 70.0
    ):
        """
        Initialize the PrinciplesIntegration.
        
        Args:
            principles_text: Text containing principle definitions (one per paragraph)
            principles_file: Path to an existing principles JSON file
            output_file: Where to save converted principles (if principles_text is provided)
            threshold: Score threshold for action compliance (0-100)
        """
        self.threshold = threshold
        
        # Process input sources
        if principles_text:
            # Convert text principles and save to file
            principles = convert_principles(principles_text)
            with open(output_file, 'w') as f:
                json.dump(principles, f, indent=2)
            logger.info(f"Converted {len(principles)} principles to {output_file}")
            self.principles_file = output_file
        elif principles_file:
            # Use existing principles file
            self.principles_file = principles_file
            logger.info(f"Using existing principles from {principles_file}")
        else:
            # No principles provided, use defaults
            self.principles_file = None
            logger.info("No principles provided, using default system principles")
        
        # Create the principle engine and evaluator
        self.principle_engine = PrincipleEngine(self.principles_file)
        self.action_evaluator = PrincipleActionEvaluator(self.principle_engine)
        
        # Load a version of principles with processed metadata
        self.principles = self.principle_engine.principles
        logger.info(f"PrinciplesIntegration initialized with {len(self.principles)} principles")
    
    def check_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if an action complies with principles.
        
        Args:
            action: Description of the action to evaluate
            context: Additional context about the action (optional)
            
        Returns:
            Dictionary with compliance result and details
        """
        return self.action_evaluator.check_action_compliance(action, context)
    
    def should_perform_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Determine whether an action should be performed based on principles.
        
        Args:
            action: Description of the action to evaluate
            context: Additional context about the action (optional)
            
        Returns:
            Tuple of (should_perform, reason)
        """
        result = self.check_action(action, context)
        should_perform = result["overall_score"] >= self.threshold
        
        if should_perform:
            reason = "Action aligns with principles"
        else:
            explanation = result["explanation"].split("\n")[0]  # Get first line of explanation
            reason = f"Action conflicts with principles: {explanation}"
        
        return should_perform, reason
    
    def get_response_for_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a complete response explaining compliance or issues with an action.
        
        Args:
            action: Description of the action to evaluate
            context: Additional context about the action (optional)
            
        Returns:
            Formatted response string
        """
        result = self.check_action(action, context)
        return self.action_evaluator.generate_explanation(result)
    
    def get_alternatives(self, action: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get alternative suggestions for an action that may violate principles.
        
        Args:
            action: Description of the action to evaluate
            context: Additional context about the action (optional)
            
        Returns:
            List of alternative suggestions
        """
        return self.action_evaluator.suggest_alternatives(action, context)
    
    def wrap_agent_action(
        self, 
        action: str, 
        action_handler: callable, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Wrap an agent's action with principle checks.
        
        Args:
            action: Description of the action
            action_handler: Function to call if action complies with principles
            context: Additional context about the action (optional)
            
        Returns:
            Response dictionary
        """
        should_perform, reason = self.should_perform_action(action, context)
        
        if should_perform:
            try:
                # Execute the action
                result = action_handler()
                return {
                    "success": True,
                    "action_performed": True,
                    "result": result,
                    "reason": reason
                }
            except Exception as e:
                return {
                    "success": False,
                    "action_performed": False,
                    "error": str(e),
                    "reason": f"Action failed with error: {str(e)}"
                }
        else:
            # Action violates principles
            alternatives = self.get_alternatives(action, context)
            
            return {
                "success": False,
                "action_performed": False,
                "reason": reason,
                "alternatives": alternatives,
                "explanation": self.get_response_for_action(action, context)
            }
    
    def get_principles_summary(self) -> str:
        """
        Get a formatted summary of all principles.
        
        Returns:
            Formatted string with principle descriptions
        """
        return self.action_evaluator.get_principles_summary()
    
    def set_compliance_threshold(self, threshold: float) -> None:
        """
        Set the compliance threshold score (0-100).
        
        Args:
            threshold: New threshold value (0-100)
        """
        self.threshold = max(0, min(100, threshold))
        logger.info(f"Compliance threshold set to {self.threshold}")


# Example usage as agent middleware
class PrincipledAgent:
    """Example agent that checks actions against principles."""
    
    def __init__(self, principles_text: Optional[str] = None, principles_file: Optional[str] = None):
        """Initialize the principled agent."""
        self.principles = PrinciplesIntegration(principles_text, principles_file)
        logger.info("PrincipledAgent initialized")
    
    def perform_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform an action after checking it against principles.
        
        Args:
            action: The action to perform
            context: Additional context
            
        Returns:
            Response dictionary
        """
        def action_handler():
            # This would normally contain the logic to actually perform the action
            return {"status": "completed", "action": action}
        
        return self.principles.wrap_agent_action(action, action_handler, context)
    
    def discuss_principles(self) -> str:
        """Get a summary of the agent's guiding principles."""
        return self.principles.get_principles_summary()


# Example function
def main():
    """Demonstrate the PrinciplesIntegration system."""
    # Example principles
    example_principles = """
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
    
    # Create a principled agent
    agent = PrincipledAgent(principles_text=example_principles)
    
    # Print the agent's principles
    print("AGENT PRINCIPLES:")
    print(agent.discuss_principles())
    print("\n" + "=" * 80 + "\n")
    
    # Test some actions
    test_actions = [
        "Send a notification to all users about a security update",
        "Collect browsing data without telling users",
        "Automatically update security settings for all accounts",
        "Add targeted ads based on users' private messages"
    ]
    
    for action in test_actions:
        print(f"ACTION: {action}")
        print("-" * 80)
        
        result = agent.perform_action(action)
        
        if result["action_performed"]:
            print(f"✅ Action performed: {result['reason']}")
        else:
            print(f"❌ Action rejected: {result['reason']}")
            
            if "alternatives" in result and result["alternatives"]:
                print("\nSuggested alternatives:")
                for i, alt in enumerate(result["alternatives"], 1):
                    print(f"{i}. {alt}")
                    
            if "explanation" in result:
                print("\nDetailed explanation:")
                for line in result["explanation"].split("\n")[:5]:  # Show first 5 lines
                    print(f"  {line}")
                if len(result["explanation"].split("\n")) > 5:
                    print("  ...")
        
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
