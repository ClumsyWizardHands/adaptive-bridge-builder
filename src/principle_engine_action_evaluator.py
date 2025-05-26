#!/usr/bin/env python3
"""
Action-oriented PrincipleEngine Extension

This module extends the core PrincipleEngine to evaluate general actions
against principles, allowing for more flexible principle application
beyond just message evaluation.
"""

import json
import logging
import textwrap
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timezone

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleActionEvaluator")


class PrincipleActionEvaluator:
    """
    Extension of PrincipleEngine that evaluates general actions against principles.
    
    This class provides methods to evaluate arbitrary actions described in natural language,
    determine principle violations, suggest alternatives, and manage interactions.
    """
    
    def __init__(self, principle_engine: Optional[PrincipleEngine] = None, principles_file: Optional[str] = None) -> None:
        """
        Initialize the PrincipleActionEvaluator.
        
        Args:
            principle_engine: Existing PrincipleEngine instance, or None to create a new one
            principles_file: Path to principles JSON file (used only if principle_engine is None)
        """
        self.principle_engine = principle_engine or PrincipleEngine(principles_file)
        self.evaluation_history = []
        self.violation_responses = {}  # Cache of response templates for principle violations
        
        logger.info(f"PrincipleActionEvaluator initialized with {len(self.principle_engine.principles)} principles")
    
    def evaluate_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate an action against all principles.
        
        Args:
            action: Description of the action to evaluate
            context: Additional context about the action (optional)
            
        Returns:
            Evaluation results with principle scores and recommendations
        """
        context = context or {}
        evaluation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "context": context,
            "principle_scores": {},
            "overall_score": 0.0,
            "violated_principles": [],
            "recommendations": []
        }
        
        # Evaluate against each principle
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for principle in self.principle_engine.principles:
            principle_id = principle["id"]
            score, violations, recommendations = self._evaluate_action_against_principle(action, principle, context)
            
            principle_result = {
                "score": score,
                "violations": violations,
                "recommendations": recommendations
            }
            
            evaluation["principle_scores"][principle_id] = principle_result
            
            # Update running weighted score
            weight = self.principle_engine.principle_weights[principle_id]
            total_weighted_score += score * weight
            total_weight += weight
            
            # Track violated principles
            if score < 70:
                evaluation["violated_principles"].append({
                    "id": principle_id,
                    "name": principle["name"],
                    "score": score,
                    "violations": violations
                })
                
                # Add recommendations to the overall list
                for rec in recommendations:
                    if rec not in evaluation["recommendations"]:
                        evaluation["recommendations"].append(rec)
        
        # Calculate overall score
        if total_weight > 0:
            evaluation["overall_score"] = total_weighted_score / total_weight
        
        # Add to history
        self.evaluation_history = [*self.evaluation_history, evaluation]
        
        # Log the evaluation
        logger.info(f"Action evaluated with overall score: {evaluation['overall_score']:.2f}")
        if evaluation["violated_principles"]:
            logger.info(f"Violated principles: {', '.join(p['name'] for p in evaluation['violated_principles'])}")
        
        return evaluation
    
    def _evaluate_action_against_principle(
        self, action: str, principle: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Evaluate an action against a specific principle.
        
        Args:
            action: Description of the action
            principle: The principle definition
            context: Additional context
            
        Returns:
            Tuple of (score, violations, recommendations)
        """
        # Default perfect score
        score = 100.0
        violations = []
        recommendations = []
        
        # Extract key attributes for evaluation
        principle_id = principle["id"]
        principle_name = principle["name"]
        principle_description = principle["description"]
        evaluation_criteria = principle.get("evaluation_criteria", [])
        
        # Analyze action against this principle
        # This is where you would implement the principle-specific evaluation logic
        # For now, we use a simple keyword-based approach
        action_lower = action.lower()
        
        # Convert principle name and description to lowercase for case-insensitive matching
        name_lower = principle_name.lower()
        desc_lower = principle_description.lower()
        
        # Extract keywords from principle
        keywords = name_lower.split() + desc_lower.split()
        keywords = [k for k in keywords if len(k) > 3]  # Only use significant words
        
        # Look for contradictory language in the action description
        contradictions = ["against", "violate", "ignore", "bypass", "override", "without regard", "disregard"]
        
        for keyword in keywords:
            for contradiction in contradictions:
                if f"{contradiction} {keyword}" in action_lower:
                    # Found explicit contradiction of this principle
                    score -= 50
                    violation = f"Action explicitly contradicts the {principle_name} principle by {contradiction} {keyword}"
                    violations.append(violation)
                    recommendations.append(f"Avoid {contradiction} {keyword}; instead, respect {principle_name}")
                    break
        
        # Universal check: Look at context for explicit overrides
        if context.get("bypass_principles") or context.get(f"bypass_{principle_id}"):
            score -= 70
            violation = f"Context explicitly seeks to bypass the {principle_name} principle"
            violations.append(violation)
            recommendations.append("Remove explicit principle bypass from context")
        
        # Balance score with overall action intent
        # Simple approach: check if action is clearly about upholding this principle
        affirmations = ["uphold", "maintain", "respect", "honor", "consider", "align with", "following", "according to"]
        for affirmation in affirmations:
            if f"{affirmation} {name_lower}" in action_lower:
                # Action explicitly mentions respecting this principle
                score = min(100, score + 20)
                break
        
        # Principle-specific evaluations
        # Here we would add custom logic for each principle type
        # This would be extended based on the specific principles provided
        
        # Handle generic keywords/phrases that might indicate principle violations
        known_violations = {
            # Add patterns specific to different principles
            # Format: "principle_id": [(negative_pattern, score_reduction, violation_message, recommendation)]
            # Generic examples:
            "any": [
                ("without consent", 40, "Action proceeds without proper consent", "Obtain proper consent before proceeding"),
                ("without permission", 40, "Action proceeds without proper permission", "Obtain proper permission before proceeding"),
                ("mandatory", 20, "Action imposes mandatory requirements", "Consider making requirements optional or providing alternatives"),
                ("secretly", 50, "Action involves secrecy", "Operate transparently instead of secretly"),
                ("track without", 50, "Action involves tracking without proper disclosure", "Ensure proper disclosure of any tracking"),
                ("collect data without", 60, "Action collects data without proper authorization", "Obtain authorization before collecting data"),
                ("override", 30, "Action overrides established protections", "Work within established protections"),
                ("backdoor", 70, "Action implements backdoor access", "Avoid backdoor implementations"),
                ("bypass security", 70, "Action bypasses security measures", "Respect security measures"),
                ("ignore warning", 40, "Action ignores warnings", "Heed warnings instead of ignoring them"),
                ("mislead", 60, "Action may mislead", "Be truthful and transparent"),
                ("hide", 40, "Action conceals information", "Be open and transparent"),
            ]
        }
        
        # Check for known violations
        for pattern, reduction, message, recommendation in known_violations.get("any", []):
            if pattern in action_lower:
                score = max(0, score - reduction)
                violations.append(message)
                recommendations.append(recommendation)
        
        # Apply principle-specific patterns if available
        for pattern, reduction, message, recommendation in known_violations.get(principle_id, []):
            if pattern in action_lower:
                score = max(0, score - reduction)
                violations.append(message)
                recommendations.append(recommendation)
        
        # Ensure score stays in valid range
        score = max(0, min(100, score))
        
        return score, violations, recommendations
    
    def check_action_compliance(self, action: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if an action complies with all principles and provide detailed feedback.
        
        Args:
            action: Description of the action to check
            context: Additional context about the action (optional)
            
        Returns:
            Dictionary with compliance result and detailed feedback
        """
        evaluation = self.evaluate_action(action, context)
        
        result = {
            "complies": evaluation["overall_score"] >= 70,
            "overall_score": evaluation["overall_score"],
            "action": action,
            "violated_principles": evaluation["violated_principles"],
            "recommendations": evaluation["recommendations"],
            "explanation": self._generate_compliance_explanation(evaluation)
        }
        
        return result
    
    def _generate_compliance_explanation(self, evaluation: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of the compliance evaluation.
        
        Args:
            evaluation: The evaluation results
            
        Returns:
            A string explaining the compliance result
        """
        if evaluation["overall_score"] >= 90:
            explanation = "This action fully complies with all principles."
        elif evaluation["overall_score"] >= 70:
            explanation = "This action generally complies with principles, with minor concerns."
        elif evaluation["overall_score"] >= 50:
            explanation = "This action raises significant principle concerns that should be addressed."
        else:
            explanation = "This action violates multiple principles and should be reconsidered."
        
        # Add details about violated principles
        if evaluation["violated_principles"]:
            explanation += "\n\nPrinciple concerns:"
            for p in evaluation["violated_principles"]:
                explanation += f"\n- {p['name']} (score: {p['score']:.0f}/100):"
                for v in p['violations']:
                    explanation += f"\n  • {v}"
        
        # Add recommendations if available
        if evaluation["recommendations"]:
            explanation += "\n\nRecommendations:"
            for i, rec in enumerate(evaluation["recommendations"], 1):
                explanation += f"\n{i}. {rec}"
        
        return explanation
    
    def suggest_alternatives(self, action: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Suggest alternative approaches that better align with principles.
        
        Args:
            action: Description of the action
            context: Additional context (optional)
            
        Returns:
            List of alternative suggestions
        """
        evaluation = self.evaluate_action(action, context)
        
        # If action already complies well with principles, no need for alternatives
        if evaluation["overall_score"] >= 90:
            return ["The proposed action already aligns well with all principles."]
        
        alternatives = []
        
        # Generate alternatives based on violated principles
        for p in evaluation["violated_principles"]:
            principle_name = p["name"]
            
            # Get recommendations for this principle
            principle_recs = evaluation["principle_scores"][p["id"]]["recommendations"]
            
            if principle_recs:
                # Generate an alternative incorporating the recommendations
                positive_suggestion = self._transform_to_positive_suggestion(action, principle_recs, principle_name)
                if positive_suggestion:
                    alternatives.append(positive_suggestion)
        
        # Add a general alternative if specific ones couldn't be generated
        if not alternatives:
            alternatives.append(
                "Consider an approach that explicitly addresses the concerns raised in the evaluation."
            )
        
        return alternatives
    
    def _transform_to_positive_suggestion(
        self, action: str, recommendations: List[str], principle_name: str
    ) -> Optional[str]:
        """
        Transform recommendations into a positive alternative suggestion.
        
        Args:
            action: Original action
            recommendations: List of recommendations
            principle_name: Name of the principle
            
        Returns:
            A positive suggestion or None
        """
        if not recommendations:
            return None
        
        # Start with a template
        suggestion = f"Instead of '{action}', consider an approach that respects {principle_name} by "
        
        # Convert recommendations into positive actions
        positive_actions = []
        for rec in recommendations:
            # Extract the positive part after words like "avoid", "instead", etc.
            parts = rec.split(", instead")
            if len(parts) > 1:
                positive_actions.append(parts[1].strip())
                continue
                
            parts = rec.split("avoid ")
            if len(parts) > 1:
                inverse = parts[1]
                positive_actions.append(f"not {inverse}")
                continue
                
            # If no clear transformation, use the recommendation as is
            positive_actions.append(rec)
        
        if positive_actions:
            suggestion += "; ".join(positive_actions).lower()
            suggestion += "."
            return suggestion
        
        return None
    
    def generate_explanation(self, compliance_result: Dict[str, Any]) -> str:
        """
        Generate a detailed explanation of compliance issues for user communication.
        
        Args:
            compliance_result: Result from check_action_compliance
            
        Returns:
            A formatted explanation string
        """
        explanation = []
        
        # Add a header based on compliance level
        if compliance_result["complies"]:
            explanation.append("I can proceed with this action while respecting all principles.")
        else:
            explanation.append("I cannot perform this action as requested because it conflicts with my principles.")
        
        # Include the explanation from the compliance result
        explanation.append("")
        explanation.append(compliance_result["explanation"])
        
        # Add negotiation text for non-compliant actions
        if not compliance_result["complies"]:
            explanation.append("")
            explanation.append("Here are some alternative approaches I could take:")
            alternatives = self.suggest_alternatives(compliance_result["action"])
            for i, alt in enumerate(alternatives, 1):
                explanation.append(f"{i}. {alt}")
        
        return "\n".join(explanation)
    
    def get_principles_summary(self) -> str:
        """
        Get a human-readable summary of all principles.
        
        Returns:
            String containing principle descriptions
        """
        summary = []
        summary.append("# Guiding Principles")
        summary.append("")
        
        for i, principle in enumerate(self.principle_engine.principles, 1):
            summary.append(f"## {i}. {principle['name']}")
            summary.append("")
            summary.append(principle['description'])
            summary.append("")
            
            if 'example' in principle and principle['example']:
                summary.append(f"*Example:* {principle['example']}")
                summary.append("")
                
        return "\n".join(summary)



    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
# Example usage
if __name__ == "__main__":
    # Create a PrincipleActionEvaluator with default principles
    evaluator = PrincipleActionEvaluator()
    
    # Example actions to evaluate
    test_actions = [
        "Send an email to all users informing them of a security update",
        "Collect user browsing data without their knowledge for targeted advertising",
        "Delete a user's account immediately upon their request",
        "Override security protocols to expedite a critical deployment",
        "Add a feature that tracks user location in the background",
    ]
    
    # Evaluate each action
    for action in test_actions:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {action}")
        print(f"{'-' * 80}")
        
        result = evaluator.check_action_compliance(action)
        explanation = evaluator.generate_explanation(result)
        
        print(f"Compliance Score: {result['overall_score']:.1f}/100")
        print(f"Complies with principles: {result['complies']}")
        print("\nExplanation:")
        print(textwrap.fill(explanation, width=80))
        
        if not result['complies']:
            print("\nSuggested Alternatives:")
            alternatives = evaluator.suggest_alternatives(action)
            for i, alt in enumerate(alternatives, 1):
                print(f"{i}. {textwrap.fill(alt, width=76, initial_indent='   ', subsequent_indent='      ')}")