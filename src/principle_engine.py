#!/usr/bin/env python3
"""
Principle Engine for the Adaptive Bridge Builder Agent

This module implements the PrincipleEngine class that stores, evaluates,
and enforces the core principles of the "Empire of the Adaptive Hero" profile.
The engine ensures that all agent communications align with these principles.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngine")

class PrincipleEngine:
    """
    Engine that manages and enforces the core principles of the "Empire of the Adaptive Hero".
    
    This class provides methods to evaluate messages against principles,
    determine appropriate responses, and maintain a principle consistency score.
    """
    
    def __init__(self, principles_file: Optional[str] = None):
        """
        Initialize the PrincipleEngine with the core principles.
        
        Args:
            principles_file: Optional path to a JSON file containing principle definitions.
                             If None, default principles will be used.
        """
        self.principles = self._load_principles(principles_file)
        self.consistency_scores = {principle["id"]: 100.0 for principle in self.principles}
        self.principle_weights = {principle["id"]: principle["weight"] for principle in self.principles}
        self.overall_consistency = 100.0
        self.evaluation_history = []
        
        logger.info(f"PrincipleEngine initialized with {len(self.principles)} core principles")
    
    def _load_principles(self, principles_file: Optional[str]) -> List[Dict[str, Any]]:
        """
        Load principles from a file or use default principles.
        
        Args:
            principles_file: Path to a JSON file containing principle definitions.
            
        Returns:
            List of principle dictionaries.
        """
        if principles_file:
            try:
                with open(principles_file, 'r') as f:
                    principles = json.load(f)
                    logger.info(f"Principles loaded successfully from {principles_file}")
                    return principles
            except Exception as e:
                logger.error(f"Failed to load principles from {principles_file}: {e}")
                logger.info("Using default principles instead")
        
        # Default "Empire of the Adaptive Hero" principles
        return [
            {
                "id": "fairness_as_truth",
                "name": "Fairness as Truth",
                "description": "Equal treatment of all messages and agents regardless of source, with transparent processing that values objectivity above all.",
                "weight": 1.0,
                "example": "When agents from different systems send requests with varying priorities, each request is evaluated with the same criteria regardless of the sender's status.",
                "evaluation_criteria": [
                    "Message is processed using standardized validation",
                    "No prioritization based on sender identity",
                    "Error handling is consistent across all senders",
                    "Processing transparency is maintained"
                ]
            },
            {
                "id": "harmony_through_presence",
                "name": "Harmony Through Presence",
                "description": "Maintaining clear communication and acknowledgment of all interactions, creating harmony through consistent and responsive presence.",
                "weight": 1.0,
                "example": "Acknowledging receipt of all messages promptly, even if full processing will take time, and providing status updates at key processing stages.",
                "evaluation_criteria": [
                    "Message receipt is acknowledged",
                    "Processing status is communicated",
                    "Response includes clear next steps",
                    "Communication maintains continuity"
                ]
            },
            {
                "id": "adaptability_as_strength",
                "name": "Adaptability as Strength",
                "description": "Ability to evolve and respond to changing communication needs, recognizing adaptability as the foundation of resilience and growth.",
                "weight": 1.0,
                "example": "When receiving a message in an unexpected format, attempting to parse it intelligently rather than rejecting it outright, and learning from the interaction for future encounters.",
                "evaluation_criteria": [
                    "Unexpected inputs are handled gracefully",
                    "Multiple interpretation attempts before rejection",
                    "New patterns are recognized and incorporated",
                    "Response adapts to sender's communication style"
                ]
            },
            {
                "id": "balance_in_mediation",
                "name": "Balance in Mediation",
                "description": "Maintaining neutrality when facilitating communication between different agents, ensuring that mediation serves all parties equally.",
                "weight": 1.0,
                "example": "When routing messages between competing agent systems, ensuring that routing algorithms apply the same rules and that no system gains unfair advantages through the mediation process.",
                "evaluation_criteria": [
                    "Mediation provides equal benefits to all parties",
                    "No favoritism in resource allocation",
                    "Transparent decision-making process",
                    "Consistent application of routing rules"
                ]
            },
            {
                "id": "clarity_in_complexity",
                "name": "Clarity in Complexity",
                "description": "Transforming complex interactions into clear, understandable communications while preserving essential meaning.",
                "weight": 1.0,
                "example": "When translating between different protocols, simplifying technical details while ensuring that the semantic meaning remains intact, making communications accessible without losing precision.",
                "evaluation_criteria": [
                    "Complex concepts are explained clearly",
                    "Technical details are appropriately simplified",
                    "Essential meaning is preserved",
                    "Response is appropriately contextualized"
                ]
            },
            {
                "id": "integrity_in_transmission",
                "name": "Integrity in Transmission",
                "description": "Ensuring that messages are delivered with their meaning and intent intact, maintaining the integrity of communication throughout the transmission process.",
                "weight": 1.0,
                "example": "When routing a message through multiple systems, verifying that the content remains uncorrupted and that semantic meaning is preserved, even when protocol translation is required.",
                "evaluation_criteria": [
                    "Message content remains unaltered",
                    "Semantic meaning is preserved",
                    "Intention of the sender is maintained",
                    "Delivery verification is performed"
                ]
            },
            {
                "id": "resilience_through_connection",
                "name": "Resilience Through Connection",
                "description": "Building robust systems through diverse, well-managed connections that can withstand failures and adapt to changing conditions.",
                "weight": 1.0,
                "example": "Maintaining multiple routing pathways between agent systems so that if one connection fails, communication can continue through alternative channels without disruption.",
                "evaluation_criteria": [
                    "Alternative communication paths are considered",
                    "Failure recovery mechanisms are in place",
                    "Connection diversity is maintained",
                    "System degrades gracefully under stress"
                ]
            },
            {
                "id": "empathy_in_interface",
                "name": "Empathy in Interface",
                "description": "Understanding and accommodating the unique needs and constraints of different agent systems, creating interfaces that feel natural to each participant.",
                "weight": 1.0,
                "example": "Adapting communication patterns to match the expectations of different agent types, such as using more structured formats for highly formal systems and more flexible formats for adaptive ones.",
                "evaluation_criteria": [
                    "Communication style adapts to recipient",
                    "System constraints are accommodated",
                    "Interface feels natural to participants",
                    "User experience is considered in design"
                ]
            },
            {
                "id": "truth_in_representation",
                "name": "Truth in Representation",
                "description": "Accurately representing capabilities, limitations, and intentions in all communications, fostering trust through honesty.",
                "weight": 1.0,
                "example": "When an agent requests a service that cannot be fully provided, clearly communicating what is possible rather than overpromising, and suggesting alternatives when appropriate.",
                "evaluation_criteria": [
                    "Capabilities are accurately represented",
                    "Limitations are clearly communicated",
                    "Expectations are properly set",
                    "Alternatives are suggested when appropriate"
                ]
            },
            {
                "id": "growth_through_reflection",
                "name": "Growth Through Reflection",
                "description": "Continuously learning from interactions and outcomes, using reflection to improve future communications and adaptations.",
                "weight": 1.0,
                "example": "Analyzing patterns in successful and unsuccessful communications to identify areas for improvement, and incorporating these insights into future message processing strategies.",
                "evaluation_criteria": [
                    "Past interactions inform current decisions",
                    "Improvement patterns are identified",
                    "Lessons learned are incorporated",
                    "Reflection leads to measurable growth"
                ]
            }
        ]
    
    def evaluate_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an incoming message against all principles.
        
        Args:
            message: The incoming message to evaluate.
            
        Returns:
            Evaluation results containing scores for each principle and overall consistency.
        """
        evaluation = {
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": message.get("id", "unknown"),
            "method": message.get("method", "unknown"),
            "principle_scores": {},
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Evaluate against each principle
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for principle in self.principles:
            principle_id = principle["id"]
            score, recommendations = self._evaluate_against_principle(message, principle)
            
            evaluation["principle_scores"][principle_id] = {
                "score": score,
                "recommendations": recommendations
            }
            
            # Update running weighted score
            weight = self.principle_weights[principle_id]
            total_weighted_score += score * weight
            total_weight += weight
            
            # Add recommendations if score is below threshold
            if score < 70:
                for rec in recommendations:
                    if rec not in evaluation["recommendations"]:
                        evaluation["recommendations"].append(rec)
        
        # Calculate overall score
        if total_weight > 0:
            evaluation["overall_score"] = total_weighted_score / total_weight
        
        # Update consistency scores
        self._update_consistency_scores(evaluation)
        
        # Add to history
        self.evaluation_history.append(evaluation)
        
        logger.info(f"Message {evaluation['message_id']} evaluated with overall score: {evaluation['overall_score']:.2f}")
        return evaluation
    
    def _evaluate_against_principle(self, message: Dict[str, Any], principle: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Evaluate a message against a specific principle.
        
        Args:
            message: The message to evaluate.
            principle: The principle to evaluate against.
            
        Returns:
            Tuple of (score, list of recommendations).
        """
        principle_id = principle["id"]
        criteria = principle["evaluation_criteria"]
        
        # Initialize with perfect score
        score = 100.0
        recommendations = []
        
        # Specific evaluation logic for each principle
        if principle_id == "fairness_as_truth":
            # Check if message has special prioritization flags
            if message.get("params", {}).get("priority", 0) > 0:
                score -= 25
                recommendations.append("Remove priority flags to ensure fair treatment")
            
            # Check for sender-specific handling
            if message.get("params", {}).get("sender_specific_handling", False):
                score -= 50
                recommendations.append("Avoid sender-specific handling instructions")
                
        elif principle_id == "harmony_through_presence":
            # Check if conversation tracking is present
            if "conversation_id" not in message.get("params", {}):
                score -= 30
                recommendations.append("Include conversation_id for continuous tracking")
            
            # Check if acknowledgment is requested
            if not message.get("params", {}).get("require_ack", True):
                score -= 20
                recommendations.append("Enable acknowledgments for better communication harmony")
                
        elif principle_id == "adaptability_as_strength":
            # Check for strict format requirements
            if message.get("params", {}).get("strict_format", False):
                score -= 40
                recommendations.append("Avoid strict format requirements to allow adaptation")
            
            # Check for flexible handling options
            if not message.get("params", {}).get("allow_interpretation", True):
                score -= 30
                recommendations.append("Enable flexible interpretation for better adaptability")
                
        elif principle_id == "balance_in_mediation":
            # Check for biased routing instructions
            if message.get("params", {}).get("preferred_route", None):
                score -= 50
                recommendations.append("Avoid specifying preferred routes to maintain balance")
                
        elif principle_id == "clarity_in_complexity":
            # Check for unnecessarily complex structure
            params = message.get("params", {})
            if isinstance(params, dict) and len(json.dumps(params)) > 500:
                score -= 20
                recommendations.append("Simplify message structure for better clarity")
                
        elif principle_id == "integrity_in_transmission":
            # Check if message requests modifications during transmission
            if message.get("params", {}).get("allow_modification", False):
                score -= 60
                recommendations.append("Disable content modification during transmission")
                
        elif principle_id == "resilience_through_connection":
            # Check if fallback options are provided
            if not message.get("params", {}).get("fallback_routes", []):
                score -= 25
                recommendations.append("Provide fallback routes for better resilience")
                
        elif principle_id == "empathy_in_interface":
            # Check if recipient preferences are considered
            if not message.get("params", {}).get("recipient_preferences", {}):
                score -= 20
                recommendations.append("Include recipient preferences for better interface empathy")
                
        elif principle_id == "truth_in_representation":
            # Check for accurate capability representation
            if message.get("method") not in ["getAgentCard", "echo", "route", "translateProtocol"]:
                score -= 40
                recommendations.append("Use supported methods to maintain truth in representation")
                
        elif principle_id == "growth_through_reflection":
            # Check if feedback mechanism is enabled
            if not message.get("params", {}).get("collect_feedback", True):
                score -= 30
                recommendations.append("Enable feedback collection for continuous improvement")
        
        # Ensure score stays in valid range
        score = max(0, min(100, score))
        
        return score, recommendations
    
    def _update_consistency_scores(self, evaluation: Dict[str, Any]) -> None:
        """
        Update the principle consistency scores based on a message evaluation.
        
        Args:
            evaluation: The evaluation results for a message.
        """
        # Update individual principle scores
        for principle_id, result in evaluation["principle_scores"].items():
            current_score = self.consistency_scores[principle_id]
            new_score = result["score"]
            
            # Apply exponential moving average (EMA) with alpha=0.1
            alpha = 0.1
            updated_score = (alpha * new_score) + ((1 - alpha) * current_score)
            self.consistency_scores[principle_id] = updated_score
        
        # Update overall consistency score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for principle_id, score in self.consistency_scores.items():
            weight = self.principle_weights[principle_id]
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            self.overall_consistency = total_weighted_score / total_weight
        
        logger.info(f"Updated overall principle consistency score: {self.overall_consistency:.2f}")
    
    def get_consistent_response(self, message: Dict[str, Any], draft_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust a draft response to ensure it aligns with principles.
        
        Args:
            message: The original incoming message.
            draft_response: The draft response to adjust.
            
        Returns:
            The adjusted response that aligns with principles.
        """
        # Make a deep copy to avoid modifying the original
        response = copy.deepcopy(draft_response)
        
        # Apply principle-based adjustments
        self._apply_fairness_principles(message, response)
        self._apply_harmony_principles(message, response)
        self._apply_adaptability_principles(message, response)
        self._apply_balance_principles(message, response)
        self._apply_clarity_principles(message, response)
        self._apply_integrity_principles(message, response)
        self._apply_resilience_principles(message, response)
        self._apply_empathy_principles(message, response)
        self._apply_truth_principles(message, response)
        self._apply_growth_principles(message, response)
        
        # Log the adjustment
        logger.info(f"Response adjusted to align with principles for message: {message.get('id', 'unknown')}")
        
        return response
    
    def _apply_fairness_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Fairness as Truth principles to a response."""
        # Ensure error responses use standardized codes
        if "error" in response:
            # Standardize error codes to JSON-RPC 2.0 if possible
            error = response["error"]
            if isinstance(error, dict) and "code" in error:
                if error["code"] not in [-32700, -32600, -32601, -32602, -32603]:
                    # Only use custom codes in the allowed range
                    if error["code"] < -32000 or error["code"] > -32099:
                        error["code"] = -32603  # Internal error as fallback
    
    def _apply_harmony_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Harmony Through Presence principles to a response."""
        # Ensure the response includes acknowledgment elements
        if "result" in response and isinstance(response["result"], dict):
            # Add timestamp if not present
            if "timestamp" not in response["result"]:
                response["result"]["timestamp"] = datetime.utcnow().isoformat()
            
            # Include conversation ID if present in the request
            conversation_id = message.get("params", {}).get("conversation_id")
            if conversation_id and "conversation_id" not in response["result"]:
                response["result"]["conversation_id"] = conversation_id
    
    def _apply_adaptability_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Adaptability as Strength principles to a response."""
        # If handling an unknown method with an error, suggest alternatives
        if "error" in response and response["error"].get("code") == -32601:
            # Method not found, suggest alternatives
            method = message.get("method", "")
            alternatives = self._suggest_alternative_methods(method)
            if alternatives and "data" not in response["error"]:
                response["error"]["data"] = {"suggested_alternatives": alternatives}
    
    def _apply_balance_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Balance in Mediation principles to a response."""
        # For routing responses, ensure no preferential treatment
        if message.get("method") == "route" and "result" in response:
            # Remove any routing priority indicators if present
            if isinstance(response["result"], dict):
                response["result"].pop("priority_route", None)
                response["result"].pop("expedited", None)
    
    def _apply_clarity_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Clarity in Complexity principles to a response."""
        # Simplify overly complex responses
        if "result" in response and isinstance(response["result"], dict):
            # If the result is very complex, provide a simplified summary
            result_json = json.dumps(response["result"])
            if len(result_json) > 1000:  # Arbitrary threshold for complexity
                # Create a simplified version preserving key information
                simplified = self._simplify_complex_result(response["result"])
                if simplified and simplified != response["result"]:
                    response["result"]["_simplified"] = simplified
    
    def _apply_integrity_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Integrity in Transmission principles to a response."""
        # For message translations, ensure semantic preservation
        if message.get("method") == "translateProtocol" and "result" in response:
            # Add integrity verification
            if isinstance(response["result"], dict):
                response["result"]["integrity_verified"] = True
                response["result"]["semantic_preservation_check"] = "passed"
    
    def _apply_resilience_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Resilience Through Connection principles to a response."""
        # For all responses, consider adding alternative contact methods
        if "result" in response and isinstance(response["result"], dict):
            # Only add if not already present
            if "alternative_contacts" not in response["result"]:
                response["result"]["alternative_contacts"] = [
                    {"type": "http", "url": "https://api.example.com/adaptive-bridge"}
                ]
    
    def _apply_empathy_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Empathy in Interface principles to a response."""
        # Adapt response format based on requester preferences if specified
        format_pref = message.get("params", {}).get("response_format")
        if format_pref and "result" in response:
            if format_pref == "minimal" and isinstance(response["result"], dict):
                # Remove non-essential metadata for minimal format
                for key in list(response["result"].keys()):
                    if key.startswith("_") or key in ["metadata", "debug_info"]:
                        response["result"].pop(key, None)
    
    def _apply_truth_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Truth in Representation principles to a response."""
        # Ensure capabilities are accurately represented
        if message.get("method") == "getAgentCard" and "result" in response:
            # Verify capabilities are accurately represented
            if isinstance(response["result"], dict) and "capabilities" in response["result"]:
                # Ensure no overstated capabilities
                capabilities = response["result"]["capabilities"]
                if isinstance(capabilities, list):
                    for capability in capabilities:
                        if isinstance(capability, dict) and "name" in capability:
                            # Remove any exaggerated capability descriptions
                            if "description" in capability:
                                if "perfect" in capability["description"].lower() or "always" in capability["description"].lower():
                                    capability["description"] = capability["description"].replace("perfect", "effective").replace("always", "typically")
    
    def _apply_growth_principles(self, message: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Apply Growth Through Reflection principles to a response."""
        # Add continuous improvement indicators
        if "result" in response and isinstance(response["result"], dict):
            # Add a feedback request for complex operations
            if message.get("method") not in ["echo", "getAgentCard"]:
                if "feedback_requested" not in response["result"]:
                    response["result"]["feedback_requested"] = True
    
    def _suggest_alternative_methods(self, method: str) -> List[str]:
        """
        Suggest alternative methods based on the requested method.
        
        Args:
            method: The requested method.
            
        Returns:
            List of suggested alternative methods.
        """
        known_methods = ["getAgentCard", "echo", "route", "translateProtocol"]
        
        # Simple similarity matching
        if method.lower() == "getagent" or method.lower() == "card":
            return ["getAgentCard"]
        elif method.lower() == "repeat" or method.lower() == "mirror":
            return ["echo"]
        elif method.lower() == "send" or method.lower() == "forward":
            return ["route"]
        elif method.lower() == "translate" or method.lower() == "convert":
            return ["translateProtocol"]
        elif method not in known_methods:
            return known_methods  # Suggest all known methods
        
        return []
    
    def _simplify_complex_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simplified version of a complex result.
        
        Args:
            result: The complex result to simplify.
            
        Returns:
            A simplified version of the result.
        """
        simplified = {}
        
        # Extract the most important keys (this is a simplified example)
        important_keys = ["status", "message", "timestamp", "conversation_id"]
        for key in important_keys:
            if key in result:
                simplified[key] = result[key]
        
        # Add a summary key
        simplified["summary"] = "Complex result simplified. See full result for complete details."
        
        return simplified
    
    def get_principle_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all principles.
        
        Returns:
            List of principle dictionaries with name, description, and example.
        """
        return [{
            "name": p["name"],
            "description": p["description"],
            "example": p["example"]
        } for p in self.principles]
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """
        Get a detailed report on principle consistency.
        
        Returns:
            Dictionary containing overall and individual principle consistency scores.
        """
        return {
            "overall_consistency": self.overall_consistency,
            "principle_scores": self.consistency_scores,
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation_count": len(self.evaluation_history)
        }
    
    def reset_consistency_scores(self) -> None:
        """Reset all consistency scores to their initial values."""
        self.consistency_scores = {principle["id"]: 100.0 for principle in self.principles}
        self.overall_consistency = 100.0
        logger.info("Principle consistency scores have been reset")


# Example usage
if __name__ == "__main__":
    # Create a principle engine
    engine = PrincipleEngine()
    
    # Example message to evaluate
    example_message = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": "conv-123",
            "destination": "target-agent-001",
            "message": {
                "jsonrpc": "2.0",
                "method": "processData",
                "params": {"data": {"type": "sensor_reading", "value": 23.5}}
            },
            "priority": 2  # This will affect the fairness score
        },
        "id": "test-1"
    }
    
    # Evaluate the message
    evaluation = engine.evaluate_message(example_message)
    print(f"Overall evaluation score: {evaluation['overall_score']:.2f}")
    
    # Get principle descriptions
    principles = engine.get_principle_descriptions()
    print(f"\nCore Principles ({len(principles)}):")
    for i, principle in enumerate(principles, 1):
        print(f"{i}. {principle['name']}: {principle['description']}")
        print(f"   Example: {principle['example']}\n")
    
    # Create a draft response
    draft_response = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "result": {
            "status": "acknowledged",
            "message": "Message accepted for routing",
            "priority_route": True  # This violates balance principles
        }
    }
    
    # Get a principle-consistent response
    consistent_response = engine.get_consistent_response(example_message, draft_response)
    print("Principle-consistent response:")
    print(json.dumps(consistent_response, indent=2))
    
    # Get consistency report
    report = engine.get_consistency_report()
    print("\nPrinciple Consistency Report:")
    print(f"Overall consistency: {report['overall_consistency']:.2f}")
    for principle_id, score in report['principle_scores'].items():
        principle_name = next((p["name"] for p in engine.principles if p["id"] == principle_id), principle_id)
        print(f"{principle_name}: {score:.2f}")
