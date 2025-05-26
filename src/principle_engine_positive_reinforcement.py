#!/usr/bin/env python3
"""
Positive Reinforcement extension for the Principle Engine

This module implements the prioritize_positive_reinforcement function that analyzes
interactions for opportunities to steer communication toward positive outcomes,
aligning with the 'Love as a Generative Force' principle.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PositiveReinforcement")

class PositiveReinforcementElement:
    """Represents an element of positive reinforcement identified in an interaction."""
    
    def __init__(self, 
                 element_type: str,
                 content: str,
                 confidence: float,
                 context: Optional[str] = None):
        """
        Initialize a positive reinforcement element.
        
        Args:
            element_type: Type of positive element (e.g., 'appreciation', 'agreement', 'opportunity')
            content: The specific content that constitutes the positive element
            confidence: Confidence score for this element (0.0-1.0)
            context: Optional context around this element
        """
        self.element_type = element_type
        self.content = content
        self.confidence = confidence
        self.context = context
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the element to a dictionary representation."""
        return {
            "element_type": self.element_type,
            "content": self.content,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp
        }

def prioritize_positive_reinforcement(
    interaction_data: Dict[str, Any], 
    agent_id: str,
    principle_engine=None,
    emotional_intelligence=None,
    learning_system=None
) -> Dict[str, Any]:
    """
    Analyze interaction data to identify opportunities for positive reinforcement.
    
    This function embodies the 'Love as a Generative Force' principle by identifying
    opportunities to steer interactions toward positive, constructive outcomes.
    
    Args:
        interaction_data: Dictionary containing message content, sender info, and
                          context from the SessionManager
        agent_id: The ID of the agent for which to prioritize positive reinforcement
        principle_engine: Optional reference to the PrincipleEngine instance
        emotional_intelligence: Optional reference to the EmotionalIntelligence instance
        learning_system: Optional reference to the LearningSystem instance
        
    Returns:
        Dictionary containing:
        - generative_potential_score: Float from -1.0 to 1.0 indicating potential for positive steering
        - suggested_modifications: Optional list of suggested response modifications
        - identified_positive_elements: Optional list of positive elements identified in the input
    """
    # Initialize result structure
    result = {
        "generative_potential_score": 0.0,
        "suggested_modifications": [],
        "identified_positive_elements": [],
        "log": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "analysis_steps": []
        }
    }
    
    # Extract relevant information from interaction_data
    message_content = interaction_data.get("message", {}).get("content", "")
    sender_info = interaction_data.get("sender", {})
    history_summary = interaction_data.get("history_summary", {})
    
    # Check if we have the necessary information to proceed
    if not message_content:
        result["log"]["analysis_steps"].append({
            "step": "input_validation",
            "status": "failed",
            "reason": "Missing message content"
        })
        return result
    
    # Step 1: Assess emotional valence using EmotionalIntelligence
    emotion_assessment = _assess_emotional_valence(
        message_content, 
        emotional_intelligence
    )
    
    result["log"]["analysis_steps"].append({
        "step": "emotional_assessment",
        "status": "completed",
        "details": emotion_assessment
    })
    
    # Step 2: Identify positive elements in the interaction
    positive_elements = _identify_positive_elements(
        message_content,
        emotion_assessment,
        sender_info,
        history_summary
    )
    
    # Add identified positive elements to the result
    result["identified_positive_elements"] = [
        element.to_dict() for element in positive_elements
    ]
    
    result["log"]["analysis_steps"].append({
        "step": "positive_element_identification",
        "status": "completed",
        "count": len(positive_elements)
    })
    
    # Step 3: Calculate generative potential score
    generative_score = _calculate_generative_potential(
        emotion_assessment,
        positive_elements,
        history_summary
    )
    
    result["generative_potential_score"] = generative_score
    
    result["log"]["analysis_steps"].append({
        "step": "generative_potential_calculation",
        "status": "completed",
        "score": generative_score
    })
    
    # Step 4: Generate suggested modifications if needed
    # If negative emotions detected or generative score is low, suggest modifications
    if generative_score < 0.2 or emotion_assessment["overall_valence"] < 0:
        suggested_mods = _generate_positive_modifications(
            message_content,
            emotion_assessment,
            positive_elements,
            agent_id,
            principle_engine
        )
        
        result["suggested_modifications"] = suggested_mods
        
        result["log"]["analysis_steps"].append({
            "step": "modification_generation",
            "status": "completed",
            "count": len(suggested_mods)
        })
    
    # Step 5: Log decision and outcome for learning if learning_system is available
    if learning_system:
        _log_to_learning_system(
            learning_system,
            interaction_data,
            agent_id,
            result
        )
        
        result["log"]["analysis_steps"].append({
            "step": "learning_system_logging",
            "status": "completed"
        })
    
    # Final log message
    logger.info(
        f"Completed positive reinforcement analysis for agent {agent_id} "
        f"with generative potential score: {generative_score:.2f}"
    )
    
    return result

def _assess_emotional_valence(
    message_content: str, 
    emotional_intelligence=None
) -> Dict[str, Any]:
    """
    Assess the emotional valence of a message using EmotionalIntelligence if available,
    or with a simple heuristic approach if not.
    
    Args:
        message_content: The message text to analyze
        emotional_intelligence: Optional EmotionalIntelligence instance
        
    Returns:
        Dictionary with emotional assessment details
    """
    assessment = {
        "overall_valence": 0.0,  # -1.0 (negative) to 1.0 (positive)
        "primary_emotions": [],
        "emotion_intensities": {},
        "emotional_ambiguity": 0.0,  # 0.0 (clear) to 1.0 (ambiguous)
        "confidence": 0.5
    }
    
    # If EmotionalIntelligence is available, use it for accurate assessment
    if emotional_intelligence:
        emotion_signals, interaction_type, _ = emotional_intelligence.process_message(message_content)
        
        if emotion_signals:
            # Calculate overall valence based on emotion categories
            valence_map = {
                "JOY": 0.8,
                "TRUST": 0.6,
                "ANTICIPATION": 0.4,
                "SURPRISE": 0.2,
                "NEUTRAL": 0.0,
                "FEAR": -0.4,
                "SADNESS": -0.6,
                "DISGUST": -0.7,
                "ANGER": -0.8
            }
            
            # Weight by confidence and intensity
            weighted_valence = 0.0
            total_weight = 0.0
            
            for signal in emotion_signals:
                emotion_name = signal.category.name
                intensity_value = signal.intensity.value / 5.0  # Normalize to 0-1
                confidence = signal.confidence
                
                if emotion_name in valence_map:
                    valence = valence_map[emotion_name] * intensity_value
                    weight = confidence
                    
                    weighted_valence += valence * weight
                    total_weight += weight
                    
                    assessment["emotion_intensities"][emotion_name] = intensity_value
                    assessment["primary_emotions"].append(emotion_name)
            
            if total_weight > 0:
                assessment["overall_valence"] = weighted_valence / total_weight
                
            # Determine emotional ambiguity (multiple conflicting emotions)
            if len(emotion_signals) > 1:
                valences = [valence_map.get(s.category.name, 0) for s in emotion_signals]
                if any(v > 0 for v in valences) and any(v < 0 for v in valences):
                    assessment["emotional_ambiguity"] = 0.7
                    
            assessment["confidence"] = max([s.confidence for s in emotion_signals]) if emotion_signals else 0.5
            
        # Set interaction type
        assessment["interaction_type"] = interaction_type.name if interaction_type else "NEUTRAL"
        
    else:
        # Simple heuristic approach if EmotionalIntelligence is not available
        positive_terms = ["thank", "appreciate", "happy", "glad", "good", "great", "excellent", 
                         "wonderful", "excited", "love", "enjoy", "pleased", "grateful"]
        negative_terms = ["sorry", "regret", "unhappy", "bad", "terrible", "disappointed", 
                        "upset", "angry", "hate", "dislike", "unfortunately", "issue", "problem"]
        
        # Count occurrences of positive and negative terms
        positive_count = sum(1 for term in positive_terms if term.lower() in message_content.lower())
        negative_count = sum(1 for term in negative_terms if term.lower() in message_content.lower())
        
        # Determine overall valence
        total_terms = positive_count + negative_count
        if total_terms > 0:
            assessment["overall_valence"] = (positive_count - negative_count) / total_terms
            
            # Higher confidence if more emotional terms found
            assessment["confidence"] = min(0.7, 0.3 + (total_terms * 0.05))
        
        # Determine ambiguity
        if positive_count > 0 and negative_count > 0:
            assessment["emotional_ambiguity"] = min(1.0, (positive_count + negative_count) / 10)
        
        # Basic emotion determination
        if positive_count > negative_count:
            assessment["primary_emotions"] = ["JOY"] if positive_count > 2 else ["ANTICIPATION"]
        elif negative_count > positive_count:
            assessment["primary_emotions"] = ["ANGER"] if negative_count > 2 else ["SADNESS"]
        else:
            assessment["primary_emotions"] = ["NEUTRAL"]
            
        # Estimate intensities
        for emotion in assessment["primary_emotions"]:
            if emotion == "JOY" or emotion == "ANGER":
                assessment["emotion_intensities"][emotion] = min(1.0, positive_count / 5.0) if emotion == "JOY" else min(1.0, negative_count / 5.0)
            else:
                assessment["emotion_intensities"][emotion] = 0.5
                
        # Guess interaction type based on content
        if "?" in message_content:
            assessment["interaction_type"] = "INQUIRY"
        elif any(term in message_content.lower() for term in ["urgent", "emergency", "immediately", "asap"]):
            assessment["interaction_type"] = "CRISIS"
        elif any(term in message_content.lower() for term in ["disagree", "don't think", "incorrect"]):
            assessment["interaction_type"] = "CONFLICT"
        else:
            assessment["interaction_type"] = "ROUTINE"
    
    return assessment

def _identify_positive_elements(
    message_content: str,
    emotion_assessment: Dict[str, Any],
    sender_info: Dict[str, Any],
    history_summary: Dict[str, Any]
) -> List[PositiveReinforcementElement]:
    """
    Identify positive elements that can be reinforced in the interaction.
    
    Args:
        message_content: The message text to analyze
        emotion_assessment: Emotional assessment data
        sender_info: Information about the message sender
        history_summary: Summary of interaction history
        
    Returns:
        List of PositiveReinforcementElement objects
    """
    positive_elements = []
    
    # Check for expressions of appreciation
    appreciation_phrases = [
        "thank you", "thanks", "appreciate", "grateful", "thankful"
    ]
    
    for phrase in appreciation_phrases:
        if phrase in message_content.lower():
            # Extract context around the appreciation (simple approach)
            start_idx = max(0, message_content.lower().find(phrase) - 30)
            end_idx = min(len(message_content), message_content.lower().find(phrase) + len(phrase) + 30)
            context = message_content[start_idx:end_idx]
            
            positive_elements.append(
                PositiveReinforcementElement(
                    element_type="appreciation",
                    content=phrase,
                    confidence=0.8,
                    context=context
                )
            )
    
    # Check for agreement expressions
    agreement_phrases = [
        "agree", "good point", "exactly", "well said", "makes sense", "you're right"
    ]
    
    for phrase in agreement_phrases:
        if phrase in message_content.lower():
            start_idx = max(0, message_content.lower().find(phrase) - 30)
            end_idx = min(len(message_content), message_content.lower().find(phrase) + len(phrase) + 30)
            context = message_content[start_idx:end_idx]
            
            positive_elements.append(
                PositiveReinforcementElement(
                    element_type="agreement",
                    content=phrase,
                    confidence=0.7,
                    context=context
                )
            )
    
    # Check for positive forward-looking statements
    future_phrases = [
        "look forward", "excited about", "future", "next steps", "plan", "goal"
    ]
    
    for phrase in future_phrases:
        if phrase in message_content.lower():
            start_idx = max(0, message_content.lower().find(phrase) - 30)
            end_idx = min(len(message_content), message_content.lower().find(phrase) + len(phrase) + 30)
            context = message_content[start_idx:end_idx]
            
            positive_elements.append(
                PositiveReinforcementElement(
                    element_type="forward_looking",
                    content=phrase,
                    confidence=0.6,
                    context=context
                )
            )
    
    # Check for opportunities to highlight shared values
    # This would be more accurate with a more sophisticated analysis
    if "value" in message_content.lower() or "principle" in message_content.lower() or "believe" in message_content.lower():
        positive_elements.append(
            PositiveReinforcementElement(
                element_type="shared_values",
                content="Expression of values",
                confidence=0.5,
                context=message_content[:100]  # Simplified context
            )
        )
    
    # Check for constructive feedback
    feedback_phrases = ["feedback", "suggestion", "recommend", "improve", "enhance"]
    
    if any(phrase in message_content.lower() for phrase in feedback_phrases):
        # Only count as positive if overall tone is not strongly negative
        if emotion_assessment["overall_valence"] > -0.5:
            positive_elements.append(
                PositiveReinforcementElement(
                    element_type="constructive_feedback",
                    content="Constructive feedback",
                    confidence=0.6,
                    context=message_content[:100]
                )
            )
    
    # Look for collaboration opportunities
    collab_phrases = ["together", "collaborate", "partnership", "join", "mutual", "our"]
    
    if any(phrase in message_content.lower() for phrase in collab_phrases):
        positive_elements.append(
            PositiveReinforcementElement(
                element_type="collaboration",
                content="Collaboration opportunity",
                confidence=0.7,
                context=message_content[:100]
            )
        )
    
    return positive_elements

def _calculate_generative_potential(
    emotion_assessment: Dict[str, Any],
    positive_elements: List[PositiveReinforcementElement],
    history_summary: Dict[str, Any]
) -> float:
    """
    Calculate a score indicating the potential for positive reinforcement.
    
    Args:
        emotion_assessment: Emotional assessment data
        positive_elements: List of identified positive elements
        history_summary: Summary of interaction history
        
    Returns:
        Float from -1.0 to 1.0 indicating generative potential
    """
    # Start with baseline based on emotional valence
    # Scale from -1..1 to 0..0.6 (leaving room for other factors)
    base_score = (emotion_assessment["overall_valence"] + 1) * 0.3
    
    # Add value for each positive element (up to a maximum of 0.3)
    element_score = min(0.3, len(positive_elements) * 0.05)
    
    # Additional score for high-confidence positive elements
    high_confidence_elements = [e for e in positive_elements if e.confidence > 0.7]
    high_confidence_score = min(0.1, len(high_confidence_elements) * 0.03)
    
    # Incorporate interaction type from emotional assessment
    interaction_type_score = 0.0
    if "interaction_type" in emotion_assessment:
        interaction_type = emotion_assessment["interaction_type"]
        if interaction_type in ["ROUTINE", "CELEBRATION", "INQUIRY"]:
            interaction_type_score = 0.1
        elif interaction_type == "CONFLICT":
            interaction_type_score = -0.1
        elif interaction_type == "CRISIS":
            interaction_type_score = -0.05
    
    # Consider history if available (focusing on trend)
    history_score = 0.0
    if history_summary and "sentiment_trend" in history_summary:
        trend = history_summary.get("sentiment_trend", 0)
        # If trend is positive, add a small bonus
        if trend > 0:
            history_score = min(0.1, trend * 0.05)
        # If trend is negative, there's more opportunity for positive steering
        elif trend < 0:
            history_score = min(0.2, abs(trend) * 0.1)
    
    # Calculate total score, ensuring it stays in range -1.0 to 1.0
    total_score = base_score + element_score + high_confidence_score + interaction_type_score + history_score
    
    # Apply confidence factor from emotional assessment
    confidence = emotion_assessment.get("confidence", 0.5)
    
    # Weighted score based on confidence
    final_score = (total_score * confidence) + (0.0 * (1 - confidence))
    
    # Ensure result is within valid range
    return max(-1.0, min(1.0, final_score))

def _generate_positive_modifications(
    message_content: str,
    emotion_assessment: Dict[str, Any],
    positive_elements: List[PositiveReinforcementElement],
    agent_id: str,
    principle_engine=None
) -> List[Dict[str, Any]]:
    """
    Generate suggested modifications to steer communication toward positive outcomes.
    
    Args:
        message_content: The message content to analyze
        emotion_assessment: Emotional assessment data
        positive_elements: List of identified positive elements
        agent_id: ID of the agent
        principle_engine: Optional reference to the PrincipleEngine
        
    Returns:
        List of suggested modifications
    """
    modifications = []
    
    # Determine primary emotion and valence
    overall_valence = emotion_assessment["overall_valence"]
    primary_emotions = emotion_assessment.get("primary_emotions", ["NEUTRAL"])
    
    # Get interaction type (default to ROUTINE if not available)
    interaction_type = emotion_assessment.get("interaction_type", "ROUTINE")
    
    # 1. If negative emotions detected, suggest reframing
    if overall_valence < 0:
        # Different strategies based on the specific negative emotion
        if "ANGER" in primary_emotions:
            modifications.append({
                "type": "reframing",
                "original_fragment": "negative expression",
                "suggestion": "Acknowledge the concern while focusing on resolution: 'I understand this is frustrating. Let's focus on how we can resolve this together.'",
                "rationale": "Transforms anger into collaborative problem-solving",
                "confidence": 0.7
            })
        elif "SADNESS" in primary_emotions:
            modifications.append({
                "type": "reframing",
                "original_fragment": "expression of disappointment",
                "suggestion": "Acknowledge the feeling while offering hope: 'I understand this is disappointing. Let's look at what possibilities remain open to us.'",
                "rationale": "Balances acknowledgment with forward movement",
                "confidence": 0.7
            })
        elif "FEAR" in primary_emotions:
            modifications.append({
                "type": "reframing",
                "original_fragment": "expression of concern",
                "suggestion": "Validate the concern while providing reassurance: 'Your concerns are valid. Here's what we can do to address them...'",
                "rationale": "Validates feelings while building security",
                "confidence": 0.7
            })
    
    # 2. For conflict situations, suggest finding common ground
    if interaction_type == "CONFLICT":
        modifications.append({
            "type": "common_ground",
            "original_fragment": "area of disagreement",
            "suggestion": "Highlight areas of agreement before addressing differences: 'We both seem to agree that [shared point]. Building on that, perhaps we can discuss...'",
            "rationale": "Establishes connection before addressing differences",
            "confidence": 0.8
        })
    
    # 3. For ambiguous emotional content, suggest clarity with positive framing
    if emotion_assessment.get("emotional_ambiguity", 0) > 0.5:
        modifications.append({
            "type": "clarity",
            "original_fragment": "ambiguous expression",
            "suggestion": "Clarify intentions with positive framing: 'To ensure I understand correctly, you're looking for [positive interpretation]. Is that right?'",
            "rationale": "Resolves ambiguity in a generative direction",
            "confidence": 0.6
        })
    
    # 4. If few positive elements found, suggest adding appreciation
    if len(positive_elements) < 2:
        modifications.append({
            "type": "appreciation",
            "original_fragment": "neutral statement",
            "suggestion": "Add appreciation for the interaction: 'I appreciate you bringing this to my attention' or 'Thank you for your thoughtful approach to this matter.'",
            "rationale": "Introduces positive element to neutral communication",
            "confidence": 0.7
        })
    
    # 5. For any message, suggest highlighting connection or shared purpose
    modifications.append({
        "type": "connection",
        "original_fragment": "task-focused statement",
        "suggestion": "Frame in terms of shared goals: 'As we work together toward [shared goal], this step will help us by...'",
        "rationale": "Reinforces relationship and shared purpose",
        "confidence": 0.6
    })
    
    # 6. For forward movement, suggest concrete next steps
    if interaction_type in ["ROUTINE", "INQUIRY"]:
        modifications.append({
            "type": "next_steps",
            "original_fragment": "end of message",
            "suggestion": "Add clear, positive next steps: 'I look forward to our continued collaboration on this. The next steps I suggest are...'",
            "rationale": "Creates momentum and clarity",
            "confidence": 0.7
        })
    
    # 7. If principle_engine is available, align with principles
    if principle_engine:
        # This would ideally use the principle engine to identify principle-aligned modifications
        # For now, use a simplified approach focused on core principles
        
        # Add modifications based on Adaptive Bridge Builder principles
        modifications.append({
            "type": "principle_alignment",
            "original_fragment": "statement that could be aligned with principles",
            "suggestion": "Frame in terms of 'Harmony Through Presence': 'I'm fully present with this situation and committed to maintaining our connection through this process.'",
            "rationale": "Aligns with core Adaptive Bridge Builder principles",
            "confidence": 0.7
        })
    
    return modifications

def _log_to_learning_system(
    learning_system,
    interaction_data: Dict[str, Any],
    agent_id: str,
    result: Dict[str, Any]
) -> None:
    """
    Log the analysis and decisions to the learning system.
    
    Args:
        learning_system: Reference to the LearningSystem
        interaction_data: The original interaction data
        agent_id: ID of the agent
        result: The result of the positive reinforcement analysis
    """
    # Prepare context for learning system
    context = {
        "message_type": interaction_data.get("message", {}).get("type", "unknown"),
        "interaction_id": interaction_data.get("id", "unknown"),
        "generative_potential_score": result["generative_potential_score"],
        "modification_count": len(result.get("suggested_modifications", [])),
        "positive_elements_count": len(result.get("identified_positive_elements", [])),
        "agent_id": agent_id
    }
    
    # Define the pattern description
    pattern_description = f"Positive reinforcement analysis for interaction with agent {agent_id}"
    
    # Determine outcome type based on generative potential score
    from learning_system import LearningDimension, OutcomeType
    
    # Map score to outcome
    if result["generative_potential_score"] > 0.7:
        outcome = OutcomeType.SUCCESSFUL
    elif result["generative_potential_score"] > 0.3:
        outcome = OutcomeType.PARTIALLY_SUCCESSFUL
    elif result["generative_potential_score"] > -0.3:
        outcome = OutcomeType.NEUTRAL
    elif result["generative_potential_score"] > -0.7:
        outcome = OutcomeType.PARTIALLY_UNSUCCESSFUL
    else:
        outcome = OutcomeType.UNSUCCESSFUL
    
    # Record in learning system
    try:
        learning_system.track_interaction(
            pattern_description=pattern_description,
            context=context,
            dimensions=[
                LearningDimension.EMOTIONAL_INTELLIGENCE,
                LearningDimension.COMMUNICATION_EFFECTIVENESS,
                LearningDimension.TRUST_BUILDING
            ],
            outcome=outcome,
            confidence=0.7,
            notes=f"Generated potential score: {result['generative_potential_score']:.2f}"
        )
    except Exception as e:
        logger.error(f"Failed to log to learning system: {e}")

# Extension methods for PrincipleEngine
def extend_principle_engine(principle_engine) -> None:
    """
    Extend a PrincipleEngine instance with the positive reinforcement function.
    
    Args:
        principle_engine: The PrincipleEngine instance to extend
    """
    principle_engine.prioritize_positive_reinforcement = lambda interaction_data, agent_id: (
        prioritize_positive_reinforcement(
            interaction_data,
            agent_id,
            principle_engine=principle_engine,
            emotional_intelligence=getattr(principle_engine, '_emotional_intelligence', None),
            learning_system=getattr(principle_engine, '_learning_system', None)
        )
    )
    
    # Ensure the PrincipleEngine instances have references to EmotionalIntelligence and LearningSystem
    # This is a convenience to avoid having to pass these explicitly each time
    if not hasattr(principle_engine, '_emotional_intelligence'):
        logger.warning("PrincipleEngine does not have a reference to EmotionalIntelligence. "
                      "Emotional assessment will use simplified heuristics.")
    
    if not hasattr(principle_engine, '_learning_system'):
        logger.warning("PrincipleEngine does not have a reference to LearningSystem. "
                      "Learning from interactions will be disabled.")
    
    logger.info("PrincipleEngine extended with prioritize_positive_reinforcement function")

def create_sample_interaction_data() -> Dict[str, Any]:
    """Create a sample interaction data dictionary for testing."""
    return {
        "id": "interaction_12345",
        "message": {
            "type": "text",
            "content": "I'm concerned about the delay in the project. It's frustrating that we haven't made progress on the key deliverables yet.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "sender": {
            "id": "agent_5678",
            "name": "Project Manager Agent",
            "type": "agent"
        },
        "history_summary": {
            "interaction_count": 5,
            "sentiment_trend": -0.2,
            "topic": "project_progress"
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_interaction_data()
    result = prioritize_positive_reinforcement(sample_data, "agent_5678")
    
    print("Positive Reinforcement Analysis:")
    print(f"Generative Potential Score: {result['generative_potential_score']:.2f}")
    print(f"Identified Positive Elements: {len(result['identified_positive_elements'])}")
    
    if result["suggested_modifications"]:
        print("\nSuggested Modifications:")
        for i, mod in enumerate(result["suggested_modifications"], 1):
            print(f"{i}. {mod['type']}: {mod['suggestion']}")
            print(f"   Rationale: {mod['rationale']}")
            print()