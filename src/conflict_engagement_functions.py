#!/usr/bin/env python3
"""
Conflict Engagement Helper Functions

This module contains helper functions for the engage_with_conflict function.
These functions handle specific aspects of engaging with conflicts and misunderstandings.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from conflict_resolver import (
    ConflictType,
    ConflictSeverity,
    ConflictRecord,
    ConflictResolutionStep
)

from conflict_engagement import (
    EngagementPlan,
    EngagementAction,
    EngagementType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConflictEngagementHelpers")

def _convert_severity_to_float(severity: ConflictSeverity) -> float:
    """Convert ConflictSeverity enum to float value."""
    mapping = {
        ConflictSeverity.MINIMAL: 0.2,
        ConflictSeverity.LOW: 0.4,
        ConflictSeverity.MODERATE: 0.6,
        ConflictSeverity.HIGH: 0.8,
        ConflictSeverity.CRITICAL: 1.0
    }
    return mapping.get(severity, 0.5)

def _add_acknowledgment_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    communication_style: Optional[Any] = None,
    emotional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add acknowledgment actions to a conflict engagement plan.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        communication_style: Optional communication style of the other agent
        emotional_context: Optional emotional context from the message
    """
    conflict_type = conflict_record.conflict_type
    agent_id = conflict_record.agents[0]
    
    # Create base acknowledgment content
    acknowledgment_content = f"I want to acknowledge that we seem to have different perspectives on this matter."
    
    # Customize based on conflict type
    if conflict_type == ConflictType.GOAL:
        acknowledgment_content = "I recognize that we may have different objectives in this situation. I'd like to understand your goals better."
    elif conflict_type == ConflictType.VALUE:
        acknowledgment_content = "I see that we may have different values or principles guiding our approach. I respect your perspective and want to find common ground."
    elif conflict_type == ConflictType.FACTUAL:
        acknowledgment_content = "I notice we may have different information or understanding of the facts. Let's clarify what we each know."
    elif conflict_type == ConflictType.PROCEDURAL:
        acknowledgment_content = "I recognize that we may prefer different methods or processes. I'd like to understand your approach better."
    elif conflict_type == ConflictType.RELATIONSHIP:
        acknowledgment_content = "I sense there may be some tension in our interaction. I value our working relationship and want to address any concerns."
    elif conflict_type == ConflictType.RESOURCE:
        acknowledgment_content = "I understand that resources are limited and we may have competing needs. Let's discuss how to allocate them fairly."
    elif conflict_type == ConflictType.COMMUNICATION:
        acknowledgment_content = "I think we may have had a miscommunication. I want to ensure we understand each other clearly."
    
    # Adjust for emotional context if available
    if emotional_context and "dominant_emotions" in emotional_context:
        emotions = emotional_context["dominant_emotions"]
        if "frustrated" in emotions or "angry" in emotions:
            acknowledgment_content += " I can hear your frustration, and I want to make sure your concerns are addressed."
        elif "anxious" in emotions or "worried" in emotions:
            acknowledgment_content += " I understand this may be causing concern, and I want to work through this together."
    
    # Add the action
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.ACKNOWLEDGMENT,
        content=acknowledgment_content,
        priority=5,  # Highest priority
        context={"conflict_type": conflict_type.value}
    ))
    
    # Add validation action as alternative
    validation_content = "Your perspective is valid, and I appreciate you sharing it with me. Let's work together to find a solution."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.ACKNOWLEDGMENT,
        content=validation_content,
        priority=4
    ))

def _add_clarification_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    communication_style: Optional[Any] = None,
    emotional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add clarification actions to a conflict engagement plan.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        communication_style: Optional communication style of the other agent
        emotional_context: Optional emotional context from the message
    """
    conflict_type = conflict_record.conflict_type
    agent_id = conflict_record.agents[0]
    
    # Create clarification questions based on conflict type
    if conflict_type == ConflictType.GOAL:
        question_content = "Can you help me understand what specific outcome you're hoping to achieve?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=question_content,
            priority=4
        ))
        
        follow_up = "What would success look like from your perspective?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=follow_up,
            priority=3
        ))
    
    elif conflict_type == ConflictType.VALUE:
        question_content = "Which principles or values are most important to you in this situation?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=question_content,
            priority=4
        ))
        
        follow_up = "How do you see these values being applied in our current context?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=follow_up,
            priority=3
        ))
    
    elif conflict_type == ConflictType.FACTUAL:
        question_content = "Could you share the information or data that's informing your perspective?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=question_content,
            priority=4
        ))
        
        rephrase_content = "Let me clarify the information I'm working with, and please let me know if I'm missing anything important."
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.CLARIFICATION,
            content=rephrase_content,
            priority=3
        ))
    
    elif conflict_type == ConflictType.PROCEDURAL:
        question_content = "What process or method do you think would work best in this situation?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=question_content,
            priority=4
        ))
        
        follow_up = "What aspects of the process are most important to you?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=follow_up,
            priority=3
        ))
    
    elif conflict_type == ConflictType.RELATIONSHIP:
        question_content = "How can I better support our working relationship?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=question_content,
            priority=4
        ))
        
        follow_up = "Is there something specific in our interactions that's causing concern?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=follow_up,
            priority=3
        ))
    
    # Add a general clarification action for any conflict type
    clarification_content = "I want to make sure I fully understand your perspective. Could you elaborate on your main concerns?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.CLARIFICATION,
        content=clarification_content,
        priority=2
    ))

def _add_mediation_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    relationship_context: Dict[str, Any]
) -> None:
    """
    Add mediation actions to a conflict engagement plan.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        relationship_context: Context about the relationship
    """
    # Suggest structured mediation
    mediation_content = "Given the importance of this matter, would it be helpful to have a more structured approach to resolve our differences? I can suggest a framework we could follow."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.MEDIATION,
        content=mediation_content,
        priority=2
    ))
    
    # Propose a break if emotions are high
    break_content = "Would it be helpful to take a short break to collect our thoughts before continuing this discussion?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.BREAK,
        content=break_content,
        priority=1
    ))
    
    # Suggest third-party mediation for critical conflicts
    if conflict_record.severity == ConflictSeverity.CRITICAL:
        third_party_content = "Given the complexity of this situation, would it be helpful to involve a neutral third party to help us navigate this discussion?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.THIRD_PARTY,
            content=third_party_content,
            priority=1
        ))
    
    # Add long-term mediation step
    plan.add_long_term_step(
        description="Establish a regular check-in process to address concerns before they escalate",
        reasoning="Regular check-ins provide a structured opportunity to address tensions early",
        expected_outcome="Reduced conflict severity and improved communication",
        timeframe="Ongoing, with weekly or bi-weekly frequency"
    )

def _add_step_implementation_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    step: ConflictResolutionStep,
    communication_style: Optional[Any] = None,
    emotional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add actions to implement a specific resolution step.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        step: The resolution step to implement
        communication_style: Optional communication style of the other agent
        emotional_context: Optional emotional context from the message
    """
    # Create action based on step type
    if step.step_type == "acknowledgment":
        content = "Let's acknowledge that we have different perspectives on this matter. I value your input and want to find a solution that works for both of us."
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.ACKNOWLEDGMENT,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    elif step.step_type == "clarification":
        content = "To ensure we're on the same page, could you help me understand your main concerns and priorities in this situation?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.CLARIFICATION,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    elif step.step_type == "negotiation":
        content = "Let's identify what's most important to each of us and see if we can find a solution that addresses our key priorities."
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.COMMON_GROUND,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    elif step.step_type == "principle_based":
        content = "Let's refer back to our shared principles to guide our decision-making. What principles do you think should guide our approach here?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.PRINCIPLE_REMINDER,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    elif step.step_type == "perspective_shift":
        content = "Let me try to see this from your perspective. If I understand correctly, your key concerns are..."
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.PERSPECTIVE_SHIFT,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    elif step.step_type == "verification":
        content = "Let's check if we've addressed the main points of our disagreement. From your perspective, have we resolved the key issues?"
        plan.add_primary_action(EngagementAction(
            engagement_type=EngagementType.QUESTION,
            content=content,
            priority=5,
            context={"step_id": step.step_id}
        ))
    
    # Add a completion prompt for the step
    completion_content = f"Does this approach address your concerns? If not, what aspects do you think we should revisit?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=completion_content,
        priority=3,
        context={"step_id": step.step_id, "purpose": "completion_check"}
    ))

def _add_progress_reflection_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    progress: float
) -> None:
    """
    Add progress reflection actions to the engagement plan.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        progress: Resolution progress (0.0-1.0)
    """
    # Create reflection based on progress
    if progress < 0.3:
        content = "We're in the early stages of addressing this matter. Let's continue focusing on understanding each other's perspectives."
    elif progress < 0.7:
        content = "We've made good progress in understanding the issues. Let's now focus on finding solutions that address our key concerns."
    else:
        content = "We've made substantial progress. Let's ensure that our solution addresses all the important aspects we've discussed."
    
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.REFLECTION,
        content=content,
        priority=3,
        context={"progress": progress}
    ))
    
    # Check for satisfaction with progress
    check_content = "How do you feel about the progress we've made so far? Are there aspects we should focus on more?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=check_content,
        priority=2
    ))

def _add_common_ground_actions(
    plan: EngagementPlan,
    conflict_record: ConflictRecord,
    relationship_context: Dict[str, Any]
) -> None:
    """
    Add common ground actions to the engagement plan.
    
    Args:
        plan: The engagement plan to add actions to
        conflict_record: The conflict record being engaged with
        relationship_context: Context about the relationship
    """
    # Create common ground statement
    content = "Despite our different perspectives, I believe we share the common goal of finding an effective solution. Let's focus on what we agree on as a starting point."
    
    # Customize based on conflict type
    if conflict_record.conflict_type == ConflictType.GOAL:
        content = "Even though we may prioritize different outcomes, I believe we both want to achieve success in this project. Let's see if we can find an approach that addresses both of our priorities."
    elif conflict_record.conflict_type == ConflictType.VALUE:
        content = "While we may value different principles, I believe we both care about achieving the best possible outcome. Let's explore how we can honor our respective values in our approach."
    
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.COMMON_GROUND,
        content=content,
        priority=4
    ))
    
    # Add long-term relationship building step
    plan.add_long_term_step(
        description="Identify and document shared goals and values to reference in future interactions",
        reasoning="Having a clear understanding of common ground provides a foundation for resolving future conflicts",
        expected_outcome="Stronger working relationship with clearer understanding of shared objectives",
        timeframe="Complete within the next week"
    )

def _generate_conflict_engagement_explanation(
    conflict_record: ConflictRecord,
    message: Dict[str, Any],
    plan: EngagementPlan
) -> str:
    """
    Generate an explanation for a conflict engagement plan.
    
    Args:
        conflict_record: The conflict record being engaged with
        message: The message being analyzed
        plan: The engagement plan
        
    Returns:
        Explanation string
    """
    conflict_type = conflict_record.conflict_type.value
    severity = conflict_record.severity.value
    
    explanation = f"Detected {severity} {conflict_type} conflict based on message content. "
    
    if conflict_record.status == "detected":
        explanation += "This is a newly detected conflict. The engagement plan focuses on acknowledging the conflict, clarifying perspectives, and setting the foundation for resolution. "
    elif conflict_record.status == "planning":
        explanation += "A resolution plan has been created but not yet implemented. The engagement plan focuses on initiating the first steps of the resolution process. "
    elif conflict_record.status == "implementing":
        explanation += "Resolution is in progress. The engagement plan focuses on continuing the implementation of the resolution steps. "
    
    explanation += f"The plan includes {len(plan.primary_actions)} primary actions and {len(plan.alternative_actions)} alternative approaches. "
    
    if plan.long_term_steps:
        explanation += f"Additionally, {len(plan.long_term_steps)} long-term steps are suggested to prevent similar conflicts in the future."
    
    return explanation

def _add_confusion_addressing_actions(
    plan: EngagementPlan,
    sign: Dict[str, Any],
    message: Dict[str, Any],
    communication_style: Optional[Any] = None,
    emotional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add actions to address confusion or requests for clarification.
    
    Args:
        plan: The engagement plan to add actions to
        sign: The misunderstanding sign detected
        message: The message being analyzed
        communication_style: Optional communication style of the other agent
        emotional_context: Optional emotional context from the message
    """
    # Extract the specific confusion point if possible
    matched_text = sign.get("matched_text", [])
    confusion_point = matched_text[0] if matched_text else "your question"
    
    # Create clarification action
    clarification_content = f"Let me clarify what I meant. I'll try to explain it differently to address {confusion_point}."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.CLARIFICATION,
        content=clarification_content,
        priority=5
    ))
    
    # Add rephrasing action
    rephrase_content = "Let me rephrase that in simpler terms to make sure we're on the same page."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.REPHRASE,
        content=rephrase_content,
        priority=4
    ))
    
    # Add check for understanding
    check_content = "After my explanation, could you please let me know if that clarifies things for you?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=check_content,
        priority=3
    ))

def _add_expectation_alignment_actions(
    plan: EngagementPlan,
    sign: Dict[str, Any],
    message: Dict[str, Any],
    communication_style: Optional[Any] = None
) -> None:
    """
    Add actions to address misaligned expectations.
    
    Args:
        plan: The engagement plan to add actions to
        sign: The misunderstanding sign detected
        message: The message being analyzed
        communication_style: Optional communication style of the other agent
    """
    # Create expectation clarification action
    content = "I think we may have different expectations. Let me clarify what I can provide and understand what you're looking for."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.CLARIFICATION,
        content=content,
        priority=5
    ))
    
    # Add question about expectations
    question_content = "What specifically were you expecting in this situation? This will help me better understand how to address your needs."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=question_content,
        priority=4
    ))
    
    # Add acknowledgment
    acknowledgment_content = "I understand that my response may not have matched what you were expecting. Let's realign our expectations."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.ACKNOWLEDGMENT,
        content=acknowledgment_content,
        priority=3
    ))

def _add_hesitation_addressing_actions(
    plan: EngagementPlan,
    sign: Dict[str, Any],
    message: Dict[str, Any],
    communication_style: Optional[Any] = None
) -> None:
    """
    Add actions to address hesitation or uncertainty.
    
    Args:
        plan: The engagement plan to add actions to
        sign: The misunderstanding sign detected
        message: The message being analyzed
        communication_style: Optional communication style of the other agent
    """
    # Create reassurance action
    content = "I notice some hesitation. Please feel free to express any concerns or ask for clarification."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.ACKNOWLEDGMENT,
        content=content,
        priority=4
    ))
    
    # Add direct question
    question_content = "Is there something specific that you're uncertain about that I can address?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=question_content,
        priority=3
    ))
    
    # Add option offering
    options_content = "Would it help if I provided some options or alternatives for us to consider?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=options_content,
        priority=2
    ))

def _add_topic_refocusing_actions(
    plan: EngagementPlan,
    sign: Dict[str, Any],
    message: Dict[str, Any]
) -> None:
    """
    Add actions to refocus the discussion when topics shift abruptly.
    
    Args:
        plan: The engagement plan to add actions to
        sign: The misunderstanding sign detected
        message: The message being analyzed
    """
    # Create topic check action
    content = "Before we move on, I want to make sure we've fully addressed the previous topic. Is there anything else we should discuss about it first?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=content,
        priority=4
    ))
    
    # Add summary action
    summary_content = "Let me summarize where we are to make sure we're aligned before continuing."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.REFLECTION,
        content=summary_content,
        priority=3
    ))
    
    # Add confirmation check
    confirmation_content = "Would you like me to summarize our discussion so far to ensure we're on the same page?"
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=confirmation_content,
        priority=2
    ))

def _add_defensive_deescalation_actions(
    plan: EngagementPlan,
    sign: Dict[str, Any],
    message: Dict[str, Any],
    emotional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add actions to de-escalate when defensive language is detected.
    
    Args:
        plan: The engagement plan to add actions to
        sign: The misunderstanding sign detected
        message: The message being analyzed
        emotional_context: Optional emotional context from the message
    """
    # Create validation action
    content = "I want to acknowledge your perspective and clarify that I'm not questioning your position. I'm trying to understand it better."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.ACKNOWLEDGMENT,
        content=content,
        priority=5
    ))
    
    # Add de-escalation question
    question_content = "What would be most helpful for us to focus on to move this conversation forward constructively?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=question_content,
        priority=4
    ))
    
    # Add perspective shift
    perspective_content = "Let me take a step back and make sure I understand your viewpoint correctly."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.PERSPECTIVE_SHIFT,
        content=perspective_content,
        priority=3
    ))
    
    # Add break option for high intensity
    if emotional_context and emotional_context.get("intensity", 0) > 0.7:
        break_content = "Would it be helpful to take a short break and return to this conversation when we're both ready?"
        plan.add_alternative_action(EngagementAction(
            engagement_type=EngagementType.BREAK,
            content=break_content,
            priority=2
        ))

def _add_general_misunderstanding_actions(
    plan: EngagementPlan,
    message: Dict[str, Any],
    communication_style: Optional[Any] = None
) -> None:
    """
    Add general actions for addressing misunderstandings.
    
    Args:
        plan: The engagement plan to add actions to
        message: The message being analyzed
        communication_style: Optional communication style of the other agent
    """
    # Add general clarification action
    clarification_content = "I want to make sure we have a shared understanding. Could you tell me how you're currently interpreting what we're discussing?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.QUESTION,
        content=clarification_content,
        priority=3
    ))
    
    # Add information sharing action
    information_content = "Let me share what I understand about our current topic to make sure we're aligned."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.CLARIFICATION,
        content=information_content,
        priority=2
    ))
    
    # Add long-term step for preventing misunderstandings
    plan.add_long_term_step(
        description="Establish a shared vocabulary or glossary for key terms to prevent future misunderstandings",
        reasoning="Many misunderstandings stem from different interpretations of the same terms",
        expected_outcome="Clearer communication and fewer misunderstandings",
        timeframe="Develop over the next few interactions"
    )

def _generate_misunderstanding_engagement_explanation(
    misunderstanding_signs: List[Dict[str, Any]],
    message: Dict[str, Any],
    plan: EngagementPlan
) -> str:
    """
    Generate an explanation for a misunderstanding engagement plan.
    
    Args:
        misunderstanding_signs: The misunderstanding signs detected
        message: The message being analyzed
        plan: The engagement plan
        
    Returns:
        Explanation string
    """
    # Count sign types
    sign_types = {}
    for sign in misunderstanding_signs:
        sign_name = sign["name"]
        sign_types[sign_name] = sign_types.get(sign_name, 0) + 1
    
    # Get the most common sign type
    most_common_sign = max(sign_types.items(), key=lambda x: x[1])[0] if sign_types else "misunderstanding"
    
    # Build explanation
    explanation = f"Detected potential misunderstanding indicators in message. "
    
    if "confusion" in most_common_sign or "clarification" in most_common_sign:
        explanation += "The message shows explicit signs of confusion or requests for clarification. "
    elif "expectation" in most_common_sign:
        explanation += "There seems to be a mismatch in expectations. "
    elif "noncommittal" in most_common_sign or "hesitation" in most_common_sign:
        explanation += "The message shows signs of uncertainty or hesitation. "
    elif "topic_shift" in most_common_sign or "repeated" in most_common_sign:
        explanation += "There are signs that we may be talking past each other. "
    elif "defensive" in most_common_sign:
        explanation += "The message contains defensive language that suggests a potential misunderstanding. "
    
    explanation += f"The engagement plan includes {len(plan.primary_actions)} primary actions to address the misunderstanding and {len(plan.alternative_actions)} alternative approaches. "
    
    if plan.long_term_steps:
        explanation += f"Additionally, {len(plan.long_term_steps)} long-term steps are suggested to prevent similar misunderstandings in the future."
    
    return explanation
