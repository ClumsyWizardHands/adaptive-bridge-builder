#!/usr/bin/env python3
"""
Self Reflection Module

This module provides functionality for conducting structured self-reflection
on failures or suboptimal outcomes, identifying root causes, generating growth
journal entries, and proposing adaptations based on lessons learned.
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from learning_system import (
    LearningSystem, LearningDimension, OutcomeType, 
    InteractionPattern, AdaptationLevel, GrowthJournalEntry
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SelfReflection")

@dataclass
class ReflectionResult:
    """Results of a self-reflection process."""
    outcome_id: str
    reflection_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    root_causes: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[Dict[str, Any]] = field(default_factory=list)
    proposed_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    growth_journal_entry_id: Optional[str] = None
    confidence_level: float = 0.5  # 0.0 (low) to 1.0 (high)
    dimensions_affected: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the reflection result to a dictionary."""
        return {
            "outcome_id": self.outcome_id,
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp,
            "root_causes": self.root_causes,
            "lessons_learned": self.lessons_learned,
            "proposed_adaptations": self.proposed_adaptations,
            "growth_journal_entry_id": self.growth_journal_entry_id,
            "confidence_level": self.confidence_level,
            "dimensions_affected": self.dimensions_affected,
            "related_patterns": self.related_patterns
        }

class CauseType(Enum):
    """Types of root causes for failures or suboptimal outcomes."""
    COMMUNICATION_MISMATCH = "communication_mismatch"
    KNOWLEDGE_GAP = "knowledge_gap" 
    PROCESS_FLAW = "process_flaw"
    RESOURCE_LIMITATION = "resource_limitation"
    ASSUMPTION_ERROR = "assumption_error"
    ENVIRONMENTAL_FACTOR = "environmental_factor"
    PRINCIPLE_MISALIGNMENT = "principle_misalignment"
    STRATEGY_FLAW = "strategy_flaw"
    TIMING_ISSUE = "timing_issue"
    COORDINATION_FAILURE = "coordination_failure"

def perform_self_reflection(
    learning_system: LearningSystem,
    outcome_description: str,
    outcome_data: Dict[str, Any],
    dimensions: List[LearningDimension],
    severity: float = 0.5,  # 0.0 (minor) to 1.0 (severe)
    context: Optional[Dict[str, Any]] = None,
    related_patterns: Optional[List[str]] = None,
    related_principles: Optional[List[str]] = None,
) -> ReflectionResult:
    """
    Perform structured self-reflection on a failure or suboptimal outcome.
    
    Args:
        learning_system: The LearningSystem to integrate with
        outcome_description: Description of the suboptimal outcome or failure
        outcome_data: Detailed data about the outcome
        dimensions: Learning dimensions affected by this outcome
        severity: How severe the failure was (0.0 to 1.0)
        context: Additional context information
        related_patterns: Optional IDs of related interaction patterns
        related_principles: Optional IDs of related principles
        
    Returns:
        ReflectionResult containing analysis, lessons, and proposed adaptations
    """
    # Generate a unique ID for this reflection
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    reflection_id = f"reflect_{timestamp}_{hash(outcome_description) % 10000}"
    outcome_id = outcome_data.get("id", f"outcome_{timestamp}")
    
    # Initialize context if not provided
    context = context or {}
    
    # Initialize result
    result = ReflectionResult(
        outcome_id=outcome_id,
        reflection_id=reflection_id,
        dimensions_affected=[d.name for d in dimensions],
        related_patterns=related_patterns or []
    )
    
    # Step 1: Identify root causes
    root_causes = _identify_root_causes(
        outcome_description, 
        outcome_data, 
        context,
        related_patterns,
        learning_system
    )
    result.root_causes = root_causes
    
    # Step 2: Extract lessons learned
    lessons = _extract_lessons_learned(
        root_causes,
        outcome_data,
        dimensions,
        related_principles
    )
    result.lessons_learned = lessons
    
    # Step 3: Propose adaptations based on lessons
    adaptations = _propose_adaptations(
        lessons,
        root_causes,
        dimensions,
        severity,
        learning_system
    )
    result.proposed_adaptations = adaptations
    
    # Step 4: Calculate confidence level based on data quality
    result.confidence_level = _calculate_confidence(
        root_causes, 
        outcome_data,
        related_patterns is not None and len(related_patterns) > 0
    )
    
    # Step 5: Create growth journal entry
    journal_entry = _create_growth_journal_entry(
        learning_system,
        result,
        outcome_description,
        dimensions
    )
    if journal_entry:
        result.growth_journal_entry_id = journal_entry.get("timestamp", "")
    
    # Log the completion of self-reflection
    logger.info(
        f"Self-reflection completed for outcome {outcome_id} with "
        f"{len(root_causes)} root causes identified and "
        f"{len(adaptations)} adaptations proposed"
    )
    
    return result

def _identify_root_causes(
    outcome_description: str,
    outcome_data: Dict[str, Any],
    context: Dict[str, Any],
    related_patterns: Optional[List[str]],
    learning_system: LearningSystem
) -> List[Dict[str, Any]]:
    """
    Identify root causes for a failure or suboptimal outcome.
    
    Args:
        outcome_description: Description of the outcome
        outcome_data: Detailed data about the outcome
        context: Additional context information
        related_patterns: Optional IDs of related interaction patterns
        learning_system: The LearningSystem instance
        
    Returns:
        List of identified root causes with types and details
    """
    root_causes = []
    
    # Extract key information from outcome data
    expected_result = outcome_data.get("expected_result", {})
    actual_result = outcome_data.get("actual_result", {})
    steps_taken = outcome_data.get("steps_taken", [])
    error_messages = outcome_data.get("error_messages", [])
    
    # 1. Check for communication mismatch
    if "communication" in outcome_data or "message" in outcome_data:
        communication_data = outcome_data.get("communication", outcome_data.get("message", {}))
        if communication_data:
            mismatches = _identify_communication_mismatches(communication_data, context)
            if mismatches:
                root_causes.append({
                    "type": CauseType.COMMUNICATION_MISMATCH.value,
                    "description": "Communication approach did not match the needs of the situation",
                    "details": mismatches,
                    "confidence": 0.7,
                    "improvement_areas": ["communication_style", "message_clarity", "audience_understanding"]
                })
    
    # 2. Check for knowledge gaps
    if expected_result and actual_result:
        knowledge_gaps = _identify_knowledge_gaps(expected_result, actual_result, context)
        if knowledge_gaps:
            root_causes.append({
                "type": CauseType.KNOWLEDGE_GAP.value,
                "description": "Insufficient knowledge or information to achieve the expected outcome",
                "details": knowledge_gaps,
                "confidence": 0.8 if error_messages else 0.6,
                "improvement_areas": ["domain_knowledge", "information_gathering", "expertise_development"]
            })
    
    # 3. Check for process flaws
    if steps_taken:
        process_flaws = _identify_process_flaws(steps_taken, error_messages, context)
        if process_flaws:
            root_causes.append({
                "type": CauseType.PROCESS_FLAW.value,
                "description": "Flaws in the process or methodology used",
                "details": process_flaws,
                "confidence": 0.75,
                "improvement_areas": ["workflow_optimization", "process_design", "error_handling"]
            })
    
    # 4. Check for resource limitations
    resource_limitations = _identify_resource_limitations(outcome_data, context)
    if resource_limitations:
        root_causes.append({
            "type": CauseType.RESOURCE_LIMITATION.value,
            "description": "Insufficient resources to achieve the desired outcome",
            "details": resource_limitations,
            "confidence": 0.65,
            "improvement_areas": ["resource_allocation", "efficiency_improvement", "prioritization"]
        })
    
    # 5. Check for assumption errors
    assumption_errors = _identify_assumption_errors(outcome_data, context)
    if assumption_errors:
        root_causes.append({
            "type": CauseType.ASSUMPTION_ERROR.value,
            "description": "Incorrect assumptions that led to the suboptimal outcome",
            "details": assumption_errors,
            "confidence": 0.7,
            "improvement_areas": ["assumption_validation", "hypothesis_testing", "feedback_integration"]
        })
    
    # 6. Check for principle misalignment
    if related_principles:
        principle_issues = _identify_principle_misalignment(outcome_data, related_principles, context)
        if principle_issues:
            root_causes.append({
                "type": CauseType.PRINCIPLE_MISALIGNMENT.value,
                "description": "Actions taken were not aligned with core principles",
                "details": principle_issues,
                "confidence": 0.85,
                "improvement_areas": ["principle_integration", "ethical_reasoning", "value_alignment"]
            })
    
    # If no specific root causes identified, add a generic one based on outcome description
    if not root_causes:
        generic_cause = _identify_generic_cause(outcome_description, outcome_data)
        if generic_cause:
            root_causes.append(generic_cause)
    
    return root_causes

def _identify_communication_mismatches(
    communication_data: Dict[str, Any],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify mismatches in communication approach."""
    mismatches = []
    
    # Check for style mismatches
    if "style" in communication_data and "recipient_preferences" in context:
        style = communication_data.get("style", {})
        preferences = context.get("recipient_preferences", {})
        
        # Check formality level
        if "formality" in style and "formality" in preferences:
            if style["formality"] != preferences["formality"]:
                mismatches.append({
                    "element": "formality",
                    "used": style["formality"],
                    "preferred": preferences["formality"],
                    "impact": "Potential disconnect or discomfort in communication"
                })
        
        # Check technical detail level
        if "technical_detail" in style and "technical_detail" in preferences:
            if style["technical_detail"] != preferences["technical_detail"]:
                mismatches.append({
                    "element": "technical_detail",
                    "used": style["technical_detail"],
                    "preferred": preferences["technical_detail"],
                    "impact": "Information may have been too detailed or too vague for recipient"
                })
    
    # Check for clarity issues
    if "message" in communication_data and "feedback" in context:
        feedback = context.get("feedback", {})
        if "clarity_rating" in feedback and feedback["clarity_rating"] < 0.7:  # Assuming 0-1 rating
            mismatches.append({
                "element": "clarity",
                "rating": feedback["clarity_rating"],
                "feedback": feedback.get("clarity_feedback", "Insufficient clarity"),
                "impact": "Message was not clearly understood"
            })
    
    # Check for emotional tone mismatches
    if "emotional_tone" in communication_data and "situation" in context:
        tone = communication_data.get("emotional_tone", "")
        situation = context.get("situation", "")
        
        # Simple heuristic for inappropriate tone
        serious_situations = ["crisis", "emergency", "formal", "sensitive"]
        lighthearted_tones = ["casual", "humorous", "playful", "excited"]
        
        if any(s in situation.lower() for s in serious_situations) and any(t == tone.lower() for t in lighthearted_tones):
            mismatches.append({
                "element": "emotional_tone",
                "used": tone,
                "situation": situation,
                "impact": "Tone was inappropriately casual for a serious situation"
            })
    
    return mismatches

def _identify_knowledge_gaps(
    expected_result: Dict[str, Any],
    actual_result: Dict[str, Any],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify knowledge gaps based on differences between expected and actual results."""
    gaps = []
    
    # Compare key fields in expected vs actual results
    for key in expected_result:
        if key in actual_result:
            expected_value = expected_result[key]
            actual_value = actual_result[key]
            
            if expected_value != actual_value:
                # For numeric values, check significant difference
                if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    if abs(expected_value - actual_value) / max(1, abs(expected_value)) > 0.1:  # >10% difference
                        gaps.append({
                            "field": key,
                            "expected": expected_value,
                            "actual": actual_value,
                            "gap_type": "accuracy"
                        })
                else:
                    gaps.append({
                        "field": key,
                        "expected": expected_value,
                        "actual": actual_value,
                        "gap_type": "correctness"
                    })
        else:
            # Missing expected fields
            gaps.append({
                "field": key,
                "expected": expected_result[key],
                "actual": "Missing",
                "gap_type": "completeness"
            })
    
    # Check for references to unknown concepts in error messages
    if "error_messages" in context:
        for error in context.get("error_messages", []):
            # Look for phrases indicating knowledge gaps
            knowledge_indicators = ["unknown", "undefined", "not recognized", "not found", "invalid"]
            for indicator in knowledge_indicators:
                if indicator in error.lower():
                    gaps.append({
                        "field": "error_handling",
                        "error": error,
                        "gap_type": "missing_knowledge",
                        "indicator": indicator
                    })
                    break
    
    return gaps

def _identify_process_flaws(
    steps_taken: List[Dict[str, Any]],
    error_messages: List[str],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify flaws in the process or methodology used."""
    flaws = []
    
    # Check for skipped steps
    expected_steps = context.get("expected_steps", [])
    if expected_steps:
        executed_steps = [s.get("name", "") for s in steps_taken]
        skipped_steps = [step for step in expected_steps if step not in executed_steps]
        
        if skipped_steps:
            flaws.append({
                "flaw_type": "skipped_steps",
                "steps": skipped_steps,
                "impact": "Critical steps were skipped in the process"
            })
    
    # Check for incorrect step order
    if len(steps_taken) > 1 and "step_dependencies" in context:
        dependencies = context.get("step_dependencies", {})
        for i, step in enumerate(steps_taken[:-1]):
            current_step = step.get("name", "")
            next_step = steps_taken[i+1].get("name", "")
            
            if current_step in dependencies and next_step in dependencies[current_step]:
                # Next step depends on the current step (correct order)
                pass
            elif next_step in dependencies and current_step in dependencies[next_step]:
                # Current step depends on the next step (incorrect order)
                flaws.append({
                    "flaw_type": "incorrect_step_order",
                    "step1": current_step,
                    "step2": next_step,
                    "correct_order": f"{next_step} → {current_step}",
                    "impact": "Steps were executed in an incorrect sequence"
                })
    
    # Check for errors in step execution
    for step in steps_taken:
        if "error" in step or step.get("status", "") == "failed":
            flaws.append({
                "flaw_type": "step_execution_error",
                "step": step.get("name", "Unknown step"),
                "error": step.get("error", "Unknown error"),
                "impact": "Error occurred during step execution"
            })
    
    # Check for error handling issues
    if error_messages and "handled_errors" not in context:
        flaws.append({
            "flaw_type": "poor_error_handling",
            "errors": error_messages,
            "impact": "Errors were not properly handled during process execution"
        })
    
    # Check for timing issues
    for i, step in enumerate(steps_taken):
        if "duration" in step and "expected_duration" in step:
            actual_duration = step.get("duration", 0)
            expected_duration = step.get("expected_duration", 0)
            
            if actual_duration > expected_duration * 2:  # 100% longer than expected
                flaws.append({
                    "flaw_type": "excessive_step_duration",
                    "step": step.get("name", f"Step {i+1}"),
                    "actual_duration": actual_duration,
                    "expected_duration": expected_duration,
                    "impact": "Step took significantly longer than expected"
                })
    
    return flaws

def _identify_resource_limitations(
    outcome_data: Dict[str, Any],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify resource limitations that contributed to the outcome."""
    limitations = []
    
    # Check for time constraints
    if "time_allocated" in outcome_data and "time_required" in outcome_data:
        allocated = outcome_data.get("time_allocated", 0)
        required = outcome_data.get("time_required", 0)
        
        if required > allocated:
            limitations.append({
                "resource_type": "time",
                "allocated": allocated,
                "required": required,
                "shortfall_percentage": (required - allocated) / allocated * 100,
                "impact": "Insufficient time allocated for the task"
            })
    
    # Check for memory/computational resource issues
    if "resource_usage" in outcome_data:
        usage = outcome_data.get("resource_usage", {})
        limits = context.get("resource_limits", {})
        
        for resource, used in usage.items():
            if resource in limits and used > limits[resource] * 0.9:  # >90% of limit
                limitations.append({
                    "resource_type": resource,
                    "used": used,
                    "limit": limits[resource],
                    "usage_percentage": (used / limits[resource]) * 100,
                    "impact": f"Resource usage approached or exceeded limits for {resource}"
                })
    
    # Check for information/data limitations
    if "data_quality" in outcome_data:
        quality = outcome_data.get("data_quality", {})
        for metric, value in quality.items():
            if value < 0.7:  # Assuming 0-1 scale where <0.7 is problematic
                limitations.append({
                    "resource_type": "data",
                    "quality_metric": metric,
                    "value": value,
                    "threshold": 0.7,
                    "impact": f"Insufficient data quality for {metric}"
                })
    
    # Check for capability limitations
    if "attempted_capabilities" in outcome_data:
        attempted = outcome_data.get("attempted_capabilities", [])
        available = context.get("available_capabilities", [])
        
        missing_capabilities = [cap for cap in attempted if cap not in available]
        if missing_capabilities:
            limitations.append({
                "resource_type": "capabilities",
                "missing": missing_capabilities,
                "available": available,
                "impact": "Attempted to use capabilities that were not available"
            })
    
    return limitations

def _identify_assumption_errors(
    outcome_data: Dict[str, Any],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify incorrect assumptions that led to the outcome."""
    assumption_errors = []
    
    # Check for explicitly stated assumptions
    if "assumptions" in outcome_data and "reality" in outcome_data:
        assumptions = outcome_data.get("assumptions", {})
        reality = outcome_data.get("reality", {})
        
        for key, assumed_value in assumptions.items():
            if key in reality and assumed_value != reality[key]:
                assumption_errors.append({
                    "assumption": key,
                    "assumed_value": assumed_value,
                    "actual_value": reality[key],
                    "impact": f"Incorrect assumption about {key}"
                })
    
    # Check for implicit assumptions from context
    if "environment" in context and "expected_environment" in outcome_data:
        actual_env = context.get("environment", {})
        expected_env = outcome_data.get("expected_environment", {})
        
        for key, expected_value in expected_env.items():
            if key in actual_env and expected_value != actual_env[key]:
                assumption_errors.append({
                    "assumption": f"environment.{key}",
                    "assumed_value": expected_value,
                    "actual_value": actual_env[key],
                    "impact": f"Mismatch between expected and actual environment for {key}"
                })
    
    # Check for user/agent assumptions
    if "user_expectations" in outcome_data and "user_feedback" in context:
        expectations = outcome_data.get("user_expectations", {})
        feedback = context.get("user_feedback", {})
        
        for key, expected in expectations.items():
            if key in feedback and expected != feedback[key]:
                assumption_errors.append({
                    "assumption": f"user.{key}",
                    "assumed_value": expected,
                    "actual_value": feedback[key],
                    "impact": f"Mismatch between assumed and actual user expectations for {key}"
                })
    
    return assumption_errors

def _identify_principle_misalignment(
    outcome_data: Dict[str, Any],
    related_principles: List[str],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify misalignments with principles."""
    misalignments = []
    
    # Check for explicit principle evaluations
    if "principle_evaluations" in outcome_data:
        evaluations = outcome_data.get("principle_evaluations", {})
        for principle_id, score in evaluations.items():
            if score < 0.7 and principle_id in related_principles:  # Assuming 0-1 scale
                misalignments.append({
                    "principle_id": principle_id,
                    "alignment_score": score,
                    "threshold": 0.7,
                    "impact": f"Actions were not well-aligned with principle {principle_id}"
                })
    
    # Check for principle-related feedback
    if "feedback" in context:
        feedback = context.get("feedback", {})
        principle_feedback = feedback.get("principle_feedback", {})
        
        for principle_id, feedback_data in principle_feedback.items():
            if principle_id in related_principles and feedback_data.get("score", 1.0) < 0.7:
                misalignments.append({
                    "principle_id": principle_id,
                    "feedback": feedback_data.get("comment", "No specific comment"),
                    "score": feedback_data.get("score", 0),
                    "impact": "Received negative feedback regarding principle alignment"
                })
    
    # Check for principle conflicts
    if "principle_conflicts" in outcome_data:
        conflicts = outcome_data.get("principle_conflicts", [])
        for conflict in conflicts:
            if conflict.get("principle1") in related_principles or conflict.get("principle2") in related_principles:
                misalignments.append({
                    "conflict_type": "principle_conflict",
                    "principles": [conflict.get("principle1"), conflict.get("principle2")],
                    "resolution_approach": conflict.get("resolution", "None"),
                    "impact": "Conflict between principles was not optimally resolved"
                })
    
    return misalignments

def _identify_generic_cause(
    outcome_description: str,
    outcome_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Identify a generic cause when specific causes can't be determined."""
    
    # Extract key phrases from description to categorize
    description_lower = outcome_description.lower()
    
    # Check for communication-related phrases
    communication_phrases = ["misunderstood", "unclear", "miscommunication", "not communicated"]
    if any(phrase in description_lower for phrase in communication_phrases):
        return {
            "type": CauseType.COMMUNICATION_MISMATCH.value,
            "description": "Possible communication issues",
            "details": [{"general_issue": "Communication problems detected from description"}],
            "confidence": 0.4,
            "improvement_areas": ["communication_clarity", "message_structure"]
        }
    
    # Check for knowledge-related phrases
    knowledge_phrases = ["didn't know", "lack of knowledge", "missing information", "incorrect information"]
    if any(phrase in description_lower for phrase in knowledge_phrases):
        return {
            "type": CauseType.KNOWLEDGE_GAP.value,
            "description": "Possible knowledge or information gaps",
            "details": [{"general_issue": "Knowledge gaps detected from description"}],
            "confidence": 0.4,
            "improvement_areas": ["information_gathering", "knowledge_validation"]
        }
    
    # Check for process-related phrases
    process_phrases = ["wrong approach", "skipped step", "incorrect process", "methodology"]
    if any(phrase in description_lower for phrase in process_phrases):
        return {
            "type": CauseType.PROCESS_FLAW.value,
            "description": "Possible process or methodology issues",
            "details": [{"general_issue": "Process flaws detected from description"}],
            "confidence": 0.4,
            "improvement_areas": ["process_review", "methodology_improvement"]
        }
    
    # If no specific category matches, provide a general assessment
    return {
        "type": CauseType.STRATEGY_FLAW.value,
        "description": "General strategy or approach issues",
        "details": [{"general_issue": "Cannot determine specific cause from available information"}],
        "confidence": 0.3,
        "improvement_areas": ["strategy_review", "approach_evaluation", "feedback_collection"]
    }

def _extract_lessons_learned(
    root_causes: List[Dict[str, Any]],
    outcome_data: Dict[str, Any],
    dimensions: List[LearningDimension],
    related_principles: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """
    Extract lessons learned from the analysis of root causes.
    
    Args:
        root_causes: List of identified root causes
        outcome_data: Detailed data about the outcome
        dimensions: Learning dimensions affected by this outcome
        related_principles: Optional IDs of related principles
        
    Returns:
        List of lessons learned with types and details
    """
    lessons = []
    
    # Map each cause type to potential lesson categories
    cause_to_lesson = {
        CauseType.COMMUNICATION_MISMATCH.value: "communication",
        CauseType.KNOWLEDGE_GAP.value: "knowledge",
        CauseType.PROCESS_FLAW.value: "process",
        CauseType.RESOURCE_LIMITATION.value: "resource_management",
        CauseType.ASSUMPTION_ERROR.value: "validation",
        CauseType.ENVIRONMENTAL_FACTOR.value: "adaptability",
        CauseType.PRINCIPLE_MISALIGNMENT.value: "principle_alignment",
        CauseType.STRATEGY_FLAW.value: "strategy",
        CauseType.TIMING_ISSUE.value: "timing",
        CauseType.COORDINATION_FAILURE.value: "coordination"
    }
    
    # Extract lessons from root causes
    for cause in root_causes:
        cause_type = cause.get("type", "")
        lesson_category = cause_to_lesson.get(cause_type, "general")
        
        # Create a lesson based on the cause type
        if cause_type == CauseType.COMMUNICATION_MISMATCH.value:
            for detail in cause.get("details", []):
                element = detail.get("element", "communication")
                lesson = {
                    "category": "communication",
                    "lesson": f"Adapt {element} to match recipient preferences and situation",
                    "reasoning": f"Mismatch between used {element} ({detail.get('used', 'unknown')}) and preferred {element} ({detail.get('preferred', 'unknown')})",
                    "importance": 0.8,
                    "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.COMMUNICATION_EFFECTIVENESS, LearningDimension.EMOTIONAL_INTELLIGENCE]]
                }
                lessons.append(lesson)
        
        elif cause_type == CauseType.KNOWLEDGE_GAP.value:
            knowledge_areas = set()
            for detail in cause.get("details", []):
                field = detail.get("field", "unknown")
                knowledge_areas.add(field)
                lesson = {
                    "category": "knowledge",
                    "lesson": f"Enhance knowledge in the area of {field}",
                    "reasoning": f"Gap between expected ({detail.get('expected', 'unknown')}) and actual ({detail.get('actual', 'unknown')}) knowledge",
                    "importance": 0.8,
                    "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.DOMAIN_EXPERTISE, LearningDimension.TECHNICAL_PROFICIENCY]]
                }
                lessons.append(lesson)
                
        elif cause_type == CauseType.PROCESS_FLAW.value:
            for detail in cause.get("details", []):
                flaw_type = detail.get("flaw_type", "process")
                lesson = {
                    "category": "process",
                    "lesson": f"Improve {flaw_type.replace('_', ' ')} in the workflow",
                    "reasoning": detail.get("impact", "Process flaw affected outcome quality"),
                    "importance": 0.75,
                    "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.PROCESS_OPTIMIZATION, LearningDimension.PROBLEM_SOLVING]]
                }
                lessons.append(lesson)
                
        elif cause_type == CauseType.RESOURCE_LIMITATION.value:
            for detail in cause.get("details", []):
                resource_type = detail.get("resource_type", "resource")
                lesson = {
                    "category": "resource_management",
                    "lesson": f"Better allocate or plan for {resource_type} resources",
                    "reasoning": detail.get("impact", "Resource limitation affected outcome quality"),
                    "importance": 0.7,
                    "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.RESOURCE_MANAGEMENT, LearningDimension.PLANNING_EFFECTIVENESS]]
                }
                lessons.append(lesson)
                
        elif cause_type == CauseType.ASSUMPTION_ERROR.value:
            for detail in cause.get("details", []):
                assumption = detail.get("assumption", "assumption")
                lesson = {
                    "category": "validation",
                    "lesson": f"Validate assumptions about {assumption} before proceeding",
                    "reasoning": f"Mismatch between assumed ({detail.get('assumed_value', 'unknown')}) and actual ({detail.get('actual_value', 'unknown')}) values",
                    "importance": 0.8,
                    "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.CRITICAL_THINKING, LearningDimension.HYPOTHESIS_TESTING]]
                }
                lessons.append(lesson)
                
        elif cause_type == CauseType.PRINCIPLE_MISALIGNMENT.value:
            for detail in cause.get("details", []):
                principle_id = detail.get("principle_id", "unknown")
                if related_principles and principle_id in related_principles:
                    lesson = {
                        "category": "principle_alignment",
                        "lesson": f"Ensure stronger alignment with principle {principle_id}",
                        "reasoning": detail.get("impact", "Insufficient alignment with core principles"),
                        "importance": 0.9,
                        "applicable_dimensions": [d.name for d in dimensions if d in [LearningDimension.ETHICAL_REASONING, LearningDimension.VALUE_ALIGNMENT]]
                    }
                    lessons.append(lesson)
        
        # Generic lesson if none of the specific types matched
        else:
            lesson = {
                "category": lesson_category,
                "lesson": f"Improve approach to {lesson_category.replace('_', ' ')}",
                "reasoning": cause.get("description", "Identified as a contributing factor"),
                "importance": 0.6,
                "applicable_dimensions": [d.name for d in dimensions]
            }
            lessons.append(lesson)
    
    # Deduplicate similar lessons
    unique_lessons = []
    lesson_keys = set()
    
    for lesson in lessons:
        key = f"{lesson['category']}:{lesson['lesson']}"
        if key not in lesson_keys:
            lesson_keys.add(key)
            unique_lessons.append(lesson)
    
    return unique_lessons

def _propose_adaptations(
    lessons: List[Dict[str, Any]],
    root_causes: List[Dict[str, Any]],
    dimensions: List[LearningDimension],
    severity: float,
    learning_system: LearningSystem
) -> List[Dict[str, Any]]:
    """
    Propose adaptations based on lessons learned.
    
    Args:
        lessons: List of extracted lessons
        root_causes: List of identified root causes
        dimensions: Learning dimensions affected
        severity: Severity of the outcome (0.0 to 1.0)
        learning_system: The LearningSystem instance
        
    Returns:
        List of proposed adaptations with types and details
    """
    adaptations = []
    
    # Determine appropriate adaptation level based on severity
    if severity >= 0.8:
        adaptation_level = AdaptationLevel.PARADIGM
    elif severity >= 0.5:
        adaptation_level = AdaptationLevel.STRATEGY
    else:
        adaptation_level = AdaptationLevel.TACTIC
    
    # Map lesson categories to adaptation approaches
    category_to_adaptation = {
        "communication": ["style_adjustment", "structure_change", "medium_selection"],
        "knowledge": ["training_focus", "information_gathering", "expertise_expansion"],
        "process": ["workflow_optimization", "step_modification", "automation_enhancement"],
        "resource_management": ["allocation_adjustment", "requirement_reduction", "efficiency_improvement"],
        "validation": ["verification_process", "testing_approach", "feedback_mechanism"],
        "principle_alignment": ["principle_integration", "ethics_enforcement", "value_emphasis"],
        "strategy": ["approach_revision", "goal_alignment", "methodology_change"],
        "timing": ["timing_optimization", "sequence_adjustment", "pace_management"],
        "coordination": ["collaboration_improvement", "role_clarification", "responsibility_allocation"]
    }
    
    # Group lessons by category
    lessons_by_category = defaultdict(list)
    for lesson in lessons:
        category = lesson.get("category", "general")
        lessons_by_category[category].append(lesson)
    
    # Process each category and generate adaptations
    for category, category_lessons in lessons_by_category.items():
        # Get potential adaptation approaches for this category
        approaches = category_to_adaptation.get(category, ["general_improvement"])
        
        # Determine importance of this category based on lesson importance and count
        importance_sum = sum(lesson.get("importance", 0.5) for lesson in category_lessons)
        category_importance = importance_sum / max(1, len(category_lessons))
        
        # Select appropriate approaches based on importance and adaptation level
        selected_approaches = []
        if category_importance >= 0.8:
            # High importance - consider all approaches
            selected_approaches = approaches
        elif category_importance >= 0.5:
            # Medium importance - consider up to 2 approaches
            selected_approaches = approaches[:2] if approaches else ["general_improvement"]
        else:
            # Low importance - consider 1 approach
            selected_approaches = approaches[:1] if approaches else ["general_improvement"]
        
        # Generate adaptation for each selected approach
        for approach in selected_approaches:
            adaptation = _generate_adaptation_for_approach(
                approach,
                category,
                category_lessons,
                adaptation_level,
                dimensions,
                learning_system
            )
            if adaptation:
                adaptations.append(adaptation)
    
    # If no adaptations were generated, create a generic one
    if not adaptations:
        generic_adaptation = {
            "type": "general_improvement",
            "level": adaptation_level.value,
            "description": "Review and improve general approach",
            "reasoning": "Based on identified issues, a general improvement in approach is recommended",
            "implementation_steps": [
                "Review the process and identify specific areas for improvement",
                "Develop a more detailed improvement plan based on this review",
                "Implement changes incrementally and monitor effects"
            ],
            "expected_impact": 0.5,
            "effort_level": 0.5,
            "priority": severity
        }
        adaptations.append(generic_adaptation)
    
    # Sort adaptations by expected impact and effort (prioritizing high impact, low effort)
    adaptations.sort(key=lambda a: (a.get("expected_impact", 0) - a.get("effort_level", 0)), reverse=True)
    
    return adaptations

def _generate_adaptation_for_approach(
    approach: str,
    category: str,
    lessons: List[Dict[str, Any]],
    adaptation_level: AdaptationLevel,
    dimensions: List[LearningDimension],
    learning_system: LearningSystem
) -> Dict[str, Any]:
    """
    Generate a specific adaptation for a given approach and category.
    
    Args:
        approach: The adaptation approach
        category: The lesson category
        lessons: Lessons within this category
        adaptation_level: The level of adaptation (tactic, strategy, paradigm)
        dimensions: Learning dimensions affected
        learning_system: The LearningSystem instance
        
    Returns:
        A dictionary containing the adaptation details
    """
    # Calculate importance, effort and impact based on lessons
    importance = sum(lesson.get("importance", 0.5) for lesson in lessons) / max(1, len(lessons))
    
    # Base effort level on adaptation level (higher levels require more effort)
    base_effort = {
        AdaptationLevel.TACTIC: 0.3,
        AdaptationLevel.STRATEGY: 0.6,
        AdaptationLevel.PARADIGM: 0.9
    }.get(adaptation_level, 0.5)
    
    # Adjust effort based on approach complexity
    approach_complexity = {
        "style_adjustment": 0.2,
        "medium_selection": 0.3,
        "structure_change": 0.5,
        "training_focus": 0.4,
        "information_gathering": 0.3,
        "expertise_expansion": 0.7,
        "workflow_optimization": 0.6,
        "step_modification": 0.4,
        "automation_enhancement": 0.8,
        "allocation_adjustment": 0.4,
        "requirement_reduction": 0.5,
        "efficiency_improvement": 0.6,
        "verification_process": 0.5,
        "testing_approach": 0.6,
        "feedback_mechanism": 0.4,
        "principle_integration": 0.7,
        "ethics_enforcement": 0.5,
        "value_emphasis": 0.3,
        "approach_revision": 0.6,
        "goal_alignment": 0.5,
        "methodology_change": 0.8,
        "timing_optimization": 0.4,
        "sequence_adjustment": 0.5,
        "pace_management": 0.3,
        "collaboration_improvement": 0.6,
        "role_clarification": 0.4,
        "responsibility_allocation": 0.5,
        "general_improvement": 0.5
    }.get(approach, 0.5)
    
    effort_level = (base_effort + approach_complexity) / 2
    
    # Estimate impact based on importance and adaptation level
    impact_factor = {
        AdaptationLevel.TACTIC: 0.6,
        AdaptationLevel.STRATEGY: 0.8,
        AdaptationLevel.PARADIGM: 1.0
    }.get(adaptation_level, 0.7)
    
    expected_impact = importance * impact_factor
    
    # Generate description and steps based on approach and category
    description, steps = _get_adaptation_details(approach, category, adaptation_level, lessons)
    
    # Create the adaptation
    adaptation = {
        "type": approach,
        "level": adaptation_level.value,
        "description": description,
        "reasoning": " ".join([lesson.get("reasoning", "") for lesson in lessons[:2]]),
        "implementation_steps": steps,
        "expected_impact": expected_impact,
        "effort_level": effort_level,
        "priority": importance,
        "related_dimensions": [d.name for d in dimensions],
        "lessons_addressed": [lesson.get("lesson", "") for lesson in lessons]
    }
    
    return adaptation

def _get_adaptation_details(
    approach: str,
    category: str,
    adaptation_level: AdaptationLevel,
    lessons: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Generate description and implementation steps for an adaptation.
    
    Args:
        approach: The adaptation approach
        category: The lesson category
        adaptation_level: The level of adaptation
        lessons: Related lessons
        
    Returns:
        Tuple of (description, implementation_steps)
    """
    # Extract relevant lesson details for customization
    lesson_focuses = []
    for lesson in lessons:
        parts = lesson.get("lesson", "").split("in the area of ")
        if len(parts) > 1:
            lesson_focuses.append(parts[1])
        else:
            parts = lesson.get("lesson", "").split("about ")
            if len(parts) > 1:
                lesson_focuses.append(parts[1])
    
    focus_area = lesson_focuses[0] if lesson_focuses else f"{category} approach"
    
    # Descriptions for different approaches and levels
    if approach == "style_adjustment" and category == "communication":
        description = f"Adjust communication style to better match recipient needs and context"
        steps = [
            "Analyze previous communication patterns and recipient responses",
            "Identify specific style elements (formality, detail level, tone) to adjust",
            "Develop guidelines for adapting style based on recipient and context",
            "Implement and monitor effectiveness of style adjustments"
        ]
    
    elif approach == "training_focus" and category == "knowledge":
        description = f"Focus training and knowledge development on {focus_area}"
        steps = [
            f"Identify specific knowledge gaps in {focus_area}",
            "Develop or source appropriate learning resources",
            "Allocate dedicated time for knowledge acquisition",
            "Apply new knowledge in practical scenarios and evaluate improvement"
        ]
    
    elif approach == "workflow_optimization" and category == "process":
        description = "Optimize workflow to address identified process flaws"
        steps = [
            "Map the current process workflow in detail",
            "Identify bottlenecks, redundancies, and error-prone steps",
            "Redesign workflow to address identified issues",
            "Implement changes incrementally and evaluate effectiveness"
        ]
    
    elif approach == "verification_process" and category == "validation":
        description = f"Implement stronger verification processes for assumptions about {focus_area}"
        steps = [
            "Identify critical assumptions that require verification",
            "Design systematic verification methods for each assumption type",
            "Integrate verification steps into standard workflows",
            "Document and share verified information to prevent future errors"
        ]
    
    elif approach == "principle_integration" and category == "principle_alignment":
        description = "Strengthen integration of principles into decision processes"
        steps = [
            "Review principles and their practical implications",
            "Develop explicit decision criteria based on principles",
            "Create checkpoints in processes to verify principle alignment",
            "Provide feedback mechanisms to report potential principle misalignments"
        ]
    
    # Generic details for other combinations
    else:
        # Customize based on adaptation level
        if adaptation_level == AdaptationLevel.TACTIC:
            description = f"Implement tactical improvements to {approach.replace('_', ' ')} in {category.replace('_', ' ')}"
            steps = [
                f"Identify specific {category.replace('_', ' ')} tactics that need improvement",
                f"Design targeted enhancements to {approach.replace('_', ' ')}",
                "Implement changes with minimal disruption to existing processes",
                "Monitor effectiveness and make incremental adjustments"
            ]
        
        elif adaptation_level == AdaptationLevel.STRATEGY:
            description = f"Revise strategic approach to {category.replace('_', ' ')} through {approach.replace('_', ' ')}"
            steps = [
                f"Conduct strategic review of current {category.replace('_', ' ')} approach",
                f"Develop new strategy focusing on {approach.replace('_', ' ')}",
                "Create implementation plan with clear milestones and metrics",
                "Roll out strategic changes and monitor effectiveness"
            ]
        
        else:  # PARADIGM
            description = f"Transform paradigm for {category.replace('_', ' ')} with focus on {approach.replace('_', ' ')}"
            steps = [
                f"Challenge fundamental assumptions about {category.replace('_', ' ')}",
                f"Develop new paradigm centered on {approach.replace('_', ' ')}",
                "Create comprehensive transformation roadmap",
                "Implement transformative changes in phases",
                "Establish new norms, processes, and evaluation methods"
            ]
    
    return description, steps

def _calculate_confidence(
    root_causes: List[Dict[str, Any]],
    outcome_data: Dict[str, Any],
    has_related_patterns: bool
) -> float:
    """
    Calculate confidence level in reflection results based on data quality.
    
    Args:
        root_causes: Identified root causes
        outcome_data: Detailed outcome data
        has_related_patterns: Whether related patterns were provided
        
    Returns:
        Confidence score (0.0-1.0)
    """
    # Base confidence
    confidence = 0.5
    
    # Adjust based on data quality
    if len(root_causes) > 0:
        # More root causes identified increases confidence (up to a point)
        confidence += min(0.2, len(root_causes) * 0.05)
    
    # Average confidence from root causes
    if root_causes:
        cause_confidences = [cause.get("confidence", 0.5) for cause in root_causes]
        avg_cause_confidence = sum(cause_confidences) / len(cause_confidences)
        confidence = (confidence + avg_cause_confidence) / 2
    
    # Data richness factors
    data_factors = [
        "steps_taken" in outcome_data and len(outcome_data.get("steps_taken", [])) > 0,
        "expected_result" in outcome_data and "actual_result" in outcome_data,
        "error_messages" in outcome_data and len(outcome_data.get("error_messages", [])) > 0,
        has_related_patterns
    ]
    
    # Each positive factor adds confidence
    confidence += sum(0.05 for factor in data_factors if factor)
    
    # Cap at 1.0
    return min(1.0, confidence)

def _create_growth_journal_entry(
    learning_system: LearningSystem,
    reflection_result: ReflectionResult,
    outcome_description: str,
    dimensions: List[LearningDimension]
) -> Dict[str, Any]:
    """
    Create a growth journal entry based on reflection results.
    
    Args:
        learning_system: The LearningSystem instance
        reflection_result: Results from the reflection process
        outcome_description: Description of the outcome
        dimensions: Learning dimensions affected
        
    Returns:
        The created growth journal entry
    """
    # Create a growth journal entry
    entry = GrowthJournalEntry(
        timestamp=reflection_result.timestamp,
        outcome_type=OutcomeType.SUBOPTIMAL,
        title=f"Reflection on: {outcome_description[:50]}{'...' if len(outcome_description) > 50 else ''}",
        situation=outcome_description,
        dimensions=[d.name for d in dimensions],
        lessons_learned=[lesson.get("lesson", "") for lesson in reflection_result.lessons_learned],
        improvement_focus=", ".join(set([adaptation.get("description", "").split(" ")[0] for adaptation in reflection_result.proposed_adaptations[:3]]))
    )
    
    # Add to learning system and return entry
    try:
        entry_data = learning_system.add_journal_entry(entry)
        return entry_data
    except Exception as e:
        logger.error(f"Failed to create growth journal entry: {e}")
        return None