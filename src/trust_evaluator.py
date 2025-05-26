#!/usr/bin/env python3
"""
Trust Evaluator for Adaptive Bridge Builder

This module implements trust impact evaluation functionality to assess
potential actions before execution, identify trust risks, and suggest
trust-preserving alternatives. It integrates with the RelationshipTracker,
PrincipleEngine, and A2ATaskHandler to provide comprehensive trust assessment.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime, timezone

from relationship_tracker import (
    RelationshipTracker, AgentRelationship, TrustLevel, 
    RelationshipStatus, InteractionType
)
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrustEvaluator")

class TrustImpactSeverity(Enum):
    """Severity levels for trust impact warnings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class ActionCategory(Enum):
    """Categories of actions that can impact trust."""
    COMMUNICATION = "communication"
    DATA_SHARING = "data_sharing"
    ACCESS_REQUEST = "access_request"
    TASK_REQUEST = "task_request"
    CONFIGURATION_CHANGE = "configuration_change"
    RELATIONSHIP_MANAGEMENT = "relationship_management"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_HANDLING = "error_handling"
    FEEDBACK = "feedback"

class TrustBuildingStrategy(Enum):
    """Strategies for building and maintaining trust."""
    TRANSPARENCY = "transparency"
    CONSISTENCY = "consistency"
    COMPETENCE = "competence"
    INTEGRITY = "integrity"
    BENEVOLENCE = "benevolence"
    ACCOUNTABILITY = "accountability"
    RECIPROCITY = "reciprocity"
    PREDICTABILITY = "predictability"

class TrustImpactType(Enum):
    """Types of trust impact for reporting."""
    TRUST_EROSION = "trust_erosion"  # Action will likely decrease trust
    TRUST_BUILDING = "trust_building"  # Action will likely build trust
    TRUST_MAINTENANCE = "trust_maintenance"  # Action will likely maintain current trust
    TRUST_RECOVERY = "trust_recovery"  # Action will likely recover damaged trust
    TRUST_NEUTRAL = "trust_neutral"  # Action has negligible impact on trust

class TrustEvaluator:
    """
    Evaluates the potential trust impact of actions before execution.
    
    This class provides functionality to assess how specific actions may impact
    trust levels with other agents, identify potentially risky actions, and suggest
    alternatives that preserve or build trust. It integrates with the RelationshipTracker
    to consider relationship history and current trust levels.
    """
    
    def __init__(
        self,
        relationship_tracker: Optional[RelationshipTracker] = None,
        principle_engine: Optional[PrincipleEngine] = None
    ):
        """
        Initialize the TrustEvaluator.
        
        Args:
            relationship_tracker: Optional RelationshipTracker for relationship history.
            principle_engine: Optional PrincipleEngine for principle alignment.
        """
        self.relationship_tracker = relationship_tracker
        self.principle_engine = principle_engine
        
        # Trust impact thresholds
        self.severe_erosion_threshold = -5.0
        self.moderate_erosion_threshold = -2.0
        self.neutral_threshold = 0.5
        self.building_threshold = 2.0
        
        # Category weights for trust impact
        self.category_weights = {
            ActionCategory.COMMUNICATION.value: 1.0,
            ActionCategory.DATA_SHARING.value: 1.5,
            ActionCategory.ACCESS_REQUEST.value: 1.8,
            ActionCategory.TASK_REQUEST.value: 1.2,
            ActionCategory.CONFIGURATION_CHANGE.value: 1.3,
            ActionCategory.RELATIONSHIP_MANAGEMENT.value: 1.7,
            ActionCategory.RESOURCE_ALLOCATION.value: 1.1,
            ActionCategory.ERROR_HANDLING.value: 1.4,
            ActionCategory.FEEDBACK.value: 0.9
        }
        
        logger.info("TrustEvaluator initialized")
    
    def evaluate_trust_impact(
        self,
        action: Dict[str, Any],
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
        a2a_context: Optional[Dict[str, Any]] = None,
        include_alternatives: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the potential trust impact of an action.
        
        Args:
            action: Dictionary describing the action to evaluate.
            agent_id: ID of the agent the action targets.
            context: Optional additional context for evaluation.
            a2a_context: Optional A2A task handler context.
            include_alternatives: Whether to include alternative suggestions.
            
        Returns:
            Dictionary with trust impact evaluation results.
        """
        # Initialize context if not provided
        context = context or {}
        
        # Get relationship if available
        relationship = None
        if self.relationship_tracker:
            relationship = self.relationship_tracker.get_relationship(agent_id, create_if_missing=False)
        
        # Extract action details
        action_type = action.get("type", "unknown")
        action_content = action.get("content", {})
        action_category = self._determine_action_category(action_type, action_content)
        
        # Calculate base trust impact
        base_impact, impact_factors = self._calculate_base_trust_impact(
            action_type, action_content, action_category
        )
        
        # Apply relationship modifiers if relationship exists
        relationship_modifiers = {}
        if relationship:
            base_impact, relationship_modifiers = self._apply_relationship_modifiers(
                base_impact, relationship, action_category, action_type
            )
        
        # Apply principle alignment if principle engine exists
        principle_alignment = None
        principle_modifiers = {}
        if self.principle_engine:
            principle_alignment, principle_impact, principle_modifiers = self._calculate_principle_alignment(
                action, context
            )
            base_impact += principle_impact
        
        # Determine trust impact type
        if base_impact <= self.severe_erosion_threshold:
            impact_type = TrustImpactType.TRUST_EROSION
        elif base_impact <= self.moderate_erosion_threshold:
            impact_type = TrustImpactType.TRUST_EROSION
        elif base_impact < self.neutral_threshold:
            impact_type = TrustImpactType.TRUST_NEUTRAL
        elif base_impact >= self.building_threshold:
            if relationship and relationship.status in [
                RelationshipStatus.DAMAGED, RelationshipStatus.STRAINED
            ]:
                impact_type = TrustImpactType.TRUST_RECOVERY
            else:
                impact_type = TrustImpactType.TRUST_BUILDING
        else:
            impact_type = TrustImpactType.TRUST_MAINTENANCE
        
        # Generate warnings for trust-eroding actions
        warnings = []
        if base_impact <= self.moderate_erosion_threshold:
            warnings = self._generate_trust_warnings(
                action, base_impact, action_category, relationship, impact_factors
            )
        
        # Generate alternative suggestions if requested
        alternatives = []
        if include_alternatives and base_impact < self.neutral_threshold:
            alternatives = self._suggest_alternatives(
                action, action_category, relationship, warnings, context
            )
        
        # Determine severity based on impact score
        severity = self._determine_severity(base_impact)
        
        # Create the evaluation result
        result = {
            "trust_impact_score": base_impact,
            "trust_impact_type": impact_type.value,
            "severity": severity.value,
            "action_category": action_category.value,
            "warnings": warnings,
            "alternatives": alternatives,
            "impact_factors": impact_factors,
            "relationship_modifiers": relationship_modifiers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add principle alignment if available
        if principle_alignment is not None:
            result["principle_alignment"] = principle_alignment
            result["principle_modifiers"] = principle_modifiers
        
        # Add relationship context if available
        if relationship:
            result["relationship_context"] = {
                "trust_level": relationship.trust_level.value,
                "trust_score": relationship.trust_score,
                "status": relationship.status.value,
                "interaction_count": relationship.interaction_count
            }
        
        return result
    
    def _determine_action_category(
        self,
        action_type: str,
        action_content: Dict[str, Any]
    ) -> ActionCategory:
        """
        Determine the category of an action based on its type and content.
        
        Args:
            action_type: Type of action.
            action_content: Content details of the action.
            
        Returns:
            ActionCategory for the action.
        """
        # Map common action types to categories
        type_to_category = {
            "message": ActionCategory.COMMUNICATION,
            "data_request": ActionCategory.DATA_SHARING,
            "data_share": ActionCategory.DATA_SHARING,
            "access_request": ActionCategory.ACCESS_REQUEST,
            "permission_request": ActionCategory.ACCESS_REQUEST,
            "task_request": ActionCategory.TASK_REQUEST,
            "task_assignment": ActionCategory.TASK_REQUEST,
            "config_change": ActionCategory.CONFIGURATION_CHANGE,
            "update_settings": ActionCategory.CONFIGURATION_CHANGE,
            "relationship_update": ActionCategory.RELATIONSHIP_MANAGEMENT,
            "trust_repair": ActionCategory.RELATIONSHIP_MANAGEMENT,
            "resource_request": ActionCategory.RESOURCE_ALLOCATION,
            "error_response": ActionCategory.ERROR_HANDLING,
            "feedback": ActionCategory.FEEDBACK,
            "evaluation": ActionCategory.FEEDBACK
        }
        
        # Check for direct match
        if action_type in type_to_category:
            return type_to_category[action_type]
        
        # Check for partial matches
        for key, category in type_to_category.items():
            if key in action_type:
                return category
        
        # Additional content-based classification
        if "message" in action_content or "communication" in action_content:
            return ActionCategory.COMMUNICATION
        elif "data" in action_content or "information" in action_content:
            return ActionCategory.DATA_SHARING
        elif "access" in action_content or "permission" in action_content:
            return ActionCategory.ACCESS_REQUEST
        elif "task" in action_content or "job" in action_content:
            return ActionCategory.TASK_REQUEST
        elif "config" in action_content or "setting" in action_content:
            return ActionCategory.CONFIGURATION_CHANGE
        elif "relationship" in action_content or "trust" in action_content:
            return ActionCategory.RELATIONSHIP_MANAGEMENT
        elif "resource" in action_content or "allocate" in action_content:
            return ActionCategory.RESOURCE_ALLOCATION
        elif "error" in action_content or "exception" in action_content:
            return ActionCategory.ERROR_HANDLING
        elif "feedback" in action_content or "review" in action_content:
            return ActionCategory.FEEDBACK
        
        # Default to communication if no other match
        return ActionCategory.COMMUNICATION
    
    def _has_transparency(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action has transparency elements."""
        # Look for transparency indicators in content
        content_str = str(action_content).lower()
        transparency_indicators = [
            "explanation", "justification", "reason", "purpose", 
            "intent", "rationale", "why", "clarification",
            "transparency", "open", "visible"
        ]
        
        # Check fields that indicate transparency
        has_transparency_fields = any(
            field in action_content for field in [
                "explanation", "justification", "rationale", "purpose", 
                "intent", "reasoning", "transparency_statement"
            ]
        )
        
        # Check content for transparency indicators
        has_transparency_content = any(
            indicator in content_str for indicator in transparency_indicators
        )
        
        return has_transparency_fields or has_transparency_content
    
    def _shows_competence(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action demonstrates competence."""
        # Look for competence indicators in content
        content_str = str(action_content).lower()
        competence_indicators = [
            "expertise", "skills", "qualified", "experienced",
            "professional", "knowledgeable", "proficient",
            "capable", "competent", "efficient", "effective"
        ]
        
        # Check fields that indicate competence
        has_competence_fields = any(
            field in action_content for field in [
                "expertise", "qualifications", "capabilities", 
                "experience", "skills", "competence_statement"
            ]
        )
        
        # Check content for competence indicators
        has_competence_content = any(
            indicator in content_str for indicator in competence_indicators
        )
        
        # Check for well-structured content
        is_well_structured = isinstance(action_content, dict) and len(action_content) >= 3
        
        return has_competence_fields or has_competence_content or is_well_structured
    
    def _demonstrates_integrity(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action demonstrates integrity."""
        # Look for integrity indicators in content
        content_str = str(action_content).lower()
        integrity_indicators = [
            "honest", "truthful", "ethical", "moral", "principle",
            "integrity", "transparent", "accountability", "responsible",
            "consistent", "fair", "equitable", "unbiased"
        ]
        
        # Check fields that indicate integrity
        has_integrity_fields = any(
            field in action_content for field in [
                "ethical_considerations", "principles", "integrity_statement", 
                "accountability", "transparency", "responsibility"
            ]
        )
        
        # Check content for integrity indicators
        has_integrity_content = any(
            indicator in content_str for indicator in integrity_indicators
        )
        
        return has_integrity_fields or has_integrity_content
    
    def _shows_benevolence(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action shows benevolence."""
        # Look for benevolence indicators in content
        content_str = str(action_content).lower()
        benevolence_indicators = [
            "help", "assist", "support", "benefit", "care",
            "well-being", "welfare", "good", "positive",
            "beneficial", "helpful", "supportive", "considerate"
        ]
        
        # Check fields that indicate benevolence
        has_benevolence_fields = any(
            field in action_content for field in [
                "benefits", "assistance", "support", "user_benefit",
                "recipient_benefit", "positive_impact", "care_statement"
            ]
        )
        
        # Check content for benevolence indicators
        has_benevolence_content = any(
            indicator in content_str for indicator in benevolence_indicators
        )
        
        return has_benevolence_fields or has_benevolence_content
    
    def _lacks_transparency(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action lacks transparency."""
        # If it has transparency, it doesn't lack it
        if self._has_transparency(action_content):
            return False
            
        # Look for signs of deliberate opacity or vagueness
        content_str = str(action_content).lower()
        opacity_indicators = [
            "confidential", "private", "secret", "restricted",
            "undisclosed", "hidden", "unavailable", "classified"
        ]
        
        # Check for minimal information
        is_minimal = (
            isinstance(action_content, dict) and len(action_content) < 2
            or isinstance(action_content, str) and len(action_content) < 20
        )
        
        # Check for opacity indicators
        has_opacity = any(
            indicator in content_str for indicator in opacity_indicators
        )
        
        # Missing important fields that would provide clarity
        missing_important_fields = True
        if isinstance(action_content, dict):
            important_fields = ["purpose", "reason", "explanation", "description", "details"]
            if any(field in action_content for field in important_fields):
                missing_important_fields = False
                
        return is_minimal or has_opacity or missing_important_fields
    
    def _shows_inconsistency(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action shows inconsistency."""
        # Look for inconsistency indicators in content
        content_str = str(action_content).lower()
        inconsistency_indicators = [
            "change", "differ", "inconsistent", "contradict",
            "contrary", "oppose", "conflict", "diverge", "deviate"
        ]
        
        # Check for contradictions within the content
        has_contradictions = False
        if isinstance(action_content, dict):
            # Look for fields that might contradict each other
            if "purpose" in action_content and "reason" in action_content:
                purpose = str(action_content["purpose"]).lower()
                reason = str(action_content["reason"]).lower()
                if not any(word in reason for word in purpose.split()):
                    has_contradictions = True
        
        # Check content for inconsistency indicators
        has_inconsistency_content = any(
            indicator in content_str for indicator in inconsistency_indicators
        )
        
        return has_contradictions or has_inconsistency_content
    
    def _indicates_incompetence(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action indicates incompetence."""
        # Look for incompetence indicators in content
        content_str = str(action_content).lower()
        incompetence_indicators = [
            "unsure", "uncertain", "don't know", "not sure",
            "inexperienced", "unfamiliar", "learning", "first time",
            "unclear", "confused", "mistake", "error", "oversight"
        ]
        
        # Check content for incompetence indicators
        has_incompetence_content = any(
            indicator in content_str for indicator in incompetence_indicators
        )
        
        # Check for poorly structured content
        is_poorly_structured = False
        if isinstance(action_content, dict):
            # Check for empty or very short values
            empty_or_short_values = 0
            for key, value in action_content.items():
                if not value or (isinstance(value, str) and len(value) < 5):
                    empty_or_short_values += 1
            
            is_poorly_structured = (empty_or_short_values / max(1, len(action_content))) > 0.3
        
        return has_incompetence_content or is_poorly_structured
    
    def _lacks_integrity(self, action_content: Dict[str, Any]) -> bool:
        """Check if the action lacks integrity."""
        # If it demonstrates integrity, it doesn't lack it
        if self._demonstrates_integrity(action_content):
            return False
            
        # Look for integrity issue indicators in content
        content_str = str(action_content).lower()
        integrity_issue_indicators = [
            "mislead", "deceive", "hide", "conceal", "manipulate",
            "trick", "false", "fake", "dishonest", "unethical",
            "immoral", "unfair", "biased", "prejudiced"
        ]
        
        # Check content for integrity issue indicators
        has_integrity_issues = any(
            indicator in content_str for indicator in integrity_issue_indicators
        )
        
        return has_integrity_issues
    
    def _calculate_base_trust_impact(
        self,
        action_type: str,
        action_content: Dict[str, Any],
        action_category: ActionCategory
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the base trust impact of an action.
        
        Args:
            action_type: Type of action.
            action_content: Content details of the action.
            action_category: Category of the action.
            
        Returns:
            Tuple of (base_impact_score, impact_factors).
        """
        impact_factors = {}
        
        # Start with neutral impact
        base_impact = 0.0
        
        # Apply category-based factor
        category_factor = self.category_weights.get(action_category.value, 1.0)
        impact_factors["category_weight"] = category_factor
        
        # Check for trust-building elements
        if self._has_transparency(action_content):
            transparency_factor = 1.5
            base_impact += transparency_factor
            impact_factors["transparency"] = transparency_factor
        
        if self._shows_competence(action_content):
            competence_factor = 1.0
            base_impact += competence_factor
            impact_factors["competence"] = competence_factor
        
        if self._demonstrates_integrity(action_content):
            integrity_factor = 2.0
            base_impact += integrity_factor
            impact_factors["integrity"] = integrity_factor
        
        if self._shows_benevolence(action_content):
            benevolence_factor = 1.5
            base_impact += benevolence_factor
            impact_factors["benevolence"] = benevolence_factor
        
        # Check for trust-eroding elements
        if self._lacks_transparency(action_content):
            transparency_penalty = -2.0
            base_impact += transparency_penalty
            impact_factors["lack_of_transparency"] = transparency_penalty
        
        if self._shows_inconsistency(action_content):
            inconsistency_penalty = -1.5
            base_impact += inconsistency_penalty
            impact_factors["inconsistency"] = inconsistency_penalty
        
        if self._indicates_incompetence(action_content):
            incompetence_penalty = -1.0
            base_impact += incompetence_penalty
            impact_factors["incompetence"] = incompetence_penalty
        
        if self._lacks_integrity(action_content):
            integrity_penalty = -3.0
            base_impact += integrity_penalty
            impact_factors["lack_of_integrity"] = integrity_penalty
        
        # Apply specific action type factors
        if action_type == "access_request" and "sensitive" in str(action_content).lower():
            sensitive_request_factor = -1.0
            base_impact += sensitive_request_factor
            impact_factors["sensitive_request"] = sensitive_request_factor
        
        if action_type == "task_request" and "urgent" in str(action_content).lower():
            urgency_factor = -0.5
            base_impact += urgency_factor
            impact_factors["urgency"] = urgency_factor
        
        if action_type == "error_response":
            error_handling_factor = 1.0
            base_impact += error_handling_factor
            impact_factors["constructive_error_handling"] = error_handling_factor
        
        # Apply category weight
        base_impact *= category_factor
        
        return base_impact, impact_factors
    
    def _apply_relationship_modifiers(
        self,
        base_impact: float,
        relationship: AgentRelationship,
        action_category: ActionCategory,
        action_type: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply relationship-specific modifiers to trust impact.
        
        Args:
            base_impact: Base trust impact score.
            relationship: The agent relationship.
            action_category: Category of the action.
            action_type: Type of action.
            
        Returns:
            Tuple of (modified_impact, modifiers_dict).
        """
        modifiers = {}
        modified_impact = base_impact
        
        # Apply trust level modifier
        trust_level_modifier = 0.0
        if relationship.trust_level == TrustLevel.NONE:
            trust_level_modifier = -0.5  # High penalty for no trust
        elif relationship.trust_level == TrustLevel.MINIMAL:
            trust_level_modifier = -0.3  # Penalty for minimal trust
        elif relationship.trust_level == TrustLevel.HIGH:
            trust_level_modifier = 0.3  # Bonus for high trust
        elif relationship.trust_level == TrustLevel.COMPLETE:
            trust_level_modifier = 0.5  # High bonus for complete trust
            
        modified_impact += trust_level_modifier
        modifiers["trust_level"] = trust_level_modifier
        
        # Apply relationship status modifier
        status_modifier = 0.0
        if relationship.status == RelationshipStatus.DAMAGED:
            status_modifier = -1.0  # High penalty for damaged relationships
        elif relationship.status == RelationshipStatus.STRAINED:
            status_modifier = -0.7  # Penalty for strained relationships
        elif relationship.status == RelationshipStatus.CLOSE:
            status_modifier = 0.7  # Bonus for close relationships
        elif relationship.status == RelationshipStatus.ESSENTIAL:
            status_modifier = 1.0  # High bonus for essential relationships
            
        modified_impact += status_modifier
        modifiers["relationship_status"] = status_modifier
        
        # Check for recent trust breaches
        has_trust_breach = any(
            memory.memory_type == "trust_breach" 
            for memory in relationship.memories[-5:]  # Only check recent memories
        )
        
        if has_trust_breach:
            breach_modifier = -1.0
            modified_impact += breach_modifier
            modifiers["recent_trust_breach"] = breach_modifier
        
        # Check interaction history
        if relationship.interaction_count > 0:
            history_modifier = 0.0
            
            # New relationships are more sensitive to trust impacts
            if relationship.interaction_count < 5:
                if base_impact < 0:
                    history_modifier = -0.5  # Higher penalty for new relationships
                else:
                    history_modifier = 0.3  # Higher bonus for new relationships
            # Established relationships have more resilience
            elif relationship.interaction_count > 50:
                if base_impact < 0:
                    history_modifier = 0.3  # Reduced penalty for established relationships
                else:
                    history_modifier = 0.2  # Moderate bonus for established relationships
                    
            modified_impact += history_modifier
            modifiers["interaction_history"] = history_modifier
        
        # Apply category-specific relationship modifiers
        if action_category == ActionCategory.DATA_SHARING and relationship.trust_level.value < TrustLevel.MODERATE.value:
            sensitive_data_modifier = -0.5
            modified_impact += sensitive_data_modifier
            modifiers["sensitive_data_with_low_trust"] = sensitive_data_modifier
            
        if action_category == ActionCategory.ACCESS_REQUEST and relationship.trust_level.value < TrustLevel.HIGH.value:
            access_modifier = -0.7
            modified_impact += access_modifier
            modifiers["access_request_with_low_trust"] = access_modifier
        
        return modified_impact, modifiers
    
    def _calculate_principle_alignment(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculate principle alignment for an action.
        
        Args:
            action: The action to evaluate.
            context: Additional context for evaluation.
            
        Returns:
            Tuple of (alignment_score, impact_modifier, principle_modifiers).
        """
        if not self.principle_engine:
            return None, 0.0, {}
            
        # Convert action to a format for principle engine
        principle_action = {
            "id": action.get("id", str(datetime.now(timezone.utc).timestamp())),
            "method": "evaluateAction",
            "params": {
                "action": action,
                "context": context
            }
        }
        
        # Evaluate with principle engine
        evaluation = self.principle_engine.evaluate_message(principle_action)
        
        # Extract overall alignment
        alignment_score = evaluation.get("overall_score", 0.0) / 100.0  # Convert to 0-1 scale
        
        # Calculate impact modifier based on alignment
        impact_modifier = 0.0
        if alignment_score >= 0.8:
            impact_modifier = 2.0  # Strong bonus for high alignment
        elif alignment_score >= 0.6:
            impact_modifier = 1.0  # Moderate bonus for good alignment
        elif alignment_score <= 0.3:
            impact_modifier = -2.0  # Strong penalty for poor alignment
        elif alignment_score <= 0.5:
            impact_modifier = -1.0  # Moderate penalty for below-average alignment
        
        # Extract principle-specific modifiers
        principle_modifiers = {}
        for principle_id, data in evaluation.get("principle_scores", {}).items():
            score = data.get("score", 0)
            principle_impact = 0.0
            
            if score < 50:  # Below threshold for acceptable alignment
                principle_impact = -1.0
            elif score > 80:  # High alignment
                principle_impact = 0.5
                
            if principle_impact != 0:
                principle_modifiers[principle_id] = principle_impact
        
        return alignment_score, impact_modifier, principle_modifiers
    
    def _generate_trust_warnings(
        self,
        action: Dict[str, Any],
        impact_score: float,
        action_category: ActionCategory,
        relationship: Optional[AgentRelationship],
        impact_factors: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate warnings for trust-eroding actions.
        
        Args:
            action: The action being evaluated.
            impact_score: The calculated trust impact score.
            action_category: Category of the action.
            relationship: Optional relationship with the target agent.
            impact_factors: Dictionary of factors affecting trust impact.
            
        Returns:
            List of warning dictionaries.
        """
        warnings = []
        
        # Determine severity based on impact score
        severity = self._determine_severity(impact_score)
        
        # Generate general warning based on trust impact
        if impact_score <= self.severe_erosion_threshold:
            warnings.append({
                "severity": severity.value,
                "message": "This action may severely damage trust in the relationship",
                "impact_score": impact_score,
                "category": action_category.value
            })
        elif impact_score <= self.moderate_erosion_threshold:
            warnings.append({
                "severity": severity.value,
                "message": "This action may moderately damage trust in the relationship",
                "impact_score": impact_score,
                "category": action_category.value
            })
        
        # Generate specific warnings based on impact factors
        for factor, value in impact_factors.items():
            if value < 0:
                # Lack of transparency warning
                if factor == "lack_of_transparency":
                    warnings.append({
                        "severity": severity.value,
                        "message": "Action lacks transparency which may damage trust",
                        "impact_factor": factor,
                        "impact_value": value,
                        "recommendation": "Add clear explanations of purpose and intended outcome"
                    })
                # Inconsistency warning
                elif factor == "inconsistency":
                    warnings.append({
                        "severity": severity.value,
                        "message": "Action shows inconsistency with previous interactions",
                        "impact_factor": factor,
                        "impact_value": value,
                        "recommendation": "Maintain consistency or explain reason for deviation"
                    })
                # Incompetence warning
                elif factor == "incompetence":
                    warnings.append({
                        "severity": severity.value,
                        "message": "Action indicates potential incompetence that may reduce trust",
                        "impact_factor": factor,
                        "impact_value": value,
                        "recommendation": "Demonstrate appropriate expertise and capability"
                    })
                # Lack of integrity warning
                elif factor == "lack_of_integrity":
                    warnings.append({
                        "severity": severity.value,
                        "message": "Action raises integrity concerns which significantly harm trust",
                        "impact_factor": factor,
                        "impact_value": value,
                        "recommendation": "Ensure actions align with ethical principles and values"
                    })
                # Other negative factors
                else:
                    warnings.append({
                        "severity": severity.value,
                        "message": f"Action has negative trust impact due to {factor.replace('_', ' ')}",
                        "impact_factor": factor,
                        "impact_value": value,
                        "recommendation": "Consider alternative approaches with more positive trust impact"
                    })
        
        # Add relationship-specific warnings
        if relationship:
            if relationship.trust_level.value <= TrustLevel.MINIMAL.value:
                warnings.append({
                    "severity": severity.value,
                    "message": f"Low existing trust level ({relationship.trust_level.value}) increases risk of this action",
                    "impact_factor": "low_trust_relationship",
                    "recommendation": "Consider building trust through lower-risk actions first"
                })
            
            if relationship.status in [RelationshipStatus.DAMAGED, RelationshipStatus.STRAINED]:
                warnings.append({
                    "severity": severity.value,
                    "message": f"Relationship in {relationship.status.value} state increases sensitivity to actions",
                    "impact_factor": "damaged_relationship",
                    "recommendation": "Focus on trust repair actions before proceeding with this action"
                })
        
        return warnings
    
    def _determine_severity(self, impact_score: float) -> TrustImpactSeverity:
        """
        Determine warning severity based on impact score.
        
        Args:
            impact_score: Trust impact score.
            
        Returns:
            TrustImpactSeverity level.
        """
        if impact_score <= self.severe_erosion_threshold:
            return TrustImpactSeverity.CRITICAL
        elif impact_score <= self.moderate_erosion_threshold:
            return TrustImpactSeverity.HIGH
        elif impact_score < 0:
            return TrustImpactSeverity.MEDIUM
        elif impact_score < self.neutral_threshold:
            return TrustImpactSeverity.LOW
        else:
            return TrustImpactSeverity.NEGLIGIBLE
    
    def _suggest_alternatives(
        self,
        action: Dict[str, Any],
        action_category: ActionCategory,
        relationship: Optional[AgentRelationship],
        warnings: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest alternative approaches that would have more positive trust impact.
        
        Args:
            action: The original action.
            action_category: Category of the action.
            relationship: Optional relationship with the target agent.
            warnings: Generated warnings about the action.
            context: Additional context for suggestions.
            
        Returns:
            List of alternative suggestions.
        """
        alternatives = []
        action_type = action.get("type", "unknown")
        action_content = action.get("content", {})
        
        # Extract warning factors to address
        warning_factors = []
        for warning in warnings:
            if "impact_factor" in warning:
                warning_factors.append(warning["impact_factor"])
        
        # Check if we need to improve transparency
        if "lack_of_transparency" in warning_factors:
            transparent_alternative = self._create_transparent_alternative(action, action_type, action_content)
            if transparent_alternative:
                alternatives.append(transparent_alternative)
        
        # Check if we need to address inconsistency
        if "inconsistency" in warning_factors:
            consistent_alternative = self._create_consistent_alternative(action, relationship)
            if consistent_alternative:
                alternatives.append(consistent_alternative)
        
        # Check if we need to demonstrate competence
        if "incompetence" in warning_factors:
            competent_alternative = self._create_competent_alternative(action, action_type, action_content)
            if competent_alternative:
                alternatives.append(competent_alternative)
        
        # Check if we need to improve integrity
        if "lack_of_integrity" in warning_factors:
            integrity_alternative = self._create_integrity_alternative(action)
            if integrity_alternative:
                alternatives.append(integrity_alternative)
        
        # Add category-specific alternatives
        if action_category == ActionCategory.DATA_SHARING:
            alternatives.append(self._create_data_sharing_alternative(action, relationship))
        elif action_category == ActionCategory.ACCESS_REQUEST:
            alternatives.append(self._create_access_request_alternative(action, relationship))
        elif action_category == ActionCategory.COMMUNICATION:
            alternatives.append(self._create_communication_alternative(action, relationship))
        
        # If no specific alternatives were created, add a general alternative
        if not alternatives:
            alternatives.append({
                "description": "Use a more collaborative approach",
                "strategy": TrustBuildingStrategy.BENEVOLENCE.value,
                "changes": [
                    "Frame the action in terms of mutual benefit",
                    "Highlight how this helps the other agent",
                    "Ask for feedback on the proposed action"
                ],
                "expected_trust_impact": "Positive",
                "rationale": "Demonstrating care for the other agent's interests builds trust"
            })
        
        return alternatives
    
    def _create_transparent_alternative(
        self,
        action: Dict[str, Any],
        action_type: str,
        action_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create alternative with improved transparency."""
        # Create new transparent content
        transparent_content = action_content.copy() if isinstance(action_content, dict) else {}
        
        # Add purpose field if missing
        if "purpose" not in transparent_content:
            transparent_content["purpose"] = f"The purpose of this {action_type} is to achieve [goal] while maintaining trust"
        
        # Add explanation field if missing
        if "explanation" not in transparent_content:
            transparent_content["explanation"] = f"This action is necessary because [specific reason], and was chosen because [decision rationale]"
        
        # Add expected_outcome field if missing
        if "expected_outcome" not in transparent_content:
            transparent_content["expected_outcome"] = "The expected outcome is [specific benefit], which helps [how it aligns with shared goals]"
        
        return {
            "description": "Increase transparency of the action",
            "strategy": TrustBuildingStrategy.TRANSPARENCY.value,
            "content_changes": transparent_content,
            "changes": [
                "Add clear explanation of purpose",
                "Clarify reasoning behind the action",
                "Describe expected outcomes",
                "Explain how this aligns with shared goals"
            ],
            "expected_trust_impact": "Positive",
            "rationale": "Being transparent about intentions and reasoning builds trust"
        }
    
    def _create_consistent_alternative(
        self,
        action: Dict[str, Any],
        relationship: Optional[AgentRelationship]
    ) -> Dict[str, Any]:
        """Create alternative with improved consistency."""
        # Extract relationship history if available
        previous_patterns = []
        if relationship and relationship.interaction_count > 0:
            # Extract pattern from previous interactions
            for memory in relationship.memories:
                if memory.memory_type == "interaction_pattern":
                    previous_patterns.append(memory.content)
        
        return {
            "description": "Ensure consistency with previous interactions",
            "strategy": TrustBuildingStrategy.PREDICTABILITY.value,
            "changes": [
                "Align with established interaction patterns" if previous_patterns else "Establish a clear interaction pattern",
                "Explain any deviations from previous patterns",
                "Reference previous similar interactions",
                "Maintain consistent communication style and frequency"
            ],
            "expected_trust_impact": "Positive",
            "rationale": "Consistency and predictability are foundations of trust"
        }
    
    def _create_competent_alternative(
        self,
        action: Dict[str, Any],
        action_type: str,
        action_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create alternative demonstrating competence."""
        # Create content with competence indicators
        competent_content = action_content.copy() if isinstance(action_content, dict) else {}
        
        # Add expertise field if missing
        if "expertise" not in competent_content:
            competent_content["expertise"] = "This action leverages expertise in [relevant domain]"
        
        # Add reasoning field if missing
        if "reasoning" not in competent_content:
            competent_content["reasoning"] = "The approach was selected based on [analytical reasoning]"
        
        return {
            "description": "Demonstrate relevant expertise and competence",
            "strategy": TrustBuildingStrategy.COMPETENCE.value,
            "content_changes": competent_content,
            "changes": [
                "Showcase relevant expertise",
                "Provide well-structured, comprehensive information",
                "Reference established methodologies or best practices",
                "Include relevant details that demonstrate domain knowledge"
            ],
            "expected_trust_impact": "Positive",
            "rationale": "Demonstrating competence builds confidence and trust"
        }
    
    def _create_integrity_alternative(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create alternative with improved integrity."""
        return {
            "description": "Emphasize ethical alignment and integrity",
            "strategy": TrustBuildingStrategy.INTEGRITY.value,
            "changes": [
                "Reference applicable ethical principles",
                "Highlight commitment to ethical standards",
                "Ensure complete honesty and accuracy",
                "Acknowledge any potential conflicts of interest"
            ],
            "expected_trust_impact": "Very Positive",
            "rationale": "Demonstrating strong ethical principles and integrity builds deep trust"
        }
    
    def _create_data_sharing_alternative(
        self,
        action: Dict[str, Any],
        relationship: Optional[AgentRelationship]
    ) -> Dict[str, Any]:
        """Create alternative for data sharing actions."""
        # Check relationship trust level
        low_trust = not relationship or relationship.trust_level.value < TrustLevel.MODERATE.value
        
        if low_trust:
            return {
                "description": "Incrementally share data with privacy safeguards",
                "strategy": TrustBuildingStrategy.RECIPROCITY.value,
                "changes": [
                    "Break down data sharing into smaller increments",
                    "Start with less sensitive information",
                    "Include additional privacy safeguards",
                    "Request reciprocal but proportional information sharing",
                    "Provide opt-out options for specific data categories"
                ],
                "expected_trust_impact": "Positive",
                "rationale": "Incremental data sharing with safeguards respects trust boundaries while building confidence"
            }
        else:
            return {
                "description": "Transparent data sharing with clear boundaries",
                "strategy": TrustBuildingStrategy.ACCOUNTABILITY.value,
                "changes": [
                    "Clearly document all data being shared",
                    "Explain specific purpose for each data category",
                    "Outline data handling and retention policies",
                    "Provide usage reporting and audit capabilities",
                    "Establish clear boundaries for data use"
                ],
                "expected_trust_impact": "Positive",
                "rationale": "Transparent boundaries and accountability measures maintain high trust during data sharing"
            }
    
    def _create_access_request_alternative(
        self,
        action: Dict[str, Any],
        relationship: Optional[AgentRelationship]
    ) -> Dict[str, Any]:
        """Create alternative for access request actions."""
        # Check relationship trust level
        low_trust = not relationship or relationship.trust_level.value < TrustLevel.HIGH.value
        
        if low_trust:
            return {
                "description": "Request limited access with trust-building steps",
                "strategy": TrustBuildingStrategy.ACCOUNTABILITY.value,
                "changes": [
                    "Request minimal access needed for immediate task",
                    "Add time limitations to access period",
                    "Offer monitoring/auditing of access usage",
                    "Suggest pilot phase with evaluation",
                    "Propose stepwise increase in access level"
                ],
                "expected_trust_impact": "Moderate Positive",
                "rationale": "Limited, accountable access requests respect trust boundaries while providing value"
            }
        else:
            return {
                "description": "Structured access request with clear safeguards",
                "strategy": TrustBuildingStrategy.INTEGRITY.value,
                "changes": [
                    "Provide detailed rationale for each access level requested",
                    "Suggest appropriate limitations and safeguards",
                    "Reference previous successful access usage",
                    "Include integrity commitments for access usage"
                ],
                "expected_trust_impact": "Positive",
                "rationale": "Well-structured access requests with safeguards respect and reinforce trust"
            }
    
    def _create_communication_alternative(
        self,
        action: Dict[str, Any],
        relationship: Optional[AgentRelationship]
    ) -> Dict[str, Any]:
        """Create alternative for communication actions."""
        # Determine appropriate communication style based on relationship
        if relationship and relationship.status == RelationshipStatus.CLOSE:
            style = "warm and collaborative"
            elements = [
                "Use friendly, personalized tone",
                "Reference shared history and successes",
                "Incorporate appropriate positive sentiment",
                "Frame in terms of continuing successful collaboration"
            ]
        elif relationship and relationship.status == RelationshipStatus.DAMAGED:
            style = "rebuilding trust"
            elements = [
                "Acknowledge previous issues without defensiveness",
                "Show understanding of concerns",
                "Focus on concrete, verifiable commitments",
                "Suggest incremental trust-building steps"
            ]
        else:
            style = "clear and professional"
            elements = [
                "Use straightforward, unambiguous language",
                "Provide complete relevant information",
                "Maintain professional, respectful tone",
                "Structure message logically with clear headings"
            ]
            
        return {
            "description": f"Communication with {style} approach",
            "strategy": TrustBuildingStrategy.TRANSPARENCY.value,
            "changes": elements,
            "expected_trust_impact": "Positive",
            "rationale": "Communication style aligned with relationship status builds appropriate trust"
        }