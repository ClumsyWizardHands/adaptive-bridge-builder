#!/usr/bin/env python3
"""
Authenticity Verifier Module

This module provides functions to verify the authenticity of actions by ensuring
they are consistent with core programming, stated principles, and historical patterns.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AuthenticityVerifier")

@dataclass
class AuthenticityWarning:
    """
    Represents a warning about potential authenticity issues.
    """
    warning_id: str
    severity: float  # 0.0 to 1.0, with 1.0 being most severe
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

@dataclass
class AuthenticityResult:
    """
    Represents the result of an authenticity verification.
    """
    is_authentic: bool
    confidence: float  # 0.0 to 1.0
    warnings: List[AuthenticityWarning] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

def verify_authenticity(
    action: Dict[str, Any],
    context: Dict[str, Any] = None,
    principle_engine: Optional[PrincipleEngine] = None,
    historical_actions: Optional[List[Dict[str, Any]]] = None
) -> AuthenticityResult:
    """
    Verifies that an action is authentic and consistent with core programming and principles.
    
    Args:
        action: The action to verify
        context: Additional context for verification
        principle_engine: Optional PrincipleEngine instance
        historical_actions: Optional list of historical actions for pattern comparison
        
    Returns:
        AuthenticityResult containing verification result and any warnings
    """
    if context is None:
        context = {}
    
    if historical_actions is None:
        historical_actions = []
    
    # Initialize result with default values
    result = AuthenticityResult(
        is_authentic=True,  # Start with assumption of authenticity
        confidence=0.5,     # Default middle confidence
        warnings=[],
        details={}
    )
    
    # Collect warnings from different verification steps
    warnings = []
    
    # 1. Verify consistency with core programming
    core_programming_warnings = _verify_core_programming_consistency(action, context)
    warnings.extend(core_programming_warnings)
    
    # 2. Verify consistency with stated principles
    principle_warnings = _verify_principle_consistency(action, context, principle_engine)
    warnings.extend(principle_warnings)
    
    # 3. Verify consistency with historical patterns
    history_warnings = _verify_historical_consistency(action, historical_actions)
    warnings.extend(history_warnings)
    
    # 4. Determine overall authenticity based on warnings
    high_severity_warnings = [w for w in warnings if w.severity >= 0.7]
    medium_severity_warnings = [w for w in warnings if 0.4 <= w.severity < 0.7]
    
    # Calculate confidence based on warnings
    confidence_reduction = sum(w.severity for w in warnings) / 10.0  # Divide by 10 to normalize impact
    confidence = max(0.0, 1.0 - confidence_reduction)
    
    # Determine authenticity - fail if there are any high severity warnings or multiple medium ones
    is_authentic = len(high_severity_warnings) == 0 and len(medium_severity_warnings) <= 1
    
    # Update result
    result.is_authentic = is_authentic
    result.confidence = confidence
    result.warnings = warnings
    result.details = {
        "high_severity_warnings_count": len(high_severity_warnings),
        "medium_severity_warnings_count": len(medium_severity_warnings),
        "total_warnings_count": len(warnings),
        "principle_consistency_score": _calculate_principle_consistency(action, principle_engine) if principle_engine else None,
        "historical_pattern_score": _calculate_historical_pattern_consistency(action, historical_actions) if historical_actions else None
    }
    
    return result

def _extract_action_content(action: Dict[str, Any]) -> Optional[str]:
    """
    Extract content from an action for analysis.
    
    Args:
        action: The action to extract content from
        
    Returns:
        Extracted action content as string, or None if not extractable
    """
    # Extract from params field first
    params = action.get("params", {})
    if isinstance(params, dict):
        # Check common content fields
        for field in ["text", "content", "message", "data", "body"]:
            if field in params and isinstance(params[field], str):
                return params[field]
        
        # If data is a dict, convert to string
        if "data" in params and isinstance(params["data"], dict):
            return json.dumps(params["data"])
    
    # Check result field for responses
    result = action.get("result")
    if result:
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return json.dumps(result)
    
    # Check error field for error messages
    error = action.get("error")
    if error:
        if isinstance(error, str):
            return error
        elif isinstance(error, dict) and "message" in error:
            return error["message"]
    
    # Try to extract from the action itself if it has text
    for field in ["text", "content", "message", "body"]:
        if field in action and isinstance(action[field], str):
            return action[field]
    
    # If no extractable content found
    return None

def _extract_data_fields(action: Dict[str, Any]) -> List[str]:
    """
    Extract data field names from an action for analysis.
    
    Args:
        action: The action to extract data fields from
        
    Returns:
        List of data field names
    """
    fields = []
    
    # Extract from params field first
    params = action.get("params", {})
    if isinstance(params, dict):
        fields.extend(params.keys())
        
        # If params contains nested data structures, extract their fields too
        for key, value in params.items():
            if isinstance(value, dict):
                # Add nested fields with dot notation (e.g., user.name)
                fields.extend([f"{key}.{k}" for k in value.keys()])
    
    # Check for data fields in result
    result = action.get("result")
    if isinstance(result, dict):
        fields.extend([f"result.{k}" for k in result.keys()])
    
    # Check for data fields in metadata
    metadata = action.get("metadata")
    if isinstance(metadata, dict):
        fields.extend([f"metadata.{k}" for k in metadata.keys()])
    
    return fields

def _verify_core_programming_consistency(
    action: Dict[str, Any],
    context: Dict[str, Any]
) -> List[AuthenticityWarning]:
    """
    Verifies that an action is consistent with core programming.
    
    Args:
        action: The action to verify
        context: Additional context for verification
        
    Returns:
        List of warnings about inconsistencies with core programming
    """
    warnings = []
    
    # Define core programming constraints
    core_constraints = [
        {
            "id": "user_safety_first",
            "check": lambda a, c: _check_action_safety(a, c),
            "message": "Action may violate user safety constraints",
            "severity": 0.9,
            "recommendation": "Modify action to prioritize user safety"
        },
        {
            "id": "data_privacy",
            "check": lambda a, c: _check_data_privacy(a, c),
            "message": "Action may compromise data privacy",
            "severity": 0.8,
            "recommendation": "Ensure action adheres to privacy principles and data handling protocols"
        },
        {
            "id": "aligned_with_directives",
            "check": lambda a, c: _check_directive_alignment(a, c),
            "message": "Action may deviate from core directives",
            "severity": 0.7,
            "recommendation": "Review action against core directives and ensure alignment"
        },
        {
            "id": "scope_appropriate",
            "check": lambda a, c: _check_scope_appropriateness(a, c),
            "message": "Action may exceed appropriate scope",
            "severity": 0.6,
            "recommendation": "Limit action to stay within authorized boundaries"
        }
    ]
    
    # Check each constraint
    for constraint in core_constraints:
        check_result = constraint["check"](action, context)
        if not check_result["passes"]:
            warning = AuthenticityWarning(
                warning_id=constraint["id"],
                severity=constraint["severity"],
                message=constraint["message"],
                details=check_result.get("details", {}),
                recommendation=constraint["recommendation"]
            )
            warnings.append(warning)
    
    return warnings

def _check_action_safety(action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Checks if an action is safe for the user."""
    # Extract action type and content
    action_type = action.get("method", "unknown")
    content = _extract_action_content(action)
    
    # Define safety red flags
    safety_red_flags = [
        "harm", "damage", "danger", "risk", "unsafe", "exploit", "vulnerability",
        "attack", "compromise", "breach", "bypass", "circumvent"
    ]
    
    # Check for red flags in content
    if content:
        found_flags = [flag for flag in safety_red_flags if flag in content.lower()]
        if found_flags:
            return {
                "passes": False,
                "details": {
                    "action_type": action_type,
                    "found_flags": found_flags,
                    "explanation": "Action contains potentially unsafe language or concepts"
                }
            }
    
    # Check for risky action types
    risky_action_types = ["system_modify", "security_override", "access_escalate"]
    if action_type in risky_action_types:
        return {
            "passes": False,
            "details": {
                "action_type": action_type,
                "explanation": "Action type is inherently risky and requires additional verification"
            }
        }
    
    return {"passes": True}

def _check_data_privacy(action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Checks if an action respects data privacy."""
    # Extract data handling aspects
    data_fields = _extract_data_fields(action)
    access_scope = action.get("scope", "default")
    
    # Check for sensitive data handling
    sensitive_fields = ["password", "credentials", "token", "key", "secret", "ssn", 
                        "social_security", "credit_card", "banking", "financial", "health"]
    
    found_sensitive = [field for field in data_fields if any(s in field.lower() for s in sensitive_fields)]
    
    if found_sensitive:
        # Verify appropriate handling based on context
        authorized_access = context.get("authorized_sensitive_access", False)
        appropriate_scope = access_scope in ["secure", "encrypted", "authenticated"]
        
        if not (authorized_access and appropriate_scope):
            return {
                "passes": False,
                "details": {
                    "sensitive_fields": found_sensitive,
                    "access_scope": access_scope,
                    "authorized_access": authorized_access,
                    "explanation": "Action involves sensitive data without appropriate authorization or handling"
                }
            }
    
    return {"passes": True}

def _check_directive_alignment(action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Checks if an action aligns with core directives."""
    # Extract action intent and core directives
    action_intent = action.get("intent", "unknown")
    core_directives = context.get("core_directives", [])
    
    if not core_directives:
        # Default directives if none provided
        core_directives = [
            "assist_user", "provide_information", "respect_boundaries",
            "ensure_accuracy", "maintain_neutrality"
        ]
    
    # Check action intent against directives
    aligned_directives = []
    misaligned_directives = []
    
    for directive in core_directives:
        if _is_aligned_with_directive(action_intent, directive):
            aligned_directives.append(directive)
        else:
            misaligned_directives.append(directive)
    
    if misaligned_directives and len(misaligned_directives) > len(aligned_directives):
        return {
            "passes": False,
            "details": {
                "action_intent": action_intent,
                "aligned_directives": aligned_directives,
                "misaligned_directives": misaligned_directives,
                "explanation": "Action intent misaligns with more directives than it aligns with"
            }
        }
    
    return {"passes": True}

def _is_aligned_with_directive(intent: str, directive: str) -> bool:
    """Check if an intent aligns with a directive (simplified)."""
    # In a real implementation, this would be more sophisticated
    related_terms = {
        "assist_user": ["help", "assist", "support", "aid"],
        "provide_information": ["inform", "explain", "describe", "detail", "clarify"],
        "respect_boundaries": ["respect", "boundary", "limit", "constrain"],
        "ensure_accuracy": ["accurate", "correct", "precise", "factual", "verify"],
        "maintain_neutrality": ["neutral", "balanced", "impartial", "objective"]
    }
    
    if directive in related_terms:
        return any(term in intent.lower() for term in related_terms[directive])
    
    # Default to assumption of alignment if we don't have specific criteria
    return True

def _check_scope_appropriateness(action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Checks if an action's scope is appropriate."""
    # Extract action scope and authorized scopes
    action_scope = action.get("scope", "default")
    authorized_scopes = context.get("authorized_scopes", ["default", "standard", "basic"])
    
    # Check if action scope is authorized
    if action_scope not in authorized_scopes:
        return {
            "passes": False,
            "details": {
                "action_scope": action_scope,
                "authorized_scopes": authorized_scopes,
                "explanation": "Action scope exceeds authorized boundaries"
            }
        }
    
    # Check for scope escalation patterns
    if _has_scope_escalation_pattern(action):
        return {
            "passes": False,
            "details": {
                "action_scope": action_scope,
                "explanation": "Action contains patterns suggesting inappropriate scope escalation"
            }
        }
    
    return {"passes": True}

def _has_scope_escalation_pattern(action: Dict[str, Any]) -> bool:
    """Check for patterns suggesting scope escalation (simplified)."""
    # Extract content
    content = _extract_action_content(action)
    if not content:
        return False
    
    # Check for scope escalation indicators
    escalation_indicators = [
        "elevate privileges", "gain access", "bypass", "override", 
        "circumvent", "disable security", "ignore restrictions"
    ]
    
    return any(indicator in content.lower() for indicator in escalation_indicators)

def _verify_principle_consistency(
    action: Dict[str, Any],
    context: Dict[str, Any],
    principle_engine: Optional[PrincipleEngine]
) -> List[AuthenticityWarning]:
    """
    Verifies that an action is consistent with stated principles.
    
    Args:
        action: The action to verify
        context: Additional context for verification
        principle_engine: PrincipleEngine instance for principle checks
        
    Returns:
        List of warnings about inconsistencies with principles
    """
    warnings = []
    
    # If no principle engine is provided, we can't check principle consistency
    if principle_engine is None:
        return warnings
    
    # Get applicable principles for this action
    applicable_principles = principle_engine.get_applicable_principles(action, context)
    
    # For each principle, check if action is consistent
    for principle in applicable_principles:
        principle_id = principle.get("id", "unknown")
        principle_weight = principle.get("weight", 0.5)
        
        # Evaluate action against the principle
        evaluation = principle_engine.evaluate_action_against_principle(action, principle, context)
        
        # If evaluation is below threshold, add a warning
        if evaluation < 0.6:  # Threshold for inconsistency
            warning = AuthenticityWarning(
                warning_id=f"principle_{principle_id}",
                severity=principle_weight * 0.8,  # Scale severity based on principle weight
                message=f"Action inconsistent with principle: {principle_id}",
                details={
                    "principle": principle,
                    "evaluation_score": evaluation,
                    "explanation": "Action's alignment with this principle is below acceptable threshold"
                },
                recommendation=f"Revise action to better align with the {principle_id} principle"
            )
            warnings.append(warning)
    
    return warnings

def _calculate_principle_consistency(
    action: Dict[str, Any],
    principle_engine: Optional[PrincipleEngine]
) -> Optional[float]:
    """Calculate overall principle consistency score."""
    if principle_engine is None:
        return None
    
    # This is a simplified version - in reality, we would use more sophisticated scoring
    principles = principle_engine.get_all_principles()
    if not principles:
        return None
    
    scores = []
    for principle in principles:
        score = principle_engine.evaluate_action_against_principle(action, principle, {})
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else None

def _verify_historical_consistency(
    action: Dict[str, Any],
    historical_actions: List[Dict[str, Any]]
) -> List[AuthenticityWarning]:
    """
    Verifies that an action is consistent with historical patterns.
    
    Args:
        action: The action to verify
        historical_actions: List of historical actions for comparison
        
    Returns:
        List of warnings about inconsistencies with historical patterns
    """
    warnings = []
    
    # If no historical actions are provided, we can't check historical consistency
    if not historical_actions:
        return warnings
    
    # Find similar actions in history
    similar_actions = _find_similar_actions(action, historical_actions)
    
    # If we can't find similar actions, this might be a deviation
    if not similar_actions and len(historical_actions) > 10:  # Only flag if we have enough history
        warning = AuthenticityWarning(
            warning_id="no_historical_precedent",
            severity=0.5,  # Medium severity - unusual but not necessarily bad
            message="Action has no precedent in historical patterns",
            details={
                "action_type": action.get("method", "unknown"),
                "explanation": "This type of action has not been observed in historical data"
            },
            recommendation="Verify action is appropriate given the lack of historical precedent"
        )
        warnings.append(warning)
        return warnings
    
    # Check for parameter pattern deviations
    parameter_deviations = _check_parameter_pattern_deviations(action, similar_actions)
    if parameter_deviations:
        warning = AuthenticityWarning(
            warning_id="parameter_pattern_deviation",
            severity=0.6,
            message="Action parameters deviate from historical patterns",
            details={
                "deviations": parameter_deviations,
                "explanation": "Action uses parameters in a way that differs from historical patterns"
            },
            recommendation="Review parameter usage for consistency with established patterns"
        )
        warnings.append(warning)
    
    # Check for outcome expectation deviations
    outcome_deviations = _check_outcome_expectation_deviations(action, similar_actions)
    if outcome_deviations:
        warning = AuthenticityWarning(
            warning_id="outcome_expectation_deviation",
            severity=0.7,
            message="Action outcome expectations deviate from historical patterns",
            details={
                "deviations": outcome_deviations,
                "explanation": "Action suggests outcomes that differ from historical patterns"
            },
            recommendation="Review expected outcomes for consistency with established patterns"
        )
        warnings.append(warning)
    
    # Check for contextual appropriateness deviations
    context_deviations = _check_contextual_appropriateness(action, similar_actions)
    if context_deviations:
        warning = AuthenticityWarning(
            warning_id="contextual_appropriateness_deviation",
            severity=0.65,
            message="Action context usage deviates from historical patterns",
            details={
                "deviations": context_deviations,
                "explanation": "Action is used in a context that differs from historical patterns"
            },
            recommendation="Review contextual usage for consistency with established patterns"
        )
        warnings.append(warning)
    
    return warnings

def _find_similar_actions(
    action: Dict[str, Any],
    historical_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Find actions in history that are similar to the given action."""
    # Extract key aspects of the action
    action_method = action.get("method", "unknown")
    action_params = action.get("params", {})
    
    # Find actions with the same method
    method_matches = [a for a in historical_actions if a.get("method", "unknown") == action_method]
    
    # If no method matches, return empty list
    if not method_matches:
        return []
    
    # Score similarity based on parameters
    similar_actions = []
    
    for hist_action in method_matches:
        hist_params = hist_action.get("params", {})
        
        # Calculate parameter similarity
        param_similarity = _calculate_parameter_similarity(action_params, hist_params)
        
        # If similarity is above threshold, add to similar actions
        if param_similarity > 0.7:  # 70% similarity threshold
            similar_action = hist_action.copy()
            similar_action["similarity"] = param_similarity
            similar_actions.append(similar_action)
    
    # Sort by similarity (most similar first)
    similar_actions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    return similar_actions

def _calculate_parameter_similarity(params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
    """Calculate similarity between two parameter sets."""
    # Get all parameter keys
    all_keys = set(params1.keys()) | set(params2.keys())
    if not all_keys:
        return 1.0  # Both empty means identical
    
    # Count matches
    matches = 0
    for key in all_keys:
        if key in params1 and key in params2:
            # If both have the parameter, check value similarity
            if params1[key] == params2[key]:
                matches += 1
            elif isinstance(params1[key], (str, int, float)) and isinstance(params2[key], (str, int, float)):
                # For simple types, check for partial match
                if str(params1[key]) in str(params2[key]) or str(params2[key]) in str(params1[key]):
                    matches += 0.5  # Partial match
    
    # Calculate similarity as proportion of matches
    return matches / len(all_keys)

def _check_parameter_pattern_deviations(
    action: Dict[str, Any],
    similar_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check for deviations in parameter usage patterns."""
    deviations = []
    
    action_params = action.get("params", {})
    
    # If no similar actions or no params, no deviations
    if not similar_actions or not action_params:
        return deviations
    
    # Track parameter statistics in similar actions
    param_stats = {}
    for similar_action in similar_actions:
        sim_params = similar_action.get("params", {})
        for key, value in sim_params.items():
            if key not in param_stats:
                param_stats[key] = {"values": [], "count": 0}
            
            param_stats[key]["values"].append(value)
            param_stats[key]["count"] += 1
    
    # Check for deviations in current action params
    for key, value in action_params.items():
        # Check if parameter is usually present
        if key in param_stats:
            # Check if value is within typical range
            typical_values = param_stats[key]["values"]
            
            if isinstance(value, (int, float)) and all(isinstance(v, (int, float)) for v in typical_values):
                # For numeric parameters, check if value is within range
                min_val = min(typical_values)
                max_val = max(typical_values)
                
                if value < min_val * 0.5 or value > max_val * 1.5:
                    deviations.append({
                        "parameter": key,
                        "current_value": value,
                        "typical_range": [min_val, max_val],
                        "explanation": "Parameter value is outside typical range"
                    })
            elif isinstance(value, str) and all(isinstance(v, str) for v in typical_values):
                # For string parameters, check if value is within typical length range
                lengths = [len(v) for v in typical_values]
                min_len = min(lengths)
                max_len = max(lengths)
                
                if len(value) < min_len * 0.5 or len(value) > max_len * 1.5:
                    deviations.append({
                        "parameter": key,
                        "current_value_length": len(value),
                        "typical_length_range": [min_len, max_len],
                        "explanation": "Parameter value length is outside typical range"
                    })
        else:
            # Parameter not typically present
            deviations.append({
                "parameter": key,
                "explanation": "Parameter not typically used in similar actions"
            })
    
    # Check for missing parameters that are typically present
    for key, stats in param_stats.items():
        frequency = stats["count"] / len(similar_actions)
        if frequency > 0.8 and key not in action_params:  # Parameter present in 80%+ of similar actions
            deviations.append({
                "parameter": key,
                "explanation": f"Parameter is typically present (in {frequency:.0%} of similar actions) but missing in current action"
            })
    
    return deviations

def _check_outcome_expectation_deviations(
    action: Dict[str, Any],
    similar_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check for deviations in expected outcomes."""
    deviations = []
    
    # Extract expected outcome from action
    expected_outcome = action.get("expected_outcome", action.get("result", {}))
    
    # If no expected outcome or no similar actions, no deviations
    if not expected_outcome or not similar_actions:
        return deviations
    
    # Track outcome patterns in similar actions
    outcome_patterns = []
    for similar_action in similar_actions:
        sim_outcome = similar_action.get("expected_outcome", similar_action.get("result", {}))
        if sim_outcome:
            outcome_patterns.append(sim_outcome)
    
    # If no outcome patterns, no deviations
    if not outcome_patterns:
        return deviations
    
    # Check for deviations in structure
    typical_keys = set()
    for pattern in outcome_patterns:
        if isinstance(pattern, dict):
            typical_keys.update(pattern.keys())
    
    if isinstance(expected_outcome, dict):
        # Check for missing keys
        missing_keys = typical_keys - set(expected_outcome.keys())
        if missing_keys and len(missing_keys) > len(typical_keys) * 0.3:  # Missing >30% of typical keys
            deviations.append({
                "type": "structure",
                "missing_keys": list(missing_keys),
                "explanation": "Expected outcome is missing keys typically present in similar actions"
            })
        
        # Check for extra keys
        extra_keys = set(expected_outcome.keys()) - typical_keys
        if extra_keys and len(extra_keys) > len(expected_outcome.keys()) * 0.3:  # >30% of keys are atypical
            deviations.append({
                "type": "structure",
                "extra_keys": list(extra_keys),
                "explanation": "Expected outcome contains keys not typically present in similar actions"
            })
    
    # Check for deviations in content (simplified)
    outcome_str = json.dumps(expected_outcome) if isinstance(expected_outcome, (dict, list)) else str(expected_outcome)
    unusual_terms = _find_unusual_terms(outcome_str, outcome_patterns)
    
    if unusual_terms:
        deviations.append({
            "type": "content",
            "unusual_terms": unusual_terms,
            "explanation": "Expected outcome contains terms not typically present in similar actions"
        })
    
    return deviations

def _find_unusual_terms(content: str, patterns: List[Any]) -> List[str]:
    """Find terms in content that are unusual compared to patterns."""
    # Convert patterns to strings
    pattern_strs = [json.dumps(p) if isinstance(p, (dict, list)) else str(p) for p in patterns]
    
    # Extract words from content
    content_words = set(w.lower() for w in content.split() if len(w) > 3)
    
    # Extract words from patterns
    pattern_words = set()
    for p_str in pattern_strs:
        pattern_words.update(w.lower() for w in p_str.split() if len(w) > 3)
    
    # Find unusual words (in content but not in patterns)
    unusual_words = content_words - pattern_words
    
    return list(unusual_words)

def _check_contextual_appropriateness(
    action: Dict[str, Any],
    similar_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check for deviations in contextual usage."""
    deviations = []
    
    # Extract action context
    action_context_ref = action.get("context_ref", {})
    
    # If no context or no similar actions, no deviations