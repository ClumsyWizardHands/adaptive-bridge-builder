#!/usr/bin/env python3
"""
Fairness Evaluator Module

This module provides functionality to evaluate fairness across different dimensions
and generate appropriate alternatives when bias or unfairness is detected.
"""

import logging
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FairnessEvaluator")

@dataclass
class FairnessMetric:
    """
    Represents a fairness measurement across a specific dimension.
    """
    dimension: str  # The dimension being measured (e.g., "language_bias", "perspective_diversity")
    score: float  # Score between 0.0 (unfair) and 1.0 (fair)
    confidence: float  # Confidence in the measurement (0.0-1.0)
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details about the measurement

@dataclass
class FairnessFlag:
    """
    Represents a specific fairness issue identified in content.
    """
    id: str  # Unique identifier for the flag
    type: str  # Type of fairness issue (e.g., "language_bias", "assumption_bias")
    description: str  # Human-readable description of the issue
    severity: float  # Severity of the issue (0.0-1.0)
    affected_groups: List[str]  # Groups potentially affected by the issue
    content_reference: Dict[str, Any]  # Reference to problematic content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata about the flag

@dataclass
class FairnessAlternative:
    """
    Represents an alternative approach that addresses a fairness issue.
    """
    id: str  # Unique identifier for the alternative
    flag_id: str  # ID of the flag this alternative addresses
    description: str  # Human-readable description of the alternative
    replacement_content: Optional[str]  # Suggested replacement content
    fairness_improvement: float  # Estimated improvement in fairness (0.0-1.0)
    impact_assessment: Dict[str, Any]  # Assessment of impact on other dimensions
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

class FairnessEvaluator:
    """
    Evaluates fairness across different dimensions and generates alternatives.
    """
    
    def __init__(self, principle_engine: Optional[PrincipleEngine] = None) -> None:
        """
        Initialize the FairnessEvaluator.
        
        Args:
            principle_engine: Optional PrincipleEngine instance for principle-based evaluation
        """
        self.principle_engine = principle_engine
        self.historical_actions = []  # History of previous actions to check for consistency
        
    def evaluate_message(
        self, 
        message: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Tuple[List[FairnessMetric], List[FairnessFlag]]:
        """
        Evaluate fairness of a message across multiple dimensions.
        
        Args:
            message: The message to evaluate
            context: Additional context for evaluation
            
        Returns:
            Tuple of (metrics, flags) where metrics are fairness measurements
            and flags are specific fairness issues identified
        """
        # Extract content for analysis
        from fairness_evaluator_implementations import _extract_message_content
        content = _extract_message_content(message)
        if not content:
            return [], []
        
        # Get applicable principles if principle engine is available
        principles = []
        if self.principle_engine:
            principles = self.principle_engine.get_applicable_principles(message, context or {})
        
        # Initialize metrics and flags
        metrics = []
        flags = []
        
        # Check decision consistency with historical actions
        from fairness_evaluator_implementations import _find_similar_actions, _check_decision_consistency
        similar_actions = _find_similar_actions(message, self.historical_actions)
        if similar_actions:
            consistency_issues = _check_decision_consistency(message, similar_actions, principles)
            for issue in consistency_issues:
                flag = FairnessFlag(
                    id=str(uuid.uuid4()),
                    type="decision_consistency",
                    description=issue["description"],
                    severity=issue["severity"],
                    affected_groups=["consistent treatment"],
                    content_reference={
                        "original": content,
                        "problematic_section": f"{issue.get('current_decision')} vs. {issue.get('historical_decision')}"
                    },
                    metadata={
                        "principles": issue.get("principles", []),
                        "similar_actions": issue.get("similar_actions", [])
                    }
                )
                flags.append(flag)
        
        # Check rule application consistency
        from fairness_evaluator_implementations import _check_rule_application_consistency
        if similar_actions:
            rule_issues = _check_rule_application_consistency(message, similar_actions, principles)
            for issue in rule_issues:
                flag = FairnessFlag(
                    id=str(uuid.uuid4()),
                    type="rule_consistency",
                    description=issue["description"],
                    severity=issue["severity"],
                    affected_groups=["consistent rule application"],
                    content_reference={
                        "original": content,
                        "problematic_section": f"Rule differences: {issue.get('rule_id')}"
                    },
                    metadata={
                        "principles": issue.get("principles", []),
                        "similar_actions": issue.get("similar_actions", []),
                        "current_rules": issue.get("current_rules", []),
                        "historical_rules": issue.get("historical_rules", [])
                    }
                )
                flags.append(flag)
        
        # Check for language bias
        from fairness_evaluator_implementations import _check_language_bias
        language_bias_issues = _check_language_bias(content)
        for issue in language_bias_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="language_bias",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=issue["affected_groups"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "biased_terms": issue.get("terms", [])
                }
            )
            flags.append(flag)
        
        # Check for assumption bias
        from fairness_evaluator_implementations import _check_assumption_bias
        agent_metadata = context.get("agent_metadata") if context else None
        assumption_bias_issues = _check_assumption_bias(content, agent_metadata)
        for issue in assumption_bias_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="assumption_bias",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=issue["affected_groups"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "assumptions": issue.get("assumptions", [])
                }
            )
            flags.append(flag)
        
        # Check for treatment bias
        from fairness_evaluator_implementations import _check_treatment_bias
        treatment_bias_issues = _check_treatment_bias(content, agent_metadata)
        for issue in treatment_bias_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="treatment_bias",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=issue["affected_groups"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "differential_treatment": issue.get("differential_treatment", [])
                }
            )
            flags.append(flag)
        
        # Check for perspective diversity
        from fairness_evaluator_implementations import _check_perspective_diversity
        perspective_issues = _check_perspective_diversity(content)
        for issue in perspective_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="perspective_diversity",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=issue["affected_groups"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "missing_perspectives": issue.get("missing_perspectives", [])
                }
            )
            flags.append(flag)
        
        # Check for balanced consideration
        from fairness_evaluator_implementations import _check_balanced_consideration
        balance_issues = _check_balanced_consideration(content)
        for issue in balance_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="balanced_consideration",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=["alternative viewpoints"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "imbalance_details": issue.get("imbalance_details", "")
                }
            )
            flags.append(flag)
        
        # Check reasoning clarity
        from fairness_evaluator_implementations import _check_reasoning_clarity
        clarity_issues = _check_reasoning_clarity(content)
        for issue in clarity_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="reasoning_clarity",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=["all users"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "unclear_elements": issue.get("unclear_elements", [])
                }
            )
            flags.append(flag)
        
        # Check process visibility
        from fairness_evaluator_implementations import _check_process_visibility
        visibility_issues = _check_process_visibility(content)
        for issue in visibility_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="process_visibility",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=["all users"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "hidden_elements": issue.get("hidden_elements", [])
                }
            )
            flags.append(flag)
        
        # Check language complexity
        from fairness_evaluator_implementations import _check_language_complexity
        complexity_issues = _check_language_complexity(content)
        for issue in complexity_issues:
            flag = FairnessFlag(
                id=str(uuid.uuid4()),
                type="language_complexity",
                description=issue["description"],
                severity=issue["severity"],
                affected_groups=["all users"],
                content_reference={
                    "original": content,
                    "problematic_section": issue.get("context", "")
                },
                metadata={
                    "complex_elements": issue.get("complex_elements", [])
                }
            )
            flags.append(flag)
        
        # Calculate overall metrics
        language_bias_score = 1.0 - sum(issue["severity"] for issue in language_bias_issues) / max(1.0, len(language_bias_issues))
        metrics.append(FairnessMetric(
            dimension="language_bias",
            score=max(0.0, language_bias_score),
            confidence=0.8,
            details={"issues_count": len(language_bias_issues)}
        ))
        
        assumption_bias_score = 1.0 - sum(issue["severity"] for issue in assumption_bias_issues) / max(1.0, len(assumption_bias_issues))
        metrics.append(FairnessMetric(
            dimension="assumption_bias",
            score=max(0.0, assumption_bias_score),
            confidence=0.7,
            details={"issues_count": len(assumption_bias_issues)}
        ))
        
        treatment_bias_score = 1.0 - sum(issue["severity"] for issue in treatment_bias_issues) / max(1.0, len(treatment_bias_issues))
        metrics.append(FairnessMetric(
            dimension="treatment_bias",
            score=max(0.0, treatment_bias_score),
            confidence=0.75,
            details={"issues_count": len(treatment_bias_issues)}
        ))
        
        perspective_diversity_score = 1.0 - sum(issue["severity"] for issue in perspective_issues) / max(1.0, len(perspective_issues))
        metrics.append(FairnessMetric(
            dimension="perspective_diversity",
            score=max(0.0, perspective_diversity_score),
            confidence=0.7,
            details={"issues_count": len(perspective_issues)}
        ))
        
        # Update historical actions for future consistency checks
        self.historical_actions = [*self.historical_actions, message]
        
        return metrics, flags

    def generate_alternatives(self, flags: List[FairnessFlag]) -> List[FairnessAlternative]:
        """
        Generate alternative approaches for identified fairness issues.
        
        Args:
            flags: List of fairness issues identified
            
        Returns:
            List of alternative approaches addressing the issues
        """
        alternatives = []
        
        for flag in flags:
            # Generate alternative based on flag type
            if flag.type == "language_bias":
                alternatives.extend(self._generate_language_alternatives(flag))
            elif flag.type == "assumption_bias":
                alternatives.extend(self._generate_assumption_alternatives(flag))
            elif flag.type == "treatment_bias":
                alternatives.extend(self._generate_treatment_alternatives(flag))
            elif flag.type == "perspective_diversity":
                alternatives.extend(self._generate_diversity_alternatives(flag))
            elif flag.type == "balanced_consideration":
                alternatives.extend(self._generate_balance_alternatives(flag))
            elif flag.type == "reasoning_clarity":
                alternatives.extend(self._generate_clarity_alternatives(flag))
            elif flag.type == "process_visibility":
                alternatives.extend(self._generate_visibility_alternatives(flag))
            elif flag.type == "language_complexity":
                alternatives.extend(self._generate_complexity_alternatives(flag))
            elif flag.type in ["decision_consistency", "rule_consistency"]:
                alternatives.extend(self._generate_consistency_alternatives(flag))
        
        return alternatives

    def _generate_language_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for language bias issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the language bias
        """
        alternatives = []
        
        # Extract problematic terms from metadata
        biased_terms = flag.metadata.get("biased_terms", [])
        if not biased_terms:
            return alternatives
        
        # Generate a more inclusive alternative
        biased_term_replacements = {
            # Gender-neutral replacements
            "he": "they",
            "his": "their",
            "him": "them",
            "man": "person",
            "men": "people",
            "guy": "person",
            "guys": "folks",
            "gentleman": "person",
            "gentlemen": "people",
            "boyish": "youthful",
            "manly": "strong",
            "manpower": "workforce",
            "mankind": "humanity",
            "chairman": "chairperson",
            "policeman": "police officer",
            "fireman": "firefighter",
            "steward": "flight attendant",
            "stewardess": "flight attendant",
            
            # Presumptive bias replacements
            "obviously": "it appears that",
            "clearly": "it seems that",
            "certainly": "likely",
            "evidently": "it appears that",
            "apparently": "it seems that",
            "undeniable": "strong evidence suggests",
            "undeniably": "the evidence suggests",
            "definitely": "likely",
            
            # Absolutist language replacements
            "always": "often",
            "never": "rarely",
            "all": "many",
            "none": "few",
            "every single": "many",
            "without exception": "with few exceptions",
            "impossible": "challenging",
            "completely": "largely",
            "totally": "largely",
            "absolutely": "largely"
        }
        
        # Create modified content with replacements
        original_content = flag.content_reference.get("original", "")
        modified_content = original_content
        
        for term in biased_terms:
            term_lower = term.lower()
            if term_lower in biased_term_replacements:
                # Preserve case pattern
                replacement = biased_term_replacements[term_lower]
                if term.isupper():
                    replacement = replacement.upper()
                elif term[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                
                # Replace the term
                modified_content = modified_content.replace(term, replacement)
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description=f"Replace biased language with more inclusive alternatives",
            replacement_content=modified_content,
            fairness_improvement=0.7,
            impact_assessment={
                "readability": 0.0,  # Neutral impact on readability
                "precision": -0.1  # Slight negative impact on precision in some cases
            },
            metadata={
                "term_replacements": {term: biased_term_replacements.get(term.lower(), "more inclusive term") 
                                     for term in biased_terms if term.lower() in biased_term_replacements}
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives

    def _generate_assumption_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for assumption bias issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the assumption bias
        """
        alternatives = []
        
        # Extract problematic assumptions from metadata
        assumptions = flag.metadata.get("assumptions", [])
        if not assumptions:
            return alternatives
        
        # Define replacement patterns for different types of assumptions
        assumption_replacements = {
            # Technical expertise assumptions
            "of course": "note that",
            "naturally": "it may be helpful to know that",
            
            # Resource assumptions
            "simply": "",  # Often can be removed entirely
            "just": "",  # Often can be removed entirely
            "easily": "can",
            "quickly": "can",
            "readily": "can",
            
            # Capability assumptions
            "anyone can": "many people can",
            "everyone can": "many people can",
            "everyone should": "one approach is to",
            "you can easily": "you can",
            "you should easily": "you can",
            
            # Default perspective assumptions
            "we all": "many people",
            "all of us": "many people",
            "everyone": "many people"
        }
        
        # Create modified content with replacements
        original_content = flag.content_reference.get("original", "")
        modified_content = original_content
        
        for assumption in assumptions:
            for pattern, replacement in assumption_replacements.items():
                if pattern in assumption.lower():
                    modified_content = modified_content.replace(assumption, assumption.replace(pattern, replacement))
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description=f"Rephrase content to avoid making assumptions about users' capabilities or resources",
            replacement_content=modified_content,
            fairness_improvement=0.6,
            impact_assessment={
                "clarity": 0.1,  # Slight positive impact on clarity
                "conciseness": -0.1  # Slight negative impact on conciseness
            },
            metadata={
                "assumption_replacements": {assumption: assumption.replace(pattern, replacement) 
                                           for assumption in assumptions 
                                           for pattern, replacement in assumption_replacements.items() 
                                           if pattern in assumption.lower()}
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives

    def _generate_treatment_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for treatment bias issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the treatment bias
        """
        alternatives = []
        
        # Extract differential treatment patterns from metadata
        differential_treatments = flag.metadata.get("differential_treatment", [])
        affected_groups = flag.affected_groups
        
        if not differential_treatments or not affected_groups:
            return alternatives
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        modified_content = original_content
        
        for treatment in differential_treatments:
            # Replace preferential treatment with more neutral language
            if "better suited for" in treatment:
                modified_content = modified_content.replace(
                    treatment, 
                    "can be used by people with various backgrounds and experiences"
                )
            elif "ideal for" in treatment:
                modified_content = modified_content.replace(
                    treatment, 
                    "designed to be accessible to users with different needs"
                )
            elif "not designed for" in treatment or "less suitable for" in treatment:
                modified_content = modified_content.replace(
                    treatment, 
                    "designed with accessibility considerations for users with different needs"
                )
            elif "struggle with" in treatment or "find it hard" in treatment:
                modified_content = modified_content.replace(
                    treatment, 
                    "may have different experiences when using this feature"
                )
            else:
                # Generic replacement for other patterns
                modified_content = modified_content.replace(
                    treatment, 
                    "designed to be accessible to all users regardless of background or experience"
                )
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description=f"Replace differential treatment language with more inclusive framing",
            replacement_content=modified_content,
            fairness_improvement=0.8,
            impact_assessment={
                "specificity": -0.2,  # Negative impact on specificity
                "inclusivity": 0.8  # Strong positive impact on inclusivity
            },
            metadata={
                "affected_groups": affected_groups,
                "treatment_replacements": {treatment: "more inclusive framing" for treatment in differential_treatments}
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives

    def _generate_diversity_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for perspective diversity issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the perspective diversity issue
        """
        alternatives = []
        
        # Extract missing perspectives from metadata
        missing_perspectives = flag.metadata.get("missing_perspectives", [])
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        problematic_section = flag.content_reference.get("problematic_section", "")
        
        if "only way" in problematic_section:
            replacement = "one approach"
            modified_content = original_content.replace("only way", replacement)
        elif "single way" in problematic_section:
            replacement = "one approach"
            modified_content = original_content.replace("single way", replacement)
        elif "must use" in problematic_section:
            replacement = "could consider using"
            modified_content = original_content.replace("must use", replacement)
        elif "should use" in problematic_section:
            replacement = "could consider using"
            modified_content = original_content.replace("should use", replacement)
        elif "have to use" in problematic_section:
            replacement = "could consider using"
            modified_content = original_content.replace("have to use", replacement)
        elif "need to use" in problematic_section:
            replacement = "could consider using"
            modified_content = original_content.replace("need to use", replacement)
        else:
            # If no specific pattern is found, add a sentence acknowledging alternatives
            modified_content = original_content + " Other approaches may also be suitable depending on specific requirements and constraints."
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description=f"Acknowledge alternative perspectives or approaches",
            replacement_content=modified_content,
            fairness_improvement=0.7,
            impact_assessment={
                "decisiveness": -0.3,  # Negative impact on decisiveness
                "completeness": 0.4  # Positive impact on completeness
            },
            metadata={
                "missing_perspectives": missing_perspectives
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives

    def _generate_balance_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for balanced consideration issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the balance issue
        """
        alternatives = []
        
        # Extract imbalance details from metadata
        imbalance_details = flag.metadata.get("imbalance_details", "")
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        
        # Replace dismissive language with more balanced considerations
        modified_content = original_content
        
        dismissive_patterns = [
            "clearly inferior", "obviously inferior", "simply inferior", "just inferior",
            "clearly worse", "obviously worse", "simply worse", "just worse",
            "clearly not as good", "obviously not as good", "simply not as good", "just not as good",
            "clearly less effective", "obviously less effective", "simply less effective", "just less effective",
            "clearly problematic", "obviously problematic", "simply problematic", "just problematic",
            "shouldn't consider", "don't consider", "should not consider", "do not consider"
        ]
        
        for pattern in dismissive_patterns:
            if pattern in modified_content.lower():
                # Replace with more balanced language
                balanced_replacement = {
                    "clearly inferior": "may have certain limitations",
                    "obviously inferior": "may have certain limitations",
                    "simply inferior": "may have certain limitations",
                    "just inferior": "may have certain limitations",
                    "clearly worse": "may be less suitable in some contexts",
                    "obviously worse": "may be less suitable in some contexts",
                    "simply worse": "may be less suitable in some contexts",
                    "just worse": "may be less suitable in some contexts",
                    "clearly not as good": "may not offer all the same benefits",
                    "obviously not as good": "may not offer all the same benefits",
                    "simply not as good": "may not offer all the same benefits",
                    "just not as good": "may not offer all the same benefits",
                    "clearly less effective": "may be less effective for certain use cases",
                    "obviously less effective": "may be less effective for certain use cases",
                    "simply less effective": "may be less effective for certain use cases",
                    "just less effective": "may be less effective for certain use cases",
                    "clearly problematic": "presents certain challenges",
                    "obviously problematic": "presents certain challenges",
                    "simply problematic": "presents certain challenges",
                    "just problematic": "presents certain challenges",
                    "shouldn't consider": "might also consider",
                    "don't consider": "might also consider",
                    "should not consider": "might also consider",
                    "do not consider": "might also consider"
                }
                
                modified_content = modified_content.replace(pattern, balanced_replacement.get(pattern, "has both advantages and disadvantages"))
        
        # Add sentence acknowledging trade-offs if no specific patterns were found
        if modified_content == original_content:
            modified_content += " Each option has its own set of trade-offs that should be considered based on specific requirements and constraints."
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description=f"Present options with more balanced consideration of advantages and disadvantages",
            replacement_content=modified_content,
            fairness_improvement=0.6,
            impact_assessment={
                "decisiveness": -0.3,  # Negative impact on decisiveness
                "completeness": 0.5  # Positive impact on completeness
            },
            metadata={
                "imbalance_details": imbalance_details
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives

    def _generate_clarity_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for reasoning clarity issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the clarity issue
        """
        alternatives = []
        
        # Extract unclear elements from metadata
        unclear_elements = flag.metadata.get("unclear_elements", [])
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        modified_content = original_content
        
        # Add explanation if missing
        if "missing explanation" in unclear_elements or "decision rationale" in unclear_elements:
            # Add a generic explanation sentence for recommendations
            modified_content += " This recommendation is based on considerations of usability, efficiency, and alignment with common best practices."
        
        # Make reasoning steps explicit if needed
        if "implicit reasoning" in unclear_elements:
            modified_content += " The reasoning behind this approach includes: 1) it follows established patterns that are familiar to users, 2) it minimizes potential for errors, and 3) it provides a good balance between simplicity and functionality."
        
        # Add decision criteria if missing
        if "missing criteria" in unclear_elements:
            modified_content += " The key criteria used in reaching this conclusion were: reliability, maintainability, performance impact, and compatibility with existing systems."
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description="Improve clarity of reasoning by making explanation explicit",
            replacement_content=modified_content,
            fairness_improvement=0.7,
            impact_assessment={
                "clarity": 0.8,  # Strong positive impact on clarity
                "conciseness": -0.3  # Negative impact on conciseness
            },
            metadata={
                "unclear_elements": unclear_elements
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_visibility_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for process visibility issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the visibility issue
        """
        alternatives = []
        
        # Extract hidden elements from metadata
        hidden_elements = flag.metadata.get("hidden_elements", [])
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        
        # Add visibility explanations based on hidden elements
        modified_content = original_content
        
        # Add decision process information if missing
        if "decision process" in hidden_elements:
            modified_content += " The decision process used here involves evaluating options based on their efficiency, maintainability, and alignment with established best practices."
        
        # Add algorithm explanation if relevant
        if "algorithm details" in hidden_elements:
            modified_content += " The algorithm works by analyzing the input, applying established transformation rules, and optimizing the output based on predefined criteria."
        
        # Add data source information if relevant
        if "data sources" in hidden_elements:
            modified_content += " This information is derived from documentation, commonly accepted industry standards, and established technical specifications."
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description="Improve process visibility by making hidden elements explicit",
            replacement_content=modified_content,
            fairness_improvement=0.6,
            impact_assessment={
                "transparency": 0.8,  # Strong positive impact on transparency
                "conciseness": -0.2  # Slight negative impact on conciseness
            },
            metadata={
                "hidden_elements": hidden_elements
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_complexity_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for language complexity issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the complexity issue
        """
        alternatives = []
        
        # Extract complex elements from metadata
        complex_elements = flag.metadata.get("complex_elements", [])
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        modified_content = original_content
        
        # Replace technical jargon with simpler explanations
        jargon_replacements = {
            # Technical jargon
            "polymorphism": "the ability for objects to take different forms",
            "encapsulation": "bundling data with methods that operate on that data",
            "instantiate": "create",
            "utilize": "use",
            "implementation": "creation",
            "functionality": "features",
            "initialize": "start",
            "terminate": "end",
            "optimize": "improve",
            "concatenate": "combine",
            "parameter": "input value",
            "execute": "run",
            "interface": "connection point",
            "synchronize": "coordinate",
            "asynchronous": "non-blocking",
            "paradigm": "approach",
            "recursion": "process where a function calls itself",
            "algorithm": "step-by-step procedure",
            "iteration": "repetition",
            "implementation": "way of coding something",
            "framework": "software tool that provides ready-made components",
            "runtime": "when the program is running",
            "middleware": "software that connects different applications",
            "abstraction": "simplified view",
            "refactor": "restructure",
            "obfuscation": "making code harder to understand",
            "instantiation": "creation of an object",
            "inheritance": "passing down properties to a new class",
            
            # Complex sentences
            "in order to": "to",
            "for the purpose of": "to",
            "with the objective of": "to",
            "in the event that": "if",
            "is able to": "can",
            "due to the fact that": "because",
            "in spite of the fact that": "although",
            "with regard to": "about",
            "in the vicinity of": "near",
            "a significant number of": "many",
            "the vast majority of": "most",
            "despite the fact that": "although",
            "in the absence of": "without",
            "subsequent to": "after",
            "prior to": "before",
            "in conjunction with": "with",
            "in accordance with": "following",
            "in relation to": "about",
            "notwithstanding": "despite"
        }
        
        # Apply replacements
        for complex_term, simple_term in jargon_replacements.items():
            if complex_term in modified_content:
                modified_content = modified_content.replace(complex_term, simple_term)
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description="Simplify language to improve accessibility",
            replacement_content=modified_content,
            fairness_improvement=0.5,
            impact_assessment={
                "accessibility": 0.8,  # Strong positive impact on accessibility
                "precision": -0.3  # Negative impact on precision
            },
            metadata={
                "complex_elements": complex_elements,
                "replacements": {complex_term: simple_term 
                                 for complex_term, simple_term in jargon_replacements.items() 
                                 if complex_term in original_content}
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_consistency_alternatives(self, flag: FairnessFlag) -> List[FairnessAlternative]:
        """
        Generate alternatives for consistency issues.
        
        Args:
            flag: The fairness flag to address
            
        Returns:
            List of alternatives addressing the consistency issue
        """
        alternatives = []
        
        # Extract relevant information from metadata
        similar_actions = flag.metadata.get("similar_actions", [])
        principles = flag.metadata.get("principles", [])
        
        # Create modified content
        original_content = flag.content_reference.get("original", "")
        problematic_section = flag.content_reference.get("problematic_section", "")
        
        # Add explanation about consistency with previous decisions
        modified_content = original_content + "\n\nNote: This approach is consistent with previous similar situations, applying the same principles and evaluation criteria."
        
        # Create alternative
        alternative = FairnessAlternative(
            id=str(uuid.uuid4()),
            flag_id=flag.id,
            description="Add explanation to highlight consistency with previous decisions",
            replacement_content=modified_content,
            fairness_improvement=0.7,
            impact_assessment={
                "transparency": 0.6,  # Positive impact on transparency
                "conciseness": -0.2  # Slight negative impact on conciseness
            },
            metadata={
                "similar_actions": similar_actions,
                "principles": principles
            }
        )
        
        alternatives.append(alternative)
        
        return alternatives