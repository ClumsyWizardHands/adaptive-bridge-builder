#!/usr/bin/env python3
"""
Enhanced Principle Evaluator

This module provides improved scoring and recommendation generation for the
principle-based evaluation system. It addresses three key improvements:

1. Better scoring mechanisms to accurately identify non-compliant actions
2. More diverse and contextually relevant recommendations
3. Support for more comprehensive testing with varied principle sets
"""

import json
import logging
import re
import textwrap
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from datetime import datetime

from principle_engine import PrincipleEngine
from principle_engine_action_evaluator import PrincipleActionEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedPrincipleEvaluator")


class EnhancedPrincipleEvaluator(PrincipleActionEvaluator):
    """
    Enhanced version of PrincipleActionEvaluator with improved scoring and recommendations.
    
    Key improvements:
    1. Better pattern recognition for principle violations
    2. More nuanced scoring based on action context and severity
    3. More diverse recommendation generation
    4. Support for specialized principle categories
    """
    
    def __init__(self, principle_engine=None, principles_file=None):
        """Initialize with parent class constructor."""
        super().__init__(principle_engine, principles_file)
        
        # Initialize enhanced pattern detection
        self._init_enhanced_patterns()
        
        # Map of principle ids to specialized evaluation functions
        self.specialized_evaluators = {}
        
        # Register default specialized evaluators
        self._register_default_evaluators()
        
        logger.info(f"EnhancedPrincipleEvaluator initialized with {len(self.principle_engine.principles)} principles")
    
    def _init_enhanced_patterns(self):
        """Initialize enhanced pattern detection for improved scoring."""
        # General problematic patterns with severity weights (1-10 scale)
        self.problematic_patterns = {
            # Privacy and consent issues (high severity)
            r"without\s+(user\s+)?consent": 9,
            r"without\s+(user\s+)?permission": 9,
            r"without\s+(user\s+)?knowledge": 9,
            r"without\s+informing": 8,
            r"track\s+users?\s+without": 9,
            r"collect\s+data\s+without": 8,
            r"monitor\s+without": 8,
            
            # Transparency issues (medium-high severity)
            r"hidden\s+from\s+users?": 7,
            r"not\s+disclose": 7,
            r"silently": 6,
            r"secretly": 8,
            r"hide": 6,
            r"conceal": 6,
            r"mislead": 8,
            
            # Security issues (high severity)
            r"bypass\s+security": 9,
            r"override\s+security": 8,
            r"ignore\s+security": 9,
            r"compromise\s+security": 8,
            
            # Ethical issues (medium-high severity)
            r"exploit": 7,
            r"manipulate": 7,
            r"force\s+users?\s+to": 7,
            r"restrict\s+access": 6,
            r"discriminate": 8,
            r"bias": 7,
            
            # General negative patterns (medium severity)
            r"ignore\s+user": 6,
            r"override\s+user": 6,
            r"without\s+notif": 5,
            r"mandatory": 4
        }
        
        # Potentially mitigating factors (can reduce severity)
        self.mitigating_patterns = {
            r"with\s+clear\s+consent": 8,
            r"with\s+explicit\s+permission": 8,
            r"with\s+user\s+approval": 7,
            r"transparent": 6,
            r"clearly\s+inform": 6,
            r"opt[\-\s]out": 5,
            r"opt[\-\s]in": 7,
            r"privacy\s+notice": 5,
            r"security\s+measure": 5,
            r"user\s+choice": 6,
            r"user\s+control": 6,
            r"emergency": 6,
            r"critical": 5
        }
        
        # Context patterns that suggest principle application areas
        self.context_patterns = {
            "privacy": [r"data", r"personal\s+information", r"track", r"monitor", r"collect"],
            "security": [r"protect", r"secure", r"vulnerab", r"breach", r"hack", r"threat"],
            "transparency": [r"inform", r"disclos", r"explain", r"communi", r"report"],
            "user_control": [r"choice", r"option", r"control", r"settings", r"preference"],
            "inclusivity": [r"access", r"disab", r"diverse", r"inclus", r"equal"],
            "autonomy": [r"decision", r"choice", r"freedom", r"independ", r"self"]
        }
    
    def _register_default_evaluators(self):
        """Register specialized evaluators for specific principle types."""
        # This would be populated with specific evaluation functions for different
        # types of principles. For simplicity, we'll add just a few examples.
        
        # Define specialized evaluators here
        def evaluate_privacy_principle(action, principle, context):
            """Specialized evaluation for privacy-related principles."""
            score = 100  # Start with perfect score
            violations = []
            recommendations = []
            
            # Look for explicit privacy violations
            privacy_violating_patterns = [
                (r"collect\s+.{0,30}\s+without\s+consent", 
                 "Collecting data without user consent",
                 "Obtain explicit consent before collecting user data"),
                
                (r"track\s+.{0,30}\s+without\s+.{0,30}\s+knowledge", 
                 "Tracking users without their knowledge",
                 "Be transparent about any tracking and provide user controls"),
                
                (r"share\s+.{0,30}\s+(data|information)\s+with\s+third\s+part", 
                 "Sharing data with third parties without adequate consent",
                 "Ensure users understand and consent to third-party data sharing"),
                
                (r"store\s+.{0,30}\s+without\s+encryption", 
                 "Storing sensitive data without encryption",
                 "Implement proper encryption for all stored user data"),
                
                (r"retain\s+.{0,30}\s+data\s+indefinitely", 
                 "Retaining user data indefinitely",
                 "Implement data retention policies with clear timeframes"),
                
                (r"access\s+.{0,20}\s+information\s+without\s+permission", 
                 "Accessing user information without permission",
                 "Only access user information with proper authorization")
            ]
            
            for pattern, violation_msg, recommendation in privacy_violating_patterns:
                if re.search(pattern, action.lower()):
                    score -= 25  # Substantial penalty
                    violations.append(violation_msg)
                    recommendations.append(recommendation)
            
            # Check for mitigating factors
            mitigating_factors = [
                (r"with\s+explicit\s+consent", 15),
                (r"anonymiz(e|ing)", 10),
                (r"pseudonymiz(e|ing)", 8),
                (r"encrypt", 10),
                (r"inform\s+users?\s+clearly", 8),
                (r"data\s+minimization", 12)
            ]
            
            for pattern, bonus in mitigating_factors:
                if re.search(pattern, action.lower()):
                    score = min(100, score + bonus)  # Apply bonus up to max 100
            
            # Context-specific adjustments
            if context and context.get("emergency_situation"):
                if score < 70:  # If failing, give slight boost for emergencies
                    score = min(70, score + 15)
                    recommendations.append("Even in emergencies, maintain privacy protections while addressing critical needs")
            
            # Overall privacy principles tend to be non-negotiable
            if score < 60:
                score = max(40, score)  # Floor at 40 for privacy principles
            
            return score, violations, recommendations
        
        # Add specialized evaluators for specific principle types
        # In a real implementation, you'd identify principles by their ID,
        # keywords in their name, or other distinguishing features
        
        # Just as an example, we'll add the privacy evaluator
        # In a real implementation, you'd have a way to map principles to these evaluators
        self.specialized_evaluators["privacy"] = evaluate_privacy_principle
    
    def _evaluate_action_against_principle(
        self, action: str, principle: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Enhanced evaluation with better pattern detection and scoring.
        
        Args:
            action: Description of the action
            principle: The principle definition
            context: Additional context
            
        Returns:
            Tuple of (score, violations, recommendations)
        """
        # Check if there's a specialized evaluator for this principle type
        principle_id = principle["id"]
        principle_name = principle["name"].lower()
        
        # See if we have a specialized evaluator for this principle
        specialized_evaluator = None
        
        # Look for direct match by ID
        if principle_id in self.specialized_evaluators:
            specialized_evaluator = self.specialized_evaluators[principle_id]
        else:
            # Look for keywords in principle name that match specialized evaluators
            for category, evaluator in self.specialized_evaluators.items():
                if category in principle_name or category.replace("_", " ") in principle_name:
                    specialized_evaluator = evaluator
                    break
        
        # Use specialized evaluator if found
        if specialized_evaluator:
            score, violations, recommendations = specialized_evaluator(action, principle, context)
            # Still do the basic evaluation and take the stricter result
            basic_score, basic_violations, basic_recommendations = self._basic_evaluation(action, principle, context)
            
            # Take the lower score (stricter)
            if basic_score < score:
                score = basic_score
                # Combine violations and recommendations
                for v in basic_violations:
                    if v not in violations:
                        violations.append(v)
                for r in basic_recommendations:
                    if r not in recommendations:
                        recommendations.append(r)
        else:
            # Use enhanced basic evaluation
            score, violations, recommendations = self._enhanced_evaluation(action, principle, context)
        
        # Ensure score stays in valid range
        score = max(0, min(100, score))
        
        return score, violations, recommendations
    
    def _basic_evaluation(
        self, action: str, principle: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Basic evaluation from parent class with minor improvements.
        This is a fallback for when specialized evaluators aren't available.
        """
        # Start with parent class implementation
        score, violations, recommendations = super()._evaluate_action_against_principle(action, principle, context)
        
        # Apply additional checks and balances
        principle_name = principle["name"]
        action_lower = action.lower()
        
        # Check action against all enhanced problematic patterns
        for pattern, severity in self.problematic_patterns.items():
            if re.search(pattern, action_lower):
                penalty = severity * 5  # Convert 1-10 scale to reduction amount
                old_score = score
                score = max(0, score - penalty)
                
                # Only add violation if score actually changed
                if old_score != score:
                    violation = f"Action contains problematic pattern: {pattern}"
                    if violation not in violations:
                        violations.append(violation)
                    
                    # Add a corresponding recommendation
                    antidote = f"Avoid {pattern}"
                    if antidote not in recommendations:
                        recommendations.append(antidote)
        
        # Check for mitigating factors
        for pattern, mitigation in self.mitigating_patterns.items():
            if re.search(pattern, action_lower):
                bonus = mitigation * 2  # Convert to score bonus
                old_score = score
                score = min(100, score + bonus)
                
                # If score improved, we can remove some violations
                if old_score != score and violations:
                    # Remove one violation if mitigating factor applies
                    violations = violations[1:]
        
        # Context-based adjustments
        if context:
            # Emergency situations may allow some flexibility
            if context.get("emergency_situation") and score < 70:
                score = min(70, score + 10)  # Boost score for emergencies but cap at 70
                recommendations.append("Even in emergencies, maintain alignment with this principle while addressing critical needs")
            
            # User consent explicitly provided
            if context.get("user_consent") and "consent" in action_lower:
                score = min(100, score + 20)  # Significant boost for explicit consent
        
        return score, violations, recommendations
    
    def _enhanced_evaluation(
        self, action: str, principle: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Enhanced evaluation with more sophisticated pattern matching and scoring.
        """
        # Start with a perfect score
        score = 100.0
        violations = []
        recommendations = []
        
        principle_id = principle["id"]
        principle_name = principle["name"]
        principle_description = principle.get("description", "")
        evaluation_criteria = principle.get("evaluation_criteria", [])
        
        action_lower = action.lower()
        
        # 1. Extract key terms from principle name and description
        key_terms = self._extract_key_terms(principle_name, principle_description)
        
        # 2. Check for direct contradictions using key terms
        contradictions = [
            "against", "violate", "ignore", "bypass", "override", 
            "without regard", "disregard", "counter to", "contrary to"
        ]
        
        for term in key_terms:
            for contradiction in contradictions:
                contradiction_pattern = f"{contradiction}\\s+\\w*\\s*{term}"
                if re.search(contradiction_pattern, action_lower):
                    score -= 40
                    violation = f"Action explicitly contradicts the principle by {contradiction} {term}"
                    violations.append(violation)
                    recommendations.append(f"Respect the principle by honoring {term}")
        
        # 3. Check against enhanced problematic patterns
        for pattern, severity in self.problematic_patterns.items():
            if re.search(pattern, action_lower):
                # Only apply if related to this principle (check key terms)
                is_relevant = False
                for term in key_terms:
                    # Either pattern contains term or term contains part of pattern
                    if term in pattern or any(p in term for p in pattern.split("\\s+")):
                        is_relevant = True
                        break
                
                # If no specific relevance is found, still apply but with lower severity
                penalty = severity * 5 if is_relevant else severity * 2
                score = max(0, score - penalty)
                
                # Format the pattern more readably for the message
                readable_pattern = pattern.replace("\\s+", " ")
                violation = f"Action contains concerning pattern: {readable_pattern}"
                violations.append(violation)
                
                # Generate a recommendation based on the pattern
                recommendation = self._generate_recommendation_from_pattern(pattern, principle_name)
                recommendations.append(recommendation)
        
        # 4. Check for mitigating factors
        for pattern, mitigation in self.mitigating_patterns.items():
            if re.search(pattern, action_lower):
                bonus = mitigation * 2  # Convert to score bonus
                score = min(100, score + bonus)
        
        # 5. Consider the evaluation criteria if they exist
        if evaluation_criteria:
            criteria_score = self._evaluate_against_criteria(action, evaluation_criteria)
            # Blend scores, giving slightly more weight to criteria-based evaluation
            score = (score * 0.4) + (criteria_score * 0.6)
        
        # 6. Context-based adjustments
        if context:
            score = self._apply_context_adjustments(score, action, principle, context, violations, recommendations)
        
        # Principle-specific scoring adjustments based on key terms
        principle_category = self._identify_principle_category(principle_name, principle_description)
        
        # Apply category-specific scoring adjustments
        if principle_category == "privacy":
            if score < 70:  # Privacy violations are generally serious
                score = max(score, score * 0.8)  # Can go as low as 80% of original score
        elif principle_category == "security":
            if score < 60:  # Security issues are critical
                score = max(score, score * 0.7)  # Can go as low as 70% of original score
        elif principle_category == "ethics":
            if score < 50:  # Ethical violations are very serious
                score = max(score, score * 0.6)  # Can go as low as 60% of original score
        
        # Ensure score stays in valid range and round to integer
        score = round(max(0, min(100, score)))
        
        return score, violations, recommendations
    
    def _extract_key_terms(self, principle_name: str, principle_description: str) -> List[str]:
        """Extract key terms from the principle name and description."""
        # Combine name and description, lowercase
        combined = (principle_name + " " + principle_description).lower()
        
        # Remove common words
        common_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "on", "off", "for", "in", "out", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "can", "will", "just", "should", "now", "to", "of", "is", "are",
            "with", "without", "as", "be", "this", "that", "these", "those"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', combined)
        filtered_words = [w for w in words if w not in common_words and len(w) > 3]
        
        # Extract phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in common_words or words[i+1] not in common_words:
                phrases.append(words[i] + " " + words[i+1])
        
        for i in range(len(words) - 2):
            if (words[i] not in common_words or 
                words[i+1] not in common_words or 
                words[i+2] not in common_words):
                phrases.append(words[i] + " " + words[i+1] + " " + words[i+2])
        
        # Combine single words and phrases, remove duplicates
        key_terms = list(set(filtered_words + phrases))
        
        return key_terms
    
    def _generate_recommendation_from_pattern(self, pattern: str, principle_name: str) -> str:
        """Generate a meaningful recommendation based on the violated pattern."""
        # Clean up the pattern for readability
        readable_pattern = pattern.replace("\\s+", " ")
        
        # Common transformations from negative patterns to positive recommendations
        transformations = {
            "without consent": "obtain explicit consent before proceeding",
            "without permission": "request permission before proceeding",
            "without knowledge": "inform users clearly",
            "without informing": "communicate transparently",
            "track": "be transparent about any tracking and provide opt-out options",
            "collect data": "only collect necessary data with clear consent",
            "hidden": "be transparent about",
            "not disclose": "disclose clearly",
            "silently": "communicate transparently about",
            "secretly": "be open about",
            "hide": "be transparent about",
            "conceal": "openly disclose",
            "mislead": "be truthful about",
            "bypass security": "follow security protocols",
            "override security": "maintain security standards",
            "ignore security": "prioritize security measures",
            "exploit": "treat fairly and respectfully",
            "manipulate": "engage honestly with",
            "force users": "provide users with choices",
            "restrict access": "ensure equitable access",
            "discriminate": "treat all users equitably",
            "mandatory": "make optional when possible"
        }
        
        # Find the best match in our transformations
        best_match = None
        for negative, positive in transformations.items():
            if negative in readable_pattern:
                best_match = positive
                break
        
        if best_match:
            return f"Respect the principle of {principle_name} by {best_match}"
        else:
            # Fallback recommendation
            return f"Ensure alignment with {principle_name} by avoiding {readable_pattern}"
    
    def _evaluate_against_criteria(self, action: str, criteria: List[str]) -> float:
        """Evaluate an action against a list of specific criteria."""
        if not criteria:
            return 100.0  # Default perfect score if no criteria
        
        total_score = 0
        action_lower = action.lower()
        
        for criterion in criteria:
            # For each criterion, start with a passing score
            criterion_score = 70
            
            # Extract key terms from the criterion
            key_terms = self._extract_key_terms("", criterion)
            
            # Check if action reflects this criterion
            term_matches = 0
            for term in key_terms:
                if term in action_lower:
                    term_matches += 1
                    criterion_score += 10  # Bonus for each matching term
            
            # If no terms match, apply penalty
            if term_matches == 0:
                criterion_score = 50
            
            # Check for negations near key terms
            negations = ["not", "don't", "doesn't", "won't", "wouldn't", "couldn't", 
                        "shouldn't", "never", "no", "none", "neither", "nor"]
            
            for term in key_terms:
                for negation in negations:
                    if re.search(f"{negation}\\s+\\w*\\s*{term}", action_lower) or \
                       re.search(f"{term}\\s+\\w*\\s*{negation}", action_lower):
                        criterion_score = max(0, criterion_score - 30)  # Major penalty for negations
            
            total_score += criterion_score
        
        # Average the scores across all criteria
        return total_score / len(criteria)
    
    def _apply_context_adjustments(
        self, 
        score: float, 
        action: str, 
        principle: Dict[str, Any], 
        context: Dict[str, Any],
        violations: List[str],
        recommendations: List[str]
    ) -> float:
        """Apply score adjustments based on context."""
        # Handle emergency situations - may allow some flexibility
        if context.get("emergency_situation"):
            potential_harm = context.get("potential_harm", "")
            
            if potential_harm and len(potential_harm) > 10:  # Substantive description of harm
                if score < 70:
                    # More significant boost for well-justified emergencies
                    score = min(75, score + 15)
                    recommendations.append(
                        "In emergency situations, maintain alignment with principles while addressing critical needs"
                    )
            else:
                # Smaller boost for generic emergencies
                if score < 70:
                    score = min(70, score + 10)
        
        # Handle explicit user consent
        if context.get("user_consent") is True:
            if "consent" in action.lower() or "permission" in action.lower():
                # Significant boost for actions with explicit consent
                score = min(100, score + 20)
                # Remove consent-related violations
                violations = [v for v in violations if "consent" not in v.lower() 
                            and "permission" not in v.lower()]
        
        # Handle explicit user consent refusal
        if context.get("user_consent") is False:
            if "without consent" in action.lower() or "without permission" in action.lower():
                # Major penalty for ignoring explicit consent refusal
                score = max(0, score - 40)
                violations.append("Action proceeds despite explicit lack of user consent")
                recommendations.append("Never proceed without consent when users have explicitly declined")
        
        # Consider the impact scope
        if context.get("impact_scope") == "limited" and score < 80:
            # Less severe for limited impact
            score = min(80, score + 10)
        elif context.get("impact_scope") == "widespread" and score < 90:
            # More severe for widespread impact
            score = max(0, score - 10)
            
        # Consider temporary vs. permanent effects
        if context.get("effect_duration") == "temporary" and score < 80:
            # Less severe for temporary effects
            score = min(80, score + 5)
        elif context.get("effect_duration") == "permanent" and score < 90:
            # More severe for permanent effects
            score = max(0, score - 10)
        
        return score
    
    def _identify_principle_category(self, principle_name: str, principle_description: str) -> str:
        """Identify the category of a principle based on name and description."""
        combined = (principle_name + " " + principle_description).lower()
        
        # Check for category matches
        category_keywords = {
            "privacy": ["privacy", "confidential", "personal data", "data protection", "anonymity"],
            "security": ["security", "protection", "secure", "vulnerability", "breach", "attack"],
            "transparency": ["transparency", "transparent", "disclose", "open", "clear", "honest"],
            "autonomy": ["autonomy", "choice", "control", "freedom", "self-determination"],
            "fairness": ["fair", "justice", "equitable", "unbiased", "impartial"],
            "ethics": ["ethic", "moral", "right", "wrong", "good", "harm"]
        }
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in combined:
                    score += 1
            if score > 0:
                category_scores[category] = score
        
        # Return the category with the most matches, or "general" if none
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"
    
    def generate_explanation(self, compliance_result: Dict[str, Any]) -> str:
        """
        Generate a more nuanced explanation of compliance issues.
        
        Args:
            compliance_result: Result from check_action_compliance
            
        Returns:
            A formatted explanation string
        """
        explanation = []
        
        # Add a header based on compliance level
        if compliance_result["complies"]:
            if compliance_result["overall_score"] >= 90:
                explanation.append("I can confidently proceed with this action as it strongly aligns with all principles.")
            else:
                explanation.append("I can proceed with this action while respecting principles, though with some minor considerations.")
        else:
            if compliance_result["overall_score"] >= 60:
                explanation.append("I have concerns about this action as it conflicts with some important principles.")
            elif compliance_result["overall_score"] >= 40:
                explanation.append("I cannot recommend this action as it significantly conflicts with multiple principles.")
            else:
                explanation.append("I strongly advise against this action as it fundamentally violates core principles.")
        
        # Include the explanation from the compliance result
        explanation.append("")
        explanation.append(compliance_result["explanation"])
        
        # Add negotiation text for non-compliant actions
        if not compliance_result["complies"]:
            explanation.append("")
            explanation.append("Here are some alternative approaches that would better align with principles:")
            alternatives = self.suggest_alternatives(compliance_result["action"])
            for i, alt in enumerate(alternatives, 1):
                explanation.append(f"{i}. {alt}")
        
        return "\n".join(explanation)
    
    def suggest_alternatives(self, action: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate more diverse and contextually relevant alternative suggestions.
        
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
        seen_recommendations = set()  # Track unique recommendations
        
        # Generate alternatives based on violated principles
        for p in evaluation["violated_principles"]:
            principle_name = p["name"]
            
            # Get recommendations for this principle
            principle_recs = evaluation["principle_scores"][p["id"]]["recommendations"]
            
            if principle_recs:
                # Generate diverse alternatives incorporating the recommendations
                for rec in principle_recs:
                    # Skip duplicate recommendations
                    rec_key = rec.lower()
                    if rec_key in seen_recommendations:
                        continue
                    
                    seen_recommendations.add(rec_key)
                    
                    # Create different alternative phrasings
                    alternatives.append(self._create_alternative_from_recommendation(
                        action, rec, principle_name, variation=1
                    ))
        
        # If we don't have enough alternatives, add some general ones
        if len(alternatives) < 3:
            general_alternatives = [
                f"Consider a more transparent approach that clearly communicates intent and respects {evaluation['violated_principles'][0]['name'] if evaluation['violated_principles'] else 'all principles'}.",
                f"Redesign this action to prioritize user agency and choice while still accomplishing the core objective.",
                f"Implement this with explicit opt-in mechanisms that give users control over their participation."
            ]
            
            for alt in general_alternatives:
                if len(alternatives) >= 5:  # Cap at 5 alternatives
                    break
                if alt not in alternatives:
                    alternatives.append(alt)
        
        # Limit to top 5 most
