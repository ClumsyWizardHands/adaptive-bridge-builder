#!/usr/bin/env python3
"""
Communication Style Module for Adaptive Bridge Builder

This module defines the CommunicationStyle class which represents an agent's
communication patterns and preferences. It provides a structured way to
characterize and adapt to different communication styles while respecting principles.
"""

import json
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommunicationStyle")

class FormalityLevel(Enum):
    """Enumeration of formality levels in communication."""
    VERY_FORMAL = 5
    FORMAL = 4
    NEUTRAL = 3
    CASUAL = 2
    VERY_CASUAL = 1
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

class DetailLevel(Enum):
    """Enumeration of detail levels in communication."""
    VERY_DETAILED = 5
    DETAILED = 4
    BALANCED = 3
    CONCISE = 2
    VERY_CONCISE = 1
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

class DirectnessLevel(Enum):
    """Enumeration of directness levels in communication."""
    VERY_DIRECT = 5
    DIRECT = 4
    BALANCED = 3
    INDIRECT = 2
    VERY_INDIRECT = 1
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

class EmotionalTone(Enum):
    """Enumeration of emotional tones in communication."""
    VERY_POSITIVE = 5
    POSITIVE = 4
    NEUTRAL = 3
    NEGATIVE = 2
    VERY_NEGATIVE = 1
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

class ResponseSpeed(Enum):
    """Enumeration of expected response speeds."""
    IMMEDIATE = 5
    QUICK = 4
    STANDARD = 3
    RELAXED = 2
    EXTENDED = 1
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

@dataclass
class CommunicationStyle:
    """
    Represents an agent's communication style and preferences.
    
    This class encapsulates various aspects of communication style
    including formality, detail level, directness, and emotional tone.
    It provides methods to adapt messages to match or complement the style.
    """
    
    # Core style attributes
    agent_id: str
    formality: FormalityLevel = FormalityLevel.NEUTRAL
    detail_level: DetailLevel = DetailLevel.BALANCED
    directness: DirectnessLevel = DirectnessLevel.BALANCED
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    response_speed: ResponseSpeed = ResponseSpeed.STANDARD
    
    # Additional preferences
    prefers_acknowledgments: bool = True
    prefers_structured_responses: bool = False
    prefers_examples: bool = True
    language_preferences: Dict[str, float] = field(default_factory=dict)
    vocabulary_level: float = 0.5  # 0.0 (simple) to 1.0 (complex)
    
    # Style consistency metrics
    consistency_score: float = 1.0  # 0.0 (inconsistent) to 1.0 (very consistent)
    confidence_level: float = 0.5  # 0.0 (low confidence) to 1.0 (high confidence)
    
    # Analysis metadata
    sample_count: int = 0
    last_updated: Optional[str] = None
    style_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the communication style to a dictionary."""
        result = asdict(self)
        
        # Convert enum values to their string representations
        result['formality'] = str(self.formality)
        result['detail_level'] = str(self.detail_level)
        result['directness'] = str(self.directness)
        result['emotional_tone'] = str(self.emotional_tone)
        result['response_speed'] = str(self.response_speed)
        
        return result
    
    def to_json(self, indent=2) -> str:
        """Convert the communication style to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationStyle':
        """Create a CommunicationStyle instance from a dictionary."""
        # Deep copy to avoid modifying the input
        style_data = dict(data)
        
        # Convert string representations back to enum values
        if 'formality' in style_data and isinstance(style_data['formality'], str):
            style_data['formality'] = next(
                (level for level in FormalityLevel if str(level) == style_data['formality']),
                FormalityLevel.NEUTRAL
            )
        
        if 'detail_level' in style_data and isinstance(style_data['detail_level'], str):
            style_data['detail_level'] = next(
                (level for level in DetailLevel if str(level) == style_data['detail_level']),
                DetailLevel.BALANCED
            )
        
        if 'directness' in style_data and isinstance(style_data['directness'], str):
            style_data['directness'] = next(
                (level for level in DirectnessLevel if str(level) == style_data['directness']),
                DirectnessLevel.BALANCED
            )
        
        if 'emotional_tone' in style_data and isinstance(style_data['emotional_tone'], str):
            style_data['emotional_tone'] = next(
                (tone for tone in EmotionalTone if str(tone) == style_data['emotional_tone']),
                EmotionalTone.NEUTRAL
            )
        
        if 'response_speed' in style_data and isinstance(style_data['response_speed'], str):
            style_data['response_speed'] = next(
                (speed for speed in ResponseSpeed if str(speed) == style_data['response_speed']),
                ResponseSpeed.STANDARD
            )
        
        return cls(**style_data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CommunicationStyle':
        """Create a CommunicationStyle instance from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_adaptation_guidance(self) -> Dict[str, Any]:
        """
        Generate guidance for adapting messages to this communication style.
        
        Returns:
            Dictionary with specific recommendations for adapting to this style.
        """
        guidance = {
            "formality_guidance": self._get_formality_guidance(),
            "detail_guidance": self._get_detail_guidance(),
            "directness_guidance": self._get_directness_guidance(),
            "tone_guidance": self._get_tone_guidance(),
            "speed_guidance": self._get_speed_guidance(),
            "structure_guidance": self._get_structure_guidance(),
            "examples_guidance": self._get_examples_guidance(),
            "vocabulary_guidance": self._get_vocabulary_guidance(),
            "general_recommendations": self._get_general_recommendations()
        }
        
        return guidance
    
    def _get_formality_guidance(self) -> Dict[str, Any]:
        """Generate guidance for adapting message formality."""
        guidance = {
            "level": str(self.formality),
            "recommendations": []
        }
        
        if self.formality in (FormalityLevel.VERY_FORMAL, FormalityLevel.FORMAL):
            guidance["recommendations"] = [
                "Use complete sentences and proper grammar",
                "Avoid contractions (use 'do not' instead of 'don't')",
                "Use professional titles and formal address",
                "Avoid colloquialisms and slang",
                "Maintain professional distance"
            ]
        elif self.formality in (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL):
            guidance["recommendations"] = [
                "Use contractions freely",
                "Incorporate conversational language",
                "Use first names",
                "Include appropriate colloquialisms",
                "Adopt a friendly, approachable tone"
            ]
        else:  # NEUTRAL
            guidance["recommendations"] = [
                "Balance formal and informal elements",
                "Use contractions selectively",
                "Maintain professional but approachable language",
                "Adjust formality based on context"
            ]
            
        return guidance
    
    def _get_detail_guidance(self) -> Dict[str, Any]:
        """Generate guidance for adapting message detail level."""
        guidance = {
            "level": str(self.detail_level),
            "recommendations": []
        }
        
        if self.detail_level in (DetailLevel.VERY_DETAILED, DetailLevel.DETAILED):
            guidance["recommendations"] = [
                "Provide comprehensive explanations",
                "Include background information and context",
                "Offer multiple examples",
                "Break down complex concepts step by step",
                "Address potential edge cases"
            ]
        elif self.detail_level in (DetailLevel.VERY_CONCISE, DetailLevel.CONCISE):
            guidance["recommendations"] = [
                "Focus on key points only",
                "Use bullet points when appropriate",
                "Minimize background information",
                "Be direct and to the point",
                "Avoid tangential information"
            ]
        else:  # BALANCED
            guidance["recommendations"] = [
                "Provide sufficient detail for clarity",
                "Include context where necessary",
                "Balance brevity with completeness",
                "Adjust detail based on complexity of topic"
            ]
            
        return guidance
    
    def _get_directness_guidance(self) -> Dict[str, Any]:
        """Generate guidance for adapting message directness."""
        guidance = {
            "level": str(self.directness),
            "recommendations": []
        }
        
        if self.directness in (DirectnessLevel.VERY_DIRECT, DirectnessLevel.DIRECT):
            guidance["recommendations"] = [
                "State conclusions and recommendations first",
                "Use active voice",
                "Make explicit requests",
                "Be clear about expectations",
                "Minimize hedging language"
            ]
        elif self.directness in (DirectnessLevel.VERY_INDIRECT, DirectnessLevel.INDIRECT):
            guidance["recommendations"] = [
                "Provide context before conclusions",
                "Use softening language ('perhaps', 'consider')",
                "Present options rather than directives",
                "Use passive voice when appropriate",
                "Frame requests as suggestions"
            ]
        else:  # BALANCED
            guidance["recommendations"] = [
                "Balance directness with tact",
                "Adjust directness based on the sensitivity of the topic",
                "Use a mix of active and passive voice",
                "Consider the impact of the message"
            ]
            
        return guidance
    
    def _get_tone_guidance(self) -> Dict[str, Any]:
        """Generate guidance for adapting message emotional tone."""
        guidance = {
            "level": str(self.emotional_tone),
            "recommendations": []
        }
        
        if self.emotional_tone in (EmotionalTone.VERY_POSITIVE, EmotionalTone.POSITIVE):
            guidance["recommendations"] = [
                "Use positive and encouraging language",
                "Highlight opportunities and possibilities",
                "Emphasize strengths and achievements",
                "Express optimism about outcomes",
                "Include affirming statements"
            ]
        elif self.emotional_tone in (EmotionalTone.VERY_NEGATIVE, EmotionalTone.NEGATIVE):
            guidance["recommendations"] = [
                "Acknowledge challenges directly",
                "Focus on problem-solving",
                "Be honest about limitations",
                "Avoid false optimism",
                "Provide realistic assessments"
            ]
        else:  # NEUTRAL
            guidance["recommendations"] = [
                "Maintain balanced perspective",
                "Present facts objectively",
                "Acknowledge both positives and negatives",
                "Focus on information rather than emotion",
                "Adjust tone based on context"
            ]
            
        return guidance
    
    def _get_speed_guidance(self) -> Dict[str, Any]:
        """Generate guidance for adapting response speed expectations."""
        guidance = {
            "level": str(self.response_speed),
            "recommendations": []
        }
        
        if self.response_speed in (ResponseSpeed.IMMEDIATE, ResponseSpeed.QUICK):
            guidance["recommendations"] = [
                "Prioritize quick responses even if incomplete",
                "Send acknowledgments immediately",
                "Break complex responses into multiple messages",
                "Indicate when a full response will take longer",
                "Set expectations for timing"
            ]
        elif self.response_speed in (ResponseSpeed.RELAXED, ResponseSpeed.EXTENDED):
            guidance["recommendations"] = [
                "Prioritize completeness over speed",
                "Take time for thorough analysis",
                "Send comprehensive responses rather than incremental updates",
                "Acknowledge receipt and set expectations for response time"
            ]
        else:  # STANDARD
            guidance["recommendations"] = [
                "Balance timeliness with thoroughness",
                "Acknowledge time-sensitive matters promptly",
                "Set realistic expectations for complex requests",
                "Provide progress updates for extended tasks"
            ]
            
        return guidance
    
    def _get_structure_guidance(self) -> Dict[str, Any]:
        """Generate guidance for message structure based on preferences."""
        guidance = {
            "prefers_structured_responses": self.prefers_structured_responses,
            "recommendations": []
        }
        
        if self.prefers_structured_responses:
            guidance["recommendations"] = [
                "Use clear section headings",
                "Include numbered lists for sequential information",
                "Organize information hierarchically",
                "Use bullet points for key takeaways",
                "Include summaries at beginning or end"
            ]
        else:
            guidance["recommendations"] = [
                "Use a more conversational flow",
                "Integrate information naturally",
                "Use paragraph structure rather than formal sections",
                "Maintain logical progression of ideas",
                "Use formatting sparingly"
            ]
            
        return guidance
    
    def _get_examples_guidance(self) -> Dict[str, Any]:
        """Generate guidance for including examples based on preferences."""
        guidance = {
            "prefers_examples": self.prefers_examples,
            "recommendations": []
        }
        
        if self.prefers_examples:
            guidance["recommendations"] = [
                "Include concrete examples to illustrate concepts",
                "Use scenarios to demonstrate application",
                "Provide use cases relevant to the agent's context",
                "Compare and contrast with familiar situations",
                "Use examples of increasing complexity"
            ]
        else:
            guidance["recommendations"] = [
                "Focus on general principles",
                "Be concise with minimal illustrations",
                "Use examples only for complex concepts",
                "Keep examples brief and targeted",
                "Prioritize clarity over illustration"
            ]
            
        return guidance
    
    def _get_vocabulary_guidance(self) -> Dict[str, Any]:
        """Generate guidance for vocabulary complexity based on preferences."""
        guidance = {
            "vocabulary_level": self.vocabulary_level,
            "recommendations": []
        }
        
        if self.vocabulary_level > 0.7:  # Complex vocabulary
            guidance["recommendations"] = [
                f"Use domain-specific terminology freely",
                "Employ nuanced vocabulary",
                "Don't simplify technical concepts",
                "Assume familiarity with advanced concepts",
                "Use precise technical language"
            ]
        elif self.vocabulary_level < 0.3:  # Simple vocabulary
            guidance["recommendations"] = [
                "Use clear, simple language",
                "Avoid jargon and technical terms",
                "Define specialized terms when used",
                "Use concrete rather than abstract language",
                "Choose common words over specialized ones"
            ]
        else:  # Moderate vocabulary
            guidance["recommendations"] = [
                "Balance technical and accessible language",
                "Define specialized terms when first used",
                "Gauge understanding and adjust accordingly",
                "Match vocabulary to the specific topic",
                "Use analogies to explain complex concepts"
            ]
            
        return guidance
    
    def _get_general_recommendations(self) -> List[str]:
        """Generate general recommendations based on overall style."""
        recommendations = [
            f"Overall formality: {str(self.formality)}",
            f"Detail preference: {str(self.detail_level)}",
            f"Directness level: {str(self.directness)}",
            f"Emotional tone: {str(self.emotional_tone)}",
            f"Response speed: {str(self.response_speed)}"
        ]
        
        if self.style_notes:
            recommendations.append("Style notes: " + "; ".join(self.style_notes))
            
        return recommendations
    
    def is_compatible_with(self, other_style: 'CommunicationStyle') -> Tuple[bool, List[str]]:
        """
        Determine if this style is compatible with another communication style.
        
        Args:
            other_style: Another CommunicationStyle to compare with.
            
        Returns:
            Tuple of (is_compatible, incompatibility_reasons).
        """
        incompatibilities = []
        
        # Check for extreme differences in key dimensions
        if abs(self.formality.value - other_style.formality.value) > 2:
            incompatibilities.append(
                f"Formality mismatch: {str(self.formality)} vs {str(other_style.formality)}"
            )
            
        if abs(self.detail_level.value - other_style.detail_level.value) > 2:
            incompatibilities.append(
                f"Detail level mismatch: {str(self.detail_level)} vs {str(other_style.detail_level)}"
            )
            
        if abs(self.directness.value - other_style.directness.value) > 2:
            incompatibilities.append(
                f"Directness mismatch: {str(self.directness)} vs {str(other_style.directness)}"
            )
            
        if abs(self.emotional_tone.value - other_style.emotional_tone.value) > 2:
            incompatibilities.append(
                f"Emotional tone mismatch: {str(self.emotional_tone)} vs {str(other_style.emotional_tone)}"
            )
            
        if abs(self.response_speed.value - other_style.response_speed.value) > 2:
            incompatibilities.append(
                f"Response speed mismatch: {str(self.response_speed)} vs {str(other_style.response_speed)}"
            )
            
        # Check for binary preference mismatches
        if self.prefers_structured_responses != other_style.prefers_structured_responses:
            incompatibilities.append(
                f"Structure preference mismatch: {self.prefers_structured_responses} vs {other_style.prefers_structured_responses}"
            )
            
        if self.prefers_examples != other_style.prefers_examples:
            incompatibilities.append(
                f"Examples preference mismatch: {self.prefers_examples} vs {other_style.prefers_examples}"
            )
            
        # Check vocabulary level
        if abs(self.vocabulary_level - other_style.vocabulary_level) > 0.4:
            incompatibilities.append(
                f"Vocabulary level mismatch: {self.vocabulary_level} vs {other_style.vocabulary_level}"
            )
            
        return (len(incompatibilities) == 0, incompatibilities)
    
    def get_alignment_strategy(self, other_style: 'CommunicationStyle') -> Dict[str, Any]:
        """
        Generate a strategy to align this style with another style.
        
        Args:
            other_style: The target CommunicationStyle to align with.
            
        Returns:
            Dictionary containing alignment recommendations.
        """
        is_compatible, incompatibilities = self.is_compatible_with(other_style)
        
        strategy = {
            "is_compatible": is_compatible,
            "incompatibilities": incompatibilities,
            "alignment_needed": not is_compatible,
            "recommendations": []
        }
        
        if not is_compatible:
            # Generate recommendations to bridge the gaps
            if any("formality" in issue for issue in incompatibilities):
                strategy["recommendations"].append(
                    f"Adjust formality towards {str(other_style.formality)}"
                )
                
            if any("detail" in issue for issue in incompatibilities):
                strategy["recommendations"].append(
                    f"Adjust detail level towards {str(other_style.detail_level)}"
                )
                
            if any("directness" in issue for issue in incompatibilities):
                strategy["recommendations"].append(
                    f"Adjust directness towards {str(other_style.directness)}"
                )
                
            if any("tone" in issue for issue in incompatibilities):
                strategy["recommendations"].append(
                    f"Adjust emotional tone towards {str(other_style.emotional_tone)}"
                )
                
            if any("speed" in issue for issue in incompatibilities):
                strategy["recommendations"].append(
                    f"Adjust response timing to match {str(other_style.response_speed)} expectations"
                )
                
            if any("structure" in issue for issue in incompatibilities):
                if other_style.prefers_structured_responses:
                    strategy["recommendations"].append("Use more structured message format")
                else:
                    strategy["recommendations"].append("Use more conversational message format")
                    
            if any("examples" in issue for issue in incompatibilities):
                if other_style.prefers_examples:
                    strategy["recommendations"].append("Include more examples and illustrations")
                else:
                    strategy["recommendations"].append("Reduce examples, focus on key information")
                    
            if any("vocabulary" in issue for issue in incompatibilities):
                if other_style.vocabulary_level > self.vocabulary_level:
                    strategy["recommendations"].append("Use more specialized/technical vocabulary")
                else:
                    strategy["recommendations"].append("Simplify language and vocabulary")
        
        # Even if compatible, include minor alignment suggestions
        else:
            strategy["recommendations"] = [
                "Styles are generally compatible, but consider:",
                f"Fine-tune formality to match {str(other_style.formality)}",
                f"Adjust detail level slightly towards {str(other_style.detail_level)}",
                f"Consider {str(other_style.directness)} approach in sensitive discussions",
                f"Match {str(other_style.emotional_tone)} tone when appropriate"
            ]
            
        return strategy
    
    def clone_with_adjustments(self, **kwargs) -> 'CommunicationStyle':
        """
        Create a new CommunicationStyle with adjustments to certain parameters.
        
        Args:
            **kwargs: Style parameters to adjust.
            
        Returns:
            A new CommunicationStyle instance with the requested adjustments.
        """
        # Start with current values
        params = self.to_dict()
        
        # Override with provided values
        for key, value in kwargs.items():
            if key in params:
                params[key] = value
                
        # Convert back to CommunicationStyle
        return CommunicationStyle.from_dict(params)
