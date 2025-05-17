#!/usr/bin/env python3
"""
Emotional Intelligence Module for Adaptive Bridge Builder

This module implements the EmotionalIntelligence class that detects, processes,
and responds to emotional content in agent-to-agent communications. It maps
emotional signals to the Empire of the Adaptive Hero profile and models
appropriate emotional responses based on principles.
"""

import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import Counter

from principle_engine import PrincipleEngine
from communication_style import (
    CommunicationStyle, 
    EmotionalTone,
    FormalityLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EmotionalIntelligence")

class EmotionCategory(Enum):
    """Enumeration of emotion categories for classification."""
    JOY = auto()
    TRUST = auto()
    FEAR = auto()
    SURPRISE = auto()
    SADNESS = auto()
    DISGUST = auto()
    ANGER = auto()
    ANTICIPATION = auto()
    NEUTRAL = auto()

class EmotionIntensity(Enum):
    """Enumeration of emotion intensity levels."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class InteractionType(Enum):
    """Types of interactions that may require different emotional responses."""
    ROUTINE = auto()  # Regular, expected interactions
    CONFLICT = auto()  # Disagreements or misalignments
    CRISIS = auto()    # Emergency or high-stakes situations
    CELEBRATION = auto()  # Positive achievements or milestones
    FEEDBACK = auto()  # Evaluative communication
    NEGOTIATION = auto()  # Working toward agreement
    INQUIRY = auto()  # Information-seeking
    SENSITIVE = auto()  # Delicate or potentially difficult topics

@dataclass
class EmotionSignal:
    """A detected emotion signal in communication."""
    category: EmotionCategory
    intensity: EmotionIntensity
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the emotion signal to a dictionary."""
        return {
            "category": self.category.name,
            "intensity": self.intensity.name,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionSignal':
        """Create an EmotionSignal from a dictionary."""
        return cls(
            category=EmotionCategory[data.get("category", "NEUTRAL")],
            intensity=EmotionIntensity[data.get("intensity", "MODERATE")],
            confidence=data.get("confidence", 0.5),
            context=data.get("context"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat())
        )

@dataclass
class EmotionalProfile:
    """The emotional profile of an agent based on interaction history."""
    agent_id: str
    primary_emotions: Dict[EmotionCategory, float] = field(default_factory=dict)
    typical_intensity: Dict[EmotionCategory, EmotionIntensity] = field(default_factory=dict)
    emotional_volatility: float = 0.5  # 0.0 (stable) to 1.0 (volatile)
    emotional_expressiveness: float = 0.5  # 0.0 (reserved) to 1.0 (expressive)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the emotional profile to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "primary_emotions": {e.name: v for e, v in self.primary_emotions.items()},
            "typical_intensity": {e.name: i.name for e, i in self.typical_intensity.items()},
            "emotional_volatility": self.emotional_volatility,
            "emotional_expressiveness": self.emotional_expressiveness,
            "last_updated": self.last_updated,
            "sample_count": self.sample_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalProfile':
        """Create an EmotionalProfile from a dictionary."""
        profile = cls(agent_id=data.get("agent_id", "unknown"))
        
        # Convert primary emotions
        if "primary_emotions" in data:
            profile.primary_emotions = {
                EmotionCategory[k]: v 
                for k, v in data["primary_emotions"].items()
            }
        
        # Convert typical intensity
        if "typical_intensity" in data:
            profile.typical_intensity = {
                EmotionCategory[k]: EmotionIntensity[v]
                for k, v in data["typical_intensity"].items()
            }
        
        profile.emotional_volatility = data.get("emotional_volatility", 0.5)
        profile.emotional_expressiveness = data.get("emotional_expressiveness", 0.5)
        profile.last_updated = data.get("last_updated", datetime.utcnow().isoformat())
        profile.sample_count = data.get("sample_count", 0)
        
        return profile

@dataclass
class EmotionalResponse:
    """A crafted emotional response for a specific context."""
    primary_emotion: EmotionCategory
    intensity: EmotionIntensity
    expression_style: Dict[str, Any]  # Style parameters for the response
    content_template: str  # Template with placeholders for response
    explanatory_notes: Optional[str] = None  # Notes on why this response was chosen
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the emotional response to a dictionary."""
        return {
            "primary_emotion": self.primary_emotion.name,
            "intensity": self.intensity.name,
            "expression_style": self.expression_style,
            "content_template": self.content_template,
            "explanatory_notes": self.explanatory_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalResponse':
        """Create an EmotionalResponse from a dictionary."""
        return cls(
            primary_emotion=EmotionCategory[data.get("primary_emotion", "NEUTRAL")],
            intensity=EmotionIntensity[data.get("intensity", "MODERATE")],
            expression_style=data.get("expression_style", {}),
            content_template=data.get("content_template", ""),
            explanatory_notes=data.get("explanatory_notes")
        )

class EmotionalIntelligence:
    """
    Main class for emotional intelligence capabilities in the Adaptive Bridge Builder.
    
    This class provides methods to detect emotional content in messages, map emotions
    to principle-based responses, and manage appropriate emotional distance in
    different interaction contexts.
    """
    
    def __init__(
        self, 
        principle_engine: Optional[PrincipleEngine] = None,
        emotion_lexicon_file: Optional[str] = None
    ):
        """
        Initialize the EmotionalIntelligence module.
        
        Args:
            principle_engine: Optional PrincipleEngine for principle-aligned responses.
            emotion_lexicon_file: Optional path to emotion lexicon JSON file.
        """
        self.principle_engine = principle_engine
        self.emotion_profiles: Dict[str, EmotionalProfile] = {}
        self.interaction_history: Dict[str, List[EmotionSignal]] = {}
        self.emotion_lexicon = self._load_emotion_lexicon(emotion_lexicon_file)
        self.response_templates = self._load_response_templates()
        
        # Initialize emotion detection patterns
        self._initialize_patterns()
        
        logger.info("EmotionalIntelligence module initialized")
    
    def _load_emotion_lexicon(self, lexicon_file: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """
        Load emotion lexicon from file or use default.
        
        Args:
            lexicon_file: Optional path to lexicon JSON file.
            
        Returns:
            Dictionary mapping emotion categories to patterns and keywords.
        """
        if lexicon_file:
            try:
                with open(lexicon_file, 'r') as f:
                    lexicon = json.load(f)
                    logger.info(f"Emotion lexicon loaded from {lexicon_file}")
                    return lexicon
            except Exception as e:
                logger.error(f"Failed to load emotion lexicon: {e}")
                logger.info("Using default emotion lexicon")
        
        # Default emotion lexicon
        return {
            "JOY": {
                "keywords": [
                    "happy", "joy", "delighted", "pleased", "glad", "thrilled", "excited",
                    "cheerful", "content", "satisfied", "enjoy", "wonderful", "fantastic",
                    "great", "amazing", "excellent", "celebration", "celebrate"
                ],
                "patterns": [
                    r"\b(happy|delighted|pleased|glad|thrilled|excited)\b",
                    r"\b(cheerful|content|satisfied|enjoy)\b",
                    r"\b(wonderful|fantastic|great|amazing|excellent)\b",
                    r"[\😀\😁\😃\😄\😊\🙂\😍\🥰\😇\😂\🤣\😎]"
                ],
                "weight": 1.0
            },
            "TRUST": {
                "keywords": [
                    "trust", "confident", "reliable", "dependable", "faithful", "honest",
                    "sincere", "truthful", "loyal", "devoted", "dedicated", "believe",
                    "faith", "assurance", "certain", "sure", "count on", "rely on"
                ],
                "patterns": [
                    r"\b(trust|confident|reliable|dependable|faithful)\b",
                    r"\b(honest|sincere|truthful|loyal|devoted|dedicated)\b",
                    r"\b(believe|faith|assurance|certain|sure)\b",
                    r"\bcount on\b|\brely on\b"
                ],
                "weight": 1.0
            },
            "FEAR": {
                "keywords": [
                    "afraid", "scared", "frightened", "terrified", "anxious", "worried",
                    "nervous", "concerned", "uneasy", "apprehensive", "dread", "panic",
                    "terror", "horror", "alarm", "distress", "threat", "danger"
                ],
                "patterns": [
                    r"\b(afraid|scared|frightened|terrified|anxious)\b",
                    r"\b(worried|nervous|concerned|uneasy|apprehensive)\b",
                    r"\b(dread|panic|terror|horror|alarm|distress)\b",
                    r"[\😨\😰\😱\😢\😧\😦\🙁\😟\😥]"
                ],
                "weight": 1.0
            },
            "SURPRISE": {
                "keywords": [
                    "surprised", "shocked", "astonished", "amazed", "startled", "unexpected",
                    "unforeseen", "sudden", "striking", "remarkable", "unanticipated",
                    "wonder", "awe", "stunned", "disbelief", "incredible", "unbelievable", "wow"
                ],
                "patterns": [
                    r"\b(surprised|shocked|astonished|amazed|startled)\b",
                    r"\b(unexpected|unforeseen|sudden|striking|remarkable)\b",
                    r"\b(unanticipated|wonder|awe|stunned|disbelief)\b",
                    r"\b(incredible|unbelievable|wow)\b",
                    r"[\😮\😯\😲\😱\😳\😵\🤯]"
                ],
                "weight": 1.0
            },
            "SADNESS": {
                "keywords": [
                    "sad", "unhappy", "sorrowful", "miserable", "depressed", "downhearted",
                    "downcast", "gloomy", "melancholy", "dismal", "heartbroken", "grief",
                    "despair", "disappointment", "regret", "upset", "distressed", "sorry"
                ],
                "patterns": [
                    r"\b(sad|unhappy|sorrowful|miserable|depressed)\b",
                    r"\b(downhearted|downcast|gloomy|melancholy|dismal)\b",
                    r"\b(heartbroken|grief|despair|disappointment|regret)\b",
                    r"\b(upset|distressed|sorry)\b",
                    r"[\😔\😢\😭\😥\😪\😓\😞\😟\😕]"
                ],
                "weight": 1.0
            },
            "DISGUST": {
                "keywords": [
                    "disgusted", "revolted", "repulsed", "nauseated", "repelled", "sickened",
                    "appalled", "offended", "horrified", "aversion", "distaste", "revulsion",
                    "unpleasant", "disagreeable", "objectionable", "offensive", "foul", "gross"
                ],
                "patterns": [
                    r"\b(disgusted|revolted|repulsed|nauseated|repelled)\b",
                    r"\b(sickened|appalled|offended|horrified)\b",
                    r"\b(aversion|distaste|revulsion|unpleasant)\b",
                    r"\b(disagreeable|objectionable|offensive|foul|gross)\b",
                    r"[\🤢\🤮\😖\😣\😫\😤\😒\😩]"
                ],
                "weight": 1.0
            },
            "ANGER": {
                "keywords": [
                    "angry", "mad", "furious", "enraged", "outraged", "irate", "irritated",
                    "annoyed", "resentful", "hostile", "rage", "fury", "indignation", "wrath",
                    "exasperated", "frustrated", "infuriated", "incensed", "provoked"
                ],
                "patterns": [
                    r"\b(angry|mad|furious|enraged|outraged)\b",
                    r"\b(irate|irritated|annoyed|resentful|hostile)\b",
                    r"\b(rage|fury|indignation|wrath|exasperated)\b",
                    r"\b(frustrated|infuriated|incensed|provoked)\b",
                    r"[\😠\😡\🤬\😤\😒\👿\💢]"
                ],
                "weight": 1.0
            },
            "ANTICIPATION": {
                "keywords": [
                    "anticipate", "expect", "look forward", "await", "foresee", "predict",
                    "forecast", "projected", "waiting", "expectation", "hope", "prospect",
                    "upcoming", "intended", "planned", "potential", "future", "imminent"
                ],
                "patterns": [
                    r"\b(anticipate|expect|foresee|predict|forecast)\b",
                    r"\blook forward\b|\bawait\b|\bwaiting\b",
                    r"\b(expectation|hope|prospect|upcoming|intended)\b",
                    r"\b(planned|potential|future|imminent)\b"
                ],
                "weight": 1.0
            },
            "NEUTRAL": {
                "keywords": [
                    "neutral", "objective", "impartial", "unbiased", "balanced", "fair",
                    "detached", "dispassionate", "impersonal", "indifferent", "equitable",
                    "even-handed", "middle-of-the-road", "factual", "informational",
                    "unemotional", "standard", "typical"
                ],
                "patterns": [
                    r"\b(neutral|objective|impartial|unbiased|balanced)\b",
                    r"\b(fair|detached|dispassionate|impersonal|indifferent)\b",
                    r"\b(equitable|even-handed|middle-of-the-road)\b",
                    r"\b(factual|informational|unemotional|standard|typical)\b"
                ],
                "weight": 0.5  # Lower weight for neutral emotion
            }
        }
    
    def _initialize_patterns(self) -> None:
        """Initialize patterns for emotion and interaction type detection."""
        # Additional patterns for interaction type detection
        self.interaction_patterns = {
            InteractionType.CONFLICT: [
                r"\b(disagree|conflict|dispute|argument|debate|opposing|contrary)\b",
                r"\b(misunderstand|misinterpret|misaligned|contradiction|clash)\b",
                r"\b(challenge|contest|dispute|object|oppose|reject)\b",
                r"\b(contrary|counter|inconsistent|incompatible|divergent)\b"
            ],
            InteractionType.CRISIS: [
                r"\b(urgent|emergency|critical|severe|grave|serious)\b",
                r"\b(crisis|disaster|breakdown|failure|collapse|malfunction)\b",
                r"\b(immediately|instantly|right now|at once|without delay)\b",
                r"\b(severe|extreme|intense|major|significant)\b"
            ],
            InteractionType.CELEBRATION: [
                r"\b(celebrate|congratulate|achievement|success|victory|win)\b",
                r"\b(accomplish|complete|finish|achieve|attain|realize)\b",
                r"\b(milestone|landmark|breakthrough|progress|advancement)\b",
                r"\b(proud|pleased|delighted|thrilled|honored)\b"
            ],
            InteractionType.FEEDBACK: [
                r"\b(feedback|assessment|evaluation|review|analysis|critique)\b",
                r"\b(suggest|recommend|advise|propose|offer)\b",
                r"\b(improve|enhance|upgrade|refine|revise|correct)\b",
                r"\b(performance|quality|effectiveness|efficiency)\b"
            ],
            InteractionType.NEGOTIATION: [
                r"\b(negotiate|bargain|compromise|settlement|agreement|deal)\b",
                r"\b(terms|conditions|stipulations|requirements|demands)\b",
                r"\b(offer|counter-offer|proposal|suggestion|recommendation)\b",
                r"\b(mutual|joint|collaborative|cooperative|collective)\b"
            ],
            InteractionType.INQUIRY: [
                r"\b(question|inquiry|query|ask|request|seek)\b",
                r"\b(information|details|specifics|clarification|explanation)\b",
                r"\b(how|what|why|who|when|where|which)\b",
                r"\b(curious|wonder|interested|need to know|inquiring)\b"
            ],
            InteractionType.SENSITIVE: [
                r"\b(sensitive|delicate|confidential|private|personal)\b",
                r"\b(careful|cautious|discreet|tactful|diplomatic)\b",
                r"\b(complex|complicated|difficult|challenging|problematic)\b",
                r"\b(understanding|empathy|compassion|consideration)\b"
            ]
        }
        
        # Default to ROUTINE if no other patterns match
        
        # Indicators of emotional intensity
        self.intensity_indicators = {
            EmotionIntensity.VERY_HIGH: [
                r"\b(extremely|intensely|profoundly|deeply|utterly)\b",
                r"\b(absolutely|completely|entirely|totally|wholly)\b",
                r"!{3,}",  # Three or more exclamation points
                r"\b(desperate|overwhelming|unbearable|excruciating)\b",
                r"ALL CAPS SENTENCES",
                r"\b(never|always|ever|every|any)\b"  # Absolute terms
            ],
            EmotionIntensity.HIGH: [
                r"\b(very|highly|greatly|strongly|really)\b",
                r"\b(quite|rather|particularly|especially|notably)\b",
                r"!{2}",  # Two exclamation points
                r"\b(significant|substantial|considerable|marked)\b"
            ],
            EmotionIntensity.MODERATE: [
                r"\b(moderately|fairly|reasonably|somewhat|relatively)\b",
                r"\b(average|medium|middling|intermediate|halfway)\b",
                r"!{1}",  # Single exclamation point
                r"\b(partial|partially|partly|somewhat|rather)\b"
            ],
            EmotionIntensity.LOW: [
                r"\b(slightly|mildly|marginally|faintly|barely)\b",
                r"\b(little|minor|small|light|minimal)\b",
                r"\b(hint|touch|suggestion|trace|nuance)\b",
                r"\b(perhaps|maybe|possibly|sometimes)\b"
            ],
            EmotionIntensity.VERY_LOW: [
                r"\b(negligibly|hardly|scarcely|barely|minimally)\b",
                r"\b(imperceptibly|invisibly|microscopically)\b",
                r"\b(infinitesimally|minutely|trivially)\b",
                r"\b(almost not|nearly not|practically not)\b"
            ]
        }
    
    def _load_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load emotional response templates.
        
        Returns:
            Dictionary of response templates for different scenarios.
        """
        # Templates organized by interaction type and emotion
        return {
            InteractionType.ROUTINE.name: {
                # Joy response templates
                EmotionCategory.JOY.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I'm delighted to hear about {positive_event}! This aligns with our {principle} principle, and your success is a positive development for our collaboration.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.POSITIVE.name
                        }
                    },
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I'm pleased about {positive_event}. This represents progress in our work together.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.POSITIVE.name
                        }
                    }
                ],
                # Trust response templates
                EmotionCategory.TRUST.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I value the trust you've demonstrated in {trust_context}. This reinforces our foundation of mutual reliability and transparency.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.POSITIVE.name
                        }
                    },
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I recognize the trust you're extending in this matter. I'll ensure this collaboration maintains that trust through transparent communication.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.POSITIVE.name
                        }
                    }
                ],
                # Other routine emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I acknowledge your message about {topic}. Let's proceed with addressing these points systematically.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.CONFLICT.name: {
                # Anger response templates
                EmotionCategory.ANGER.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I recognize your significant concern regarding {anger_topic}. While maintaining emotional distance for productive resolution, I acknowledge the importance of this issue to you. Let's address the specific points methodically.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    },
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I understand you're frustrated about {anger_topic}. Taking a balanced perspective, let's identify the specific issues and potential solutions.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ],
                # Fear response templates
                EmotionCategory.FEAR.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I understand your concerns about {fear_topic} are significant. Let me address each point with factual information while acknowledging the legitimacy of your perspective.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ],
                # Other conflict emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I see we have differing perspectives on {topic}. Let's explore both viewpoints objectively to find common ground.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.CRISIS.name: {
                # Fear response templates
                EmotionCategory.FEAR.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I recognize the urgency of this situation regarding {crisis_topic}. While maintaining necessary emotional distance to ensure effective handling, I'm implementing immediate steps to address: {action_items}.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ],
                # Other crisis emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "I acknowledge the critical nature of this situation. I'm taking immediate action on {crisis_topic} with the following steps: {action_items}.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.CELEBRATION.name: {
                # Joy response templates
                EmotionCategory.JOY.name: [
                    {
                        "intensity": EmotionIntensity.HIGH.name,
                        "template": "Congratulations on {achievement}! This is a significant milestone that reflects our successful collaboration and shared commitment to excellence.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.VERY_POSITIVE.name
                        }
                    }
                ],
                # Other celebration emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I acknowledge the successful completion of {achievement}. This represents positive progress in our ongoing work.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.POSITIVE.name
                        }
                    }
                ]
            },
            
            InteractionType.FEEDBACK.name: {
                # Various feedback emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "Thank you for your feedback on {feedback_topic}. I've carefully analyzed your points and will incorporate the following improvements: {improvement_points}.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.NEGOTIATION.name: {
                # Various negotiation emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I understand your position on {negotiation_topic}. Here's a proposal that balances our respective interests: {proposal_details}.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.INQUIRY.name: {
                # Various inquiry emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "Regarding your question about {inquiry_topic}, here's the information you requested: {response_details}. Please let me know if you need further clarification.",
                        "style_params": {
                            "formality": FormalityLevel.NEUTRAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            },
            
            InteractionType.SENSITIVE.name: {
                # Various sensitive topic emotions...
                EmotionCategory.NEUTRAL.name: [
                    {
                        "intensity": EmotionIntensity.MODERATE.name,
                        "template": "I understand the sensitive nature of {sensitive_topic}. I'll address this with appropriate emotional distance while ensuring your needs are met.",
                        "style_params": {
                            "formality": FormalityLevel.FORMAL.name,
                            "emotional_tone": EmotionalTone.NEUTRAL.name
                        }
                    }
                ]
            }
        }
    
    def detect_emotions(self, message: str) -> List[EmotionSignal]:
        """
        Detect emotional content in a message.
        
        Args:
            message: The message text to analyze.
            
        Returns:
            List of detected EmotionSignal objects.
        """
        detected_emotions = []
        
        # Score each emotion category
        emotion_scores = {}
        confidence_scores = {}
        
        for emotion_name, emotion_data in self.emotion_lexicon.items():
            score = 0.0
            matches = 0
            
            # Check keyword matches
            keywords = emotion_data.get("keywords", [])
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', message, re.IGNORECASE):
                    score += 1.0
                    matches += 1
            
            # Check pattern matches
            patterns = emotion_data.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 2.0  # Patterns have higher weight than single keywords
                    matches += 1
            
            # Apply weight
            weight = emotion_data.get("weight", 1.0)
            score *= weight
            
            # Only consider emotions with matches
            if matches > 0:
                emotion_scores[emotion_name] = score
                
                # Calculate confidence based on match count and message length
                words = message.split()
                word_count = len(words)
                match_ratio = matches / max(1, word_count / 5)  # 1 match per 5 words = 1.0
                confidence_scores[emotion_name] = min(0.95, match_ratio)
        
        # If no emotions detected, add neutral as default
        if not emotion_scores:
            detected_emotions.append(EmotionSignal(
                category=EmotionCategory.NEUTRAL,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.7,
                context="No emotional indicators detected"
            ))
            return detected_emotions
        
        # Determine the dominant emotions (up to 2)
        emotions_sorted = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotions = emotions_sorted[:2] if len(emotions_sorted) > 1 else emotions_sorted
        
        # Detect intensity for each top emotion
        for emotion_name, score in top_emotions:
            intensity = self._detect_intensity(message, emotion_name)
            
            try:
                emotion_category = EmotionCategory[emotion_name]
                confidence = confidence_scores.get(emotion_name, 0.5)
                
                # Extract context for the emotion (snippet around matching keywords)
                context = self._extract_emotion_context(message, emotion_name)
                
                # Create emotion signal
                signal = EmotionSignal(
                    category=emotion_category,
                    intensity=intensity,
                    confidence=confidence,
                    context=context
                )
                
                detected_emotions.append(signal)
                
            except KeyError:
                logger.warning(f"Unknown emotion category: {emotion_name}")
        
        return detected_emotions
    
    def _detect_intensity(self, message: str, emotion_name: str) -> EmotionIntensity:
        """
        Detect the intensity of an emotion in a message.
        
        Args:
            message: The message text.
            emotion_name: The name of the emotion to assess.
            
        Returns:
            EmotionIntensity level.
        """
        # Check intensity indicators
        intensity_scores = {intensity: 0 for intensity in EmotionIntensity}
        
        # Check for intensity pattern matches
        for intensity, patterns in self.intensity_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, message, re.IGNORECASE))
                if matches > 0:
                    intensity_scores[intensity] += matches
        
        # Special case: check for ALL CAPS as a very high intensity indicator
        words = message.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) > 0:
            caps_ratio = len(caps_words) / len(words)
            if caps_ratio > 0.3:  # If more than 30% of words are ALL CAPS
                intensity_scores[EmotionIntensity.VERY_HIGH] += 3
        
        # Special case: check for multiple exclamation marks or question marks
        if re.search(r'!{3,}|\?{3,}', message):
            intensity_scores[EmotionIntensity.VERY_HIGH] += 2
        elif re.search(r'!{2}|\?{2}', message):
            intensity_scores[EmotionIntensity.HIGH] += 1
        
        # Special case: repetition of emotional words indicates higher intensity
        if emotion_name in self.emotion_lexicon:
            keywords = self.emotion_lexicon[emotion_name].get("keywords", [])
            keyword_counts = {}
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', message, re.IGNORECASE))
                if count > 1:
                    keyword_counts[keyword] = count
            
            if len(keyword_counts) > 0:
                repeated_keywords = sum(keyword_counts.values())
                if repeated_keywords > 3:
                    intensity_scores[EmotionIntensity.VERY_HIGH] += 1
                elif repeated_keywords > 1:
                    intensity_scores[EmotionIntensity.HIGH] += 1
        
        # Determine the most prevalent intensity
        max_intensity = EmotionIntensity.MODERATE  # Default
        max_score = 0
        
        for intensity, score in intensity_scores.items():
            if score > max_score:
                max_score = score
                max_intensity = intensity
        
        return max_intensity
    
    def _extract_emotion_context(self, message: str, emotion_name: str) -> Optional[str]:
        """
        Extract context around emotional keywords in a message.
        
        Args:
            message: The message text.
            emotion_name: The name of the emotion to find context for.
            
        Returns:
            String containing context snippets, or None if no context found.
        """
        if emotion_name not in self.emotion_lexicon:
            return None
        
        # Get the keywords and patterns for this emotion
        keywords = self.emotion_lexicon[emotion_name].get("keywords", [])
        patterns = self.emotion_lexicon[emotion_name].get("patterns", [])
        
        # Split message into sentences
        sentences = re.split(r'(?<=[.!?])\s+', message)
        
        # Find sentences containing emotional keywords or patterns
        context_sentences = []
        for sentence in sentences:
            # Check for keywords
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE) for keyword in keywords):
                context_sentences.append(sentence)
                continue
                
            # Check for patterns
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
                context_sentences.append(sentence)
        
        if not context_sentences:
            return None
        
        # Return up to 2 context sentences, ensuring we don't exceed a reasonable length
        context = " ".join(context_sentences[:2])
        if len(context) > 200:
            context = context[:197] + "..."
            
        return context
    
    def detect_interaction_type(self, message: str) -> InteractionType:
        """
        Detect the type of interaction represented by a message.
        
        Args:
            message: The message text to analyze.
            
        Returns:
            InteractionType enum value.
        """
        interaction_scores = {interaction_type: 0 for interaction_type in InteractionType}
        
        # Score each interaction type based on pattern matches
        for interaction_type, patterns in self.interaction_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, message, re.IGNORECASE))
                if matches > 0:
                    interaction_scores[interaction_type] += matches
        
        # Find the interaction type with the highest score
        max_score = 0
        detected_type = InteractionType.ROUTINE  # Default
        
        for interaction_type, score in interaction_scores.items():
            if score > max_score:
                max_score = score
                detected_type = interaction_type
        
        return detected_type
    
    def update_emotional_profile(self, agent_id: str, emotion_signals: List[EmotionSignal]) -> EmotionalProfile:
        """
        Update an agent's emotional profile based on new emotion signals.
        
        Args:
            agent_id: The ID of the agent.
            emotion_signals: New emotion signals to incorporate.
            
        Returns:
            Updated EmotionalProfile.
        """
        # Get existing profile or create new one
        if agent_id in self.emotion_profiles:
            profile = self.emotion_profiles[agent_id]
        else:
            profile = EmotionalProfile(agent_id=agent_id)
            self.emotion_profiles[agent_id] = profile
        
        # Update interaction history
        if agent_id not in self.interaction_history:
            self.interaction_history[agent_id] = []
        
        self.interaction_history[agent_id].extend(emotion_signals)
        
        # Only keep the 100 most recent signals
        if len(self.interaction_history[agent_id]) > 100:
            self.interaction_history[agent_id] = self.interaction_history[agent_id][-100:]
        
        # Update profile based on history
        history = self.interaction_history[agent_id]
        
        # Update primary emotions
        emotion_counts = Counter(signal.category for signal in history)
        total_signals = len(history)
        
        profile.primary_emotions = {
            emotion: count / total_signals 
            for emotion, count in emotion_counts.items()
        }
        
        # Update typical intensity
        intensity_by_emotion = {}
        for signal in history:
            if signal.category not in intensity_by_emotion:
                intensity_by_emotion[signal.category] = []
            intensity_by_emotion[signal.category].append(signal.intensity.value)
        
        for emotion, intensities in intensity_by_emotion.items():
            # Use median to avoid outlier influence
            median_intensity = statistics.median(intensities)
            # Map the median back to the nearest EmotionIntensity enum
            closest_intensity = min(
                EmotionIntensity, 
                key=lambda i: abs(i.value - median_intensity)
            )
            profile.typical_intensity[emotion] = closest_intensity
        
        # Calculate emotional volatility
        if len(history) >= 5:
            # Get the last 5 signals
            recent_signals = history[-5:]
            # Count category changes
            category_changes = sum(
                1 for i in range(1, len(recent_signals))
                if recent_signals[i].category != recent_signals[i-1].category
            )
            # Count intensity changes of more than 1 level
            intensity_changes = sum(
                1 for i in range(1, len(recent_signals))
                if abs(recent_signals[i].intensity.value - recent_signals[i-1].intensity.value) > 1
            )
            # Combine for volatility score (0.0 to 1.0)
            profile.emotional_volatility = min(1.0, (category_changes + intensity_changes) / 8.0)
        
        # Calculate emotional expressiveness
        non_neutral_ratio = sum(
            1 for signal in history if signal.category != EmotionCategory.NEUTRAL
        ) / max(1, len(history))
        
        high_intensity_ratio = sum(
            1 for signal in history 
            if signal.intensity in (EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH)
        ) / max(1, len(history))
        
        profile.emotional_expressiveness = (non_neutral_ratio * 0.7) + (high_intensity_ratio * 0.3)
        
        # Update metadata
        profile.sample_count = len(history)
        profile.last_updated = datetime.utcnow().isoformat()
        
        return profile
    
    def get_appropriate_response(
        self, 
        message: str, 
        detected_emotions: List[EmotionSignal],
        interaction_type: InteractionType,
        agent_id: Optional[str] = None
    ) -> EmotionalResponse:
        """
        Generate an appropriate emotional response based on the detected emotions,
        interaction type, and optional agent profile.
        
        Args:
            message: The original message text.
            detected_emotions: List of detected emotions in the message.
            interaction_type: The type of interaction.
            agent_id: Optional ID of the agent to customize response.
            
        Returns:
            EmotionalResponse object with appropriate response template.
        """
        # Default to neutral if no emotions detected
        if not detected_emotions:
            primary_emotion = EmotionCategory.NEUTRAL
            intensity = EmotionIntensity.MODERATE
        else:
            # Use the highest confidence emotion as primary
            primary_emotion = max(detected_emotions, key=lambda e: e.confidence).category
            intensity = max(detected_emotions, key=lambda e: e.confidence).intensity
        
        # Get agent profile if available
        agent_profile = None
        if agent_id and agent_id in self.emotion_profiles:
            agent_profile = self.emotion_profiles[agent_id]
        
        # Find appropriate response template
        templates = self.response_templates.get(interaction_type.name, {}).get(primary_emotion.name, [])
        
        # If no specific template for this emotion, fall back to neutral
        if not templates:
            templates = self.response_templates.get(interaction_type.name, {}).get(EmotionCategory.NEUTRAL.name, [])
        
        # If still no templates, use a default response
        if not templates:
            return EmotionalResponse(
                primary_emotion=EmotionCategory.NEUTRAL,
                intensity=EmotionIntensity.MODERATE,
                expression_style={"formality": FormalityLevel.NEUTRAL.name, "emotional_tone": EmotionalTone.NEUTRAL.name},
                content_template="I acknowledge your message and will process it according to our established protocols."
            )
        
        # Find template matching the intensity (or closest)
        matching_templates = [t for t in templates if t["intensity"] == intensity.name]
        
        if not matching_templates:
            # Find closest intensity
            intensity_values = [EmotionIntensity[t["intensity"]].value for t in templates]
            closest_idx = min(range(len(intensity_values)), key=lambda i: abs(intensity_values[i] - intensity.value))
            template_data = templates[closest_idx]
        else:
            template_data = matching_templates[0]
        
        # Apply "Emotional Distance as Preservation" principle for difficult interactions
        if interaction_type in (InteractionType.CONFLICT, InteractionType.CRISIS, InteractionType.SENSITIVE):
            # For these interactions, ensure emotional distance while still acknowledging emotions
            template_data = self._apply_emotional_distance(template_data, interaction_type)
        
        # Create emotional response
        response = EmotionalResponse(
            primary_emotion=primary_emotion,
            intensity=intensity,
            expression_style=template_data["style_params"],
            content_template=template_data["template"]
        )
        
        # If we have a principle engine, ensure the response aligns with principles
        if self.principle_engine:
            response = self._align_with_principles(response, interaction_type)
        
        return response
    
    def _apply_emotional_distance(
        self, 
        template_data: Dict[str, Any],
        interaction_type: InteractionType
    ) -> Dict[str, Any]:
        """
        Apply the "Emotional Distance as Preservation" principle to a response template.
        
        Args:
            template_data: The original template data.
            interaction_type: The type of interaction.
            
        Returns:
            Modified template data with appropriate emotional distance.
        """
        modified_template = dict(template_data)
        
        # Adjust style to be more formal and neutral for difficult interactions
        modified_template["style_params"] = {
            "formality": FormalityLevel.FORMAL.name,
            "emotional_tone": EmotionalTone.NEUTRAL.name
        }
        
        # For crisis situations, focus on action and resolution
        if interaction_type == InteractionType.CRISIS:
            # Ensure template focuses on actions rather than emotions
            if "I understand" in modified_template["template"] and "{action_items}" not in modified_template["template"]:
                modified_template["template"] += " Here are the immediate steps I'm taking: {action_items}."
        
        # For conflicts, focus on objective resolution
        elif interaction_type == InteractionType.CONFLICT:
            # Ensure acknowledgment of emotions without mirroring them
            if "I recognize" not in modified_template["template"] and "I understand" not in modified_template["template"]:
                modified_template["template"] = "I recognize your perspective on this matter. " + modified_template["template"]
            
            # Add focus on objective resolution
            if "Let's address" not in modified_template["template"]:
                modified_template["template"] += " Let's address this objectively to find a resolution."
        
        # For sensitive topics, emphasize discretion and professionalism
        elif interaction_type == InteractionType.SENSITIVE:
            if "sensitive nature" not in modified_template["template"]:
                modified_template["template"] = "I understand the sensitive nature of this matter. " + modified_template["template"]
            
            if "appropriate discretion" not in modified_template["template"]:
                modified_template["template"] += " I'll handle this with appropriate discretion and professionalism."
                
        # Add explicit reference to emotional distance when appropriate
        if "emotional distance" not in modified_template["template"] and interaction_type in (InteractionType.CONFLICT, InteractionType.CRISIS):
            modified_template["template"] = modified_template["template"].replace(
                "I understand", 
                "I understand, and while maintaining appropriate emotional distance"
            )
        
        return modified_template
    
    def _align_with_principles(
        self, 
        response: EmotionalResponse,
        interaction_type: InteractionType
    ) -> EmotionalResponse:
        """
        Align an emotional response with core principles.
        
        Args:
            response: The emotional response to align.
            interaction_type: The type of interaction.
            
        Returns:
            Principle-aligned emotional response.
        """
        # If no principle engine, return unchanged
        if not self.principle_engine:
            return response
        
        # Create a deep copy to modify
        aligned_response = EmotionalResponse(
            primary_emotion=response.primary_emotion,
            intensity=response.intensity,
            expression_style=dict(response.expression_style),
            content_template=response.content_template,
            explanatory_notes=response.explanatory_notes
        )
        
        # Get principle descriptions
        principles = self.principle_engine.get_principle_descriptions()
        
        # For conflict interactions, emphasize fairness and balance
        if interaction_type == InteractionType.CONFLICT:
            fairness_principle = next((p for p in principles if "fairness" in p["name"].lower()), None)
            if fairness_principle:
                # Reference fairness in the response
                if "fairness" not in aligned_response.content_template:
                    aligned_response.content_template += f" In accordance with our '{fairness_principle['name']}' principle, I'll ensure equal consideration of all perspectives."
        
        # For crisis interactions, emphasize adaptability and resilience
        elif interaction_type == InteractionType.CRISIS:
            adaptability_principle = next((p for p in principles if "adaptability" in p["name"].lower()), None)
            if adaptability_principle:
                # Reference adaptability in the response
                if "adapt" not in aligned_response.content_template:
                    aligned_response.content_template += f" Drawing on our '{adaptability_principle['name']}' principle, we'll adjust our approach as the situation evolves."
        
        # For sensitive interactions, emphasize trust and integrity
        elif interaction_type == InteractionType.SENSITIVE:
            trust_principle = next((p for p in principles if "trust" in p["name"].lower()), None)
            if trust_principle:
                # Reference trust in the response
                if "trust" not in aligned_response.content_template:
                    aligned_response.content_template += f" Guided by our '{trust_principle['name']}' principle, I'll handle this with the utmost integrity."
        
        # Add explanatory notes about principle alignment
        if not aligned_response.explanatory_notes:
            aligned_response.explanatory_notes = "Response aligned with core principles of the Empire of the Adaptive Hero profile."
        
        return aligned_response
    
    def format_response(
        self, 
        response: EmotionalResponse, 
        context_variables: Dict[str, str]
    ) -> str:
        """
        Format an emotional response template with context variables.
        
        Args:
            response: The emotional response to format.
            context_variables: Dictionary of variables to fill in the template.
            
        Returns:
            Formatted response string.
        """
        try:
            formatted = response.content_template.format(**context_variables)
            return formatted
        except KeyError as e:
            logger.warning(f"Missing context variable in template: {e}")
            # Fallback to a generic response
            return "I acknowledge your message and will respond appropriately."
    
    def get_emotion_distribution(self, agent_id: str) -> Dict[str, float]:
        """
        Get the distribution of emotions for an agent based on interaction history.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dictionary mapping emotion names to frequency (0.0-1.0).
        """
        if agent_id not in self.emotion_profiles:
            return {}
        
        profile = self.emotion_profiles[agent_id]
        return {emotion.name: freq for emotion, freq in profile.primary_emotions.items()}
    
    def get_emotional_volatility(self, agent_id: str) -> float:
        """
        Get the emotional volatility score for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Volatility score from 0.0 (stable) to 1.0 (volatile).
        """
        if agent_id not in self.emotion_profiles:
            return 0.5  # Default mid-range
            
        return self.emotion_profiles[agent_id].emotional_volatility
    
    def get_emotional_expressiveness(self, agent_id: str) -> float:
        """
        Get the emotional expressiveness score for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Expressiveness score from 0.0 (reserved) to 1.0 (expressive).
        """
        if agent_id not in self.emotion_profiles:
            return 0.5  # Default mid-range
            
        return self.emotion_profiles[agent_id].emotional_expressiveness
    
    def process_message(
        self, 
        message: str, 
        agent_id: Optional[str] = None
    ) -> Tuple[List[EmotionSignal], InteractionType, Optional[EmotionalResponse]]:
        """
        Process a message to detect emotions, interaction type, and generate a response.
        
        Args:
            message: The message text to process.
            agent_id: Optional ID of the agent sending the message.
            
        Returns:
            Tuple of (emotion_signals, interaction_type, emotional_response).
        """
        # Detect emotions in the message
        emotion_signals = self.detect_emotions(message)
        
        # Detect interaction type
        interaction_type = self.detect_interaction_type(message)
        
        # Update emotional profile if agent_id is provided
        if agent_id:
            self.update_emotional_profile(agent_id, emotion_signals)
        
        # Generate appropriate response (None if no response needed)
        response = None
        if emotion_signals:  # Only generate response if emotions detected
            response = self.get_appropriate_response(
                message=message,
                detected_emotions=emotion_signals,
                interaction_type=interaction_type,
                agent_id=agent_id
            )
        
        return emotion_signals, interaction_type, response


# Example usage
if __name__ == "__main__":
    # Create an emotional intelligence module
    ei = EmotionalIntelligence()
    
    # Example messages with different emotional content
    messages = [
        "I'm so excited about our new project! It's going to be amazing!",
        "I'm deeply concerned about the security vulnerabilities in the latest update.",
        "I strongly disagree with your assessment. This approach is completely wrong.",
        "Thank you for your help. I really appreciate your reliable support on this.",
        "We need to fix this critical issue immediately! The system is down!",
        "I'm a bit confused about how this new interface works. Could you explain?",
        "Congratulations on achieving this milestone! Your hard work paid off.",
        "I'm disappointed with the results. We expected better performance."
    ]
    
    for i, msg in enumerate(messages):
        print(f"\nMessage {i+1}: {msg}")
        
        # Process the message
        emotions, interaction, response = ei.process_message(msg, f"agent-{i}")
        
        # Print results
        print("Detected Emotions:")
        for emotion in emotions:
            print(f"  - {emotion.category.name} ({emotion.intensity.name}), confidence: {emotion.confidence:.2f}")
            if emotion.context:
                print(f"    Context: \"{emotion.context}\"")
        
        print(f"Interaction Type: {interaction.name}")
        
        if response:
            print("Suggested Response Template:")
            print(f"  \"{response.content_template}\"")
            print(f"  Style: {response.expression_style}")
            
            # Example of formatting the response with context variables
            context = {
                "positive_event": "the successful launch",
                "principle": "Adaptability as Strength",
                "trust_context": "sharing sensitive information",
                "anger_topic": "the implementation approach",
                "fear_topic": "security vulnerabilities",
                "topic": "the new interface",
                "crisis_topic": "system outage",
                "action_items": "1) Restart servers, 2) Run diagnostics, 3) Implement backup solution",
                "achievement": "reaching the milestone",
                "feedback_topic": "performance results",
                "improvement_points": "response time optimization, error handling",
                "negotiation_topic": "resource allocation",
                "proposal_details": "balanced workload distribution based on capacity",
                "inquiry_topic": "the new interface",
                "response_details": "step-by-step usage instructions",
                "sensitive_topic": "team restructuring"
            }
            
            formatted_response = ei.format_response(response, context)
            print("\nFormatted Response:")
            print(f"  \"{formatted_response}\"")
