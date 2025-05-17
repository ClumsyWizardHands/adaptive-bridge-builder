import json
import re
import datetime
import random
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from enum import Enum, auto
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionCategory(Enum):
    """Categories of emotions that can be detected in emoji sequences."""
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    SURPRISE = auto()
    DISGUST = auto()
    TRUST = auto()
    ANTICIPATION = auto()
    NEUTRAL = auto()
    MIXED = auto()
    
    # Secondary emotions
    EXCITEMENT = auto()
    CONTENTMENT = auto()
    PRIDE = auto()
    LOVE = auto()
    JEALOUSY = auto()
    ENVY = auto()
    SHAME = auto()
    GUILT = auto()
    ANXIETY = auto()
    HOPE = auto()
    DISAPPOINTMENT = auto()
    CONFUSION = auto()
    CURIOSITY = auto()
    EMPATHY = auto()
    GRATITUDE = auto()
    RELIEF = auto()
    EMBARRASSMENT = auto()

class EmotionIntensity(Enum):
    """Intensity levels for emotions."""
    VERY_LOW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()

class CulturalContext(Enum):
    """Cultural contexts for adapting emoji emotional interpretations."""
    WESTERN = auto()
    EASTERN_ASIAN = auto()
    SOUTH_ASIAN = auto()
    MIDDLE_EASTERN = auto()
    LATIN_AMERICAN = auto()
    AFRICAN = auto()
    OCEANIAN = auto()
    GLOBAL = auto()  # Universal interpretations

class ResponseTone(Enum):
    """Possible emotional tones for responses."""
    MATCHING = auto()         # Match the emotion expressed
    SUPPORTIVE = auto()       # Provide emotional support
    NEUTRAL = auto()          # Maintain neutral tone
    REDIRECTING = auto()      # Shift to more constructive emotion
    VALIDATING = auto()       # Acknowledge without matching intensity
    CALMING = auto()          # De-escalate high intensity emotions
    ENCOURAGING = auto()      # Positive and motivational
    EMPATHETIC = auto()       # Show understanding of emotions
    PROFESSIONAL = auto()     # Formal and business-like
    CURIOUS = auto()          # Interested and inquiring
    CELEBRATORY = auto()      # Joining in celebration

@dataclass
class EmotionalState:
    """Representation of an emotional state with probabilities."""
    primary_emotion: EmotionCategory
    primary_probability: float
    secondary_emotion: Optional[EmotionCategory] = None
    secondary_probability: float = 0.0
    intensity: EmotionIntensity = EmotionIntensity.MEDIUM
    confidence: float = 1.0
    context_notes: List[str] = field(default_factory=list)

@dataclass
class EmotionalShift:
    """Representation of a shift in emotional state during conversation."""
    from_state: EmotionalState
    to_state: EmotionalState
    magnitude: float  # 0.0 to 1.0, how significant the shift is
    detected_trigger: Optional[str] = None
    temporal_pattern: Optional[str] = None  # e.g., "gradual", "sudden", "oscillating"
    confidence: float = 1.0

@dataclass
class EmojiEmotionalResponse:
    """Representation of an emotional response in emoji form."""
    emoji_sequence: str
    emotional_intent: str
    principle_alignment_score: float
    alternative_sequences: List[Tuple[str, str, float]] = field(default_factory=list)  # (sequence, intent, alignment_score)
    cultural_adaptations: Dict[CulturalContext, str] = field(default_factory=dict)
    confidence: float = 1.0

class EmotionDetectionEngine:
    """Engine for detecting emotional content in emoji sequences."""
    
    def __init__(self, emoji_knowledge_base=None):
        """Initialize the emotion detection engine."""
        self.emoji_kb = emoji_knowledge_base
        self._initialize_emotion_mappings()
        
    def _initialize_emotion_mappings(self):
        """Initialize the mappings between emojis and emotions."""
        # Mapping emojis to emotions with probabilities
        self.primary_emotion_mappings = {
            # Joy
            "😀": (EmotionCategory.JOY, 0.9),
            "😄": (EmotionCategory.JOY, 0.9),
            "😊": (EmotionCategory.JOY, 0.8),
            "🎉": (EmotionCategory.JOY, 0.8),
            
            # Sadness
            "😢": (EmotionCategory.SADNESS, 0.9),
            "😭": (EmotionCategory.SADNESS, 0.95),
            "😔": (EmotionCategory.SADNESS, 0.8),
            
            # Anger
            "😠": (EmotionCategory.ANGER, 0.85),
            "😡": (EmotionCategory.ANGER, 0.95),
            "🤬": (EmotionCategory.ANGER, 0.95),
            
            # Fear
            "😨": (EmotionCategory.FEAR, 0.85),
            "😱": (EmotionCategory.FEAR, 0.95),
            
            # Surprise
            "😮": (EmotionCategory.SURPRISE, 0.85),
            "😲": (EmotionCategory.SURPRISE, 0.9),
            
            # Love
            "❤️": (EmotionCategory.LOVE, 0.9),
            "😍": (EmotionCategory.LOVE, 0.85),
            
            # Others
            "🤔": (EmotionCategory.CURIOSITY, 0.8),
            "😐": (EmotionCategory.NEUTRAL, 0.9),
        }
        
        # Modifiers that affect emotion intensity
        self.modifier_mappings = {
            "❗": 1.2,  # Intensifies
            "‼️": 1.5,  # Greatly intensifies
            "❓": 0.8,  # Reduces certainty
        }
        
        # Complex patterns
        self.complex_patterns = [
            (["😊", "👍"], EmotionCategory.CONTENTMENT, 0.85),
            (["😭", "💔"], EmotionCategory.SADNESS, 0.95),
            (["😡", "💢"], EmotionCategory.ANGER, 0.9),
        ]
        
    def detect_emotion(self, emoji_sequence: str) -> EmotionalState:
        """Detect the emotional content in an emoji sequence."""
        # Extract individual emojis
        emojis = self._extract_emojis(emoji_sequence)
        
        if not emojis:
            return EmotionalState(
                primary_emotion=EmotionCategory.NEUTRAL,
                primary_probability=1.0,
                confidence=0.5
            )
        
        # Check for complex patterns first
        for pattern, emotion, probability in self.complex_patterns:
            if all(emoji in emojis for emoji in pattern):
                intensity = self._calculate_intensity(emojis)
                return EmotionalState(
                    primary_emotion=emotion,
                    primary_probability=probability,
                    intensity=intensity,
                    confidence=probability
                )
        
        # Calculate emotion scores
        emotion_scores = self._calculate_emotion_scores(emojis)
        
        # Default to neutral if no emotions detected
        if not emotion_scores:
            return EmotionalState(
                primary_emotion=EmotionCategory.NEUTRAL,
                primary_probability=1.0,
                confidence=0.5
            )
            
        # Get primary emotion
        primary_emotion, primary_score = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Get secondary emotion if any
        secondary_emotion = None
        secondary_score = 0.0
        
        emotion_scores_copy = emotion_scores.copy()
        emotion_scores_copy.pop(primary_emotion)
        if emotion_scores_copy:
            secondary_emotion, secondary_score = max(emotion_scores_copy.items(), key=lambda x: x[1])
            if secondary_score < 0.3:  # Only significant secondary emotions
                secondary_emotion = None
                secondary_score = 0.0
        
        # Calculate intensity and confidence
        intensity = self._calculate_intensity(emojis)
        confidence = self._calculate_confidence(emotion_scores, primary_score)
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            primary_probability=primary_score,
            secondary_emotion=secondary_emotion,
            secondary_probability=secondary_score,
            intensity=intensity,
            confidence=confidence
        )
        
    def _extract_emojis(self, text: str) -> List[str]:
        """Extract individual emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U00002702-\U000027B0"  # Dingbats
            "]+"
        )
        
        return emoji_pattern.findall(text)
    
    def _calculate_emotion_scores(self, emojis: List[str]) -> Dict[EmotionCategory, float]:
        """Calculate scores for each emotion category based on the emojis."""
        scores = {}
        
        for emoji in emojis:
            if emoji in self.primary_emotion_mappings:
                emotion, score = self.primary_emotion_mappings[emoji]
                
                # Apply modifiers if present
                for mod_emoji, mod_factor in self.modifier_mappings.items():
                    if mod_emoji in emojis:
                        score *= mod_factor
                
                # Cap at 1.0
                score = min(score, 1.0)
                
                # Add to existing score or create new entry
                if emotion in scores:
                    scores[emotion] = scores[emotion] + score * (1 - scores[emotion])
                else:
                    scores[emotion] = score
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _calculate_intensity(self, emojis: List[str]) -> EmotionIntensity:
        """Calculate the emotional intensity."""
        # Factors that increase intensity
        intensity_factors = {
            "repetition": len(emojis) - len(set(emojis)),  # Repeated emojis
            "emphasis": emojis.count("❗") + emojis.count("‼️") * 2,
            "strong_emojis": sum(1 for e in emojis if e in ["😡", "😭", "😱"]),
        }
        
        intensity_score = sum(intensity_factors.values()) * 0.2 + 0.5  # 0.5 is baseline
        
        # Map to enum
        if intensity_score < 0.2:
            return EmotionIntensity.VERY_LOW
        elif intensity_score < 0.4:
            return EmotionIntensity.LOW
        elif intensity_score < 0.6:
            return EmotionIntensity.MEDIUM
        elif intensity_score < 0.8:
            return EmotionIntensity.HIGH
        else:
            return EmotionIntensity.VERY_HIGH
    
    def _calculate_confidence(self, emotion_scores: Dict[EmotionCategory, float], primary_score: float) -> float:
        """Calculate confidence in the emotion detection."""
        if len(emotion_scores) <= 1:
            return 0.9
            
        other_scores = [s for e, s in emotion_scores.items()]
        avg_other = sum(other_scores) / len(other_scores)
        score_ratio = primary_score / (avg_other + 0.001)  # Avoid division by zero
        
        confidence = min(0.9, 0.5 + 0.4 * (score_ratio - 1))
        return max(0.1, confidence)  # Ensure minimum confidence


class EmotionMappingSystem:
    """System for mapping emoji combinations to emotional states with probability scores."""
    
    def __init__(self, emoji_knowledge_base=None):
        """Initialize the emotion mapping system."""
        self.emoji_kb = emoji_knowledge_base
        self.emotion_detection = EmotionDetectionEngine(emoji_knowledge_base)
        
    def map_to_emotional_states(self, emoji_sequence: str) -> Dict[EmotionCategory, float]:
        """Map an emoji sequence to a dictionary of emotional states with probability scores."""
        # Get primary emotional state
        emotional_state = self.emotion_detection.detect_emotion(emoji_sequence)
        
        # Start with primary and secondary emotions
        emotion_map = {
            emotional_state.primary_emotion: emotional_state.primary_probability
        }
        
        if emotional_state.secondary_emotion:
            emotion_map[emotional_state.secondary_emotion] = emotional_state.secondary_probability
        
        # Add related emotions with lower probabilities
        for emotion, probability in emotion_map.copy().items():
            related = self._get_related_emotions(emotion)
            for rel_emotion, rel_score in related.items():
                if rel_emotion not in emotion_map:
                    emotion_map[rel_emotion] = probability * rel_score
                else:
                    emotion_map[rel_emotion] = max(emotion_map[rel_emotion], probability * rel_score)
        
        # Normalize
        total = sum(emotion_map.values())
        if total > 0:
            emotion_map = {k: v / total for k, v in emotion_map.items()}
        
        return emotion_map
    
    def _get_related_emotions(self, emotion: EmotionCategory) -> Dict[EmotionCategory, float]:
        """Get emotions related to the given emotion with relatedness scores."""
        related_emotions_map = {
            EmotionCategory.JOY: {
                EmotionCategory.EXCITEMENT: 0.7,
                EmotionCategory.CONTENTMENT: 0.6
            },
            EmotionCategory.SADNESS: {
                EmotionCategory.DISAPPOINTMENT: 0.6,
                EmotionCategory.SHAME: 0.4
            },
            EmotionCategory.ANGER: {
                EmotionCategory.DISGUST: 0.5,
                EmotionCategory.JEALOUSY: 0.4
            },
            EmotionCategory.FEAR: {
                EmotionCategory.ANXIETY: 0.7,
                EmotionCategory.SURPRISE: 0.4
            }
        }
        
        return related_emotions_map.get(emotion, {})


class EmotionalShiftTracker:
    """System for tracking emotional shifts in emoji conversations."""
    
    def __init__(self):
        """Initialize the emotional shift tracker."""
        self.emotion_detection = EmotionDetectionEngine()
        self.conversation_history = []
        self.emotion_history = []
        
    def track_conversation(self, emoji_sequence: str, is_user_message: bool = True) -> None:
        """Track a new message in the conversation."""
        # Record message
        message_record = {
            "sequence": emoji_sequence,
            "is_user_message": is_user_message,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.conversation_history.append(message_record)
        
        # Detect and record emotion
        emotional_state = self.emotion_detection.detect_emotion(emoji_sequence)
        emotion_record = {
            "state": emotional_state,
            "message_index": len(self.conversation_history) - 1,
            "is_user_emotion": is_user_message,
        }
        self.emotion_history.append(emotion_record)
        
    def detect_emotional_shift(self, window_size: int = 3) -> Optional[EmotionalShift]:
        """Detect if there has been a significant emotional shift."""
        if len(self.emotion_history) < 2:
            return None
        
        # Get user emotions only
        user_emotions = [record for record in self.emotion_history if record["is_user_emotion"]]
        if len(user_emotions) < 2:
            return None
        
        # Get current and previous emotions
        current_emotion = user_emotions[-1]["state"]
        previous_emotion = user_emotions[-2]["state"]
        
        # Calculate shift magnitude
        magnitude = self._calculate_shift_magnitude(previous_emotion, current_emotion)
        
        # Only report significant shifts
        if magnitude < 0.3:
            return None
        
        # Detect trigger and pattern
        trigger = self._detect_shift_trigger(previous_emotion, current_emotion)
        pattern = self._determine_temporal_pattern(user_emotions[-min(window_size, len(user_emotions)):])
        
        return EmotionalShift(
            from_state=previous_emotion,
            to_state=current_emotion,
            magnitude=magnitude,
            detected_trigger=trigger,
            temporal_pattern=pattern,
            confidence=min(previous_emotion.confidence, current_emotion.confidence)
        )
        
    def _calculate_shift_magnitude(self, from_state: EmotionalState, to_state: EmotionalState) -> float:
        """Calculate the magnitude of an emotional shift."""
        # Check if primary emotion changed
        if from_state.primary_emotion == to_state.primary_emotion:
            # Same emotion, check intensity change
            intensity_change = abs(self._intensity_to_value(to_state.intensity) - 
                                   self._intensity_to_value(from_state.intensity))
            return 0.3 * intensity_change  # Lower magnitude for intensity changes
        
        # Different emotions, high magnitude shift
        base_magnitude = 0.7
        distance = self._calculate_emotion_distance(from_state.primary_emotion, to_state.primary_emotion)
        intensity_factor = (self._intensity_to_value(to_state.intensity) + 
                           self._intensity_to_value(from_state.intensity)) / 2
        
        return base_magnitude * distance * intensity_factor
    
    def _intensity_to_value(self, intensity: EmotionIntensity) -> float:
        """Convert intensity enum to numeric value."""
        intensity_values = {
            EmotionIntensity.VERY_LOW: 0.2,
            EmotionIntensity.LOW: 0.4,
            EmotionIntensity.MEDIUM: 0.6,
            EmotionIntensity.HIGH: 0.8,
            EmotionIntensity.VERY_HIGH: 1.0,
        }
        return intensity_values.get(intensity, 0.6)
    
    def _calculate_emotion_distance(self, emotion1: EmotionCategory, emotion2: EmotionCategory) -> float:
        """Calculate conceptual distance between emotions."""
        # Basic distances between select emotions
        distances = {
            (EmotionCategory.JOY, EmotionCategory.SADNESS): 1.0,
            (EmotionCategory.JOY, EmotionCategory.ANGER): 0.9,
            (EmotionCategory.FEAR, EmotionCategory.RELIEF): 0.8,
        }
        
        key = (emotion1, emotion2)
        if key in distances:
            return distances[key]
        
        key = (emotion2, emotion1)
        if key in distances:
            return distances[key]
        
        if emotion1 == emotion2:
            return 0.0
        
        return 0.7  # Default distance
    
    def _detect_shift_trigger(self, prev_emotion: EmotionalState, curr_emotion: EmotionalState) -> str:
        """Detect what might have triggered an emotional shift."""
        # Check common patterns
        patterns = {
            (EmotionCategory.NEUTRAL, EmotionCategory.JOY): "positive_development",
            (EmotionCategory.NEUTRAL, EmotionCategory.SADNESS): "negative_development",
            (EmotionCategory.JOY, EmotionCategory.DISAPPOINTMENT): "expectation_violation",
            (EmotionCategory.FEAR, EmotionCategory.RELIEF): "threat_resolution",
        }
        
        key = (prev_emotion.primary_emotion, curr_emotion.primary_emotion)
        return patterns.get(key, "unknown_trigger")
    
    def _determine_temporal_pattern(self, emotion_records) -> str:
        """Determine the temporal pattern of emotional changes."""
        if len(emotion_records) < 3:
            return "insufficient_data"
        
        # Extract intensities
        intensities = [self._intensity_to_value(r["state"].intensity) for r in emotion_records]
        
        # Calculate differences
        diffs = [intensities[i+1] - intensities[i] for i in range(len(intensities)-1)]
        
        if all(d > 0 for d in diffs):
            return "steadily_increasing"
        elif all(d < 0 for d in diffs):
            return "steadily_decreasing"
        elif diffs[0] * diffs[-1] < 0:  # Sign change
            return "fluctuating"
        elif abs(diffs[0]) > 0.4:
            return "sudden_shift"
        else:
            return "gradual_change"


class EmojiResponseGenerator:
    """System for generating emotionally appropriate emoji responses."""
    
    def __init__(self, principle_engine=None, cultural_context=CulturalContext.GLOBAL):
        """Initialize the response generator."""
        self.principle_engine = principle_engine
        self.cultural_context = cultural_context
        self.detection_engine = EmotionDetectionEngine()
        self._initialize_response_templates()
        
    def _initialize_response_templates(self):
        """Initialize response templates for different emotions and tones."""
        self.response_templates = {
            EmotionCategory.JOY: {
                ResponseTone.MATCHING: ["😄👍", "😊✨", "🎉😃"],
                ResponseTone.SUPPORTIVE: ["😊👍", "🙌😄"],
            },
            EmotionCategory.SADNESS: {
                ResponseTone.SUPPORTIVE: ["🫂❤️", "💪💙", "🙏✨"],
                ResponseTone.ENCOURAGING: ["🌈✨", "🌻💙"],
            },
            EmotionCategory.ANGER: {
                ResponseTone.CALMING: ["🧘‍♀️✨", "🫂💙"],
                ResponseTone.VALIDATING: ["🫂👍", "👀💙"],
            },
            EmotionCategory.FEAR: {
                ResponseTone.SUPPORTIVE: ["🫂💙", "🙏✨"],
                ResponseTone.CALMING: ["🧘‍♀️🌊", "✨🛡️"],
            },
            EmotionCategory.NEUTRAL: {
                ResponseTone.NEUTRAL: ["👍", "🙂", "✨"],
                ResponseTone.CURIOUS: ["🤔✨", "👀❓"],
            },
        }
        
        # Cultural variations
        self.cultural_variations = {
            CulturalContext.EASTERN_ASIAN: {
                EmotionCategory.JOY: {
                    ResponseTone.MATCHING: ["😊🌸", "✨👍"],
                }
            },
            CulturalContext.WESTERN: {
                EmotionCategory.JOY: {
                    ResponseTone.MATCHING: ["😄🙌", "🎉👍"],
                }
            }
        }
    
    def generate_response(self, emoji_sequence: str, desired_tone: Optional[ResponseTone] = None) -> EmojiEmotionalResponse:
        """Generate an appropriate emoji response."""
        # Detect emotion in input
        input_state = self.detection_engine.detect_emotion(emoji_sequence)
        input_emotion = input_state.primary_emotion
        
        # Determine appropriate tone if not specified
        if desired_tone is None:
            desired_tone = self._determine_response_tone(input_state)
        
        # Get templates for this emotion and tone
        templates = self._get_response_templates(input_emotion, desired_tone)
        
        # Fallback to neutral if no templates
        if not templates:
            templates = self.response_templates[EmotionCategory.NEUTRAL][ResponseTone.NEUTRAL]
        
        # Select a template
        base_sequence = random.choice(templates)
        
        # Cultural adaptations
        cultural_adaptations = self._create_cultural_adaptations(input_emotion, desired_tone)
        
        # Check principle alignment
        alignment_score = 1.0
        alternatives = []
        
        if self.principle_engine:
            alignment_score = self._check_principle_alignment(base_sequence)
            if alignment_score < 0.7:
                alternatives = self._generate_alternatives(input_emotion, desired_tone)
        
        return EmojiEmotionalResponse(
            emoji_sequence=base_sequence,
            emotional_intent=f"Responding to {input_emotion.name} with {desired_tone.name} tone",
            principle_alignment_score=alignment_score,
            alternative_sequences=alternatives,
            cultural_adaptations=cultural_adaptations,
            confidence=input_state.confidence
        )
    
    def _determine_response_tone(self, emotional_state: EmotionalState) -> ResponseTone:
        """Determine appropriate response tone based on input state."""
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Default tone mappings
        if emotion == EmotionCategory.JOY:
            return ResponseTone.MATCHING
        elif emotion == EmotionCategory.SADNESS:
            return ResponseTone.SUPPORTIVE
        elif emotion == EmotionCategory.ANGER:
            return ResponseTone.CALMING if intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH] else ResponseTone.VALIDATING
        elif emotion == EmotionCategory.FEAR:
            return ResponseTone.CALMING if intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH] else ResponseTone.SUPPORTIVE
        else:
            return ResponseTone.NEUTRAL
    
    def _get_response_templates(self, emotion: EmotionCategory, tone: ResponseTone) -> List[str]:
        """Get appropriate response templates for the emotion and tone."""
        # Check if we have templates for this emotion and tone
        if emotion in self.response_templates and tone in self.response_templates[emotion]:
            return self.response_templates[emotion][tone]
        
        # Try to find any templates for this emotion
        if emotion in self.response_templates:
            for tone_templates in self.response_templates[emotion].values():
                if tone_templates:
                    return tone_templates
        
        # Fall back to neutral responses
        return self.response_templates[EmotionCategory.NEUTRAL][ResponseTone.NEUTRAL]
    
    def _create_cultural_adaptations(self, emotion: EmotionCategory, tone: ResponseTone) -> Dict[CulturalContext, str]:
        """Create culturally adapted responses."""
        adaptations = {}
        
        # Check if we have cultural variations for this emotion and tone
        for context, variations in self.cultural_variations.items():
            if emotion in variations and tone in variations[emotion]:
                adaptations[context] = random.choice(variations[emotion][tone])
        
        return adaptations
    
    def _check_principle_alignment(self, emoji_sequence: str) -> float:
        """Check alignment with agent principles."""
        # This would use the principle engine to evaluate alignment
        # For this implementation, return a placeholder score
        if self.principle_engine:
            # In a real implementation, call the principle engine
            return 0.9  # Placeholder
        return 1.0
    
    def _generate_alternatives(self, emotion: EmotionCategory, tone: ResponseTone) -> List[Tuple[str, str, float]]:
        """Generate alternative responses with better principle alignment."""
        # Get all templates for this emotion
        alternatives = []
        
        if emotion in self.response_templates:
            for alt_tone, templates in self.response_templates[emotion].items():
                if alt_tone != tone and templates:  # Different tone
                    seq = random.choice(templates)
                    alignment = self._check_principle_alignment(seq)
                    if alignment > 0.7:  # Only include good alignments
                        alternatives.append(
                            (seq, f"Alternative with {alt_tone.name} tone", alignment)
                        )
        
        return alternatives[:2]  # Limit to 2 alternatives


class EmojiEmotionalAnalyzer:
    """Main analyzer that combines all the emotional analysis components."""
    
    def __init__(self, principle_engine=None, cultural_context=CulturalContext.GLOBAL):
        """Initialize the analyzer with all sub-components."""
        self.detection_engine = EmotionDetectionEngine()
        self.mapping_system = EmotionMappingSystem()
        self.shift_tracker = EmotionalShiftTracker()
        self.response_generator = EmojiResponseGenerator(principle_engine, cultural_context)
        self.cultural_context = cultural_context
        
    def detect_emotion(self, emoji_sequence: str) -> EmotionalState:
        """Detect the emotional content in an emoji sequence."""
        return self.detection_engine.detect_emotion(emoji_sequence)
        
    def map_to_emotional_states(self, emoji_sequence: str) -> Dict[EmotionCategory, float]:
        """Map an emoji sequence to emotional states with probability scores."""
        return self.mapping_system.map_to_emotional_states(emoji_sequence)
        
    def track_conversation(self, emoji_sequence: str, is_user_message: bool = True) -> None:
        """Track a message in the conversation history."""
        self.shift_tracker.track_conversation(emoji_sequence, is_user_message)
        
    def detect_emotional_shift(self, window_size: int = 3) -> Optional[EmotionalShift]:
        """Detect significant emotional shifts in the conversation."""
        return self.shift_tracker.detect_emotional_shift(window_size)
        
    def generate_response(self, emoji_sequence: str, desired_tone: Optional[ResponseTone] = None) -> EmojiEmotionalResponse:
        """Generate an emotionally appropriate response to the input emojis."""
        return self.response_generator.generate_response(emoji_sequence, desired_tone)
        
    def adapt_to_cultural_context(self, new_context: CulturalContext) -> None:
        """Update the cultural context for responses."""
        self.cultural_context = new_context
        self.response_generator = EmojiResponseGenerator(
            principle_engine=self.response_generator.principle_engine,
            cultural_context=new_context
        )


# Example usage demonstrating the EmojiEmotionalAnalyzer
def demonstrate_emoji_emotional_analysis():
    """Demonstrate the capabilities of the EmojiEmotionalAnalyzer."""
    # Create analyzer
    analyzer = EmojiEmotionalAnalyzer()
    
    # Example 1: Detect emotion in a simple emoji sequence
    emoji_sequence = "😊👍"
    print(f"Analyzing emoji sequence: {emoji_sequence}")
    emotion = analyzer.detect_emotion(emoji_sequence)
    print(f"Detected emotion: {emotion.primary_emotion.name} with {emotion.primary_probability:.2f} probability")
    print(f"Intensity: {emotion.intensity.name}, Confidence: {emotion.confidence:.2f}")
    print()
    
    # Example 2: Map to emotional states with probabilities
    emoji_sequence = "😀😀🎉"
    print(f"Mapping emotional states for: {emoji_sequence}")
    emotion_map = analyzer.map_to_emotional_states(emoji_sequence)
    for emotion, probability in emotion_map.items():
        print(f"  {emotion.name}: {probability:.2f}")
    print()
    
    # Example 3: Track a conversation and detect shifts
    print("Tracking conversation for emotional shifts:")
    analyzer.track_conversation("😊👍")  # User starts happy
    analyzer.track_conversation("👍", is_user_message=False)  # Agent response
    analyzer.track_conversation("😊👍")  # User remains happy
    analyzer.track_conversation("✨", is_user_message=False)  # Agent response
    analyzer.track_conversation("😔")  # User becomes sad - emotional shift
    
    # Check for shift
    shift = analyzer.detect_emotional_shift()
    if shift:
        print(f"Detected shift from {shift.from_state.primary_emotion.name} to {shift.to_state.primary_emotion.name}")
        print(f"Magnitude: {shift.magnitude:.2f}, Trigger: {shift.detected_trigger}")
        print(f"Pattern: {shift.temporal_pattern}")
    else:
        print("No significant emotional shift detected")
    print()
    
    # Example 4: Generate appropriate response
    emoji_sequence = "😭💔"
    print(f"Generating response to: {emoji_sequence}")
    response = analyzer.generate_response(emoji_sequence)
    print(f"Generated response: {response.emoji_sequence}")
    print(f"Emotional intent: {response.emotional_intent}")
    print(f"Principle alignment: {response.principle_alignment_score:.2f}")
    print()
    
    # Example 5: Cultural adaptation
    print("Adapting to Eastern Asian cultural context")
    analyzer.adapt_to_cultural_context(CulturalContext.EASTERN_ASIAN)
    response = analyzer.generate_response("😊")
    print(f"Generated response: {response.emoji_sequence}")
    
    if CulturalContext.EASTERN_ASIAN in response.cultural_adaptations:
        print(f"Cultural adaptation: {response.cultural_adaptations[CulturalContext.EASTERN_ASIAN]}")


if __name__ == "__main__":
    demonstrate_emoji_emotional_analysis()
