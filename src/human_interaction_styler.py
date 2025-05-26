#!/usr/bin/env python3
"""
Human Interaction Styler Module for Adaptive Bridge Builder

This module implements the HumanInteractionStyler class that builds and maintains
profiles of human communication preferences, detects emotional states, respects
cultural differences, and adapts responses accordingly while maintaining authenticity.
"""

import json
import logging
import re
import os
from typing import Dict, Any, List, Tuple, Optional, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timezone
import copy

from principle_engine import PrincipleEngine
from communication_style import (
    CommunicationStyle,
    FormalityLevel,
    DetailLevel,
    DirectnessLevel,
    EmotionalTone,
    ResponseSpeed
)
from emotional_intelligence import (
    EmotionalIntelligence,
    EmotionCategory,
    EmotionIntensity,
    EmotionSignal,
    InteractionType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HumanInteractionStyler")

class CulturalContext(Enum):
    """Enumeration of cultural communication contexts."""
    HIGH_CONTEXT = auto()      # Implicit communication, reliance on context
    LOW_CONTEXT = auto()       # Explicit communication, less reliance on context
    COLLECTIVIST = auto()      # Group harmony, indirect communication
    INDIVIDUALIST = auto()     # Direct communication, individual focus
    HIERARCHICAL = auto()      # Formal, status-conscious communication
    EGALITARIAN = auto()       # Informal, status-neutral communication
    POLYCHRONIC = auto()       # Flexible time, relationship-focused
    MONOCHRONIC = auto()       # Structured time, task-focused
    
    def __str__(self) -> str:
        return self.name.replace('_', ' ').title()

class CommunicationChannel(Enum):
    """Enumeration of communication channels."""
    TEXT_CHAT = auto()
    VOICE = auto()
    VIDEO = auto()
    EMAIL = auto()
    FORMAL_DOCUMENT = auto()
    SOCIAL_MEDIA = auto()
    API = auto()
    
    def __str__(self) -> str:
        return self.name.replace('_', ' ').title()

@dataclass
class HumanProfile:
    """
    Represents a human's communication preferences and patterns.
    """
    # Core identity
    human_id: str
    name: Optional[str] = None
    
    # Communication style preferences
    communication_style: CommunicationStyle = field(default_factory=lambda: CommunicationStyle(agent_id="human"))
    
    # Cultural context factors
    cultural_contexts: List[CulturalContext] = field(default_factory=list)
    language_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Emotional patterns
    emotional_expressiveness: float = 0.5  # 0.0 (reserved) to 1.0 (expressive)
    emotional_stability: float = 0.5  # 0.0 (volatile) to 1.0 (stable)
    primary_emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    
    # Communication content preferences
    topic_interests: Dict[str, float] = field(default_factory=dict)
    topic_expertise: Dict[str, float] = field(default_factory=dict)
    preferred_detail_level_by_topic: Dict[str, DetailLevel] = field(default_factory=dict)
    
    # Channel preferences
    channel_preferences: Dict[CommunicationChannel, float] = field(default_factory=dict)
    
    # Learning and adaptation preferences
    adaptability_preference: float = 0.5  # 0.0 (prefer consistency) to 1.0 (prefer adaptation)
    learning_style: Dict[str, float] = field(default_factory=dict)
    
    # Relationship factors
    relationship_history: List[Dict[str, Any]] = field(default_factory=list)
    trust_level: float = 0.5  # 0.0 (low trust) to 1.0 (high trust)
    
    # Profile metadata
    confidence_level: float = 0.1  # 0.0 (uncertain profile) to 1.0 (certain profile)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interaction_count: int = 0
    
    # Adaptive features
    observed_preferences: Dict[str, Any] = field(default_factory=dict)
    preference_strength: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the human profile to a dictionary."""
        result = {
            "human_id": self.human_id,
            "name": self.name,
            "communication_style": self.communication_style.to_dict(),
            "cultural_contexts": [ctx.name for ctx in self.cultural_contexts],
            "language_preferences": self.language_preferences,
            "emotional_expressiveness": self.emotional_expressiveness,
            "emotional_stability": self.emotional_stability,
            "primary_emotional_tone": self.primary_emotional_tone.name,
            "topic_interests": self.topic_interests,
            "topic_expertise": self.topic_expertise,
            "preferred_detail_level_by_topic": {
                topic: level.name for topic, level in self.preferred_detail_level_by_topic.items()
            },
            "channel_preferences": {
                channel.name: preference for channel, preference in self.channel_preferences.items()
            },
            "adaptability_preference": self.adaptability_preference,
            "learning_style": self.learning_style,
            "relationship_history": self.relationship_history,
            "trust_level": self.trust_level,
            "confidence_level": self.confidence_level,
            "last_updated": self.last_updated,
            "interaction_count": self.interaction_count,
            "observed_preferences": self.observed_preferences,
            "preference_strength": self.preference_strength
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanProfile':
        """Create a HumanProfile from a dictionary."""
        profile = cls(human_id=data.get("human_id", "unknown"))
        
        if "name" in data:
            profile.name = data["name"]
        
        if "communication_style" in data:
            profile.communication_style = CommunicationStyle.from_dict(data["communication_style"])
        
        if "cultural_contexts" in data:
            profile.cultural_contexts = [
                CulturalContext[ctx] for ctx in data["cultural_contexts"]
                if ctx in CulturalContext.__members__
            ]
        
        if "language_preferences" in data:
            profile.language_preferences = data["language_preferences"]
        
        if "emotional_expressiveness" in data:
            profile.emotional_expressiveness = data["emotional_expressiveness"]
        
        if "emotional_stability" in data:
            profile.emotional_stability = data["emotional_stability"]
        
        if "primary_emotional_tone" in data:
            try:
                profile.primary_emotional_tone = EmotionalTone[data["primary_emotional_tone"]]
            except KeyError:
                profile.primary_emotional_tone = EmotionalTone.NEUTRAL
        
        if "topic_interests" in data:
            profile.topic_interests = data["topic_interests"]
        
        if "topic_expertise" in data:
            profile.topic_expertise = data["topic_expertise"]
        
        if "preferred_detail_level_by_topic" in data:
            profile.preferred_detail_level_by_topic = {
                topic: DetailLevel[level] 
                for topic, level in data["preferred_detail_level_by_topic"].items()
            }
        
        if "channel_preferences" in data:
            profile.channel_preferences = {
                CommunicationChannel[channel]: preference
                for channel, preference in data["channel_preferences"].items()
                if channel in CommunicationChannel.__members__
            }
        
        if "adaptability_preference" in data:
            profile.adaptability_preference = data["adaptability_preference"]
        
        if "learning_style" in data:
            profile.learning_style = data["learning_style"]
        
        if "relationship_history" in data:
            profile.relationship_history = data["relationship_history"]
        
        if "trust_level" in data:
            profile.trust_level = data["trust_level"]
        
        if "confidence_level" in data:
            profile.confidence_level = data["confidence_level"]
        
        if "last_updated" in data:
            profile.last_updated = data["last_updated"]
        
        if "interaction_count" in data:
            profile.interaction_count = data["interaction_count"]
        
        if "observed_preferences" in data:
            profile.observed_preferences = data["observed_preferences"]
        
        if "preference_strength" in data:
            profile.preference_strength = data["preference_strength"]
        
        return profile

class HumanInteractionStyler:
    """
    Main class for adapting communication to human preferences.
    
    This class builds and maintains human communication profiles,
    detects emotional states, respects cultural differences,
    and adapts responses accordingly while maintaining authenticity.
    """
    
    def __init__(
        self,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        profiles_directory: Optional[str] = None
    ):
        """Initialize the HumanInteractionStyler."""
        self.emotional_intelligence = emotional_intelligence or EmotionalIntelligence()
        self.principle_engine = principle_engine
        self.profiles_directory = profiles_directory
        
        # Dictionary of human profiles
        self.human_profiles: Dict[str, HumanProfile] = {}
        
        # Load existing profiles if directory provided
        if profiles_directory and os.path.exists(profiles_directory):
            self._load_profiles()
        
        logger.info("HumanInteractionStyler initialized")
    
    def _load_profiles(self) -> None:
        """Load human profiles from the profiles directory."""
        if not os.path.exists(self.profiles_directory):
            os.makedirs(self.profiles_directory)
            logger.info(f"Created profiles directory: {self.profiles_directory}")
            return
        
        profile_files = [f for f in os.listdir(self.profiles_directory) if f.endswith('.json')]
        
        for file_name in profile_files:
            file_path = os.path.join(self.profiles_directory, file_name)
            try:
                with open(file_path, 'r') as f:
                    profile_data = json.load(f)
                    profile = HumanProfile.from_dict(profile_data)
                    self.human_profiles = {**self.human_profiles, profile.human_id: profile}
                    logger.info(f"Loaded profile for human: {profile.human_id}")
            except Exception as e:
                logger.error(f"Error loading profile from {file_path}: {e}")
    
    def save_profile(self, human_id: str) -> bool:
        """Save a human profile to disk."""
        if not self.profiles_directory or human_id not in self.human_profiles:
            return False
        
        if not os.path.exists(self.profiles_directory):
            os.makedirs(self.profiles_directory)
        
        profile = self.human_profiles[human_id]
        file_path = os.path.join(self.profiles_directory, f"{human_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.info(f"Saved profile for human: {human_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving profile for {human_id}: {e}")
            return False
    
    def get_or_create_profile(self, human_id: str, name: Optional[str] = None) -> HumanProfile:
        """Get an existing profile or create a new one if it doesn't exist."""
        if human_id in self.human_profiles:
            return self.human_profiles[human_id]
        
        profile = HumanProfile(human_id=human_id, name=name)
        self.human_profiles = {**self.human_profiles, human_id: profile}
        logger.info(f"Created new profile for human: {human_id}")
        return profile
    
    def detect_emotional_state(self, message: str) -> List[EmotionSignal]:
        """Detect the emotional state from a message using EmotionalIntelligence."""
        emotions, _, _ = self.emotional_intelligence.process_message(message)
        return emotions
    
    def update_profile_from_message(self, message: str, human_id: str) -> HumanProfile:
        """Update a human's profile based on a message."""
        profile = self.get_or_create_profile(human_id)
        
        # Detect emotions
        emotions = self.detect_emotional_state(message)
        
        # Update emotional patterns
        if emotions:
            # Calculate emotional expressiveness from intensity
            intensity_values = [e.intensity.value for e in emotions]
            avg_intensity = sum(intensity_values) / len(intensity_values) if intensity_values else 3
            normalized_intensity = (avg_intensity - 1) / 4  # Convert from 1-5 to 0-1
            
            # Update expressiveness with some smoothing
            profile.emotional_expressiveness = (
                profile.emotional_expressiveness * 0.8 + normalized_intensity * 0.2
            )
            
            # Update emotional tone
            primary_emotion = max(emotions, key=lambda e: e.confidence)
            if primary_emotion.category == EmotionCategory.JOY:
                tone = EmotionalTone.POSITIVE
            elif primary_emotion.category == EmotionCategory.SADNESS:
                tone = EmotionalTone.NEGATIVE
            elif primary_emotion.category == EmotionCategory.ANGER:
                tone = EmotionalTone.VERY_NEGATIVE
            elif primary_emotion.category == EmotionCategory.FEAR:
                tone = EmotionalTone.NEGATIVE
            elif primary_emotion.category == EmotionCategory.NEUTRAL:
                tone = EmotionalTone.NEUTRAL
            else:
                tone = EmotionalTone.NEUTRAL
                
            # Update tone with smoothing
            current = profile.primary_emotional_tone.value
            observed = tone.value
            confidence = primary_emotion.confidence
            profile.primary_emotional_tone = EmotionalTone(
                max(1, min(5, round(current * (1 - confidence) + observed * confidence)))
            )
        
        # Update interaction count and metadata
        profile.interaction_count += 1
        profile.last_updated = datetime.now(timezone.utc).isoformat()
        profile.confidence_level = min(0.95, profile.confidence_level + 0.01)
        
        # Save the updated profile
        if self.profiles_directory:
            self.save_profile(human_id)
        
        return profile
    
    def adapt_response(self, response: str, human_id: str) -> str:
        """
        Adapt a response based on the human's communication preferences.
        
        Args:
            response: The original response text.
            human_id: The ID of the human recipient.
            
        Returns:
            Adapted response text.
        """
        profile = self.get_or_create_profile(human_id)
        
        # Apply formality adaptation
        response = self._adapt_formality(response, profile.communication_style.formality)
        
        # Apply detail level adaptation
        response = self._adapt_detail_level(response, profile.communication_style.detail_level)
        
        # Apply directness adaptation
        response = self._adapt_directness(response, profile.communication_style.directness)
        
        # Apply emotional tone adaptation
        response = self._adapt_emotional_tone(response, profile.primary_emotional_tone)
        
        # Apply cultural adaptations
        for cultural_context in profile.cultural_contexts:
            response = self._adapt_cultural_context(response, cultural_context)
        
        # Apply authenticity principle
        response = self._apply_authenticity_principle(response, profile)
        
        return response
    
    def _adapt_formality(self, response: str, formality: FormalityLevel) -> str:
        """Adapt response formality."""
        if formality in (FormalityLevel.VERY_FORMAL, FormalityLevel.FORMAL):
            # Replace contractions
            response = re.sub(r"(\w+)'(\w+)", r"\1\2", response)  # don't -> do not
            # Add formal greeting/closing if missing
            if not re.search(r'^(Dear|Hello|Greetings)', response):
                response = f"Hello,\n\n{response}"
            if not re.search(r'(Sincerely|Regards|Respectfully),$', response):
                response = f"{response}\n\nRegards,"
        elif formality in (FormalityLevel.VERY_CASUAL, FormalityLevel.CASUAL):
            # Add casual greeting if missing
            if not re.search(r'^(Hey|Hi)', response):
                response = f"Hey,\n\n{response}"
            # Add casual closing if missing
            if not re.search(r'(Thanks|Cheers|Later)!$', response):
                response = f"{response}\n\nCheers!"
        
        return response
    
    def _adapt_detail_level(self, response: str, detail_level: DetailLevel) -> str:
        """Adapt response detail level."""
        paragraphs = response.split('\n\n')
        
        if detail_level in (DetailLevel.VERY_DETAILED, DetailLevel.DETAILED):
            # For detailed preferences, keep the full response
            return response
        elif detail_level == DetailLevel.BALANCED:
            # For balanced, keep all paragraphs but might trim very long ones
            return response
        elif detail_level in (DetailLevel.CONCISE, DetailLevel.VERY_CONCISE):
            # For concise, keep first and last paragraph, and add a note if content was trimmed
            if len(paragraphs) > 2:
                summary = f"{paragraphs[0]}\n\n{paragraphs[-1]}"
                if len(paragraphs) > 3:
                    summary += "\n\n(Note: Some details have been omitted for brevity.)"
                return summary
            return response
        
        return response
    
    def _adapt_directness(self, response: str, directness: DirectnessLevel) -> str:
        """Adapt response directness."""
        if directness in (DirectnessLevel.VERY_DIRECT, DirectnessLevel.DIRECT):
            # For direct preferences, move conclusions/recommendations to the beginning
            if "In conclusion" in response or "To summarize" in response:
                # Find conclusion paragraph
                conclusion_match = re.search(r'(In conclusion|To summarize).*?(\n\n|$)', response, re.DOTALL)
                if conclusion_match:
                    conclusion = conclusion_match.group(0)
                    rest = response.replace(conclusion, '')
                    return f"{conclusion}\n\n{rest}"
        elif directness in (DirectnessLevel.VERY_INDIRECT, DirectnessLevel.INDIRECT):
            # For indirect preferences, add softening phrases
            response = response.replace("You should", "You might consider")
            response = response.replace("You must", "It might be helpful to")
            response = response.replace("I recommend", "Perhaps consider")
        
        return response
    
    def _adapt_emotional_tone(self, response: str, tone: EmotionalTone) -> str:
        """Adapt response emotional tone."""
        if tone in (EmotionalTone.VERY_POSITIVE, EmotionalTone.POSITIVE):
            # Add positive phrases
            if not re.search(r'(excellent|great|wonderful|appreciate)', response, re.IGNORECASE):
                response = f"I'm pleased to {response[0].lower()}{response[1:]}"
        elif tone in (EmotionalTone.VERY_NEGATIVE, EmotionalTone.NEGATIVE):
            # Add acknowledging phrases for negative tone
            if not re.search(r'(understand your concern|acknowledge|recognize)', response, re.IGNORECASE):
                response = f"I understand your concern. {response}"
        
        return response
    
    def _adapt_cultural_context(self, response: str, cultural_context: CulturalContext) -> str:
        """Adapt response for cultural context."""
        if cultural_context == CulturalContext.HIGH_CONTEXT:
            # For high-context cultures, add context and relationship focus
            if not re.search(r'(our (relationship|conversation|discussion))', response, re.IGNORECASE):
                response = f"Considering our conversation, {response}"
        elif cultural_context == CulturalContext.COLLECTIVIST:
            # For collectivist cultures, emphasize group harmony and consensus
            response = response.replace("I think", "We might find")
            response = response.replace("You should", "We could")
        elif cultural_context == CulturalContext.HIERARCHICAL:
            # For hierarchical cultures, add proper formality and deference
            response = response.replace("Let me know", "Please let me know")
            response = response.replace("I think", "I would suggest")
        
        return response
    
    def _apply_authenticity_principle(self, response: str, profile: HumanProfile) -> str:
        """
        Apply the 'Authenticity Beyond Performance' principle to ensure genuine
        communication while adapting to the human's preferences.
        """
        # Add an authentic note if adaptation level is high
        if profile.confidence_level > 0.7 and profile.interaction_count > 5:
            authenticity_note = "\n\nNote: While I've adapted my communication style to your preferences, " \
                                "I remain committed to authentic and genuine communication."
            if authenticity_note not in response:
                response += authenticity_note
        
        return response


    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
# Example usage
def main() -> None:
    """Example usage of HumanInteractionStyler."""
    # Initialize the styler
    styler = HumanInteractionStyler()
    
    # Example human IDs
    humans = ["alice", "bob", "charlie"]
    
    # Example messages from each human
    messages = {
        "alice": "Hello, could you please provide a detailed analysis of this project? I would appreciate a formal report.",
        "bob": "Hey! Just give me the quick version. What's the bottom line here?",
        "charlie": "We need to understand the implications for our team. How does this affect us collectively?"
    }
    
    # Example response to adapt
    original_response = """
    Here is the project analysis:
    
    The project has several key components that need attention. First, the timeline is quite aggressive and may require additional resources. Second, the budget constraints may limit our options for technology choices.
    
    Additionally, there are some technical challenges related to integration with existing systems that need to be addressed early in the planning phase.
    
    In conclusion, this project is feasible but requires careful planning and resource allocation to be successful.
    """
    
    # Process each human's message and adapt the response
    for human_id in humans:
        # Update profile based on message
        profile = styler.update_profile_from_message(messages[human_id], human_id)
        
        # Adapt response based on profile
        adapted_response = styler.adapt_response(original_response, human_id)
        
        print(f"\n--- Adapted response for {human_id} ---")
        print(adapted_response)
        print("----------------------------\n")

if __name__ == "__main__":
    main()