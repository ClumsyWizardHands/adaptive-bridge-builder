#!/usr/bin/env python3
"""
Multilingual Engine for Adaptive Bridge Builder

This module provides language detection, translation, and cultural adaptation
capabilities to enable the Adaptive Bridge Builder agent to maintain consistent
principles and personality across multiple languages while respecting cultural
context and nuances.
"""

import logging
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field

from communication_style import CommunicationStyle, FormalityLevel, DetailLevel
from content_handler import ContentHandler, ContentFormat
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MultilingualEngine")


class Language(str, Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    HINDI = "hi"
    KOREAN = "ko"
    ITALIAN = "it"
    DUTCH = "nl"
    SWEDISH = "sv"
    POLISH = "pl"
    
    @classmethod
    def from_code(cls, code: str) -> 'Language':
        """Get Language enum from ISO code."""
        for lang in cls:
            if lang.value == code.lower():
                return lang
        # Default to English if language not found
        logger.warning(f"Language code '{code}' not found, defaulting to English")
        return Language.ENGLISH
    
    def __str__(self) -> str:
        """Return human-readable language name."""
        language_names = {
            Language.ENGLISH: "English",
            Language.SPANISH: "Spanish",
            Language.FRENCH: "French",
            Language.GERMAN: "German",
            Language.CHINESE: "Chinese",
            Language.JAPANESE: "Japanese",
            Language.ARABIC: "Arabic",
            Language.RUSSIAN: "Russian",
            Language.PORTUGUESE: "Portuguese",
            Language.HINDI: "Hindi",
            Language.KOREAN: "Korean",
            Language.ITALIAN: "Italian",
            Language.DUTCH: "Dutch",
            Language.SWEDISH: "Swedish",
            Language.POLISH: "Polish"
        }
        return language_names.get(self, self.name)


class CulturalContext(Enum):
    """Cultural dimensions that affect communication."""
    HIGH_CONTEXT = "high_context"          # Communication relies heavily on context and relationships
    LOW_CONTEXT = "low_context"            # Communication is explicit with less reliance on context
    INDIRECT = "indirect"                  # Preference for indirect communication
    DIRECT = "direct"                      # Preference for direct communication
    COLLECTIVIST = "collectivist"          # Emphasis on group harmony and consensus
    INDIVIDUALIST = "individualist"        # Emphasis on individual opinions and perspectives
    HIERARCHICAL = "hierarchical"          # Strong respect for authority and hierarchy
    EGALITARIAN = "egalitarian"            # Preference for equality in interactions
    FORMAL = "formal"                      # Preference for formal language and interactions
    INFORMAL = "informal"                  # Preference for informal language and interactions
    POLYCHRONIC = "polychronic"            # Flexible attitude toward time and scheduling
    MONOCHRONIC = "monochronic"            # Structured attitude toward time and scheduling


@dataclass
class LanguageProfile:
    """
    Represents linguistic and cultural characteristics of a language.
    
    Contains information about formality conventions, cultural contexts,
    common expressions, specialized vocabulary, and other language-specific
    considerations that affect communication.
    """
    
    language: Language
    cultural_dimensions: List[CulturalContext] = field(default_factory=list)
    formality_conventions: Dict[str, Any] = field(default_factory=dict)
    honorifics: Dict[str, str] = field(default_factory=dict)
    greeting_formats: List[str] = field(default_factory=list)
    specialized_vocabulary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cultural_references: Dict[str, str] = field(default_factory=dict)
    idiomatic_expressions: Dict[str, str] = field(default_factory=dict)
    writing_direction: str = "ltr"  # ltr = left-to-right, rtl = right-to-left
    default_formality: FormalityLevel = FormalityLevel.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert language profile to dictionary."""
        result = {
            "language": self.language.value,
            "cultural_dimensions": [dim.value for dim in self.cultural_dimensions],
            "formality_conventions": self.formality_conventions,
            "honorifics": self.honorifics,
            "greeting_formats": self.greeting_formats,
            "specialized_vocabulary": self.specialized_vocabulary,
            "cultural_references": self.cultural_references,
            "idiomatic_expressions": self.idiomatic_expressions,
            "writing_direction": self.writing_direction,
            "default_formality": str(self.default_formality)
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LanguageProfile':
        """Create LanguageProfile from dictionary."""
        # Convert raw values to enums
        language = Language.from_code(data["language"])
        cultural_dimensions = [CulturalContext(dim) for dim in data.get("cultural_dimensions", [])]
        
        # Handle formality level
        formality_str = data.get("default_formality", "Neutral")
        default_formality = next(
            (level for level in FormalityLevel if str(level) == formality_str),
            FormalityLevel.NEUTRAL
        )
        
        return cls(
            language=language,
            cultural_dimensions=cultural_dimensions,
            formality_conventions=data.get("formality_conventions", {}),
            honorifics=data.get("honorifics", {}),
            greeting_formats=data.get("greeting_formats", []),
            specialized_vocabulary=data.get("specialized_vocabulary", {}),
            cultural_references=data.get("cultural_references", {}),
            idiomatic_expressions=data.get("idiomatic_expressions", {}),
            writing_direction=data.get("writing_direction", "ltr"),
            default_formality=default_formality
        )
    
    def has_cultural_dimension(self, dimension: CulturalContext) -> bool:
        """Check if language has a specific cultural dimension."""
        return dimension in self.cultural_dimensions
    
    def get_domain_vocabulary(self, domain: str) -> Dict[str, str]:
        """Get specialized vocabulary for a specific domain."""
        return self.specialized_vocabulary.get(domain, {})
    
    def get_formality_guidance(self) -> Dict[str, Any]:
        """Get language-specific formality guidance."""
        return {
            "default_level": str(self.default_formality),
            "conventions": self.formality_conventions,
            "honorifics": self.honorifics
        }


class MultilingualEngine:
    """
    Enables multilingual communication while maintaining agent identity.
    
    This class provides:
    1. Language detection for incoming messages
    2. Principle-consistent translation and adaptation
    3. Cultural context awareness
    4. Terminology management across languages
    5. Communication style adaptation for language contexts
    """
    
    def __init__(
        self,
        agent_id: str,
        default_language: Language = Language.ENGLISH,
        principle_engine: Optional[PrincipleEngine] = None,
        content_handler: Optional[ContentHandler] = None
    ):
        """
        Initialize the multilingual engine.
        
        Args:
            agent_id: ID of the agent using this engine
            default_language: Default language for the agent
            principle_engine: Reference to the agent's principle engine
            content_handler: Handler for content formatting
        """
        self.agent_id = agent_id
        self.default_language = default_language
        self.principle_engine = principle_engine
        self.content_handler = content_handler or ContentHandler()
        
        # Dictionary of language profiles
        self.language_profiles: Dict[Language, LanguageProfile] = {}
        
        # Core language-independent identity elements
        self.core_identity: Dict[str, Any] = {
            "principles": [],  # Core principles that remain consistent across languages
            "personality_traits": [],  # Core personality traits
            "key_terminology": {}  # Domain-specific terms that need consistent translation
        }
        
        # Translation and detection services
        self.translation_service = None  # Would be initialized with actual translation service
        self.detection_service = None  # Would be initialized with actual language detection service
        
        # Cache for detected languages in conversations
        self.agent_language_cache: Dict[str, Language] = {}
        
        # Initialize with base language profiles
        self._initialize_base_profiles()
        
        logger.info(f"MultilingualEngine initialized for agent {agent_id} with default language {default_language}")
    
    def _initialize_base_profiles(self) -> None:
        """Initialize base language profiles for supported languages."""
        # English profile (used as reference)
        english_profile = LanguageProfile(
            language=Language.ENGLISH,
            cultural_dimensions=[
                CulturalContext.LOW_CONTEXT,
                CulturalContext.DIRECT,
                CulturalContext.INDIVIDUALIST,
                CulturalContext.EGALITARIAN,
                CulturalContext.INFORMAL,
                CulturalContext.MONOCHRONIC
            ],
            formality_conventions={
                "professional": "Use titles (Mr./Ms.) for initial contacts, shift to first names when relationship established",
                "academic": "Use professional titles (Dr./Prof.) until invited to use first names",
                "casual": "First names are standard, titles rarely used except in very formal situations"
            },
            honorifics={
                "general": "Mr./Ms./Mrs.",
                "academic": "Dr./Prof.",
                "formal": "Sir/Madam"
            },
            greeting_formats=[
                "Hello {name}",
                "Hi {name}",
                "Good morning/afternoon/evening {name}"
            ],
            default_formality=FormalityLevel.NEUTRAL
        )
        self.register_language_profile(english_profile)
        
        # Japanese profile (as an example of high-context culture)
        japanese_profile = LanguageProfile(
            language=Language.JAPANESE,
            cultural_dimensions=[
                CulturalContext.HIGH_CONTEXT,
                CulturalContext.INDIRECT,
                CulturalContext.COLLECTIVIST,
                CulturalContext.HIERARCHICAL,
                CulturalContext.FORMAL,
                CulturalContext.POLYCHRONIC
            ],
            formality_conventions={
                "professional": "Use surname with appropriate honorific (-san, -sama), avoid first names unless invited",
                "academic": "Use title and surname with -sensei honorific",
                "casual": "Surnames with -san for colleagues, first names rarely used except with close friends"
            },
            honorifics={
                "general": "-san",
                "respectable": "-sama",
                "teacher/master": "-sensei",
                "familiar": "-kun (male) / -chan (female, intimate)"
            },
            greeting_formats=[
                "こんにちは、{surname}{honorific}",
                "お疲れ様です、{surname}{honorific}"
            ],
            default_formality=FormalityLevel.FORMAL
        )
        self.register_language_profile(japanese_profile)
        
        # Additional profiles would be initialized here
        # (Simplified for brevity - in production, all supported languages would have detailed profiles)
    
    def register_language_profile(self, profile: LanguageProfile) -> None:
        """
        Register or update a language profile.
        
        Args:
            profile: The language profile to register
        """
        self.language_profiles = {**self.language_profiles, profile.language: profile}
        logger.info(f"Registered profile for language {profile.language}")
    
    def get_language_profile(self, language: Language) -> Optional[LanguageProfile]:
        """
        Get a language profile.
        
        Args:
            language: The language to get profile for
            
        Returns:
            Language profile if found, None otherwise
        """
        profile = self.language_profiles.get(language)
        if not profile:
            logger.warning(f"No profile found for language {language}, using default language profile")
            profile = self.language_profiles.get(self.default_language)
        return profile
    
    def detect_language(self, text: str) -> Language:
        """
        Detect the language of a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Detected language
        """
        # This is a placeholder. In a real implementation, this would use:
        # 1. A language detection library (e.g., langdetect, fastText)
        # 2. External API (e.g., Google Cloud Translation API)
        # 3. Or a locally trained model
        
        # Placeholder implementation for demonstration
        if self.detection_service:
            # Use actual detection service
            code = self.detection_service.detect(text)
            return Language.from_code(code)
        
        # Fallback detection logic (simplified for demo)
        # In production, this would be replaced with a proper language detection algorithm
        language_indicators = {
            "こんにちは": Language.JAPANESE,
            "你好": Language.CHINESE,
            "hola": Language.SPANISH,
            "bonjour": Language.FRENCH,
            "guten tag": Language.GERMAN,
            "ciao": Language.ITALIAN,
            "привет": Language.RUSSIAN,
            "안녕하세요": Language.KOREAN,
            "مرحبا": Language.ARABIC,
            "नमस्ते": Language.HINDI,
            "olá": Language.PORTUGUESE
        }
        
        text_lower = text.lower()
        for indicator, language in language_indicators.items():
            if indicator in text_lower:
                return language
                
        # Default to English if no language is detected
        return Language.ENGLISH
    
    def update_agent_language(self, agent_id: str, language: Language) -> None:
        """
        Update the cached language for an agent.
        
        Args:
            agent_id: ID of the agent
            language: Detected or specified language
        """
        self.agent_language_cache = {**self.agent_language_cache, agent_id: language}
        logger.debug(f"Updated language for agent {agent_id} to {language}")
    
    def get_agent_language(self, agent_id: str) -> Language:
        """
        Get the cached language for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The agent's language, or default language if not found
        """
        return self.agent_language_cache.get(agent_id, self.default_language)
    
    def translate_message(
        self,
        message: Any,
        from_language: Language,
        to_language: Language,
        preserve_terms: Optional[List[str]] = None
    ) -> Tuple[Any, bool]:
        """
        Translate a message between languages while preserving key elements.
        
        Args:
            message: The message to translate
            from_language: Source language
            to_language: Target language
            preserve_terms: List of terms to preserve in their original form
            
        Returns:
            Tuple of (translated_message, success)
        """
        # This is a placeholder. In a real implementation, this would:
        # 1. Use a translation service/API
        # 2. Implement term preservation logic
        # 3. Handle various content types
        
        # Simple implementation for demonstration
        if from_language == to_language:
            return message, True
        
        # Extract content for translation
        content = message
        if isinstance(message, dict) and "content" in message:
            content = message["content"]
        
        # Extract terms to preserve (if any)
        terms_to_preserve = preserve_terms or []
        for term_category, terms in self.core_identity.get("key_terminology", {}).items():
            terms_to_preserve.extend(terms)
        
        # Translation would happen here
        # For demonstration, we'll just return the original with a note
        translated_content = f"[Translated from {from_language} to {to_language}]: {content}"
        
        # If original was a dict with content field, update that field
        if isinstance(message, dict) and "content" in message:
            translated_message = dict(message)
            translated_message["content"] = translated_content
            translated_message["source_language"] = from_language.value
            translated_message["target_language"] = to_language.value
        else:
            translated_message = translated_content
        
        return translated_message, True
    
    def adapt_communication_style(
        self,
        style: CommunicationStyle,
        target_language: Language
    ) -> CommunicationStyle:
        """
        Adapt a communication style for a specific language and cultural context.
        
        Args:
            style: The base communication style
            target_language: The target language
            
        Returns:
            Adapted communication style
        """
        language_profile = self.get_language_profile(target_language)
        if not language_profile:
            return style
        
        # Start with a clone of the original style
        adapted_style = style.clone_with_adjustments()
        
        # Adjust formality based on language profile
        adapted_style.formality = language_profile.default_formality
        
        # Adjust directness based on cultural dimensions
        if language_profile.has_cultural_dimension(CulturalContext.INDIRECT):
            # More indirect cultures generally prefer less direct communication
            if adapted_style.directness.value > 3:  # If originally direct or very direct
                from communication_style import DirectnessLevel
                adapted_style.directness = DirectnessLevel.BALANCED
        elif language_profile.has_cultural_dimension(CulturalContext.DIRECT):
            # More direct cultures generally prefer more direct communication
            if adapted_style.directness.value < 3:  # If originally indirect or very indirect
                from communication_style import DirectnessLevel
                adapted_style.directness = DirectnessLevel.DIRECT
        
        # Adjust detail level based on cultural context
        if language_profile.has_cultural_dimension(CulturalContext.HIGH_CONTEXT):
            # High-context cultures often expect more background and context
            if adapted_style.detail_level.value < 3:  # If originally concise or very concise
                from communication_style import DetailLevel
                adapted_style.detail_level = DetailLevel.BALANCED
        elif language_profile.has_cultural_dimension(CulturalContext.LOW_CONTEXT):
            # Low-context cultures often prefer explicit, detailed information
            if adapted_style.detail_level.value < 4:  # If not already detailed
                from communication_style import DetailLevel
                adapted_style.detail_level = DetailLevel.DETAILED
        
        # Additional adaptations could be implemented based on other cultural dimensions
        
        return adapted_style
    
    def process_incoming_message(
        self,
        message: Any,
        sender_id: str,
        detect_language: bool = True
    ) -> Dict[str, Any]:
        """
        Process an incoming message, detecting language and preparing for response.
        
        Args:
            message: The message to process
            sender_id: ID of the message sender
            detect_language: Whether to detect language or use cached language
            
        Returns:
            Processed message with language information
        """
        # Extract content for language detection
        content = message
        if isinstance(message, dict) and "content" in message:
            content = message["content"]
        
        # Convert content to string for language detection if needed
        if not isinstance(content, str):
            # For non-string content, use cached language or default
            detected_language = self.get_agent_language(sender_id)
        else:
            # Detect language if requested, otherwise use cached language
            if detect_language:
                detected_language = self.detect_language(content)
                self.update_agent_language(sender_id, detected_language)
            else:
                detected_language = self.get_agent_language(sender_id)
        
        # Prepare processed message
        if isinstance(message, dict):
            processed = dict(message)
            processed["detected_language"] = detected_language.value
            processed["language_profile"] = self.get_language_profile(detected_language).to_dict()
        else:
            processed = {
                "content": message,
                "detected_language": detected_language.value,
                "language_profile": self.get_language_profile(detected_language).to_dict()
            }
        
        return processed
    
    def prepare_response(
        self,
        message: Any,
        recipient_id: str,
        source_language: Optional[Language] = None,
        target_language: Optional[Language] = None,
        style: Optional[CommunicationStyle] = None
    ) -> Dict[str, Any]:
        """
        Prepare a response in the appropriate language with cultural adaptation.
        
        Args:
            message: The message to prepare
            recipient_id: ID of the message recipient
            source_language: Original language of the message
            target_language: Target language for the response
            style: Communication style to adapt
            
        Returns:
            Prepared message with language adaptations
        """
        # Determine source and target languages
        actual_source = source_language or self.default_language
        actual_target = target_language or self.get_agent_language(recipient_id)
        
        # Extract content for translation
        content = message
        if isinstance(message, dict) and "content" in message:
            content = message["content"]
        
        # Translate content if languages differ
        if actual_source != actual_target:
            translated_content, success = self.translate_message(
                content,
                actual_source,
                actual_target
            )
            if not success:
                logger.warning(f"Translation failed from {actual_source} to {actual_target}")
                # Fallback to original content
                translated_content = content
        else:
            translated_content = content
        
        # Apply cultural adaptation
        language_profile = self.get_language_profile(actual_target)
        cultural_adaptations = self._apply_cultural_adaptations(
            content=translated_content,
            language=actual_target,
            style=style
        )
        
        # Prepare final message
        if isinstance(message, dict):
            prepared = dict(message)
            prepared["content"] = cultural_adaptations["content"]
            prepared["source_language"] = actual_source.value
            prepared["target_language"] = actual_target.value
            prepared["cultural_adaptations"] = cultural_adaptations["applied_adaptations"]
        else:
            prepared = {
                "content": cultural_adaptations["content"],
                "source_language": actual_source.value,
                "target_language": actual_target.value,
                "cultural_adaptations": cultural_adaptations["applied_adaptations"]
            }
        
        return prepared
    
    def _apply_cultural_adaptations(
        self,
        content: Any,
        language: Language,
        style: Optional[CommunicationStyle] = None
    ) -> Dict[str, Any]:
        """
        Apply cultural adaptations to content based on target language.
        
        Args:
            content: The content to adapt
            language: Target language
            style: Communication style to consider
            
        Returns:
            Dictionary with adapted content and list of applied adaptations
        """
        language_profile = self.get_language_profile(language)
        if not language_profile:
            return {"content": content, "applied_adaptations": []}
        
        applied_adaptations = []
        adapted_content = content
        
        # Adapt only string content
        if isinstance(content, str):
            # Apply greeting adaptations if content starts with a greeting
            common_greetings = ["hello", "hi", "good morning", "good afternoon", "greetings"]
            for greeting in common_greetings:
                if content.lower().startswith(greeting):
                    if language_profile.greeting_formats:
                        # Replace with culturally appropriate greeting
                        adapted_greeting = language_profile.greeting_formats[0].format(name="{name}")
                        adapted_content = adapted_content.replace(greeting, adapted_greeting, 1)
                        applied_adaptations.append(f"Adapted greeting to {language} convention")
                        break
            
            # Apply formality adaptations based on language profile
            if style and language_profile.default_formality != style.formality:
                # Note the formality adaptation
                applied_adaptations.append(
                    f"Adjusted formality to {language_profile.default_formality} (standard for {language})"
                )
            
            # Apply honorific adaptations if there's a name reference with Mr./Ms./etc.
            honorific_patterns = {
                "Mr.": "general",
                "Ms.": "general",
                "Mrs.": "general",
                "Dr.": "academic",
                "Prof.": "academic"
            }
            
            for eng_honorific, honorific_type in honorific_patterns.items():
                if eng_honorific in adapted_content and honorific_type in language_profile.honorifics:
                    target_honorific = language_profile.honorifics[honorific_type]
                    # Adaptation would be more sophisticated in a real implementation
                    # This is simplified for demonstration
                    if target_honorific != eng_honorific:
                        adapted_content = adapted_content.replace(eng_honorific, target_honorific)
                        applied_adaptations.append(
                            f"Adapted honorific from {eng_honorific} to {target_honorific}"
                        )
        
        return {
            "content": adapted_content,
            "applied_adaptations": applied_adaptations
        }
    
    def register_core_principle(self, principle: Dict[str, Any]) -> None:
        """
        Register a core principle that should be maintained across languages.
        
        Args:
            principle: Principle definition
        """
        self.core_identity["principles"].append(principle)
        logger.info(f"Registered core principle: {principle.get('name', 'Unnamed')}")
    
    def register_key_terminology(self, domain: str, terms: Dict[str, Dict[str, str]]) -> None:
        """
        Register key terminology that should be consistently translated.
        
        Args:
            domain: The domain/category for the terminology
            terms: Dictionary of terms with translations
        """
        if domain not in self.core_identity["key_terminology"]:
            self.core_identity["key_terminology"][domain] = {}
            
        self.core_identity["key_terminology"][domain].update(terms)
        logger.info(f"Registered {len(terms)} terms for domain: {domain}")
    
    def get_core_principles(self) -> List[Dict[str, Any]]:
        """Get the registered core principles."""
        return self.core_identity["principles"]
    
    def get_domain_terminology(self, domain: str) -> Dict[str, Dict[str, str]]:
        """Get terminology for a specific domain."""
        return self.core_identity["key_terminology"].get(domain, {})
    
    def support_new_language(self, language_code: str, base_profile: Optional[Dict[str, Any]] = None) -> Language:
        """
        Add support for a new language.
        
        Args:
            language_code: ISO 639-1 language code
            base_profile: Base profile data for the language
            
        Returns:
            Language enum for the added language
        """
        # This would be expanded in a real implementation to dynamically add languages
        # For now, we'll just verify if the language is already supported
        try:
            language = Language.from_code(language_code)
            if language in self.language_profiles:
                logger.info(f"Language {language} already supported")
                return language
                
            # If language is in enum but not in profiles, create profile
            if base_profile:
                profile_data = dict(base_profile)
                profile_data["language"] = language_code
                profile = LanguageProfile.from_dict(profile_data)
                self.register_language_profile(profile)
                logger.info(f"Added support for language: {language}")
                return language
            else:
                # Create minimal profile
                profile = LanguageProfile(
                    language=language,
                    default_formality=FormalityLevel.NEUTRAL
                )
                self.register_language_profile(profile)
                logger.info(f"Added minimal support for language: {language}")
                return language
                
        except ValueError:
            # Language not in enum - would require extending the enum
            logger.error(f"Cannot add language {language_code} - not in Language enum")
            return Language.ENGLISH
    
    def get_cultural_guidance(self, language: Language) -> Dict[str, Any]:
        """
        Get cultural guidance for communicating in a specific language.
        
        Args:
            language: The target language
            
        Returns:
            Dictionary with cultural guidance
        """
        profile = self.get_language_profile(language)
        if not profile:
            return {}
            
        # Generate comprehensive cultural guidance
        guidance = {
            "language": str(language),
            "cultural_dimensions": [dim.value for dim in profile.cultural_dimensions],
            "formality_guidance": profile.get_formality_guidance(),
            "greeting_formats": profile.greeting_formats,
            "writing_direction": profile.writing_direction,
            "communication_recommendations": []
        }
        
        # Add dimension-specific recommendations
        if profile.has_cultural_dimension(CulturalContext.HIGH_CONTEXT):
            guidance["communication_recommendations"].append(
                "Pay attention to non-verbal cues and context. What is not said may be as important as what is said."
            )
            
        if profile.has_cultural_dimension(CulturalContext.INDIRECT):
            guidance["communication_recommendations"].append(
                "Use indirect language and avoid confrontation. Phrase criticism or disagreement carefully."
            )
            
        if profile.has_cultural_dimension(CulturalContext.COLLECTIVIST):
            guidance["communication_recommendations"].append(
                "Emphasize group harmony and consensus rather than individual achievement or disagreement."
            )
            
        if profile.has_cultural_dimension(CulturalContext.HIERARCHICAL):
            guidance["communication_recommendations"].append(
                "Show proper respect for authority and status. Use appropriate honorifics and formal language."
            )
            
        return guidance
