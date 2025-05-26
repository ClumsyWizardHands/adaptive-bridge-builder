import emoji
"""
EmojiKnowledgeBase Component for Adaptive Bridge Builder Agent

This component provides a comprehensive knowledge base for emoji mappings,
including context-specific meanings, cultural variations, usage frequency,
combination patterns, and versioning support.
"""

import json
import time
import copy
import re
import os
import hashlib
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


class EmojiDomain(Enum):
    """Domains/contexts for emoji usage."""
    GENERAL = "general"               # General/common usage
    TECHNICAL = "technical"           # Technical and development contexts
    BUSINESS = "business"             # Business and professional settings
    EDUCATION = "education"           # Educational contexts
    HEALTHCARE = "healthcare"         # Healthcare and medical contexts
    SOCIAL = "social"                 # Social media and casual communication
    EMOTIONAL = "emotional"           # Emotional expression-focused
    CULTURAL = "cultural"             # Culturally specific usages
    SCIENTIFIC = "scientific"         # Scientific and research contexts
    CREATIVE = "creative"             # Creative and artistic domains


class CulturalContext(Enum):
    """Cultural contexts for emoji interpretation."""
    GLOBAL = "global"                 # Generally understood across cultures
    WESTERN = "western"               # Western cultural interpretation
    EASTERN_ASIAN = "eastern_asian"   # East Asian interpretation (China, Japan, Korea)
    SOUTH_ASIAN = "south_asian"       # South Asian interpretation (India, Pakistan, etc.)
    LATIN_AMERICAN = "latin_american" # Latin American interpretation
    MIDDLE_EASTERN = "middle_eastern" # Middle Eastern interpretation
    AFRICAN = "african"               # African interpretation
    OCEANIC = "oceanic"               # Australian, New Zealand, Pacific Islands


class EmojiCategory(Enum):
    """Categories for emoji classification."""
    FACE_EMOTION = "face_emotion"     # Faces and emotions
    PERSON = "person"                 # People and body parts
    ANIMAL = "animal"                 # Animals and nature
    FOOD_DRINK = "food_drink"         # Food and drink
    ACTIVITY = "activity"             # Activities and sports
    TRAVEL = "travel"                 # Travel and places
    OBJECT = "object"                 # Objects and tools
    SYMBOL = "symbol"                 # Symbols and signs
    FLAG = "flag"                     # Flags
    ABSTRACT = "abstract"             # Abstract concepts
    TIME = "time"                     # Time-related
    WEATHER = "weather"               # Weather-related
    GESTURE = "gesture"               # Gestures and body language
    TECHNICAL = "technical"           # Technical symbols


class SentimentValue(Enum):
    """Sentiment values for emoji."""
    VERY_POSITIVE = "very_positive"   # Strongly positive emotion
    POSITIVE = "positive"             # Positive emotion
    NEUTRAL = "neutral"               # Neutral emotion
    NEGATIVE = "negative"             # Negative emotion
    VERY_NEGATIVE = "very_negative"   # Strongly negative emotion
    AMBIGUOUS = "ambiguous"           # Can be interpreted multiple ways


class FamiliarityLevel(Enum):
    """Levels of emoji familiarity among users."""
    UNIVERSAL = "universal"           # Universally recognized
    COMMON = "common"                 # Commonly recognized
    FAMILIAR = "familiar"             # Familiar to regular emoji users
    SPECIALIZED = "specialized"       # Known in specific domains/contexts
    RARE = "rare"                     # Rarely used or recognized
    NOVEL = "novel"                   # New or emerging usage


@dataclass
class EmojiMetadata:
    """Comprehensive metadata for an emoji."""
    # Basic information
    emoji: str                                            # The emoji character(s)
    unicode_representation: str                           # Unicode representation
    short_name: str                                       # Short name like ":smile:"
    description: str                                      # Human-readable description
    keywords: List[str] = field(default_factory=list)     # Related keywords
    category: EmojiCategory = EmojiCategory.OBJECT        # Primary category
    
    # Semantic information
    primary_meaning: str = ""                             # Primary meaning in English
    alternate_meanings: List[str] = field(default_factory=list)  # Alternative meanings
    
    # Domain-specific meanings
    domain_meanings: Dict[EmojiDomain, str] = field(default_factory=dict)  # Meanings in different domains
    
    # Cultural variations
    cultural_meanings: Dict[CulturalContext, str] = field(default_factory=dict)  # Cultural interpretations
    
    # Usage statistics
    frequency_score: float = 0.0                          # Usage frequency (0.0-1.0)
    familiarity_level: FamiliarityLevel = FamiliarityLevel.FAMILIAR  # How widely recognized
    context_specificity: float = 0.0                      # How context-dependent (0.0-1.0)
    
    # Emotional/semantic properties
    sentiment: SentimentValue = SentimentValue.NEUTRAL    # Emotional sentiment
    formality_score: float = 0.5                          # Formality level (0.0-1.0, 0=informal)
    intensity_score: float = 0.5                          # Emotional intensity (0.0-1.0)
    ambiguity_score: float = 0.0                          # Ambiguity level (0.0-1.0)
    
    # Combination information
    common_prefixes: List[str] = field(default_factory=list)   # Emojis often used before this one
    common_suffixes: List[str] = field(default_factory=list)   # Emojis often used after this one
    
    # Versioning information
    first_version: str = "1.0"                            # Version when first added
    last_updated: str = "1.0"                             # Version when last updated
    deprecation_status: bool = False                      # Whether it's deprecated
    replacement_emoji: Optional[str] = None               # Recommended replacement if deprecated
    
    # Timestamp information
    added_date: float = field(default_factory=time.time)  # When this emoji was added to the KB
    last_modified: float = field(default_factory=time.time)  # When this emoji was last modified


@dataclass
class ConceptMapping:
    """Mapping of a concept to emojis across different contexts."""
    # Basic information
    concept: str                                          # The concept being mapped
    description: str                                      # Description of the concept
    
    # Primary emoji mappings
    primary_emoji: Dict[EmojiDomain, str] = field(default_factory=dict)  # Best emoji by domain
    
    # Alternative emoji mappings
    alternative_emojis: Dict[EmojiDomain, List[str]] = field(default_factory=dict)  # Alternatives by domain
    
    # Cultural variations
    cultural_variations: Dict[CulturalContext, str] = field(default_factory=dict)  # Cultural preferences
    
    # Complex concept representation
    emoji_combinations: Dict[EmojiDomain, List[str]] = field(default_factory=dict)  # Emoji sequences by domain
    
    # Related concepts
    related_concepts: List[str] = field(default_factory=list)  # Semantically related concepts
    
    # Versioning information
    first_version: str = "1.0"                            # Version when first added
    last_updated: str = "1.0"                             # Version when last updated
    
    # Timestamp information
    added_date: float = field(default_factory=time.time)  # When this mapping was added
    last_modified: float = field(default_factory=time.time)  # When this mapping was last modified


@dataclass
class EmojiCombinationPattern:
    """Pattern for combining emojis to represent complex concepts."""
    # Basic information
    name: str                                             # Name of the combination pattern
    description: str                                      # Description of the pattern
    
    # Pattern components
    emoji_sequence: List[str]                             # Sequence of emojis in the pattern
    resulting_concept: str                                # The concept expressed by the combination
    
    # Context information
    primary_domain: EmojiDomain = EmojiDomain.GENERAL     # Primary domain for this pattern
    applicable_domains: List[EmojiDomain] = field(default_factory=list)  # Where this pattern is applicable
    cultural_context: CulturalContext = CulturalContext.GLOBAL  # Primary cultural context
    
    # Usage information
    frequency_score: float = 0.0                          # Usage frequency (0.0-1.0)
    recognition_score: float = 0.0                        # How widely recognized (0.0-1.0)
    
    # Versioning information
    first_version: str = "1.0"                            # Version when first added
    last_updated: str = "1.0"                             # Version when last updated
    
    # Timestamp information
    added_date: float = field(default_factory=time.time)  # When this pattern was added
    last_modified: float = field(default_factory=time.time)  # When this pattern was last modified


@dataclass
class KnowledgeBaseVersion:
    """Version information for the emoji knowledge base."""
    # Version identifiers
    version: str                                          # Version number (semantic versioning)
    name: str                                             # Version name
    
    # Changes in this version
    new_emojis: List[str] = field(default_factory=list)   # Newly added emojis
    updated_emojis: List[str] = field(default_factory=list)  # Updated emoji metadata
    deprecated_emojis: List[str] = field(default_factory=list)  # Deprecated emojis
    
    new_concepts: List[str] = field(default_factory=list)  # Newly added concepts
    updated_concepts: List[str] = field(default_factory=list)  # Updated concept mappings
    
    new_patterns: List[str] = field(default_factory=list)  # Newly added combination patterns
    updated_patterns: List[str] = field(default_factory=list)  # Updated patterns
    
    # Timestamp information
    release_date: float = field(default_factory=time.time)  # Release timestamp
    notes: str = ""                                       # Release notes


class EmojiKnowledgeBase:
    """
    Comprehensive knowledge base for emoji mappings, meanings, and combinations.
    
    This component serves as the foundation for the emoji communication system,
    providing rich contextual information about emojis across different domains,
    cultures, and use cases.
    """
    
    def __init__(
        self,
        load_default: bool = True,
        storage_path: str = "./emoji_kb_data"
    ):
        """
        Initialize the EmojiKnowledgeBase.
        
        Args:
            load_default: Whether to load the default emoji dataset on initialization
            storage_path: Path to the storage directory for the knowledge base data
        """
        # Storage for emoji information
        self.emojis: Dict[str, EmojiMetadata] = {}
        
        # Storage for concept mappings
        self.concepts: Dict[str, ConceptMapping] = {}
        
        # Storage for combination patterns
        self.patterns: Dict[str, EmojiCombinationPattern] = {}
        
        # Version history
        self.versions: List[KnowledgeBaseVersion] = []
        self.current_version = "1.0.0"
        
        # Storage paths
        self.storage_path = storage_path
        self.ensure_storage_directory()
        
        # Statistics
        self.last_updated = time.time()
        self.emoji_count = 0
        self.concept_count = 0
        self.pattern_count = 0
        
        # Load default data if requested
        if load_default:
            self.load_default_dataset()
    
    def ensure_storage_directory(self) -> None:
        """Ensure the storage directory exists."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        # Create subdirectories for different types of data
        for subdir in ["emojis", "concepts", "patterns", "versions"]:
            path = os.path.join(self.storage_path, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def load_default_dataset(self) -> None:
        """Load default emoji dataset."""
        # This would typically load from a bundled dataset file
        # For demonstration, we'll add a few example entries
        
        # Add some basic emojis
        self.add_emoji(
            emoji="😊",
            unicode_representation="U+1F60A",
            short_name=":smiling_face_with_smiling_eyes:",
            description="Smiling face with smiling eyes",
            keywords=["smile", "happy", "joy", "pleased"],
            category=EmojiCategory.FACE_EMOTION,
            primary_meaning="Happiness or joy",
            alternate_meanings=["Satisfaction", "Contentment"],
            sentiment=SentimentValue.POSITIVE,
            frequency_score=0.95,
            familiarity_level=FamiliarityLevel.UNIVERSAL
        )
        
        self.add_emoji(
            emoji="👨‍💻",
            unicode_representation="U+1F468 U+200D U+1F4BB",
            short_name=":man_technologist:",
            description="Man technologist",
            keywords=["programmer", "developer", "coder", "tech"],
            category=EmojiCategory.PERSON,
            primary_meaning="Male technology worker",
            domain_meanings={
                EmojiDomain.TECHNICAL: "Software developer or programmer",
                EmojiDomain.BUSINESS: "IT professional"
            },
            frequency_score=0.75,
            familiarity_level=FamiliarityLevel.COMMON
        )
        
        # Add some concept mappings
        self.add_concept_mapping(
            concept="success",
            description="Achievement of a goal or desired outcome",
            primary_emoji={
                EmojiDomain.GENERAL: "🎉",
                EmojiDomain.BUSINESS: "✅",
                EmojiDomain.TECHNICAL: "🚀"
            },
            alternative_emojis={
                EmojiDomain.GENERAL: ["🏆", "👏", "💯"],
                EmojiDomain.BUSINESS: ["📈", "🏆", "👍"],
                EmojiDomain.TECHNICAL: ["✅", "🔥", "💪"]
            },
            cultural_variations={
                CulturalContext.GLOBAL: "🎉",
                CulturalContext.EASTERN_ASIAN: "👍"
            },
            emoji_combinations={
                EmojiDomain.GENERAL: ["🎉🏆", "🥳🎊"],
                EmojiDomain.BUSINESS: ["📈✅", "🏆💼"]
            }
        )
        
        self.add_concept_mapping(
            concept="code review",
            description="Process of reviewing software code for quality and correctness",
            primary_emoji={
                EmojiDomain.TECHNICAL: "👀",
                EmojiDomain.BUSINESS: "🔍"
            },
            alternative_emojis={
                EmojiDomain.TECHNICAL: ["🧐", "🔍", "💻"],
                EmojiDomain.BUSINESS: ["👓", "👀", "✅"]
            },
            emoji_combinations={
                EmojiDomain.TECHNICAL: ["👀💻", "🔍👨‍💻", "💻🧐"],
                EmojiDomain.BUSINESS: ["📝👀", "🔍📊"]
            }
        )
        
        # Add some combination patterns
        self.add_combination_pattern(
            name="project_success",
            description="Successful completion of a project",
            emoji_sequence=["📋", "✅", "🎉"],
            resulting_concept="project completion success",
            primary_domain=EmojiDomain.BUSINESS,
            applicable_domains=[EmojiDomain.BUSINESS, EmojiDomain.TECHNICAL],
            frequency_score=0.7
        )
        
        self.add_combination_pattern(
            name="bug_fixed",
            description="Bug has been identified and fixed",
            emoji_sequence=["🐛", "🔨", "✅"],
            resulting_concept="bug fixed",
            primary_domain=EmojiDomain.TECHNICAL,
            applicable_domains=[EmojiDomain.TECHNICAL],
            frequency_score=0.85
        )
        
        # Create initial version
        self.create_new_version(
            version="1.0.0",
            name="Initial Release",
            notes="Initial release of the emoji knowledge base with basic emoji set."
        )
    
    def add_emoji(
        self,
        emoji: str,
        unicode_representation: str,
        short_name: str,
        description: str,
        keywords: List[str] = None,
        category: EmojiCategory = EmojiCategory.OBJECT,
        primary_meaning: str = "",
        alternate_meanings: List[str] = None,
        domain_meanings: Dict[EmojiDomain, str] = None,
        cultural_meanings: Dict[CulturalContext, str] = None,
        frequency_score: float = 0.0,
        familiarity_level: FamiliarityLevel = FamiliarityLevel.FAMILIAR,
        context_specificity: float = 0.0,
        sentiment: SentimentValue = SentimentValue.NEUTRAL,
        formality_score: float = 0.5,
        intensity_score: float = 0.5,
        ambiguity_score: float = 0.0,
        common_prefixes: List[str] = None,
        common_suffixes: List[str] = None,
        first_version: str = None,
        update_version: bool = True
    ) -> EmojiMetadata:
        """
        Add a new emoji to the knowledge base.
        
        Args:
            emoji: The emoji character(s)
            unicode_representation: Unicode representation
            short_name: Short name like ":smile:"
            description: Human-readable description
            keywords: Related keywords
            category: Primary category
            primary_meaning: Primary meaning in English
            alternate_meanings: Alternative meanings
            domain_meanings: Meanings in different domains
            cultural_meanings: Cultural interpretations
            frequency_score: Usage frequency (0.0-1.0)
            familiarity_level: How widely recognized
            context_specificity: How context-dependent (0.0-1.0)
            sentiment: Emotional sentiment
            formality_score: Formality level (0.0-1.0, 0=informal)
            intensity_score: Emotional intensity (0.0-1.0)
            ambiguity_score: Ambiguity level (0.0-1.0)
            common_prefixes: Emojis often used before this one
            common_suffixes: Emojis often used after this one
            first_version: Version when first added
            update_version: Whether to record this as a version change
            
        Returns:
            The created EmojiMetadata object
        """
        # Set default values for lists and dictionaries
        if keywords is None:
            keywords = []
        if alternate_meanings is None:
            alternate_meanings = []
        if domain_meanings is None:
            domain_meanings = {}
        if cultural_meanings is None:
            cultural_meanings = {}
        if common_prefixes is None:
            common_prefixes = []
        if common_suffixes is None:
            common_suffixes = []
        
        # Set version info
        if first_version is None:
            first_version = self.current_version
        
        # Create the metadata object
        metadata = EmojiMetadata(
            emoji=emoji,
            unicode_representation=unicode_representation,
            short_name=short_name,
            description=description,
            keywords=keywords,
            category=category,
            primary_meaning=primary_meaning,
            alternate_meanings=alternate_meanings,
            domain_meanings=domain_meanings,
            cultural_meanings=cultural_meanings,
            frequency_score=frequency_score,
            familiarity_level=familiarity_level,
            context_specificity=context_specificity,
            sentiment=sentiment,
            formality_score=formality_score,
            intensity_score=intensity_score,
            ambiguity_score=ambiguity_score,
            common_prefixes=common_prefixes,
            common_suffixes=common_suffixes,
            first_version=first_version,
            last_updated=self.current_version
        )
        
        # Check if we're updating an existing emoji
        is_update = emoji in self.emojis
        
        # Store the emoji metadata
        self.emojis = {**self.emojis, emoji: metadata}
        
        # Update statistics
        if not is_update:
            self.emoji_count = self.emoji_count + 1
        
        # Add to version tracking if requested
        if update_version:
            if is_update:
                self._add_to_version_tracking("updated_emojis", emoji)
            else:
                self._add_to_version_tracking("new_emojis", emoji)
        
        self.last_updated = time.time()
        
        return metadata
    
    def update_emoji(
        self,
        emoji: str,
        **kwargs
    ) -> Optional[EmojiMetadata]:
        """
        Update an existing emoji's metadata.
        
        Args:
            emoji: The emoji to update
            **kwargs: Metadata fields to update
            
        Returns:
            The updated EmojiMetadata or None if not found
        """
        if emoji not in self.emojis:
            return None
        
        # Get the existing metadata
        metadata = self.emojis[emoji]
        
        # Update the fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Update version and timestamp
        metadata.last_updated = self.current_version
        metadata.last_modified = time.time()
        
        # Add to version tracking
        self._add_to_version_tracking("updated_emojis", emoji)
        
        self.last_updated = time.time()
        
        return metadata
    
    def deprecate_emoji(
        self,
        emoji: str,
        replacement_emoji: Optional[str] = None
    ) -> bool:
        """
        Mark an emoji as deprecated.
        
        Args:
            emoji: The emoji to deprecate
            replacement_emoji: Recommended replacement emoji
            
        Returns:
            True if successful, False if emoji not found
        """
        if emoji not in self.emojis:
            return False
        
        # Update deprecation status
        metadata = self.emojis[emoji]
        metadata.deprecation_status = True
        metadata.replacement_emoji = replacement_emoji
        metadata.last_updated = self.current_version
        metadata.last_modified = time.time()
        
        # Add to version tracking
        self._add_to_version_tracking("deprecated_emojis", emoji)
        
        self.last_updated = time.time()
        
        return True
    
    def add_concept_mapping(
        self,
        concept: str,
        description: str,
        primary_emoji: Dict[EmojiDomain, str] = None,
        alternative_emojis: Dict[EmojiDomain, List[str]] = None,
        cultural_variations: Dict[CulturalContext, str] = None,
        emoji_combinations: Dict[EmojiDomain, List[str]] = None,
        related_concepts: List[str] = None,
        first_version: str = None,
        update_version: bool = True
    ) -> ConceptMapping:
        """
        Add a new concept mapping to the knowledge base.
        
        Args:
            concept: The concept being mapped
            description: Description of the concept
            primary_emoji: Best emoji by domain
            alternative_emojis: Alternatives by domain
            cultural_variations: Cultural preferences
            emoji_combinations: Emoji sequences by domain
            related_concepts: Semantically related concepts
            first_version: Version when first added
            update_version: Whether to record this as a version change
            
        Returns:
            The created ConceptMapping object
        """
        # Set default values
        if primary_emoji is None:
            primary_emoji = {}
        if alternative_emojis is None:
            alternative_emojis = {}
        if cultural_variations is None:
            cultural_variations = {}
        if emoji_combinations is None:
            emoji_combinations = {}
        if related_concepts is None:
            related_concepts = []
        
        # Set version info
        if first_version is None:
            first_version = self.current_version
        
        # Create the concept mapping
        mapping = ConceptMapping(
            concept=concept,
            description=description,
            primary_emoji=primary_emoji,
            alternative_emojis=alternative_emojis,
            cultural_variations=cultural_variations,
            emoji_combinations=emoji_combinations,
            related_concepts=related_concepts,
            first_version=first_version,
            last_updated=self.current_version
        )
        
        # Check if we're updating an existing concept
        is_update = concept in self.concepts
        
        # Store the concept mapping
        self.concepts = {**self.concepts, concept: mapping}
        
        # Update statistics
        if not is_update:
            self.concept_count = self.concept_count + 1
        
        # Add to version tracking if requested
        if update_version:
            if is_update:
                self._add_to_version_tracking("updated_concepts", concept)
            else:
                self._add_to_version_tracking("new_concepts", concept)
        
        self.last_updated = time.time()
        
        return mapping
    
    def update_concept_mapping(
        self,
        concept: str,
        **kwargs
    ) -> Optional[ConceptMapping]:
        """
        Update an existing concept mapping.
        
        Args:
            concept: The concept to update
            **kwargs: Mapping fields to update
            
        Returns:
            The updated ConceptMapping or None if not found
        """
        if concept not in self.concepts:
            return None
        
        # Get the existing mapping
        mapping = self.concepts[concept]
        
        # Update the fields
        for key, value in kwargs.items():
            if hasattr(mapping, key):
                setattr(mapping, key, value)
        
        # Update version and timestamp
        mapping.last_updated = self.current_version
        mapping.last_modified = time.time()
        
        # Add to version tracking
        self._add_to_version_tracking("updated_concepts", concept)
        
        self.last_updated = time.time()
        
        return mapping
    
    def get_emoji(self, emoji: str) -> Optional[EmojiMetadata]:
        """Get emoji metadata by emoji character.
        
        Args:
            emoji: The emoji character
            
        Returns:
            Emoji metadata dictionary or None if not found
        """
        return self.emojis.get(emoji)
    

    def find_concept_for_emoji(self, emoji: str) -> List[str]:
        """Find concepts that use this emoji.
        
        Args:
            emoji: The emoji to find concepts for
            
        Returns:
            List of concept names that include this emoji
        """
        concepts = []
        
        # Search in primary emojis
        for concept, mapping in self.concepts.items():
            for domain, primary_emoji in mapping.primary_emoji.items():
                if primary_emoji == emoji:
                    concepts.append(concept)
                    break
            
            # Search in alternative emojis
            for domain, alt_emojis in mapping.alternative_emojis.items():
                if emoji in alt_emojis:
                    concepts.append(concept)
                    break
            
            # Search in emoji combinations
            for domain, combinations in mapping.emoji_combinations.items():
                for combo in combinations:
                    if emoji in combo:
                        concepts.append(concept)
                        break
        
        return list(set(concepts))  # Remove duplicates

    def add_combination_pattern(
        self,
        name: str,
        description: str,
        emoji_sequence: List[str],
        resulting_concept: str,
        primary_domain: EmojiDomain = EmojiDomain.GENERAL,
        applicable_domains: List[EmojiDomain] = None,
        cultural_context: CulturalContext = CulturalContext.GLOBAL,
        frequency_score: float = 0.0,
        recognition_score: float = 0.0,
        first_version: str = None,
        update_version: bool = True
    ) -> EmojiCombinationPattern:
        """
        Add a new emoji combination pattern to the knowledge base.
        
        Args:
            name: Name of the combination pattern
            description: Description of the pattern
            emoji_sequence: Sequence of emojis in the pattern
            resulting_concept: The concept expressed by the combination
            primary_domain: Primary domain for this pattern
            applicable_domains: Where this pattern is applicable
            cultural_context: Primary cultural context
            frequency_score: Usage frequency (0.0-1.0)
            recognition_score: How widely recognized (0.0-1.0)
            first_version: Version when first added
            update_version: Whether to record this as a version change
            
        Returns:
            The created EmojiCombinationPattern object
        """
        # Set default values
        if applicable_domains is None:
            applicable_domains = [primary_domain]
        
        # Set version info
        if first_version is None:
            first_version = self.current_version
        
        # Create pattern ID
        pattern_id = f"{name}_{hashlib.md5(''.join(emoji_sequence).encode()).hexdigest()[:8]}"
        
        # Create the pattern object
        pattern = EmojiCombinationPattern(
            name=name,
            description=description,
            emoji_sequence=emoji_sequence,
            resulting_concept=resulting_concept,
            primary_domain=primary_domain,
            applicable_domains=applicable_domains,
            cultural_context=cultural_context,
            frequency_score=frequency_score,
            recognition_score=recognition_score,
            first_version=first_version,
            last_updated=self.current_version
        )
        
        # Check if we're updating an existing pattern
        is_update = pattern_id in self.patterns
        
        # Store the pattern
        self.patterns = {**self.patterns, pattern_id: pattern}
        
        # Update statistics
        if not is_update:
            self.pattern_count = self.pattern_count + 1
        
        # Add to version tracking if requested
        if update_version:
            if is_update:
                self._add_to_version_tracking("updated_patterns", pattern_id)
            else:
                self._add_to_version_tracking("new_patterns", pattern_id)
        
        self.last_updated = time.time()
        
        return pattern
    
    def update_combination_pattern(
        self,
        pattern_id: str,
        **kwargs
    ) -> Optional[EmojiCombinationPattern]:
        """
        Update an existing combination pattern.
        
        Args:
            pattern_id: The ID of the pattern to update
            **kwargs: Pattern fields to update
            
        Returns:
            The updated EmojiCombinationPattern or None if not found
        """
        if pattern_id not in self.patterns:
            return None
        
        # Get the existing pattern
        pattern = self.patterns[pattern_id]
        
        # Update the fields
        for key, value in kwargs.items():
            if hasattr(pattern, key):
                setattr(pattern, key, value)
        
        # Update version and timestamp
        pattern.last_updated = self.current_version
        pattern.last_modified = time.time()
        
        # Add to version tracking
        self._add_to_version_tracking("updated_patterns", pattern_id)
        
        self.last_updated = time.time()
        
        return pattern
    
    def create_new_version(
        self,
        version: str,
        name: str,
        notes: str = ""
    ) -> KnowledgeBaseVersion:
        """
        Create a new version of the knowledge base.
        
        Args:
            version: Version number (semantic versioning)
            name: Version name
            notes: Release notes
            
        Returns:
            The created KnowledgeBaseVersion object
        """
        # Get pending changes from the current version
        if self.versions and hasattr(self.versions[-1], "pending_changes"):
            pending_changes = self.versions[-1].pending_changes
        else:
            pending_changes = {}
            
        # Create new version record
        version_record = KnowledgeBaseVersion(
            version=version,
            name=name,
            notes=notes,
            new_emojis=pending_changes.get("new_emojis", []),
            updated_emojis=pending_changes.get("updated_emojis", []),
            deprecated_emojis=pending_changes.get("deprecated_emojis", []),
            new_concepts=pending_changes.get("new_concepts", []),
            updated_concepts=pending_changes.get("updated_concepts", []),
            new_patterns=pending_changes.get("new_patterns", []),
            updated_patterns=pending_changes.get("updated_patterns", [])
        )
        
        # Add to version history
        self.versions = [*self.versions, version_record]
        self.current_version = version
        
        # Clear pending changes for next version
        if self.versions:
            self.versions[-1].pending_changes = {
                "new_emojis": [],
                "updated_emojis": [],
                "deprecated_emojis": [],
                "new_concepts": [],
                "updated_concepts": [],
                "new_patterns": [],
                "updated_patterns": []
            }
            
        return version_record
    
    def _add_to_version_tracking(self, category: str, item: str) -> None:
        """
        Add an item to version tracking for the current version.
        
        Args:
            category: The category to add to (e.g., "new_emojis", "updated_concepts")
            item: The item identifier to add
        """
        # Ensure we have a version to track changes in
        if not self.versions:
            self.create_new_version("1.0.0", "Initial Version", "Initial version with tracking")
            
        # Get or create pending changes on the last version
        if not hasattr(self.versions[-1], "pending_changes"):
            self.versions[-1].pending_changes = {
                "new_emojis": [],
                "updated_emojis": [],
                "deprecated_emojis": [],
                "new_concepts": [],
                "updated_concepts": [],
                "new_patterns": [],
                "updated_patterns": []
            }
            
        # Add item to the appropriate category if not already present
        if category in self.versions[-1].pending_changes:
            if item not in self.versions[-1].pending_changes[category]:
                self.versions[-1].pending_changes[category].append(item)