import emoji
"""
EmojiGrammarSystem Component for Adaptive Bridge Builder Agent

This component defines structural rules and patterns for organizing emojis into
coherent "sentences" with grammatical meaning, supporting complex communication
through emoji sequences.

It builds upon the EmojiTranslationEngine and provides a formalized grammar system
for emoji-based communication.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from emoji_translation_engine import (
    EmojiTranslationEngine,
    EmojiDictionary,
    EmojiEntry,
    EmojiCategory
)


class GrammaticalRole(Enum):
    """Defines grammatical roles for emojis in a sentence."""
    SUBJECT = "subject"           # Who/what is performing the action
    PREDICATE = "predicate"       # The action or state
    OBJECT = "object"             # Who/what receives the action
    MODIFIER = "modifier"         # Adjusts meaning of other elements
    CONNECTOR = "connector"       # Links elements together
    PUNCTUATION = "punctuation"   # Indicates sentence structure
    TENSE = "tense"               # Indicates when action occurs
    QUANTITY = "quantity"         # Indicates amount or number
    INTERROGATIVE = "interrogative"  # Indicates a question
    NEGATIVE = "negative"         # Negates meaning
    CONDITIONAL = "conditional"   # Indicates contingent relationship


class SentenceType(Enum):
    """Defines the types of emoji sentences that can be constructed."""
    STATEMENT = "statement"       # Declarative sentence
    QUESTION = "question"         # Interrogative sentence
    COMMAND = "command"           # Imperative sentence
    EXCLAMATION = "exclamation"   # Exclamatory sentence
    CONDITIONAL = "conditional"   # Expresses condition/result relationship


class Tense(Enum):
    """Represents time aspects in emoji grammar."""
    PAST = "past"                 # Action occurred in the past
    PRESENT = "present"           # Action occurring now
    FUTURE = "future"             # Action will occur in the future
    CONTINUOUS = "continuous"     # Ongoing action
    PERFECT = "perfect"           # Completed action with relevance to now


class Quantity(Enum):
    """Represents quantity indicators in emoji grammar."""
    SINGULAR = "singular"         # One
    PLURAL = "plural"             # More than one
    ZERO = "zero"                 # None
    FEW = "few"                   # Small number
    MANY = "many"                 # Large number
    ALL = "all"                   # Everything


class EmotionalNuance(Enum):
    """Represents emotional nuances that can be applied to emoji sentences."""
    NEUTRAL = "neutral"           # No specific emotional emphasis
    EXCITED = "excited"           # High energy, enthusiasm
    SERIOUS = "serious"           # Formal, grave
    HUMOROUS = "humorous"         # Funny, not serious
    SARCASTIC = "sarcastic"       # Saying opposite of what's meant
    URGENT = "urgent"             # Requiring immediate attention
    GENTLE = "gentle"             # Soft, kind approach
    FIRM = "firm"                 # Strong, decisive approach


class EmojiGrammarRule:
    """
    Defines a structural rule for emoji grammar.
    
    Each rule specifies a pattern for how emojis should be arranged
    to express a particular grammatical construct.
    """
    
    def __init__(
        self,
        name: str,
        pattern: List[GrammaticalRole],
        example: str,
        description: str,
        sentence_type: SentenceType
    ):
        self.name = name
        self.pattern = pattern  # Sequence of roles that make up this pattern
        self.example = example  # Example emoji sequence using this pattern
        self.description = description
        self.sentence_type = sentence_type
    
    def __str__(self) -> str:
        roles_str = " + ".join([role.value for role in self.pattern])
        return f"{self.name}: {roles_str}"
    
    def matches(self, roles: List[GrammaticalRole]) -> bool:
        """Check if a sequence of roles matches this pattern."""
        if len(roles) != len(self.pattern):
            return False
            
        for i, role in enumerate(roles):
            if role != self.pattern[i]:
                return False
                
        return True


class EmojiGrammarRuleSet:
    """
    Collection of grammar rules for different types of emoji sentences.
    
    Provides methods to find appropriate rules for constructing or
    parsing emoji sequences.
    """
    
    def __init__(self) -> None:
        self.rules: List[EmojiGrammarRule] = []
        self._initialize_default_rules()
    
    def add_rule(self, rule: EmojiGrammarRule) -> None:
        """Add a grammar rule to the rule set."""
        self.rules = [*self.rules, rule]
    
    def get_rules_by_sentence_type(self, sentence_type: SentenceType) -> List[EmojiGrammarRule]:
        """Get all rules for a specific sentence type."""
        return [rule for rule in self.rules if rule.sentence_type == sentence_type]
    
    def find_matching_rule(self, roles: List[GrammaticalRole]) -> Optional[EmojiGrammarRule]:
        """Find a rule that matches a sequence of grammatical roles."""
        for rule in self.rules:
            if rule.matches(roles):
                return rule
        return None
    
    def _initialize_default_rules(self) -> None:
        """Initialize the default set of grammar rules."""
        # Basic statement patterns
        self.add_rule(EmojiGrammarRule(
            name="Simple Statement",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE],
            example="👤 👍",  # "I approve"
            description="Basic subject-verb structure for simple statements",
            sentence_type=SentenceType.STATEMENT
        ))
        
        self.add_rule(EmojiGrammarRule(
            name="Statement with Object",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE, GrammaticalRole.OBJECT],
            example="👤 ❤️ 🍕",  # "I love pizza"
            description="Subject-verb-object structure for statements about actions on objects",
            sentence_type=SentenceType.STATEMENT
        ))
        
        self.add_rule(EmojiGrammarRule(
            name="Statement with Modifier",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.MODIFIER, GrammaticalRole.PREDICATE],
            example="👤 👍 😊",  # "I happily agree"
            description="Subject with modified verb for more nuanced statements",
            sentence_type=SentenceType.STATEMENT
        ))
        
        # Question patterns
        self.add_rule(EmojiGrammarRule(
            name="Simple Question",
            pattern=[GrammaticalRole.INTERROGATIVE, GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE],
            example="❓ 👤 👍",  # "Do I approve?"
            description="Basic interrogative structure for yes/no questions",
            sentence_type=SentenceType.QUESTION
        ))
        
        self.add_rule(EmojiGrammarRule(
            name="Question with Object",
            pattern=[GrammaticalRole.INTERROGATIVE, GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE, GrammaticalRole.OBJECT],
            example="❓ 👤 ❤️ 🍕",  # "Do I love pizza?"
            description="Interrogative structure for questions about subject-object relationships",
            sentence_type=SentenceType.QUESTION
        ))
        
        # Command patterns
        self.add_rule(EmojiGrammarRule(
            name="Simple Command",
            pattern=[GrammaticalRole.PREDICATE],
            example="👍",  # "Approve"
            description="Single verb structure for basic commands",
            sentence_type=SentenceType.COMMAND
        ))
        
        self.add_rule(EmojiGrammarRule(
            name="Command with Object",
            pattern=[GrammaticalRole.PREDICATE, GrammaticalRole.OBJECT],
            example="👀 📝",  # "Look at the document"
            description="Verb-object structure for commands involving objects",
            sentence_type=SentenceType.COMMAND
        ))
        
        # Conditional patterns
        self.add_rule(EmojiGrammarRule(
            name="Simple Conditional",
            pattern=[
                GrammaticalRole.SUBJECT, 
                GrammaticalRole.PREDICATE, 
                GrammaticalRole.CONDITIONAL, 
                GrammaticalRole.SUBJECT, 
                GrammaticalRole.PREDICATE
            ],
            example="☔ 👇 ➡️ 👤 🏠",  # "If it rains, I stay home"
            description="If-then structure for expressing conditional relationships",
            sentence_type=SentenceType.CONDITIONAL
        ))
        
        # Negative patterns
        self.add_rule(EmojiGrammarRule(
            name="Negative Statement",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.NEGATIVE, GrammaticalRole.PREDICATE],
            example="👤 🚫 👍",  # "I don't approve"
            description="Subject with negated verb for negative statements",
            sentence_type=SentenceType.STATEMENT
        ))
        
        # Tense-specific patterns
        self.add_rule(EmojiGrammarRule(
            name="Past Tense Statement",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE, GrammaticalRole.TENSE],
            example="👤 👍 ⏮️",  # "I approved (in the past)"
            description="Subject-verb structure with past tense indicator",
            sentence_type=SentenceType.STATEMENT
        ))
        
        self.add_rule(EmojiGrammarRule(
            name="Future Tense Statement",
            pattern=[GrammaticalRole.SUBJECT, GrammaticalRole.PREDICATE, GrammaticalRole.TENSE],
            example="👤 👍 ⏭️",  # "I will approve (in the future)"
            description="Subject-verb structure with future tense indicator",
            sentence_type=SentenceType.STATEMENT
        ))


class EmojiGrammarElement:
    """
    Represents a single element in an emoji grammar structure.
    
    Each element consists of an emoji and its grammatical role
    in the sentence structure.
    """
    
    def __init__(
        self,
        emoji: str,
        role: GrammaticalRole,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.emoji = emoji
        self.role = role
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"{self.emoji} ({self.role.value})"


class EmojiSentence:
    """
    Represents a grammatically structured emoji sentence.
    
    An emoji sentence consists of a sequence of emoji grammar elements
    arranged according to a specific grammar rule.
    """
    
    def __init__(
        self,
        elements: List[EmojiGrammarElement],
        sentence_type: SentenceType,
        emotional_nuance: EmotionalNuance = EmotionalNuance.NEUTRAL
    ):
        self.elements = elements
        self.sentence_type = sentence_type
        self.emotional_nuance = emotional_nuance
        
        # Derived properties
        self.roles = [element.role for element in elements]
        self.emoji_sequence = ''.join([element.emoji for element in elements])
    
    def __str__(self) -> str:
        return self.emoji_sequence
    
    def get_elements_by_role(self, role: GrammaticalRole) -> List[EmojiGrammarElement]:
        """Get all elements with a specific grammatical role."""
        return [element for element in self.elements if element.role == role]
    
    def has_role(self, role: GrammaticalRole) -> bool:
        """Check if the sentence has at least one element with the specified role."""
        return role in self.roles
    
    def get_subject(self) -> Optional[EmojiGrammarElement]:
        """Get the subject of the sentence, if present."""
        subjects = self.get_elements_by_role(GrammaticalRole.SUBJECT)
        return subjects[0] if subjects else None
    
    def get_predicate(self) -> Optional[EmojiGrammarElement]:
        """Get the predicate (verb) of the sentence, if present."""
        predicates = self.get_elements_by_role(GrammaticalRole.PREDICATE)
        return predicates[0] if predicates else None
    
    def get_object(self) -> Optional[EmojiGrammarElement]:
        """Get the object of the sentence, if present."""
        objects = self.get_elements_by_role(GrammaticalRole.OBJECT)
        return objects[0] if objects else None
    
    def get_tense(self) -> Optional[Tense]:
        """Determine the tense of the sentence, if indicated."""
        tense_elements = self.get_elements_by_role(GrammaticalRole.TENSE)
        if not tense_elements:
            return Tense.PRESENT  # Default to present tense if not specified
            
        # Extract tense from metadata
        tense_element = tense_elements[0]
        return tense_element.metadata.get('tense', Tense.PRESENT)


class EmojiModifiers:
    """
    Defines emoji modifiers for expressing grammatical concepts.
    
    This class provides mappings between grammatical concepts like
    tense, quantity, and relationships and the emoji modifiers used
    to express them.
    """
    
    # Tense indicators
    TENSE_MARKERS = {
        Tense.PAST: "⏮️",      # Reverse button for past
        Tense.PRESENT: "▶️",    # Play button for present
        Tense.FUTURE: "⏭️",     # Forward button for future
        Tense.CONTINUOUS: "🔄",  # Cycle arrows for continuous
        Tense.PERFECT: "✓️"      # Check mark for perfect/completed
    }
    
    # Quantity indicators
    QUANTITY_MARKERS = {
        Quantity.SINGULAR: "1️⃣",  # Number one for singular
        Quantity.PLURAL: "🔢",     # Input numbers for plural
        Quantity.ZERO: "0️⃣",      # Number zero for none
        Quantity.FEW: "🔉",       # Low volume for few
        Quantity.MANY: "🔊",      # High volume for many
        Quantity.ALL: "💯"        # Hundred points for all
    }
    
    # Relationship indicators
    RELATIONSHIP_MARKERS = {
        "possessive": "🔒",      # Lock for possession
        "belongs_to": "📎",      # Paperclip for belonging
        "part_of": "🧩",         # Puzzle piece for part-whole relationship
        "located_at": "📍",      # Pin for location
        "causes": "➡️",          # Right arrow for causation
        "member_of": "👥"        # Group for membership
    }
    
    # Grammatical markers
    PUNCTUATION_MARKERS = {
        "statement": ".",       # Period (represented as a dot)
        "question": "❓",        # Question mark
        "exclamation": "❗",     # Exclamation mark
        "comma": "💨",          # Wind blowing for pause/comma
        "semicolon": "💨💨",     # Double wind for semicolon
        "colon": "🔍"           # Magnifying glass for explanatory/expanding
    }
    
    # Logical operators
    LOGICAL_MARKERS = {
        "and": "➕",             # Plus sign for 'and'
        "or": "🔀",              # Shuffle for 'or'
        "not": "🚫",             # Prohibited for 'not'
        "if": "❔",              # White question mark for 'if'
        "then": "➡️",            # Right arrow for 'then'
        "else": "↪️"             # Left arrow hook for 'else'
    }
    
    @classmethod
    def get_tense_marker(cls, tense: Tense) -> str:
        """Get the emoji marker for a specific tense."""
        return cls.TENSE_MARKERS.get(tense, cls.TENSE_MARKERS[Tense.PRESENT])
    
    @classmethod
    def get_quantity_marker(cls, quantity: Quantity) -> str:
        """Get the emoji marker for a specific quantity."""
        return cls.QUANTITY_MARKERS.get(quantity, cls.QUANTITY_MARKERS[Quantity.SINGULAR])
    
    @classmethod
    def get_relationship_marker(cls, relationship: str) -> Optional[str]:
        """Get the emoji marker for a specific relationship type."""
        return cls.RELATIONSHIP_MARKERS.get(relationship)
    
    @classmethod
    def get_punctuation_marker(cls, punctuation_type: str) -> Optional[str]:
        """Get the emoji marker for a specific punctuation type."""
        return cls.PUNCTUATION_MARKERS.get(punctuation_type)
    
    @classmethod
    def get_logical_marker(cls, logical_type: str) -> Optional[str]:
        """Get the emoji marker for a specific logical operator."""
        return cls.LOGICAL_MARKERS.get(logical_type)


class EmojiGrammarParser:
    """
    Parses emoji sequences into grammatical structures.
    
    This class provides functionality to analyze emoji sequences and
    determine their grammatical structure according to defined rules.
    """
    
    def __init__(self, rule_set: EmojiGrammarRuleSet, emoji_engine: EmojiTranslationEngine) -> None:
        self.rule_set = rule_set
        self.emoji_engine = emoji_engine
    
    def parse_emoji_sequence(self, emoji_sequence: str) -> Optional[EmojiSentence]:
        """
        Parse an emoji sequence into a grammatically structured sentence.
        
        Args:
            emoji_sequence: The sequence of emojis to parse
            
        Returns:
            An EmojiSentence object if parsing is successful, None otherwise
        """
        # Extract individual emojis
        emoji_pattern = re.compile(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])')
        emojis = emoji_pattern.findall(emoji_sequence)
        
        if not emojis:
            return None
        
        # Determine the sentence type
        sentence_type = self._determine_sentence_type(emojis)
        
        # Assign grammatical roles to each emoji
        elements = self._assign_grammatical_roles(emojis, sentence_type)
        
        # Determine emotional nuance if present
        emotional_nuance = self._determine_emotional_nuance(emojis)
        
        # Create and return the emoji sentence
        return EmojiSentence(elements, sentence_type, emotional_nuance)
    
    def _determine_sentence_type(self, emojis: List[str]) -> SentenceType:
        """Determine the type of sentence based on the emoji sequence."""
        # Check for question marker
        if "❓" in emojis:
            return SentenceType.QUESTION
        
        # Check for exclamation marker
        if "❗" in emojis:
            return SentenceType.EXCLAMATION
        
        # Check for conditional marker
        if "➡️" in emojis and any(emoji in emojis for emoji in ["❔", "🤔"]):
            return SentenceType.CONDITIONAL
        
        # Check for command (typically starts with verb without subject)
        # This is a simplified heuristic; real implementation would be more sophisticated
        if len(emojis) >= 1 and emojis[0] not in ["👤", "👨", "👩", "🧑", "👪", "👥"]:
            return SentenceType.COMMAND
        
        # Default to statement
        return SentenceType.STATEMENT
    
    def _assign_grammatical_roles(
        self, 
        emojis: List[str], 
        sentence_type: SentenceType
    ) -> List[EmojiGrammarElement]:
        """
        Assign grammatical roles to each emoji in the sequence.
        
        This is a complex task that would use a combination of pattern matching,
        context, and possibly machine learning in a real implementation.
        For this example, we'll use a simplified rule-based approach.
        """
        elements = []
        
        # Different handling based on sentence type
        if sentence_type == SentenceType.QUESTION:
            # Question structure: ❓ + Subject + Predicate [+ Object]
            for i, emoji in enumerate(emojis):
                if emoji == "❓":
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.INTERROGATIVE))
                elif i == 1:  # Assuming the subject follows the question marker
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.SUBJECT))
                elif i == 2:  # Assuming the predicate follows the subject
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.PREDICATE))
                elif i == 3:  # Assuming the object follows the predicate (if present)
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.OBJECT))
                else:
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.MODIFIER))
        
        elif sentence_type == SentenceType.COMMAND:
            # Command structure: Predicate [+ Object]
            for i, emoji in enumerate(emojis):
                if i == 0:  # First emoji is the verb/predicate
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.PREDICATE))
                elif i == 1:  # Second emoji is potentially the object
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.OBJECT))
                else:
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.MODIFIER))
        
        elif sentence_type == SentenceType.CONDITIONAL:
            # Conditional structure: [Subject +] Predicate + Conditional + [Subject +] Predicate
            # This is simplified; real implementation would be more sophisticated
            conditional_marker_index = emojis.index("➡️") if "➡️" in emojis else -1
            
            if conditional_marker_index > 0:
                # Process "if" part (before the conditional marker)
                for i in range(conditional_marker_index):
                    if i == 0:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.SUBJECT))
                    elif i == 1:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.PREDICATE))
                    else:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.MODIFIER))
                
                # Add conditional marker
                elements.append(EmojiGrammarElement("➡️", GrammaticalRole.CONDITIONAL))
                
                # Process "then" part (after the conditional marker)
                for i in range(conditional_marker_index + 1, len(emojis)):
                    if i == conditional_marker_index + 1:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.SUBJECT))
                    elif i == conditional_marker_index + 2:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.PREDICATE))
                    else:
                        elements.append(EmojiGrammarElement(emojis[i], GrammaticalRole.MODIFIER))
            else:
                # Fallback parsing if conditional structure isn't clear
                for i, emoji in enumerate(emojis):
                    if i == 0:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.SUBJECT))
                    elif i == 1:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.PREDICATE))
                    elif i == 2:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.OBJECT))
                    else:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.MODIFIER))
        
        else:  # Statement or Exclamation
            # Statement structure: Subject + Predicate [+ Object]
            for i, emoji in enumerate(emojis):
                if emoji in EmojiModifiers.TENSE_MARKERS.values():
                    # Identify tense markers
                    tense = next(
                        (t for t, marker in EmojiModifiers.TENSE_MARKERS.items() if marker == emoji),
                        Tense.PRESENT
                    )
                    elements.append(EmojiGrammarElement(
                        emoji, GrammaticalRole.TENSE, {'tense': tense}
                    ))
                elif emoji in EmojiModifiers.LOGICAL_MARKERS.values():
                    if emoji == EmojiModifiers.LOGICAL_MARKERS['not']:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.NEGATIVE))
                    else:
                        elements.append(EmojiGrammarElement(emoji, GrammaticalRole.CONNECTOR))
                elif i == 0:  # First emoji is typically the subject
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.SUBJECT))
                elif i == 1:  # Second emoji is typically the predicate
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.PREDICATE))
                elif i == 2:  # Third emoji is potentially the object
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.OBJECT))
                else:
                    elements.append(EmojiGrammarElement(emoji, GrammaticalRole.MODIFIER))
        
        return elements
    
    def _determine_emotional_nuance(self, emojis: List[str]) -> EmotionalNuance:
        """Determine the emotional nuance of the emoji sequence."""
        # This is a simplified approach; a real implementation would be more sophisticated
        
        # Check for excited/enthusiastic indicators
        excited_emojis = ["😄", "🎉", "✨", "🔥", "💯"]
        if any(emoji in emojis for emoji in excited_emojis):
            return EmotionalNuance.EXCITED
        
        # Check for serious indicators
        serious_emojis = ["😐", "🧐", "🤨", "📊", "📈"]
        if any(emoji in emojis for emoji in serious_emojis):
            return EmotionalNuance.SERIOUS
        
        # Check for humorous indicators
        humorous_emojis = ["😂", "🤣", "😜", "😋", "🙃"]
        if any(emoji in emojis for emoji in humorous_emojis):
            return EmotionalNuance.HUMOROUS
        
        # Check for sarcastic indicators
        sarcastic_emojis = ["😏", "🙄", "😒", "💅", "🤦"]
        if any(emoji in emojis for emoji in sarcastic_emojis):
            return EmotionalNuance.SARCASTIC
        
        # Check for urgent indicators
        urgent_emojis = ["⚠️", "🚨", "❗", "⏰", "🔴"]
        if any(emoji in emojis for emoji in urgent_emojis):
            return EmotionalNuance.URGENT
        
        # Default to neutral
        return EmotionalNuance.NEUTRAL


class EmojiGrammarGenerator:
    """
    Generates grammatically structured emoji sequences.
    
    This class provides functionality to create emoji sentences
    following defined grammatical rules.
    """
    
    def __init__(self, rule_set: EmojiGrammarRuleSet, emoji_engine: EmojiTranslationEngine) -> None:
        self.rule_set = rule_set
        self.emoji_engine = emoji_engine
    
    def generate_emoji_sentence(
        self,
        sentence_type: SentenceType,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        tense: Tense = Tense.PRESENT,
        modifiers: List[str] = None,
        emotional_nuance: EmotionalNuance = EmotionalNuance.NEUTRAL,
        is_negative: bool = False
    ) -> Optional[EmojiSentence]:
        """
        Generate an emoji sentence based on the specified parameters.
        
        Args:
            sentence_type: The type of sentence to generate
            subject: The subject of the sentence (for statements/questions)
            predicate: The predicate/verb of the sentence
            object_: The object of the sentence (if applicable)
            tense: The tense of the sentence
            modifiers: Additional modifiers to apply
            emotional_nuance: The emotional tone of the sentence
            is_negative: Whether the sentence is negative
            
        Returns:
            An EmojiSentence object if generation is successful, None otherwise
        """
        if not predicate:
            return None  # Predicate is required for all sentence types
            
        # Find suitable emojis for each component
        predicate_emoji = self._find_emoji_for_concept(predicate)
        if not predicate_emoji:
            return None
            
        subject_emoji = None
        if subject and sentence_type != SentenceType.COMMAND:
            subject_emoji = self._find_emoji_for_concept(subject)
            if not subject_emoji:
                return None
                
        object_emoji = None
        if object_:
            object_emoji = self._find_emoji_for_concept(object_)
            if not object_emoji:
                return None
        
        # Create grammar elements
        elements = []
        
        # Process based on sentence type
        if sentence_type == SentenceType.STATEMENT:
            # Add subject (if provided)
            if subject_emoji:
                elements.append(EmojiGrammarElement(subject_emoji, GrammaticalRole.SUBJECT))
            
            # Add negation if needed
            if is_negative:
                elements.append(EmojiGrammarElement(
                    EmojiModifiers.LOGICAL_MARKERS['not'], 
                    GrammaticalRole.NEGATIVE
                ))
            
            # Add predicate
            elements.append(EmojiGrammarElement(predicate_emoji, GrammaticalRole.PREDICATE))
            
            # Add object if provided
            if object_emoji:
                elements.append(EmojiGrammarElement(object_emoji, GrammaticalRole.OBJECT))
            
            # Add tense marker if not present
            if tense != Tense.PRESENT:
                tense_marker = EmojiModifiers.get_tense_marker(tense)
                elements.append(EmojiGrammarElement(
                    tense_marker, 
                    GrammaticalRole.TENSE,
                    {'tense': tense}
                ))
            
            # Add modifiers if provided
            if modifiers:
                for modifier in modifiers:
                    modifier_emoji = self._find_emoji_for_concept(modifier)
                    if modifier_emoji:
                        elements.append(EmojiGrammarElement(modifier_emoji, GrammaticalRole.MODIFIER))
        
        elif sentence_type == SentenceType.QUESTION:
            # Add question marker
            elements.append(EmojiGrammarElement(
                EmojiModifiers.PUNCTUATION_MARKERS['question'], 
                GrammaticalRole.INTERROGATIVE
            ))
            
            # Add subject (if provided)
            if subject_emoji:
                elements.append(EmojiGrammarElement(subject_emoji, GrammaticalRole.SUBJECT))
            
            # Add negation if needed
            if is_negative:
                elements.append(EmojiGrammarElement(
                    EmojiModifiers.LOGICAL_MARKERS['not'], 
                    GrammaticalRole.NEGATIVE
                ))
            
            # Add predicate
            elements.append(EmojiGrammarElement(predicate_emoji, GrammaticalRole.PREDICATE))
            
            # Add object if provided
            if object_emoji:
                elements.append(EmojiGrammarElement(object_emoji, GrammaticalRole.OBJECT))
            
            # Add tense marker if not present
            if tense != Tense.PRESENT:
                tense_marker = EmojiModifiers.get_tense_marker(tense)
                elements.append(EmojiGrammarElement(
                    tense_marker, 
                    GrammaticalRole.TENSE,
                    {'tense': tense}
                ))
        
        elif sentence_type == SentenceType.COMMAND:
            # Commands typically start with the predicate
            elements.append(EmojiGrammarElement(predicate_emoji, GrammaticalRole.PREDICATE))
            
            # Add object if provided
            if object_emoji:
                elements.append(EmojiGrammarElement(object_emoji, GrammaticalRole.OBJECT))
            
            # Add modifiers if provided
            if modifiers:
                for modifier in modifiers:
                    modifier_emoji = self._find_emoji_for_concept(modifier)
                    if modifier_emoji:
                        elements.append(EmojiGrammarElement(modifier_emoji, GrammaticalRole.MODIFIER))
        
        elif sentence_type == SentenceType.CONDITIONAL:
            # Add first part (condition)
            if subject_emoji:
                elements.append(EmojiGrammarElement(subject_emoji, GrammaticalRole.SUBJECT))
            
            elements.append(EmojiGrammarElement(predicate_emoji, GrammaticalRole.PREDICATE))
            
            # Add conditional marker
            elements.append(EmojiGrammarElement(
                EmojiModifiers.LOGICAL_MARKERS['then'], 
                GrammaticalRole.CONDITIONAL
            ))
            
            # Add second part (result) - this would typically need another subject and predicate
            # For simplicity, we'll use the same subject but would need different predicates in practice
            if subject_emoji:
                elements.append(EmojiGrammarElement(subject_emoji, GrammaticalRole.SUBJECT))
            
            # For a complete implementation, we would need a result predicate here
            # Using the same predicate for demonstration
            elements.append(EmojiGrammarElement(predicate_emoji, GrammaticalRole.PREDICATE))
        
        # Apply emotional nuance (if not neutral)
        if emotional_nuance != EmotionalNuance.NEUTRAL:
            # Determine emoji to represent emotional nuance
            emotion_emoji = self._get_emotion_emoji(emotional_nuance)
            if emotion_emoji:
                elements.append(EmojiGrammarElement(emotion_emoji, GrammaticalRole.MODIFIER))
        
        return EmojiSentence(elements, sentence_type, emotional_nuance)
    
    def translate_text_to_emoji_sentence(
        self,
        text: str,
        sentence_type: Optional[SentenceType] = None
    ) -> Optional[EmojiSentence]:
        """
        Translate natural language text to an emoji sentence.
        
        Args:
            text: The text to translate
            sentence_type: Optional sentence type to enforce (if None, will be inferred)
            
        Returns:
            An EmojiSentence if translation is successful, None otherwise
        """
        # Use the EmojiTranslationEngine to get emojis for the text
        emoji_sequence = self.emoji_engine.translate_text_to_emoji(text)
        
        # Infer sentence type if not provided
        if sentence_type is None:
            # Simple rules to infer sentence type
            if text.endswith('?'):
                sentence_type = SentenceType.QUESTION
            elif text.endswith('!'):
                sentence_type = SentenceType.EXCLAMATION
            elif text.startswith(('Do ', 'Could ', 'Would ', 'Will ', 'Can ', 'Should ')):
                # Imperative sentence
                sentence_type = SentenceType.COMMAND
            elif any(conditional in text.lower() for conditional in ('if', 'when', 'unless', 'until')):
                sentence_type = SentenceType.CONDITIONAL
            else:
                sentence_type = SentenceType.STATEMENT
        
        # Create grammar elements based on the emoji sequence
        # This would need sophisticated NLP in a real implementation
        # Here we're using a simplified approach that relies on EmojiGrammarParser
        parser = EmojiGrammarParser(self.rule_set, self.emoji_engine)
        return parser.parse_emoji_sequence(emoji_sequence)
    
    def _find_emoji_for_concept(self, concept: str) -> Optional[str]:
        """Find the most appropriate emoji for a concept."""
        # Use the EmojiTranslationEngine to find an emoji
        result = self.emoji_engine.translate_text_to_emoji(concept)
        
        # Return the first emoji if available
        if result:
            emoji_pattern = re.compile(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])')
            emojis = emoji_pattern.findall(result)
            if emojis:
                return emojis[0]
        
        return None
    
    def _get_emotion_emoji(self, emotional_nuance: EmotionalNuance) -> Optional[str]:
        """Get an emoji that represents a specific emotional nuance."""
        # Mapping of emotional nuances to representative emojis
        emotion_emojis = {
            EmotionalNuance.EXCITED: "😄",
            EmotionalNuance.SERIOUS: "😐",
            EmotionalNuance.HUMOROUS: "😂",
            EmotionalNuance.SARCASTIC: "😏",
            EmotionalNuance.URGENT: "⚠️",
            EmotionalNuance.GENTLE: "🙂",
            EmotionalNuance.FIRM: "😠"
        }
        
        return emotion_emojis.get(emotional_nuance)


class EmojiGrammarSystem:
    """
    Main component for creating and interpreting emoji sentences with grammatical structure.
    
    This system builds upon the EmojiTranslationEngine to provide formal
    grammar rules for emoji-based communication.
    """
    
    def __init__(self, emoji_engine: Optional[EmojiTranslationEngine] = None) -> None:
        """
        Initialize the emoji grammar system.
        
        Args:
            emoji_engine: Optional EmojiTranslationEngine to use.
                If None, a new instance will be created.
        """
        self.emoji_engine = emoji_engine or EmojiTranslationEngine()
        self.rule_set = EmojiGrammarRuleSet()
        self.parser = EmojiGrammarParser(self.rule_set, self.emoji_engine)
        self.generator = EmojiGrammarGenerator(self.rule_set, self.emoji_engine)
    
    def parse_emoji_sequence(self, emoji_sequence: str) -> Optional[EmojiSentence]:
        """
        Parse an emoji sequence into a grammatical structure.
        
        Args:
            emoji_sequence: The emoji sequence to parse
            
        Returns:
            An EmojiSentence object representing the grammatical structure,
            or None if parsing fails
        """
        return self.parser.parse_emoji_sequence(emoji_sequence)
    
    def translate_to_emoji_sentence(
        self,
        text: str,
        sentence_type: Optional[SentenceType] = None
    ) -> Optional[EmojiSentence]:
        """
        Translate natural language text to an emoji sentence with grammatical structure.
        
        Args:
            text: The text to translate
            sentence_type: Optional sentence type to enforce
            
        Returns:
            An EmojiSentence object representing the grammatical structure,
            or None if translation fails
        """
        return self.generator.translate_text_to_emoji_sentence(text, sentence_type)
    
    def create_emoji_sentence(
        self,
        sentence_type: SentenceType,
        subject: Optional[str] = None,
        predicate: str = None,
        object_: Optional[str] = None,
        tense: Tense = Tense.PRESENT,
        modifiers: List[str] = None,
        emotional_nuance: EmotionalNuance = EmotionalNuance.NEUTRAL,
        is_negative: bool = False
    ) -> Optional[EmojiSentence]:
        """
        Create an emoji sentence with the specified grammatical components.
        
        Args:
            sentence_type: The type of sentence to create
            subject: The subject of the sentence
            predicate: The predicate/verb of the sentence
            object_: The object of the sentence
            tense: The tense of the sentence
            modifiers: Additional modifiers to apply
            emotional_nuance: The emotional tone to convey
            is_negative: Whether the sentence is negative
            
        Returns:
            An EmojiSentence object representing the grammatical structure,
            or None if creation fails
        """
        return self.generator.generate_emoji_sentence(
            sentence_type=sentence_type,
            subject=subject,
            predicate=predicate,
            object_=object_,
            tense=tense,
            modifiers=modifiers,
            emotional_nuance=emotional_nuance,
            is_negative=is_negative
        )
    
    def interpret_emoji_sentence(self, emoji_sentence: EmojiSentence) -> str:
        """
        Interpret an emoji sentence into natural language.
        
        Args:
            emoji_sentence: The emoji sentence to interpret
            
        Returns:
            A natural language interpretation of the emoji sentence
        """
        # Use the EmojiTranslationEngine to get a basic translation
        basic_translation = self.emoji_engine.translate_emoji_to_text(emoji_sentence.emoji_sequence)
        
        # Apply grammatical knowledge to refine the translation
        # This would involve sophisticated NLP in a real implementation
        # For simplicity, we'll just return the basic translation
        return basic_translation


# Example usage
def emoji_grammar_system_example() -> None:
    """Example usage of the EmojiGrammarSystem."""
    
    # Initialize the system
    emoji_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(emoji_engine)
    
    print("="*80)
    print("                EmojiGrammarSystem Demonstration")
    print("="*80)
    
    # Example 1: Basic statement
    print("\n1. Basic Statement Grammar Example:\n")
    
    statement = grammar_system.create_emoji_sentence(
        sentence_type=SentenceType.STATEMENT,
        subject="I",
        predicate="like",
        object_="pizza"
    )
    
    if statement:
        print(f"Emoji Sentence: {statement}")
        print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in statement.elements])}")
        print(f"Interpretation: {grammar_system.interpret_emoji_sentence(statement)}")
    
    # Example 2: Question
    print("\n2. Question Grammar Example:\n")
    
    question = grammar_system.create_emoji_sentence(
        sentence_type=SentenceType.QUESTION,
        subject="you",
        predicate="understand",
        object_="emoji grammar"
    )
    
    if question:
        print(f"Emoji Sentence: {question}")
        print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in question.elements])}")
        print(f"Interpretation: {grammar_system.interpret_emoji_sentence(question)}")
    
    # Example 3: Command
    print("\n3. Command Grammar Example:\n")
    
    command = grammar_system.create_emoji_sentence(
        sentence_type=SentenceType.COMMAND,
        predicate="look",
        object_="document"
    )
    
    if command:
        print(f"Emoji Sentence: {command}")
        print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in command.elements])}")
        print(f"Interpretation: {grammar_system.interpret_emoji_sentence(command)}")
    
    # Example 4: Tense variations
    print("\n4. Tense Grammar Examples:\n")
    
    for tense in [Tense.PAST, Tense.PRESENT, Tense.FUTURE]:
        sentence = grammar_system.create_emoji_sentence(
            sentence_type=SentenceType.STATEMENT,
            subject="we",
            predicate="work",
            object_="project",
            tense=tense
        )
        
        if sentence:
            print(f"Tense: {tense.value}")
            print(f"Emoji Sentence: {sentence}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in sentence.elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(sentence)}")
            print()
    
    # Example 5: Negation
    print("\n5. Negative Grammar Example:\n")
    
    negative = grammar_system.create_emoji_sentence(
        sentence_type=SentenceType.STATEMENT,
        subject="I",
        predicate="agree",
        is_negative=True
    )
    
    if negative:
        print(f"Emoji Sentence: {negative}")
        print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in negative.elements])}")
        print(f"Interpretation: {grammar_system.interpret_emoji_sentence(negative)}")
    
    # Example 6: Conditional
    print("\n6. Conditional Grammar Example:\n")
    
    conditional = grammar_system.create_emoji_sentence(
        sentence_type=SentenceType.CONDITIONAL,
        subject="it",
        predicate="rain",
        object_="umbrella"
    )
    
    if conditional:
        print(f"Emoji Sentence: {conditional}")
        print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in conditional.elements])}")
        print(f"Interpretation: {grammar_system.interpret_emoji_sentence(conditional)}")
    
    # Example 7: Emotional nuance
    print("\n7. Emotional Nuance Examples:\n")
    
    for nuance in [EmotionalNuance.EXCITED, EmotionalNuance.SERIOUS, EmotionalNuance.URGENT]:
        sentence = grammar_system.create_emoji_sentence(
            sentence_type=SentenceType.STATEMENT,
            subject="meeting",
            predicate="start",
            tense=Tense.FUTURE,
            emotional_nuance=nuance
        )
        
        if sentence:
            print(f"Emotional Nuance: {nuance.value}")
            print(f"Emoji Sentence: {sentence}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in sentence.elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(sentence)}")
            print()
    
    # Example 8: Parsing emoji sequences
    print("\n8. Parsing Emoji Sequences:\n")
    
    sequences = [
        "👤👍🍕",    # "I like pizza"
        "❓👤👍🍕",   # "Do I like pizza?"
        "👀📝",      # "Look at the document"
        "👤🚫👍",    # "I don't agree"
        "☔👇➡️👤🏠"  # "If it rains, I stay home"
    ]
    
    for sequence in sequences:
        sentence = grammar_system.parse_emoji_sequence(sequence)
        if sentence:
            print(f"Emoji Sequence: {sequence}")
            print(f"Sentence Type: {sentence.sentence_type.value}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in sentence.elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(sentence)}")
            print()
    
    # Example 9: Translation with different sentence types
    print("\n9. Text to Emoji Grammar Translation:\n")
    
    texts = [
        "I am happy about the new project.",
        "Do you understand the emoji grammar system?",
        "Look at the document please.",
        "If it rains tomorrow, bring an umbrella.",
        "I don't like this proposal."
    ]
    
    for text in texts:
        sentence = grammar_system.translate_to_emoji_sentence(text)
        if sentence:
            print(f"Text: '{text}'")
            print(f"Emoji Sentence: {sentence}")
            print(f"Sentence Type: {sentence.sentence_type.value}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in sentence.elements])}")
            print()
    
    print("="*80)
    print("                Demonstration Complete")
    print("="*80)


if __name__ == "__main__":
    emoji_grammar_system_example()