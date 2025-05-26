import cmd
import emoji
"""
EmojiDialogueManager Component for Adaptive Bridge Builder Agent

This component manages multi-turn emoji-based conversations, maintaining context
and ensuring clarity across emoji-only exchanges. It builds upon the 
EmojiTranslationEngine and EmojiGrammarSystem to provide a complete system
for emoji-based dialogue.
"""

import re
import time
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Callable
from dataclasses import dataclass, field
from collections import deque

from emoji_translation_engine import (
    EmojiTranslationEngine,
    TranslationMode,
    AmbiguityResolutionStrategy
)

from emoji_grammar_system import (
    EmojiGrammarSystem,
    EmojiSentence,
    SentenceType,
    Tense,
    EmotionalNuance,
    GrammaticalRole
)


class DialogueState(Enum):
    """Represents the current state of an emoji dialogue."""
    GREETING = "greeting"            # Initial greeting phase
    ACTIVE = "active"                # Normal active conversation
    CLARIFICATION = "clarification"  # Requesting/providing clarification
    TRANSITION = "transition"        # Transitioning between modes
    FEEDBACK = "feedback"            # Collecting/providing feedback
    SUMMARIZING = "summarizing"      # Summarizing conversation
    CLOSING = "closing"              # Closing conversation
    IDLE = "idle"                    # No active conversation


class CommunicationMode(Enum):
    """Represents the current mode of communication."""
    EMOJI_ONLY = "emoji_only"        # Pure emoji-based communication
    TEXT_ONLY = "text_only"          # Pure text-based communication
    MIXED = "mixed"                  # Combination of emoji and text
    EMOJI_DOMINANT = "emoji_dominant"  # Mostly emoji with some text
    TEXT_DOMINANT = "text_dominant"  # Mostly text with some emoji


class ComplexityLevel(Enum):
    """Represents the complexity level of the conversation."""
    VERY_SIMPLE = "very_simple"      # Basic greetings, simple reactions
    SIMPLE = "simple"                # Simple statements, questions
    MODERATE = "moderate"            # Multi-part exchanges, some nuance
    COMPLEX = "complex"              # Abstract concepts, multi-step processes
    VERY_COMPLEX = "very_complex"    # Technical discussions, complex reasoning


class FeedbackType(Enum):
    """Types of feedback that can be exchanged in emoji conversations."""
    CONFIRMATION = "confirmation"    # Confirming understanding
    CONFUSION = "confusion"          # Indicating confusion
    AGREEMENT = "agreement"          # Expressing agreement
    DISAGREEMENT = "disagreement"    # Expressing disagreement
    EMPHASIS = "emphasis"            # Emphasizing a point
    QUESTION = "question"            # Asking for clarification
    ELABORATION = "elaboration"      # Requesting more details
    SUCCESS = "success"              # Indicating successful outcome
    FAILURE = "failure"              # Indicating unsuccessful outcome
    PARTIAL = "partial"              # Partial understanding/agreement


@dataclass
class ConversationContext:
    """
    Represents the context of an emoji conversation, storing state
    and relevant information to maintain context across exchanges.
    """
    # Core conversation tracking
    dialogue_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_state: DialogueState = DialogueState.IDLE
    communication_mode: CommunicationMode = CommunicationMode.MIXED
    
    # Subject tracking
    current_topics: List[str] = field(default_factory=list)
    referenced_entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    topic_history: List[str] = field(default_factory=list)
    
    # Complexity and adaptation
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    emoji_density: float = 0.5  # 0.0 = no emojis, 1.0 = all emojis
    
    # Context window management
    active_context_window: List[Dict[str, Any]] = field(default_factory=list)
    max_context_window_size: int = 10
    
    # User-specific adaptations
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    user_feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Time tracking
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    # Ambiguity management
    ambiguity_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarifications: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history."""
        self.conversation_history = [*self.conversation_history, message]
        self.active_context_window = [*self.active_context_window, message]
        
        # Prune context window if needed
        if len(self.active_context_window) > self.max_context_window_size:
            self.active_context_window.pop(0)
        
        # Update time
        self.last_update_time = time.time()
        
        # Track topics
        if "topics" in message:
            for topic in message["topics"]:
                if topic not in self.current_topics:
                    self.current_topics = [*self.current_topics, topic]
                if topic not in self.topic_history:
                    self.topic_history = [*self.topic_history, topic]
        
        # Track entities
        if "entities" in message:
            for entity_id, entity_data in message["entities"].items():
                self.referenced_entities = {**self.referenced_entities, entity_id: entity_data}
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent messages from the conversation history."""
        return self.conversation_history[-count:] if len(self.conversation_history) >= count else self.conversation_history
    
    def update_complexity(self, new_complexity: ComplexityLevel) -> None:
        """Update the complexity level of the conversation."""
        self.complexity_level = new_complexity
        
        # Adjust emoji density based on complexity
        if new_complexity == ComplexityLevel.VERY_SIMPLE:
            self.emoji_density = 0.9
        elif new_complexity == ComplexityLevel.SIMPLE:
            self.emoji_density = 0.7
        elif new_complexity == ComplexityLevel.MODERATE:
            self.emoji_density = 0.5
        elif new_complexity == ComplexityLevel.COMPLEX:
            self.emoji_density = 0.3
        elif new_complexity == ComplexityLevel.VERY_COMPLEX:
            self.emoji_density = 0.2
    
    def add_pending_clarification(self, emoji: str, possible_meanings: List[str], context: str) -> None:
        """Add a pending clarification for an ambiguous emoji."""
        self.pending_clarifications.append({
            "emoji": emoji,
            "possible_meanings": possible_meanings,
            "context": context,
            "time": time.time()
        })
    
    def resolve_clarification(self, emoji: str, selected_meaning: str) -> None:
        """Resolve a pending clarification with the selected meaning."""
        for i, clarification in enumerate(self.pending_clarifications):
            if clarification["emoji"] == emoji:
                # Move from pending to history
                clarification["selected_meaning"] = selected_meaning
                clarification["resolution_time"] = time.time()
                self.ambiguity_history = [*self.ambiguity_history, clarification]
                
                # Remove from pending
                self.pending_clarifications.pop(i)
                break
    
    def switch_communication_mode(self, new_mode: CommunicationMode) -> None:
        """Switch the current communication mode."""
        self.communication_mode = new_mode


class EmojiDialogueManager:
    """
    Manages multi-turn emoji-based conversations, maintaining context and clarity.
    
    This component builds upon the EmojiTranslationEngine and EmojiGrammarSystem
    to provide a complete system for emoji-based dialogue.
    """
    
    def __init__(
        self, 
        emoji_translation_engine: Optional[EmojiTranslationEngine] = None,
        emoji_grammar_system: Optional[EmojiGrammarSystem] = None
    ):
        """
        Initialize the EmojiDialogueManager.
        
        Args:
            emoji_translation_engine: Optional EmojiTranslationEngine instance.
                If None, a new instance will be created.
            emoji_grammar_system: Optional EmojiGrammarSystem instance.
                If None, a new instance will be created.
        """
        self.translation_engine = emoji_translation_engine or EmojiTranslationEngine()
        self.grammar_system = emoji_grammar_system or EmojiGrammarSystem(self.translation_engine)
        
        # Active conversations
        self.active_contexts: Dict[str, ConversationContext] = {}
        
        # Conversation history archive
        self.conversation_archive: Dict[str, ConversationContext] = {}
        
        # Feedback emoji sets for different feedback types
        self.feedback_emoji_sets = {
            FeedbackType.CONFIRMATION: ["👍", "✅", "👌", "🆗", "👏"],
            FeedbackType.CONFUSION: ["🤔", "❓", "🧐", "😕", "🙃"],
            FeedbackType.AGREEMENT: ["👍", "✅", "✔️", "💯", "🤝"],
            FeedbackType.DISAGREEMENT: ["👎", "❌", "🚫", "🔴", "⛔"],
            FeedbackType.EMPHASIS: ["‼️", "❗", "📢", "⭐", "🔥"],
            FeedbackType.QUESTION: ["❓", "🤷", "🧐", "🔍", "👀"],
            FeedbackType.ELABORATION: ["🔄", "📝", "🔍", "📈", "➕"],
            FeedbackType.SUCCESS: ["🎉", "🏆", "🌟", "💪", "🚀"],
            FeedbackType.FAILURE: ["💔", "🛑", "🚨", "😢", "⚠️"],
            FeedbackType.PARTIAL: ["🟠", "↔️", "😐", "➗", "🤏"]
        }
        
        # Transition patterns between modes
        self.transition_patterns = {
            "emoji_to_text": ["📝", "🔤", "ABC"],
            "text_to_emoji": ["😀", "🎭", "👻"],
            "mixed_mode": ["🔄", "🔀", "⚖️"]
        }
    
    def create_conversation(
        self, 
        dialogue_id: str,
        initial_mode: CommunicationMode = CommunicationMode.MIXED,
        complexity_level: ComplexityLevel = ComplexityLevel.MODERATE,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """
        Create a new emoji conversation context.
        
        Args:
            dialogue_id: Unique identifier for the conversation
            initial_mode: Initial communication mode
            complexity_level: Initial complexity level
            user_preferences: Optional dictionary of user preferences
            
        Returns:
            A new ConversationContext object
        """
        context = ConversationContext(
            dialogue_id=dialogue_id,
            communication_mode=initial_mode,
            complexity_level=complexity_level,
            user_preferences=user_preferences or {},
            current_state=DialogueState.GREETING
        )
        
        # Set initial emoji density based on complexity
        context.update_complexity(complexity_level)
        
        # Store in active contexts
        self.active_contexts = {**self.active_contexts, dialogue_id: context}
        
        return context
    
    def get_conversation(self, dialogue_id: str) -> Optional[ConversationContext]:
        """
        Get an active conversation context by ID.
        
        Args:
            dialogue_id: The ID of the conversation to retrieve
            
        Returns:
            The ConversationContext if found, None otherwise
        """
        return self.active_contexts.get(dialogue_id)
    
    def archive_conversation(self, dialogue_id: str) -> bool:
        """
        Archive a conversation, moving it from active to archived.
        
        Args:
            dialogue_id: The ID of the conversation to archive
            
        Returns:
            True if the conversation was archived, False otherwise
        """
        if dialogue_id in self.active_contexts:
            # Move to archive
            context = self.active_contexts[dialogue_id]
            context.current_state = DialogueState.IDLE
            self.conversation_archive = {**self.conversation_archive, dialogue_id: context}
            
            # Remove from active
            self.active_contexts = {k: v for k, v in self.active_contexts.items() if k != dialogue_id}
            
            return True
        
        return False
    
    def process_incoming_emoji_message(
        self, 
        dialogue_id: str, 
        emoji_sequence: str,
        sender_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incoming emoji message in a conversation.
        
        Args:
            dialogue_id: The ID of the conversation
            emoji_sequence: The emoji sequence to process
            sender_id: ID of the message sender
            metadata: Optional additional metadata
            
        Returns:
            A dictionary with processed message information
        """
        # Get or create conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            context = self.create_conversation(dialogue_id)
        
        # Parse the emoji sequence
        parsed_sentence = self.grammar_system.parse_emoji_sequence(emoji_sequence)
        
        # Translate to natural language
        natural_language = None
        confidence = 1.0
        ambiguities = []
        
        if parsed_sentence:
            # Use grammar-aware interpretation
            natural_language = self.grammar_system.interpret_emoji_sentence(parsed_sentence)
            
            # Extract grammatical information
            grammatical_info = {
                "sentence_type": parsed_sentence.sentence_type.value,
                "emotional_nuance": parsed_sentence.emotional_nuance.value,
                "subject": str(parsed_sentence.get_subject()) if parsed_sentence.get_subject() else None,
                "predicate": str(parsed_sentence.get_predicate()) if parsed_sentence.get_predicate() else None,
                "object": str(parsed_sentence.get_object()) if parsed_sentence.get_object() else None,
                "tense": parsed_sentence.get_tense().value if parsed_sentence.get_tense() else None
            }
        else:
            # Fall back to direct translation
            translation_result = self.translation_engine.translate_emoji_to_text(
                emoji_sequence, 
                resolution_strategy=AmbiguityResolutionStrategy.CONFIDENCE
            )
            
            if isinstance(translation_result, dict):
                natural_language = translation_result.get('translation')
                confidence = translation_result.get('confidence', 0.5)
                
                # Extract ambiguities
                if confidence < 0.7 and translation_result.get('alternatives'):
                    for alt, conf in translation_result.get('alternatives', []):
                        ambiguities.append({
                            "text": alt,
                            "confidence": conf
                        })
            else:
                natural_language = translation_result
            
            grammatical_info = None
        
        # Check for feedback indicators
        feedback_type = self._detect_feedback_type(emoji_sequence)
        
        # Check for transition indicators
        transition_to = self._detect_transition_indicator(emoji_sequence)
        if transition_to:
            context.current_state = DialogueState.TRANSITION
        
        # Check for ambiguity
        needs_clarification = len(ambiguities) > 0 and confidence < 0.7
        if needs_clarification:
            context.current_state = DialogueState.CLARIFICATION
            
            # Add ambiguous emojis to pending clarifications
            ambiguous_emojis = self._identify_ambiguous_emojis(emoji_sequence)
            for emoji, meanings in ambiguous_emojis.items():
                context.add_pending_clarification(emoji, meanings, emoji_sequence)
        
        # Extract potential topics
        topics = self._extract_topics(natural_language) if natural_language else []
        
        # Extract potential entities
        entities = self._extract_entities(natural_language) if natural_language else {}
        
        # Create message object
        message = {
            "id": f"{dialogue_id}-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": sender_id,
            "emoji_sequence": emoji_sequence,
            "natural_language": natural_language,
            "confidence": confidence,
            "ambiguities": ambiguities,
            "needs_clarification": needs_clarification,
            "feedback_type": feedback_type.value if feedback_type else None,
            "transition_to": transition_to,
            "grammatical_info": grammatical_info,
            "topics": topics,
            "entities": entities,
            "metadata": metadata or {}
        }
        
        # Add to conversation history
        context.add_message(message)
        
        # Update conversation state if needed
        self._update_conversation_state(context, message)
        
        return message
    
    def generate_emoji_response(
        self, 
        dialogue_id: str,
        intent: str,
        response_type: Optional[SentenceType] = None,
        emotional_nuance: Optional[EmotionalNuance] = None,
        include_feedback: bool = True,
        include_clarification: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an emoji response based on conversation context.
        
        Args:
            dialogue_id: The ID of the conversation
            intent: The intended message in natural language
            response_type: Optional specific type of response (question, statement, etc.)
            emotional_nuance: Optional emotional tone to convey
            include_feedback: Whether to include feedback indicators
            include_clarification: Whether to include clarification for potential ambiguities
            
        Returns:
            A dictionary with the generated response
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            # Create a new context if none exists
            context = self.create_conversation(dialogue_id)
        
        # Determine best sentence type if not specified
        if response_type is None:
            if "?" in intent:
                response_type = SentenceType.QUESTION
            elif any(cmd in intent.lower() for cmd in ["please", "would you", "could you", "do this"]):
                response_type = SentenceType.COMMAND
            elif any(cond in intent.lower() for cond in ["if", "when", "unless"]):
                response_type = SentenceType.CONDITIONAL
            else:
                response_type = SentenceType.STATEMENT
        
        # Determine emotional nuance if not specified
        if emotional_nuance is None:
            # Default to neutral but can be made more sophisticated
            emotional_nuance = EmotionalNuance.NEUTRAL
        
        # Generate base emoji response using grammar system
        emoji_sentence = self.grammar_system.translate_to_emoji_sentence(
            intent,
            sentence_type=response_type
        )
        
        # If translation with grammar rules failed, fall back to direct translation
        base_emoji_sequence = ""
        if emoji_sentence:
            base_emoji_sequence = str(emoji_sentence)
        else:
            # Use direct translation with appropriate mode based on complexity
            mode = TranslationMode.LITERAL
            if context.complexity_level == ComplexityLevel.SIMPLE:
                mode = TranslationMode.SEMANTIC
            elif context.complexity_level == ComplexityLevel.MODERATE:
                mode = TranslationMode.EMOTIONAL
            elif context.complexity_level == ComplexityLevel.COMPLEX:
                mode = TranslationMode.SUMMARIZED
            elif context.complexity_level == ComplexityLevel.VERY_COMPLEX:
                mode = TranslationMode.SEMANTIC
            
            base_emoji_sequence = self.translation_engine.translate_text_to_emoji(intent, mode)
        
        # Add feedback indicators if needed
        final_emoji_sequence = base_emoji_sequence
        feedback_type = None
        
        if include_feedback:
            # Determine appropriate feedback type
            if "confirm" in intent.lower() or "understood" in intent.lower():
                feedback_type = FeedbackType.CONFIRMATION
            elif "confused" in intent.lower() or "don't understand" in intent.lower():
                feedback_type = FeedbackType.CONFUSION
            elif "agree" in intent.lower():
                feedback_type = FeedbackType.AGREEMENT
            elif "disagree" in intent.lower():
                feedback_type = FeedbackType.DISAGREEMENT
            
            # Add feedback emoji if a type was determined
            if feedback_type:
                feedback_emoji = self._get_feedback_emoji(feedback_type)
                final_emoji_sequence = f"{feedback_emoji} {final_emoji_sequence}"
        
        # Add clarification helpers if needed
        clarification_info = None
        if include_clarification and context.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            # Identify potentially ambiguous emojis in our response
            ambiguous_emojis = self._identify_ambiguous_emojis(base_emoji_sequence)
            
            if ambiguous_emojis:
                # Add clarification info
                clarification_info = {
                    "ambiguous_emojis": ambiguous_emojis,
                    "suggestions": {}
                }
                
                # For each ambiguous emoji, suggest a clearer alternative or explanation
                for emoji, meanings in ambiguous_emojis.items():
                    if len(meanings) > 1:
                        primary_meaning = meanings[0]
                        clarification_info["suggestions"][emoji] = {
                            "intended_meaning": primary_meaning,
                            "alternative": f"{emoji}({primary_meaning})"
                        }
        
        # Create response message
        response = {
            "id": f"{dialogue_id}-response-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": "system",
            "original_intent": intent,
            "emoji_sequence": final_emoji_sequence,
            "base_sequence": base_emoji_sequence,
            "sentence_type": response_type.value if response_type else None,
            "emotional_nuance": emotional_nuance.value if emotional_nuance else None,
            "feedback_type": feedback_type.value if feedback_type else None,
            "clarification_info": clarification_info,
            "topics": self._extract_topics(intent),
            "entities": self._extract_entities(intent)
        }
        
        # Add to conversation history
        context.add_message(response)
        
        return response
    
    def request_clarification(
        self, 
        dialogue_id: str,
        ambiguous_emojis: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Generate a clarification request for ambiguous emojis.
        
        Args:
            dialogue_id: The ID of the conversation
            ambiguous_emojis: Dictionary mapping emojis to possible meanings
            
        Returns:
            A clarification request message
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            return {"error": "Conversation not found"}
        
        # Set state to clarification
        context.current_state = DialogueState.CLARIFICATION
        
        # Build clarification message
        clarification_parts = []
        for emoji, meanings in ambiguous_emojis.items():
            # Add ambiguous emoji with question mark
            clarification_parts.append(f"{emoji}❓")
            
            # Add to pending clarifications
            context.add_pending_clarification(emoji, meanings, "".join(clarification_parts))
        
        # Create the clarification message
        emoji_sequence = "".join(clarification_parts)
        natural_language = f"Could you clarify what you meant by these emojis: {', '.join(ambiguous_emojis.keys())}?"
        
        clarification_message = {
            "id": f"{dialogue_id}-clarification-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": "system",
            "emoji_sequence": emoji_sequence,
            "natural_language": natural_language,
            "ambiguous_emojis": ambiguous_emojis,
            "is_clarification_request": True
        }
        
        # Add to conversation history
        context.add_message(clarification_message)
        
        return clarification_message
    
    def provide_clarification(
        self, 
        dialogue_id: str,
        emoji: str,
        selected_meaning: str
    ) -> Dict[str, Any]:
        """
        Provide clarification for an ambiguous emoji.
        
        Args:
            dialogue_id: The ID of the conversation
            emoji: The emoji being clarified
            selected_meaning: The selected meaning for the emoji
            
        Returns:
            A response message with the clarification
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            return {"error": "Conversation not found"}
        
        # Resolve the clarification
        context.resolve_clarification(emoji, selected_meaning)
        
        # Create confirmation message
        confirmation_emoji = self._get_feedback_emoji(FeedbackType.CONFIRMATION)
        emoji_sequence = f"{confirmation_emoji} {emoji}"
        
        clarification_message = {
            "id": f"{dialogue_id}-clarification-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": "system",
            "emoji_sequence": emoji_sequence,
            "natural_language": f"Thanks for clarifying that {emoji} means '{selected_meaning}'.",
            "clarified_emoji": emoji,
            "selected_meaning": selected_meaning,
            "is_clarification_response": True
        }
        
        # Add to conversation history
        context.add_message(clarification_message)
        
        # Update conversation state if all pending clarifications are resolved
        if len(context.pending_clarifications) == 0:
            context.current_state = DialogueState.ACTIVE
        
        return clarification_message
    
    def generate_mode_transition(
        self, 
        dialogue_id: str,
        target_mode: CommunicationMode
    ) -> Dict[str, Any]:
        """
        Generate a transition message to switch communication modes.
        
        Args:
            dialogue_id: The ID of the conversation
            target_mode: The communication mode to transition to
            
        Returns:
            A transition message
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            return {"error": "Conversation not found"}
        
        # Set state to transition
        context.current_state = DialogueState.TRANSITION
        
        # Get appropriate transition emoji
        transition_emoji = ""
        if target_mode == CommunicationMode.EMOJI_ONLY:
            transition_emoji = "".join(self.transition_patterns["text_to_emoji"])
        elif target_mode == CommunicationMode.TEXT_ONLY:
            transition_emoji = "".join(self.transition_patterns["emoji_to_text"])
        else:
            transition_emoji = "".join(self.transition_patterns["mixed_mode"])
        
        # Create transition message
        natural_language = f"Switching to {target_mode.value} communication mode."
        
        transition_message = {
            "id": f"{dialogue_id}-transition-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": "system",
            "emoji_sequence": transition_emoji,
            "natural_language": natural_language,
            "transition_to": target_mode.value,
            "is_transition": True
        }
        
        # Update the context with the new mode
        context.switch_communication_mode(target_mode)
        
        # Add to conversation history
        context.add_message(transition_message)
        
        # After transition is complete, return to active state
        context.current_state = DialogueState.ACTIVE
        
        return transition_message
    
    def provide_feedback(
        self, 
        dialogue_id: str,
        feedback_type: FeedbackType,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a feedback message.
        
        Args:
            dialogue_id: The ID of the conversation
            feedback_type: The type of feedback to provide
            message_id: Optional ID of the message being responded to
            
        Returns:
            A feedback message
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            return {"error": "Conversation not found"}
        
        # Get appropriate feedback emoji
        feedback_emoji = self._get_feedback_emoji(feedback_type)
        
        # Create feedback message
        feedback_message = {
            "id": f"{dialogue_id}-feedback-{len(context.conversation_history) + 1}",
            "timestamp": time.time(),
            "sender_id": "system",
            "emoji_sequence": feedback_emoji,
            "natural_language": self._get_feedback_text(feedback_type),
            "feedback_type": feedback_type.value,
            "response_to": message_id,
            "is_feedback": True
        }
        
        # Add to conversation history
        context.add_message(feedback_message)
        
        return feedback_message
    
    def get_conversation_summary(
        self, 
        dialogue_id: str,
        include_translations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a summary of the conversation.
        
        Args:
            dialogue_id: The ID of the conversation
            include_translations: Whether to include natural language translations
            
        Returns:
            A summary of the conversation
        """
        # Get conversation context
        context = self.get_conversation(dialogue_id)
        if context is None:
            context = self.conversation_archive.get(dialogue_id)
            if context is None:
                return {"error": "Conversation not found"}
        
        # Extract emoji-only and parallel text sequences
        emoji_sequence = []
        translations = []
        
        for message in context.conversation_history:
            if "emoji_sequence" in message:
                emoji_sequence.append({
                    "sender_id": message["sender_id"],
                    "emoji": message["emoji_sequence"],
                    "timestamp": message["timestamp"]
                })
                
                if include_translations and "natural_language" in message:
                    translations.append({
                        "sender_id": message["sender_id"],
                        "text": message["natural_language"],
                        "timestamp": message["timestamp"]
                    })
        
        # Extract topics and entities
        all_topics = context.topic_history
        all_entities = dict(context.referenced_entities)
        
        # Create summary
        summary = {
            "dialogue_id": dialogue_id,
            "creation_time": context.creation_time,
            "last_update_time": context.last_update_time,
            "message_count": len(context.conversation_history),
            "complexity_level": context.complexity_level.value,
            "communication_mode": context.communication_mode.value,
            "current_state": context.current_state.value,
            "emoji_density": context.emoji_density,
            "emoji_sequence": emoji_sequence
        }
        
        if include_translations:
            summary["translations"] = translations
        
        summary["topics"] = all_topics
        summary["entities"] = all_entities
        
        # Include ambiguity history if available
        if context.ambiguity_history:
            summary["ambiguity_resolutions"] = [
                {
                    "emoji": item["emoji"],
                    "selected_meaning": item["selected_meaning"],
                    "resolution_time": item["resolution_time"]
                }
                for item in context.ambiguity_history
            ]
        
        return summary
    
    def adjust_emoji_density(
        self, 
        dialogue_id: str,
        new_complexity: ComplexityLevel
    ) -> bool:
        """
        Adjust the emoji density based on conversation complexity.
        
        Args:
            dialogue_id: The ID of the conversation
            new_complexity: The new complexity level
            
        Returns:
            True if adjustment was successful, False otherwise
        """
        context = self.get_conversation(dialogue_id)
        if context is None:
            return False
        
        context.update_complexity(new_complexity)
        return True
    
    def _update_conversation_state(self, context: ConversationContext, message: Dict[str, Any]) -> None:
        """Update conversation state based on the latest message."""
        # If we're in greeting state and this isn't the first message, move to active
        if context.current_state == DialogueState.GREETING and len(context.conversation_history) > 1:
            context.current_state = DialogueState.ACTIVE
        
        # If we're in clarification and there are no pending clarifications, return to active
        if context.current_state == DialogueState.CLARIFICATION and len(context.pending_clarifications) == 0:
            context.current_state = DialogueState.ACTIVE
        
        # If we're in transition and the message isn't a transition, return to active
        if context.current_state == DialogueState.TRANSITION and not message.get("is_transition", False):
            context.current_state = DialogueState.ACTIVE
        
        # If we're in feedback and the message isn't feedback, return to active
        if context.current_state == DialogueState.FEEDBACK and not message.get("is_feedback", False):
            context.current_state = DialogueState.ACTIVE
        
        # Check for closing indicators
        if self._is_closing_message(message.get("emoji_sequence", "")):
            context.current_state = DialogueState.CLOSING
    
    def _is_closing_message(self, emoji_sequence: str) -> bool:
        """Check if an emoji sequence indicates conversation closing."""
        closing_indicators = ["👋", "🏁", "✅", "🔚", "👍"]
        if emoji_sequence and any(indicator in emoji_sequence for indicator in closing_indicators):
            # Additional check to filter out false positives
            # Only consider it a closing if it's a short message with mainly closing indicators
            if len(emoji_sequence) <= 5 and sum(emoji_sequence.count(ind) for ind in closing_indicators) >= 1:
                return True
        return False
    
    def _detect_feedback_type(self, emoji_sequence: str) -> Optional[FeedbackType]:
        """
        Detect the feedback type from an emoji sequence.
        
        Args:
            emoji_sequence: The emoji sequence to analyze
            
        Returns:
            The detected feedback type, or None if no feedback detected
        """
        # Check for each feedback type
        for feedback_type, emoji_set in self.feedback_emoji_sets.items():
            # If any of the emojis in the set are found at the start of the sequence
            for emoji in emoji_set:
                if emoji_sequence.startswith(emoji) or emoji in emoji_sequence[:3]:
                    return feedback_type
        
        return None
    
    def _detect_transition_indicator(self, emoji_sequence: str) -> Optional[str]:
        """
        Detect if an emoji sequence contains mode transition indicators.
        
        Args:
            emoji_sequence: The emoji sequence to analyze
            
        Returns:
            The target mode if a transition is detected, None otherwise
        """
        # Check for each transition pattern
        if any(emoji in emoji_sequence for emoji in self.transition_patterns["emoji_to_text"]):
            return CommunicationMode.TEXT_ONLY.value
        
        if any(emoji in emoji_sequence for emoji in self.transition_patterns["text_to_emoji"]):
            return CommunicationMode.EMOJI_ONLY.value
        
        if any(emoji in emoji_sequence for emoji in self.transition_patterns["mixed_mode"]):
            return CommunicationMode.MIXED.value
        
        return None
    
    def _identify_ambiguous_emojis(self, emoji_sequence: str) -> Dict[str, List[str]]:
        """
        Identify potentially ambiguous emojis in a sequence.
        
        Args:
            emoji_sequence: The emoji sequence to analyze
            
        Returns:
            A dictionary mapping ambiguous emojis to their possible meanings
        """
        # Extract individual emojis
        emoji_pattern = re.compile(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])')
        emojis = emoji_pattern.findall(emoji_sequence)
        
        ambiguous_emojis = {}
        
        for emoji in emojis:
            # Get meanings using the translation engine
            result = self.translation_engine.translate_emoji_to_text(
                emoji, 
                resolution_strategy=AmbiguityResolutionStrategy.MULTIPLE
            )
            
            # If there are multiple meanings, consider it ambiguous
            if isinstance(result, list) and len(result) > 1:
                ambiguous_emojis[emoji] = result
            elif isinstance(result, dict) and result.get('alternatives') and len(result.get('alternatives', [])) > 1:
                ambiguous_emojis[emoji] = [result.get('translation')] + [alt for alt, _ in result.get('alternatives', [])]
        
        return ambiguous_emojis
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topics from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A list of extracted topics
        """
        # This would be more sophisticated in a real implementation,
        # potentially using NLP techniques for topic extraction
        
        # Simple approach: extract nouns and noun phrases
        if not text:
            return []
        
        # Split into words and remove punctuation
        words = text.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
        
        # Filter out common stop words
        stop_words = {'a', 'an', 'the', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'}
        
        topics = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique topics
        return list(set(topics))
    
    def _extract_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract potential entities from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary of entities with their properties
        """
        # This would be more sophisticated in a real implementation,
        # potentially using named entity recognition techniques
        
        # For this example, we'll use a very simplified approach
        entities = {}
        
        if not text:
            return entities
        
        # Look for capitalized words as potential named entities
        words = text.split()
        
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 1 and word.lower() not in {'i', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d'}:
                # Clean up punctuation
                clean_word = word.strip('.,!?;:(){}[]"\'')
                
                if clean_word:
                    # Generate a unique ID for the entity
                    entity_id = f"entity_{len(entities) + 1}"
                    
                    # Try to determine entity type from context
                    entity_type = "unknown"
                    
                    # Look at the previous word for clues
                    if i > 0:
                        prev_word = words[i-1].lower()
                        if prev_word in {'mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'miss'}:
                            entity_type = "person"
                        elif prev_word in {'company', 'corporation', 'inc', 'org', 'organization'}:
                            entity_type = "organization"
                        elif prev_word in {'project', 'initiative', 'program'}:
                            entity_type = "project"
                    
                    entities[entity_id] = {
                        "name": clean_word,
                        "type": entity_type,
                        "mentions": 1
                    }
        
        return entities
    
    def _get_feedback_emoji(self, feedback_type: FeedbackType) -> str:
        """
        Get an emoji representing a feedback type.
        
        Args:
            feedback_type: The type of feedback
            
        Returns:
            An emoji representing the feedback type
        """
        emoji_set = self.feedback_emoji_sets.get(feedback_type, [])
        if emoji_set:
            # Choose a random emoji from the set for variety
            import random
            return random.choice(emoji_set)
        
        # Default fallback
        return "👍"
    
    def _get_feedback_text(self, feedback_type: FeedbackType) -> str:
        """
        Get text representing a feedback type.
        
        Args:
            feedback_type: The type of feedback
            
        Returns:
            Text representing the feedback type
        """
        feedback_texts = {
            FeedbackType.CONFIRMATION: "I understand and confirm.",
            FeedbackType.CONFUSION: "I'm not sure I understand.",
            FeedbackType.AGREEMENT: "I agree with that.",
            FeedbackType.DISAGREEMENT: "I disagree with that.",
            FeedbackType.EMPHASIS: "This is important to note.",
            FeedbackType.QUESTION: "I have a question about this.",
            FeedbackType.ELABORATION: "Could you provide more details?",
            FeedbackType.SUCCESS: "Great! That was successful.",
            FeedbackType.FAILURE: "Unfortunately, that didn't work.",
            FeedbackType.PARTIAL: "I partially understand or agree."
        }
        
        return feedback_texts.get(feedback_type, "Acknowledged.")


# Example usage
def emoji_dialogue_manager_example() -> None:
    """Example demonstrating the EmojiDialogueManager."""
    print("\n=== EmojiDialogueManager Demonstration ===\n")
    
    # Initialize components
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    dialogue_id = "demo-conversation-1"
    context = dialogue_manager.create_conversation(
        dialogue_id=dialogue_id,
        initial_mode=CommunicationMode.MIXED,
        complexity_level=ComplexityLevel.MODERATE
    )
    
    print(f"Created conversation with ID: {dialogue_id}")
    print(f"Initial mode: {context.communication_mode.value}")
    print(f"Initial complexity: {context.complexity_level.value}")
    print(f"Emoji density: {context.emoji_density}")
    
    # Simulate a conversation
    print("\n1. User sends a greeting")
    user_message = dialogue_manager.process_incoming_emoji_message(
        dialogue_id=dialogue_id,
        emoji_sequence="👋😊",
        sender_id="user"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Interpreted as: {user_message['natural_language']}")
    
    # System responds
    print("\n2. System responds with a greeting")
    system_response = dialogue_manager.generate_emoji_response(
        dialogue_id=dialogue_id,
        intent="Hello! How can I help you today?",
        emotional_nuance=EmotionalNuance.EXCITED
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Original intent: {system_response['original_intent']}")
    
    # User asks a question
    print("\n3. User asks a question")
    user_message = dialogue_manager.process_incoming_emoji_message(
        dialogue_id=dialogue_id,
        emoji_sequence="❓🌦️🔜",
        sender_id="user"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Interpreted as: {user_message['natural_language']}")
    
    # Check if clarification is needed
    if user_message['needs_clarification']:
        print("\n4. System requests clarification for ambiguous emoji")
        clarification_request = dialogue_manager.request_clarification(
            dialogue_id=dialogue_id,
            ambiguous_emojis={"🌦️": ["weather", "rainy", "partly cloudy", "light rain"]}
        )
        print(f"System: {clarification_request['emoji_sequence']}")
        print(f"Request: {clarification_request['natural_language']}")
        
        # User provides clarification
        print("\n5. User provides clarification")
        user_clarification = dialogue_manager.provide_clarification(
            dialogue_id=dialogue_id,
            emoji="🌦️",
            selected_meaning="weather"
        )
        print(f"User selected: {user_clarification['selected_meaning']} for {user_clarification['clarified_emoji']}")
    
    # System answers the question
    print("\n6. System answers the weather question")
    system_response = dialogue_manager.generate_emoji_response(
        dialogue_id=dialogue_id,
        intent="The weather tomorrow will be partly sunny with a chance of light rain in the afternoon.",
        emotional_nuance=EmotionalNuance.NEUTRAL
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Original intent: {system_response['original_intent']}")
    
    # User sends a more complex query
    print("\n7. User sends a more complex query about project")
    user_message = dialogue_manager.process_incoming_emoji_message(
        dialogue_id=dialogue_id,
        emoji_sequence="📋🔄🕒❓",
        sender_id="user"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Interpreted as: {user_message['natural_language']}")
    
    # System identifies increased complexity and adjusts density
    print("\n8. System identifies increased complexity and adjusts")
    dialogue_manager.adjust_emoji_density(dialogue_id, ComplexityLevel.COMPLEX)
    updated_context = dialogue_manager.get_conversation(dialogue_id)
    print(f"Updated complexity: {updated_context.complexity_level.value}")
    print(f"Updated emoji density: {updated_context.emoji_density}")
    
    # System responds to complex query
    system_response = dialogue_manager.generate_emoji_response(
        dialogue_id=dialogue_id,
        intent="The project schedule has been updated. Phase 1 is complete, Phase 2 is in progress (about 60% done), and Phase 3 will start next month. We're currently on track to meet the final deadline.",
        emotional_nuance=EmotionalNuance.NEUTRAL,
        include_clarification=True
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Original intent: {system_response['original_intent']}")
    
    # Display clarification information if provided
    if system_response.get('clarification_info'):
        print("\nClarification info provided:")
        for emoji, suggestion in system_response['clarification_info'].get('suggestions', {}).items():
            print(f"  {emoji} means '{suggestion['intended_meaning']}'")
    
    # User indicates confusion
    print("\n9. User indicates confusion")
    user_message = dialogue_manager.process_incoming_emoji_message(
        dialogue_id=dialogue_id,
        emoji_sequence="🤔❓",
        sender_id="user"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Interpreted as: {user_message['natural_language']}")
    print(f"Detected feedback type: {user_message['feedback_type']}")
    
    # System provides simplification
    print("\n10. System provides simplified explanation")
    dialogue_manager.adjust_emoji_density(dialogue_id, ComplexityLevel.SIMPLE)
    system_response = dialogue_manager.generate_emoji_response(
        dialogue_id=dialogue_id,
        intent="Let me simplify: Project is going well. Part 1: done. Part 2: working on it now. Part 3: starts soon. We'll finish on time.",
        emotional_nuance=EmotionalNuance.GENTLE
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Original intent: {system_response['original_intent']}")
    
    # User confirms understanding
    print("\n11. User confirms understanding")
    user_message = dialogue_manager.process_incoming_emoji_message(
        dialogue_id=dialogue_id,
        emoji_sequence="👍✅",
        sender_id="user"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Interpreted as: {user_message['natural_language']}")
    print(f"Detected feedback type: {user_message['feedback_type']}")
    
    # System suggests switching communication modes
    print("\n12. System suggests switching to text")
    transition_message = dialogue_manager.generate_mode_transition(
        dialogue_id=dialogue_id,
        target_mode=CommunicationMode.TEXT_ONLY
    )
    print(f"System: {transition_message['emoji_sequence']}")
    print(f"Message: {transition_message['natural_language']}")
    
    # Generate conversation summary
    print("\n13. Generate conversation summary")
    summary = dialogue_manager.get_conversation_summary(dialogue_id)
    print(f"Conversation summary:")
    print(f"  Dialogue ID: {summary['dialogue_id']}")
    print(f"  Message count: {summary['message_count']}")
    print(f"  Current mode: {summary['communication_mode']}")
    print(f"  Current state: {summary['current_state']}")
    print(f"  Topics discussed: {', '.join(summary['topics']) if summary['topics'] else 'None'}")
    
    print("\n=== Demonstration Complete ===\n")


if __name__ == "__main__":
    emoji_dialogue_manager_example()