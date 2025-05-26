import emoji
"""
Example demonstrating the EmojiDialogueManager component functionality.

This file provides concrete examples of multi-turn emoji conversations
and how the system maintains context and clarity across exchanges.
"""

from emoji_dialogue_manager import (
    EmojiDialogueManager,
    DialogueState,
    CommunicationMode,
    ComplexityLevel,
    FeedbackType,
    EmotionalNuance,
    SentenceType
)
from emoji_translation_engine import EmojiTranslationEngine
from emoji_grammar_system import EmojiGrammarSystem

def demonstrate_multi_turn_conversation() -> None:
    """Demonstrate a multi-turn conversation with context maintenance."""
    print("\n=== Multi-Turn Conversation with Context Maintenance ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    conversation_id = "demo-1"
    context = dialogue_manager.create_conversation(conversation_id)
    
    print("Starting a new conversation about project planning...")
    
    # User greeting
    user_greeting = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👋😊",
        "user-1"
    )
    print(f"User: {user_greeting['emoji_sequence']}")
    print(f"System interpretation: {user_greeting['natural_language']}")
    
    # System response
    system_greeting = dialogue_manager.generate_emoji_response(
        conversation_id,
        "Hello! I'm ready to help with project planning. What would you like to discuss?",
        emotional_nuance=EmotionalNuance.EXCITED
    )
    print(f"System: {system_greeting['emoji_sequence']}")
    print(f"Intent: {system_greeting['original_intent']}")
    
    # User asks about project status
    user_question = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "📋📊❓",
        "user-1"
    )
    print(f"User: {user_question['emoji_sequence']}")
    print(f"System interpretation: {user_question['natural_language']}")
    
    # System response about project status
    system_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        "The project is currently in the planning phase. We've completed 25% of the initial tasks and are on track to meet our first milestone next week.",
        emotional_nuance=EmotionalNuance.NEUTRAL
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Intent: {system_response['original_intent']}")
    
    # User asks about team members
    user_question = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👥❓",
        "user-1"
    )
    print(f"User: {user_question['emoji_sequence']}")
    print(f"System interpretation: {user_question['natural_language']}")
    
    # System response about team members
    system_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        "We have 5 team members assigned to this project: 2 developers, 1 designer, 1 product manager, and 1 QA specialist.",
        emotional_nuance=EmotionalNuance.NEUTRAL
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Intent: {system_response['original_intent']}")
    
    # User asks when the project will be completed
    user_question = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "📋✅⏰❓",
        "user-1"
    )
    print(f"User: {user_question['emoji_sequence']}")
    print(f"System interpretation: {user_question['natural_language']}")
    
    # System response about timeline
    system_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        "Based on our current progress, the project is expected to be completed in approximately 3 months. The exact target date is March 15th, 2026.",
        emotional_nuance=EmotionalNuance.NEUTRAL
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Intent: {system_response['original_intent']}")
    
    # User expresses satisfaction
    user_feedback = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👍💯",
        "user-1"
    )
    print(f"User: {user_feedback['emoji_sequence']}")
    print(f"System interpretation: {user_feedback['natural_language']}")
    print(f"Detected feedback type: {user_feedback['feedback_type']}")
    
    # Check conversation state and context
    context = dialogue_manager.get_conversation(conversation_id)
    print("\nConversation summary after multi-turn exchange:")
    print(f"Current state: {context.current_state.value}")
    print(f"Topics discussed: {', '.join(context.topic_history)}")
    
    # Generate a conversation summary
    summary = dialogue_manager.get_conversation_summary(conversation_id)
    print(f"Total messages: {summary['message_count']}")
    print(f"Active topics: {', '.join(summary['topics']) if summary['topics'] else 'None'}")


def demonstrate_ambiguity_handling() -> None:
    """Demonstrate ambiguity detection and resolution in emoji conversations."""
    print("\n=== Ambiguity Detection and Resolution ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    conversation_id = "demo-2"
    context = dialogue_manager.create_conversation(conversation_id)
    
    print("Starting a conversation with potentially ambiguous emojis...")
    
    # User sends a message with potentially ambiguous emoji
    user_message = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👋 🔥📱❓",
        "user-2"
    )
    print(f"User: {user_message['emoji_sequence']}")
    print(f"Initial interpretation: {user_message['natural_language']}")
    
    # Check if clarification is needed
    if user_message['needs_clarification']:
        print("System detected ambiguity and needs clarification.")
        
        # For demonstration purposes, we'll create ambiguous emojis manually
        # In a real scenario, these would come from the message processing
        ambiguous_emojis = {
            "🔥": ["fire", "hot", "trending", "popular", "destruction"],
            "📱": ["phone", "mobile", "device", "app", "technology"]
        }
        
        # Request clarification
        clarification_request = dialogue_manager.request_clarification(
            conversation_id,
            ambiguous_emojis
        )
        print(f"System clarification request: {clarification_request['emoji_sequence']}")
        print(f"Natural language request: {clarification_request['natural_language']}")
        
        # User provides clarification for 🔥
        fire_clarification = dialogue_manager.provide_clarification(
            conversation_id,
            "🔥",
            "popular"
        )
        print(f"User clarifies that 🔥 means: {fire_clarification['selected_meaning']}")
        
        # User provides clarification for 📱
        phone_clarification = dialogue_manager.provide_clarification(
            conversation_id,
            "📱",
            "app"
        )
        print(f"User clarifies that 📱 means: {phone_clarification['selected_meaning']}")
        
        # System now responds with the clarified understanding
        system_response = dialogue_manager.generate_emoji_response(
            conversation_id,
            "Hello! Yes, I can recommend some popular apps. What type are you interested in?",
            emotional_nuance=EmotionalNuance.EXCITED
        )
        print(f"System: {system_response['emoji_sequence']}")
        print(f"Intent: {system_response['original_intent']}")
        
        # User responds with a more specific request
        user_message = dialogue_manager.process_incoming_emoji_message(
            conversation_id,
            "🎮🎯",
            "user-2"
        )
        print(f"User: {user_message['emoji_sequence']}")
        print(f"Interpretation: {user_message['natural_language']}")
        
        # System response with game recommendations
        system_response = dialogue_manager.generate_emoji_response(
            conversation_id,
            "For gaming apps with good precision/targeting, I recommend: Rocket League Sideswipe, Archer Legend, and Golf Rival. All are popular and highly rated.",
            emotional_nuance=EmotionalNuance.NEUTRAL
        )
        print(f"System: {system_response['emoji_sequence']}")
        print(f"Intent: {system_response['original_intent']}")
    else:
        print("No ambiguity detected.")
    
    # Check ambiguity history to see what's been resolved
    context = dialogue_manager.get_conversation(conversation_id)
    print("\nAmbiguity resolution history:")
    for item in context.ambiguity_history:
        print(f"Emoji {item['emoji']} was clarified to mean '{item['selected_meaning']}'")


def demonstrate_complexity_adaptation() -> None:
    """Demonstrate adaptive emoji density based on conversation complexity."""
    print("\n=== Adaptive Emoji Density Based on Complexity ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    conversation_id = "demo-3"
    
    print("Demonstrating emoji density adaptation across complexity levels...\n")
    
    # Very simple conversation (high emoji density)
    context = dialogue_manager.create_conversation(
        conversation_id, 
        complexity_level=ComplexityLevel.VERY_SIMPLE
    )
    print(f"Very Simple Conversation (Emoji Density: {context.emoji_density})")
    
    simple_message = "Hello! How are you today?"
    simple_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        simple_message
    )
    print(f"Text: '{simple_message}'")
    print(f"Emoji: {simple_response['emoji_sequence']}")
    print()
    
    # Switch to moderate complexity
    dialogue_manager.adjust_emoji_density(conversation_id, ComplexityLevel.MODERATE)
    context = dialogue_manager.get_conversation(conversation_id)
    print(f"Moderate Complexity Conversation (Emoji Density: {context.emoji_density})")
    
    moderate_message = "I'd like to discuss the project timeline and identify any potential risks that might affect our delivery date."
    moderate_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        moderate_message
    )
    print(f"Text: '{moderate_message}'")
    print(f"Emoji: {moderate_response['emoji_sequence']}")
    print()
    
    # Switch to complex conversation
    dialogue_manager.adjust_emoji_density(conversation_id, ComplexityLevel.COMPLEX)
    context = dialogue_manager.get_conversation(conversation_id)
    print(f"Complex Conversation (Emoji Density: {context.emoji_density})")
    
    complex_message = "The implementation of the new authentication system requires us to refactor the existing user management module while ensuring backward compatibility with the legacy APIs that some clients still depend on."
    complex_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        complex_message
    )
    print(f"Text: '{complex_message}'")
    print(f"Emoji: {complex_response['emoji_sequence']}")
    print()
    
    # Switch to very complex conversation
    dialogue_manager.adjust_emoji_density(conversation_id, ComplexityLevel.VERY_COMPLEX)
    context = dialogue_manager.get_conversation(conversation_id)
    print(f"Very Complex Conversation (Emoji Density: {context.emoji_density})")
    
    very_complex_message = "The distributed transaction processing system must maintain ACID properties across multiple microservices while handling network partitioning through a combination of the Saga pattern and eventual consistency models, all while minimizing latency and ensuring data integrity."
    very_complex_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        very_complex_message
    )
    print(f"Text: '{very_complex_message}'")
    print(f"Emoji: {very_complex_response['emoji_sequence']}")


def demonstrate_mode_transitions() -> None:
    """Demonstrate transitions between different communication modes."""
    print("\n=== Communication Mode Transitions ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    conversation_id = "demo-4"
    context = dialogue_manager.create_conversation(conversation_id, initial_mode=CommunicationMode.MIXED)
    
    print(f"Starting in {context.communication_mode.value} mode...")
    
    # Initial exchange in mixed mode
    user_message = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👋 Can we discuss the project?",
        "user-4"
    )
    print(f"User: {user_message['emoji_sequence']}")
    
    system_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        "Hello! Yes, I'd be happy to discuss the project. What aspects would you like to focus on?"
    )
    print(f"System: {system_response['emoji_sequence']}")
    
    # Transition to emoji-only mode
    print("\nUser requests to switch to emoji-only mode...")
    transition_request = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "🎭👻",  # Using emoji mode transition indicator
        "user-4"
    )
    
    if transition_request.get('transition_to') == CommunicationMode.EMOJI_ONLY.value:
        # System acknowledges and transitions
        transition_message = dialogue_manager.generate_mode_transition(
            conversation_id,
            CommunicationMode.EMOJI_ONLY
        )
        print(f"System: {transition_message['emoji_sequence']}")
        print(f"Transition message: {transition_message['natural_language']}")
        
        # Continue conversation in emoji-only mode
        print("\nContinuing in emoji-only mode...")
        
        user_message = dialogue_manager.process_incoming_emoji_message(
            conversation_id,
            "📆🔍",
            "user-4"
        )
        print(f"User: {user_message['emoji_sequence']}")
        print(f"Interpretation: {user_message['natural_language']}")
        
        system_response = dialogue_manager.generate_emoji_response(
            conversation_id,
            "Here's the project timeline. We have milestones in March, June, and September, with final delivery in December."
        )
        print(f"System: {system_response['emoji_sequence']}")
        print(f"Intent: {system_response['original_intent']}")
        
        # Transition back to text mode
        print("\nSystem suggests switching back to text mode for complex explanation...")
        transition_message = dialogue_manager.generate_mode_transition(
            conversation_id,
            CommunicationMode.TEXT_ONLY
        )
        print(f"System: {transition_message['emoji_sequence']}")
        print(f"Transition message: {transition_message['natural_language']}")
        
        # Continue in text mode
        context = dialogue_manager.get_conversation(conversation_id)
        print(f"\nCurrent mode: {context.communication_mode.value}")
    else:
        print("Mode transition not detected or unsuccessful.")


def demonstrate_feedback_mechanisms() -> None:
    """Demonstrate emoji-based feedback mechanisms."""
    print("\n=== Emoji-Based Feedback Mechanisms ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create a new conversation
    conversation_id = "demo-5"
    context = dialogue_manager.create_conversation(conversation_id)
    
    print("Demonstrating various feedback types...\n")
    
    # Display available feedback types and their emoji representations
    for feedback_type in FeedbackType:
        feedback_emoji = dialogue_manager._get_feedback_emoji(feedback_type)
        feedback_text = dialogue_manager._get_feedback_text(feedback_type)
        print(f"{feedback_type.value}: {feedback_emoji} - {feedback_text}")
    
    print("\nSimulating a conversation with feedback...")
    
    # System provides information
    system_info = dialogue_manager.generate_emoji_response(
        conversation_id,
        "The team has completed 60% of the planned work for this sprint."
    )
    print(f"System: {system_info['emoji_sequence']}")
    
    # User sends confirmation feedback
    user_feedback = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👍",
        "user-5"
    )
    print(f"User: {user_feedback['emoji_sequence']}")
    print(f"Detected feedback: {user_feedback['feedback_type']}")
    
    # System provides more information
    system_info = dialogue_manager.generate_emoji_response(
        conversation_id,
        "We have identified a potential risk with the API integration that might delay the release."
    )
    print(f"System: {system_info['emoji_sequence']}")
    
    # User sends question/confusion feedback
    user_feedback = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "🤔❓",
        "user-5"
    )
    print(f"User: {user_feedback['emoji_sequence']}")
    print(f"Detected feedback: {user_feedback['feedback_type']}")
    
    # System provides clarification with emphasis
    system_response = dialogue_manager.generate_emoji_response(
        conversation_id,
        "The API integration might be delayed because the third-party provider changed their authentication mechanism.",
        include_feedback=True
    )
    print(f"System: {system_response['emoji_sequence']}")
    
    # User sends disagreement feedback
    user_feedback = dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👎❌",
        "user-5"
    )
    print(f"User: {user_feedback['emoji_sequence']}")
    print(f"Detected feedback: {user_feedback['feedback_type']}")
    
    # System acknowledges disagreement
    system_response = dialogue_manager.provide_feedback(
        conversation_id,
        FeedbackType.CONFIRMATION
    )
    print(f"System: {system_response['emoji_sequence']}")
    print(f"Feedback message: {system_response['natural_language']}")


def demonstrate_conversation_history() -> None:
    """Demonstrate storing conversation history with translations."""
    print("\n=== Conversation History with Parallel Translations ===\n")
    
    # Initialize the system
    translation_engine = EmojiTranslationEngine()
    grammar_system = EmojiGrammarSystem(translation_engine)
    dialogue_manager = EmojiDialogueManager(translation_engine, grammar_system)
    
    # Create and populate a conversation
    conversation_id = "demo-6"
    context = dialogue_manager.create_conversation(conversation_id)
    
    print("Creating a sample conversation...")
    
    # User greeting
    dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👋😊",
        "user-6"
    )
    
    # System response
    dialogue_manager.generate_emoji_response(
        conversation_id,
        "Hello! How can I help you today?"
    )
    
    # User question
    dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "📊📈❓",
        "user-6"
    )
    
    # System answer
    dialogue_manager.generate_emoji_response(
        conversation_id,
        "The latest metrics show a 15% increase in user engagement and a 7% increase in conversion rate."
    )
    
    # User follow-up
    dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "🔍🤔",
        "user-6"
    )
    
    # System elaboration
    dialogue_manager.generate_emoji_response(
        conversation_id,
        "Looking deeper into the data, we see that the biggest improvements came from mobile users and new visitors."
    )
    
    # User confirmation
    dialogue_manager.process_incoming_emoji_message(
        conversation_id,
        "👍✅",
        "user-6"
    )
    
    # Get conversation summary with translations
    summary = dialogue_manager.get_conversation_summary(conversation_id, include_translations=True)
    
    print("Complete conversation history with parallel translations:")
    
    # Display conversation as parallel emoji and text
    for i in range(len(summary['emoji_sequence'])):
        emoji_message = summary['emoji_sequence'][i]
        text_message = summary['translations'][i] if i < len(summary['translations']) else None
        
        sender = "User" if emoji_message['sender_id'].startswith('user') else "System"
        print(f"\n{sender}: {emoji_message['emoji']}")
        
        if text_message:
            print(f"Translation: {text_message['text']}")
        else:
            print("Translation not available")
    
    print(f"\nTopics discussed: {', '.join(summary['topics'])}")


def main() -> None:
    """Run all emoji dialogue manager demonstrations."""
    print("="*80)
    print("                EmojiDialogueManager Advanced Demonstrations")
    print("="*80)
    
    demonstrate_multi_turn_conversation()
    demonstrate_ambiguity_handling()
    demonstrate_complexity_adaptation()
    demonstrate_mode_transitions()
    demonstrate_feedback_mechanisms()
    demonstrate_conversation_history()
    
    print("="*80)
    print("                        Demonstrations Complete")
    print("="*80)


if __name__ == "__main__":
    main()