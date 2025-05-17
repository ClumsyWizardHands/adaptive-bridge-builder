"""
Example demonstrating the EmojiTranslationEngine component functionality.

This file provides concrete examples of how to use the EmojiTranslationEngine
for translating between natural language and emoji sequences with various options.
"""

from emoji_translation_engine import (
    EmojiTranslationEngine,
    TranslationMode,
    AmbiguityResolutionStrategy,
    EmojiCategory,
    EmojiEntry
)

def demonstrate_text_to_emoji():
    """Demonstrate text to emoji translation with different modes and contexts."""
    print("\n=== Text to Emoji Translation ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Example texts for translation
    examples = [
        {
            "text": "I'm happy today and excited about our new project!",
            "description": "Simple positive sentiment"
        },
        {
            "text": "I need some time to think about this complex problem.",
            "description": "Cognitive process with abstract concept"
        },
        {
            "text": "This proposal makes me sad, we need to reconsider our approach.",
            "description": "Negative sentiment with action needed"
        },
        {
            "text": "Let's search for innovative solutions to grow our business.",
            "description": "Multiple abstract concepts (search and growth)"
        },
        {
            "text": "We need to wait for the system to process the data before proceeding.",
            "description": "Abstract concept of time and process"
        }
    ]
    
    # Demonstrate different translation modes for each example
    for example in examples:
        text = example["text"]
        print(f"Text: {text}")
        print(f"Description: {example['description']}")
        
        # Translate with different modes
        print(f"  LITERAL:     {engine.translate_text_to_emoji(text, TranslationMode.LITERAL)}")
        print(f"  SEMANTIC:    {engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)}")
        print(f"  EMOTIONAL:   {engine.translate_text_to_emoji(text, TranslationMode.EMOTIONAL)}")
        print(f"  SUMMARIZED:  {engine.translate_text_to_emoji(text, TranslationMode.SUMMARIZED)}")
        print(f"  EXPRESSIVE:  {engine.translate_text_to_emoji(text, TranslationMode.EXPRESSIVE)}")
        print()
    
    # Demonstrate the impact of context on translation
    contexts = [
        ["business", "meeting", "professional"],
        ["personal", "friendship", "casual"],
        ["technical", "development", "programming"]
    ]
    
    text = "I have an idea that could solve our problem."
    print(f"Text with different contexts: {text}")
    
    for context in contexts:
        print(f"  Context: {', '.join(context)}")
        print(f"  Result:  {engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC, context)}")
    
    print()


def demonstrate_emoji_to_text():
    """Demonstrate emoji to text translation with different resolution strategies."""
    print("\n=== Emoji to Text Translation ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Example emoji sequences for translation
    examples = [
        {
            "emojis": "😊👍",
            "description": "Simple positive sentiment"
        },
        {
            "emojis": "🤔💡🔍",
            "description": "Thinking process leading to idea and investigation"
        },
        {
            "emojis": "😢👎🔄",
            "description": "Disappointment and need to try again"
        },
        {
            "emojis": "⏳🔄🌱",
            "description": "Abstract concepts (time, repetition, growth)"
        },
        {
            "emojis": "💡🔍👍⚡",
            "description": "Idea investigation with positive outcome"
        }
    ]
    
    # Demonstrate different resolution strategies for each example
    for example in examples:
        emojis = example["emojis"]
        print(f"Emoji Sequence: {emojis}")
        print(f"Description: {example['description']}")
        
        # Most common interpretation
        result = engine.translate_emoji_to_text(
            emojis, 
            resolution_strategy=AmbiguityResolutionStrategy.MOST_COMMON
        )
        print(f"  MOST_COMMON: {result}")
        
        # Contextual interpretation
        context = ["project", "development", "meeting"]
        result = engine.translate_emoji_to_text(
            emojis, 
            context=context,
            resolution_strategy=AmbiguityResolutionStrategy.CONTEXTUAL
        )
        print(f"  CONTEXTUAL (context: {', '.join(context)}): {result}")
        
        # Multiple interpretations
        result = engine.translate_emoji_to_text(
            emojis, 
            resolution_strategy=AmbiguityResolutionStrategy.MULTIPLE
        )
        print(f"  MULTIPLE: {result}")
        
        # Clarification needed
        result = engine.translate_emoji_to_text(
            emojis, 
            resolution_strategy=AmbiguityResolutionStrategy.CLARIFY
        )
        print(f"  CLARIFY: {result['translation']}")
        if result['clarification_needed']:
            print(f"    Ambiguous emojis: {result['ambiguities']}")
            for emoji, options in result['options'].items():
                print(f"      {emoji}: {options}")
        
        # Confidence-based interpretation
        result = engine.translate_emoji_to_text(
            emojis, 
            resolution_strategy=AmbiguityResolutionStrategy.CONFIDENCE
        )
        print(f"  CONFIDENCE: {result['translation']} (confidence: {result['confidence']:.2f})")
        if result['alternatives']:
            print("    Alternatives:")
            for alt, conf in result['alternatives']:
                print(f"      {alt} (confidence: {conf:.2f})")
        
        print()


def demonstrate_abstract_concept_handling():
    """Demonstrate how the engine handles abstract concepts."""
    print("\n=== Abstract Concept Handling ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # List of abstract concepts to demonstrate
    abstract_concepts = [
        "time", 
        "idea", 
        "growth", 
        "search", 
        "repetition",
        "progress",
        "complexity",
        "balance",
        "connection",
        "transformation"
    ]
    
    # Show available emojis for each abstract concept
    for concept in abstract_concepts:
        emojis = engine.get_emoji_for_abstract_concept(concept)
        if emojis:
            print(f"Abstract concept: {concept}")
            print(f"  Represented by: {' '.join(emojis)}")
        else:
            print(f"Abstract concept: {concept} (no specific emojis available)")
            
        # Demonstrate translating a sentence containing this abstract concept
        text = f"We need to consider the {concept} of our system."
        emoji_translation = engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        print(f"  Example text: '{text}'")
        print(f"  Translated to: '{emoji_translation}'")
        print()
    
    # Demonstrate handling complex sentences with multiple abstract concepts
    complex_examples = [
        "We need to balance growth with stability over time.",
        "The search for innovative ideas requires transformation of our thinking.",
        "Establishing connections between concepts helps with complexity management.",
        "Progress often requires repetition and incremental improvements."
    ]
    
    print("\nComplex sentences with multiple abstract concepts:")
    for text in complex_examples:
        emoji_translation = engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        print(f"  Text: '{text}'")
        print(f"  Emojis: '{emoji_translation}'")
        # Translate back to show how well the meaning is preserved
        back_translation = engine.translate_emoji_to_text(emoji_translation)
        print(f"  Back-translation: '{back_translation}'")
        print(f"  Meaning preservation: {'High' if len(back_translation.split()) >= len(text.split())/2 else 'Medium' if len(back_translation.split()) >= len(text.split())/3 else 'Low'}")
        print()


def demonstrate_ambiguity_resolution():
    """Demonstrate ambiguity resolution through user feedback."""
    print("\n=== Ambiguity Resolution ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Example emoji with multiple possible meanings
    emoji = "💡"
    
    # Before resolution
    print(f"Emoji: {emoji}")
    
    # Initial translations in different contexts
    contexts = [
        ["project", "innovation"],
        ["electricity", "power"],
        ["education", "learning"]
    ]
    
    print("Before resolution:")
    for context in contexts:
        result = engine.translate_emoji_to_text(emoji, context=context)
        print(f"  Context [{', '.join(context)}]: {result}")
    
    # Simulate user selecting a specific meaning in a specific context
    selected_meaning = "insight"
    resolution_context = ["brainstorming", "creativity"]
    
    print(f"\nUser selects meaning '{selected_meaning}' in context [{', '.join(resolution_context)}]")
    engine.resolve_ambiguity(emoji, selected_meaning, resolution_context)
    
    # After resolution
    print("\nAfter resolution:")
    for context in contexts + [resolution_context]:
        result = engine.translate_emoji_to_text(emoji, context=context)
        print(f"  Context [{', '.join(context)}]: {result}")
    
    print("\nDemonstrating preference learning over time:")
    
    # Simulate multiple users selecting different meanings in different contexts
    examples = [
        {"emoji": "🔍", "meaning": "investigate", "context": ["research", "analysis"]},
        {"emoji": "⏳", "meaning": "patience", "context": ["waiting", "development"]},
        {"emoji": "🌱", "meaning": "new project", "context": ["startup", "initiative"]},
        {"emoji": "🔄", "meaning": "repeat process", "context": ["iteration", "development"]}
    ]
    
    for example in examples:
        print(f"\nEmoji: {example['emoji']}")
        
        # Before selection
        before = engine.translate_emoji_to_text(
            example['emoji'], 
            context=example['context']
        )
        print(f"  Before selection (context: {', '.join(example['context'])}): {before}")
        
        # Apply user selection
        engine.resolve_ambiguity(
            example['emoji'], 
            example['meaning'], 
            example['context']
        )
        
        # After selection
        after = engine.translate_emoji_to_text(
            example['emoji'], 
            context=example['context']
        )
        print(f"  After selection (context: {', '.join(example['context'])}): {after}")
        
        # Check if learning occurred
        if before != after and example['meaning'] in after:
            print("  ✓ Successfully learned user preference")
        else:
            print("  ✕ Learning not visible in this case")


def demonstrate_dictionary_customization():
    """Demonstrate customizing the emoji dictionary."""
    print("\n=== Dictionary Customization ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Example: Adding a custom emoji for a technical concept
    print("Adding custom emoji entries for technical concepts")
    
    # Create a custom emoji entry for "API"
    api_emoji = EmojiEntry(
        emoji="🔌",
        name="electric plug",
        keywords=["API", "interface", "connection", "integration"],
        categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
        sentiment_score=0.2,
        common_contexts=["development", "programming", "system", "integration"],
        related_emojis=["🖥️", "📡", "🔄"],
        abstract_concepts=["connection", "integration", "communication"]
    )
    
    # Create a custom emoji entry for "Database"
    db_emoji = EmojiEntry(
        emoji="🗄️",
        name="file cabinet",
        keywords=["database", "storage", "data", "repository"],
        categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
        sentiment_score=0.1,
        common_contexts=["development", "programming", "system", "storage"],
        related_emojis=["💾", "📊", "📁"],
        abstract_concepts=["storage", "organization", "persistence"]
    )
    
    # Create a custom emoji entry for "Bug"
    bug_emoji = EmojiEntry(
        emoji="🐛",
        name="bug",
        keywords=["bug", "error", "issue", "defect"],
        categories=[EmojiCategory.ANIMAL, EmojiCategory.ABSTRACT],
        sentiment_score=-0.5,
        common_contexts=["development", "programming", "testing", "debugging"],
        related_emojis=["🔍", "⚠️", "🔧"],
        abstract_concepts=["problem", "error", "flaw"]
    )
    
    # Add the custom entries to the dictionary
    engine.add_emoji_to_dictionary(api_emoji)
    engine.add_emoji_to_dictionary(db_emoji)
    engine.add_emoji_to_dictionary(bug_emoji)
    
    # Test the custom entries
    technical_texts = [
        "We need to integrate with the API",
        "The database needs optimization",
        "I found a bug in the code",
        "We need to connect our API to the database and fix any bugs"
    ]
    
    print("\nTexts with technical terms:")
    for text in technical_texts:
        emoji_translation = engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        print(f"  Text: '{text}'")
        print(f"  Emojis: '{emoji_translation}'")
        print()
    
    # Demonstrate saving and loading the dictionary
    print("Saving and loading the custom dictionary...")
    
    # In a real implementation, this would save to a file
    # engine.save_dictionary("custom_emoji_dictionary.json")
    # new_engine = EmojiTranslationEngine("custom_emoji_dictionary.json")
    
    # For example purposes, we'll just note that this would persist the dictionary
    print("  (In a real implementation, the dictionary would be saved to a file)")
    print("  (and could be loaded by a new instance of the engine)")


def demonstrate_real_communication_scenarios():
    """Demonstrate using the engine in real communication scenarios."""
    print("\n=== Real Communication Scenarios ===\n")
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Scenario 1: Meeting summary
    print("Scenario 1: Converting a meeting summary to emoji for quick visual reference")
    meeting_summary = """
    Project kickoff meeting went well. Team is excited about the new approach.
    We identified several challenges but also came up with innovative solutions.
    Next steps: research technical options, develop prototype, and test with users.
    Timeline is tight but achievable if we prioritize correctly.
    """
    
    print(f"Original text:\n{meeting_summary}")
    
    # Convert to emoji summary
    emoji_summary = engine.translate_text_to_emoji(meeting_summary, TranslationMode.SUMMARIZED)
    print(f"\nEmoji summary: {emoji_summary}")
    
    # Scenario 2: Status update
    print("\nScenario 2: Translating status emojis in a dashboard")
    status_emojis = {
        "task_status": {
            "not_started": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "blocked": "⛔"
        },
        "project_health": {
            "good": "🟢",
            "warning": "🟡",
            "critical": "🔴"
        }
    }
    
    print("Status dashboard with emojis:")
    dashboard = f"""
    Project Alpha: {status_emojis['project_health']['good']}
    - Feature 1: {status_emojis['task_status']['completed']}
    - Feature 2: {status_emojis['task_status']['in_progress']}
    - Feature 3: {status_emojis['task_status']['not_started']}
    
    Project Beta: {status_emojis['project_health']['warning']}
    - Database migration: {status_emojis['task_status']['in_progress']}
    - API integration: {status_emojis['task_status']['blocked']}
    - User interface: {status_emojis['task_status']['completed']}
    """
    
    print(dashboard)
    
    # Translate dashboard to text for accessibility
    text_lines = []
    for line in dashboard.strip().split('\n'):
        for emoji, text in status_emojis['task_status'].items():
            if status_emojis['task_status'][emoji] in line:
                line = line.replace(status_emojis['task_status'][emoji], f"[{emoji}]")
        
        for emoji, text in status_emojis['project_health'].items():
            if status_emojis['project_health'][emoji] in line:
                line = line.replace(status_emojis['project_health'][emoji], f"[{emoji}]")
        
        text_lines.append(line)
    
    print("\nAccessible text version:")
    print('\n'.join(text_lines))
    
    # Scenario 3: Cross-cultural communication
    print("\nScenario 3: Using emoji as a universal language for cross-cultural communication")
    
    messages = [
        "I'm excited to work with your team on this project.",
        "Could you please provide more details about the requirements?",
        "We've encountered a technical issue that will delay the delivery.",
        "Great progress! The prototype is working well."
    ]
    
    print("Original messages and their emoji translations:")
    for message in messages:
        emoji_message = engine.translate_text_to_emoji(message, TranslationMode.SEMANTIC)
        print(f"  Original: {message}")
        print(f"  Emoji:    {emoji_message}")
        print(f"  Back to text: {engine.translate_emoji_to_text(emoji_message)}")
        print()


def main():
    """Main function to run all demonstrations."""
    print("="*80)
    print("                  EmojiTranslationEngine Demonstration")
    print("="*80)
    
    demonstrate_text_to_emoji()
    demonstrate_emoji_to_text()
    demonstrate_abstract_concept_handling()
    demonstrate_ambiguity_resolution()
    demonstrate_dictionary_customization()
    demonstrate_real_communication_scenarios()
    
    print("="*80)
    print("                  Demonstration Complete")
    print("="*80)


if __name__ == "__main__":
    main()
