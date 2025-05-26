import emoji
"""
Advanced examples demonstrating the EmojiGrammarSystem component functionality.

This file provides detailed examples of emoji grammar patterns for different
types of communication, showcasing the advanced linguistic capabilities of
the EmojiGrammarSystem.
"""

from emoji_grammar_system import (
    EmojiGrammarSystem,
    SentenceType,
    Tense,
    Quantity,
    EmotionalNuance,
    GrammaticalRole,
    EmojiModifiers
)
from emoji_translation_engine import EmojiTranslationEngine


def demonstrate_basic_grammar_patterns() -> None:
    """Demonstrate basic emoji grammar patterns for different sentence types."""
    print("\n=== Basic Grammar Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Basic sentence patterns
    patterns = [
        {
            "description": "Simple statement (Subject + Predicate)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="I",
                predicate="agree"
            )
        },
        {
            "description": "Statement with object (Subject + Predicate + Object)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="she",
                predicate="read",
                object_="book"
            )
        },
        {
            "description": "Simple question (Interrogative + Subject + Predicate)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.QUESTION,
                subject="you",
                predicate="understand"
            )
        },
        {
            "description": "Question with object (Interrogative + Subject + Predicate + Object)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.QUESTION,
                subject="you",
                predicate="need",
                object_="help"
            )
        },
        {
            "description": "Simple command (Predicate)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="stop"
            )
        },
        {
            "description": "Command with object (Predicate + Object)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="check",
                object_="email"
            )
        },
        {
            "description": "Simple conditional (Subject + Predicate + Conditional + Subject + Predicate)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.CONDITIONAL,
                subject="weather",
                predicate="sunny",
                object_="picnic"
            )
        },
        {
            "description": "Negative statement (Subject + Negative + Predicate)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="approve",
                is_negative=True
            )
        }
    ]
    
    # Display each pattern
    for pattern in patterns:
        if pattern["sentence"]:
            print(f"Pattern: {pattern['description']}")
            print(f"Emoji Sentence: {pattern['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in pattern['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(pattern['sentence'])}")
            print()


def demonstrate_tense_modifiers() -> None:
    """Demonstrate tense modifiers in emoji grammar."""
    print("\n=== Tense Modifiers ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Create the same sentence in different tenses
    base_sentence = "We discuss the project"
    
    tense_examples = [
        {
            "tense": Tense.PAST,
            "description": "Past tense (action occurred in the past)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="discuss",
                object_="project",
                tense=Tense.PAST
            )
        },
        {
            "tense": Tense.PRESENT,
            "description": "Present tense (action occurring now)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="discuss",
                object_="project",
                tense=Tense.PRESENT
            )
        },
        {
            "tense": Tense.FUTURE,
            "description": "Future tense (action will occur in the future)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="discuss",
                object_="project",
                tense=Tense.FUTURE
            )
        },
        {
            "tense": Tense.CONTINUOUS,
            "description": "Continuous tense (ongoing action)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="discuss",
                object_="project",
                tense=Tense.CONTINUOUS
            )
        },
        {
            "tense": Tense.PERFECT,
            "description": "Perfect tense (completed action with relevance to now)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="we",
                predicate="discuss",
                object_="project",
                tense=Tense.PERFECT
            )
        }
    ]
    
    print(f"Base sentence: '{base_sentence}'")
    print(f"Tense markers: {', '.join([f'{tense.value}: {EmojiModifiers.TENSE_MARKERS[tense]}' for tense in Tense])}")
    print()
    
    # Display each tense example
    for example in tense_examples:
        if example["sentence"]:
            print(f"Tense: {example['tense'].value} ({example['description']})")
            print(f"Emoji Sentence: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
            print()


def demonstrate_quantity_modifiers() -> None:
    """Demonstrate quantity modifiers in emoji grammar."""
    print("\n=== Quantity Modifiers ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # We'll create sentences about apples with different quantities
    quantity_examples = [
        {
            "quantity": Quantity.SINGULAR,
            "description": "One apple",
            "text": "There is one apple"
        },
        {
            "quantity": Quantity.PLURAL,
            "description": "Multiple apples",
            "text": "There are several apples"
        },
        {
            "quantity": Quantity.ZERO,
            "description": "No apples",
            "text": "There are no apples"
        },
        {
            "quantity": Quantity.FEW,
            "description": "A few apples",
            "text": "There are a few apples"
        },
        {
            "quantity": Quantity.MANY,
            "description": "Many apples",
            "text": "There are many apples"
        },
        {
            "quantity": Quantity.ALL,
            "description": "All the apples",
            "text": "All of the apples are here"
        }
    ]
    
    print(f"Quantity markers: {', '.join([f'{qty.value}: {EmojiModifiers.QUANTITY_MARKERS[qty]}' for qty in Quantity])}")
    print()
    
    # Since our current implementation doesn't have explicit quantity support in create_emoji_sentence,
    # we'll translate the text directly and then parse the result
    for example in quantity_examples:
        print(f"Quantity: {example['quantity'].value} ({example['description']})")
        print(f"Text: '{example['text']}'")
        
        # Translate to emoji
        engine = EmojiTranslationEngine()
        emoji_sequence = engine.translate_text_to_emoji(example['text'])
        
        # Add quantity marker manually (would be automatic in a full implementation)
        quantity_marker = EmojiModifiers.QUANTITY_MARKERS[example['quantity']]
        augmented_sequence = emoji_sequence + quantity_marker
        
        print(f"Emoji Sequence: {augmented_sequence}")
        
        # Translate back for interpretation
        interpretation = engine.translate_emoji_to_text(augmented_sequence)
        print(f"Interpretation: {interpretation}")
        print()


def demonstrate_relationship_indicators() -> None:
    """Demonstrate relationship indicators in emoji grammar."""
    print("\n=== Relationship Indicators ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Examples of different relationship types
    relationship_examples = [
        {
            "relationship": "possessive",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["possessive"],
            "description": "Possession (belongs to the subject)",
            "text": "This is my book"
        },
        {
            "relationship": "belongs_to",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["belongs_to"],
            "description": "Belonging (connected to but not owned)",
            "text": "The document belongs to the project"
        },
        {
            "relationship": "part_of",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["part_of"],
            "description": "Part-whole relationship",
            "text": "The chapter is part of the book"
        },
        {
            "relationship": "located_at",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["located_at"],
            "description": "Location relationship",
            "text": "The meeting is at the office"
        },
        {
            "relationship": "causes",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["causes"],
            "description": "Causation relationship",
            "text": "The rain caused the flood"
        },
        {
            "relationship": "member_of",
            "marker": EmojiModifiers.RELATIONSHIP_MARKERS["member_of"],
            "description": "Membership relationship",
            "text": "She is a member of the team"
        }
    ]
    
    print(f"Relationship markers: {', '.join([f'{rel}: {marker}' for rel, marker in EmojiModifiers.RELATIONSHIP_MARKERS.items()])}")
    print()
    
    # Create examples for each relationship type
    for example in relationship_examples:
        print(f"Relationship: {example['relationship']} ({example['description']})")
        print(f"Marker: {example['marker']}")
        print(f"Text: '{example['text']}'")
        
        # Translate to emoji
        engine = EmojiTranslationEngine()
        emoji_sequence = engine.translate_text_to_emoji(example['text'])
        
        # For a full implementation, we would insert the relationship marker at the appropriate position
        # Here we'll just append it as a demonstration
        modified_sequence = emoji_sequence + example['marker']
        
        print(f"Emoji Sequence: {modified_sequence}")
        print(f"Base Translation: {engine.translate_emoji_to_text(emoji_sequence)}")
        print(f"With Relationship Marker: {engine.translate_emoji_to_text(modified_sequence)}")
        print()


def demonstrate_question_patterns() -> None:
    """Demonstrate patterns for forming questions in emoji grammar."""
    print("\n=== Question Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Different question types
    question_examples = [
        {
            "description": "Yes/No question",
            "text": "Do you like coffee?",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.QUESTION,
                subject="you",
                predicate="like",
                object_="coffee"
            )
        },
        {
            "description": "Who question",
            "text": "Who sent the message?",
            "custom_sequence": "❓👤📤📝"
        },
        {
            "description": "What question",
            "text": "What happened yesterday?",
            "custom_sequence": "❓📦⏮️"
        },
        {
            "description": "When question",
            "text": "When is the meeting?",
            "custom_sequence": "❓⏰🤝"
        },
        {
            "description": "Where question",
            "text": "Where is the conference?",
            "custom_sequence": "❓📍🏢"
        },
        {
            "description": "Why question",
            "text": "Why did you cancel?",
            "custom_sequence": "❓🔍👤🚫"
        },
        {
            "description": "How question",
            "text": "How does this work?",
            "custom_sequence": "❓🛠️⚙️"
        }
    ]
    
    # Display each question pattern
    for example in question_examples:
        print(f"Question Type: {example['description']}")
        print(f"Example: '{example['text']}'")
        
        if 'sentence' in example and example['sentence']:
            print(f"Generated Emoji: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
        elif 'custom_sequence' in example:
            # For special question types that our implementation doesn't directly support yet
            print(f"Emoji Sequence: {example['custom_sequence']}")
            
            # Parse the custom sequence
            parsed = grammar_system.parse_emoji_sequence(example['custom_sequence'])
            if parsed:
                print(f"Parsed Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in parsed.elements])}")
                print(f"Interpretation: {grammar_system.interpret_emoji_sentence(parsed)}")
            else:
                # Fallback to direct interpretation
                engine = EmojiTranslationEngine()
                print(f"Direct Translation: {engine.translate_emoji_to_text(example['custom_sequence'])}")
        
        print()


def demonstrate_command_patterns() -> None:
    """Demonstrate patterns for forming commands in emoji grammar."""
    print("\n=== Command Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Different command types
    command_examples = [
        {
            "description": "Simple imperative",
            "text": "Stop!",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="stop"
            )
        },
        {
            "description": "Command with object",
            "text": "Check the document.",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="check",
                object_="document"
            )
        },
        {
            "description": "Command with modifier (manner)",
            "text": "Run quickly!",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="run",
                modifiers=["quickly"]
            )
        },
        {
            "description": "Negative command (prohibition)",
            "text": "Don't touch!",
            "custom_sequence": "🚫👆"
        },
        {
            "description": "Conditional command",
            "text": "Call me if you need help.",
            "custom_sequence": "📞👤➡️👤🆘"
        },
        {
            "description": "Polite request",
            "text": "Please send the report when ready.",
            "custom_sequence": "🙏📤📊⏰✅"
        },
        {
            "description": "Urgent command",
            "text": "Evacuate immediately!",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.COMMAND,
                predicate="evacuate",
                emotional_nuance=EmotionalNuance.URGENT
            )
        }
    ]
    
    # Display each command pattern
    for example in command_examples:
        print(f"Command Type: {example['description']}")
        print(f"Example: '{example['text']}'")
        
        if 'sentence' in example and example['sentence']:
            print(f"Generated Emoji: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
        elif 'custom_sequence' in example:
            # For command types that our implementation doesn't directly support yet
            print(f"Emoji Sequence: {example['custom_sequence']}")
            
            # Parse the custom sequence
            parsed = grammar_system.parse_emoji_sequence(example['custom_sequence'])
            if parsed:
                print(f"Parsed Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in parsed.elements])}")
                print(f"Interpretation: {grammar_system.interpret_emoji_sentence(parsed)}")
            else:
                # Fallback to direct interpretation
                engine = EmojiTranslationEngine()
                print(f"Direct Translation: {engine.translate_emoji_to_text(example['custom_sequence'])}")
        
        print()


def demonstrate_conditional_expressions() -> None:
    """Demonstrate patterns for conditional expressions in emoji grammar."""
    print("\n=== Conditional Expression Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Different conditional expression types
    conditional_examples = [
        {
            "description": "Simple if-then",
            "text": "If it rains, I'll stay home.",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.CONDITIONAL,
                subject="rain",
                predicate="fall",
                object_="home"
            )
        },
        {
            "description": "If-then with negative consequence",
            "text": "If you're late, you can't enter.",
            "custom_sequence": "❔👤⏰➡️👤🚫🚪"
        },
        {
            "description": "Unless condition",
            "text": "Let's go unless it rains.",
            "custom_sequence": "👥🚶↪️☔👇"
        },
        {
            "description": "When condition",
            "text": "When the file is ready, I'll send it.",
            "custom_sequence": "⏰📄✅➡️👤📤📄"
        },
        {
            "description": "Either-or condition",
            "text": "Either we go now or we miss the train.",
            "custom_sequence": "👥🚶⏰🔀👥🚫🚂"
        },
        {
            "description": "If and only if",
            "text": "The system works if and only if all tests pass.",
            "custom_sequence": "⚙️✅⬅➡️🧪💯✅"
        }
    ]
    
    # Display each conditional pattern
    for example in conditional_examples:
        print(f"Conditional Type: {example['description']}")
        print(f"Example: '{example['text']}'")
        
        if 'sentence' in example and example['sentence']:
            print(f"Generated Emoji: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
        elif 'custom_sequence' in example:
            # For conditional types that our implementation doesn't directly support yet
            print(f"Emoji Sequence: {example['custom_sequence']}")
            
            # Parse the custom sequence
            parsed = grammar_system.parse_emoji_sequence(example['custom_sequence'])
            if parsed:
                print(f"Parsed Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in parsed.elements])}")
                print(f"Interpretation: {grammar_system.interpret_emoji_sequence(parsed)}")
            else:
                # Fallback to direct interpretation
                engine = EmojiTranslationEngine()
                print(f"Direct Translation: {engine.translate_emoji_to_text(example['custom_sequence'])}")
        
        print()


def demonstrate_negation_patterns() -> None:
    """Demonstrate patterns for negation in emoji grammar."""
    print("\n=== Negation Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Different negation patterns
    negation_examples = [
        {
            "description": "Simple negation of statement",
            "text": "I don't agree.",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="I",
                predicate="agree",
                is_negative=True
            )
        },
        {
            "description": "Negation of statement with object",
            "text": "She doesn't like coffee.",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="she",
                predicate="like",
                object_="coffee",
                is_negative=True
            )
        },
        {
            "description": "Negative question",
            "text": "Don't you understand?",
            "custom_sequence": "❓🚫👤🧠"
        },
        {
            "description": "Negation with exception",
            "text": "Everyone except John is here.",
            "custom_sequence": "👥🚫👨📍"
        },
        {
            "description": "Double negation",
            "text": "I can't not help you.",
            "custom_sequence": "👤🚫🚫🆘👤"
        },
        {
            "description": "Partial negation",
            "text": "The project is not completely finished.",
            "custom_sequence": "📋🚫💯✅"
        }
    ]
    
    # Display each negation pattern
    for example in negation_examples:
        print(f"Negation Type: {example['description']}")
        print(f"Example: '{example['text']}'")
        
        if 'sentence' in example and example['sentence']:
            print(f"Generated Emoji: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
        elif 'custom_sequence' in example:
            # For negation patterns that our implementation doesn't directly support yet
            print(f"Emoji Sequence: {example['custom_sequence']}")
            
            # Parse the custom sequence
            parsed = grammar_system.parse_emoji_sequence(example['custom_sequence'])
            if parsed:
                print(f"Parsed Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in parsed.elements])}")
                print(f"Interpretation: {grammar_system.interpret_emoji_sentence(parsed)}")
            else:
                # Fallback to direct interpretation
                engine = EmojiTranslationEngine()
                print(f"Direct Translation: {engine.translate_emoji_to_text(example['custom_sequence'])}")
        
        print()


def demonstrate_emotional_nuance() -> None:
    """Demonstrate emotional nuance in emoji grammar."""
    print("\n=== Emotional Nuance Patterns ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Base statement to express with different emotional nuances
    base_statement = "The meeting is tomorrow."
    
    emotional_examples = [
        {
            "nuance": EmotionalNuance.NEUTRAL,
            "description": "Neutral (no specific emotional emphasis)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.NEUTRAL
            )
        },
        {
            "nuance": EmotionalNuance.EXCITED,
            "description": "Excited (high energy, enthusiasm)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.EXCITED
            )
        },
        {
            "nuance": EmotionalNuance.SERIOUS,
            "description": "Serious (formal, grave)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.SERIOUS
            )
        },
        {
            "nuance": EmotionalNuance.HUMOROUS,
            "description": "Humorous (funny, not serious)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.HUMOROUS
            )
        },
        {
            "nuance": EmotionalNuance.SARCASTIC,
            "description": "Sarcastic (saying opposite of what's meant)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.SARCASTIC
            )
        },
        {
            "nuance": EmotionalNuance.URGENT,
            "description": "Urgent (requiring immediate attention)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.URGENT
            )
        },
        {
            "nuance": EmotionalNuance.GENTLE,
            "description": "Gentle (soft, kind approach)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.GENTLE
            )
        },
        {
            "nuance": EmotionalNuance.FIRM,
            "description": "Firm (strong, decisive approach)",
            "sentence": grammar_system.create_emoji_sentence(
                SentenceType.STATEMENT,
                subject="meeting",
                predicate="scheduled",
                tense=Tense.FUTURE,
                emotional_nuance=EmotionalNuance.FIRM
            )
        }
    ]
    
    print(f"Base statement: '{base_statement}'")
    print()
    
    # Display each emotional nuance example
    for example in emotional_examples:
        if example["sentence"]:
            print(f"Emotional Nuance: {example['nuance'].value} ({example['description']})")
            print(f"Emoji Sentence: {example['sentence']}")
            print(f"Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in example['sentence'].elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(example['sentence'])}")
            print()


def demonstrate_complex_communication() -> None:
    """Demonstrate complex communication scenarios with emoji grammar."""
    print("\n=== Complex Communication Scenarios ===\n")
    
    # Initialize the system
    grammar_system = EmojiGrammarSystem()
    
    # Complex communication scenarios
    scenarios = [
        {
            "description": "Meeting Invitation",
            "text": "Would you like to join our team meeting tomorrow at 2 PM?",
            "custom_sequence": "❓👤👍🤝👥⏭️🕑"
        },
        {
            "description": "Project Status Update",
            "text": "The development phase is 75% complete, but testing has not started yet.",
            "custom_sequence": "⚙️🔄7️⃣5️⃣✅➕🧪🚫▶️⏳"
        },
        {
            "description": "Technical Troubleshooting",
            "text": "If the system shows an error, try restarting it first.",
            "custom_sequence": "❔⚙️⚠️➡️👤🔄⚙️1️⃣"
        },
        {
            "description": "Task Assignment with Deadline",
            "text": "Please complete the documentation by Friday. It's urgent!",
            "custom_sequence": "🙏👤✅📝⏰📅5️⃣⚠️"
        },
        {
            "description": "Collaborative Decision Making",
            "text": "We have three options: continue as planned, modify the approach, or start over.",
            "custom_sequence": "👥3️⃣🔀1️⃣➡️▶️📋2️⃣➡️🔄📋3️⃣➡️🔄0️⃣"
        },
        {
            "description": "Expressing Conditional Satisfaction",
            "text": "I'll approve the changes if you address all the comments first.",
            "custom_sequence": "👤👍🔄❔👤✅💬💯1️⃣"
        },
        {
            "description": "Nuanced Feedback",
            "text": "Your presentation was good, but could be improved by adding more examples.",
            "custom_sequence": "👤📊👍➕🔍👍➕👤➕🧩🌟"
        }
    ]
    
    # Display each complex scenario
    for scenario in scenarios:
        print(f"Scenario: {scenario['description']}")
        print(f"Message: '{scenario['text']}'")
        print(f"Emoji Sequence: {scenario['custom_sequence']}")
        
        # Parse the custom sequence
        parsed = grammar_system.parse_emoji_sequence(scenario['custom_sequence'])
        if parsed:
            print(f"Parsed Structure: {' '.join([f'{e.emoji}({e.role.value})' for e in parsed.elements])}")
            print(f"Interpretation: {grammar_system.interpret_emoji_sentence(parsed)}")
        else:
            # Fallback to direct interpretation
            engine = EmojiTranslationEngine()
            print(f"Direct Translation: {engine.translate_emoji_to_text(scenario['custom_sequence'])}")
        
        print()


def main() -> None:
    """Run all emoji grammar system demonstrations."""
    print("="*80)
    print("             Advanced EmojiGrammarSystem Demonstrations")
    print("="*80)
    
    demonstrate_basic_grammar_patterns()
    demonstrate_tense_modifiers()
    demonstrate_quantity_modifiers()
    demonstrate_relationship_indicators()
    demonstrate_question_patterns()
    demonstrate_command_patterns()
    demonstrate_conditional_expressions()
    demonstrate_negation_patterns()
    demonstrate_emotional_nuance()
    demonstrate_complex_communication()
    
    print("="*80)
    print("                      Demonstration Complete")
    print("="*80)


if __name__ == "__main__":
    main()