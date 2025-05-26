import emoji
"""
Demonstration of the Domain-Specific Emoji Sets.

This script provides examples of how to use the domain-specific emoji sets
for specialized communication contexts.
"""

from domain_specific_emoji_sets import (
    TechnicalSupportEmojiSet,
    ProjectManagementEmojiSet,
    EducationalEmojiSet,
    FinancialEmojiSet
)

def demonstrate_domain_specific_emoji_sets() -> None:
    """Demonstrate the usage of domain-specific emoji sets."""
    print("=" * 80)
    print("Domain-Specific Emoji Sets Demonstration")
    print("=" * 80)
    
    # Technical Support Domain
    demonstrate_technical_support()
    
    # Project Management Domain
    demonstrate_project_management()
    
    # Educational Domain
    demonstrate_educational()
    
    # Financial Domain
    demonstrate_financial()

def demonstrate_technical_support() -> None:
    """Demonstrate the technical support emoji set."""
    print("\n\n1. TECHNICAL SUPPORT EMOJI SET")
    print("-" * 40)
    
    tech_support = TechnicalSupportEmojiSet()
    
    # Show emoji mappings
    print("\nTechnical Support Emoji Vocabulary:")
    for emoji, mapping in tech_support.emoji_mappings.items():
        print(f"{emoji} - {mapping.concept}: {mapping.description}")
        print(f"  Usage examples: {', '.join(mapping.usage_examples)}")
    
    # Show common sequences
    print("\nCommon Technical Support Sequences:")
    for sequence in tech_support.common_sequences:
        print(f"{sequence.sequence} - {sequence.meaning}")
        print(f"  Context: {sequence.context}")
        print(f"  Possible responses: {', '.join(sequence.possible_responses)}")
    
    # Demonstrate usage
    print("\nExample Usage Scenarios:")
    
    # Scenario 1: Server issue
    print("\nScenario 1: Reporting a critical server issue")
    user_message = "🛑🖥️⚡"
    meaning = tech_support.interpret_sequence(user_message)
    responses = tech_support.suggest_response(user_message)
    
    print(f"User: {user_message}")
    print(f"Meaning: {meaning}")
    print(f"Suggested responses: {', '.join(responses)}")
    
    # Scenario 2: Looking up emoji for a concept
    concept = "Database"
    emoji = tech_support.get_emoji_for_concept(concept)
    print(f"\nEmoji for '{concept}': {emoji}")

def demonstrate_project_management() -> None:
    """Demonstrate the project management emoji set."""
    print("\n\n2. PROJECT MANAGEMENT EMOJI SET")
    print("-" * 40)
    
    project_mgmt = ProjectManagementEmojiSet()
    
    # Show high priority task sequence
    print("\nExample: High Priority Task Assignment")
    sequence = "🆕🔥👤"
    meaning = project_mgmt.interpret_sequence(sequence)
    responses = project_mgmt.suggest_response(sequence)
    
    print(f"PM: {sequence}")
    print(f"Meaning: {meaning}")
    print(f"Team member responses: {', '.join(responses)}")
    
    # Show deadline approaching sequence
    print("\nExample: Urgent deadline notification")
    sequence = "⏰🔥⚠️"
    meaning = project_mgmt.interpret_sequence(sequence)
    responses = project_mgmt.suggest_response(sequence)
    
    print(f"PM: {sequence}")
    print(f"Meaning: {meaning}")
    print(f"Team member responses: {', '.join(responses)}")
    
    # Show milestone completion
    print("\nExample: Milestone completion celebration")
    sequence = "✅📊🎉"
    meaning = project_mgmt.interpret_sequence(sequence)
    responses = project_mgmt.suggest_response(sequence)
    
    print(f"PM: {sequence}")
    print(f"Meaning: {meaning}")
    print(f"Team member responses: {', '.join(responses)}")

def demonstrate_educational() -> None:
    """Demonstrate the educational emoji set."""
    print("\n\n3. EDUCATIONAL EMOJI SET")
    print("-" * 40)
    
    educational = EducationalEmojiSet()
    
    # Show emoji for different knowledge areas
    print("\nKnowledge Area Emojis:")
    knowledge_areas = ["Mathematics", "Science", "Literature", "Computer Science"]
    for area in knowledge_areas:
        emoji = educational.get_emoji_for_concept(area)
        print(f"{area}: {emoji}")
    
    # Show study sequence
    print("\nExample: Intensive study session coordination")
    sequence = "📚🧠⏰"
    meaning = educational.interpret_sequence(sequence)
    responses = educational.suggest_response(sequence)
    
    print(f"Teacher: {sequence}")
    print(f"Meaning: {meaning}")
    print(f"Student responses: {', '.join(responses)}")
    
    # Show science project sequence
    print("\nExample: Science experiment instructions")
    sequence = "🧪🔬📊"
    meaning = educational.interpret_sequence(sequence)
    responses = educational.suggest_response(sequence)
    
    print(f"Teacher: {sequence}")
    print(f"Meaning: {meaning}")
    print(f"Student responses: {', '.join(responses)}")

def demonstrate_financial() -> None:
    """Demonstrate the financial emoji set."""
    print("\n\n4. FINANCIAL EMOJI SET")
    print("-" * 40)
    
    financial = FinancialEmojiSet()
    
    # Show currency emojis
    print("\nCurrency Emojis:")
    currencies = ["Dollar", "Euro", "Pound", "Yen"]
    for currency in currencies:
        emoji = financial.get_emoji_for_concept(currency)
        print(f"{currency}: {emoji}")
    
    # Show financial concepts
    print("\nFinancial Concept Emojis:")
    concepts = ["Growth/Increase", "Decline/Decrease", "Balance/Budget", "Bank/Financial Institution"]
    for concept in concepts:
        emoji = financial.get_emoji_for_concept(concept)
        print(f"{concept}: {emoji}")
    
    # Example transaction communication
    print("\nExample Financial Communications:")
    message = "💰💵✅"
    print(f"Accountant: {message}")
    print("Meaning: Cash payment received and confirmed")
    
    message = "📊📈🎉"
    print(f"\nFinancial Advisor: {message}")
    print("Meaning: Quarterly report shows strong growth")
    
    message = "💼💰⚖️"
    print(f"\nBusiness Manager: {message}")
    print("Meaning: Investment funds balanced properly")

if __name__ == "__main__":
    demonstrate_domain_specific_emoji_sets()