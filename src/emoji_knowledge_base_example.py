"""
Example demonstrating the EmojiKnowledgeBase component functionality.

This file provides concrete examples of using the EmojiKnowledgeBase
for domain-specific emoji mappings, cultural variations, and complex concepts.
"""

from datetime import datetime
from emoji_knowledge_base import (
    EmojiKnowledgeBase,
    EmojiDomain,
    CulturalContext,
    EmojiCategory,
    SentimentValue,
    FamiliarityLevel
)


def demonstrate_concept_mapping():
    """Demonstrate concept-to-emoji mapping across different domains and cultures."""
    print("\n=== Concept-to-Emoji Mapping ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Concepts to demonstrate
    concepts = [
        "success",
        "code review",
        "meeting",
        "deadline",
        "bug"
    ]
    
    # Domains to demonstrate
    domains = [
        EmojiDomain.GENERAL,
        EmojiDomain.TECHNICAL,
        EmojiDomain.BUSINESS
    ]
    
    # Show mappings across domains
    print("Concept mappings across different domains:\n")
    for concept in concepts:
        print(f"Concept: {concept}")
        for domain in domains:
            emojis = kb.find_emojis_for_concept(concept, domain=domain)
            
            primary = ", ".join(emojis["primary"]) if emojis["primary"] else "None"
            alternatives = ", ".join(emojis["alternatives"]) if emojis["alternatives"] else "None"
            
            print(f"  {domain.value}: Primary: {primary} | Alternatives: {alternatives}")
        
        # Show combination patterns if available
        patterns = kb.get_combination_patterns_for_concept(concept)
        if patterns:
            print("  Combination patterns:")
            for pattern in patterns:
                emoji_sequence = " ".join(pattern.emoji_sequence)
                print(f"    {emoji_sequence} → {pattern.resulting_concept} ({pattern.primary_domain.value})")
        
        print()


def demonstrate_cultural_variations():
    """Demonstrate cultural variations in emoji interpretation."""
    print("\n=== Cultural Variations in Emoji Interpretation ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Add some cultural variations for demonstration
    kb.add_emoji(
        emoji="👍",
        unicode_representation="U+1F44D",
        short_name=":thumbs_up:",
        description="Thumbs up sign",
        keywords=["approval", "like", "positive", "yes"],
        category=EmojiCategory.GESTURE,
        primary_meaning="Approval or agreement",
        cultural_meanings={
            CulturalContext.GLOBAL: "General approval or agreement",
            CulturalContext.WESTERN: "Positive affirmation, 'good job'",
            CulturalContext.MIDDLE_EASTERN: "Potentially offensive gesture in some countries",
            CulturalContext.EASTERN_ASIAN: "Number 'one' in Japan, general agreement in China"
        },
        frequency_score=0.98,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    kb.add_emoji(
        emoji="🙏",
        unicode_representation="U+1F64F",
        short_name=":folded_hands:",
        description="Person with folded hands",
        keywords=["please", "thank you", "pray", "request", "grateful"],
        category=EmojiCategory.GESTURE,
        primary_meaning="Thank you or please",
        cultural_meanings={
            CulturalContext.GLOBAL: "Gratitude or pleading",
            CulturalContext.WESTERN: "Prayer or spirituality",
            CulturalContext.EASTERN_ASIAN: "Thank you or please",
            CulturalContext.SOUTH_ASIAN: "Greeting (namaste)"
        },
        frequency_score=0.94,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    # Display cultural variations
    emojis_to_demonstrate = ["👍", "🙏", "👌", "👋"]
    cultural_contexts = [
        CulturalContext.GLOBAL,
        CulturalContext.WESTERN,
        CulturalContext.EASTERN_ASIAN,
        CulturalContext.MIDDLE_EASTERN,
        CulturalContext.SOUTH_ASIAN
    ]
    
    print("Cultural variations in emoji interpretation:\n")
    for emoji in emojis_to_demonstrate:
        emoji_metadata = kb.get_emoji(emoji)
        if not emoji_metadata:
            print(f"{emoji}: No information available")
            continue
            
        print(f"{emoji} ({emoji_metadata.short_name}): {emoji_metadata.primary_meaning}")
        
        for context in cultural_contexts:
            meaning = emoji_metadata.cultural_meanings.get(context)
            if meaning:
                print(f"  {context.value}: {meaning}")
        
        print()

def demonstrate_domain_specific_meanings():
    """Demonstrate domain-specific emoji meanings."""
    print("\n=== Domain-Specific Emoji Meanings ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Add some domain-specific meanings for demonstration
    kb.add_emoji(
        emoji="🔥",
        unicode_representation="U+1F525",
        short_name=":fire:",
        description="Fire",
        keywords=["hot", "flame", "burn", "trending"],
        category=EmojiCategory.SYMBOL,
        primary_meaning="Fire or flame",
        domain_meanings={
            EmojiDomain.GENERAL: "Fire, flame, or heat",
            EmojiDomain.SOCIAL: "Trending, popular, or exciting",
            EmojiDomain.TECHNICAL: "High CPU usage or performance issue",
            EmojiDomain.BUSINESS: "Hot deal or urgent priority"
        },
        frequency_score=0.91,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    kb.add_emoji(
        emoji="💯",
        unicode_representation="U+1F4AF",
        short_name=":hundred_points:",
        description="Hundred points symbol",
        keywords=["100", "perfect", "score", "complete"],
        category=EmojiCategory.SYMBOL,
        primary_meaning="Perfect score or 100%",
        domain_meanings={
            EmojiDomain.GENERAL: "Perfect or excellent",
            EmojiDomain.EDUCATION: "Perfect score on test or assignment",
            EmojiDomain.SOCIAL: "Strong agreement or 'keeping it real'",
            EmojiDomain.BUSINESS: "100% completion or full achievement"
        },
        frequency_score=0.89,
        familiarity_level=FamiliarityLevel.COMMON
    )
    
    # Display domain-specific meanings
    emojis_to_demonstrate = ["🔥", "💯", "👨‍💻", "🚀"]
    domains_to_show = [
        EmojiDomain.GENERAL,
        EmojiDomain.TECHNICAL,
        EmojiDomain.BUSINESS,
        EmojiDomain.SOCIAL,
        EmojiDomain.EDUCATION
    ]
    
    print("Domain-specific emoji meanings:\n")
    for emoji in emojis_to_demonstrate:
        emoji_metadata = kb.get_emoji(emoji)
        if not emoji_metadata:
            print(f"{emoji}: No information available")
            continue
            
        print(f"{emoji} ({emoji_metadata.short_name}): {emoji_metadata.primary_meaning}")
        
        for domain in domains_to_show:
            meaning = emoji_metadata.domain_meanings.get(domain)
            if meaning:
                print(f"  {domain.value}: {meaning}")
        
        print()

def demonstrate_combination_patterns():
    """Demonstrate emoji combination patterns for complex concepts."""
    print("\n=== Emoji Combination Patterns ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Add more combination patterns for demonstration
    kb.add_combination_pattern(
        name="feature_request",
        description="Request for a new feature",
        emoji_sequence=["✨", "💡", "🔍"],
        resulting_concept="feature request or idea",
        primary_domain=EmojiDomain.TECHNICAL,
        applicable_domains=[EmojiDomain.TECHNICAL, EmojiDomain.BUSINESS],
        frequency_score=0.72
    )
    
    kb.add_combination_pattern(
        name="critical_deadline",
        description="Approaching critical deadline",
        emoji_sequence=["⏰", "🔴", "📅"],
        resulting_concept="urgent deadline approaching",
        primary_domain=EmojiDomain.BUSINESS,
        applicable_domains=[EmojiDomain.BUSINESS, EmojiDomain.TECHNICAL, EmojiDomain.EDUCATION],
        frequency_score=0.81
    )
    
    kb.add_combination_pattern(
        name="security_vulnerability",
        description="Security vulnerability or breach",
        emoji_sequence=["🔒", "⚠️", "🐛"],
        resulting_concept="security vulnerability",
        primary_domain=EmojiDomain.TECHNICAL,
        applicable_domains=[EmojiDomain.TECHNICAL, EmojiDomain.BUSINESS],
        frequency_score=0.65
    )
    
    kb.add_combination_pattern(
        name="deployment_successful",
        description="Successful software deployment",
        emoji_sequence=["🚀", "✅", "🎉"],
        resulting_concept="successful deployment or launch",
        primary_domain=EmojiDomain.TECHNICAL,
        applicable_domains=[EmojiDomain.TECHNICAL],
        frequency_score=0.88
    )
    
    # Display combination patterns by domain
    domains_to_demonstrate = [
        EmojiDomain.TECHNICAL,
        EmojiDomain.BUSINESS,
        EmojiDomain.EDUCATION
    ]
    
    print("Emoji combination patterns by domain:\n")
    for domain in domains_to_demonstrate:
        print(f"Domain: {domain.value}")
        
        # Find patterns for the domain
        domain_patterns = []
        for pattern_id, pattern in kb.patterns.items():
            if domain in pattern.applicable_domains:
                domain_patterns.append(pattern)
        
        # Sort by frequency score
        domain_patterns.sort(key=lambda p: p.frequency_score, reverse=True)
        
        if domain_patterns:
            for pattern in domain_patterns:
                emoji_sequence = " ".join(pattern.emoji_sequence)
                frequency = f"{pattern.frequency_score:.2f}" if pattern.frequency_score > 0 else "N/A"
                print(f"  {emoji_sequence} → {pattern.resulting_concept} (Frequency: {frequency})")
        else:
            print("  No patterns found")
        
        print()

def demonstrate_searching_and_querying():
    """Demonstrate searching and querying the knowledge base."""
    print("\n=== Searching and Querying the Knowledge Base ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Add more data for demonstration
    kb.add_emoji(
        emoji="📊",
        unicode_representation="U+1F4CA",
        short_name=":bar_chart:",
        description="Bar chart",
        keywords=["stats", "metrics", "data", "analytics", "graph"],
        category=EmojiCategory.OBJECT,
        primary_meaning="Statistical data or analytics",
        domain_meanings={
            EmojiDomain.GENERAL: "Statistics or data visualization",
            EmojiDomain.BUSINESS: "Business metrics or performance indicators",
            EmojiDomain.TECHNICAL: "Performance analytics or monitoring"
        },
        frequency_score=0.78,
        familiarity_level=FamiliarityLevel.COMMON
    )
    
    kb.add_concept_mapping(
        concept="analytics",
        description="Analysis of data or metrics",
        primary_emoji={
            EmojiDomain.GENERAL: "📊",
            EmojiDomain.BUSINESS: "📈",
            EmojiDomain.TECHNICAL: "📊"
        },
        alternative_emojis={
            EmojiDomain.GENERAL: ["📈", "📉", "🔍"],
            EmojiDomain.BUSINESS: ["📊", "📉", "💹"],
            EmojiDomain.TECHNICAL: ["📈", "🔍", "⚙️"]
        }
    )
    
    # Demonstrate keyword search
    print("Searching emojis by keyword:\n")
    keywords = ["data", "code", "success", "meeting", "happy"]
    
    for keyword in keywords:
        results = kb.search_emoji_by_keyword(keyword)
        print(f"Keyword: '{keyword}'")
        if results:
            print(f"  Results: {', '.join(results)}")
        else:
            print("  No results found")
        print()
    
    # Demonstrate finding concepts for emoji
    print("Finding concepts represented by emoji:\n")
    emojis = ["🎉", "👀", "📊", "🚀"]
    
    for emoji in emojis:
        concepts = kb.find_concept_for_emoji(emoji)
        print(f"Emoji: {emoji}")
        if concepts:
            print(f"  Represents concepts: {', '.join(concepts)}")
        else:
            print("  No concepts found")
        print()
    
    # Demonstrate finding emoji sequences for concepts
    print("Finding emoji sequences that represent complex concepts:\n")
    emoji_sequences = [
        ["🐛", "🔨", "✅"],
        ["📋", "✅", "🎉"],
        ["🚀", "✅", "🎉"],
        ["🔒", "⚠️", "🐛"]
    ]
    
    for sequence in emoji_sequences:
        sequence_str = " ".join(sequence)
        concepts = kb.find_concepts_for_emoji_sequence(sequence)
        print(f"Sequence: {sequence_str}")
        if concepts:
            print(f"  Represents: {', '.join(concepts)}")
        else:
            print("  No concepts found")
        print()

def demonstrate_emoji_by_sentiment_and_familiarity():
    """Demonstrate retrieving emojis by sentiment and familiarity."""
    print("\n=== Emojis by Sentiment and Familiarity ===\n")
    
    # Initialize the knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    
    # Add more data with sentiment values
    kb.add_emoji(
        emoji="😀",
        unicode_representation="U+1F600",
        short_name=":grinning_face:",
        description="Grinning face",
        keywords=["happy", "joy", "smile", "grin"],
        category=EmojiCategory.FACE_EMOTION,
        primary_meaning="Happiness or joy",
        sentiment=SentimentValue.POSITIVE,
        frequency_score=0.96,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    kb.add_emoji(
        emoji="😢",
        unicode_representation="U+1F622",
        short_name=":crying_face:",
        description="Crying face",
        keywords=["sad", "unhappy", "tear", "cry"],
        category=EmojiCategory.FACE_EMOTION,
        primary_meaning="Sadness or disappointment",
        sentiment=SentimentValue.NEGATIVE,
        frequency_score=0.93,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    kb.add_emoji(
        emoji="😐",
        unicode_representation="U+1F610",
        short_name=":neutral_face:",
        description="Neutral face",
        keywords=["neutral", "blank", "expressionless"],
        category=EmojiCategory.FACE_EMOTION,
        primary_meaning="Neutral emotion or indifference",
        sentiment=SentimentValue.NEUTRAL,
        frequency_score=0.89,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    kb.add_emoji(
        emoji="🤩",
        unicode_representation="U+1F929",
        short_name=":star_struck:",
        description="Star-struck face",
        keywords=["excited", "amazed", "impressed", "star eyes"],
        category=EmojiCategory.FACE_EMOTION,
        primary_meaning="Excitement or amazement",
        sentiment=SentimentValue.VERY_POSITIVE,
        frequency_score=0.87,
        familiarity_level=FamiliarityLevel.COMMON
    )
    
    kb.add_emoji(
        emoji="😡",
        unicode_representation="U+1F621",
        short_name=":angry_face:",
        description="Angry face",
        keywords=["angry", "mad", "upset", "frustrated"],
        category=EmojiCategory.FACE_EMOTION,
        primary_meaning="Anger or frustration",
        sentiment=SentimentValue.VERY_NEGATIVE,
        frequency_score=0.85,
        familiarity_level=FamiliarityLevel.UNIVERSAL
    )
    
    # Demonstrate getting emojis by sentiment
    print("Emojis by sentiment:\n")
    sentiment_values = [
        SentimentValue.VERY_POSITIVE,
        SentimentValue.POSITIVE,
        SentimentValue.NEUTRAL,
        SentimentValue.NEGATIVE,
        SentimentValue.VERY_NEGATIVE
    ]
    
    for sentiment in sentiment_values:
        emojis = kb.get_emoji_by_sentiment(sentiment, limit=5)
        print(f"Sentiment: {sentiment.value}")
        if emojis:
            print(f"  Emojis: {' '.join(emojis)}")
        else:
            print("  No emojis found")
        print()
    
    # Demonstrate getting emojis by familiarity
    print("Emojis by familiarity level:\n")
    familiarity_levels = [
        FamiliarityLevel.UNIVERSAL,
        FamiliarityLevel.COMMON,
        FamiliarityLevel.FAMILIAR,
        FamiliarityLevel.SPECIALIZED,
        FamiliarityLevel.RARE
    ]
    
    for familiarity in familiarity_levels:
        emojis = kb.get_emojis_by_familiarity(familiarity, limit=5)
        print(f"Familiarity: {familiarity.value}")
        if emojis:
            print(f"  Emojis: {' '.join(emojis)}")
        else:
            print("  No emojis found")
        print()

def demonstrate_version_tracking():
    """Demonstrate version tracking in the knowledge base."""
    print("\n=== Version Tracking ===\n")
    
    # Initialize the knowledge base with no default data
    kb = EmojiKnowledgeBase(load_default=False)
    
    print("Creating version history with multiple releases...\n")
    
    # Create initial version with basic emojis
    kb.create_new_version(
        version="1.0.0",
        name="Initial Release",
        notes="Basic emoji set with common emotions and gestures."
    )
    
    kb.add_emoji(
        emoji="👍",
        unicode_representation="U+1F44D",
        short_name=":thumbs_up:",
        description="Thumbs up sign",
        keywords=["approval", "like", "positive"],
        update_version=True
    )
    
    kb.add_emoji(
        emoji="👋",
        unicode_representation="U+1F44B",
        short_name=":waving_hand:",
        description="Waving hand",
        keywords=["hello", "goodbye", "wave"],
        update_version=True
    )
    
    # Create version for technical emojis
    kb.create_new_version(
        version="1.1.0",
        name="Technical Emoji Pack",
        notes="Added technical and development-related emojis."
    )
    
    kb.add_emoji(
        emoji="💻",
        unicode_representation="U+1F4BB",
        short_name=":laptop:",
        description="Laptop computer",
        keywords=["computer", "work", "laptop"],
        category=EmojiCategory.OBJECT,
        domain_meanings={
            EmojiDomain.TECHNICAL: "Development environment or coding",
            EmojiDomain.BUSINESS: "Work or office tasks"
        },
        update_version=True
    )
    
    kb.add_emoji(
        emoji="🐛",
        unicode_representation="U+1F41B",
        short_name=":bug:",
        description="Bug",
        keywords=["bug", "insect", "error"],
        category=EmojiCategory.ANIMAL,
        domain_meanings={
            EmojiDomain.TECHNICAL: "Software bug or error",
            EmojiDomain.GENERAL: "Insect"
        },
        update_version=True
    )
    
    # Create version with updates
    kb.create_new_version(
        version="1.1.1",
        name="Bug Fix Release",
        notes="Updated metadata and fixed inconsistencies."
    )
    
    kb.update_emoji(
        emoji="👍",
        cultural_meanings={
            CulturalContext.GLOBAL: "General approval",
            CulturalContext.MIDDLE_EASTERN: "Can be offensive in some regions"
        }
    )
    
    # Display version history
    print("Version History:")
    for version in kb.versions:
        print(f"\nVersion: {version.version} - {version.name}")
        print(f"Release Notes: {version.notes}")
        print(f"Release Date: {datetime.fromtimestamp(version.release_date).strftime('%Y-%m-%d')}")
        
        # Show changes
        if version.new_emojis:
            print(f"  Added Emojis: {', '.join(version.new_emojis)}")
        
        if version.updated_emojis:
            print(f"  Updated Emojis: {', '.join(version.updated_emojis)}")
        
        if version.deprecated_emojis:
            print(f"  Deprecated Emojis: {', '.join(version.deprecated_emojis)}")
    
    # Show current statistics
    print("\nCurrent Knowledge Base Statistics:")
    print(f"  Total Emojis: {kb.emoji_count}")
    print(f"  Total Concepts: {kb.concept_count}")
    print(f"  Total Patterns: {kb.pattern_count}")
    print(f"  Current Version: {kb.current_version}")


def main():
    """Run all emoji knowledge base demonstrations."""
    print("="*80)
    print("                EmojiKnowledgeBase Demonstrations")
    print("="*80)
    
    demonstrate_concept_mapping()
    demonstrate_cultural_variations()
    demonstrate_domain_specific_meanings()
    demonstrate_combination_patterns()
    demonstrate_searching_and_querying()
    demonstrate_emoji_by_sentiment_and_familiarity()
    demonstrate_version_tracking()
    
    print("="*80)
    print("                        Demonstrations Complete")
    print("="*80)


if __name__ == "__main__":
    main()
