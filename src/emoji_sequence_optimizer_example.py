"""
Example demonstrating the EmojiSequenceOptimizer component functionality.

This file provides concrete examples of how the optimizer improves emoji
sequences for different communication contexts and optimization profiles.
"""

from emoji_sequence_optimizer import (
    EmojiSequenceOptimizer,
    OptimizationContext,
    OptimizationProfile,
    GroupingStrategy,
    OptimizationWeight
)

from emoji_knowledge_base import (
    EmojiKnowledgeBase,
    EmojiDomain,
    CulturalContext,
    FamiliarityLevel
)


def demonstrate_optimization_profiles():
    """Show how different optimization profiles affect emoji sequences."""
    print("\n=== Optimization Profiles ===\n")
    
    # Initialize optimizer with knowledge base
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test message with multiple emojis
    test_message = "🚀🎉🏆💻📊👨‍💻🤔🐛🔨✅👀💯🔥"
    
    # Profiles to demonstrate
    profiles = [
        OptimizationProfile.PRECISE,
        OptimizationProfile.CONCISE,
        OptimizationProfile.EXPRESSIVE,
        OptimizationProfile.UNIVERSAL,
        OptimizationProfile.TECHNICAL,
        OptimizationProfile.BUSINESS,
        OptimizationProfile.SOCIAL
    ]
    
    print(f"Original sequence: {test_message}")
    print(f"Original length: {len(test_message)}")
    print()
    
    # Optimize for each profile
    for profile in profiles:
        context = OptimizationContext(profile=profile)
        result = optimizer.optimize_sequence(test_message, context)
        
        print(f"{profile.value.capitalize()} profile:")
        print(f"  Optimized: {result.optimized_sequence}")
        print(f"  Score: {result.optimization_score:.2f}")
        
        # Show changes
        if result.removals:
            removed = ", ".join(f"{r[0]}" for r in result.removals)
            print(f"  Removed: {removed}")
        
        if result.substitutions:
            substituted = ", ".join(f"{s[0]}→{s[1]}" for s in result.substitutions)
            print(f"  Substituted: {substituted}")
        
        if result.additions:
            added = ", ".join(f"{a[0]}" for a in result.additions)
            print(f"  Added: {added}")
        
        if result.groups:
            print(f"  Groups: {result.groups}")
        
        print()


def demonstrate_domain_specific_optimization():
    """Show how domain context affects emoji sequence optimization."""
    print("\n=== Domain-Specific Optimization ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test message with multiple interpretations
    test_message = "🔥🚀🐛💻✅"
    
    # Domains to demonstrate
    domains = [
        EmojiDomain.GENERAL,
        EmojiDomain.TECHNICAL,
        EmojiDomain.BUSINESS,
        EmojiDomain.SOCIAL
    ]
    
    print(f"Original sequence: {test_message}")
    print()
    
    # Optimize for each domain
    for domain in domains:
        context = OptimizationContext(
            domain=domain,
            profile=OptimizationProfile.PRECISE
        )
        result = optimizer.optimize_sequence(test_message, context)
        
        print(f"{domain.value.capitalize()} domain:")
        print(f"  Optimized: {result.optimized_sequence}")
        
        # Show substitutions with reasons
        if result.substitutions:
            for original, new, reason in result.substitutions:
                print(f"  • {original} → {new}: {reason}")
        
        print()

    # Technical context example - code deployment message
    tech_message = "💻🔄🚀✅🎉"
    tech_context = OptimizationContext(
        domain=EmojiDomain.TECHNICAL,
        profile=OptimizationProfile.TECHNICAL
    )
    tech_result = optimizer.optimize_sequence(tech_message, tech_context)
    
    print("Technical deployment message:")
    print(f"  Original: {tech_message}")
    print(f"  Optimized: {tech_result.optimized_sequence}")
    print(f"  Interpretation: Code updated, deployed successfully, celebration")
    print()
    
    # Business context example - meeting reminder message
    business_message = "📅🕐👥📊💼"
    business_context = OptimizationContext(
        domain=EmojiDomain.BUSINESS,
        profile=OptimizationProfile.BUSINESS
    )
    business_result = optimizer.optimize_sequence(business_message, business_context)
    
    print("Business meeting reminder message:")
    print(f"  Original: {business_message}")
    print(f"  Optimized: {business_result.optimized_sequence}")
    print(f"  Interpretation: Scheduled time for team to discuss metrics/performance")
    print()


def demonstrate_cultural_adaptations():
    """Show how cultural context affects emoji sequence optimization."""
    print("\n=== Cultural Adaptations ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test message with cultural variations
    test_message = "👍🙏👌🤙✌️"
    
    # Cultural contexts to demonstrate
    cultures = [
        CulturalContext.GLOBAL,
        CulturalContext.WESTERN,
        CulturalContext.EASTERN_ASIAN,
        CulturalContext.MIDDLE_EASTERN
    ]
    
    print(f"Original sequence: {test_message}")
    print(f"Message sentiment: Positive affirmation and agreement")
    print()
    
    # Optimize for each cultural context
    for culture in cultures:
        context = OptimizationContext(
            cultural_context=culture,
            profile=OptimizationProfile.CULTURAL
        )
        result = optimizer.optimize_sequence(test_message, context)
        
        print(f"{culture.value.capitalize()} cultural context:")
        print(f"  Optimized: {result.optimized_sequence}")
        
        # Show substitutions with reasons
        if result.substitutions:
            for original, new, reason in result.substitutions:
                print(f"  • {original} → {new}: {reason}")
        
        print()


def demonstrate_grouping_strategies():
    """Show how different grouping strategies affect readability."""
    print("\n=== Grouping Strategies ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Complex emoji sequence
    test_message = "🌞🌈🌧️🌪️🌨️🔥💧❄️🌊💨🌱🌳🌲🌸🍂"
    
    # Grouping strategies to demonstrate
    strategies = [
        GroupingStrategy.NONE,
        GroupingStrategy.SEMANTIC,
        GroupingStrategy.VISUAL,
        GroupingStrategy.SYNTACTIC
    ]
    
    print(f"Original sequence: {test_message}")
    print()
    
    # Optimize with each grouping strategy
    for strategy in strategies:
        context = OptimizationContext(
            grouping_strategy=strategy,
            space_between_groups=True
        )
        result = optimizer.optimize_sequence(test_message, context)
        
        print(f"{strategy.value.capitalize()} grouping:")
        print(f"  Optimized: {result.optimized_sequence}")
        print(f"  Groups: {result.groups}")
        print()


def demonstrate_conciseness_vs_expressiveness():
    """Compare concise vs expressive optimization for the same message."""
    print("\n=== Conciseness vs Expressiveness ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test messages
    messages = [
        "😀😃😁😄😆😊🥰😍🤗",  # Happy emotions
        "🏃‍♂️🏋️‍♂️🤸‍♂️🏊‍♂️🚴‍♂️⛹️‍♂️🎯🏆🏅",  # Sports activities
        "🎮🎲🎯🎪🎨🎭🎬🎤🎧🎼",  # Entertainment
        "🚗🚕🚙🚌🚎🚓🚑🚒🦼🚲"   # Transportation
    ]
    
    # Create contexts
    concise_context = OptimizationContext(profile=OptimizationProfile.CONCISE)
    expressive_context = OptimizationContext(profile=OptimizationProfile.EXPRESSIVE)
    
    # Compare optimizations
    for i, message in enumerate(messages):
        print(f"Message {i+1}: {message}")
        
        # Concise optimization
        concise_result = optimizer.optimize_sequence(message, concise_context)
        print(f"  Concise: {concise_result.optimized_sequence}")
        print(f"  Length reduction: {len(message)} → {len(concise_result.optimized_sequence.replace(' ', ''))} emojis")
        
        # Expressive optimization
        expressive_result = optimizer.optimize_sequence(message, expressive_context)
        print(f"  Expressive: {expressive_result.optimized_sequence}")
        
        print()


def demonstrate_custom_optimization_profile():
    """Show how to create and use a custom optimization profile."""
    print("\n=== Custom Optimization Profile ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test message
    test_message = "🚀📈💻🐛🔥👀✅📊🤔😊🎉💯👍"
    
    print(f"Original message: {test_message}")
    print()
    
    # Default weights for reference
    print("Default profiles for reference:")
    profiles = optimizer.get_optimization_profiles()
    for profile_name in [OptimizationProfile.PRECISE, OptimizationProfile.TECHNICAL]:
        weights = profiles[profile_name]
        print(f"  {profile_name.value}:")
        for weight, value in weights.items():
            print(f"    {weight.value}: {value:.2f}")
        print()
    
    # Create custom profile - "Technical but Expressive"
    custom_weights = {
        OptimizationWeight.PRECISION: 0.9,
        OptimizationWeight.CLARITY: 0.8,
        OptimizationWeight.UNIVERSALITY: 0.7,
        OptimizationWeight.EXPRESSIVENESS: 0.7,  # Higher than default technical
        OptimizationWeight.EMOTIONALITY: 0.5,    # Higher than default technical
        OptimizationWeight.CONCISENESS: 0.6,
        OptimizationWeight.CREATIVITY: 0.4,      # Higher than default technical
        OptimizationWeight.CONSISTENCY: 0.7
    }
    
    custom_context = optimizer.create_custom_profile(custom_weights)
    custom_context.domain = EmojiDomain.TECHNICAL
    
    # Optimize with custom profile
    custom_result = optimizer.optimize_sequence(test_message, custom_context)
    
    print("Custom Technical-Expressive profile:")
    print(f"  Optimized: {custom_result.optimized_sequence}")
    print(f"  Score: {custom_result.optimization_score:.2f}")
    
    if custom_result.removals:
        removed = ", ".join(f"{r[0]}" for r in custom_result.removals)
        print(f"  Removed: {removed}")
    
    if custom_result.substitutions:
        substituted = ", ".join(f"{s[0]}→{s[1]}" for s in custom_result.substitutions)
        print(f"  Substituted: {substituted}")
    
    if custom_result.additions:
        added = ", ".join(f"{a[0]}" for a in custom_result.additions)
        print(f"  Added: {added}")
    
    print()
    
    # Compare with standard profiles
    technical_context = OptimizationContext(profile=OptimizationProfile.TECHNICAL)
    technical_result = optimizer.optimize_sequence(test_message, technical_context)
    
    expressive_context = OptimizationContext(profile=OptimizationProfile.EXPRESSIVE)
    expressive_result = optimizer.optimize_sequence(test_message, expressive_context)
    
    print("Comparison with standard profiles:")
    print(f"  Technical: {technical_result.optimized_sequence}")
    print(f"  Expressive: {expressive_result.optimized_sequence}")
    print(f"  Custom Technical-Expressive: {custom_result.optimized_sequence}")
    print()


def demonstrate_sequence_analysis():
    """Show how to analyze emoji sequences without optimizing them."""
    print("\n=== Emoji Sequence Analysis ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Test sequences
    sequences = [
        "🚀✅🎉",  # Simple, common sequence (deployment success)
        "🔥💻🐛⚠️🔒",  # Technical sequence (security vulnerability)
        "👍👌😊",  # Common but culturally variable
        "🌮🎭🧠🦄🔮"  # Unusual/creative sequence
    ]
    
    for sequence in sequences:
        print(f"Analyzing: {sequence}")
        analysis = optimizer.analyze_sequence(sequence)
        
        print(f"  Length: {analysis['length']} emojis")
        print(f"  Familiarity score: {analysis['familiarity']:.2f}")
        print(f"  Ambiguity score: {analysis['ambiguity']:.2f}")
        print(f"  Universality score: {analysis['universality']:.2f}")
        
        if analysis['most_ambiguous']:
            print(f"  Most ambiguous emoji: {analysis['most_ambiguous']}")
        
        if analysis['least_familiar']:
            print(f"  Least familiar emoji: {analysis['least_familiar']}")
        
        print(f"  Cultural specificity: {analysis['cultural_specificity']:.2f}")
        print(f"  Domain specificity: {analysis['domain_specificity']:.2f}")
        print()


def demonstrate_real_world_examples():
    """Show optimization of realistic emoji messages in different contexts."""
    print("\n=== Real-World Examples ===\n")
    
    # Initialize optimizer
    kb = EmojiKnowledgeBase(load_default=True)
    optimizer = EmojiSequenceOptimizer(knowledge_base=kb)
    
    # Example 1: Bug report message
    bug_report = "🐛🚨💻😱❓"
    technical_context = OptimizationContext(
        profile=OptimizationProfile.TECHNICAL,
        domain=EmojiDomain.TECHNICAL
    )
    
    bug_result = optimizer.optimize_sequence(bug_report, technical_context)
    
    print("Bug report message:")
    print(f"  Original: {bug_report}")
    print(f"  Optimized for technical: {bug_result.optimized_sequence}")
    print(f"  Interpretation: Critical bug found in code, needs investigation")
    print()
    
    # Example 2: Meeting invitation
    meeting_invite = "📅👥💼🕒☕"
    business_context = OptimizationContext(
        profile=OptimizationProfile.BUSINESS,
        domain=EmojiDomain.BUSINESS,
        grouping_strategy=GroupingStrategy.SEMANTIC
    )
    
    meeting_result = optimizer.optimize_sequence(meeting_invite, business_context)
    
    print("Meeting invitation message:")
    print(f"  Original: {meeting_invite}")
    print(f"  Optimized for business: {meeting_result.optimized_sequence}")
    print(f"  Interpretation: Calendar meeting with team, business discussion, scheduled time, coffee break")
    print()
    
    # Example 3: Social celebration message
    celebration = "🎉🎂🎁🎊🥳🍾🥂👏😊❤️"
    social_context = OptimizationContext(
        profile=OptimizationProfile.SOCIAL,
        domain=EmojiDomain.SOCIAL,
        grouping_strategy=GroupingStrategy.SEMANTIC
    )
    
    celebration_result = optimizer.optimize_sequence(celebration, social_context)
    
    print("Celebration message:")
    print(f"  Original: {celebration}")
    print(f"  Optimized for social: {celebration_result.optimized_sequence}")
    print(f"  Interpretation: Party celebration with cake, gifts, cheers, and love")
    print()
    
    # Example 4: Cross-cultural greeting
    greeting = "👋😊🌍👐🙏"
    universal_context = OptimizationContext(
        profile=OptimizationProfile.UNIVERSAL,
        cultural_context=CulturalContext.GLOBAL
    )
    
    greeting_result = optimizer.optimize_sequence(greeting, universal_context)
    
    print("Cross-cultural greeting message:")
    print(f"  Original: {greeting}")
    print(f"  Optimized for universal: {greeting_result.optimized_sequence}")
    print(f"  Interpretation: Friendly greeting to people worldwide, with respect")
    print()


def main():
    """Run all emoji sequence optimizer demonstrations."""
    print("="*80)
    print("              EmojiSequenceOptimizer Demonstrations")
    print("="*80)
    
    demonstrate_optimization_profiles()
    demonstrate_domain_specific_optimization()
    demonstrate_cultural_adaptations()
    demonstrate_grouping_strategies()
    demonstrate_conciseness_vs_expressiveness()
    demonstrate_custom_optimization_profile()
    demonstrate_sequence_analysis()
    demonstrate_real_world_examples()
    
    print("="*80)
    print("                     Demonstrations Complete")
    print("="*80)


if __name__ == "__main__":
    main()
