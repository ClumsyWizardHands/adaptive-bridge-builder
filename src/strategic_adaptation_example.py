#!/usr/bin/env python3
"""
Strategic Adaptation Example

This module demonstrates how to use the strategic_adaptation module to evaluate and adapt
communication strategies based on historical performance data while maintaining alignment
with core principles.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import uuid

from learning_system import (
    LearningSystem, LearningDimension, OutcomeType,
    InteractionPattern, AdaptationLevel
)
from principle_engine import PrincipleEngine
from communication_style import (
    CommunicationStyle, EmotionalTone, FormalityLevel
)
from strategic_adaptation import (
    CommunicationStrategy, adapt_strategy, 
    integrate_with_continuous_evolution
)
from continuous_evolution_system import ContinuousEvolutionSystem

def create_sample_learning_data(learning_system: LearningSystem) -> None:
    """Create sample interaction patterns for testing strategy adaptation."""
    
    # Pattern 1: High formality, professional tone, high detail level
    pattern1 = InteractionPattern(
        pattern_id="comm_pattern_formal_professional",
        description="Formal professional communication style with high detail",
        context={
            "style_formality_level": "FORMAL",
            "style_emotional_tone": "PROFESSIONAL",
            "style_detail_level": "HIGH",
            "style_technical_density": "MEDIUM",
            "style_response_time": "NORMAL",
            "audience": "technical_professionals"
        },
        occurrences=12,
        successful_count=10,
        unsuccessful_count=1,
        neutral_count=1,
        success_rate=0.85,
        confidence=0.8
    )
    learning_system.interaction_patterns[pattern1.pattern_id] = pattern1
    
    # Pattern 2: Neutral formality, friendly tone, medium detail level
    pattern2 = InteractionPattern(
        pattern_id="comm_pattern_neutral_friendly",
        description="Neutral friendly communication style with medium detail",
        context={
            "style_formality_level": "NEUTRAL",
            "style_emotional_tone": "FRIENDLY",
            "style_detail_level": "MEDIUM",
            "style_technical_density": "LOW",
            "style_response_time": "FAST",
            "audience": "general_users"
        },
        occurrences=20,
        successful_count=18,
        unsuccessful_count=1,
        neutral_count=1,
        success_rate=0.9,
        confidence=0.9
    )
    learning_system.interaction_patterns[pattern2.pattern_id] = pattern2
    
    # Pattern 3: Casual formality, enthusiastic tone, low detail level
    pattern3 = InteractionPattern(
        pattern_id="comm_pattern_casual_enthusiastic",
        description="Casual enthusiastic communication style with low detail",
        context={
            "style_formality_level": "CASUAL",
            "style_emotional_tone": "ENTHUSIASTIC",
            "style_detail_level": "LOW",
            "style_technical_density": "VERY_LOW",
            "style_response_time": "VERY_FAST",
            "audience": "novice_users"
        },
        occurrences=8,
        successful_count=4,
        unsuccessful_count=3,
        neutral_count=1,
        success_rate=0.5,
        confidence=0.7
    )
    learning_system.interaction_patterns[pattern3.pattern_id] = pattern3
    
    # Pattern 4: Formal, professional, but with low detail (performing poorly)
    pattern4 = InteractionPattern(
        pattern_id="comm_pattern_formal_low_detail",
        description="Formal professional style with low detail",
        context={
            "style_formality_level": "FORMAL",
            "style_emotional_tone": "PROFESSIONAL",
            "style_detail_level": "LOW",
            "style_technical_density": "MEDIUM",
            "style_response_time": "NORMAL",
            "audience": "technical_professionals"
        },
        occurrences=5,
        successful_count=1,
        unsuccessful_count=3,
        neutral_count=1,
        success_rate=0.3,
        confidence=0.6
    )
    learning_system.interaction_patterns[pattern4.pattern_id] = pattern4
    
    # Pattern 5: Neutral formality, educational tone, high detail level
    pattern5 = InteractionPattern(
        pattern_id="comm_pattern_neutral_educational",
        description="Neutral educational style with high detail",
        context={
            "style_formality_level": "NEUTRAL",
            "style_emotional_tone": "EDUCATIONAL",
            "style_detail_level": "HIGH",
            "style_technical_density": "MEDIUM_HIGH",
            "style_response_time": "NORMAL",
            "audience": "students"
        },
        occurrences=15,
        successful_count=12,
        unsuccessful_count=1,
        neutral_count=2,
        success_rate=0.85,
        confidence=0.85
    )
    learning_system.interaction_patterns[pattern5.pattern_id] = pattern5

def create_sample_principles(principle_engine: PrincipleEngine) -> None:
    """Create sample principles for testing strategy adaptation."""
    
    # Principle 1: Clear Communication
    principle1 = {
        "id": "clear_communication",
        "name": "Clear Communication",
        "description": "Communication should be clear, concise, and accessible to the intended audience.",
        "keywords": ["clarity", "accessibility", "conciseness", "understanding"],
        "evaluation_criteria": [
            "Uses language appropriate for the audience",
            "Avoids unnecessary jargon",
            "Presents information in a structured manner",
            "Explains complex concepts in understandable terms"
        ],
        "weight": 0.8
    }
    principle_engine.principles.append(principle1)
    
    # Principle 2: Respectful Dialogue
    principle2 = {
        "id": "respectful_dialogue",
        "name": "Respectful Dialogue",
        "description": "All communication should respect the dignity and autonomy of all participants.",
        "keywords": ["respect", "dignity", "autonomy", "equality"],
        "evaluation_criteria": [
            "Uses respectful language",
            "Acknowledges different perspectives",
            "Avoids condescension or superiority",
            "Treats all participants as equals"
        ],
        "weight": 0.9
    }
    principle_engine.principles.append(principle2)
    
    # Principle 3: Technical Accuracy
    principle3 = {
        "id": "technical_accuracy",
        "name": "Technical Accuracy",
        "description": "Technical information must be accurate, up-to-date, and well-sourced.",
        "keywords": ["accuracy", "correctness", "sources", "verification"],
        "evaluation_criteria": [
            "Provides accurate technical information",
            "Cites sources when appropriate",
            "Updates information as needed",
            "Acknowledges uncertainty when present"
        ],
        "weight": 0.7
    }
    principle_engine.principles.append(principle3)

def create_sample_strategy() -> CommunicationStrategy:
    """Create a sample communication strategy for testing adaptation."""
    
    # Create a strategy for technical professionals
    strategy = CommunicationStrategy(
        strategy_id=str(uuid.uuid4()),
        name="Technical Professional Engagement",
        description="Communication strategy for engaging with technical professionals",
        target_audiences=["technical_professionals", "engineers", "developers"],
        style_parameters={
            "detail_level": "MEDIUM",  # Currently medium, but we expect this to get adapted to HIGH
            "technical_density": "MEDIUM",
            "response_time": "NORMAL",
            "vocabulary_complexity": "HIGH",
            "feedback_frequency": "LOW"
        },
        formality_level=FormalityLevel.FORMAL,
        emotional_tone=EmotionalTone.PROFESSIONAL,
        core_values=["accuracy", "efficiency", "respect"],
        immutable_aspects=["emotional_tone"]  # We don't want to change the professional tone
    )
    
    # Add some performance history
    strategy.performance_history = [
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "effectiveness_score": 0.65,
            "context": {"audience": "technical_professionals"}
        },
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
            "effectiveness_score": 0.68,
            "context": {"audience": "engineers"}
        },
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            "effectiveness_score": 0.63,
            "context": {"audience": "developers"}
        }
    ]
    
    return strategy

def main() -> None:
    """Main function to demonstrate strategy adaptation."""
    
    print("\n" + "="*80)
    print("STRATEGIC ADAPTATION EXAMPLE")
    print("="*80 + "\n")
    
    # Step 1: Set up the necessary components
    learning_system = LearningSystem()
    principle_engine = PrincipleEngine()
    
    # Step 2: Create sample data
    create_sample_learning_data(learning_system)
    create_sample_principles(principle_engine)
    strategy = create_sample_strategy()
    
    # Step 3: Display initial strategy
    print("INITIAL COMMUNICATION STRATEGY:")
    print(f"Name: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Target Audiences: {', '.join(strategy.target_audiences)}")
    print(f"Formality Level: {strategy.formality_level.name if isinstance(strategy.formality_level, FormalityLevel) else strategy.formality_level}")
    print(f"Emotional Tone: {strategy.emotional_tone.name if isinstance(strategy.emotional_tone, EmotionalTone) else strategy.emotional_tone}")
    print("Style Parameters:")
    for param, value in strategy.style_parameters.items():
        print(f"  - {param}: {value}")
    print(f"Core Values: {', '.join(strategy.core_values)}")
    print(f"Immutable Aspects: {', '.join(strategy.immutable_aspects)}")
    
    print("\n" + "-"*80 + "\n")
    
    # Step 4: Apply moderate adaptation
    print("APPLYING MODERATE ADAPTATION...")
    adapted_strategy, adaptation_details = adapt_strategy(
        strategy=strategy,
        learning_system=learning_system,
        principle_engine=principle_engine,
        adaptation_level=AdaptationLevel.MODERATE
    )
    
    # Step 5: Display adaptation results
    print("\nADAPTATION RESULTS:")
    if adaptation_details.get("adapted", False):
        print("Strategy was adapted successfully.")
        print(f"Effectiveness score before adaptation: {adaptation_details['evaluation']['effectiveness_score']:.2f}")
        print("\nChanges applied:")
        for change in adaptation_details.get("changes", []):
            print(f"  • {change['component']}: {change['from_value']} → {change['to_value']}")
            print(f"    Rationale: {change['rationale']}")
    else:
        print(f"Strategy was not adapted. Reason: {adaptation_details.get('reason', 'Unknown')}")
    
    print("\n" + "-"*80 + "\n")
    
    # Step 6: Display adapted strategy
    print("ADAPTED COMMUNICATION STRATEGY:")
    print(f"Name: {adapted_strategy.name}")
    print(f"Description: {adapted_strategy.description}")
    print(f"Formality Level: {adapted_strategy.formality_level.name if isinstance(adapted_strategy.formality_level, FormalityLevel) else adapted_strategy.formality_level}")
    print(f"Emotional Tone: {adapted_strategy.emotional_tone.name if isinstance(adapted_strategy.emotional_tone, EmotionalTone) else adapted_strategy.emotional_tone}")
    print("Style Parameters:")
    for param, value in adapted_strategy.style_parameters.items():
        # Highlight changed parameters
        unchanged = strategy.style_parameters.get(param) == value
        marker = "" if unchanged else " [CHANGED]"
        print(f"  - {param}: {value}{marker}")
    
    print("\n" + "-"*80 + "\n")
    
    # Step 7: Now try with the Continuous Evolution System
    print("INTEGRATING WITH CONTINUOUS EVOLUTION SYSTEM...")
    
    # Set up a Continuous Evolution System
    continuous_system = ContinuousEvolutionSystem(
        learning_system=learning_system,
        principle_engine=principle_engine
    )
    
    # Create a slightly different strategy for this demo
    strategy2 = create_sample_strategy()
    strategy2.strategy_id = str(uuid.uuid4())
    strategy2.name = "Technical Community Engagement"
    strategy2.style_parameters["detail_level"] = "LOW"  # This is likely suboptimal
    
    # Integrate with the system
    print("\nApplying adaptation through the Continuous Evolution System...")
    adapted_strategy2, adaptation_details2 = integrate_with_continuous_evolution(
        continuous_system, strategy2
    )
    
    # Display results
    print("\nINTEGRATED ADAPTATION RESULTS:")
    if adaptation_details2.get("adapted", False):
        print("Strategy was adapted through the Continuous Evolution System.")
        print(f"Adaptation level: {adaptation_details2.get('adaptation_level', 'Unknown')}")
        print("\nChanges applied:")
        for change in adaptation_details2.get("changes", []):
            print(f"  • {change['component']}: {change['from_value']} → {change['to_value']}")
            print(f"    Rationale: {change['rationale']}")
    else:
        print(f"Strategy was not adapted. Reason: {adaptation_details2.get('reason', 'Unknown')}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()