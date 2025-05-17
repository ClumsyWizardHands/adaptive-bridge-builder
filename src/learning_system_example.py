#!/usr/bin/env python3
"""
Example usage of the LearningSystem module for Adaptive Bridge Builder.

This example demonstrates how to use the LearningSystem to:
1. Track interaction patterns and their outcomes
2. Refine communication approaches based on learning
3. Maintain a growth journal tracking evolution over time
4. Implement the "Growth as a Shared Journey" principle
5. Balance adaptation with core identity maintenance
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligence
from learning_system import (
    LearningSystem,
    LearningDimension,
    OutcomeType,
    AdaptationLevel,
    GrowthJournalEntry
)

def print_divider(title: str = None):
    """Print a divider with an optional title."""
    width = 80
    if title:
        print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}")
    else:
        print(f"\n{'=' * width}")


def main():
    """Demonstrate the LearningSystem functionality."""
    print_divider("LEARNING SYSTEM DEMONSTRATION")
    
    # Initialize PrincipleEngine and EmotionalIntelligence
    principle_engine = PrincipleEngine()
    emotional_intelligence = EmotionalIntelligence(principle_engine=principle_engine)
    
    # Initialize LearningSystem with growth journal directory
    learning_system = LearningSystem(
        principle_engine=principle_engine,
        emotional_intelligence=emotional_intelligence,
        growth_journal_dir="./growth_journal"
    )
    
    print("Initialized Learning System with PrincipleEngine and EmotionalIntelligence")
    
    # Track a variety of interaction patterns with different outcomes
    print_divider("TRACKING INTERACTION PATTERNS")
    
    # Example 1: Technical Support Interaction (Successful)
    pattern_id_1 = learning_system.track_interaction(
        pattern_description="Respond to inquiry with detailed technical information",
        context={
            "agent_type": "technical_support",
            "topic": "system_configuration",
            "inquiry_type": "technical",
            "style_formality": "FORMAL",
            "style_detail_level": "DETAILED"
        },
        dimensions=[
            LearningDimension.COMMUNICATION_EFFECTIVENESS,
            LearningDimension.TASK_COLLABORATION
        ],
        outcome=OutcomeType.SUCCESSFUL,
        notes="Agent expressed satisfaction with detailed technical response"
    )
    print(f"Tracked successful technical interaction pattern (ID: {pattern_id_1})")
    
    # Example 2: Empathetic Response to Complaint (Partially Successful)
    pattern_id_2 = learning_system.track_interaction(
        pattern_description="Respond to complaint with empathetic acknowledgment",
        context={
            "agent_type": "customer_service",
            "topic": "service_disruption",
            "emotion": "angry",
            "style_emotional_tone": "NEGATIVE",
            "style_formality": "FORMAL"
        },
        dimensions=[
            LearningDimension.EMOTIONAL_INTELLIGENCE,
            LearningDimension.CONFLICT_RESOLUTION
        ],
        outcome=OutcomeType.PARTIALLY_SUCCESSFUL,
        notes="Agent acknowledged complaint but wanted more concrete solutions"
    )
    print(f"Tracked partially successful empathetic interaction pattern (ID: {pattern_id_2})")
    
    # Example 3: Solution-Focused Response (Successful)
    pattern_id_3 = learning_system.track_interaction(
        pattern_description="Respond to complaint with solution-focused approach",
        context={
            "agent_type": "customer_service",
            "topic": "service_disruption",
            "emotion": "angry",
            "style_emotional_tone": "NEUTRAL",
            "style_formality": "FORMAL"
        },
        dimensions=[
            LearningDimension.EMOTIONAL_INTELLIGENCE,
            LearningDimension.CONFLICT_RESOLUTION
        ],
        outcome=OutcomeType.SUCCESSFUL,
        notes="Agent appreciated focus on concrete solutions"
    )
    print(f"Tracked successful solution-focused interaction pattern (ID: {pattern_id_3})")
    
    # Example 4: Casual Collaborative Discussion (Unsuccessful)
    pattern_id_4 = learning_system.track_interaction(
        pattern_description="Engage in collaborative planning with casual style",
        context={
            "agent_type": "project_manager",
            "topic": "project_planning",
            "team_size": 5,
            "style_formality": "CASUAL",
            "style_detail_level": "LOW"
        },
        dimensions=[
            LearningDimension.TASK_COLLABORATION,
            LearningDimension.TRUST_BUILDING
        ],
        outcome=OutcomeType.UNSUCCESSFUL,
        notes="Team found casual approach confusing for complex planning tasks"
    )
    print(f"Tracked unsuccessful casual planning interaction pattern (ID: {pattern_id_4})")
    
    # Example 5: Structured Collaborative Discussion (Successful)
    pattern_id_5 = learning_system.track_interaction(
        pattern_description="Engage in collaborative planning with structured approach",
        context={
            "agent_type": "project_manager",
            "topic": "project_planning",
            "team_size": 5,
            "style_formality": "NEUTRAL",
            "style_detail_level": "HIGH"
        },
        dimensions=[
            LearningDimension.TASK_COLLABORATION,
            LearningDimension.TRUST_BUILDING
        ],
        outcome=OutcomeType.SUCCESSFUL,
        notes="Team appreciated structured approach with clear action items"
    )
    print(f"Tracked successful structured planning interaction pattern (ID: {pattern_id_5})")
    
    # Reflect on patterns and identify potential adaptations
    print_divider("REFLECTING ON PATTERNS")
    
    # Reflect on task collaboration patterns
    task_collaboration_insights = learning_system.reflect_on_patterns(LearningDimension.TASK_COLLABORATION)
    print(f"Generated {len(task_collaboration_insights)} insights for Task Collaboration dimension:")
    for i, insight in enumerate(task_collaboration_insights, 1):
        print(f"  {i}. {insight['description']} (confidence: {insight['confidence']:.2f})")
        if "from_value" in insight and "to_value" in insight:
            print(f"     Suggested adaptation: Change {insight['factor']} from '{insight['from_value']}' to '{insight['to_value']}'")
    
    # Reflect on emotional intelligence patterns
    emotional_intelligence_insights = learning_system.reflect_on_patterns(LearningDimension.EMOTIONAL_INTELLIGENCE)
    print(f"Generated {len(emotional_intelligence_insights)} insights for Emotional Intelligence dimension:")
    for i, insight in enumerate(emotional_intelligence_insights, 1):
        print(f"  {i}. {insight['description']} (confidence: {insight['confidence']:.2f})")
        if "from_value" in insight and "to_value" in insight:
            print(f"     Suggested adaptation: Change {insight['factor']} from '{insight['from_value']}' to '{insight['to_value']}'")
    
    # Apply an adaptation based on insights
    print_divider("APPLYING ADAPTATIONS")
    
    # Find a suitable style adaptation
    adaptation_to_apply = None
    for insight in task_collaboration_insights + emotional_intelligence_insights:
        if insight["type"] == "style_adaptation" and insight.get("confidence", 0) > 0.6:
            adaptation_to_apply = insight
            break
    
    if adaptation_to_apply:
        # Apply adaptation to the first pattern
        applied = learning_system.apply_adaptation(
            pattern_id=pattern_id_1,
            adaptation=adaptation_to_apply,
            apply_level=AdaptationLevel.MODERATE
        )
        
        if applied:
            print(f"Applied adaptation: {adaptation_to_apply['description']}")
            
            # Later, record a successful outcome
            learning_system.record_adaptation_outcome(
                pattern_id=pattern_id_1,
                adaptation_index=0,  # First adaptation
                outcome=OutcomeType.SUCCESSFUL,
                notes="Adaptation improved communication effectiveness significantly"
            )
            print("Recorded successful outcome for the adaptation")
        else:
            print("Adaptation could not be applied (possibly violates core identity)")
    else:
        print("No suitable adaptation found with sufficient confidence")
    
    # Add a milestone to mark significant progress
    learning_system._add_milestone_to_journal(
        description="Achieved balanced approach to technical and emotional interactions",
        dimension=LearningDimension.ADAPTABILITY.name,
        notes="System now adapts communication style based on agent type and interaction context"
    )
    print("Added milestone to growth journal")
    
    # Generate and display growth summary
    print_divider("GROWTH SUMMARY")
    
    summary = learning_system.generate_growth_summary()
    
    print("Learning Metrics:")
    print(f"  Overall Success Rate: {summary['metrics']['overall_success_rate']:.2f}")
    print(f"  Adaptability Score: {summary['metrics']['adaptability_score']:.2f}")
    print(f"  Identity Preservation Score: {summary['metrics']['identity_preservation_score']:.2f}")
    print(f"  Balance Score: {summary['metrics']['balance_score']:.2f}")
    
    print("\nDimension Performance:")
    for dim_name, dim_data in summary["dimensions"].items():
        print(f"  {dim_name}: Success Rate {dim_data['success_rate']:.2f}, {dim_data['pattern_count']} patterns")
    
    # Display growth journal entries
    print_divider("GROWTH JOURNAL")
    
    journal_entries = learning_system.get_growth_journal(limit=5)
    print(f"Recent growth journal entries ({len(journal_entries)} shown):")
    
    for i, entry in enumerate(journal_entries, 1):
        print(f"\n{i}. {entry['entry_type'].upper()} - {entry['dimension']}")
        print(f"   Date: {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        content_preview = entry['content'].split('\n')[0]
        print(f"   Content: {content_preview[:80]}{'...' if len(content_preview) > 80 else ''}")
    
    print_divider("DEMONSTRATION COMPLETE")
    print("The Learning System has successfully tracked interactions, reflected on patterns,")
    print("applied adaptations, and maintained a growth journal that tracks the agent's evolution")
    print("while balancing adaptation with core identity preservation.")


if __name__ == "__main__":
    main()
