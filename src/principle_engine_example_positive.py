#!/usr/bin/env python3
"""
Example demonstrating the use of positive reinforcement functionality with the PrincipleEngine.

This example shows how to:
1. Extend a PrincipleEngine instance with the positive reinforcement function
2. Use the function to analyze interactions
3. Interpret and act on the results
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from principle_engine import PrincipleEngine
from principle_engine_positive_reinforcement import (
    extend_principle_engine,
    create_sample_interaction_data
)

# Optional imports for additional capabilities
try:
    from emotional_intelligence import EmotionalIntelligence
    EMOTIONAL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    EMOTIONAL_INTELLIGENCE_AVAILABLE = False

try:
    from learning_system import LearningSystem, LearningDimension, OutcomeType
    LEARNING_SYSTEM_AVAILABLE = True
except ImportError:
    LEARNING_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEnginePositiveExample")

def main() -> None:
    """Run the positive reinforcement example."""
    # Create PrincipleEngine instance
    engine = PrincipleEngine()
    
    # Initialize optional systems for enhanced functionality
    emotional_intelligence = None
    learning_system = None
    
    if EMOTIONAL_INTELLIGENCE_AVAILABLE:
        emotional_intelligence = EmotionalIntelligence()
        setattr(engine, '_emotional_intelligence', emotional_intelligence)
        logger.info("EmotionalIntelligence system initialized and attached to PrincipleEngine")
    
    if LEARNING_SYSTEM_AVAILABLE:
        learning_system = LearningSystem()
        setattr(engine, '_learning_system', learning_system)
        logger.info("LearningSystem initialized and attached to PrincipleEngine")
    
    # Extend the PrincipleEngine with positive reinforcement capability
    extend_principle_engine(engine)
    logger.info("PrincipleEngine extended with positive reinforcement capability")
    
    # Create sample interaction data for different scenarios
    interaction_negative = create_sample_interaction_data()
    logger.info("Created sample negative interaction data")
    
    interaction_positive = create_sample_interaction_data()
    interaction_positive["message"]["content"] = (
        "I'm excited about the progress we're making on the project. "
        "Thanks to everyone's hard work, we're back on track with our deliverables."
    )
    logger.info("Created sample positive interaction data")
    
    interaction_neutral = create_sample_interaction_data()
    interaction_neutral["message"]["content"] = (
        "The project status update is ready for review. "
        "I've included all the metrics from the last sprint."
    )
    logger.info("Created sample neutral interaction data")
    
    # Analyze the interactions
    print("\n=== Negative Interaction Analysis ===")
    result_negative = engine.prioritize_positive_reinforcement(interaction_negative, "agent_1")
    print_analysis_result(result_negative)
    
    print("\n=== Positive Interaction Analysis ===")
    result_positive = engine.prioritize_positive_reinforcement(interaction_positive, "agent_2")
    print_analysis_result(result_positive)
    
    print("\n=== Neutral Interaction Analysis ===")
    result_neutral = engine.prioritize_positive_reinforcement(interaction_neutral, "agent_3")
    print_analysis_result(result_neutral)
    
    # Demonstrate how to apply a suggested modification
    demonstrate_modification_application(interaction_negative, result_negative)
    
    # Generate a summary of findings
    print("\n=== Summary of Findings ===")
    print("1. Negative interactions have the most potential for positive steering")
    print(f"   Score: {result_negative['generative_potential_score']:.2f}, Modifications: {len(result_negative['suggested_modifications'])}")
    
    print("2. Positive interactions already contain positive elements")
    print(f"   Score: {result_positive['generative_potential_score']:.2f}, Positive Elements: {len(result_positive['identified_positive_elements'])}")
    
    print("3. Neutral interactions can benefit from added positive framing")
    print(f"   Score: {result_neutral['generative_potential_score']:.2f}, Modifications: {len(result_neutral['suggested_modifications'])}")
    
    print("\nThis example demonstrates how the 'Love as a Generative Force' principle")
    print("can be applied to steer interactions toward positive, constructive outcomes.")

def print_analysis_result(result: Dict[str, Any]) -> None:
    """Print the analysis results in a formatted way."""
    print(f"Generative Potential Score: {result['generative_potential_score']:.2f}")
    
    if result["identified_positive_elements"]:
        print("\nIdentified Positive Elements:")
        for i, element in enumerate(result["identified_positive_elements"], 1):
            print(f"  {i}. {element['element_type']}: {element['content']}")
            print(f"     Confidence: {element['confidence']:.2f}")
            if element.get('context'):
                print(f"     Context: \"{element['context'][:50]}...\"")
    else:
        print("\nNo positive elements identified.")
    
    if result["suggested_modifications"]:
        print("\nSuggested Modifications:")
        for i, mod in enumerate(result["suggested_modifications"], 1):
            print(f"  {i}. {mod['type'].capitalize()}:")
            print(f"     Suggestion: {mod['suggestion'][:80]}...")
            print(f"     Rationale: {mod['rationale']}")
    else:
        print("\nNo modifications suggested.")

def demonstrate_modification_application(
    interaction: Dict[str, Any],
    analysis_result: Dict[str, Any]
) -> None:
    """Demonstrate how to apply a suggested modification to a message."""
    if not analysis_result["suggested_modifications"]:
        print("\n=== No modifications to apply ===")
        return
    
    original_message = interaction["message"]["content"]
    print("\n=== Applying Modification Example ===")
    print(f"Original message: \"{original_message}\"")
    
    # Select the first modification with highest confidence
    modifications = sorted(
        analysis_result["suggested_modifications"], 
        key=lambda m: m.get("confidence", 0), 
        reverse=True
    )
    selected_mod = modifications[0]
    
    # Extract suggestion text (simplified for this example)
    suggestion_text = selected_mod["suggestion"]
    suggestion_sample = suggestion_text.split("'")[1] if "'" in suggestion_text else suggestion_text
    
    # Apply modification (in a real system, this would be more sophisticated)
    if selected_mod["type"] == "reframing":
        # Example of reframing the entire message
        modified_message = suggestion_sample
    elif selected_mod["type"] == "appreciation":
        # Example of adding appreciation at the beginning
        modified_message = f"{suggestion_sample} {original_message}"
    elif selected_mod["type"] == "next_steps":
        # Example of adding next steps at the end
        modified_message = f"{original_message} {suggestion_sample}"
    else:
        # Generic approach for other modification types
        modified_message = f"{original_message}\n\n{suggestion_sample}"
    
    print(f"Modified message: \"{modified_message}\"")
    print(f"Modification type: {selected_mod['type']}")
    print(f"Rationale: {selected_mod['rationale']}")
    
    # In a real system, you would update the message in the interaction data
    # and possibly send it through the principle engine for validation

if __name__ == "__main__":
    main()
