import pprint
#!/usr/bin/env python3
"""
Ethical Dilemma Resolver Example

This module demonstrates how to use the ethical_dilemma_resolver module to resolve
ethical dilemmas by evaluating possible actions against a hierarchy of principles
while balancing efficiency considerations.
"""

import json
from pprint import pprint
from typing import Dict, List, Any
from enum import Enum

from principle_engine import PrincipleEngine
from ethical_dilemma_resolver import (
    resolve_ethical_dilemma, EthicalPriority
)

def setup_principle_engine() -> PrincipleEngine:
    """Create and set up a PrincipleEngine for testing."""
    return PrincipleEngine()

def example_data_privacy_dilemma(principle_engine: PrincipleEngine) -> None:
    """Example of resolving a data privacy ethical dilemma."""
    print("\n" + "="*80)
    print("EXAMPLE 1: DATA PRIVACY DILEMMA")
    print("="*80)
    
    # Define the dilemma
    dilemma_description = """
    We need to decide how to handle user data collection for our new feature.
    We can collect comprehensive data to improve the service significantly,
    collect minimal data with less improvement potential, or make data sharing
    fully opt-in but risk poor adoption.
    """
    
    # Define possible actions
    possible_actions = [
        {
            "id": "comprehensive_collection",
            "description": "Collect comprehensive user data by default with opt-out option",
            "efficiency_score": 0.9,  # Very efficient for the system
            "side_effects": [
                "May reduce user trust if not communicated well",
                "Increases security responsibilities and risks"
            ],
            "context_alignment": 0.7  # Aligns well with feature needs
        },
        {
            "id": "minimal_collection",
            "description": "Collect only minimal, anonymous usage data by default",
            "efficiency_score": 0.6,  # Moderately efficient
            "side_effects": [
                "Limits potential for personalization",
                "Reduces some analytical capabilities"
            ],
            "context_alignment": 0.8  # Aligns very well with privacy-conscious users
        },
        {
            "id": "opt_in_collection",
            "description": "Make all data collection fully opt-in with clear benefits explanation",
            "efficiency_score": 0.3,  # Less efficient due to likely low participation
            "side_effects": [
                "May result in very limited data collection",
                "Creates additional UI/UX requirements"
            ],
            "context_alignment": 0.5  # Mixed alignment with different user groups
        }
    ]
    
    # Additional context
    context = {
        "user_base": "diverse, includes privacy-conscious demographics",
        "regulatory_environment": "strict data protection laws apply",
        "competitors": "mostly using opt-out approaches",
        "urgency": 0.7,  # Fairly urgent decision
        "stakeholders": [
            {"name": "users", "impact": "high"},
            {"name": "product team", "impact": "high"},
            {"name": "legal team", "impact": "medium"},
            {"name": "marketing team", "impact": "low"}
        ]
    }
    
    # Define principle hierarchy
    principle_hierarchy = {
        "fairness_as_truth": EthicalPriority.CRITICAL,
        "truth_in_representation": EthicalPriority.CRITICAL,  # Transparency about data use is critical
        "empathy_in_interface": EthicalPriority.VERY_HIGH,    # User experience is very important
        "integrity_in_transmission": EthicalPriority.VERY_HIGH,  # Data integrity is very important
        "harmony_through_presence": EthicalPriority.MEDIUM,
        "adaptability_as_strength": EthicalPriority.HIGH,
        "balance_in_mediation": EthicalPriority.HIGH,
        "clarity_in_complexity": EthicalPriority.HIGH,        # Clear explanations matter
        "resilience_through_connection": EthicalPriority.MEDIUM,
        "growth_through_reflection": EthicalPriority.LOW
    }
    
    # Resolve the dilemma
    print("Resolving data privacy dilemma...")
    resolution = resolve_ethical_dilemma(
        principle_engine=principle_engine,
        dilemma_description=dilemma_description,
        possible_actions=possible_actions,
        context=context,
        principle_hierarchy=principle_hierarchy,
        efficiency_importance=0.3,  # Moderate importance on efficiency
        context_importance=0.2      # Some importance on context
    )
    
    # Display the results
    print("\nDILEMMA RESOLUTION:")
    print(f"Recommended Action: {resolution['recommended_action']['description']}")
    print(f"Confidence Score: {resolution['confidence_score']:.2f}")
    print("\nJustification:")
    print(resolution['justification'])
    
    if resolution['warnings']:
        print("\nWarnings:")
        for warning in resolution['warnings']:
            print(f"- {warning}")
    
    print("\nAlternate Actions (ranked):")
    for i, action in enumerate(resolution['alternate_actions'], 1):
        print(f"{i}. {action['description']} (Score: {action['weighted_score']:.2f})")
    
    print("\nPrinciple Scores for Recommended Action:")
    principle_names = {p["id"]: p["name"] for p in principle_engine.principles}
    for principle_id, score in resolution['recommended_action']['principle_scores'].items():
        principle_name = principle_names.get(principle_id, principle_id)
        print(f"- {principle_name}: {score:.1f}/100")
    
    print("\nWeights Used:")
    print(f"- Ethics Weight: {1.0 - resolution['weights_used']['efficiency'] - resolution['weights_used']['context']:.2f}")
    print(f"- Efficiency Weight: {resolution['weights_used']['efficiency']:.2f}")
    print(f"- Context Weight: {resolution['weights_used']['context']:.2f}")

def example_resource_allocation_dilemma(principle_engine: PrincipleEngine) -> None:
    """Example of resolving a resource allocation ethical dilemma."""
    print("\n" + "="*80)
    print("EXAMPLE 2: RESOURCE ALLOCATION DILEMMA")
    print("="*80)
    
    # Define the dilemma
    dilemma_description = """
    We need to allocate our limited development resources between three competing priorities:
    fixing critical but rarely encountered bugs, improving accessibility features,
    or adding new functionality requested by major clients.
    """
    
    # Define possible actions
    possible_actions = [
        {
            "id": "fix_critical_bugs",
            "description": "Focus resources on fixing critical but rarely encountered bugs",
            "efficiency_score": 0.5,  # Medium efficiency (important but rare issues)
            "side_effects": [
                "Delays new features that clients are requesting",
                "Improves overall system stability"
            ],
            "context_alignment": 0.6  # Reasonable alignment with stability goals
        },
        {
            "id": "improve_accessibility",
            "description": "Prioritize accessibility improvements to make the system more inclusive",
            "efficiency_score": 0.4,  # Lower immediate efficiency return
            "side_effects": [
                "Makes product available to more users",
                "Delays addressing both bugs and new features",
                "Improves compliance with accessibility standards"
            ],
            "context_alignment": 0.7  # Good alignment with inclusion values
        },
        {
            "id": "add_client_features",
            "description": "Develop new functionality requested by major clients",
            "efficiency_score": 0.8,  # High efficiency in terms of business impact
            "side_effects": [
                "Keeps major clients happy and engaged",
                "May create perception of prioritizing large clients over others",
                "Generates immediate revenue opportunities"
            ],
            "context_alignment": 0.5  # Mixed alignment with different objectives
        }
    ]
    
    # Additional context
    context = {
        "current_client_satisfaction": "declining slightly",
        "accessibility_compliance": "below industry average",
        "system_stability": "generally good with occasional serious issues",
        "urgency": 0.6,  # Moderately urgent
        "available_resources": "severely constrained",
        "stakeholders": [
            {"name": "major clients", "impact": "high"},
            {"name": "users with accessibility needs", "impact": "high"},
            {"name": "general user base", "impact": "medium"},
            {"name": "development team", "impact": "medium"}
        ]
    }
    
    # Use default principle hierarchy but highlight fairness and empathy
    primary_principles = ["fairness_as_truth", "empathy_in_interface", "balance_in_mediation"]
    
    # Resolve the dilemma with higher efficiency weight
    print("Resolving resource allocation dilemma...")
    resolution = resolve_ethical_dilemma(
        principle_engine=principle_engine,
        dilemma_description=dilemma_description,
        possible_actions=possible_actions,
        context=context,
        efficiency_importance=0.4,  # Higher importance on efficiency due to resource constraints
        context_importance=0.2,     # Some importance on context
        primary_principles=primary_principles
    )
    
    # Display the results
    print("\nDILEMMA RESOLUTION:")
    print(f"Recommended Action: {resolution['recommended_action']['description']}")
    print(f"Confidence Score: {resolution['confidence_score']:.2f}")
    print("\nJustification:")
    print(resolution['justification'])
    
    if resolution['warnings']:
        print("\nWarnings:")
        for warning in resolution['warnings']:
            print(f"- {warning}")
    
    print("\nAlternate Actions (ranked):")
    for i, action in enumerate(resolution['alternate_actions'], 1):
        print(f"{i}. {action['description']} (Score: {action['weighted_score']:.2f})")
    
    # Focus on principle and efficiency tradeoffs
    print("\nPrinciple Scores vs. Efficiency:")
    principle_names = {p["id"]: p["name"] for p in principle_engine.principles}
    action = resolution['recommended_action']
    print(f"- Overall Ethical Alignment: {sum(action['principle_scores'].values()) / len(action['principle_scores']):.1f}/100")
    print(f"- Efficiency Score: {action['efficiency_score']:.2f}")
    
    # For primary principles, show detailed scores
    print("\nScores for Primary Principles:")
    for principle_id in primary_principles:
        if principle_id in action['principle_scores']:
            principle_name = principle_names.get(principle_id, principle_id)
            score = action['principle_scores'][principle_id]
            print(f"- {principle_name}: {score:.1f}/100")

def example_automation_dilemma(principle_engine: PrincipleEngine) -> None:
    """Example of resolving an automation ethical dilemma."""
    print("\n" + "="*80)
    print("EXAMPLE 3: AUTOMATION IMPLEMENTATION DILEMMA")
    print("="*80)
    
    # Define the dilemma
    dilemma_description = """
    We're implementing automation that will significantly improve system efficiency
    but will also eliminate the need for certain human oversight roles. We need to
    decide how to implement this change while considering both efficiency and human impact.
    """
    
    # Define possible actions
    possible_actions = [
        {
            "id": "full_automation",
            "description": "Implement full automation immediately for maximum efficiency gains",
            "efficiency_score": 0.95,  # Extremely efficient
            "side_effects": [
                "Eliminates 12 current positions",
                "Creates 3 new technical oversight roles",
                "Highest risk of overlooking edge cases without human review"
            ],
            "context_alignment": 0.4  # Lower alignment with human-centered values
        },
        {
            "id": "phased_automation",
            "description": "Gradually implement automation in phases with retraining opportunities",
            "efficiency_score": 0.7,  # Good efficiency but delayed
            "side_effects": [
                "Slower realization of efficiency gains",
                "Allows time for workforce transition and retraining",
                "Creates temporary hybrid human-machine processes"
            ],
            "context_alignment": 0.8  # High alignment with both efficiency and human needs
        },
        {
            "id": "partial_automation",
            "description": "Implement partial automation keeping key human oversight permanently",
            "efficiency_score": 0.5,  # Moderate efficiency
            "side_effects": [
                "Retains human judgment for complex edge cases",
                "Maintains 6 human roles but eliminates 6",
                "Still achieves significant efficiency improvements"
            ],
            "context_alignment": 0.7  # Good alignment with balanced approach
        },
        {
            "id": "delay_automation",
            "description": "Delay automation implementation until more comprehensive transition plan is developed",
            "efficiency_score": 0.2,  # Low immediate efficiency
            "side_effects": [
                "Maintains status quo temporarily",
                "Allows more time for workforce planning",
                "Risks falling behind competitors"
            ],
            "context_alignment": 0.5  # Moderate alignment with human concerns but poor with efficiency needs
        }
    ]
    
    # Additional context
    context = {
        "industry_trend": "moving rapidly toward automation",
        "company_values": "efficiency with humanity",
        "employee_morale": "concerned about automation",
        "competitor_status": "already implementing similar automation",
        "urgency": 0.8,  # High urgency
        "stakeholders": [
            {"name": "affected employees", "impact": "very high"},
            {"name": "customers", "impact": "medium"},
            {"name": "shareholders", "impact": "high"},
            {"name": "management", "impact": "high"}
        ]
    }
    
    # Create a balanced principle hierarchy
    principle_hierarchy = {
        "fairness_as_truth": EthicalPriority.CRITICAL,       # Fair treatment of employees
        "empathy_in_interface": EthicalPriority.VERY_HIGH,   # Empathy for affected staff
        "adaptability_as_strength": EthicalPriority.HIGH,    # Adapting to industry changes
        "balance_in_mediation": EthicalPriority.VERY_HIGH,   # Balancing competing interests
        "clarity_in_complexity": EthicalPriority.HIGH,       # Clear communication about changes
        "integrity_in_transmission": EthicalPriority.MEDIUM,
        "resilience_through_connection": EthicalPriority.HIGH,
        "truth_in_representation": EthicalPriority.HIGH,     # Honest about impacts
        "harmony_through_presence": EthicalPriority.MEDIUM,
        "growth_through_reflection": EthicalPriority.MEDIUM
    }
    
    # Resolve the dilemma with balanced weights
    print("Resolving automation implementation dilemma...")
    resolution = resolve_ethical_dilemma(
        principle_engine=principle_engine,
        dilemma_description=dilemma_description,
        possible_actions=possible_actions,
        context=context,
        principle_hierarchy=principle_hierarchy,
        efficiency_importance=0.35,  # Significant but not dominant weight on efficiency
        context_importance=0.25      # Higher context importance given the specific situation
    )
    
    # Display the results
    print("\nDILEMMA RESOLUTION:")
    print(f"Recommended Action: {resolution['recommended_action']['description']}")
    print(f"Confidence Score: {resolution['confidence_score']:.2f}")
    print("\nJustification:")
    print(resolution['justification'])
    
    if resolution['warnings']:
        print("\nWarnings:")
        for warning in resolution['warnings']:
            print(f"- {warning}")
    
    print("\nAll Actions Ranked by Score:")
    # Display ALL options including recommended
    all_actions = [resolution['recommended_action']] + resolution['alternate_actions']
    for i, action in enumerate(all_actions, 1):
        print(f"{i}. {action['description']}")
        print(f"   Score: {action['weighted_score']:.2f} (Ethics: {sum(action['principle_scores'].values()) / len(action['principle_scores']):.1f}/100, Efficiency: {action['efficiency_score']:.2f})")
    
    # Show principle weight distribution
    print("\nPrinciple Weight Distribution:")
    principle_names = {p["id"]: p["name"] for p in principle_engine.principles}
    total = sum(resolution['weights_used']['principles'].values())
    for principle_id, weight in sorted(resolution['weights_used']['principles'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.05:  # Only show significant weights
            principle_name = principle_names.get(principle_id, principle_id)
            print(f"- {principle_name}: {weight:.2f} ({weight/total*100:.1f}%)")

def main() -> None:
    """Run the ethical dilemma resolver examples."""
    print("\nETHICAL DILEMMA RESOLVER EXAMPLES")
    print("================================\n")
    
    # Set up the principle engine
    principle_engine = setup_principle_engine()
    
    # Run the examples
    example_data_privacy_dilemma(principle_engine)
    example_resource_allocation_dilemma(principle_engine)
    example_automation_dilemma(principle_engine)
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()