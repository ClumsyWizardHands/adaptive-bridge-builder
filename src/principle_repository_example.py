#!/usr/bin/env python3
"""
Principle Repository Example

This module demonstrates how to use the PrincipleRepository class to manage
principles in the database. It shows common operations like creating, retrieving,
updating, and deleting principles and related data.
"""

import json
import os
import logging
from typing import Dict, Any, List

from principle_repository import PrincipleRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleRepositoryExample")


def create_sample_principles(repo: PrincipleRepository) -> None:
    """
    Create sample principles in the database.
    
    Args:
        repo: The PrincipleRepository instance
    """
    print("\n=== Creating Sample Principles ===")
    
    # Get existing categories and importance levels
    categories = repo.get_categories()
    importance_levels = repo.get_importance_levels()
    
    if not categories:
        print("No categories found in database. Initial schema may not be loaded.")
        return
    
    if not importance_levels:
        print("No importance levels found in database. Initial schema may not be loaded.")
        return
    
    # Find category and importance level IDs
    ethical_category = next((c for c in categories if c["name"] == "Ethical"), None)
    epistemic_category = next((c for c in categories if c["name"] == "Epistemic"), None)
    relational_category = next((c for c in categories if c["name"] == "Relational"), None)
    
    critical_importance = next((i for i in importance_levels if i["name"] == "Critical"), None)
    high_importance = next((i for i in importance_levels if i["name"] == "High"), None)
    
    # Create fairness as truth principle
    fairness_principle_id = repo.create_principle(
        name="Fairness as Truth",
        short_name="fairness_as_truth",
        description="The principle of fairness as truth requires that all agents be treated with "
                  "equal consideration and that their interests be given equal weight. Truth "
                  "demands we recognize the inherent value of all perspectives.",
        category_id=ethical_category["id"],
        importance_level_id=critical_importance["id"],
        tags=["ethical", "fairness", "truth", "equality"],
        is_active=True,
        user_id="system"
    )
    print(f"Created 'Fairness as Truth' principle with ID {fairness_principle_id}")
    
    # Add evaluation criteria for fairness principle
    fairness_criteria_id = repo.add_evaluation_criteria(
        principle_id=fairness_principle_id,
        type_id=1,  # LLM Prompt type
        content="""
You are evaluating whether an action aligns with the principle of "Fairness as Truth."

Fairness as Truth requires that:
1. All agents are treated with equal consideration
2. All perspectives are given equal weight (unless there is a compelling reason not to)
3. No bias or favoritism is shown toward any specific agent or perspective
4. Decisions are made based on objective criteria rather than arbitrary distinctions

Consider the following action:
{action_description}

Context:
{context}

Evaluate whether this action aligns with the principle of Fairness as Truth.
Your response should include:
1. An overall alignment score between 0.0 and 1.0, where 1.0 is perfect alignment
2. A brief explanation of your reasoning
3. Specific aspects of the action that align or misalign with the principle
4. Recommendations for how the action could better align with the principle

Format your response as a JSON object with the following keys:
{
  "overall_score": (float between 0.0 and 1.0),
  "reasoning": (string),
  "alignment_aspects": (array of strings),
  "misalignment_aspects": (array of strings),
  "recommendations": (array of strings)
}
""",
        parameters={
            "minimum_score_threshold": 0.7,
            "consider_context_weight": 0.8,
            "action_weight": 1.0
        },
        user_id="system"
    )
    print(f"Added evaluation criteria (ID: {fairness_criteria_id}) to fairness principle")
    
    # Create clarity in complexity principle
    clarity_principle_id = repo.create_principle(
        name="Clarity in Complexity",
        short_name="clarity_in_complexity",
        description="The principle of clarity in complexity recognizes that while the world "
                   "is inherently complex, our communication should strive for clarity. "
                   "We value precision and understandability even when addressing complex topics.",
        category_id=epistemic_category["id"],
        importance_level_id=high_importance["id"],
        tags=["epistemic", "clarity", "communication", "understanding"],
        is_active=True,
        user_id="system"
    )
    print(f"Created 'Clarity in Complexity' principle with ID {clarity_principle_id}")
    
    # Add evaluation criteria for clarity principle
    clarity_criteria_id = repo.add_evaluation_criteria(
        principle_id=clarity_principle_id,
        type_id=1,  # LLM Prompt type
        content="""
You are evaluating whether an action aligns with the principle of "Clarity in Complexity."

Clarity in Complexity requires that:
1. Communication is clear and understandable, even when addressing complex topics
2. Precision in language is maintained without overwhelming with unnecessary details
3. Technical concepts are explained in accessible terms when appropriate
4. Complex ideas are structured logically to aid understanding

Consider the following action:
{action_description}

Context:
{context}

Evaluate whether this action aligns with the principle of Clarity in Complexity.
Your response should include:
1. An overall alignment score between 0.0 and 1.0, where 1.0 is perfect alignment
2. A brief explanation of your reasoning
3. Specific aspects of the action that align or misalign with the principle
4. Recommendations for how the action could better align with the principle

Format your response as a JSON object with the following keys:
{
  "overall_score": (float between 0.0 and 1.0),
  "reasoning": (string),
  "alignment_aspects": (array of strings),
  "misalignment_aspects": (array of strings),
  "recommendations": (array of strings)
}
""",
        parameters={
            "minimum_score_threshold": 0.7,
            "consider_audience": True
        },
        user_id="system"
    )
    print(f"Added evaluation criteria (ID: {clarity_criteria_id}) to clarity principle")
    
    # Create harmony through presence principle
    harmony_principle_id = repo.create_principle(
        name="Harmony Through Presence",
        short_name="harmony_through_presence",
        description="The principle of harmony through presence values being fully engaged "
                    "and attentive in interactions. True harmony comes from authentic presence "
                    "and deep listening rather than mere conflict avoidance.",
        category_id=relational_category["id"],
        importance_level_id=high_importance["id"],
        tags=["relational", "harmony", "presence", "authenticity"],
        is_active=True,
        user_id="system"
    )
    print(f"Created 'Harmony Through Presence' principle with ID {harmony_principle_id}")
    
    # Assign principles to decision points
    decision_points = repo.get_decision_points()
    
    # Find decision points by name
    a2a_decision_point = next((dp for dp in decision_points if dp["name"] == "a2a_task_response_generation"), None)
    orchestrator_decision_point = next((dp for dp in decision_points if dp["name"] == "orchestrator_task_assignment"), None)
    conflict_decision_point = next((dp for dp in decision_points if dp["name"] == "conflict_resolution_generation"), None)
    
    if a2a_decision_point:
        # Assign fairness principle to A2A task response decision point
        repo.assign_principle_to_decision_point(
            principle_id=fairness_principle_id,
            decision_point_id=a2a_decision_point["id"],
            alignment_threshold=0.7,
            priority=100,
            user_id="system"
        )
        print(f"Assigned 'Fairness as Truth' principle to A2A task response decision point")
        
        # Assign clarity principle to A2A task response decision point
        repo.assign_principle_to_decision_point(
            principle_id=clarity_principle_id,
            decision_point_id=a2a_decision_point["id"],
            alignment_threshold=0.7,
            priority=80,
            user_id="system"
        )
        print(f"Assigned 'Clarity in Complexity' principle to A2A task response decision point")
    
    if conflict_decision_point:
        # Assign harmony principle to conflict resolution decision point
        repo.assign_principle_to_decision_point(
            principle_id=harmony_principle_id,
            decision_point_id=conflict_decision_point["id"],
            alignment_threshold=0.8,
            priority=100,
            user_id="system"
        )
        print(f"Assigned 'Harmony Through Presence' principle to conflict resolution decision point")


def query_and_display_principles(repo: PrincipleRepository) -> None:
    """
    Query and display principles from the database.
    
    Args:
        repo: The PrincipleRepository instance
    """
    print("\n=== Querying Principles ===")
    
    # Get all active principles
    principles = repo.get_principles(is_active=True)
    print(f"Found {len(principles)} active principles:")
    
    for p in principles:
        print(f"  - {p['name']} ({p['short_name'] or 'no short name'}) "
              f"[Category: {p['category_name']}, Importance: {p['importance_level_name']}]")
        print(f"    Tags: {', '.join(tag['name'] for tag in p['tags'])}")
    
    # Get a specific principle with all details
    principle = repo.get_principle(principles[0]["id"])
    
    print(f"\nDetailed view of '{principle['name']}' principle:")
    print(f"  Description: {principle['description']}")
    print(f"  Category: {principle['category_name']}")
    print(f"  Importance: {principle['importance_level_name']} (Value: {principle['importance_value']})")
    print(f"  Tags: {', '.join(tag['name'] for tag in principle['tags'])}")
    
    print("  Evaluation Criteria:")
    for ec in principle["evaluation_criteria"]:
        print(f"    - Type: {ec['type_name']}")
        print(f"      Content excerpt: {ec['content'][:100]}...")
        
    print("  Decision Points:")
    for dp in principle["decision_points"]:
        print(f"    - {dp['name']} (Component: {dp['component']})")
        print(f"      Alignment Threshold: {dp['alignment_threshold']}, Priority: {dp['priority']}")


def update_principle_example(repo: PrincipleRepository) -> None:
    """
    Demonstrate updating a principle.
    
    Args:
        repo: The PrincipleRepository instance
    """
    print("\n=== Updating a Principle ===")
    
    # Get principles to find one to update
    principles = repo.get_principles()
    if not principles:
        print("No principles to update.")
        return
    
    # Choose the clarity principle
    clarity_principle = next((p for p in principles if p["short_name"] == "clarity_in_complexity"), None)
    if not clarity_principle:
        print("Clarity principle not found.")
        return
    
    principle_id = clarity_principle["id"]
    
    # Update the principle description
    updated_description = (
        "The principle of clarity in complexity recognizes that while the world "
        "is inherently complex, our communication should strive for clarity and accessibility. "
        "We value precision, understandability, and structured presentation even when "
        "addressing highly complex or technical topics."
    )
    
    success = repo.update_principle(
        principle_id=principle_id,
        description=updated_description,
        user_id="system"
    )
    
    if success:
        print(f"Successfully updated description of principle ID {principle_id}")
        
        # Add new tags
        repo.manage_principle_tags(
            principle_id=principle_id,
            tags_to_add=["accessibility", "structure"],
            user_id="system"
        )
        print(f"Added new tags to principle ID {principle_id}")
    else:
        print(f"Failed to update principle ID {principle_id}")
    
    # Fetch and display the updated principle
    updated_principle = repo.get_principle(principle_id)
    print(f"\nUpdated '{updated_principle['name']}' principle:")
    print(f"  Description: {updated_principle['description']}")
    print(f"  Tags: {', '.join(tag['name'] for tag in updated_principle['tags'])}")


def print_section_break() -> None:
    """Print a section break for better readability."""
    print("\n" + "="*80)


def main() -> None:
    """
    Main function to demonstrate the PrincipleRepository class.
    """
    print("Principle Repository Example")
    print_section_break()
    
    # Database path
    db_path = "principles.db"
    
    # Remove existing database file to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}")
    
    # Create repository instance
    repo = PrincipleRepository(db_path)
    print(f"Initialized PrincipleRepository with database at {db_path}")
    
    # Create sample principles
    create_sample_principles(repo)
    print_section_break()
    
    # Query and display principles
    query_and_display_principles(repo)
    print_section_break()
    
    # Update principle example
    update_principle_example(repo)
    print_section_break()
    
    print("Principle Repository Example completed successfully.")


if __name__ == "__main__":
    main()
