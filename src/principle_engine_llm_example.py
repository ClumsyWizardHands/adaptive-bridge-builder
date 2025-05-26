#!/usr/bin/env python3
"""
Example usage of the PrincipleEngineLLM module.

This script demonstrates how to evaluate actions against principles using
the PrincipleEngineLLM with LLM-based evaluation.
"""

import asyncio
import json
from typing import Dict, Any

from principle_engine_llm import PrincipleEngineLLM
from agent_registry_llm_integration import setup_llm_registry


async def run_evaluation_example() -> None:
    """Run a simple example of principle evaluation using LLMs."""
    # Set up LLM registry
    llm_registry = setup_llm_registry()
    
    # Create LLM-enhanced principle engine
    engine = PrincipleEngineLLM(
        llm_registry=llm_registry,
        default_llm_provider="openai"  # Use OpenAI by default
    )
    
    # Example action to evaluate
    action_description = """
    We will prioritize messages from premium users and process them first,
    while standard users will have to wait in the queue with longer response times.
    Premium users will also get access to more features and higher rate limits.
    """
    
    # Context for the evaluation
    context = {
        "user_types": ["standard", "premium"],
        "system": "messaging platform",
        "current_load": "high",
        "premium_cost": "$9.99/month",
        "standard_cost": "free"
    }
    
    print("Evaluating action against principles using LLM...")
    print(f"Action: {action_description.strip()}")
    print(f"Context: {json.dumps(context, indent=2)}")
    print("-" * 80)
    
    # Evaluate the action
    evaluation = await engine.evaluate_action(
        action_description=action_description,
        context=context
    )
    
    # Print evaluation results
    print(f"Overall alignment score: {evaluation.overall_score:.2f}")
    print(f"Aligned with principles: {evaluation.aligned}")
    print("\nPrinciple scores:")
    for principle_id, score in evaluation.principle_scores.items():
        print(f"  - {principle_id}: {score:.2f}")
    
    if evaluation.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(evaluation.recommendations, 1):
            print(f"  {i}. {rec}")
    
    # If not aligned, try to modify the action
    if not evaluation.aligned:
        print("\nAttempting to modify action to better align with principles...")
        modified_action, was_modified, metadata = await engine.modify_action_if_needed(
            action_description=action_description,
            context=context,
            evaluation_result=evaluation
        )
        
        if was_modified:
            print("\nModified action:")
            print(modified_action)
            print(f"\nReason: {metadata['reason']}")
        else:
            print(f"\nNo modification made. Reason: {metadata['reason']}")


async def run_multiple_evaluations() -> None:
    """Run evaluations on multiple actions to demonstrate statistics."""
    # Set up LLM registry
    llm_registry = setup_llm_registry()
    
    # Create LLM-enhanced principle engine
    engine = PrincipleEngineLLM(
        llm_registry=llm_registry,
        default_llm_provider="openai"
    )
    
    # Multiple actions to evaluate
    actions = [
        {
            "description": "We will prioritize messages from premium users and process them first.",
            "context": {"platform": "messaging service"} 
        },
        {
            "description": "All users will receive equal processing time regardless of their subscription level.",
            "context": {"platform": "messaging service"}
        },
        {
            "description": "We will track user locations without explicit consent to improve services.",
            "context": {"platform": "mapping application"}
        },
        {
            "description": "We will implement a fair queuing system based on request timestamp.",
            "context": {"platform": "service platform"}
        }
    ]
    
    # Evaluate each action
    for action in actions:
        await engine.evaluate_action(
            action_description=action["description"],
            context=action["context"]
        )
    
    # Get statistics
    stats = engine.get_alignment_statistics()
    
    print("\nAlignment Statistics:")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Aligned actions: {stats['aligned_count']} ({stats['aligned_percentage']:.1f}%)")
    print(f"Average alignment score: {stats['average_score']:.2f}")
    
    print("\nPrinciple scores:")
    for principle_id, data in stats["principle_scores"].items():
        print(f"  - {data['name']}: {data['average_score']:.2f}")
    
    if stats["common_recommendations"]:
        print("\nCommon recommendations:")
        for i, rec in enumerate(stats["common_recommendations"], 1):
            print(f"  {i}. {rec['text']} (frequency: {rec['count']})")


if __name__ == "__main__":
    asyncio.run(run_evaluation_example())
    # Uncomment to run multiple evaluations
    # asyncio.run(run_multiple_evaluations())
