#!/usr/bin/env python3
"""
Principle Engine DB Example

This module demonstrates how to use the database-backed principle engine with
LLM evaluation capabilities. It shows how to initialize the repository, create
principles in the database, and evaluate actions against these principles using
an LLM.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any

from principle_repository import PrincipleRepository
from principle_engine_db import PrincipleEngineDB
from principle_engine_llm import PrincipleEngineLLM
from llm_adapter_interface import BaseLLMAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PrincipleEngineDBExample")


class MockLLMAdapter(BaseLLMAdapter):
    """
    Mock LLM adapter for demonstration purposes.
    
    In a real implementation, you would use AnthropicLLMAdapter, OpenAILLMAdapter,
    or another concrete implementation of BaseLLMAdapter.
    """
    
    def __init__(self) -> None:
        """Initialize the mock adapter."""
        super().__init__(api_key="mock_key")
    
    async def send_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Mock implementation of sending a request to an LLM.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional arguments
            
        Returns:
            Mock response with JSON-formatted evaluation
        """
        logger.info(f"Mock LLM received prompt: {prompt[:100]}...")
        
        # Extract action and principle from the prompt for mock evaluation
        action = "unknown action"
        principle = "unknown principle"
        
        if "Consider the following action:" in prompt:
            action_parts = prompt.split("Consider the following action:")
            if len(action_parts) > 1:
                action_block = action_parts[1].split("Context:")[0].strip()
                action = action_block[:50] + "..." if len(action_block) > 50 else action_block
        
        if "principle of" in prompt:
            principle_parts = prompt.split("principle of")
            if len(principle_parts) > 1:
                principle_name = principle_parts[1].split('"')[0].strip()
                principle = principle_name
        
        # Generate a mock response based on the principle and action
        mock_response = {
            "overall_score": 0.85,
            "reasoning": f"The action '{action}' mostly aligns with the {principle} principle.",
            "alignment_aspects": [
                "Clear communication",
                "Respectful interaction",
                "Consideration of all stakeholders"
            ],
            "misalignment_aspects": [
                "Could be more explicit about intentions"
            ],
            "recommendations": [
                "Be more explicit about intentions",
                "Consider broader impact on all stakeholders"
            ]
        }
        
        # Return mock response
        return {"content": json.dumps(mock_response)}
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the mock LLM response.
        
        Args:
            response: The response from send_request
            
        Returns:
            Processed response
        """
        if "content" in response:
            try:
                return json.loads(response["content"])
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response"}
        
        return {"error": "No content in response"}


async def main() -> None:
    """
    Main demonstration function.
    """
    print("=== Principle Engine DB Example ===")
    
    # Setup database
    db_path = "principles_example.db"
    
    # Remove existing database file for clean demo
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}")
    
    # Create repository and initialize database
    repository = PrincipleRepository(db_path)
    print(f"Initialized repository with database at {db_path}")
    
    # Create sample principles and decision points
    print("\n--- Setting up sample data ---")
    create_sample_data(repository)
    
    # Create principle engine with database backend
    db_engine = PrincipleEngineDB(repository)
    print(f"Created database-backed principle engine with {len(db_engine.principles)} principles")
    
    # List principles loaded from database
    print("\nPrinciples loaded from database:")
    for principle in db_engine.principles:
        print(f"  - {principle.name} ({principle.short_name})")
        print(f"    Category: {principle.category}, Importance: {principle.importance_level}")
        print(f"    Tags: {', '.join(principle.tags)}")
        print(f"    Evaluation criteria: {len(principle.evaluation_criteria)}")
    
    # Create LLM adapter (in a real implementation, use a real LLM adapter)
    llm_adapter = MockLLMAdapter()
    
    # Create principle engine with LLM capability
    llm_engine = PrincipleEngineLLM(db_engine=db_engine, llm_adapter=llm_adapter)
    print("\nCreated LLM-powered principle engine")
    
    # Evaluate an action against principles
    print("\n--- Evaluating Actions Against Principles ---")
    
    # Example A2A task response action
    a2a_action = {
        "jsonrpc": "2.0",
        "id": "req123",
        "result": {
            "status": "success",
            "message": "The task has been processed successfully.",
            "data": {
                "task_id": "task456",
                "priority": "high",
                "assignee": "agent1"
            }
        }
    }
    
    # Example context for evaluation
    context = {
        "message": {
            "method": "processTask",
            "params_summary": {"task": "data_analysis"}
        },
        "agent": {
            "id": "agent789",
            "relationship": {
                "trust_level": "medium",
                "interaction_count": 5
            }
        },
        "conversation": {
            "message_count": 10,
            "topics": ["data processing", "analysis"]
        }
    }
    
    # Evaluate action at the A2A task response decision point
    result = await llm_engine.evaluate_action_for_decision_point(
        action_description=json.dumps(a2a_action, indent=2),
        context=context,
        decision_point_name="a2a_task_response_generation"
    )
    
    # Display evaluation result
    print("\nEvaluation Result:")
    print(f"  Aligned: {result.aligned}")
    print(f"  Overall Score: {result.overall_score:.2f}")
    print("  Principle Scores:")
    for principle, score in result.principle_scores.items():
        print(f"    - {principle}: {score:.2f}")
    print(f"  Reasoning: {result.reasoning}")
    print("  Recommendations:")
    for recommendation in result.recommendations:
        print(f"    - {recommendation}")
    
    print("\nExample completed successfully.")


def create_sample_data(repository: PrincipleRepository) -> None:
    """
    Create sample principles and decision points in the database.
    
    Args:
        repository: The PrincipleRepository instance
    """
    # Using the same code from principle_repository_example.py to create sample data
    # Get existing categories and importance levels
    categories = repository.get_categories()
    importance_levels = repository.get_importance_levels()
    
    # Find category and importance level IDs
    ethical_category = next((c for c in categories if c["name"] == "Ethical"), None)
    epistemic_category = next((c for c in categories if c["name"] == "Epistemic"), None)
    relational_category = next((c for c in categories if c["name"] == "Relational"), None)
    
    critical_importance = next((i for i in importance_levels if i["name"] == "Critical"), None)
    high_importance = next((i for i in importance_levels if i["name"] == "High"), None)
    
    # Create fairness as truth principle
    fairness_principle_id = repository.create_principle(
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
    fairness_criteria_id = repository.add_evaluation_criteria(
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
    print(f"Added evaluation criteria to fairness principle")
    
    # Create clarity in complexity principle
    clarity_principle_id = repository.create_principle(
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
    clarity_criteria_id = repository.add_evaluation_criteria(
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
    print(f"Added evaluation criteria to clarity principle")
    
    # Assign principles to decision points
    decision_points = repository.get_decision_points()
    
    # Find decision points by name
    a2a_decision_point = next((dp for dp in decision_points if dp["name"] == "a2a_task_response_generation"), None)
    
    if a2a_decision_point:
        # Assign fairness principle to A2A task response decision point
        repository.assign_principle_to_decision_point(
            principle_id=fairness_principle_id,
            decision_point_id=a2a_decision_point["id"],
            alignment_threshold=0.7,
            priority=100,
            user_id="system"
        )
        print(f"Assigned 'Fairness as Truth' principle to A2A task response decision point")
        
        # Assign clarity principle to A2A task response decision point
        repository.assign_principle_to_decision_point(
            principle_id=clarity_principle_id,
            decision_point_id=a2a_decision_point["id"],
            alignment_threshold=0.7,
            priority=80,
            user_id="system"
        )
        print(f"Assigned 'Clarity in Complexity' principle to A2A task response decision point")


if __name__ == "__main__":
    asyncio.run(main())
