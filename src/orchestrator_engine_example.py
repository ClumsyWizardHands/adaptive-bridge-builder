"""
Orchestrator Engine Example

This module demonstrates how to use the OrchestratorEngine for coordinating tasks across
multiple specialized agents. It provides concrete examples of task decomposition, dependency
management, scheduling, and error recovery.
"""

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from collaborative_task_handler import Task, TaskStatus, TaskPriority, TaskCoordinator
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability,
    DependencyType, TaskDecompositionStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OrchestratorExample")


def setup_orchestrator() -> OrchestratorEngine:
    """
    Set up and configure an OrchestratorEngine instance with multiple agents.
    """
    # Create the orchestrator
    orchestrator = OrchestratorEngine(
        agent_id="orchestrator-agent",
        storage_dir="data/orchestration"
    )
    
    # Register agents with different specializations
    
    # Data analysis agent
    orchestrator.register_agent(
        agent_id="analysis-agent",
        roles=[AgentRole.ANALYZER],
        capabilities=["data_analysis", "statistics", "visualization"],
        specialization={
            TaskType.ANALYSIS: 0.9,
            TaskType.VALIDATION: 0.7,
            TaskType.RESEARCH: 0.6,
            TaskType.AGGREGATION: 0.5,
        },
        max_load=3
    )
    
    # Content generation agent
    orchestrator.register_agent(
        agent_id="content-agent",
        roles=[AgentRole.GENERATOR],
        capabilities=["text_generation", "summarization", "paraphrasing"],
        specialization={
            TaskType.GENERATION: 0.9,
            TaskType.TRANSFORMATION: 0.7,
            TaskType.COMMUNICATION: 0.8,
        },
        max_load=5
    )
    
    # Research agent
    orchestrator.register_agent(
        agent_id="research-agent",
        roles=[AgentRole.RESEARCHER],
        capabilities=["web_search", "information_gathering", "fact_checking"],
        specialization={
            TaskType.RESEARCH: 0.9,
            TaskType.EXTRACTION: 0.8,
            TaskType.VALIDATION: 0.6,
        },
        max_load=2
    )
    
    # Code execution agent
    orchestrator.register_agent(
        agent_id="executor-agent",
        roles=[AgentRole.EXECUTOR],
        capabilities=["code_execution", "api_integration", "database_queries"],
        specialization={
            TaskType.EXECUTION: 0.9,
            TaskType.MONITORING: 0.7,
            TaskType.VALIDATION: 0.5,
        },
        max_load=4
    )
    
    # Validation agent
    orchestrator.register_agent(
        agent_id="validator-agent",
        roles=[AgentRole.VALIDATOR],
        capabilities=["fact_checking", "code_review", "quality_assurance"],
        specialization={
            TaskType.VALIDATION: 0.9,
            TaskType.ANALYSIS: 0.6,
            TaskType.MONITORING: 0.7,
        },
        max_load=3
    )
    
    # Domain specialist agent (e.g., financial domain)
    orchestrator.register_agent(
        agent_id="finance-specialist",
        roles=[AgentRole.SPECIALIST],
        capabilities=["financial_analysis", "market_prediction", "risk_assessment"],
        specialization={
            TaskType.ANALYSIS: 0.8,
            TaskType.DECISION: 0.9,
            TaskType.VALIDATION: 0.7,
        },
        max_load=2
    )
    
    return orchestrator


def example_task_decomposition(orchestrator: OrchestratorEngine) -> str:
    """
    Example of how to decompose a complex task into subtasks.
    
    Args:
        orchestrator: The OrchestratorEngine instance
        
    Returns:
        ID of the decomposed task
    """
    # Create a complex task
    task = orchestrator.task_coordinator.create_task(
        title="Market Analysis Report Creation",
        description=(
            "Create a comprehensive market analysis report for the tech sector, "
            "including data analysis, trend research, content creation, and validation."
        ),
        required_capabilities=["research", "analysis", "content_creation"],
        priority=TaskPriority.HIGH,
        metadata={
            "domain": "finance",
            "sector": "technology",
            "requested_by": "client-123",
            "deadline": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        }
    )
    
    # Decompose the task using a functional strategy
    decomposed = orchestrator.decompose_task(
        task=task,
        strategy=TaskDecompositionStrategy.FUNCTIONAL,
        metadata={
            "complexity": "high",
            "estimated_total_duration": 240,  # minutes
        }
    )
    
    if not decomposed:
        logger.error("Failed to decompose task")
        return None
    
    # Add subtasks with dependencies
    
    # 1. Initial data gathering
    data_gathering = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Gather Market Data",
        description=(
            "Collect current market data for the tech sector including stock prices, "
            "market cap, P/E ratios, and recent news for major companies."
        ),
        task_type=TaskType.RESEARCH,
        required_capabilities=["web_search", "information_gathering"],
        dependencies=None,  # No dependencies
        estimated_duration=30,
        priority=TaskPriority.HIGH,
        metadata={"data_sources": ["financial_apis", "news_sources", "sec_filings"]}
    )
    
    # 2. Data analysis
    data_analysis = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Analyze Market Data",
        description=(
            "Analyze the collected market data to identify trends, correlations, "
            "and potential opportunities or risks."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["data_analysis", "statistics"],
        dependencies=[(data_gathering.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=45,
        priority=TaskPriority.HIGH,
        metadata={"analysis_methods": ["trend_analysis", "statistical_tests", "comparative_analysis"]}
    )
    
    # 3. Trend research (can start in parallel with data analysis)
    trend_research = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Research Industry Trends",
        description=(
            "Research current and emerging trends in the tech industry, including "
            "technological innovations, regulatory changes, and market shifts."
        ),
        task_type=TaskType.RESEARCH,
        required_capabilities=["web_search", "information_gathering"],
        dependencies=[(data_gathering.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=40,
        priority=TaskPriority.MEDIUM,
        metadata={"research_focus": ["innovation", "regulation", "market_sentiment"]}
    )
    
    # 4. Competitor analysis
    competitor_analysis = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Analyze Competitors",
        description=(
            "Analyze major competitors in the tech sector, their market positions, "
            "recent strategic moves, and financial performance."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["data_analysis", "financial_analysis"],
        dependencies=[(data_gathering.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=35,
        priority=TaskPriority.MEDIUM,
        metadata={"competitors": ["major_tech_companies", "emerging_startups"]}
    )
    
    # 5. Financial projections (depends on data analysis)
    financial_projections = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Create Financial Projections",
        description=(
            "Develop financial projections for the tech sector based on historical data "
            "and current trends, with multiple scenario analyses."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["financial_analysis", "market_prediction"],
        dependencies=[(data_analysis.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=40,
        priority=TaskPriority.HIGH,
        metadata={"projection_period": "12_months", "scenarios": ["baseline", "optimistic", "pessimistic"]}
    )
    
    # 6. Report drafting (depends on all analyses)
    report_drafting = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Draft Market Analysis Report",
        description=(
            "Create a comprehensive draft report integrating all analyses, findings, "
            "and projections with appropriate sections and visualizations."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["text_generation", "summarization"],
        dependencies=[
            (data_analysis.task_id, DependencyType.SEQUENTIAL),
            (trend_research.task_id, DependencyType.SEQUENTIAL),
            (competitor_analysis.task_id, DependencyType.SEQUENTIAL),
            (financial_projections.task_id, DependencyType.SEQUENTIAL)
        ],
        estimated_duration=60,
        priority=TaskPriority.HIGH,
        metadata={"report_format": "pdf", "sections": ["executive_summary", "market_overview", "trends", "competitive_landscape", "financial_analysis", "recommendations"]}
    )
    
    # 7. Validation and fact-checking
    validation = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Validate Report Content",
        description=(
            "Validate all facts, figures, and conclusions in the draft report for "
            "accuracy, consistency, and completeness."
        ),
        task_type=TaskType.VALIDATION,
        required_capabilities=["fact_checking", "quality_assurance"],
        dependencies=[(report_drafting.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=30,
        priority=TaskPriority.CRITICAL,
        metadata={"validation_aspects": ["factual_accuracy", "methodology_soundness", "logical_consistency"]}
    )
    
    # 8. Final report (depends on validation)
    final_report = orchestrator.add_subtask_to_decomposition(
        decomposed_task_id=task.task_id,
        title="Finalize Market Analysis Report",
        description=(
            "Incorporate validation feedback and finalize the market analysis report "
            "with executive summary, key findings, and visual elements."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["text_generation", "summarization"],
        dependencies=[(validation.task_id, DependencyType.SEQUENTIAL)],
        estimated_duration=30,
        priority=TaskPriority.CRITICAL,
        metadata={"deliverable_format": "pdf", "includes_presentation": True}
    )
    
    logger.info(f"Decomposed task {task.task_id} into 8 subtasks with dependencies")
    
    # Assign tasks to agents based on capabilities
    orchestrator.assign_tasks_to_agents(task.task_id)
    
    # Start orchestration process
    orchestrator.start_orchestration()
    
    return task.task_id


def example_error_recovery(orchestrator: OrchestratorEngine, task_id: str) -> None:
    """
    Example of how to implement error recovery strategies.
    
    Args:
        orchestrator: The OrchestratorEngine instance
        task_id: ID of the task to set up recovery for
    """
    # Get the decomposed task
    decomposed = orchestrator.decomposed_tasks.get(task_id)
    if not decomposed:
        logger.error(f"Decomposed task {task_id} not found")
        return
    
    # Register recovery strategies for each subtask
    for subtask_id, subtask in decomposed.subtasks.items():
        # For research and data gathering tasks - use retry with alternate sources
        if subtask.metadata.get("task_type") == TaskType.RESEARCH.name:
            orchestrator.register_recovery_strategy(
                subtask_id,
                {
                    "type": "retry",
                    "max_attempts": 3,
                    "backoff": 10,  # seconds
                    "alternate_sources": ["secondary_data_provider", "academic_databases"]
                }
            )
        
        # For analysis tasks - use reassignment to different agents
        elif subtask.metadata.get("task_type") == TaskType.ANALYSIS.name:
            orchestrator.register_recovery_strategy(
                subtask_id,
                {
                    "type": "reassign",
                    "max_attempts": 2,
                    "fallback_agents": ["finance-specialist"],  # domain expert as fallback
                    "simplified_requirements": True
                }
            )
        
        # For generation tasks - use simplification
        elif subtask.metadata.get("task_type") == TaskType.GENERATION.name:
            orchestrator.register_recovery_strategy(
                subtask_id,
                {
                    "type": "simplify",
                    "max_attempts": 2,
                    "simplification": {
                        "simplified_description": f"Simplified version of: {subtask.description}",
                        "reduced_capabilities": ["text_generation"]
                    }
                }
            )
        
        # For validation tasks - use fallback result with manual review flag
        elif subtask.metadata.get("task_type") == TaskType.VALIDATION.name:
            orchestrator.register_recovery_strategy(
                subtask_id,
                {
                    "type": "fallback",
                    "fallback_result": {
                        "validation_status": "pending_manual_review",
                        "validation_confidence": 0.0,
                        "requires_human_review": True
                    }
                }
            )
        
        # Default strategy for other task types
        else:
            orchestrator.register_recovery_strategy(
                subtask_id,
                {
                    "type": "retry",
                    "max_attempts": 2,
                    "backoff": 5  # seconds
                }
            )
    
    logger.info(f"Registered recovery strategies for all subtasks of task {task_id}")


def example_harmony_presence_implementation(orchestrator: OrchestratorEngine) -> None:
    """
    Example of how the "Harmony Through Presence" principle is implemented.
    
    Args:
        orchestrator: The OrchestratorEngine instance
    """
    # 1. Balanced workload distribution
    # Get current orchestration status
    status = orchestrator.get_orchestration_status()
    
    # Log the current workload distribution for visibility
    logger.info("Agent workload distribution:")
    for agent_id, agent_status in status["agent_status"].items():
        load_percent = (agent_status["current_load"] / agent_status["max_load"]) * 100
        logger.info(f"  {agent_id}: {agent_status['current_load']}/{agent_status['max_load']} tasks ({load_percent:.1f}%)")
    
    # 2. Active status updates
    # The orchestrator automatically sends status updates based on the status_update_frequency
    # We can adjust this frequency for more responsive presence
    orchestrator.status_update_frequency = timedelta(seconds=15)
    logger.info(f"Set status update frequency to {orchestrator.status_update_frequency.seconds} seconds")
    
    # 3. Transparent progress tracking
    # Monitor a specific decomposed task
    task_ids = list(orchestrator.decomposed_tasks.keys())
    if task_ids:
        task_id = task_ids[0]
        decomposed = orchestrator.decomposed_tasks[task_id]
        
        # Display progress of subtasks
        logger.info(f"Progress for task {task_id}:")
        for subtask_id, subtask in decomposed.subtasks.items():
            logger.info(f"  {subtask.title}: {subtask.progress * 100:.1f}% - {subtask.status.value}")
            
        # Display critical path to show task scheduling
        critical_path = decomposed.get_critical_path()
        logger.info(f"Critical path for task {task_id}: {' -> '.join(critical_path)}")
    
    # 4. Responsive error handling
    # When errors occur, the orchestrator will automatically apply recovery strategies
    # and notify agents about the changes
    logger.info("Error recovery notification example:")
    recovery_message = {
        "type": "error_recovery",
        "task_id": "task-123",
        "recovery_action": "retry",
        "reason": "Data source temporarily unavailable",
        "next_attempt": datetime.now(timezone.utc).isoformat(),
        "remaining_attempts": 2
    }
    logger.info(f"  Recovery notification: {recovery_message}")
    
    # 5. Relationship-aware communication
    # Adjust communication based on the relationship with each agent
    logger.info("Relationship-aware communication example:")
    
    # For an agent with high trust and good relationship
    high_trust_message = {
        "type": "task_update",
        "task_id": "task-456",
        "status": "needs_assistance",
        "message": "Could you help with this analysis? I trust your expertise in this area.",
        "priority": "high",
        "context": {
            "previous_collaboration": 12,
            "success_rate": 0.95,
            "relationship_level": "trusted_partner"
        }
    }
    logger.info(f"  High trust communication: {high_trust_message}")
    
    # For a new agent with limited relationship history
    new_agent_message = {
        "type": "task_update",
        "task_id": "task-789",
        "status": "assigned",
        "message": "This task is assigned based on your capabilities. Please follow the provided guidelines carefully.",
        "priority": "medium",
        "context": {
            "previous_collaboration": 1,
            "success_rate": 0.0,  # No history yet
            "relationship_level": "new_collaborator"
        }
    }
    logger.info(f"  New agent communication: {new_agent_message}")


def simulate_orchestration_execution() -> None:
    """
    Simulate the execution of tasks in the orchestration system.
    """
    # Set up orchestrator and create a decomposed task
    orchestrator = setup_orchestrator()
    task_id = example_task_decomposition(orchestrator)
    
    if not task_id:
        logger.error("Failed to create and decompose task")
        return
    
    # Register error recovery strategies
    example_error_recovery(orchestrator, task_id)
    
    # Start the orchestration process
    orchestrator.start_orchestration()
    
    try:
        # Simulate task execution
        logger.info("Simulating task execution (press Ctrl+C to stop)...")
        
        # Run for a fixed amount of time in this example
        simulation_time = 60  # seconds
        end_time = time.time() + simulation_time
        
        while time.time() < end_time:
            # Get current task status
            status = orchestrator.get_orchestration_status()
            
            # Log task counts
            logger.info(f"Task status: {status['task_counts']}")
            
            # Demonstrate the "Harmony Through Presence" principle
            example_harmony_presence_implementation(orchestrator)
            
            # Simulate some task progress updates
            decomposed = orchestrator.decomposed_tasks.get(task_id)
            if decomposed:
                # Find tasks that are in progress
                in_progress_tasks = [
                    subtask for subtask in decomposed.subtasks.values()
                    if subtask.status == TaskStatus.IN_PROGRESS
                ]
                
                # Update progress for these tasks
                for subtask in in_progress_tasks:
                    # Random progress increase between 0.05 and 0.2
                    import random
                    progress_increase = random.uniform(0.05, 0.2)
                    new_progress = min(1.0, subtask.progress + progress_increase)
                    
                    # Pick the first assigned agent to update progress
                    if subtask.assigned_agents:
                        agent_id = subtask.assigned_agents[0]
                        orchestrator.task_coordinator.update_task_progress(
                            subtask.task_id, new_progress, agent_id
                        )
                        
                        # If task is complete, add a result and mark as complete
                        if new_progress >= 1.0:
                            orchestrator.task_coordinator.add_task_result(
                                subtask.task_id,
                                agent_id,
                                {"data": f"Completed result for {subtask.title}"}
                            )
                            orchestrator.task_coordinator.complete_task(subtask.task_id, agent_id)
                            
                            logger.info(f"Completed task: {subtask.title}")
            
            # Show updated progress after simulation step
            decomposed = orchestrator.decomposed_tasks.get(task_id)
            if decomposed:
                decomposed.update_progress()
                logger.info(f"Overall task progress: {decomposed.progress * 100:.1f}%")
            
            # Wait before next update
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Stop the orchestration process
        orchestrator.stop_orchestration()
        logger.info("Stopped orchestration")


if __name__ == "__main__":
    simulate_orchestration_execution()
