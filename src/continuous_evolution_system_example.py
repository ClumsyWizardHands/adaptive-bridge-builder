"""
Continuous Evolution System Example

This example demonstrates the usage of the ContinuousEvolutionSystem to track
orchestration patterns, reflect on performance, adapt agent selection and task
allocation, evolve capabilities, and maintain a growth journal - all while
embodying the "Resilience Through Reflection" principle.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from continuous_evolution_system import (
    ContinuousEvolutionSystem,
    OrchestrationPattern,
    CapabilityEvolution,
    GrowthMilestone,
    OrchestrationDimension
)
from learning_system import (
    LearningSystem,
    OutcomeType,
    LearningDimension
)
from orchestration_analytics import OrchestrationAnalytics
from orchestrator_engine import (
    OrchestratorEngine, 
    TaskType, 
    AgentRole,
    TaskDecompositionStrategy,
    DecomposedTask
)
from principle_engine import PrincipleEngine
from collaborative_task_handler import Task, TaskStatus, TaskPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContinuousEvolutionSystemExample")

def run_example():
    """Run the example demonstration of ContinuousEvolutionSystem."""
    
    # Initialize needed components
    logger.info("Initializing components...")
    principle_engine = PrincipleEngine(agent_id="principle-engine-001")
    
    # Add the "Resilience Through Reflection" principle
    principle_engine.add_principle(
        name="Resilience Through Reflection",
        description="Continuously evaluate experiences, adapt strategies, and maintain core identity through thoughtful reflection.",
        weight=1.0
    )
    
    orchestrator_engine = OrchestratorEngine(
        agent_id="orchestrator-001",
        principle_engine=principle_engine
    )
    
    analytics = OrchestrationAnalytics(
        agent_id="analytics-001",
        orchestrator_engine=orchestrator_engine,
        principle_engine=principle_engine
    )
    
    # Initialize learning system with growth journal directory
    learning_system = LearningSystem(
        principle_engine=principle_engine,
        growth_journal_dir="data/growth_journal"
    )
    
    # Initialize the continuous evolution system
    logger.info("Setting up ContinuousEvolutionSystem...")
    evolution_system = ContinuousEvolutionSystem(
        learning_system=learning_system,
        orchestration_analytics=analytics,
        orchestrator_engine=orchestrator_engine,
        principle_engine=principle_engine,
        growth_journal_dir="data/growth_journal",
        evolution_data_dir="data/evolution"
    )
    
    # Step1: Create an example orchestration task
    logger.info("Creating sample orchestration task...")
    original_task = Task(
        task_id="task-original-123",
        title="Analyze customer feedback and improve service",
        description="Process customer feedback data, identify key issues, and develop service improvements",
        required_capabilities=["data_analysis", "service_design", "customer_insights"],
        priority=TaskPriority.HIGH
    )
    
    # Step 2: Create a decomposed task
    decomposed_task = DecomposedTask(
        original_task=original_task,
        strategy=TaskDecompositionStrategy.FUNCTIONAL
    )
    
    # Add subtasks to the decomposed task
    subtask1 = Task(
        task_id="subtask-1",
        title="Extract key themes from customer feedback",
        description="Analyze feedback data to identify recurring themes and issues",
        required_capabilities=["data_analysis", "text_processing"],
        priority=TaskPriority.HIGH
    )
    subtask1.status = TaskStatus.COMPLETED
    subtask1.results = {
        "analyzer-agent-001": {
            "status": "success",
            "data": {"themes": ["response_time", "product_quality", "support_experience"]}
        }
    }
    
    subtask2 = Task(
        task_id="subtask-2",
        title="Quantify impact of identified issues",
        description="Determine frequency and severity of each identified issue",
        required_capabilities=["data_analysis", "statistics"],
        priority=TaskPriority.MEDIUM
    )
    subtask2.status = TaskStatus.COMPLETED
    subtask2.results = {
        "stats-agent-002": {
            "status": "success",
            "data": {
                "impact_scores": {
                    "response_time": 0.85,
                    "product_quality": 0.65,
                    "support_experience": 0.72
                }
            }
        }
    }
    
    subtask3 = Task(
        task_id="subtask-3",
        title="Design service improvements",
        description="Develop specific service improvements to address key issues",
        required_capabilities=["service_design", "customer_insights"],
        priority=TaskPriority.HIGH
    )
    subtask3.status = TaskStatus.COMPLETED
    subtask3.results = {
        "design-agent-003": {
            "status": "success",
            "data": {
                "improvements": [
                    {"area": "response_time", "solution": "Implement automated initial response system"},
                    {"area": "product_quality", "solution": "Add additional QA steps in manufacturing"},
                    {"area": "support_experience", "solution": "Enhance staff training program"}
                ]
            }
        }
    }
    
    # Add dependencies between subtasks
    decomposed_task.add_subtask(subtask1, dependencies=[])
    decomposed_task.add_subtask(subtask2, dependencies=[("subtask-1", "sequential")])
    decomposed_task.add_subtask(subtask3, dependencies=[("subtask-2", "sequential")])
    
    # Step 3: Track the orchestration pattern
    logger.info("Tracking orchestration pattern...")
    performance_metrics = {
        "total_completion_time": 45.5,  # minutes
        "resource_utilization": 0.82,   # 82% utilized
        "quality_rating": 0.91,         # 91% quality
        "agent_coordination_score": 0.89  # 89% coordination efficiency
    }
    
    agent_performances = {
        "analyzer-agent-001": {
            "success_rate": 1.0,
            "processing_time": 15.2,  # minutes
            "quality_score": 0.94
        },
        "stats-agent-002": {
            "success_rate": 1.0,
            "processing_time": 12.8,  # minutes
            "quality_score": 0.88
        },
        "design-agent-003": {
            "success_rate": 1.0,
            "processing_time": 17.5,  # minutes
            "quality_score": 0.92
        }
    }
    
    # Track the pattern with a successful outcome
    pattern_id = evolution_system.track_orchestration_pattern(
        decomposed_task=decomposed_task,
        outcome=OutcomeType.SUCCESSFUL,
        performance_metrics=performance_metrics,
        agent_performances=agent_performances,
        notes="Initial implementation of functional decomposition for a customer feedback analysis task"
    )
    
    logger.info(f"Tracked orchestration pattern with ID: {pattern_id}")
    
    # Step 4: Manually trigger a reflection cycle
    logger.info("Performing orchestration reflection...")
    insights = evolution_system.reflect_on_orchestration()
    
    for i, insight in enumerate(insights):
        logger.info(f"Insight {i+1}: {insight['description']}")
        logger.info(f"Recommendation: {insight['recommendation']}")
        logger.info(f"Confidence: {insight['confidence']:.2f}")
    
    # Step 5: Create a new capability
    logger.info("Creating new capability from reflection...")
    capability_id = "feedback-analysis-capability"
    capability = CapabilityEvolution(
        capability_id=capability_id,
        name="Customer Feedback Analysis",
        description="Capability to efficiently analyze and extract insights from customer feedback",
        created_at=datetime.utcnow().isoformat(),
        evolution_stages=[
            {
                "stage": 0,
                "name": "Initial Implementation",
                "description": "Basic analysis of customer feedback themes",
                "implemented_at": datetime.utcnow().isoformat()
            }
        ],
        current_stage=0,
        performance_metrics={
            "accuracy": [0.85],
            "processing_time": [45.5]
        },
        development_focus="Improve theme extraction precision"
    )
    
    evolution_system.capabilities[capability_id] = capability
    evolution_system._save_evolution_data()
    
    logger.info(f"Created capability: {capability.name}")
    
    # Step 6: Record a growth milestone
    logger.info("Recording growth milestone...")
    milestone = GrowthMilestone(
        milestone_id=f"milestone-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        title="Initial Customer Feedback Analysis Implementation",
        description="Successfully implemented first version of customer feedback analysis orchestration pattern",
        achieved_at=datetime.utcnow().isoformat(),
        category="capability",
        impact_score=0.8,
        metrics_before={},
        metrics_after={
            "success_rate": 1.0,
            "completion_time": 45.5,
            "quality_score": 0.91
        },
        references=[pattern_id, capability_id]
    )
    
    evolution_system.growth_milestones.append(milestone)
    evolution_system._save_evolution_data()
    
    logger.info(f"Recorded milestone: {milestone.title}")
    
    # Step 7: Simulate capability evolution
    logger.info("Evolving capability based on experience...")
    capability = evolution_system.capabilities[capability_id]
    
    # Add a new evolution stage
    capability.evolution_stages.append({
        "stage": 1,
        "name": "Enhanced Theme Extraction",
        "description": "Improved theme extraction with semantic clustering and sentiment analysis",
        "implemented_at": datetime.utcnow().isoformat(),
        "improvements": [
            "Added semantic clustering for better theme identification",
            "Integrated sentiment analysis for deeper insights",
            "Implemented priority ranking of feedback items"
        ]
    })
    
    # Update current stage and performance metrics
    capability.current_stage = 1
    capability.performance_metrics["accuracy"].append(0.92)  # Improved accuracy
    capability.performance_metrics["processing_time"].append(38.2)  # Faster processing
    capability.development_focus = "Improve scalability for larger feedback datasets"
    
    evolution_system._save_evolution_data()
    logger.info(f"Evolved capability to stage {capability.current_stage}: {capability.evolution_stages[-1]['name']}")
    
    # Step 8: Perform deep reflection (manually triggered)
    logger.info("Performing deep reflection on all patterns and capabilities...")
    evolution_system.deep_reflection()
    
    logger.info("Example complete!")
    return evolution_system

if __name__ == "__main__":
    run_example()
