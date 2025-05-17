"""
OrchestrationAnalytics Example

This example demonstrates the usage of the OrchestrationAnalytics system to track
performance metrics across an orchestrated workflow, identify bottlenecks, recommend 
optimizations, and visualize orchestration patterns.

The example showcases how to:
1. Initialize the analytics system with existing orchestrator components
2. Configure and collect performance metrics
3. Analyze workflow bottlenecks
4. Generate optimization recommendations
5. Create visualizations of orchestration patterns
6. Measure principle alignment
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from orchestration_analytics import (
    OrchestrationAnalytics, MetricType, VisualizationType,
    AnalysisPeriod, BottleneckType, RecommendationCategory
)
from orchestrator_engine import OrchestratorEngine, TaskType, AgentRole
from project_orchestrator import ProjectOrchestrator
from principle_engine import PrincipleEngine
from collaborative_task_handler import Task, TaskStatus, TaskPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrchestrationAnalyticsExample")

def create_sample_task() -> Task:
    """Create a sample task for demonstration purposes."""
    return Task(
        task_id="task-123",
        title="Process customer data",
        description="Extract, transform, and load customer data from API to database",
        required_capabilities=["data_processing", "api_integration", "database"],
        priority=TaskPriority.HIGH,
        dependencies=[],
        metadata={
            "created_at": datetime.utcnow().isoformat(),
            "estimated_duration": 60,  # minutes
            "task_type": TaskType.EXTRACTION.name
        }
    )

def run_example():
    """Run the example demonstration of OrchestrationAnalytics."""
    # Initialize sample components
    logger.info("Initializing components...")
    orchestrator_engine = OrchestratorEngine(agent_id="orchestrator-001")
    project_orchestrator = ProjectOrchestrator(agent_id="project-orchestrator-001")
    principle_engine = PrincipleEngine(agent_id="principle-engine-001")
    
    # Initialize the analytics system
    logger.info("Setting up OrchestrationAnalytics...")
    analytics = OrchestrationAnalytics(
        agent_id="analytics-001",
        orchestrator_engine=orchestrator_engine,
        project_orchestrator=project_orchestrator,
        principle_engine=principle_engine
    )
    
    # Register a custom metric
    data_throughput_metric = analytics.register_metric(
        name="Data Throughput",
        description="Amount of data processed per minute",
        type=MetricType.PERFORMANCE,
        unit="MB/min",
        aggregation_method="avg",
        ideal_trend="increase",
        warning_threshold=5.0,
        critical_threshold=1.0
    )
    
    # Record some sample metrics
    logger.info("Recording sample metrics...")
    
    # Simulate task processing metrics
    for i in range(10):
        analytics.record_metric(
            metric_id=data_throughput_metric,
            value=8.5 + (i * 0.5),  # Increasing throughput
            timestamp=(datetime.utcnow() - timedelta(minutes=10-i)).isoformat(),
            context={"task_id": "task-123", "agent_id": "processor-agent-001"}
        )
    
    # Record agent utilization
    agent_utilization_metrics = [m for m, d in analytics.metric_definitions.items() 
                               if d.name == "Agent Utilization"][0]
    analytics.record_metric(
        metric_id=agent_utilization_metrics,
        value=85.0,  # 85% utilization
        context={"agent_id": "processor-agent-001"}
    )
    
    # Task completion rate
    task_completion_metrics = [m for m, d in analytics.metric_definitions.items() 
                             if d.name == "Task Completion Rate"][0]
    analytics.record_metric(
        metric_id=task_completion_metrics,
        value=12.5,  # 12.5 tasks per hour
        context={"period": "last_hour"}
    )
    
    # Get metric values
    logger.info("Retrieving aggregated metrics...")
    throughput = analytics.get_metric_value(
        metric_id=data_throughput_metric,
        aggregation_period=AnalysisPeriod.HOURLY
    )
    logger.info(f"Average data throughput: {throughput:.2f} MB/min")
    
    # Analyze bottlenecks
    logger.info("Analyzing potential bottlenecks...")
    # For demo purposes, we'll simulate an agent capacity bottleneck
    # by adding a sample agent profile and manually triggering analysis
    orchestrator_engine.agent_profiles["processor-agent-001"] = orchestrator_engine.AgentProfile(
        agent_id="processor-agent-001",
        roles=[AgentRole.EXECUTOR],
        capabilities=["data_processing", "api_integration", "database"],
        specialization={TaskType.EXTRACTION: 0.9, TaskType.TRANSFORMATION: 0.8},
        current_load=4,
        max_load=5,
        task_history=["task-101", "task-102", "task-103", "task-123"]
    )
    
    bottlenecks = analytics.analyze_bottlenecks()
    logger.info(f"Found {len(bottlenecks)} potential bottlenecks")
    
    for bottleneck in bottlenecks:
        logger.info(f"Bottleneck: {bottleneck.bottleneck_type.name}, Severity: {bottleneck.severity:.2f}")
        logger.info(f"Impact: {bottleneck.impact_assessment}")
    
    # Generate optimization recommendations
    logger.info("Generating optimization recommendations...")
    recommendations = analytics.generate_recommendations(bottlenecks)
    
    for recommendation in recommendations:
        logger.info(f"Recommendation: {recommendation.title}")
        logger.info(f"Description: {recommendation.description}")
        logger.info(f"Priority: {recommendation.priority:.2f}")
        logger.info(f"Implementation steps:")
        for step in recommendation.implementation_steps:
            logger.info(f"- {step}")
    
    # Create a visualization
    logger.info("Creating visualization of orchestration patterns...")
    visualization = analytics.create_visualization(
        visualization_type=VisualizationType.TIMELINE,
        title="Task Execution Timeline",
        description="Timeline of task execution across agents",
        time_range=(
            (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            datetime.utcnow().isoformat()
        ),
        data_sources={"tasks": True, "agents": ["processor-agent-001"]},
        filters={"task_types": [TaskType.EXTRACTION.name, TaskType.TRANSFORMATION.name]},
        parameters={"show_dependencies": True, "show_critical_path": True}
    )
    
    if visualization:
        logger.info(f"Created visualization: {visualization.request_id}")
        logger.info(f"Format: {visualization.render_format}")
    
    # Measure principle alignment
    logger.info("Measuring principle alignment...")
    # For demo purposes, let's add a sample principle to the principle engine
    principle_engine.principles = {
        "principle-001": principle_engine.Principle(
            principle_id="principle-001",
            name="Efficiency",
            description="Optimize resource usage and minimize waste",
            weight=1.0
        )
    }
    
    alignment = analytics.measure_principle_alignment(["principle-001"])
    
    for principle_id, measurement in alignment.items():
        logger.info(f"Principle: {principle_id}")
        logger.info(f"Alignment score: {measurement.alignment_score:.2f}")
        logger.info(f"Improvement opportunities:")
        for opportunity in measurement.improvement_opportunities:
            logger.info(f"- {opportunity}")
    
    logger.info("Example complete!")
    return analytics

if __name__ == "__main__":
    run_example()
