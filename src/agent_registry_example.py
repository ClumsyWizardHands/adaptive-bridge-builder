"""
Agent Registry Example

This module demonstrates how to use the AgentRegistry to maintain a database of agents,
discover capabilities, match tasks to agents, and implement fair task distribution.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from agent_registry import (
    AgentRegistry, CapabilityLevel, CapabilityInfo, FairnessPolicy,
    MatchCriteria, TaskType, AgentRole, DiscoveryMethod
)
from agent_card import AgentCard
from orchestrator_engine import TaskType, AgentRole, AgentProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentRegistryExample")


def setup_agent_registry() -> AgentRegistry:
    """
    Set up and configure an AgentRegistry with example agents.
    
    Returns:
        Configured AgentRegistry instance
    """
    # Create the registry with hybrid fairness policy
    registry = AgentRegistry(
        fairness_policy=FairnessPolicy.HYBRID,
        discovery_methods=[
            DiscoveryMethod.AGENT_CARD,
            DiscoveryMethod.SELF_DECLARATION,
            DiscoveryMethod.OBSERVATION
        ]
    )
    
    # Register a language model agent
    lm_agent_capabilities = {
        "text_generation": CapabilityInfo(
            name="text_generation",
            description="Generate text based on prompts",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.GENERATION, TaskType.COMMUNICATION]
        ),
        "text_summarization": CapabilityInfo(
            name="text_summarization",
            description="Summarize text content",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.TRANSFORMATION, TaskType.EXTRACTION]
        ),
        "content_analysis": CapabilityInfo(
            name="content_analysis",
            description="Analyze text for sentiment, entities, and topics",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.ANALYSIS, TaskType.EXTRACTION]
        )
    }
    
    registry.register_agent(
        agent_id="language-model-agent",
        roles=[AgentRole.GENERATOR, AgentRole.TRANSFORMER],
        declared_capabilities=lm_agent_capabilities
    )
    
    # Register a research agent
    research_agent_capabilities = {
        "web_search": CapabilityInfo(
            name="web_search",
            description="Search the web for information",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.RESEARCH, TaskType.EXTRACTION]
        ),
        "fact_checking": CapabilityInfo(
            name="fact_checking",
            description="Verify factual information",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.VALIDATION, TaskType.RESEARCH]
        ),
        "data_synthesis": CapabilityInfo(
            name="data_synthesis", 
            description="Synthesize information from multiple sources",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.AGGREGATION, TaskType.ANALYSIS]
        )
    }
    
    registry.register_agent(
        agent_id="research-agent",
        roles=[AgentRole.RESEARCHER, AgentRole.VALIDATOR],
        declared_capabilities=research_agent_capabilities
    )
    
    # Register a code execution agent
    code_agent_capabilities = {
        "code_execution": CapabilityInfo(
            name="code_execution",
            description="Execute code in various languages",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.EXECUTION]
        ),
        "data_analysis": CapabilityInfo(
            name="data_analysis",
            description="Analyze and visualize data using code",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.ANALYSIS, TaskType.TRANSFORMATION]
        ),
        "api_integration": CapabilityInfo(
            name="api_integration",
            description="Integrate with external APIs",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.EXECUTION, TaskType.COMMUNICATION]
        )
    }
    
    registry.register_agent(
        agent_id="code-execution-agent",
        roles=[AgentRole.EXECUTOR, AgentRole.ANALYZER],
        declared_capabilities=code_agent_capabilities
    )
    
    # Register a specialized financial agent
    finance_agent_capabilities = {
        "financial_analysis": CapabilityInfo(
            name="financial_analysis",
            description="Analyze financial data and market trends",
            level=CapabilityLevel.MASTER,
            task_types=[TaskType.ANALYSIS, TaskType.DECISION]
        ),
        "risk_assessment": CapabilityInfo(
            name="risk_assessment",
            description="Assess financial risks",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.ANALYSIS, TaskType.VALIDATION]
        ),
        "financial_reporting": CapabilityInfo(
            name="financial_reporting",
            description="Generate financial reports",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.GENERATION, TaskType.AGGREGATION]
        )
    }
    
    registry.register_agent(
        agent_id="finance-specialist",
        roles=[AgentRole.SPECIALIST, AgentRole.ANALYZER],
        declared_capabilities=finance_agent_capabilities
    )
    
    # Register a monitoring agent
    monitoring_agent_capabilities = {
        "system_monitoring": CapabilityInfo(
            name="system_monitoring",
            description="Monitor system performance and health",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.MONITORING, TaskType.ANALYSIS]
        ),
        "alert_management": CapabilityInfo(
            name="alert_management",
            description="Manage and respond to system alerts",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.MONITORING, TaskType.COMMUNICATION]
        ),
        "error_recovery": CapabilityInfo(
            name="error_recovery",
            description="Recover from system errors",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.RECOVERY, TaskType.EXECUTION]
        )
    }
    
    registry.register_agent(
        agent_id="monitoring-agent",
        roles=[AgentRole.MONITOR, AgentRole.EXECUTOR],
        declared_capabilities=monitoring_agent_capabilities
    )
    
    return registry


def simulate_performance_history(registry: AgentRegistry) -> None:
    """
    Simulate a performance history for registered agents.
    
    Args:
        registry: The AgentRegistry instance
    """
    # Simulate performance for language model agent
    registry.update_agent_performance(
        agent_id="language-model-agent",
        task_type=TaskType.GENERATION,
        success=True,
        response_time=0.8,
        completion_time=5.2,
        task_id="task-1",
        quality_score=0.92
    )
    
    registry.update_agent_performance(
        agent_id="language-model-agent",
        task_type=TaskType.TRANSFORMATION,
        success=True,
        response_time=0.5,
        completion_time=2.1,
        task_id="task-2",
        quality_score=0.95
    )
    
    registry.update_agent_performance(
        agent_id="language-model-agent",
        task_type=TaskType.ANALYSIS,
        success=True,
        response_time=1.2,
        completion_time=8.7,
        task_id="task-3",
        quality_score=0.75  # Not as good at analysis
    )
    
    # Simulate performance for research agent
    registry.update_agent_performance(
        agent_id="research-agent",
        task_type=TaskType.RESEARCH,
        success=True,
        response_time=1.5,
        completion_time=12.3,
        task_id="task-4",
        quality_score=0.97
    )
    
    registry.update_agent_performance(
        agent_id="research-agent",
        task_type=TaskType.VALIDATION,
        success=True,
        response_time=2.1,
        completion_time=15.6,
        task_id="task-5",
        quality_score=0.88
    )
    
    # Failed research task
    registry.update_agent_performance(
        agent_id="research-agent",
        task_type=TaskType.RESEARCH,
        success=False,
        response_time=4.5,
        completion_time=0,  # Failed, so no completion
        task_id="task-6",
        error_type="source_unavailable"
    )
    
    # Simulate performance for code execution agent
    registry.update_agent_performance(
        agent_id="code-execution-agent",
        task_type=TaskType.EXECUTION,
        success=True,
        response_time=0.3,
        completion_time=3.2,
        task_id="task-7",
        quality_score=0.99
    )
    
    registry.update_agent_performance(
        agent_id="code-execution-agent",
        task_type=TaskType.ANALYSIS,
        success=True,
        response_time=0.8,
        completion_time=10.5,
        task_id="task-8",
        quality_score=0.93
    )
    
    # Simulate performance for finance specialist
    registry.update_agent_performance(
        agent_id="finance-specialist",
        task_type=TaskType.ANALYSIS,
        success=True,
        response_time=1.7,
        completion_time=15.3,
        task_id="task-9",
        quality_score=0.98  # Very good at financial analysis
    )
    
    # More tasks for some agents to create imbalance
    for i in range(10, 20):
        registry.update_agent_performance(
            agent_id="language-model-agent",
            task_type=TaskType.GENERATION,
            success=True,
            response_time=0.7 + (i % 3) * 0.1,
            completion_time=4.5 + (i % 5) * 0.5,
            task_id=f"task-{i}",
            quality_score=0.9 + (i % 10) * 0.01
        )
    
    logger.info("Simulated performance history for all agents")


def demonstrate_capability_discovery(registry: AgentRegistry) -> None:
    """
    Demonstrate capability discovery for a new agent.
    
    Args:
        registry: The AgentRegistry instance
    """
    # Create a mock agent card
    agent_card = {
        "name": "multimedia-processor",
        "description": "Processes and analyzes multimedia content",
        "functions": [
            {
                "name": "analyze_image",
                "description": "Analyze image content including objects, scenes, and text",
                "parameters": {
                    "image_url": {"type": "string", "description": "URL of the image to analyze"}
                }
            },
            {
                "name": "transcribe_audio",
                "description": "Transcribe speech in audio files to text",
                "parameters": {
                    "audio_url": {"type": "string", "description": "URL of the audio file to transcribe"}
                }
            },
            {
                "name": "extract_video_frames",
                "description": "Extract key frames from video content",
                "parameters": {
                    "video_url": {"type": "string", "description": "URL of the video to process"},
                    "frame_interval": {"type": "number", "description": "Interval between frames in seconds"}
                }
            }
        ]
    }
    
    # Register the agent with just the card
    registry.register_agent(
        agent_id="multimedia-agent",
        agent_card=agent_card,
        roles=[AgentRole.ANALYZER, AgentRole.TRANSFORMER]
    )
    
    # Discover additional capabilities
    registry.discover_capabilities(
        "multimedia-agent", 
        methods=[DiscoveryMethod.AGENT_CARD, DiscoveryMethod.SELF_DECLARATION]
    )
    
    # List discovered capabilities
    capabilities = registry.capabilities_by_agent.get("multimedia-agent", {})
    logger.info(f"Discovered {len(capabilities)} capabilities for multimedia-agent:")
    for name, info in capabilities.items():
        logger.info(f"  - {name} ({info.level.name}): {', '.join(t.name for t in info.task_types)}")


def demonstrate_agent_matching(registry: AgentRegistry) -> None:
    """
    Demonstrate matching tasks to the most appropriate agents.
    
    Args:
        registry: The AgentRegistry instance
    """
    # Example 1: Find best agent for text generation
    best_generator = registry.find_best_agent_for_task(
        task_type=TaskType.GENERATION,
        required_capabilities=["text_generation"],
        criteria=MatchCriteria.PERFORMANCE
    )
    
    logger.info(f"Best agent for text generation: {best_generator}")
    
    # Example 2: Find best agent for financial analysis
    best_financial_analyst = registry.find_best_agent_for_task(
        task_type=TaskType.ANALYSIS,
        required_capabilities=["financial_analysis"],
        criteria=MatchCriteria.CAPABILITY
    )
    
    logger.info(f"Best agent for financial analysis: {best_financial_analyst}")
    
    # Example 3: Find agents capable of data analysis with minimum performance
    data_analysts = registry.find_agents_by_task_type(
        task_type=TaskType.ANALYSIS,
        min_performance=0.8
    )
    
    logger.info(f"Qualified data analysts: {data_analysts}")
    
    # Example 4: Find agents with web search capability
    researchers = registry.find_agents_with_capability(
        capability_name="web_search",
        task_type=TaskType.RESEARCH,
        min_level=CapabilityLevel.PROFICIENT
    )
    
    logger.info(f"Qualified researchers: {researchers}")
    
    # Example 5: Apply fairness policy for selecting agents
    for i in range(5):
        fair_selection = registry.find_best_agent_for_task(
            task_type=TaskType.ANALYSIS,
            criteria=MatchCriteria.COMPOSITE,
            apply_fairness=True  # Apply fairness policy
        )
        logger.info(f"Fair selection for analysis task {i+1}: {fair_selection}")


def demonstrate_capability_negotiation(registry: AgentRegistry) -> None:
    """
    Demonstrate capability negotiation between agents.
    
    Args:
        registry: The AgentRegistry instance
    """
    # Create a capability request
    from agent_registry import CapabilityRequest, CapabilityResponse
    
    request = CapabilityRequest(
        request_id="req-001",
        requesting_agent_id="orchestrator-agent",
        capability_name="financial_analysis",
        task_types=[TaskType.ANALYSIS, TaskType.DECISION],
        minimum_level=CapabilityLevel.PROFICIENT,
        parameters={
            "data_format": "json",
            "analysis_depth": "comprehensive",
            "include_recommendations": True
        },
        response_by=datetime.now(timezone.utc) + timedelta(seconds=30),
        priority=3
    )
    
    # Find capable agents
    capable_agents = registry.find_agents_with_capability(
        capability_name=request.capability_name,
        task_type=request.task_types[0],
        min_level=request.minimum_level
    )
    
    logger.info(f"Found {len(capable_agents)} agents capable of handling the request")
    
    # Simulate responses from agents
    responses = []
    for agent_id in capable_agents:
        # Simulate capability check and response generation
        profile = registry.capabilities_by_agent.get(agent_id, {}).get(request.capability_name)
        if profile:
            response = CapabilityResponse(
                request_id=request.request_id,
                responding_agent_id=agent_id,
                capability_name=request.capability_name,
                available=True,
                level=profile.level,
                supported_parameters={
                    "data_format": ["json", "csv", "xml"],
                    "analysis_depth": ["basic", "standard", "comprehensive"],
                    "include_recommendations": True
                },
                estimated_response_time=1.5,
                estimated_completion_time=12.0,
                cost=10.0 if agent_id == "finance-specialist" else 15.0
            )
            responses.append(response)
    
    # Select the best agent based on responses
    if responses:
        # Sort by a combination of level, cost, and estimated completion time
        responses.sort(key=lambda r: (
            -r.level.value,  # Higher level is better (negative for descending)
            r.cost,          # Lower cost is better
            r.estimated_completion_time  # Faster completion is better
        ))
        
        best_response = responses[0]
        logger.info(f"Selected {best_response.responding_agent_id} for the capability request")
        logger.info(f"  Level: {best_response.level.name}")
        logger.info(f"  Cost: {best_response.cost}")
        logger.info(f"  Estimated completion time: {best_response.estimated_completion_time}s")
    else:
        logger.warning("No compatible agents found for the capability request")


def demonstrate_fairness_principle(registry: AgentRegistry) -> None:
    """
    Demonstrate the "Fairness as a Fundamental Truth" principle in agent selection.
    
    Args:
        registry: The AgentRegistry instance
    """
    # Print initial task distribution
    logger.info("Initial task distribution:")
    for agent_id, task_counts in registry.task_distribution.items():
        total_tasks = sum(task_counts.values())
        logger.info(f"  {agent_id}: {total_tasks} total tasks")
    
    # Simulate multiple task assignments with fairness policy
    task_types = [
        TaskType.ANALYSIS,
        TaskType.GENERATION,
        TaskType.RESEARCH,
        TaskType.EXECUTION
    ]
    
    # Run multiple selections to show fairness in action
    logger.info("Running 20 fair selections...")
    task_assignments = {}
    
    for i in range(20):
        task_type = task_types[i % len(task_types)]
        selected_agent = registry.find_best_agent_for_task(
            task_type=task_type,
            criteria=MatchCriteria.COMPOSITE,
            apply_fairness=True
        )
        
        if selected_agent:
            task_assignments[selected_agent] = task_assignments.get(selected_agent, 0) + 1
            
            # Update performance to simulate task completion
            registry.update_agent_performance(
                agent_id=selected_agent,
                task_type=task_type,
                success=True,
                response_time=1.0,
                completion_time=5.0,
                task_id=f"fairness-task-{i}"
            )
    
    # Show distribution of assignments
    logger.info("Fairness-based task assignment distribution:")
    for agent_id, count in task_assignments.items():
        logger.info(f"  {agent_id}: {count} tasks")
    
    # Compare with performance-only selection (no fairness)
    perf_assignments = {}
    for i in range(20):
        task_type = task_types[i % len(task_types)]
        selected_agent = registry.find_best_agent_for_task(
            task_type=task_type,
            criteria=MatchCriteria.PERFORMANCE,
            apply_fairness=False
        )
        
        if selected_agent:
            perf_assignments[selected_agent] = perf_assignments.get(selected_agent, 0) + 1
    
    logger.info("Performance-only task assignment distribution:")
    for agent_id, count in perf_assignments.items():
        logger.info(f"  {agent_id}: {count} tasks")
    
    # Calculate fairness metrics
    fair_std_dev = calculate_distribution_std_dev(task_assignments)
    perf_std_dev = calculate_distribution_std_dev(perf_assignments)
    
    logger.info(f"Fairness-based distribution standard deviation: {fair_std_dev:.2f}")
    logger.info(f"Performance-only distribution standard deviation: {perf_std_dev:.2f}")
    logger.info(f"Lower standard deviation indicates more balanced distribution")


def calculate_distribution_std_dev(assignments: Dict[str, int]) -> float:
    """
    Calculate the standard deviation of task assignments.
    
    Args:
        assignments: Dictionary of agent ID to assignment count
        
    Returns:
        Standard deviation of the distribution
    """
    if not assignments:
        return 0.0
        
    values = list(assignments.values())
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5)


def main() -> None:
    """Run the Agent Registry example."""
    # Set up agent registry with example agents
    registry = setup_agent_registry()
    
    # Simulate performance history
    simulate_performance_history(registry)
    
    # Demonstrate capability discovery
    demonstrate_capability_discovery(registry)
    
    # Demonstrate agent matching
    demonstrate_agent_matching(registry)
    
    # Demonstrate capability negotiation
    demonstrate_capability_negotiation(registry)
    
    # Demonstrate fairness principle
    demonstrate_fairness_principle(registry)


if __name__ == "__main__":
    main()
