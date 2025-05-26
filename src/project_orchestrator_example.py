"""
Project Orchestrator Example

This module demonstrates how to use the ProjectOrchestrator to manage complex multi-stage
projects requiring various agent capabilities. It provides a concrete example of orchestrating 
a business strategy development project, coordinating research, analysis, and creative agents.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from project_orchestrator import (
    ProjectOrchestrator, MilestoneStatus, ResourceType,
    Resource, Milestone, Project, ScheduleEvent, ProjectIssue, StatusUpdate
)
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability,
    DependencyType, TaskDecompositionStrategy
)
from collaborative_task_handler import TaskStatus, TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ProjectOrchestratorExample")


def setup_project_orchestrator() -> ProjectOrchestrator:
    """
    Set up and configure a ProjectOrchestrator with a base OrchestratorEngine.
    
    Returns:
        Configured ProjectOrchestrator instance
    """
    # Create underlying orchestrator engine
    orchestrator_engine = OrchestratorEngine(
        agent_id="project-orchestrator-agent",
        storage_dir="data/orchestration"
    )
    
    # Register specialized agents with different roles and capabilities
    
    # Strategy agent
    orchestrator_engine.register_agent(
        agent_id="strategy-agent",
        roles=[AgentRole.SPECIALIST],
        capabilities=["business_strategy", "market_analysis", "competitive_analysis"],
        specialization={
            TaskType.ANALYSIS: 0.9,
            TaskType.DECISION: 0.8,
            TaskType.ORCHESTRATION: 0.7,
        },
        max_load=3
    )
    
    # Research agent
    orchestrator_engine.register_agent(
        agent_id="research-agent",
        roles=[AgentRole.RESEARCHER],
        capabilities=["market_research", "data_collection", "trend_analysis"],
        specialization={
            TaskType.RESEARCH: 0.9,
            TaskType.EXTRACTION: 0.8,
            TaskType.ANALYSIS: 0.6,
        },
        max_load=4
    )
    
    # Data analysis agent
    orchestrator_engine.register_agent(
        agent_id="data-analysis-agent",
        roles=[AgentRole.ANALYZER],
        capabilities=["data_analysis", "statistical_modeling", "forecasting"],
        specialization={
            TaskType.ANALYSIS: 0.9,
            TaskType.TRANSFORMATION: 0.7,
            TaskType.VALIDATION: 0.6,
        },
        max_load=3
    )
    
    # Creative agent
    orchestrator_engine.register_agent(
        agent_id="creative-agent",
        roles=[AgentRole.GENERATOR],
        capabilities=["content_creation", "visual_design", "storytelling"],
        specialization={
            TaskType.GENERATION: 0.9,
            TaskType.TRANSFORMATION: 0.7,
            TaskType.COMMUNICATION: 0.8,
        },
        max_load=3
    )
    
    # Financial analysis agent
    orchestrator_engine.register_agent(
        agent_id="financial-agent",
        roles=[AgentRole.ANALYZER, AgentRole.SPECIALIST],
        capabilities=["financial_analysis", "roi_calculation", "budget_planning"],
        specialization={
            TaskType.ANALYSIS: 0.8,
            TaskType.VALIDATION: 0.7,
            TaskType.DECISION: 0.6,
        },
        max_load=2
    )
    
    # Implementation planning agent
    orchestrator_engine.register_agent(
        agent_id="implementation-agent",
        roles=[AgentRole.EXECUTOR, AgentRole.COORDINATOR],
        capabilities=["implementation_planning", "resource_management", "timeline_planning"],
        specialization={
            TaskType.ORCHESTRATION: 0.8,
            TaskType.EXECUTION: 0.7,
            TaskType.MONITORING: 0.6,
        },
        max_load=2
    )
    
    # Communication agent
    orchestrator_engine.register_agent(
        agent_id="communication-agent",
        roles=[AgentRole.COMMUNICATOR],
        capabilities=["presentation_creation", "report_writing", "stakeholder_communication"],
        specialization={
            TaskType.COMMUNICATION: 0.9,
            TaskType.GENERATION: 0.8,
            TaskType.TRANSFORMATION: 0.7,
        },
        max_load=3
    )
    
    # Create project orchestrator using the configured engine
    project_orchestrator = ProjectOrchestrator(
        agent_id="project-orchestrator-agent",
        orchestrator_engine=orchestrator_engine,
        storage_dir="data/projects"
    )
    
    logger.info("Project orchestrator set up with 7 specialized agents")
    return project_orchestrator


def create_business_strategy_project(project_orchestrator: ProjectOrchestrator) -> Project:
    """
    Create a complex business strategy development project with milestones and resources.
    
    Args:
        project_orchestrator: The configured ProjectOrchestrator
        
    Returns:
        The created Project instance
    """
    # Create project
    now = datetime.now(timezone.utc)
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=90)).strftime("%Y-%m-%d")
    
    project = project_orchestrator.create_project(
        name="Business Strategy Development for Expansion",
        description=(
            "Develop a comprehensive business strategy for expanding into new markets "
            "including market analysis, competitive positioning, financial projections, "
            "and implementation roadmap."
        ),
        start_date=start_date,
        end_date=end_date,
        stakeholders=[
            {
                "id": "ceo@company.com",
                "name": "Alex Johnson",
                "role": "CEO",
                "communication_preference": "weekly_update"
            },
            {
                "id": "cfo@company.com",
                "name": "Morgan Smith",
                "role": "CFO",
                "communication_preference": "financial_updates"
            },
            {
                "id": "cmo@company.com",
                "name": "Jamie Williams",
                "role": "CMO",
                "communication_preference": "marketing_updates"
            },
            {
                "id": "board@company.com",
                "name": "Board of Directors",
                "role": "oversight",
                "communication_preference": "milestone_updates"
            }
        ],
        tags=["business_strategy", "market_expansion", "growth_planning"],
        metadata={
            "priority": "high",
            "strategic_importance": "critical",
            "budget": 250000,
            "sponsor": "CEO"
        }
    )
    
    # Register key resources
    
    # Human resources
    project_orchestrator.register_resource(
        resource_type=ResourceType.HUMAN,
        name="Strategy Team",
        capacity=1.0,
        project_id=project.project_id,
        cost_per_unit=150.0,  # hourly rate
        tags=["strategy", "executive"],
        constraints={"availability": "limited", "max_hours_per_week": 10}
    )
    
    project_orchestrator.register_resource(
        resource_type=ResourceType.HUMAN,
        name="Research Team",
        capacity=1.0,
        project_id=project.project_id,
        cost_per_unit=100.0,  # hourly rate
        tags=["research", "analysis"],
        constraints={"availability": "standard", "max_hours_per_week": 20}
    )
    
    # Technology resources
    project_orchestrator.register_resource(
        resource_type=ResourceType.API_ACCESS,
        name="Market Data API",
        capacity=5000,  # API calls
        project_id=project.project_id,
        cost_per_unit=0.05,  # per call
        tags=["data", "market_research"],
        constraints={"rate_limit": 100, "daily_limit": 1000}
    )
    
    project_orchestrator.register_resource(
        resource_type=ResourceType.COMPUTE,
        name="Data Analysis Cluster",
        capacity=100.0,  # compute hours
        project_id=project.project_id,
        cost_per_unit=2.0,  # per compute hour
        tags=["analysis", "modeling"],
        constraints={"max_concurrent_jobs": 5}
    )
    
    # Budget resources
    project_orchestrator.register_resource(
        resource_type=ResourceType.CUSTOM,
        name="Research Budget",
        capacity=50000.0,  # dollars
        project_id=project.project_id,
        cost_per_unit=1.0,  # dollar for dollar
        tags=["budget", "research"],
        constraints={"approval_required_above": 5000}
    )
    
    # Create project milestones
    
    # Milestone 1: Market Research and Analysis
    milestone1 = project_orchestrator.add_project_milestone(
        project_id=project.project_id,
        name="Market Research and Analysis",
        description="Complete comprehensive market research and analysis",
        target_date=(now + timedelta(days=30)).strftime("%Y-%m-%d"),
        completion_criteria={
            "required_deliverables": [
                "market_size_report", 
                "target_segment_analysis", 
                "competitor_analysis"
            ],
            "approval_required": ["cmo@company.com"]
        },
        stakeholders=["cmo@company.com", "ceo@company.com"]
    )
    
    # Milestone 2: Strategy Development
    milestone2 = project_orchestrator.add_project_milestone(
        project_id=project.project_id,
        name="Strategy Development",
        description="Develop core business strategy based on market analysis",
        target_date=(now + timedelta(days=50)).strftime("%Y-%m-%d"),
        dependencies=[milestone1.milestone_id],  # Depends on market research
        completion_criteria={
            "required_deliverables": [
                "strategic_positioning", 
                "value_proposition", 
                "go_to_market_strategy"
            ],
            "approval_required": ["ceo@company.com"]
        },
        stakeholders=["ceo@company.com", "cmo@company.com", "board@company.com"]
    )
    
    # Milestone 3: Financial Projection and ROI Analysis
    milestone3 = project_orchestrator.add_project_milestone(
        project_id=project.project_id,
        name="Financial Projection and ROI Analysis",
        description="Complete financial projections and ROI analysis for the new strategy",
        target_date=(now + timedelta(days=65)).strftime("%Y-%m-%d"),
        dependencies=[milestone2.milestone_id],  # Depends on strategy development
        completion_criteria={
            "required_deliverables": [
                "financial_model", 
                "roi_analysis", 
                "budget_requirements"
            ],
            "approval_required": ["cfo@company.com"]
        },
        stakeholders=["cfo@company.com", "ceo@company.com", "board@company.com"]
    )
    
    # Milestone 4: Implementation Planning
    milestone4 = project_orchestrator.add_project_milestone(
        project_id=project.project_id,
        name="Implementation Planning",
        description="Develop detailed implementation plan and timeline",
        target_date=(now + timedelta(days=80)).strftime("%Y-%m-%d"),
        dependencies=[milestone2.milestone_id, milestone3.milestone_id],  # Depends on strategy and financials
        completion_criteria={
            "required_deliverables": [
                "implementation_roadmap", 
                "resource_plan", 
                "risk_mitigation_plan"
            ],
            "approval_required": ["ceo@company.com", "cfo@company.com"]
        },
        stakeholders=["ceo@company.com", "cfo@company.com", "cmo@company.com"]
    )
    
    # Milestone 5: Executive Presentation and Final Approval
    milestone5 = project_orchestrator.add_project_milestone(
        project_id=project.project_id,
        name="Executive Presentation and Final Approval",
        description="Present final strategy to executives and board for approval",
        target_date=(now + timedelta(days=90)).strftime("%Y-%m-%d"),
        dependencies=[milestone4.milestone_id],  # Depends on implementation planning
        completion_criteria={
            "required_deliverables": [
                "executive_presentation", 
                "comprehensive_strategy_document", 
                "approval_document"
            ],
            "approval_required": ["ceo@company.com", "board@company.com"]
        },
        stakeholders=["ceo@company.com", "cfo@company.com", "cmo@company.com", "board@company.com"]
    )
    
    logger.info(f"Created business strategy project with 5 milestones: {project.project_id}")
    return project


def setup_milestone_tasks(
    project_orchestrator: ProjectOrchestrator, 
    project: Project
) -> Dict[str, List[str]]:
    """
    Set up tasks for each milestone in the business strategy project.
    
    Args:
        project_orchestrator: The ProjectOrchestrator instance
        project: The created Project
        
    Returns:
        Dictionary mapping milestone IDs to lists of task IDs
    """
    milestone_tasks = {}
    
    # Get milestone IDs (assuming the order they were created)
    milestone_ids = list(project.milestones.keys())
    if len(milestone_ids) < 5:
        logger.error("Expected 5 milestones but found fewer")
        return {}
    
    market_research_id = milestone_ids[0]
    strategy_dev_id = milestone_ids[1]
    financial_id = milestone_ids[2]
    implementation_id = milestone_ids[3]
    final_presentation_id = milestone_ids[4]
    
    # Create tasks for Milestone 1: Market Research and Analysis
    market_research_tasks = []
    
    # Task 1.1: Market Size and Growth Analysis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=market_research_id,
        title="Market Size and Growth Analysis",
        description=(
            "Research and analyze the target market size, growth trends, and "
            "key drivers affecting the industry."
        ),
        task_type=TaskType.RESEARCH,
        required_capabilities=["market_research", "data_analysis"],
        priority=TaskPriority.HIGH,
        estimated_duration=80  # minutes
    )
    market_research_tasks.append(task.task_id)
    
    # Task 1.2: Target Customer Segmentation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=market_research_id,
        title="Target Customer Segmentation",
        description=(
            "Identify and analyze key customer segments, including needs, "
            "preferences, and buying behaviors."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["market_research", "data_analysis"],
        priority=TaskPriority.HIGH,
        estimated_duration=100  # minutes
    )
    market_research_tasks.append(task.task_id)
    
    # Task 1.3: Competitive Landscape Analysis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=market_research_id,
        title="Competitive Landscape Analysis",
        description=(
            "Research and analyze key competitors, their market positions, "
            "strategies, strengths, and weaknesses."
        ),
        task_type=TaskType.RESEARCH,
        required_capabilities=["competitive_analysis", "market_research"],
        priority=TaskPriority.HIGH,
        estimated_duration=120  # minutes
    )
    market_research_tasks.append(task.task_id)
    
    # Task 1.4: Market Trend Identification
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=market_research_id,
        title="Market Trend Identification",
        description=(
            "Identify and analyze emerging trends, technologies, and "
            "regulatory changes affecting the market."
        ),
        task_type=TaskType.RESEARCH,
        required_capabilities=["trend_analysis", "market_research"],
        priority=TaskPriority.MEDIUM,
        estimated_duration=90  # minutes
    )
    market_research_tasks.append(task.task_id)
    
    # Task 1.5: Market Research Synthesis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=market_research_id,
        title="Market Research Synthesis",
        description=(
            "Synthesize all market research findings into a cohesive report "
            "with key insights and implications."
        ),
        task_type=TaskType.AGGREGATION,
        required_capabilities=["data_analysis", "report_writing"],
        priority=TaskPriority.HIGH,
        dependencies=market_research_tasks.copy(),  # Depends on all previous tasks
        estimated_duration=120  # minutes
    )
    market_research_tasks.append(task.task_id)
    
    milestone_tasks[market_research_id] = market_research_tasks
    
    # Create tasks for Milestone 2: Strategy Development
    strategy_dev_tasks = []
    
    # Task 2.1: Strategic Positioning Development
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=strategy_dev_id,
        title="Strategic Positioning Development",
        description=(
            "Develop strategic positioning based on market analysis, identifying "
            "the unique value proposition and competitive advantages."
        ),
        task_type=TaskType.DECISION,
        required_capabilities=["business_strategy", "competitive_analysis"],
        priority=TaskPriority.CRITICAL,
        dependencies=[market_research_tasks[-1]],  # Depends on market research synthesis
        estimated_duration=150  # minutes
    )
    strategy_dev_tasks.append(task.task_id)
    
    # Task 2.2: Go-to-Market Strategy
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=strategy_dev_id,
        title="Go-to-Market Strategy",
        description=(
            "Develop a comprehensive go-to-market strategy including channels, "
            "pricing, positioning, and initial marketing approach."
        ),
        task_type=TaskType.DECISION,
        required_capabilities=["business_strategy", "market_analysis"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[0]],  # Depends on strategic positioning
        estimated_duration=180  # minutes
    )
    strategy_dev_tasks.append(task.task_id)
    
    # Task 2.3: Product/Service Offering Strategy
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=strategy_dev_id,
        title="Product/Service Offering Strategy",
        description=(
            "Define the product or service offering strategy, including features, "
            "benefits, and differentiation points."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["business_strategy", "product_development"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[0]],  # Depends on strategic positioning
        estimated_duration=120  # minutes
    )
    strategy_dev_tasks.append(task.task_id)
    
    # Task 2.4: Growth Strategy Framework
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=strategy_dev_id,
        title="Growth Strategy Framework",
        description=(
            "Develop a framework for sustainable growth, including customer "
            "acquisition, retention, and expansion strategies."
        ),
        task_type=TaskType.DECISION,
        required_capabilities=["business_strategy", "market_analysis"],
        priority=TaskPriority.MEDIUM,
        dependencies=[strategy_dev_tasks[1]],  # Depends on go-to-market strategy
        estimated_duration=150  # minutes
    )
    strategy_dev_tasks.append(task.task_id)
    
    # Task 2.5: Strategy Document Compilation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=strategy_dev_id,
        title="Strategy Document Compilation",
        description=(
            "Compile all strategy components into a comprehensive strategy document "
            "with executive summary and key recommendations."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["report_writing", "business_strategy"],
        priority=TaskPriority.HIGH,
        dependencies=strategy_dev_tasks.copy(),  # Depends on all previous strategy tasks
        estimated_duration=120  # minutes
    )
    strategy_dev_tasks.append(task.task_id)
    
    milestone_tasks[strategy_dev_id] = strategy_dev_tasks
    
    # Create tasks for Milestone 3: Financial Projection and ROI Analysis
    financial_tasks = []
    
    # Task 3.1: Revenue Projection Model
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=financial_id,
        title="Revenue Projection Model",
        description=(
            "Develop detailed revenue projections based on market size, penetration "
            "rates, pricing strategy, and growth assumptions."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["financial_analysis", "forecasting"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[-1]],  # Depends on strategy document
        estimated_duration=180  # minutes
    )
    financial_tasks.append(task.task_id)
    
    # Task 3.2: Cost Structure Analysis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=financial_id,
        title="Cost Structure Analysis",
        description=(
            "Analyze the cost structure including fixed and variable costs, "
            "cost of customer acquisition, and operational expenses."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["financial_analysis", "budget_planning"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[-1]],  # Depends on strategy document
        estimated_duration=150  # minutes
    )
    financial_tasks.append(task.task_id)
    
    # Task 3.3: ROI and Break-even Analysis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=financial_id,
        title="ROI and Break-even Analysis",
        description=(
            "Calculate return on investment, break-even point, and payback period "
            "based on financial projections."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["roi_calculation", "financial_analysis"],
        priority=TaskPriority.HIGH,
        dependencies=[financial_tasks[0], financial_tasks[1]],  # Depends on revenue and cost analyses
        estimated_duration=120  # minutes
    )
    financial_tasks.append(task.task_id)
    
    # Task 3.4: Sensitivity Analysis
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=financial_id,
        title="Sensitivity Analysis",
        description=(
            "Conduct sensitivity analysis to identify key financial risks and "
            "their potential impact on profitability and ROI."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["financial_analysis", "risk_assessment"],
        priority=TaskPriority.MEDIUM,
        dependencies=[financial_tasks[2]],  # Depends on ROI analysis
        estimated_duration=120  # minutes
    )
    financial_tasks.append(task.task_id)
    
    # Task 3.5: Financial Summary Report
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=financial_id,
        title="Financial Summary Report",
        description=(
            "Create a comprehensive financial summary report with key metrics, "
            "projections, and recommendations for financial viability."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["report_writing", "financial_analysis"],
        priority=TaskPriority.HIGH,
        dependencies=financial_tasks.copy(),  # Depends on all previous financial tasks
        estimated_duration=150  # minutes
    )
    financial_tasks.append(task.task_id)
    
    milestone_tasks[financial_id] = financial_tasks
    
    # Create tasks for Milestone 4: Implementation Planning
    implementation_tasks = []
    
    # Task 4.1: Resource Requirements Planning
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=implementation_id,
        title="Resource Requirements Planning",
        description=(
            "Identify and plan for all necessary resources including personnel, "
            "technology, partnerships, and infrastructure."
        ),
        task_type=TaskType.ORCHESTRATION,
        required_capabilities=["implementation_planning", "resource_management"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[-1], financial_tasks[-1]],  # Depends on strategy and financial reports
        estimated_duration=150  # minutes
    )
    implementation_tasks.append(task.task_id)
    
    # Task 4.2: Timeline and Milestone Planning
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=implementation_id,
        title="Timeline and Milestone Planning",
        description=(
            "Develop a detailed implementation timeline with key milestones, "
            "dependencies, and critical path analysis."
        ),
        task_type=TaskType.ORCHESTRATION,
        required_capabilities=["timeline_planning", "implementation_planning"],
        priority=TaskPriority.HIGH,
        dependencies=[implementation_tasks[0]],  # Depends on resource planning
        estimated_duration=120  # minutes
    )
    implementation_tasks.append(task.task_id)
    
    # Task 4.3: Risk Assessment and Mitigation Planning
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=implementation_id,
        title="Risk Assessment and Mitigation Planning",
        description=(
            "Identify potential implementation risks and develop mitigation "
            "strategies to address them proactively."
        ),
        task_type=TaskType.ANALYSIS,
        required_capabilities=["risk_assessment", "implementation_planning"],
        priority=TaskPriority.MEDIUM,
        dependencies=[implementation_tasks[1]],  # Depends on timeline planning
        estimated_duration=150  # minutes
    )
    implementation_tasks.append(task.task_id)
    
    # Task 4.4: KPI and Success Metrics Definition
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=implementation_id,
        title="KPI and Success Metrics Definition",
        description=(
            "Define key performance indicators and success metrics to track "
            "and measure implementation effectiveness."
        ),
        task_type=TaskType.DECISION,
        required_capabilities=["business_strategy", "performance_measurement"],
        priority=TaskPriority.MEDIUM,
        dependencies=[strategy_dev_tasks[-1]],  # Depends on strategy document
        estimated_duration=90  # minutes
    )
    implementation_tasks.append(task.task_id)
    
    # Task 4.5: Implementation Plan Compilation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=implementation_id,
        title="Implementation Plan Compilation",
        description=(
            "Compile all implementation components into a comprehensive plan "
            "with resource requirements, timeline, risks, and KPIs."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["report_writing", "implementation_planning"],
        priority=TaskPriority.HIGH,
        dependencies=implementation_tasks.copy(),  # Depends on all previous implementation tasks
        estimated_duration=120  # minutes
    )
    implementation_tasks.append(task.task_id)
    
    milestone_tasks[implementation_id] = implementation_tasks
    
    # Create tasks for Milestone 5: Executive Presentation and Final Approval
    presentation_tasks = []
    
    # Task 5.1: Executive Summary Creation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=final_presentation_id,
        title="Executive Summary Creation",
        description=(
            "Create a concise executive summary highlighting key strategy points, "
            "financials, and implementation plan."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["report_writing", "executive_communication"],
        priority=TaskPriority.CRITICAL,
        dependencies=[strategy_dev_tasks[-1], financial_tasks[-1], implementation_tasks[-1]],
        estimated_duration=120  # minutes
    )
    presentation_tasks.append(task.task_id)
    
    # Task 5.2: Presentation Deck Creation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=final_presentation_id,
        title="Presentation Deck Creation",
        description=(
            "Develop a compelling presentation deck for executive and board audiences "
            "with clear visualizations and key messages."
        ),
        task_type=TaskType.GENERATION,
        required_capabilities=["presentation_creation", "visual_design"],
        priority=TaskPriority.CRITICAL,
        dependencies=[presentation_tasks[0]],  # Depends on executive summary
        estimated_duration=180  # minutes
    )
    presentation_tasks.append(task.task_id)
    
    # Task 5.3: Supporting Documentation Compilation
    task = project_orchestrator.create_project_task(
        project_id=project.project_id,
        milestone_id=final_presentation_id,
        title="Supporting Documentation Compilation",
        description=(
            "Compile all supporting documentation, appendices, and reference materials "
            "to substantiate the strategy and implementation plan."
        ),
        task_type=TaskType.AGGREGATION,
        required_capabilities=["report_writing", "data_organization"],
        priority=TaskPriority.HIGH,
        dependencies=[strategy_dev_tasks[-1], financial_tasks[-1], implementation_tasks[-1]],
        estimated_duration=150  # minutes
    )
    presentation_tasks.append(task.task_id)
    
    # Task 5.4: Presentation Rehearsal and Refin