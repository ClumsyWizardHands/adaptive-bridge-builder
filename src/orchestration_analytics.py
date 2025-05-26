import html
"""
Orchestration Analytics System

This module provides a comprehensive analytics system for tracking, analyzing, and
optimizing the performance of orchestrated agent workflows. It collects metrics across
all orchestration activities, identifies bottlenecks and inefficiencies, recommends
optimization strategies, provides visualizations, measures principle alignment, and
supports continuous improvement of the orchestration process.

The OrchestrationAnalytics system implements the "Empirical Grace" principle by
gracefully measuring performance while maintaining a deep understanding of the
human and agent contexts in which work is performed.
"""

import json
import uuid
import time
import logging
import heapq
from datetime import datetime, timedelta, timezone
import threading
import copy
import math
import statistics
from collections import defaultdict, Counter, deque
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable, NamedTuple, DefaultDict
from enum import Enum, auto
from dataclasses import dataclass, field

# Import related modules
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability, 
    DependencyType, TaskDecompositionStrategy, DecomposedTask, AgentProfile
)
from project_orchestrator import (
    ProjectOrchestrator, Project, Milestone, Resource, ResourceType,
    MilestoneStatus, ProjectIssue, StatusUpdate
)
from collaborative_task_handler import Task, TaskStatus, TaskPriority
from principle_engine import PrincipleEngine
from communication_adapter import CommunicationAdapter
from relationship_tracker import RelationshipTracker
from content_handler import ContentHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OrchestrationAnalytics")


class MetricType(Enum):
    """Types of metrics tracked by the analytics system."""
    PERFORMANCE = "performance"       # Time-based performance metrics
    EFFICIENCY = "efficiency"         # Resource efficiency metrics
    QUALITY = "quality"               # Output quality metrics
    ALIGNMENT = "alignment"           # Principle alignment metrics
    INTERACTION = "interaction"       # Agent interaction metrics
    BOTTLENECK = "bottleneck"         # Bottleneck-related metrics
    RESILIENCE = "resilience"         # Error handling and recovery metrics
    RESOURCE = "resource"             # Resource utilization metrics
    PROGRESS = "progress"             # Task completion and progress metrics
    SATISFACTION = "satisfaction"     # Agent/human satisfaction metrics


class AnalysisPeriod(Enum):
    """Time periods for analytics aggregation."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class BottleneckType(Enum):
    """Types of bottlenecks that can be identified."""
    AGENT_CAPACITY = "agent_capacity"        # Agent is overloaded
    RESOURCE_CONTENTION = "resource_contention"  # Resource conflict
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"  # Task waiting on dependencies
    COMMUNICATION_LATENCY = "communication_latency"  # Slow communication
    TASK_COMPLEXITY = "task_complexity"      # Task is too complex
    SKILL_MISMATCH = "skill_mismatch"        # Agent lacks required skills
    COORDINATION_OVERHEAD = "coordination_overhead"  # Too much coordination required
    EXTERNAL_DEPENDENCY = "external_dependency"  # Waiting on external system


class RecommendationCategory(Enum):
    """Categories of optimization recommendations."""
    AGENT_ALLOCATION = "agent_allocation"    # Changes to agent assignments
    TASK_DECOMPOSITION = "task_decomposition"  # Different task breakdown
    DEPENDENCY_MANAGEMENT = "dependency_management"  # Dependency changes
    RESOURCE_ALLOCATION = "resource_allocation"  # Resource distribution
    SCHEDULING = "scheduling"                # Timing changes
    CAPABILITY_ENHANCEMENT = "capability_enhancement"  # Add agent capabilities
    PROCESS_IMPROVEMENT = "process_improvement"  # Change orchestration process
    PRINCIPLE_ALIGNMENT = "principle_alignment"  # Better align with principles


class VisualizationType(Enum):
    """Types of visualizations supported by the analytics system."""
    TIMELINE = "timeline"                 # Timeline of task execution
    GANTT = "gantt"                       # Gantt chart of tasks and dependencies
    NETWORK = "network"                   # Network diagram of agent interactions
    HEATMAP = "heatmap"                   # Heatmap of activity or performance
    BAR_CHART = "bar_chart"               # Bar chart of comparative metrics
    LINE_CHART = "line_chart"             # Line chart of trends over time
    SCATTER_PLOT = "scatter_plot"         # Scatter plot of relationships
    SANKEY = "sankey"                     # Sankey diagram of flows
    RADAR = "radar"                       # Radar chart of multi-dimensional metrics
    TREEMAP = "treemap"                   # Treemap of hierarchical data


@dataclass
class MetricDefinition:
    """Definition of a metric to be tracked by the analytics system."""
    metric_id: str
    name: str
    description: str
    type: MetricType
    unit: str                           # e.g., "ms", "percent", "count"
    aggregation_method: str             # e.g., "avg", "sum", "min", "max"
    ideal_trend: str                    # "increase", "decrease", "maintain"
    warning_threshold: Optional[float] = None  # Threshold for warnings
    critical_threshold: Optional[float] = None  # Threshold for critical alerts
    related_principle: Optional[str] = None  # Related principle ID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDataPoint:
    """A single data point for a tracked metric."""
    timestamp: str                       # ISO format timestamp
    value: float                         # Metric value
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class MetricTimeSeries:
    """Time series data for a specific metric."""
    metric_id: str
    data_points: List[MetricDataPoint] = field(default_factory=list)
    
    def add_data_point(
        self, 
        value: float, 
        timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new data point to the time series."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
            
        self.data_points.append(MetricDataPoint(
            timestamp=timestamp,
            value=value,
            context=context or {}
        ))
    
    def get_values(self) -> List[float]:
        """Get all values in the time series."""
        return [dp.value for dp in self.data_points]
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent value in the time series."""
        if not self.data_points:
            return None
        return self.data_points[-1].value
    
    def get_trend(self, window: int = 10) -> Optional[float]:
        """
        Calculate the trend over the last 'window' data points.
        
        Returns:
            Trend coefficient or None if insufficient data
        """
        values = self.get_values()
        if len(values) < window:
            return None
            
        recent_values = values[-window:]
        indices = list(range(len(recent_values)))
        
        # Simple linear regression
        n = len(recent_values)
        sum_x = sum(indices)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(indices, recent_values))
        sum_xx = sum(x * x for x in indices)
        
        # Calculate slope
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0


@dataclass
class BottleneckAnalysis:
    """Analysis of an identified bottleneck in the orchestration process."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    severity: float                      # 0.0 (minor) to 1.0 (critical)
    detected_at: str                     # ISO format timestamp
    affected_items: Dict[str, List[str]] = field(default_factory=dict)  # Type -> IDs map
    metrics: Dict[str, float] = field(default_factory=dict)  # Relevant metrics
    impact_assessment: str = ""          # Description of impact
    root_cause_analysis: str = ""        # Description of root cause
    estimated_impact: Dict[str, Any] = field(default_factory=dict)  # Performance impact
    recommendations: List[str] = field(default_factory=list)  # Recommendation IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """A recommended optimization to improve orchestration performance."""
    recommendation_id: str
    category: RecommendationCategory
    title: str
    description: str
    created_at: str                      # ISO format timestamp
    estimated_impact: Dict[str, float] = field(default_factory=dict)  # Metric -> impact map
    implementation_complexity: float = 0.5  # 0.0 (simple) to 1.0 (complex)
    priority: float = 0.5                # 0.0 (low) to 1.0 (high)
    related_bottlenecks: List[str] = field(default_factory=list)  # Bottleneck IDs
    implementation_steps: List[str] = field(default_factory=list)  # Implementation guide
    preconditions: Dict[str, Any] = field(default_factory=dict)  # Required conditions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrincipleAlignmentMeasurement:
    """Measurement of alignment with a specific principle."""
    principle_id: str
    timestamp: str                       # ISO format timestamp
    alignment_score: float               # 0.0 (misaligned) to 1.0 (fully aligned)
    contributing_factors: Dict[str, float] = field(default_factory=dict)  # Factor -> weight
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # Supporting evidence
    improvement_opportunities: List[str] = field(default_factory=list)  # Areas to improve
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class VisualizationRequest:
    """Request for generating a visualization of orchestration analytics."""
    visualization_type: VisualizationType
    title: str
    description: str
    time_range: Tuple[str, str]          # (start, end) timestamps
    data_sources: Dict[str, Any]         # Configuration of data sources
    filters: Dict[str, Any] = field(default_factory=dict)  # Data filters
    parameters: Dict[str, Any] = field(default_factory=dict)  # Visualization parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of a visualization generation request."""
    request_id: str
    visualization_type: VisualizationType
    created_at: str                      # ISO format timestamp
    data: Any                            # Visualization data (format depends on type)
    render_format: str                   # "svg", "png", "html", "json", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestrationAnalytics:
    """
    Comprehensive analytics system for tracking and optimizing orchestrated agent workflows.
    
    The OrchestrationAnalytics system collects metrics across all orchestration activities,
    identifies bottlenecks and inefficiencies, recommends optimization strategies, provides
    visualizations, measures principle alignment, and supports continuous improvement.
    """
    
    def __init__(
        self,
        agent_id: str,
        orchestrator_engine: Optional[OrchestratorEngine] = None,
        project_orchestrator: Optional[ProjectOrchestrator] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        storage_dir: str = "data/analytics"
    ):
        """
        Initialize the orchestration analytics system.
        
        Args:
            agent_id: ID of the analytics agent
            orchestrator_engine: Existing OrchestratorEngine or None
            project_orchestrator: Existing ProjectOrchestrator or None
            principle_engine: Engine for principle-based reasoning
            storage_dir: Directory for storing analytics data
        """
        self.agent_id = agent_id
        self.orchestrator_engine = orchestrator_engine
        self.project_orchestrator = project_orchestrator
        self.principle_engine = principle_engine
        self.storage_dir = storage_dir
        
        # Metric definitions and data
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_data: Dict[str, MetricTimeSeries] = {}
        
        # Bottleneck analysis
        self.bottleneck_analyses: Dict[str, BottleneckAnalysis] = {}
        
        # Optimization recommendations
        self.recommendations: Dict[str, OptimizationRecommendation] = {}
        
        # Principle alignment tracking
        self.principle_alignment: Dict[str, List[PrincipleAlignmentMeasurement]] = {}
        
        # Visualizations
        self.visualization_results: Dict[str, VisualizationResult] = {}
        
        # Historical performance data
        self.historical_data: Dict[str, Any] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 60  # seconds
        
        # Locks
        self.metric_lock = threading.Lock()
        self.bottleneck_lock = threading.Lock()
        self.recommendation_lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info(f"OrchestrationAnalytics initialized for agent {agent_id}")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics for tracking orchestration performance."""
        
        # Performance metrics
        self.register_metric(
            name="Task Processing Time",
            description="Average time to process a task from assignment to completion",
            type=MetricType.PERFORMANCE,
            unit="seconds",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=60.0,
            critical_threshold=300.0
        )
        
        self.register_metric(
            name="Agent Response Time",
            description="Average time for an agent to respond to a task assignment",
            type=MetricType.PERFORMANCE,
            unit="seconds",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=10.0,
            critical_threshold=30.0
        )
        
        self.register_metric(
            name="Task Queue Time",
            description="Average time tasks spend in queue before assignment",
            type=MetricType.PERFORMANCE,
            unit="seconds",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=30.0,
            critical_threshold=120.0
        )
        
        # Efficiency metrics
        self.register_metric(
            name="Agent Utilization",
            description="Percentage of agent capacity being utilized",
            type=MetricType.EFFICIENCY,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="maintain",
            warning_threshold=85.0,
            critical_threshold=95.0
        )
        
        self.register_metric(
            name="Task Parallelism",
            description="Average number of tasks executing in parallel",
            type=MetricType.EFFICIENCY,
            unit="count",
            aggregation_method="avg",
            ideal_trend="increase"
        )
        
        self.register_metric(
            name="Resource Utilization",
            description="Percentage of available resources being utilized",
            type=MetricType.RESOURCE,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="maintain",
            warning_threshold=85.0,
            critical_threshold=95.0
        )
        
        # Quality metrics
        self.register_metric(
            name="Task Success Rate",
            description="Percentage of tasks completed successfully",
            type=MetricType.QUALITY,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=90.0,
            critical_threshold=80.0
        )
        
        self.register_metric(
            name="Error Rate",
            description="Percentage of tasks that result in errors",
            type=MetricType.QUALITY,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=5.0,
            critical_threshold=10.0
        )
        
        # Alignment metrics
        self.register_metric(
            name="Principle Alignment Score",
            description="Overall alignment with defined principles",
            type=MetricType.ALIGNMENT,
            unit="score",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=0.7,
            critical_threshold=0.5,
            related_principle="all"
        )
        
        # Interaction metrics
        self.register_metric(
            name="Communication Volume",
            description="Number of messages exchanged between agents",
            type=MetricType.INTERACTION,
            unit="count",
            aggregation_method="sum",
            ideal_trend="optimize"
        )
        
        self.register_metric(
            name="Coordination Overhead",
            description="Percentage of time spent on coordination vs. execution",
            type=MetricType.INTERACTION,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=30.0,
            critical_threshold=50.0
        )
        
        # Bottleneck metrics
        self.register_metric(
            name="Critical Path Delay",
            description="Delays in critical path task execution",
            type=MetricType.BOTTLENECK,
            unit="seconds",
            aggregation_method="sum",
            ideal_trend="decrease",
            warning_threshold=60.0,
            critical_threshold=300.0
        )
        
        self.register_metric(
            name="Resource Contention Count",
            description="Number of resource contention incidents",
            type=MetricType.BOTTLENECK,
            unit="count",
            aggregation_method="sum",
            ideal_trend="decrease",
            warning_threshold=5.0,
            critical_threshold=10.0
        )
        
        # Resilience metrics
        self.register_metric(
            name="Recovery Time",
            description="Average time to recover from task failures",
            type=MetricType.RESILIENCE,
            unit="seconds",
            aggregation_method="avg",
            ideal_trend="decrease",
            warning_threshold=120.0,
            critical_threshold=300.0
        )
        
        self.register_metric(
            name="Adaptability Score",
            description="System's ability to adapt to changing conditions",
            type=MetricType.RESILIENCE,
            unit="score",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=0.6,
            critical_threshold=0.4
        )
        
        # Progress metrics
        self.register_metric(
            name="Task Completion Rate",
            description="Number of tasks completed per hour",
            type=MetricType.PROGRESS,
            unit="tasks/hour",
            aggregation_method="avg",
            ideal_trend="increase"
        )
        
        self.register_metric(
            name="Milestone Completion Rate",
            description="Percentage of milestones completed on time",
            type=MetricType.PROGRESS,
            unit="percent",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=80.0,
            critical_threshold=60.0
        )
        
        # Satisfaction metrics
        self.register_metric(
            name="Agent Satisfaction Score",
            description="Composite score of agent satisfaction with orchestration",
            type=MetricType.SATISFACTION,
            unit="score",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=0.7,
            critical_threshold=0.5
        )
    
    def register_metric(
        self,
        name: str,
        description: str,
        type: MetricType,
        unit: str,
        aggregation_method: str,
        ideal_trend: str,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        related_principle: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new metric to be tracked by the analytics system.
        
        Args:
            name: Human-readable name of the metric
            description: Description of what the metric measures
            type: Type of metric (performance, efficiency, etc.)
            unit: Unit of measurement (seconds, percent, count, etc.)
            aggregation_method: How the metric should be aggregated
            ideal_trend: Whether higher or lower values are better
            warning_threshold: Threshold for warnings
            critical_threshold: Threshold for critical alerts
            related_principle: ID of related principle, if any
            metadata: Additional metadata about the metric
            
        Returns:
            ID of the registered metric
        """
        metric_id = f"metric-{str(uuid.uuid4())}"
        
        with self.metric_lock:
            metric_def = MetricDefinition(
                metric_id=metric_id,
                name=name,
                description=description,
                type=type,
                unit=unit,
                aggregation_method=aggregation_method,
                ideal_trend=ideal_trend,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                related_principle=related_principle,
                metadata=metadata or {}
            )
            
            self.metric_definitions = {**self.metric_definitions, metric_id: metric_def}
            self.metric_data = {**self.metric_data, metric_id: MetricTimeSeries(metric_id=metric_id)}
            
            logger.info(f"Registered metric '{name}' with ID {metric_id}")
            
            return metric_id
    
    def record_metric(
        self,
        metric_id: str,
        value: float,
        timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a data point for a specific metric.
        
        Args:
            metric_id: ID of the metric
            value: Metric value to record
            timestamp: ISO format timestamp (default: current time)
            context: Additional context about the data point
            
        Returns:
            Whether the recording was successful
        """
        with self.metric_lock:
            if metric_id not in self.metric_data:
                logger.error(f"Metric {metric_id} not found")
                return False
                
            self.metric_data[metric_id].add_data_point(
                value=value,
                timestamp=timestamp,
                context=context
            )
            
            return True
    
    def get_metric_value(
        self,
        metric_id: str,
        aggregation_period: AnalysisPeriod = AnalysisPeriod.DAILY,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Optional[float]:
        """
        Get an aggregated value for a metric over the specified period.
        
        Args:
            metric_id: ID of the metric
            aggregation_period: Period for aggregation
            start_time: Start of period (ISO format)
            end_time: End of period (ISO format)
            
        Returns:
            Aggregated metric value or None if not available
        """
        with self.metric_lock:
            if metric_id not in self.metric_data:
                logger.error(f"Metric {metric_id} not found")
                return None
                
            metric_def = self.metric_definitions[metric_id]
            time_series = self.metric_data[metric_id]
            
            # Filter data points by time range
            filtered_points = time_series.data_points
            
            if start_time or end_time:
                now = datetime.now(timezone.utc).isoformat()
                start = start_time or "0001-01-01T00:00:00Z"
                end = end_time or now
                
                filtered_points = [
                    dp for dp in time_series.data_points
                    if start <= dp.timestamp <= end
                ]
            
            if not filtered_points:
                return None
                
            # Aggregate based on method
            values = [dp.value for dp in filtered_points]
            
            if metric_def.aggregation_method == "avg":
                return sum(values) / len(values)
            elif metric_def.aggregation_method == "sum":
                return sum(values)
            elif metric_def.aggregation_method == "min":
                return min(values)
            elif metric_def.aggregation_method == "max":
                return max(values)
            elif metric_def.aggregation_method == "median":
                return statistics.median(values)
            elif metric_def.aggregation_method == "latest":
                return values[-1]
            else:
                # Default to average
                return sum(values) / len(values)
    
    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """
        Analyze current orchestration performance to identify bottlenecks.
        
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Only proceed if we have an orchestrator engine and metrics data
        if not self.orchestrator_engine:
            logger.warning("No orchestrator engine available for bottleneck analysis")
            return bottlenecks
            
        with self.bottleneck_lock:
            # Analyze agent capacity bottlenecks
            self._analyze_agent_capacity_bottlenecks(bottlenecks)
            
            # Analyze resource contention bottlenecks
            self._analyze_resource_contention_bottlenecks(bottlenecks)
            
            # Analyze dependency bottlenecks
            self._analyze_dependency_bottlenecks(bottlenecks)
            
            # Analyze communication bottlenecks
            self._analyze_communication_bottlenecks(bottlenecks)
            
            # Save bottleneck analyses
            now = datetime.now(timezone.utc).isoformat()
            for bottleneck in bottlenecks:
                bottleneck.detected_at = now
                self.bottleneck_analyses = {**self.bottleneck_analyses, bottleneck.bottleneck_id: bottleneck}
                
            return bottlenecks
    
    def _analyze_agent_capacity_bottlenecks(self, bottlenecks: List[BottleneckAnalysis]) -> None:
        """
        Analyze agent capacity to identify overloaded agents.
        
        Args:
            bottlenecks: List to append identified bottlenecks to
        """
        # Check for agents with high utilization
        for agent_id, profile in self.orchestrator_engine.agent_profiles.items():
            utilization = profile.current_load / profile.max_load if profile.max_load > 0 else 1.0
            
            if utilization >= 0.9:  # 90% capacity or higher
                bottleneck_id = f"bottleneck-agent-capacity-{agent_id}-{str(uuid.uuid4())[:8]}"
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=bottleneck_id,
                    bottleneck_type=BottleneckType.AGENT_CAPACITY,
                    severity=min(1.0, utilization),
                    detected_at="",  # Will be set later
                    affected_items={"agents": [agent_id]},
                    metrics={"utilization": utilization},
                    impact_assessment=f"Agent {agent_id} is at {utilization*100:.1f}% capacity, causing task queuing and delays",
                    root_cause_analysis="High task assignment rate relative to agent processing capacity",
                    estimated_impact={
                        "task_delay": profile.current_load * 60,  # Estimated delay in seconds
                        "affected_tasks": profile.task_history[-5:] if profile.task_history else []
                    }
                )
                
                bottlenecks.append(bottleneck)
    
    def _analyze_resource_contention_bottlenecks(self, bottlenecks: List[BottleneckAnalysis]) -> None:
        """
        Analyze resource usage to identify contention bottlenecks.
        
        Args:
            bottlenecks: List to append identified bottlenecks to
        """
        if not self.project_orchestrator:
            return
            
        # Check for highly contended resources
        for project_id, project in self.project_orchestrator.projects.items():
            for resource_id, resource in project.resources.items():
                utilization = resource.utilization_percentage() / 100.0
                
                if utilization >= 0.9:  # 90% utilization or higher
                    bottleneck_id = f"bottleneck-resource-{resource_id}-{str(uuid.uuid4())[:8]}"
                    
                    # Identify tasks that might be affected
                    affected_tasks = []
                    for milestone in project.milestones.values():
                        for task_id in milestone.task_ids:
                            task = self.orchestrator_engine.task_coordinator.get_task(task_id)
                            if task and "required_resources" in task.metadata:
                                if resource_id in task.metadata["required_resources"]:
                                    affected_tasks.append(task_id)
                    
                    bottleneck = BottleneckAnalysis(
                        bottleneck_id=bottleneck_id,
                        bottleneck_type=BottleneckType.RESOURCE_CONTENTION,
                        severity=min(1.0, utilization),
                        detected_at="",  # Will be set later
                        affected_items={
                            "resources": [resource_id],
                            "tasks": affected_tasks,
                            "projects": [project_id]
                        },
                        metrics={"utilization": utilization},
                        impact_assessment=f"Resource {resource.name} is at {utilization*100:.1f}% utilization, causing contention and delays",
                        root_cause_analysis="Multiple tasks requiring the same resource at the same time",
                        estimated_impact={
                            "task_delay": len(affected_tasks) * 30,  # Estimated delay in seconds
                            "affected_tasks": affected_tasks
                        }
                    )
                    
                    bottlenecks.append(bottleneck)
    
    def _analyze_dependency_bottlenecks(self, bottlenecks: List[BottleneckAnalysis]) -> None:
        """
        Analyze task dependencies to identify bottlenecks in the critical path.
        
        Args:
            bottlenecks: List to append identified bottlenecks to
        """
        # Check for blocked tasks
        blocked_tasks = []
        dependency_map = {}
        
        # Find tasks blocked by dependencies
        if self.orchestrator_engine and hasattr(self.orchestrator_engine, 'task_coordinator'):
            for task_id, task in self.orchestrator_engine.task_coordinator.tasks.items():
                if task.status == TaskStatus.PENDING:
                    # Check if task has unmet dependencies
                    dependencies = task.metadata.get("dependencies", [])
                    if dependencies:
                        blocked_tasks.append(task_id)
                        dependency_map[task_id] = dependencies
        
        # Analyze critical path delays
        if blocked_tasks:
            critical_delay = 0
            for task_id in blocked_tasks:
                # Estimate delay impact
                critical_delay += 60  # Base delay estimate
            
            if critical_delay > 120:  # More than 2 minutes of critical path delay
                bottleneck_id = f"bottleneck-dependency-{str(uuid.uuid4())[:8]}"
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=bottleneck_id,
                    bottleneck_type=BottleneckType.SEQUENTIAL_DEPENDENCY,
                    severity=min(1.0, critical_delay / 600),  # 10 minutes = severity 1.0
                    detected_at="",  # Will be set later
                    affected_items={
                        "tasks": blocked_tasks,
                        "dependencies": list(set(sum(dependency_map.values(), [])))
                    },
                    metrics={"critical_delay": critical_delay},
                    impact_assessment=f"Critical path delayed by {critical_delay} seconds due to sequential dependencies",
                    root_cause_analysis="Tasks arranged in sequential chain preventing parallelization",
                    estimated_impact={
                        "total_delay": critical_delay,
                        "blocked_tasks": blocked_tasks
                    }
                )
                
                bottlenecks.append(bottleneck)
    
    def _analyze_communication_bottlenecks(self, bottlenecks: List[BottleneckAnalysis]) -> None:
        """
        Analyze communication patterns to identify latency and overhead bottlenecks.
        
        Args:
            bottlenecks: List to append identified bottlenecks to
        """
        # Check communication metrics
        comm_volume_metric = None
        comm_overhead_metric = None
        
        # Find communication-related metrics
        for metric_id, metric_def in self.metric_definitions.items():
            if metric_def.name == "Communication Volume":
                comm_volume_metric = metric_id
            elif metric_def.name == "Coordination Overhead":
                comm_overhead_metric = metric_id
        
        # Analyze communication volume
        if comm_volume_metric:
            comm_volume = self.get_metric_value(comm_volume_metric, AnalysisPeriod.HOURLY)
            if comm_volume and comm_volume > 1000:  # More than 1000 messages per hour
                bottleneck_id = f"bottleneck-comm-volume-{str(uuid.uuid4())[:8]}"
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=bottleneck_id,
                    bottleneck_type=BottleneckType.COMMUNICATION_LATENCY,
                    severity=min(1.0, comm_volume / 5000),  # 5000 messages/hour = severity 1.0
                    detected_at="",  # Will be set later
                    affected_items={"system": ["communication_subsystem"]},
                    metrics={"communication_volume": comm_volume},
                    impact_assessment=f"High communication volume ({comm_volume:.0f} messages/hour) causing system overhead",
                    root_cause_analysis="Excessive inter-agent communication or inefficient message patterns",
                    estimated_impact={
                        "overhead_percentage": min(50, comm_volume / 100),
                        "latency_increase": comm_volume / 50  # ms
                    }
                )
                
                bottlenecks.append(bottleneck)
        
        # Analyze coordination overhead
        if comm_overhead_metric:
            coord_overhead = self.get_metric_value(comm_overhead_metric, AnalysisPeriod.HOURLY)
            if coord_overhead and coord_overhead > 40:  # More than 40% coordination overhead
                bottleneck_id = f"bottleneck-coord-overhead-{str(uuid.uuid4())[:8]}"
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=bottleneck_id,
                    bottleneck_type=BottleneckType.COORDINATION_OVERHEAD,
                    severity=min(1.0, coord_overhead / 60),  # 60% overhead = severity 1.0
                    detected_at="",  # Will be set later
                    affected_items={"system": ["orchestration_layer"]},
                    metrics={"coordination_overhead": coord_overhead},
                    impact_assessment=f"Coordination overhead at {coord_overhead:.1f}%, reducing effective work time",
                    root_cause_analysis="Complex task dependencies or inefficient orchestration patterns",
                    estimated_impact={
                        "efficiency_loss": coord_overhead,
                        "effective_capacity": 100 - coord_overhead
                    }
                )
                
                bottlenecks.append(bottleneck)