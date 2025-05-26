"""
Continuous Evolution System

This module implements the ContinuousEvolutionSystem class that enables the agent to learn from
orchestration experiences, refine its approach over time, and develop new capabilities while
maintaining its core identity. It embodies the "Resilience Through Reflection" principle
by systematically evaluating past performance and implementing targeted improvements.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from enum import Enum, auto
from dataclasses import dataclass, field
import statistics
from collections import Counter, defaultdict
import copy
import uuid

from learning_system import (
    LearningSystem, LearningDimension, OutcomeType, 
    InteractionPattern, AdaptationLevel, GrowthJournalEntry
)
from orchestration_analytics import (
    OrchestrationAnalytics, MetricType, BottleneckType, 
    RecommendationCategory
)
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentProfile, 
    DecomposedTask, TaskDecompositionStrategy
)
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ContinuousEvolutionSystem")


class OrchestrationPattern(InteractionPattern):
    """
    An orchestration pattern extends an interaction pattern with specific
    orchestration-related metadata and performance metrics.
    """
    
    def __init__(
        self,
        pattern_id: str,
        description: str,
        context: Dict[str, Any],
        decomposition_strategy: Optional[TaskDecompositionStrategy] = None,
        agent_selection_criteria: Optional[Dict[str, Any]] = None,
        resource_allocation_strategy: Optional[Dict[str, Any]] = None,
        communication_template: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize an orchestration pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            description: Description of the pattern
            context: Context in which the pattern was used
            decomposition_strategy: Task decomposition strategy used
            agent_selection_criteria: Criteria used for agent selection
            resource_allocation_strategy: Resource allocation approach 
            communication_template: Communication patterns used
            performance_metrics: Performance metrics for this pattern
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(pattern_id, description, context, **kwargs)
        
        # Orchestration-specific pattern data
        self.decomposition_strategy = decomposition_strategy
        self.agent_selection_criteria = agent_selection_criteria or {}
        self.resource_allocation_strategy = resource_allocation_strategy or {}
        self.communication_template = communication_template or {}
        self.performance_metrics = performance_metrics or {}
        
        # Track which agents were involved and their performance
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
        # Track which task types were involved and their success rates
        self.task_type_performance: Dict[str, Dict[str, float]] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the orchestration pattern to a dictionary."""
        data = super().to_dict()
        data.update({
            "decomposition_strategy": self.decomposition_strategy.value if self.decomposition_strategy else None,
            "agent_selection_criteria": self.agent_selection_criteria,
            "resource_allocation_strategy": self.resource_allocation_strategy,
            "communication_template": self.communication_template,
            "performance_metrics": self.performance_metrics,
            "agent_performance": self.agent_performance,
            "task_type_performance": self.task_type_performance
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchestrationPattern':
        """Create an OrchestrationPattern from a dictionary."""
        # Extract orchestration-specific fields
        orchestration_fields = {
            "decomposition_strategy": data.get("decomposition_strategy"),
            "agent_selection_criteria": data.get("agent_selection_criteria", {}),
            "resource_allocation_strategy": data.get("resource_allocation_strategy", {}),
            "communication_template": data.get("communication_template", {}),
            "performance_metrics": data.get("performance_metrics", {})
        }
        
        # Convert decomposition_strategy string to enum if present
        if orchestration_fields["decomposition_strategy"]:
            try:
                orchestration_fields["decomposition_strategy"] = TaskDecompositionStrategy(
                    orchestration_fields["decomposition_strategy"]
                )
            except:
                orchestration_fields["decomposition_strategy"] = None
        
        # Extract base pattern fields
        pattern = super().from_dict(data)
        
        # Create new instance with all fields
        result = cls(
            pattern_id=pattern.pattern_id,
            description=pattern.description,
            context=pattern.context,
            occurrences=pattern.occurrences,
            successful_count=pattern.successful_count,
            unsuccessful_count=pattern.unsuccessful_count,
            neutral_count=pattern.neutral_count,
            success_rate=pattern.success_rate,
            confidence=pattern.confidence,
            last_observed=pattern.last_observed,
            first_observed=pattern.first_observed,
            adaptations=pattern.adaptations,
            **orchestration_fields
        )
        
        # Add agent and task type performance data if present
        result.agent_performance = data.get("agent_performance", {})
        result.task_type_performance = data.get("task_type_performance", {})
        
        return result


class OrchestrationDimension(Enum):
    """Specific dimensions for orchestration learning."""
    TASK_DECOMPOSITION = auto()
    AGENT_SELECTION = auto()
    RESOURCE_ALLOCATION = auto()
    DEPENDENCY_MANAGEMENT = auto()
    COMMUNICATION_ADAPTATION = auto()
    ERROR_RECOVERY = auto()
    CAPABILITY_DEVELOPMENT = auto()
    PRINCIPLE_ALIGNMENT = auto()


@dataclass
class CapabilityEvolution:
    """
    Tracks the evolution of a specific capability over time.
    """
    capability_id: str
    name: str
    description: str
    created_at: str
    evolution_stages: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: int = 0
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)
    development_focus: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "evolution_stages": self.evolution_stages,
            "current_stage": self.current_stage,
            "performance_metrics": self.performance_metrics,
            "development_focus": self.development_focus
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapabilityEvolution':
        """Create from dictionary representation."""
        return cls(
            capability_id=data["capability_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"],
            evolution_stages=data.get("evolution_stages", []),
            current_stage=data.get("current_stage", 0),
            performance_metrics=data.get("performance_metrics", {}),
            development_focus=data.get("development_focus")
        )


@dataclass
class GrowthMilestone:
    """
    A significant milestone in the agent's evolution journey.
    """
    milestone_id: str
    title: str
    description: str
    achieved_at: str
    category: str  # "capability", "pattern", "principle", etc.
    impact_score: float  # 0.0 to 1.0
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "milestone_id": self.milestone_id,
            "title": self.title,
            "description": self.description,
            "achieved_at": self.achieved_at,
            "category": self.category,
            "impact_score": self.impact_score,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "references": self.references
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrowthMilestone':
        """Create from dictionary representation."""
        return cls(
            milestone_id=data["milestone_id"],
            title=data["title"],
            description=data["description"],
            achieved_at=data["achieved_at"],
            category=data["category"],
            impact_score=data["impact_score"],
            metrics_before=data.get("metrics_before", {}),
            metrics_after=data.get("metrics_after", {}),
            references=data.get("references", [])
        )


class ContinuousEvolutionSystem:
    """
    System for continuously evolving the agent's capabilities, orchestration patterns,
    and communication approaches based on experience and feedback.
    
    The ContinuousEvolutionSystem extends the LearningSystem with a specific focus on
    orchestration patterns and embodies the "Resilience Through Reflection" principle
    by systematically evaluating performance and implementing improvements.
    """
    
    def __init__(
        self,
        learning_system: Optional[LearningSystem] = None,
        orchestration_analytics: Optional[OrchestrationAnalytics] = None,
        orchestrator_engine: Optional[OrchestratorEngine] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        growth_journal_dir: str = "data/growth_journal",
        evolution_data_dir: str = "data/evolution"
    ):
        """
        Initialize the ContinuousEvolutionSystem.
        
        Args:
            learning_system: Optional LearningSystem for base learning functionality
            orchestration_analytics: Optional OrchestrationAnalytics for metrics
            orchestrator_engine: Optional OrchestratorEngine for orchestration
            principle_engine: Optional PrincipleEngine for principle alignment
            growth_journal_dir: Directory for growth journal
            evolution_data_dir: Directory for evolution data
        """
        # Initialize or store component references
        self.learning_system = learning_system or LearningSystem(
            principle_engine=principle_engine,
            growth_journal_dir=growth_journal_dir
        )
        self.orchestration_analytics = orchestration_analytics
        self.orchestrator_engine = orchestrator_engine
        self.principle_engine = principle_engine or self.learning_system.principle_engine
        
        # Paths for storing evolution data
        self.growth_journal_dir = growth_journal_dir
        self.evolution_data_dir = evolution_data_dir
        os.makedirs(evolution_data_dir, exist_ok=True)
        
        # Orchestration pattern storage
        self.orchestration_patterns: Dict[str, OrchestrationPattern] = {}
        
        # Capability evolution tracking
        self.capabilities: Dict[str, CapabilityEvolution] = {}
        
        # Growth milestones
        self.growth_milestones: List[GrowthMilestone] = []
        
        # Reflection schedule
        self.last_reflection = datetime.now(timezone.utc) - timedelta(days=1)  # Start with reflection due
        self.reflection_frequency = timedelta(hours=6)  # Reflect every 6 hours
        self.last_deep_reflection = datetime.now(timezone.utc) - timedelta(days=7)  # Start with deep reflection due
        self.deep_reflection_frequency = timedelta(days=7)  # Deep reflect weekly
        
        # Load saved data
        self._load_evolution_data()
        
        logger.info("ContinuousEvolutionSystem initialized")
    
    def _load_evolution_data(self) -> None:
        """Load saved evolution data."""
        try:
            # Load orchestration patterns
            patterns_file = os.path.join(self.evolution_data_dir, "orchestration_patterns.json")
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    for pattern_id, pattern_data in patterns_data.items():
                        self.orchestration_patterns = {**self.orchestration_patterns, pattern_id: OrchestrationPattern.from_dict(pattern_data)}
            
            # Load capabilities
            capabilities_file = os.path.join(self.evolution_data_dir, "capabilities.json")
            if os.path.exists(capabilities_file):
                with open(capabilities_file, 'r') as f:
                    capabilities_data = json.load(f)
                    for capability_id, capability_data in capabilities_data.items():
                        self.capabilities = {**self.capabilities, capability_id: CapabilityEvolution.from_dict(capability_data)}
            
            # Load growth milestones
            milestones_file = os.path.join(self.evolution_data_dir, "growth_milestones.json")
            if os.path.exists(milestones_file):
                with open(milestones_file, 'r') as f:
                    milestones_data = json.load(f)
                    for milestone_data in milestones_data:
                        self.growth_milestones = [*self.growth_milestones, GrowthMilestone.from_dict(milestone_data)]
            
            # Load reflection timestamps
            reflection_file = os.path.join(self.evolution_data_dir, "reflection_schedule.json")
            if os.path.exists(reflection_file):
                with open(reflection_file, 'r') as f:
                    reflection_data = json.load(f)
                    self.last_reflection = datetime.fromisoformat(reflection_data.get(
                        "last_reflection", datetime.now(timezone.utc).isoformat()
                    ))
                    self.last_deep_reflection = datetime.fromisoformat(reflection_data.get(
                        "last_deep_reflection", datetime.now(timezone.utc).isoformat()
                    ))
            
            logger.info(f"Loaded {len(self.orchestration_patterns)} orchestration patterns, "
                        f"{len(self.capabilities)} capabilities, and "
                        f"{len(self.growth_milestones)} growth milestones")
        except Exception as e:
            logger.error(f"Error loading evolution data: {e}")
    
    def _save_evolution_data(self) -> None:
        """Save evolution data to files."""
        try:
            # Save orchestration patterns
            patterns_data = {
                pattern_id: pattern.to_dict() 
                for pattern_id, pattern in self.orchestration_patterns.items()
            }
            with open(os.path.join(self.evolution_data_dir, "orchestration_patterns.json"), 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save capabilities
            capabilities_data = {
                capability_id: capability.to_dict() 
                for capability_id, capability in self.capabilities.items()
            }
            with open(os.path.join(self.evolution_data_dir, "capabilities.json"), 'w') as f:
                json.dump(capabilities_data, f, indent=2)
            
            # Save growth milestones
            milestones_data = [milestone.to_dict() for milestone in self.growth_milestones]
            with open(os.path.join(self.evolution_data_dir, "growth_milestones.json"), 'w') as f:
                json.dump(milestones_data, f, indent=2)
            
            # Save reflection timestamps
            reflection_data = {
                "last_reflection": self.last_reflection.isoformat(),
                "last_deep_reflection": self.last_deep_reflection.isoformat()
            }
            with open(os.path.join(self.evolution_data_dir, "reflection_schedule.json"), 'w') as f:
                json.dump(reflection_data, f, indent=2)
            
            logger.info("Saved evolution data")
        except Exception as e:
            logger.error(f"Error saving evolution data: {e}")
    
    def track_orchestration_pattern(
        self,
        decomposed_task: DecomposedTask,
        outcome: OutcomeType,
        performance_metrics: Dict[str, float],
        agent_performances: Dict[str, Dict[str, float]],
        notes: Optional[str] = None
    ) -> str:
        """
        Track an orchestration pattern after task completion.
        
        Args:
            decomposed_task: The decomposed task that was executed
            outcome: The outcome of the orchestration
            performance_metrics: Performance metrics for this orchestration
            agent_performances: Performance data for each agent
            notes: Optional notes about the orchestration
            
        Returns:
            The pattern_id of the tracked pattern
        """
        # Create a pattern description based on the task
        pattern_description = f"Task decomposition for {decomposed_task.original_title} using {decomposed_task.strategy.name} strategy"
        
        # Extract context information
        context = {
            "original_task_id": decomposed_task.original_task_id,
            "original_title": decomposed_task.original_title,
            "strategy": decomposed_task.strategy.name,
            "subtask_count": len(decomposed_task.subtasks),
            "task_types": [
                task.metadata.get("task_type", "unknown") 
                for task in decomposed_task.subtasks.values()
            ],
            "agent_count": len(set(
                [agent_id for task in decomposed_task.subtasks.values() 
                 for agent_id in task.results.keys()]
            ))
        }
        
        # Generate a consistent pattern ID based on decomposition strategy and task characteristics
        strategy_hash = hash(decomposed_task.strategy.name)
        task_hash = hash(f"{len(decomposed_task.subtasks)}:{context['agent_count']}:{','.join(context['task_types'])}")
        pattern_id = f"orch_pattern_{strategy_hash % 10000}_{task_hash % 10000}"
        
        # Check if this pattern already exists
        if pattern_id in self.orchestration_patterns:
            pattern = self.orchestration_patterns[pattern_id]
            
            # Update pattern statistics
            pattern.occurrences += 1
            pattern.last_observed = datetime.now(timezone.utc).isoformat()
            
            # Update outcome counts
            if outcome == OutcomeType.SUCCESSFUL:
                pattern.successful_count += 1
            elif outcome == OutcomeType.UNSUCCESSFUL:
                pattern.unsuccessful_count += 1
            elif outcome == OutcomeType.PARTIALLY_SUCCESSFUL:
                pattern.successful_count += 0.5
                pattern.neutral_count += 0.5
            elif outcome == OutcomeType.PARTIALLY_UNSUCCESSFUL:
                pattern.unsuccessful_count += 0.5
                pattern.neutral_count += 0.5
            else:  # NEUTRAL or INDETERMINATE
                pattern.neutral_count += 1
            
            # Recalculate success rate
            total_outcomes = pattern.successful_count + pattern.unsuccessful_count + pattern.neutral_count
            if total_outcomes > 0:
                pattern.success_rate = (pattern.successful_count + (pattern.neutral_count * 0.5)) / total_outcomes
            
            # Update confidence based on number of occurrences
            pattern.confidence = min(0.95, 1.0 - (1.0 / (1.0 + pattern.occurrences * 0.1)))
            
            # Update performance metrics (weighted average with new data)
            for metric, value in performance_metrics.items():
                weight = 1.0 / pattern.occurrences
                if metric in pattern.performance_metrics:
                    pattern.performance_metrics[metric] = (
                        pattern.performance_metrics[metric] * (1 - weight) + value * weight
                    )
                else:
                    pattern.performance_metrics[metric] = value
            
            # Update agent performance data
            for agent_id, agent_metrics in agent_performances.items():
                if agent_id not in pattern.agent_performance:
                    pattern.agent_performance[agent_id] = agent_metrics
                else:
                    for metric, value in agent_metrics.items():
                        weight = 1.0 / pattern.occurrences
                        if metric in pattern.agent_performance[agent_id]:
                            pattern.agent_performance[agent_id][metric] = (
                                pattern.agent_performance[agent_id][metric] * (1 - weight) + 
                                value * weight
                            )
                        else:
                            pattern.agent_performance[agent_id][metric] = value
            
            # Update task type performance
            task_types = [task.metadata.get("task_type", "unknown") for task in decomposed_task.subtasks.values()]
            task_type_counts = Counter(task_types)
            for task_type, count in task_type_counts.items():
                if task_type not in pattern.task_type_performance:
                    pattern.task_type_performance[task_type] = {
                        "count": count,
                        "success_rate": pattern.success_rate
                    }
                else:
                    old_count = pattern.task_type_performance[task_type]["count"]
                    total_count = old_count + count
                    pattern.task_type_performance[task_type]["count"] = total_count
                    pattern.task_type_performance[task_type]["success_rate"] = (
                        (pattern.task_type_performance[task_type]["success_rate"] * old_count +
                         pattern.success_rate * count) / total_count
                    )
            
        else:
            # Create a new orchestration pattern
            pattern = OrchestrationPattern(
                pattern_id=pattern_id,
                description=pattern_description,
                context=context,
                decomposition_strategy=decomposed_task.strategy,
                agent_selection_criteria={},  # Will be populated through reflection
                resource_allocation_strategy={},  # Will be populated through reflection
                communication_template={},  # Will be populated through reflection
                performance_metrics=performance_metrics,
                occurrences=1,
                successful_count=1 if outcome == OutcomeType.SUCCESSFUL else 0,
                unsuccessful_count=1 if outcome == OutcomeType.UNSUCCESSFUL else 0,
                neutral_count=1 if outcome in (OutcomeType.NEUTRAL, OutcomeType.INDETERMINATE) else 0,
                success_rate=1.0 if outcome == OutcomeType.SUCCESSFUL else (
                    0.0 if outcome == OutcomeType.UNSUCCESSFUL else 0.5
                ),
                confidence=0.1  # Low confidence for new patterns
            )
            
            # Special handling for partially successful/unsuccessful
            if outcome == OutcomeType.PARTIALLY_SUCCESSFUL:
                pattern.successful_count = 0.5
                pattern.neutral_count = 0.5
                pattern.success_rate = 0.75  # Between success (1.0) and neutral (0.5)
            elif outcome == OutcomeType.PARTIALLY_UNSUCCESSFUL:
                pattern.unsuccessful_count = 0.5
                pattern.neutral_count = 0.5
                pattern.success_rate = 0.25  # Between neutral (0.5) and failure (0.0)
            
            # Add agent performance data
            pattern.agent_performance = agent_performances
            
            # Add task type performance data
            task_types = [task.metadata.get("task_type", "unknown") for task in decomposed_task.subtasks.values()]
            task_type_counts = Counter(task_types)
            for task_type, count in task_type_counts.items():
                pattern.task_type_performance[task_type] = {
                    "count": count,
                    "success_rate": pattern.success_rate
                }
            
            self.orchestration_patterns = {**self.orchestration_patterns, pattern_id: pattern}
        
        # Add corresponding entry to the learning system
        dimensions = [
            LearningDimension.TASK_COLLABORATION,
            LearningDimension.ADAPTABILITY
        ]
        
        self.learning_system.track_interaction(
            pattern_description=pattern_description,
            context=context,
            dimensions=dimensions,
            outcome=outcome,
            confidence=pattern.confidence,
            notes=notes
        )
        
        # Add entry to the growth journal
        self._add_orchestration_to_journal(
            pattern=pattern,
            decomposed_task=decomposed_task,
            outcome=outcome,
            metrics=performance_metrics,
            notes=notes
        )
        
        # Save updated data
        self._save_evolution_data()
        
        # Check if reflection is due
        now = datetime.now(timezone.utc)
        if now - self.last_reflection >= self.reflection_frequency:
            self.reflect_on_orchestration()
        
        if now - self.last_deep_reflection >= self.deep_reflection_frequency:
            self.deep_reflection()
        
        return pattern_id
    
    def _add_orchestration_to_journal(
        self,
        pattern: OrchestrationPattern,
        decomposed_task: DecomposedTask,
        outcome: OutcomeType,
        metrics: Dict[str, float],
        notes: Optional[str] = None
    ) -> None:
        """
        Add an orchestration entry to the growth journal.
        
        Args:
            pattern: The orchestration pattern
            decomposed_task: The decomposed task
            outcome: The outcome of the orchestration
            metrics: Performance metrics
            notes: Optional notes
        """
        # Create journal entry content
        content = f"Orchestration Pattern: {pattern.description}\n"
        content += f"Outcome: {outcome.name}\n"
        content += f"Strategy: {decomposed_task.strategy.name}\n"
        content += f"Subtasks: {len(decomposed_task.subtasks)}\n"
        content += f"Success rate: {pattern.success_rate:.2f} (confidence: {pattern.confidence:.2f})\n"
        content += f"Occurrences: {pattern.occurrences}\n\n"
        
        content += "Performance Metrics:\n"
        for metric, value in metrics.items():
            content += f"- {metric}: {value}\n"
        
        if notes:
            content += f"\nNotes: {notes}\n"
        
        # Create and add the entry
        entry = GrowthJournalEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="orchestration",
            dimension="ORCHESTRATION",
            content=content,
            metrics=metrics,
            references=[pattern.pattern_id, decomposed_task.original_task_id]
        )
        
        self.learning_system.growth_journal.append(entry)
        
        # Save to file if directory is configured
        if self.learning_system._save_growth_journal_entry:
            self.learning_system._save_growth_journal_entry(entry)
    
    def reflect_on_orchestration(self) -> List[Dict[str, Any]]:
        """
        Reflect on orchestration patterns to identify improvements.
        
        Returns:
            List of insights and recommended improvements
        """
        logger.info("Performing orchestration reflection")
        
        # Update reflection timestamp
        self.last_reflection = datetime.now(timezone.utc)
        
        insights = []
        
        # Group patterns by success rate
        successful_patterns = []
        unsuccessful_patterns = []
        
        for pattern in self.orchestration_patterns.values():
            # Only consider patterns with sufficient confidence
            if pattern.confidence < 0.3:
                continue
                
            if pattern.success_rate >= 0.7:  # 70% success threshold
                successful_patterns.append(pattern)
            elif pattern.success_rate <= 0.4:  # 40% failure threshold
                unsuccessful_patterns.append(pattern)
        
        # Analyze successful patterns for common traits
        if successful_patterns:
            successful_strategies = Counter([
                p.decomposition_strategy.name if p.decomposition_strategy else "unknown"
                for p in successful_patterns
            ])
            most_successful_strategy = successful_strategies.most_common(1)[0][0]
            
            insights.append({
                "type": "successful_strategy",
                "description": f"The {most_successful_strategy} decomposition strategy has shown high success rates",
                "recommendation": f"Consider using {most_successful_strategy} more frequently for task decomposition",
                "confidence": min(0.9, 0.5 + (len(successful_patterns) * 0.05))
            })
            
            # Analyze agent selection patterns
            agent_success_rates = defaultdict(list)
            for pattern in successful_patterns:
                for agent_id, metrics in pattern.agent_performance.items():
                    if "success_rate" in metrics:
                        agent_success_rates[agent_id].append(metrics["success_rate"])
            
            # Identify consistently high-performing agents
            high_performing_agents = []
            for agent_id, rates in agent_success_rates.items():
                if len(rates) >= 3 and statistics.mean(rates) >= 0.8:
                    high_performing_agents.append(agent_id)
            
            if high_performing_agents:
                insights.append({
                    "type": "agent_selection",
                    "description": f"Identified {len(high_performing_agents)} consistently high-performing agents",
                    "recommendation": "Prefer these agents for future task assignments",
                    "data": {
                        "high_performing_agents": high_performing_agents
                    },
                    "confidence": min(0.9, 0.5 + (len(high_performing_agents) * 0.05))
                })
        
        # Analyze unsuccessful patterns
        if unsuccessful_patterns:
            unsuccessful_strategies = Counter([
                p.decomposition_strategy.name if p.decomposition_strategy else "unknown"
                for p in unsuccessful_patterns
            ])
            most_unsuccessful_strategy = unsuccessful_strategies.most_common(1)[0][0]
            
            insights.append({
                "type": "unsuccessful_strategy",
                "description": f"The {most_unsuccessful_strategy} decomposition strategy has shown low success rates",
                "recommendation": f"Consider using alternative strategies to {most_unsuccessful_strategy}",
                "confidence": min(0.9, 0.5 + (len(unsuccessful_patterns) * 0.05))
            })
            
            # Analyze common bottlenecks
            bottlenecks = []
            for pattern in unsuccessful_patterns:
                # Look for performance metrics that indicate bottlenecks
                if "response_time" in pattern.performance_metrics and pattern.performance_metrics["response_time"] > 5.0:
                    bottlenecks.append("high_response_time")
                if "completion_rate" in pattern.performance_metrics and pattern.performance_metrics["completion_rate"] < 0.7:
                    bottlenecks.append("low_completion_rate")
                if "error_rate" in pattern.performance_metrics and pattern.performance_metrics["error_rate"] > 0.2:
                    bottlenecks.append("high_error_rate")
                
                # Also check agent performance
                for agent_id, metrics in pattern.agent_performance.items():
                    if "error_rate" in metrics and metrics["error_rate"] > 0.3:
                        bottlenecks.append(f"agent_errors_{agent_id}")
            
            # Count bottleneck occurrences
            bottleneck_counts = Counter(bottlenecks)
            if bottleneck_counts:
                most_common_bottleneck = bottleneck_counts.most_common(1)[0][0]
                insights.append({
                    "type": "bottleneck",
                    "description": f"Identified common bottleneck: {most_common_bottleneck}",
                    "recommendation": "Address this bottleneck in future orchestrations",
                    "confidence": min(0.85, 0.4 + (bottleneck_counts[most_common_bottleneck] * 0.05))
                })
