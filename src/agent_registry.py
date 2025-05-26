"""
Agent Registry System

This module provides a centralized registry for tracking, discovering, and selecting agents
based on their capabilities, performance metrics, and availability. It enables fair and
efficient agent selection for tasks orchestrated by the OrchestratorEngine.

The AgentRegistry implements the "Fairness as a Fundamental Truth" principle ensuring that
agent selection is objective, transparent, and based on merit while maintaining fair
distribution of opportunities.
"""

import json
import logging
import uuid
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
import threading
import heapq
import statistics
import random
from dataclasses import dataclass, field

# Import related modules
from agent_card import AgentCard
from orchestrator_engine import TaskType, AgentRole, AgentProfile
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentRegistry")


class CapabilityLevel(Enum):
    """Level of proficiency for a capability."""
    NOVICE = 1       # Basic understanding, needs guidance
    COMPETENT = 2    # Can perform adequately, sometimes needs assistance
    PROFICIENT = 3   # Can perform well independently
    EXPERT = 4       # Deep expertise, can handle complex cases
    MASTER = 5       # Exceptional skill, can teach others


class MatchCriteria(Enum):
    """Criteria for matching agents to tasks."""
    CAPABILITY = auto()      # Match based on specific capabilities
    PERFORMANCE = auto()     # Match based on past performance
    AVAILABILITY = auto()    # Match based on current availability
    LATENCY = auto()         # Match based on response time
    COST = auto()            # Match based on resource cost
    TRUST = auto()           # Match based on trust relationship
    FAIRNESS = auto()        # Match based on fair opportunity distribution
    COMPOSITE = auto()       # Match based on weighted combination


@dataclass
class CapabilityInfo:
    """Information about an agent's capability."""
    name: str
    description: str
    level: CapabilityLevel
    task_types: List[TaskType]
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    last_verified: Optional[datetime] = None
    verification_method: Optional[str] = None


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    success_rate: float = 1.0                          # 0.0 to 1.0
    average_response_time: float = 0.0                 # In seconds
    average_completion_time: Dict[TaskType, float] = field(default_factory=dict)  # By task type
    quality_score: float = 1.0                         # 0.0 to 1.0
    task_count: int = 0                                # Number of tasks performed
    tasks_by_type: Dict[TaskType, int] = field(default_factory=dict)  # Count by task type
    recent_tasks: List[str] = field(default_factory=list)  # Recent task IDs
    last_success: Optional[datetime] = None             # Time of last successful task
    last_failure: Optional[datetime] = None             # Time of last failed task
    error_count: int = 0                                # Number of errors
    common_errors: Dict[str, int] = field(default_factory=dict)  # Error type to count


@dataclass
class CapabilityRequest:
    """Request for capability negotiation."""
    request_id: str
    requesting_agent_id: str
    capability_name: str
    task_types: List[TaskType]
    minimum_level: CapabilityLevel = CapabilityLevel.COMPETENT
    parameters: Dict[str, Any] = field(default_factory=dict)
    response_by: Optional[datetime] = None
    priority: int = 1  # 1 (lowest) to 5 (highest)


@dataclass
class CapabilityResponse:
    """Response to a capability request."""
    request_id: str
    responding_agent_id: str
    capability_name: str
    available: bool
    level: Optional[CapabilityLevel] = None
    supported_parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_response_time: Optional[float] = None  # In seconds
    estimated_completion_time: Optional[float] = None  # In seconds
    cost: Optional[float] = None  # Arbitrary cost unit


class DiscoveryMethod(Enum):
    """Methods for capability discovery."""
    SELF_DECLARATION = "self_declaration"  # Agent declares own capabilities
    AGENT_CARD = "agent_card"              # Extract from A2A agent card
    TEST_EXECUTION = "test_execution"      # Execute test tasks to validate
    OBSERVATION = "observation"            # Observe behavior during regular tasks
    PEER_VALIDATION = "peer_validation"    # Other agents validate capabilities


class FairnessPolicy(Enum):
    """Policies for implementing fairness in agent selection."""
    MERIT_BASED = "merit_based"    # Select purely based on capability and performance
    ROUND_ROBIN = "round_robin"    # Rotate between qualified agents
    WEIGHTED_RANDOM = "weighted_random"  # Random selection weighted by qualification
    OPPORTUNITY_BALANCED = "opportunity_balanced"  # Balance opportunity among agents
    HYBRID = "hybrid"  # Combine multiple policies


class AgentRegistry:
    """
    A central registry for agent capabilities, performance metrics, and selection.
    
    The AgentRegistry maintains a database of known agents, discovers their capabilities,
    tracks their performance, and provides optimal agent selection for tasks based on
    various matching criteria and fairness policies.
    """
    
    def __init__(
        self,
        fairness_policy: FairnessPolicy = FairnessPolicy.HYBRID,
        principle_engine: Optional[PrincipleEngine] = None,
        discovery_methods: Optional[List[DiscoveryMethod]] = None,
        storage_path: str = "data/agent_registry"
    ):
        """
        Initialize the agent registry.
        
        Args:
            fairness_policy: Policy for implementing fairness in agent selection
            principle_engine: Engine for principle-based reasoning
            discovery_methods: Methods to use for capability discovery
            storage_path: Directory for storing registry data
        """
        self.fairness_policy = fairness_policy
        self.principle_engine = principle_engine
        self.discovery_methods = discovery_methods or [
            DiscoveryMethod.AGENT_CARD,
            DiscoveryMethod.SELF_DECLARATION,
            DiscoveryMethod.OBSERVATION
        ]
        self.storage_path = storage_path
        
        # Agent registry data
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Set[str]]] = {}  # capability -> task_type -> agent_ids
        self.capabilities_by_agent: Dict[str, Dict[str, CapabilityInfo]] = {}  # agent_id -> capability -> info
        self.roles_by_agent: Dict[str, List[AgentRole]] = {}  # agent_id -> roles
        
        # Agent performance data
        self.performance: Dict[str, AgentPerformance] = {}  # agent_id -> performance
        
        # Task distribution tracking for fairness
        self.task_distribution: Dict[str, Dict[TaskType, int]] = {}  # agent_id -> task_type -> count
        self.opportunity_scores: Dict[str, float] = {}  # agent_id -> opportunity score
        
        # Locks
        self.registry_lock = threading.Lock()
        self.performance_lock = threading.Lock()
        
        logger.info("AgentRegistry initialized with fairness policy: %s", fairness_policy.value)
    
    def register_agent(
        self,
        agent_id: str,
        agent_card: Optional[AgentCard] = None,
        agent_profile: Optional[AgentProfile] = None,
        roles: Optional[List[AgentRole]] = None,
        declared_capabilities: Optional[Dict[str, CapabilityInfo]] = None,
        verify_capabilities: bool = True
    ) -> bool:
        """
        Register a new agent with the registry.
        
        Args:
            agent_id: ID of the agent to register
            agent_card: Optional agent card for A2A protocol
            agent_profile: Optional agent profile data
            roles: Optional list of roles the agent can fulfill
            declared_capabilities: Optional dictionary of capabilities declared by the agent
            verify_capabilities: Whether to verify declared capabilities
            
        Returns:
            Whether registration was successful
        """
        with self.registry_lock:
            # Check if agent already exists
            if agent_id in self.agents:
                logger.warning("Agent %s already registered", agent_id)
                return False
            
            # Create agent entry
            self.agents = {**self.agents, agent_id: {
                "agent_id": agent_id,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "agent_card": agent_card.to_dict() if agent_card else None,
                "agent_profile": agent_profile.__dict__ if agent_profile else None,
                "roles": [role.value for role in (roles or [])]
            }}
            
            # Initialize capability and performance records
            self.capabilities_by_agent = {**self.capabilities_by_agent, agent_id: {}}
            self.roles_by_agent = {**self.roles_by_agent, agent_id: roles or []}
            self.performance = {**self.performance, agent_id: AgentPerformance()}
            self.task_distribution = {**self.task_distribution, agent_id: {task_type: 0 for task_type in TaskType}}
            self.opportunity_scores = {**self.opportunity_scores, agent_id: 1.0}
            
            # Process agent card if provided
            if agent_card:
                self._extract_capabilities_from_agent_card(agent_id, agent_card)
            
            # Process declared capabilities if provided
            if declared_capabilities:
                self._register_declared_capabilities(
                    agent_id, declared_capabilities, verify_capabilities
                )
            
            logger.info("Registered agent %s with %d capabilities", 
                       agent_id, len(self.capabilities_by_agent[agent_id]))
            return True
    
    def _select_by_capability_level(
        self,
        candidates: List[str],
        task_type: TaskType,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select agent with highest capability levels for required capabilities.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            required_capabilities: Optional list of required capabilities
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not candidates:
            return None
            
        if not required_capabilities:
            # Without specific capabilities, select randomly
            return random.choice(candidates)
            
        # Calculate average capability level for each agent
        agent_levels = {}
        for agent_id in candidates:
            if agent_id not in self.capabilities_by_agent:
                continue
                
            capabilities = self.capabilities_by_agent[agent_id]
            total_level = 0
            count = 0
            
            for capability_name in required_capabilities:
                if capability_name in capabilities:
                    total_level += capabilities[capability_name].level.value
                    count += 1
            
            if count > 0:
                agent_levels[agent_id] = total_level / count
        
        if not agent_levels:
            return None
            
        # Return agent with highest average capability level
        return max(agent_levels.items(), key=lambda x: x[1])[0]
    
    def _select_by_performance(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Select agent with best performance for the task type.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not candidates:
            return None
            
        # Calculate performance score for each agent
        agent_scores = {}
        for agent_id in candidates:
            if agent_id not in self.performance:
                continue
                
            perf = self.performance[agent_id]
            
            # Weight success rate higher than response/completion time
            success_rate = perf.success_rate
            response_time = perf.average_response_time
            completion_time = perf.average_completion_time.get(task_type, 3600)
            
            # Normalize response and completion times
            # Lower is better, so we invert
            response_factor = 1.0 / (1.0 + response_time / 60.0)  # Minutes scale
            completion_factor = 1.0 / (1.0 + completion_time / 3600.0)  # Hours scale
            
            # Calculate overall score (higher is better)
            score = 0.6 * success_rate + 0.2 * response_factor + 0.2 * completion_factor
            
            agent_scores[agent_id] = score
        
        if not agent_scores:
            return None
            
        # Return agent with highest performance score
        return max(agent_scores.items(), key=lambda x: x[1])[0]
    
    def _select_by_availability(
        self,
        candidates: List[str]
    ) -> Optional[str]:
        """
        Select most available agent.
        
        Args:
            candidates: List of candidate agent IDs
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        # For now, just randomly select from candidates
        # In a real implementation, this would check agent availability status
        if not candidates:
            return None
            
        return random.choice(candidates)
    
    def _select_by_latency(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Select agent with lowest response latency.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not candidates:
            return None
            
        # Find agent with lowest average response time
        response_times = {}
        for agent_id in candidates:
            if agent_id in self.performance:
                response_times[agent_id] = self.performance[agent_id].average_response_time
        
        if not response_times:
            return None
            
        # Return agent with lowest response time
        return min(response_times.items(), key=lambda x: x[1])[0]
    
    def _select_by_cost(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Select agent with lowest cost.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        # For now, just randomly select from candidates
        # In a real implementation, this would check agent cost models
        if not candidates:
            return None
            
        return random.choice(candidates)
    
    def _select_by_trust(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Select agent with highest trust level.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not candidates:
            return None
            
        # For now, use success rate as a proxy for trust
        trust_scores = {}
        for agent_id in candidates:
            if agent_id in self.performance:
                trust_scores[agent_id] = self.performance[agent_id].success_rate
        
        if not trust_scores:
            return None
            
        # Return agent with highest trust score
        return max(trust_scores.items(), key=lambda x: x[1])[0]
    
    def _select_by_fairness(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Select agent based on fairness policy.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not candidates:
            return None
            
        # Get opportunity scores for candidates
        candidate_scores = {
            agent_id: self.opportunity_scores.get(agent_id, 1.0)
            for agent_id in candidates
        }
        
        # Apply fairness policy
        if self.fairness_policy == FairnessPolicy.MERIT_BASED:
            # Just use performance
            return self._select_by_performance(candidates, task_type)
            
        elif self.fairness_policy == FairnessPolicy.ROUND_ROBIN:
            # Select agent with fewest tasks of this type
            task_counts = {}
            for agent_id in candidates:
                if agent_id in self.task_distribution:
                    task_counts[agent_id] = self.task_distribution[agent_id].get(task_type, 0)
                else:
                    task_counts[agent_id] = 0
                    
            if not task_counts:
                return None
                
            # Return agent with fewest tasks
            return min(task_counts.items(), key=lambda x: x[1])[0]
            
        elif self.fairness_policy == FairnessPolicy.WEIGHTED_RANDOM:
            # Select randomly weighted by opportunity score
            weights = list(candidate_scores.values())
            if sum(weights) == 0:
                # If all weights are 0, use uniform distribution
                return random.choice(candidates)
            else:
                return random.choices(
                    list(candidate_scores.keys()),
                    weights=weights,
                    k=1
                )[0]
                
        elif self.fairness_policy == FairnessPolicy.OPPORTUNITY_BALANCED:
            # Simply select agent with highest opportunity score
            return max(candidate_scores.items(), key=lambda x: x[1])[0]
            
        elif self.fairness_policy == FairnessPolicy.HYBRID:
            # Combine merit and fairness
            agent_scores = self._calculate_composite_scores(candidates, task_type)
            
            if not agent_scores:
                return None
                
            # Apply opportunity weighting
            for i, (agent_id, score) in enumerate(agent_scores):
                opportunity = candidate_scores.get(agent_id, 1.0)
                # Adjust score by opportunity (higher opportunity = higher score)
                agent_scores[i] = (agent_id, score * (0.5 + 0.5 * opportunity))
                
            # Return agent with highest adjusted score
            return max(agent_scores, key=lambda x: x[1])[0]
            
        # Fallback to random selection
        return random.choice(candidates)
    
    def _calculate_composite_scores(
        self,
        candidates: List[str],
        task_type: TaskType
    ) -> List[Tuple[str, float]]:
        """
        Calculate composite scores for all candidates.
        
        Args:
            candidates: List of candidate agent IDs
            task_type: Type of task
            
        Returns:
            List of (agent_id, score) tuples
        """
        scores = []
        
        for agent_id in candidates:
            # Default score components
            capability_score = 0.5  # Default middle score
            performance_score = 0.5
            latency_score = 0.5
            
            # Capability score
            if agent_id in self.capabilities_by_agent:
                capabilities = self.capabilities_by_agent[agent_id]
                relevant_capabilities = [
                    cap for cap in capabilities.values()
                    if task_type in cap.task_types
                ]
                
                if relevant_capabilities:
                    # Average level of relevant capabilities
                    avg_level = sum(
                        cap.level.value for cap in relevant_capabilities
                    ) / len(relevant_capabilities)
                    
                    # Normalize to 0.0-1.0 range
                    capability_score = avg_level / CapabilityLevel.MASTER.value
            
            # Performance score
            if agent_id in self.performance:
                perf = self.performance[agent_id]
                success_rate = perf.success_rate
                quality = perf.quality_score
                
                # Combine success rate and quality
                performance_score = 0.6 * success_rate + 0.4 * quality
            
            # Latency score
            if agent_id in self.performance:
                perf = self.performance[agent_id]
                response_time = perf.average_response_time
                
                # Normalize to 0.0-1.0 range (lower is better)
                # 5 seconds is considered ideal (1.0), 5 minutes is poor (0.0)
                latency_score = max(0.0, min(1.0, 1.0 - (response_time - 5) / 295))
            
            # Calculate composite score
            # Weights should sum to 1.0
            composite_score = (
                0.4 * capability_score +
                0.4 * performance_score +
                0.2 * latency_score
            )
            
            scores.append((agent_id, composite_score))
        
        return scores
    
    def _apply_fairness_policy(
        self,
        agent_scores: List[Tuple[str, float]],
        task_type: TaskType
    ) -> Optional[str]:
        """
        Apply fairness policy to selection process.
        
        Args:
            agent_scores: List of (agent_id, score) tuples
            task_type: Type of task
            
        Returns:
            ID of selected agent, or None if no suitable agent found
        """
        if not agent_scores:
            return None
            
        # Get opportunity scores
        opportunity_scores = {
            agent_id: self.opportunity_scores.get(agent_id, 1.0)
            for agent_id, _ in agent_scores
        }
        
        # Apply policy based on fairness type
        if self.fairness_policy == FairnessPolicy.MERIT_BASED:
            # Simply return highest scored agent
            return max(agent_scores, key=lambda x: x[1])[0]
            
        elif self.fairness_policy == FairnessPolicy.ROUND_ROBIN:
            # Select based on task count
            task_counts = {}
            for agent_id, _ in agent_scores:
                if agent_id in self.task_distribution:
                    task_counts[agent_id] = self.task_distribution[agent_id].get(task_type, 0)
                else:
                    task_counts[agent_id] = 0
                    
            # Return agent with lowest task count
            return min(task_counts.items(), key=lambda x: x[1])[0]
            
        elif self.fairness_policy == FairnessPolicy.WEIGHTED_RANDOM:
            # Get quality threshold (80% of max score)
            if not agent_scores:
                return None
                
            max_score = max(score for _, score in agent_scores)
            threshold = max_score * 0.8
            
            # Filter to qualified agents
            qualified = [(agent_id, score) for agent_id, score in agent_scores if score >= threshold]
            
            if not qualified:
                qualified = agent_scores  # Fallback to all agents
                
            # Select randomly weighted by opportunity score
            weights = [opportunity_scores[agent_id] for agent_id, _ in qualified]
            
            if sum(weights) == 0:
                # If all weights are 0, use uniform distribution
                return random.choice([agent_id for agent_id, _ in qualified])
            else:
                return random.choices(
                    [agent_id for agent_id, _ in qualified],
                    weights=weights,
                    k=1
                )[0]
                
        elif self.fairness_policy == FairnessPolicy.OPPORTUNITY_BALANCED:
            # Get quality threshold (70% of max score)
            if not agent_scores:
                return None
                
            max_score = max(score for _, score in agent_scores)
            threshold = max_score * 0.7
            
            # Filter to qualified agents
            qualified = [(agent_id, score) for agent_id, score in agent_scores if score >= threshold]
            
            if not qualified:
                qualified = agent_scores  # Fallback to all agents
                
            # Select agent with highest opportunity score
            qualified_agents = [agent_id for agent_id, _ in qualified]
            return max(
                [(agent_id, opportunity_scores[agent_id]) for agent_id in qualified_agents],
                key=lambda x: x[1]
            )[0]
            
        elif self.fairness_policy == FairnessPolicy.HYBRID:
            # Combine merit and fairness
            # Adjust scores based on opportunity
            adjusted_scores = []
            for agent_id, score in agent_scores:
                opportunity = opportunity_scores.get(agent_id, 1.0)
                
                # Adjust score by opportunity (higher opportunity = higher score)
                # Use weighted combination: 70% merit, 30% fairness
                adjusted_score = 0.7 * score + 0.3 * opportunity
                
                adjusted_scores.append((agent_id, adjusted_score))
                
            # Return agent with highest adjusted score
            return max(adjusted_scores, key=lambda x: x[1])[0]
            
        # Fallback to highest scoring agent
        return max(agent_scores, key=lambda x: x[1])[0]
    
    def _extract_capabilities_from_agent_card(self, agent_id: str, agent_card: AgentCard) -> None:
        """
        Extract capabilities from an agent card.
        
        Args:
            agent_id: ID of the agent
            agent_card: Agent card to extract capabilities from
        """
        if not agent_card:
            return
            
        # Get core information from card
        name = agent_card.name
        description = agent_card.description
        
        # Extract capabilities from functions
        if hasattr(agent_card, 'functions') and agent_card.functions:
            for func in agent_card.functions:
                capability_name = func.get('name', '')
                if not capability_name:
                    continue
                    
                capability_desc = func.get('description', '')
                parameters = func.get('parameters', {})
                
                # Determine task types based on function name and description
                task_types = self._infer_task_types(capability_name, capability_desc)
                
                # Create capability info
                capability_info = CapabilityInfo(
                    name=capability_name,
                    description=capability_desc,
                    level=CapabilityLevel.COMPETENT,  # Default level until verified
                    task_types=task_types,
                    parameters=parameters,
                    examples=[],
                    last_verified=datetime.now(timezone.utc),
                    verification_method=DiscoveryMethod.AGENT_CARD.value
                )
                
                # Register capability
                self._add_capability(agent_id, capability_name, capability_info)
        
        # Extract capabilities from tools
        if hasattr(agent_card, 'tools') and agent_card.tools:
            for tool in agent_card.tools:
                capability_name = tool.get('name', '')
                if not capability_name:
                    continue
                    
                capability_desc = tool.get('description', '')
                parameters = tool.get('parameters', {})
                
                # Determine task types based on tool name and description
                task_types = self._infer_task_types(capability_name, capability_desc)
                
                # Create capability info
                capability_info = CapabilityInfo(
                    name=capability_name,
                    description=capability_desc,
                    level=CapabilityLevel.COMPETENT,  # Default level until verified
                    task_types=task_types,
                    parameters=parameters,
                    examples=[],
                    last_verified=datetime.now(timezone.utc),
                    verification_method=DiscoveryMethod.AGENT_CARD.value
                )
                
                # Register capability
                self._add_capability(agent_id, capability_name, capability_info)
    
    def _register_declared_capabilities(
        self,
        agent_id: str,
        capabilities: Dict[str, CapabilityInfo],
        verify: bool = True
    ) -> None:
        """
        Register capabilities declared by an agent.
        
        Args:
            agent_id: ID of the agent
            capabilities: Dictionary of capability name to capability info
            verify: Whether to verify the capabilities
        """
        for capability_name, capability_info in capabilities.items():
            # Verify the capability if requested
            if verify:
                verified = self._verify_capability(agent_id, capability_name, capability_info)
                if not verified:
                    logger.warning(
                        "Capability %s for agent %s failed verification",
                        capability_name, agent_id
                    )
                    continue
            
            # Register the capability
            self._add_capability(agent_id, capability_name, capability_info)
    
    def _add_capability(self, agent_id: str, capability_name: str, capability_info: CapabilityInfo) -> None:
        """
        Add a capability to the registry.
        
        Args:
            agent_id: ID of the agent
            capability_name: Name of the capability
            capability_info: Information about the capability
        """
        # Add to capabilities by agent
        self.capabilities_by_agent[agent_id][capability_name] = capability_info
        
        # Add to capability index
        for task_type in capability_info.task_types:
            task_type_str = task_type.name
            
            if capability_name not in self.capabilities:
                self.capabilities = {**self.capabilities, capability_name: {}}
                
            if task_type_str not in self.capabilities[capability_name]:
                self.capabilities[capability_name][task_type_str] = set()
                
            self.capabilities[capability_name][task_type_str].add(agent_id)
    
    def _infer_task_types(self, capability_name: str, description: str) -> List[TaskType]:
        """
        Infer task types from capability name and description.
        
        Args:
            capability_name: Name of the capability
            description: Description of the capability
            
        Returns:
            List of inferred task types
        """
        task_types = []
        name_lower = capability_name.lower()
        desc_lower = description.lower()
        
        # Match by keywords in name and description
        if any(kw in name_lower or kw in desc_lower for kw in ['analyze', 'analysis', 'assess']):
            task_types.append(TaskType.ANALYSIS)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['generate', 'create', 'produce']):
            task_types.append(TaskType.GENERATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['transform', 'convert', 'translate']):
            task_types.append(TaskType.TRANSFORMATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['extract', 'retrieve', 'obtain']):
            task_types.append(TaskType.EXTRACTION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['validate', 'verify', 'check']):
            task_types.append(TaskType.VALIDATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['aggregate', 'combine', 'merge']):
            task_types.append(TaskType.AGGREGATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['decide', 'recommend', 'choose']):
            task_types.append(TaskType.DECISION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['orchestrate', 'coordinate', 'manage']):
            task_types.append(TaskType.ORCHESTRATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['communicate', 'message', 'notify']):
            task_types.append(TaskType.COMMUNICATION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['research', 'investigate', 'explore']):
            task_types.append(TaskType.RESEARCH)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['execute', 'run', 'perform']):
            task_types.append(TaskType.EXECUTION)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['monitor', 'observe', 'track']):
            task_types.append(TaskType.MONITORING)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['recover', 'repair', 'restore']):
            task_types.append(TaskType.RECOVERY)
            
        if any(kw in name_lower or kw in desc_lower for kw in ['negotiate', 'mediate', 'resolve']):
            task_types.append(TaskType.NEGOTIATION)
            
        # Default to OTHER if no matches
        if not task_types:
            task_types.append(TaskType.OTHER)
            
        return task_types
    
    def _verify_capability(
        self,
        agent_id: str,
        capability_name: str,
        capability_info: CapabilityInfo
    ) -> bool:
        """
        Verify that an agent has a claimed capability.
        
        Args:
            agent_id: ID of the agent
            capability_name: Name of the capability
            capability_info: Information about the capability
            
        Returns:
            Whether the capability was verified
        """
        # For now, we just accept the declaration as true
        # Future implementations could test capabilities with sample tasks
        capability_info.verification_method = DiscoveryMethod.SELF_DECLARATION.value
        capability_info.last_verified = datetime.now(timezone.utc)
        return True
    
    def discover_capabilities(self, agent_id: str, methods: Optional[List[DiscoveryMethod]] = None) -> Dict[str, CapabilityInfo]:
        """
        Actively discover capabilities of an agent.
        
        Args:
            agent_id: ID of the agent
            methods: Optional list of discovery methods to use
            
        Returns:
            Dictionary of discovered capabilities
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not registered")
            return {}
            
        methods = methods or self.discovery_methods
        discovered = {}
        
        for method in methods:
            if method == DiscoveryMethod.AGENT_CARD:
                # Already done during registration if agent card was provided
                pass
                
            elif method == DiscoveryMethod.SELF_DECLARATION:
                # Implement asking agent to declare capabilities
                # This would involve sending a structured query to the agent
                pass
                
            elif method == DiscoveryMethod.TEST_EXECUTION:
                # Implement sending test tasks to verify capabilities
                pass
                
            elif method == DiscoveryMethod.OBSERVATION:
                # Capabilities are observed during regular task execution
                pass
                
            elif method == DiscoveryMethod.PEER_VALIDATION:
                # Implement asking other agents to validate capabilities
                pass
        
        # Add newly discovered capabilities to the registry
        with self.registry_lock:
            for capability_name, capability_info in discovered.items():
                self._add_capability(agent_id, capability_name, capability_info)
                
        return discovered
    
    def find_agents_with_capability(
        self,
        capability_name: str,
        task_type: Optional[TaskType] = None,
        min_level: CapabilityLevel = CapabilityLevel.COMPETENT
    ) -> List[str]:
        """
        Find agents with a specific capability.
        
        Args:
            capability_name: Name of the capability
            task_type: Optional task type to filter by
            min_level: Minimum capability level required
            
        Returns:
            List of agent IDs with the capability
        """
        if capability_name not in self.capabilities:
            return []
            
        if task_type:
            # Get agents with capability for specific task type
            task_type_str = task_type.name
            if task_type_str not in self.capabilities[capability_name]:
                return []
                
            agents = self.capabilities[capability_name][task_type_str]
        else:
            # Get all agents with capability across all task types
            agents = set()
            for task_agents in self.capabilities[capability_name].values():
                agents.update(task_agents)
        
        # Filter by capability level
        if min_level != CapabilityLevel.NOVICE:
            agents = [
                agent_id for agent_id in agents
                if (agent_id in self.capabilities_by_agent and
                    capability_name in self.capabilities_by_agent[agent_id] and
                    self.capabilities_by_agent[agent_id][capability_name].level.value >= min_level.value)
            ]
            
        return list(agents)
    
    def find_agents_by_task_type(
        self,
        task_type: TaskType,
        min_performance: float = 0.0,
        max_response_time: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find agents capable of handling a specific task type.
        
        Args:
            task_type: Type of task
            min_performance: Minimum performance score required
            max_response_time: Maximum acceptable response time
            
        Returns:
            List of (agent_id, suitability_score) tuples
        """
        suitable_agents = []
        
        for agent_id, capabilities in self.capabilities_by_agent.items():
            # Check if agent has capabilities for this task type
            has_capability = any(
                task_type in capability.task_types
                for capability in capabilities.values()
            )
            
            if not has_capability:
                continue
                
            # Check performance requirements
            if agent_id in self.performance:
                perf = self.performance[agent_id]
                
                if perf.success_rate < min_performance:
                    continue
                    
                if max_response_time is not None and perf.average_response_time > max_response_time:
                    continue
                
                # Calculate suitability score
                # Higher score means more suitable
                task_count = perf.tasks_by_type.get(task_type, 0)
                success_rate = perf.success_rate
                response_time = perf.average_response_time
                completion_time = perf.average_completion_time.get(task_type, 3600)
                
                # More tasks = more experience = better score
                experience_factor = min(1.0, task_count / 10)
                
                # Higher success rate = better score
                success_factor = success_rate
                
                # Lower response time = better score (normalized)
                response_factor = 1.0 / (1.0 + response_time / 60.0)
                
                # Lower completion time = better score (normalized)
                completion_factor = 1.0 / (1.0 + completion_time / 3600.0)
                
                # Calculate overall score
                score = (
                    0.3 * experience_factor +
                    0.4 * success_factor +
                    0.15 * response_factor +
                    0.15 * completion_factor
                )
                
                suitable_agents.append((agent_id, score))
        
        # Sort by suitability score (descending)
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        return suitable_agents
    
    def update_agent_performance(
        self,
        agent_id: str,
        task_type: TaskType,
        success: bool,
        response_time: float,
        completion_time: float,
        task_id: str,
        quality_score: Optional[float] = None,
        error_type: Optional[str] = None
    ) -> None:
        """
        Update performance metrics for an agent.
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task
            success: Whether the task was successful
            response_time: Time taken to respond (seconds)
            completion_time: Time taken to complete (seconds)
            task_id: ID of the task
            quality_score: Optional quality score (0.0-1.0)
            error_type: Optional error type if task failed
        """
        if agent_id not in self.performance:
            logger.warning(f"Agent {agent_id} not found in performance registry")
            return
            
        with self.performance_lock:
            perf = self.performance[agent_id]
            
            # Update task count
            perf.task_count += 1
            perf.tasks_by_type[task_type] = perf.tasks_by_type.get(task_type, 0) + 1
            
            # Update recent tasks
            perf.recent_tasks.append(task_id)
            if len(perf.recent_tasks) > 10:
                perf.recent_tasks.pop(0)
            
            # Update success rate
            success_weight = 1.0 / max(1, perf.task_count)  # Weight decreases with more tasks
            if success:
                # Successful task
                perf.success_rate = perf.success_rate * (1 - success_weight) + 1.0 * success_weight
                perf.last_success = datetime.now(timezone.utc)
            else:
                # Failed task
                perf.success_rate = perf.success_rate * (1 - success_weight) + 0.0 * success_weight
                perf.last_failure = datetime.now(timezone.utc)
                perf.error_count += 1
                
                # Track error type
                if error_type:
                    perf.common_errors[error_type] = perf.common_errors.get(error_type, 0) + 1
            
            # Update response time
            prev_avg = perf.average_response_time
            perf.average_response_time = prev_avg * (1 - success_weight) + response_time * success_weight
            
            # Update completion time for task type
            prev_completion = perf.average_completion_time.get(task_type, completion_time)
            perf.average_completion_time[task_type] = prev_completion * (1 - success_weight) + completion_time * success_weight
            
            # Update quality score if provided
            if quality_score is not None:
                prev_quality = perf.quality_score
                perf.quality_score = prev_quality * (1 - success_weight) + quality_score * success_weight
            
            # Update task distribution tracking
            self.task_distribution[agent_id][task_type] += 1
            
            # Recalculate opportunity score
            self._update_opportunity_score(agent_id)
    
    def _update_opportunity_score(self, agent_id: str) -> None:
        """
        Update the opportunity score for an agent based on task distribution.
        
        Args:
            agent_id: ID of the agent to update score for
        """
        if agent_id not in self.task_distribution:
            return
            
        # Calculate total tasks across all agents for each task type
        total_tasks_by_type = {}
        for agent, tasks in self.task_distribution.items():
            for task_type, count in tasks.items():
                if task_type not in total_tasks_by_type:
                    total_tasks_by_type[task_type] = 0
                total_tasks_by_type[task_type] += count
        
        # Calculate opportunity score based on task distribution
        # Higher score means agent has received fewer opportunities relative to others
        agent_tasks = self.task_distribution[agent_id]
        total_opportunity_score = 0.0
        
        for task_type, total in total_tasks_by_type.items():
            if total == 0:
                continue
                
            agent_count = agent_tasks.get(task_type, 0)
            
            # Calculate ratio of tasks this agent has done
            # 0.0 means none, 1.0 means all tasks of this type
            if total > 0:
                ratio = agent_count / total
            else:
                ratio = 0.0
                
            # Invert to get opportunity score (higher means fewer tasks)
            # Apply diminishing returns to avoid extreme values
            opportunity = 1.0 / (1.0 + ratio * 3.0)
            
            # Add to total score
            total_opportunity_score += opportunity
        
        # Normalize by number of task types
        num_task_types = len(total_tasks_by_type)
        if num_task_types > 0:
            self.opportunity_scores = {**self.opportunity_scores, agent_id: total_opportunity_score / num_task_types}
        else:
            self.opportunity_scores = {**self.opportunity_scores, agent_id: 1.0}
