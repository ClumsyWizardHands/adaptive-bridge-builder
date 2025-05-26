"""
Crisis Response Coordinator Extension

This module extends the OrchestratorEngine with specialized capabilities for
rapidly assessing and responding to urgent situations requiring multiple agent
expertise and coordination.

The CrisisResponseCoordinator implements sophisticated crisis assessment, prioritization,
information gathering, decision support, and communication management, particularly
in high-stress, rapidly changing environments.

It embodies the "Adaptability as a Form of Strength" principle throughout the response
process, allowing for rapid adjustment to new information and changing circumstances.
"""

import json
import uuid
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
import threading
import copy
from dataclasses import dataclass, field
import heapq

# Import related modules
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability, 
    DependencyType, TaskDecompositionStrategy, DecomposedTask, AgentProfile
)
from collaborative_task_handler import Task, TaskStatus, TaskPriority
from principle_engine import PrincipleEngine
from communication_adapter import CommunicationAdapter
from relationship_tracker import RelationshipTracker
from content_handler import ContentHandler
from emotional_intelligence import EmotionalIntelligence
from project_orchestrator import (
    Project, Resource, ResourceType, ResourceConflictStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CrisisResponseCoordinator")


class CrisisSeverity(Enum):
    """Severity levels for crisis situations."""
    LOW = "low"                # Minor incident with limited impact
    MEDIUM = "medium"          # Moderate incident with notable impact
    HIGH = "high"              # Serious incident with significant impact
    CRITICAL = "critical"      # Severe incident with major impact
    CATASTROPHIC = "catastrophic"  # Extreme incident with widespread impact


class ResponsePhase(Enum):
    """Phases of crisis response."""
    DETECTION = "detection"            # Initial detection of potential crisis
    ASSESSMENT = "assessment"          # Assessment of situation and impact
    IMMEDIATE_RESPONSE = "immediate"   # Immediate response actions
    STABILIZATION = "stabilization"    # Stabilizing the situation
    RECOVERY = "recovery"              # Recovery and return to normal operations
    EVALUATION = "evaluation"          # Post-crisis evaluation and learning


class InformationReliability(Enum):
    """Reliability levels for information sources."""
    UNVERIFIED = "unverified"      # Information not yet verified
    LOW = "low"                    # Low reliability
    MEDIUM = "medium"              # Medium reliability
    HIGH = "high"                  # High reliability
    CONFIRMED = "confirmed"        # Fully confirmed information


class InformationPriority(Enum):
    """Priority levels for information processing."""
    ROUTINE = "routine"            # Routine information
    IMPORTANT = "important"        # Important information
    URGENT = "urgent"              # Urgent information
    CRITICAL = "critical"          # Critical information


class CommunicationChannel(Enum):
    """Communication channels for crisis response."""
    INTERNAL = "internal"          # Internal team communications
    LEADERSHIP = "leadership"      # Communication with leadership
    STAKEHOLDERS = "stakeholders"  # Communication with stakeholders
    PUBLIC = "public"              # Public communications
    AUTHORITIES = "authorities"    # Communication with authorities
    PARTNERS = "partners"          # Communication with partners
    

@dataclass
class InformationSource:
    """Represents a source of information during a crisis."""
    source_id: str
    name: str
    source_type: str  # agent, sensor, system, human, external, etc.
    reliability: InformationReliability
    access_method: str  # API, direct, observation, report, etc.
    capabilities: List[str]  # What kinds of information can this source provide
    refresh_rate: Optional[int] = None  # How often new information is available (seconds)
    last_updated: Optional[str] = None  # ISO format timestamp
    contact_details: Optional[Dict[str, str]] = None
    credentials: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "source_type": self.source_type,
            "reliability": self.reliability.value,
            "access_method": self.access_method,
            "capabilities": self.capabilities,
            "refresh_rate": self.refresh_rate,
            "last_updated": self.last_updated,
            "contact_details": self.contact_details,
            "credentials": self.credentials,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InformationSource':
        """Create from dictionary representation."""
        return cls(
            source_id=data["source_id"],
            name=data["name"],
            source_type=data["source_type"],
            reliability=InformationReliability(data["reliability"]),
            access_method=data["access_method"],
            capabilities=data["capabilities"],
            refresh_rate=data.get("refresh_rate"),
            last_updated=data.get("last_updated"),
            contact_details=data.get("contact_details"),
            credentials=data.get("credentials"),
            metadata=data.get("metadata", {})
        )


@dataclass
class CrisisInformation:
    """Represents a piece of information relevant to a crisis."""
    info_id: str
    title: str
    content: str
    source_id: str
    timestamp: str  # ISO format timestamp
    reliability: InformationReliability
    priority: InformationPriority
    categories: List[str]  # Types of information (damage, casualties, etc.)
    verified: bool = False
    verification_source: Optional[str] = None
    verification_time: Optional[str] = None
    expiration_time: Optional[str] = None
    related_info_ids: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "info_id": self.info_id,
            "title": self.title,
            "content": self.content,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "reliability": self.reliability.value,
            "priority": self.priority.value,
            "categories": self.categories,
            "verified": self.verified,
            "verification_source": self.verification_source,
            "verification_time": self.verification_time,
            "expiration_time": self.expiration_time,
            "related_info_ids": self.related_info_ids,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrisisInformation':
        """Create from dictionary representation."""
        return cls(
            info_id=data["info_id"],
            title=data["title"],
            content=data["content"],
            source_id=data["source_id"],
            timestamp=data["timestamp"],
            reliability=InformationReliability(data["reliability"]),
            priority=InformationPriority(data["priority"]),
            categories=data["categories"],
            verified=data.get("verified", False),
            verification_source=data.get("verification_source"),
            verification_time=data.get("verification_time"),
            expiration_time=data.get("expiration_time"),
            related_info_ids=data.get("related_info_ids", []),
            confidence_score=data.get("confidence_score", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class ResponseAction:
    """Represents a specific action to be taken in response to a crisis."""
    action_id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.CREATED
    assigned_agent_id: Optional[str] = None
    deadline: Optional[str] = None  # ISO format timestamp
    estimated_duration: Optional[int] = None  # minutes
    dependencies: List[str] = field(default_factory=list)  # Other action IDs
    expected_outcomes: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    start_time: Optional[str] = None  # ISO format timestamp
    completion_time: Optional[str] = None  # ISO format timestamp
    resources_required: Dict[str, float] = field(default_factory=dict)  # resource_id -> amount
    resources_allocated: Dict[str, float] = field(default_factory=dict)  # resource_id -> amount
    communication_channel: Optional[CommunicationChannel] = None
    phase: ResponsePhase = ResponsePhase.IMMEDIATE_RESPONSE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_id": self.action_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "deadline": self.deadline,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "expected_outcomes": self.expected_outcomes,
            "progress": self.progress,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "resources_required": self.resources_required,
            "resources_allocated": self.resources_allocated,
            "communication_channel": self.communication_channel.value if self.communication_channel else None,
            "phase": self.phase.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseAction':
        """Create from dictionary representation."""
        return cls(
            action_id=data["action_id"],
            title=data["title"],
            description=data["description"],
            priority=TaskPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            assigned_agent_id=data.get("assigned_agent_id"),
            deadline=data.get("deadline"),
            estimated_duration=data.get("estimated_duration"),
            dependencies=data.get("dependencies", []),
            expected_outcomes=data.get("expected_outcomes", []),
            progress=data.get("progress", 0.0),
            start_time=data.get("start_time"),
            completion_time=data.get("completion_time"),
            resources_required=data.get("resources_required", {}),
            resources_allocated=data.get("resources_allocated", {}),
            communication_channel=CommunicationChannel(data["communication_channel"]) if data.get("communication_channel") else None,
            phase=ResponsePhase(data["phase"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class DecisionPoint:
    """Represents a critical decision that needs to be made during crisis response."""
    decision_id: str
    title: str
    description: str
    options: List[Dict[str, Any]]  # List of possible options with consequences
    deadline: Optional[str] = None  # ISO format timestamp
    decision_maker: Optional[str] = None  # Agent ID or stakeholder who needs to decide
    required_information: List[str] = field(default_factory=list)  # Information needed
    priority: TaskPriority = TaskPriority.HIGH
    status: str = "pending"  # pending, in_progress, decided, postponed
    decision_time: Optional[str] = None  # When was decision made
    selected_option: Optional[int] = None  # Index of selected option
    rationale: Optional[str] = None  # Why was this decision made
    resulting_actions: List[str] = field(default_factory=list)  # Action IDs resulting from decision
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "description": self.description,
            "options": self.options,
            "deadline": self.deadline,
            "decision_maker": self.decision_maker,
            "required_information": self.required_information,
            "priority": self.priority.value,
            "status": self.status,
            "decision_time": self.decision_time,
            "selected_option": self.selected_option,
            "rationale": self.rationale,
            "resulting_actions": self.resulting_actions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionPoint':
        """Create from dictionary representation."""
        return cls(
            decision_id=data["decision_id"],
            title=data["title"],
            description=data["description"],
            options=data["options"],
            deadline=data.get("deadline"),
            decision_maker=data.get("decision_maker"),
            required_information=data.get("required_information", []),
            priority=TaskPriority(data["priority"]),
            status=data["status"],
            decision_time=data.get("decision_time"),
            selected_option=data.get("selected_option"),
            rationale=data.get("rationale"),
            resulting_actions=data.get("resulting_actions", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class CommunicationMessage:
    """Represents a communication message during crisis response."""
    message_id: str
    sender_id: str
    recipients: List[str]
    channel: CommunicationChannel
    subject: str
    content: str
    timestamp: str  # ISO format timestamp
    priority: TaskPriority
    status: str = "draft"  # draft, sent, delivered, read, responded
    response_required: bool = False
    response_deadline: Optional[str] = None  # ISO format timestamp
    response_received: bool = False
    response_time: Optional[str] = None  # ISO format timestamp
    response_content: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    related_info_ids: List[str] = field(default_factory=list)
    related_action_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipients": self.recipients,
            "channel": self.channel.value,
            "subject": self.subject,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "status": self.status,
            "response_required": self.response_required,
            "response_deadline": self.response_deadline,
            "response_received": self.response_received,
            "response_time": self.response_time,
            "response_content": self.response_content,
            "attachments": self.attachments,
            "related_info_ids": self.related_info_ids,
            "related_action_ids": self.related_action_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationMessage':
        """Create from dictionary representation."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            recipients=data["recipients"],
            channel=CommunicationChannel(data["channel"]),
            subject=data["subject"],
            content=data["content"],
            timestamp=data["timestamp"],
            priority=TaskPriority(data["priority"]),
            status=data.get("status", "draft"),
            response_required=data.get("response_required", False),
            response_deadline=data.get("response_deadline"),
            response_received=data.get("response_received", False),
            response_time=data.get("response_time"),
            response_content=data.get("response_content"),
            attachments=data.get("attachments", []),
            related_info_ids=data.get("related_info_ids", []),
            related_action_ids=data.get("related_action_ids", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class SituationReport:
    """A structured report on the current crisis situation."""
    report_id: str
    title: str
    timestamp: str  # ISO format timestamp
    summary: str
    severity: CrisisSeverity
    current_phase: ResponsePhase
    key_developments: List[Dict[str, Any]]
    current_status: Dict[str, Any]
    key_metrics: Dict[str, Any]
    immediate_concerns: List[Dict[str, Any]]
    actions_in_progress: List[Dict[str, Any]]
    planned_actions: List[Dict[str, Any]]
    resource_status: Dict[str, Any]
    information_gaps: List[str]
    recommendations: List[Dict[str, Any]]
    distribution_list: List[str]
    prepared_by: str
    approved_by: Optional[str] = None
    report_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "severity": self.severity.value,
            "current_phase": self.current_phase.value,
            "key_developments": self.key_developments,
            "current_status": self.current_status,
            "key_metrics": self.key_metrics,
            "immediate_concerns": self.immediate_concerns,
            "actions_in_progress": self.actions_in_progress,
            "planned_actions": self.planned_actions,
            "resource_status": self.resource_status,
            "information_gaps": self.information_gaps,
            "recommendations": self.recommendations,
            "distribution_list": self.distribution_list,
            "prepared_by": self.prepared_by,
            "approved_by": self.approved_by,
            "report_version": self.report_version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SituationReport':
        """Create from dictionary representation."""
        return cls(
            report_id=data["report_id"],
            title=data["title"],
            timestamp=data["timestamp"],
            summary=data["summary"],
            severity=CrisisSeverity(data["severity"]),
            current_phase=ResponsePhase(data["current_phase"]),
            key_developments=data["key_developments"],
            current_status=data["current_status"],
            key_metrics=data["key_metrics"],
            immediate_concerns=data["immediate_concerns"],
            actions_in_progress=data["actions_in_progress"],
            planned_actions=data["planned_actions"],
            resource_status=data["resource_status"],
            information_gaps=data["information_gaps"],
            recommendations=data["recommendations"],
            distribution_list=data["distribution_list"],
            prepared_by=data["prepared_by"],
            approved_by=data.get("approved_by"),
            report_version=data.get("report_version", "1.0"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Crisis:
    """Represents a crisis situation requiring coordinated response."""
    crisis_id: str
    name: str
    description: str
    type: str  # Type of crisis (natural disaster, security breach, etc.)
    severity: CrisisSeverity
    current_phase: ResponsePhase
    detected_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    status: str  # active, contained, resolved
    location: Optional[Dict[str, Any]] = None
    affected_systems: List[str] = field(default_factory=list)
    affected_stakeholders: List[Dict[str, Any]] = field(default_factory=list)
    lead_coordinator: Optional[str] = None  # Agent ID or person
    response_team: List[Dict[str, Any]] = field(default_factory=list)
    information: Dict[str, CrisisInformation] = field(default_factory=dict)
    actions: Dict[str, ResponseAction] = field(default_factory=dict)
    decisions: Dict[str, DecisionPoint] = field(default_factory=dict)
    communications: Dict[str, CommunicationMessage] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    situation_reports: Dict[str, SituationReport] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "crisis_id": self.crisis_id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "severity": self.severity.value,
            "current_phase": self.current_phase.value,
            "detected_at": self.detected_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "location": self.location,
            "affected_systems": self.affected_systems,
            "affected_stakeholders": self.affected_stakeholders,
            "lead_coordinator": self.lead_coordinator,
            "response_team": self.response_team,
            "information": {
                info_id: info.to_dict() 
                for info_id, info in self.information.items()
            },
            "actions": {
                action_id: action.to_dict() 
                for action_id, action in self.actions.items()
            },
            "decisions": {
                decision_id: decision.to_dict() 
                for decision_id, decision in self.decisions.items()
            },
            "communications": {
                message_id: message.to_dict() 
                for message_id, message in self.communications.items()
            },
            "resources": {
                resource_id: resource.to_dict() 
                for resource_id, resource in self.resources.items()
            },
            "situation_reports": {
                report_id: report.to_dict() 
                for report_id, report in self.situation_reports.items()
            },
            "timeline": self.timeline,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Crisis':
        """Create from dictionary representation."""
        crisis = cls(
            crisis_id=data["crisis_id"],
            name=data["name"],
            description=data["description"],
            type=data["type"],
            severity=CrisisSeverity(data["severity"]),
            current_phase=ResponsePhase(data["current_phase"]),
            detected_at=data["detected_at"],
            updated_at=data["updated_at"],
            status=data["status"],
            location=data.get("location"),
            affected_systems=data.get("affected_systems", []),
            affected_stakeholders=data.get("affected_stakeholders", []),
            lead_coordinator=data.get("lead_coordinator"),
            response_team=data.get("response_team", []),
            timeline=data.get("timeline", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        
        # Load complex nested objects
        for info_id, info_data in data.get("information", {}).items():
            crisis.information[info_id] = CrisisInformation.from_dict(info_data)
            
        for action_id, action_data in data.get("actions", {}).items():
            crisis.actions[action_id] = ResponseAction.from_dict(action_data)
            
        for decision_id, decision_data in data.get("decisions", {}).items():
            crisis.decisions[decision_id] = DecisionPoint.from_dict(decision_data)
            
        for message_id, message_data in data.get("communications", {}).items():
            crisis.communications[message_id] = CommunicationMessage.from_dict(message_data)
            
        for resource_id, resource_data in data.get("resources", {}).items():
            crisis.resources[resource_id] = Resource.from_dict(resource_data)
            
        for report_id, report_data in data.get("situation_reports", {}).items():
            crisis.situation_reports[report_id] = SituationReport.from_dict(report_data)
            
        return crisis
    
    def add_timeline_event(self, event_type: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the crisis timeline."""
        self.timeline.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "description": description,
            "details": details or {}
        })
        self.updated_at = datetime.now(timezone.utc).isoformat()


class CrisisResponseCoordinator:
    """
    Advanced coordinator for rapidly assessing and responding to urgent situations requiring
    multiple agent expertise.
    
    The CrisisResponseCoordinator extends OrchestratorEngine with specialized capabilities for
    crisis assessment, prioritization, information gathering, decision support, and communication
    management, particularly in high-stress, rapidly changing environments.
    
    It embodies the "Adaptability as a Form of Strength" principle throughout the response process.
    """
    
    def __init__(
        self,
        agent_id: str,
        orchestrator_engine: Optional[OrchestratorEngine] = None,
        communication_adapter: Optional[CommunicationAdapter] = None,
        content_handler: Optional[ContentHandler] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        storage_dir: str = "data/crisis_response"
    ):
        """
        Initialize the crisis response coordinator.
        
        Args:
            agent_id: ID of the coordinator agent
            orchestrator_engine: Existing OrchestratorEngine or None to create new
            communication_adapter: Adapter for agent communication
            content_handler: Handler for content format conversion
            principle_engine: Engine for principle-based reasoning
            relationship_tracker: Tracker for agent relationships
            emotional_intelligence: Module for emotional intelligence
            storage_dir: Directory for storing crisis response data
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.communication_adapter = communication_adapter
        self.content_handler = content_handler
        self.principle_engine = principle_engine
        self.relationship_tracker = relationship_tracker
        self.emotional_intelligence = emotional_intelligence
        
        # Create or use existing orchestrator engine
        if orchestrator_engine:
            self.orchestrator_engine = orchestrator_engine
        else:
            self.orchestrator_engine = OrchestratorEngine(
                agent_id=agent_id,
                communication_adapter=communication_adapter,
                content_handler=content_handler,
                principle_engine=principle_engine,
                relationship_tracker=relationship_tracker,
                storage_dir=f"{storage_dir}/orchestration"
            )
        
        # Active crises
        self.crises: Dict[str, Crisis] = {}
        
        # Information sources
        self.information_sources: Dict[str, InformationSource] = {}
        
        # Agent specializations for crisis response
        self.agent_specializations: Dict[str, Dict[str, float]] = {}
        
        # Crisis response templates
        self.response_templates: Dict[str, Dict[str, Any]] = {}
        
        # Action priority queue
        self.action_queue: List[Tuple[int, float, str, str]] = []  # (priority value, creation time, crisis_id, action_id)
        
        # Locks
        self.crisis_lock = threading.Lock()
        self.source_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        
        # Background processors
        self.processors: Dict[str, threading.Thread] = {}
        self.processor_running: Dict[str, bool] = {}
        
        # Adaptability metrics
        self.adaptability_metrics = {
            "plan_adjustment_rate": 0.0,  # How often plans change based on new information
            "information_processing_time": 0.0,  # Average time to process and act on new information
            "priority_shift_frequency": 0.0,  # How often priorities are re-evaluated
            "resource_reallocation_rate": 0.0,  # How often resources are reallocated
            "communication_responsiveness": 0.0,  # Speed of communication response
            "coordination_effectiveness": 0.0,  # Effectiveness of multi-agent coordination
            "decision_speed": 0.0,  # Speed of decision making
            "adaptation_score": 0.0  # Overall adaptation capability score
        }
        
        logger.info(f"CrisisResponseCoordinator initialized for agent {agent_id}")
    
    def register_information_source(
        self,
        name: str,
        source_type: str,
        reliability: InformationReliability,
        access_method: str,
        capabilities: List[str],
        refresh_rate: Optional[int] = None,
        contact_details: Optional[Dict[str, str]] = None,
        credentials: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InformationSource:
        """
        Register a new information source for crisis situations.
        
        Args:
            name: Source name
            source_type: Type of source (agent, sensor, system, human, external, etc.)
            reliability: Initial reliability assessment
            access_method: How to access the source (API, direct, observation, report, etc.)
            capabilities: What kinds of information this source can provide
            refresh_rate: Optional seconds between updates
            contact_details: Optional contact information for human sources
            credentials: Optional credentials for accessing the source
            metadata: Additional metadata
        
        Returns:
            Newly registered InformationSource
        """
        source_id = f"source-{str(uuid.uuid4())}"
        now = datetime.now(timezone.utc).isoformat()
