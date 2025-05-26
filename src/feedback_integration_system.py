"""
Feedback Integration System

This module implements the FeedbackIntegrationSystem class that actively solicits, processes,
and integrates human feedback into the agent's orchestration and evolution processes.
It applies the "Fairness as a Fundamental Truth" principle by treating all feedback
equitably while balancing potentially conflicting stakeholder needs.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional, Set, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import statistics
from collections import Counter, defaultdict, deque
import copy
import uuid

from principle_engine import PrincipleEngine
from orchestration_analytics import OrchestrationAnalytics
from continuous_evolution_system import ContinuousEvolutionSystem
from learning_system import LearningSystem, OutcomeType
from communication_adapter import CommunicationAdapter
from human_interaction_styler import HumanInteractionStyler
from emotional_intelligence import EmotionalIntelligence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeedbackIntegrationSystem")


class FeedbackType(Enum):
    """Types of feedback the system can handle."""
    ORCHESTRATION_QUALITY = auto()    # Quality of orchestration process
    RESULT_QUALITY = auto()          # Quality of orchestration results
    COMMUNICATION = auto()           # Communication effectiveness
    AGENT_SELECTION = auto()         # Agent selection appropriateness
    CAPABILITY_GAP = auto()          # Missing capability identification
    PRINCIPLE_ALIGNMENT = auto()     # Alignment with principles
    OPTIMIZATION = auto()            # Efficiency and optimization
    FEATURE_REQUEST = auto()         # Request for new features
    GENERAL = auto()                 # General feedback


class FeedbackSource(Enum):
    """Sources of feedback."""
    END_USER = auto()               # Direct users of orchestration outputs
    OPERATOR = auto()               # System operators/administrators
    AGENT_OWNER = auto()            # Stakeholders who own specific agents
    DEVELOPER = auto()              # System developers
    BUSINESS_STAKEHOLDER = auto()   # Business stakeholders
    GOVERNANCE_BODY = auto()        # Governance or oversight bodies
    EXTERNAL_EVALUATOR = auto()     # External review or audit
    UNKNOWN = auto()                # Source not clearly identified


class FeedbackUrgency(Enum):
    """Urgency levels for feedback."""
    CRITICAL = 4     # Must be addressed immediately
    HIGH = 3         # Should be addressed soon
    MEDIUM = 2       # Should be addressed in normal course of operations
    LOW = 1          # Can be addressed when convenient
    INFORMATIONAL = 0  # No action required, just for information


class FeedbackStatus(Enum):
    """Status of feedback processing."""
    RECEIVED = auto()               # Feedback received, not yet processed
    PROCESSING = auto()             # Feedback being analyzed
    PRIORITIZED = auto()            # Feedback prioritized
    ACTION_PLANNED = auto()         # Action plan created
    IN_PROGRESS = auto()            # Implementation in progress
    IMPLEMENTED = auto()            # Changes implemented
    VERIFIED = auto()               # Changes verified
    CLOSED = auto()                 # Feedback loop closed
    DEFERRED = auto()               # Deferred to later
    REJECTED = auto()               # Rejected with explanation


class FeedbackFormat(Enum):
    """Formats for collecting feedback."""
    STRUCTURED_SURVEY = auto()      # Predefined questions with rating scales
    FREE_TEXT = auto()              # Open-ended textual feedback
    VOICE_RECORDING = auto()        # Spoken feedback
    COMPARATIVE_RANKING = auto()    # Ranking of options
    BINARY_CHOICE = auto()          # Yes/no or thumbs up/down feedback
    NUMERICAL_RATING = auto()       # Star or numerical rating
    MULTI_DIMENSIONAL = auto()      # Feedback across multiple dimensions
    INTERACTIVE_DIALOG = auto()     # Conversational feedback collection


@dataclass
class FeedbackItem:
    """A single item of feedback received from a human."""
    feedback_id: str
    content: str                              # The actual feedback content
    feedback_type: FeedbackType               # Type of feedback
    source: FeedbackSource                    # Source of the feedback
    urgency: FeedbackUrgency                  # Urgency/priority
    received_at: str                          # ISO format timestamp
    format: FeedbackFormat                    # Format the feedback was received in
    status: FeedbackStatus = FeedbackStatus.RECEIVED  # Current status
    numerical_ratings: Dict[str, float] = field(default_factory=dict)  # Any numerical ratings
    related_orchestration_id: Optional[str] = None  # Related orchestration ID
    related_agent_ids: List[str] = field(default_factory=list)  # Related agent IDs
    related_capability_ids: List[str] = field(default_factory=list)  # Related capability IDs
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    # Processing information
    sentiment_score: Optional[float] = None   # -1.0 (negative) to 1.0 (positive)
    priority_score: float = 0.0               # Calculated priority (0.0-1.0)
    processed_content: Optional[Dict[str, Any]] = None  # Processed/structured content
    tags: List[str] = field(default_factory=list)  # Extracted tags/themes
    
    # Response tracking
    response_plan: Optional[Dict[str, Any]] = None  # Plan to address feedback
    response_actions: List[Dict[str, Any]] = field(default_factory=list)  # Actions taken
    response_sent_at: Optional[str] = None    # When response was sent
    feedback_resolution: Optional[str] = None  # How feedback was resolved
    stakeholder_satisfaction: Optional[float] = None  # Satisfaction with resolution (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "feedback_id": self.feedback_id,
            "content": self.content,
            "feedback_type": self.feedback_type.name,
            "source": self.source.name,
            "urgency": self.urgency.name,
            "received_at": self.received_at,
            "format": self.format.name,
            "status": self.status.name,
            "numerical_ratings": self.numerical_ratings,
            "related_orchestration_id": self.related_orchestration_id,
            "related_agent_ids": self.related_agent_ids,
            "related_capability_ids": self.related_capability_ids,
            "metadata": self.metadata,
            "sentiment_score": self.sentiment_score,
            "priority_score": self.priority_score,
            "processed_content": self.processed_content,
            "tags": self.tags,
            "response_plan": self.response_plan,
            "response_actions": self.response_actions,
            "response_sent_at": self.response_sent_at,
            "feedback_resolution": self.feedback_resolution,
            "stakeholder_satisfaction": self.stakeholder_satisfaction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create from dictionary representation."""
        return cls(
            feedback_id=data["feedback_id"],
            content=data["content"],
            feedback_type=FeedbackType[data["feedback_type"]],
            source=FeedbackSource[data["source"]],
            urgency=FeedbackUrgency[data["urgency"]],
            received_at=data["received_at"],
            format=FeedbackFormat[data["format"]],
            status=FeedbackStatus[data["status"]],
            numerical_ratings=data.get("numerical_ratings", {}),
            related_orchestration_id=data.get("related_orchestration_id"),
            related_agent_ids=data.get("related_agent_ids", []),
            related_capability_ids=data.get("related_capability_ids", []),
            metadata=data.get("metadata", {}),
            sentiment_score=data.get("sentiment_score"),
            priority_score=data.get("priority_score", 0.0),
            processed_content=data.get("processed_content"),
            tags=data.get("tags", []),
            response_plan=data.get("response_plan"),
            response_actions=data.get("response_actions", []),
            response_sent_at=data.get("response_sent_at"),
            feedback_resolution=data.get("feedback_resolution"),
            stakeholder_satisfaction=data.get("stakeholder_satisfaction")
        )


@dataclass
class FeedbackCollection:
    """Collection of related feedback items, possibly from multiple sources."""
    collection_id: str
    name: str
    description: str
    created_at: str
    feedback_items: List[str] = field(default_factory=list)  # List of feedback IDs
    associated_topic: Optional[str] = None
    status: FeedbackStatus = FeedbackStatus.RECEIVED
    aggregated_priority: float = 0.0
    aggregated_sentiment: Optional[float] = None
    common_tags: List[str] = field(default_factory=list)
    resolution_summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "feedback_items": self.feedback_items,
            "associated_topic": self.associated_topic,
            "status": self.status.name,
            "aggregated_priority": self.aggregated_priority,
            "aggregated_sentiment": self.aggregated_sentiment,
            "common_tags": self.common_tags,
            "resolution_summary": self.resolution_summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackCollection':
        """Create from dictionary representation."""
        return cls(
            collection_id=data["collection_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"],
            feedback_items=data.get("feedback_items", []),
            associated_topic=data.get("associated_topic"),
            status=FeedbackStatus[data["status"]],
            aggregated_priority=data.get("aggregated_priority", 0.0),
            aggregated_sentiment=data.get("aggregated_sentiment"),
            common_tags=data.get("common_tags", []),
            resolution_summary=data.get("resolution_summary")
        )


@dataclass
class FeedbackSolicitationTemplate:
    """Template for soliciting feedback."""
    template_id: str
    name: str
    description: str
    target_audience: List[FeedbackSource]
    format: FeedbackFormat
    questions: List[Dict[str, Any]]
    introduction: str
    conclusion: str
    estimated_completion_time: int  # in minutes
    contextual_data_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "target_audience": [audience.name for audience in self.target_audience],
            "format": self.format.name,
            "questions": self.questions,
            "introduction": self.introduction,
            "conclusion": self.conclusion,
            "estimated_completion_time": self.estimated_completion_time,
            "contextual_data_requirements": self.contextual_data_requirements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackSolicitationTemplate':
        """Create from dictionary representation."""
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            target_audience=[FeedbackSource[src] for src in data["target_audience"]],
            format=FeedbackFormat[data["format"]],
            questions=data["questions"],
            introduction=data["introduction"],
            conclusion=data["conclusion"],
            estimated_completion_time=data["estimated_completion_time"],
            contextual_data_requirements=data.get("contextual_data_requirements", {})
        )


@dataclass
class FeedbackSolicitationCampaign:
    """A campaign to solicit feedback from stakeholders."""
    campaign_id: str
    name: str
    description: str
    template_id: str
    created_at: str
    start_date: str
    end_date: str
    status: str  # "planned", "active", "completed", "cancelled"
    target_stakeholders: List[str]  # IDs or groups
    distribution_channels: List[str]
    response_count: int = 0
    contextual_data: Dict[str, Any] = field(default_factory=dict)
    feedback_item_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "template_id": self.template_id,
            "created_at": self.created_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "target_stakeholders": self.target_stakeholders,
            "distribution_channels": self.distribution_channels,
            "response_count": self.response_count,
            "contextual_data": self.contextual_data,
            "feedback_item_ids": self.feedback_item_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackSolicitationCampaign':
        """Create from dictionary representation."""
        return cls(
            campaign_id=data["campaign_id"],
            name=data["name"],
            description=data["description"],
            template_id=data["template_id"],
            created_at=data["created_at"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            status=data["status"],
            target_stakeholders=data["target_stakeholders"],
            distribution_channels=data["distribution_channels"],
            response_count=data.get("response_count", 0),
            contextual_data=data.get("contextual_data", {}),
            feedback_item_ids=data.get("feedback_item_ids", [])
        )


@dataclass
class StakeholderProfile:
    """Profile information for stakeholders who provide feedback."""
    stakeholder_id: str
    name: str
    roles: List[FeedbackSource]
    contact_info: Dict[str, str]
    preferred_communication_channels: List[str]
    feedback_history: List[str] = field(default_factory=list)  # List of feedback IDs
    response_rate: float = 0.0
    average_satisfaction: Optional[float] = None
    feedback_preferences: Dict[str, Any] = field(default_factory=dict)
    last_contact: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stakeholder_id": self.stakeholder_id,
            "name": self.name,
            "roles": [role.name for role in self.roles],
            "contact_info": self.contact_info,
            "preferred_communication_channels": self.preferred_communication_channels,
            "feedback_history": self.feedback_history,
            "response_rate": self.response_rate,
            "average_satisfaction": self.average_satisfaction,
            "feedback_preferences": self.feedback_preferences,
            "last_contact": self.last_contact,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StakeholderProfile':
        """Create from dictionary representation."""
        return cls(
            stakeholder_id=data["stakeholder_id"],
            name=data["name"],
            roles=[FeedbackSource[role] for role in data["roles"]],
            contact_info=data["contact_info"],
            preferred_communication_channels=data["preferred_communication_channels"],
            feedback_history=data.get("feedback_history", []),
            response_rate=data.get("response_rate", 0.0),
            average_satisfaction=data.get("average_satisfaction"),
            feedback_preferences=data.get("feedback_preferences", {}),
            last_contact=data.get("last_contact"),
            metadata=data.get("metadata", {})
        )


class FeedbackIntegrationSystem:
    """
    System for soliciting, processing, and integrating human feedback into the 
    agent's orchestration and evolution processes.
    
    The FeedbackIntegrationSystem applies the "Fairness as a Fundamental Truth" principle
    by treating all feedback equitably while balancing potentially conflicting
    stakeholder needs to drive continuous improvement.
    """
    
    def __init__(
        self,
        agent_id: str,
        principle_engine: Optional[PrincipleEngine] = None,
        orchestration_analytics: Optional[OrchestrationAnalytics] = None,
        continuous_evolution_system: Optional[ContinuousEvolutionSystem] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        human_interaction_styler: Optional[HumanInteractionStyler] = None,
        communication_adapter: Optional[CommunicationAdapter] = None,
        feedback_storage_dir: str = "data/feedback"
    ):
        """
        Initialize the feedback integration system.
        
        Args:
            agent_id: ID of the agent
            principle_engine: Engine for principle-based reasoning
            orchestration_analytics: Analytics system for orchestration
            continuous_evolution_system: System for continuous evolution
            emotional_intelligence: System for emotional intelligence
            human_interaction_styler: System for styling human interactions
            communication_adapter: Adapter for communication
            feedback_storage_dir: Directory for storing feedback data
        """
        self.agent_id = agent_id
        self.principle_engine = principle_engine
        self.orchestration_analytics = orchestration_analytics
        self.continuous_evolution_system = continuous_evolution_system
        self.emotional_intelligence = emotional_intelligence
        self.human_interaction_styler = human_interaction_styler
        self.communication_adapter = communication_adapter
        
        # Feedback storage
        self.feedback_storage_dir = feedback_storage_dir
        os.makedirs(feedback_storage_dir, exist_ok=True)
        
        # Feedback items and collections
        self.feedback_items: Dict[str, FeedbackItem] = {}
        self.feedback_collections: Dict[str, FeedbackCollection] = {}
        
        # Solicitation templates and campaigns
        self.solicitation_templates: Dict[str, FeedbackSolicitationTemplate] = {}
        self.solicitation_campaigns: Dict[str, FeedbackSolicitationCampaign] = {}
        
        # Stakeholder profiles
        self.stakeholder_profiles: Dict[str, StakeholderProfile] = {}
        
        # Feedback processing configuration
        self.priority_weights = {
            "urgency": 0.3,
            "source": 0.2,
            "sentiment": 0.1,
            "feedback_type": 0.2,
            "recency": 0.1,
            "impact": 0.1
        }
        
        # Source weights for prioritization
        self.source_weights = {
            FeedbackSource.END_USER: 0.9,
            FeedbackSource.OPERATOR: 0.8,
            FeedbackSource.AGENT_OWNER: 0.7,
            FeedbackSource.BUSINESS_STAKEHOLDER: 0.7,
            FeedbackSource.DEVELOPER: 0.6,
            FeedbackSource.GOVERNANCE_BODY: 0.8,
            FeedbackSource.EXTERNAL_EVALUATOR: 0.6,
            FeedbackSource.UNKNOWN: 0.5
        }
        
        # Type weights for prioritization
        self.type_weights = {
            FeedbackType.ORCHESTRATION_QUALITY: 0.8,
            FeedbackType.RESULT_QUALITY: 0.9,
            FeedbackType.COMMUNICATION: 0.7,
            FeedbackType.AGENT_SELECTION: 0.7,
            FeedbackType.CAPABILITY_GAP: 0.8,
            FeedbackType.PRINCIPLE_ALIGNMENT: 0.9,
            FeedbackType.OPTIMIZATION: 0.6,
            FeedbackType.FEATURE_REQUEST: 0.5,
            FeedbackType.GENERAL: 0.4
        }
        
        # Initialize default solicitation templates
        self._initialize_default_templates()
        
        # Load any existing feedback data
        self._load_feedback_data()
        
        logger.info(f"FeedbackIntegrationSystem initialized for agent {agent_id}")
    
    def _initialize_default_templates(self) -> None:
        """Initialize default feedback solicitation templates."""
        # Post-orchestration feedback template
        post_orchestration_template = FeedbackSolicitationTemplate(
            template_id=f"template-post-orchestration-{uuid.uuid4()}",
            name="Post-Orchestration Feedback",
            description="Collect feedback immediately after an orchestration task completes",
            target_audience=[FeedbackSource.END_USER, FeedbackSource.OPERATOR],
            format=FeedbackFormat.STRUCTURED_SURVEY,
            questions=[
                {
                    "id": "overall_satisfaction",
                    "type": "rating",
                    "text": "How satisfied are you with the overall orchestration process?",
                    "scale": "1-5",
                    "required": True
                },
                {
                    "id": "result_quality",
                    "type": "rating",
                    "text": "How would you rate the quality of the results?",
                    "scale": "1-5",
                    "required": True
                },
                {
                    "id": "timeliness",
                    "type": "rating",
                    "text": "How satisfied are you with the timeliness of the orchestration?",
                    "scale": "1-5",
                    "required": True
                },
                {
                    "id": "communication",
                    "type": "rating",
                    "text": "How effective was the communication during orchestration?",
                    "scale": "1-5",
                    "required": True
                },
                {
                    "id": "strengths",
                    "type": "text",
                    "text": "What worked well in this orchestration?",
                    "required": False
                },
                {
                    "id": "improvements",
                    "type": "text",
                    "text": "What could be improved?",
                    "required": False
                },
                {
                    "id": "additional_comments",
                    "type": "text",
                    "text": "Any additional comments or suggestions?",
                    "required": False
                }
            ],
            introduction="Your feedback helps us improve our orchestration capabilities. Please take a moment to share your thoughts on the recently completed task.",
            conclusion="Thank you for your valuable feedback! We'll use this to continuously improve our orchestration process.",
            estimated_completion_time=5,
            contextual_data_requirements={
                "orchestration_id": True,
                "task_description": True,
                "completion_time": True,
                "agents_involved": True
            }
        )
        
        # Comprehensive system evaluation template
        system_evaluation_template = FeedbackSolicitationTemplate(
            template_id=f"template-system-evaluation-{uuid.uuid4()}",
            name="Comprehensive System Evaluation",
            description="In-depth evaluation of the orchestration system's effectiveness",
            target_audience=[
                FeedbackSource.OPERATOR, 
                FeedbackSource.AGENT_OWNER,
                FeedbackSource.BUSINESS_STAKEHOLDER
            ],
            format=FeedbackFormat.MULTI_DIMENSIONAL,
            questions=[
                {
                    "id": "orchestration_effectiveness",
                    "type": "category",
                    "text": "Orchestration Effectiveness",
                    "subcategories": [
                        {
                            "id": "task_decomposition",
                            "type": "rating",
                            "text": "How effectively are tasks being decomposed?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "agent_selection",
                            "type": "rating",
                            "text": "How appropriate is the selection of agents for tasks?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "resource_allocation",
                            "type": "rating",
                            "text": "How efficient is the allocation of resources?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "dependency_management",
                            "type": "rating",
                            "text": "How well are task dependencies managed?",
                            "scale": "1-5",
                            "required": True
                        }
                    ]
                },
                {
                    "id": "output_quality",
                    "type": "category",
                    "text": "Output Quality",
                    "subcategories": [
                        {
                            "id": "accuracy",
                            "type": "rating",
                            "text": "How accurate are the orchestration results?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "consistency",
                            "type": "rating",
                            "text": "How consistent are the results across similar tasks?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "completeness",
                            "type": "rating",
                            "text": "How complete are the orchestration results?",
                            "scale": "1-5",
                            "required": True
                        }
                    ]
                },
                {
                    "id": "principle_alignment",
                    "type": "category",
                    "text": "Principle Alignment",
                    "subcategories": [
                        {
                            "id": "fairness",
                            "type": "rating",
                            "text": "How well does the system embody the 'Fairness as a Fundamental Truth' principle?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "adaptability",
                            "type": "rating",
                            "text": "How effectively does the system demonstrate 'Adaptability as Strength'?",
                            "scale": "1-5",
                            "required": True
                        },
                        {
                            "id": "resilience",
                            "type": "rating",
                            "text": "How well does the system implement 'Resilience Through Reflection'?",
                            "scale": "1-5",
                            "required": True
                        }
                    ]
                },
                {
                    "id": "improvement_opportunities",
                    "type": "text",
                    "text": "What specific areas of the orchestration system need improvement?",
                    "required": True
                },
                {
                    "id": "capability_gaps",
                    "type": "text",
                    "text": "Are there any missing capabilities that would enhance orchestration?",
                    "required": False
                }
            ],
            introduction="This comprehensive evaluation helps us understand the overall effectiveness of our orchestration system. Your detailed feedback will guide strategic improvements.",
            conclusion="Thank you for completing this comprehensive evaluation. Your insights are invaluable for our continuous improvement process.",
            estimated_completion_time=15,
            contextual_data_requirements={
                "system_version": True,
                "evaluation_period": True,
                "performance_metrics": True
            }
        )
        
        # Interactive feedback dialog template
        interactive_feedback_template = FeedbackSolicitationTemplate(
            template_id=f"template-interactive-dialog-{uuid.uuid4()}",
            name="Interactive Feedback Dialog",
            description="Conversational feedback collection through guided dialog",
            target_audience=[FeedbackSource.END_USER, FeedbackSource.BUSINESS_STAKEHOLDER],
            format=FeedbackFormat.INTERACTIVE_DIALOG,
            questions=[
                {
                    "id": "initial_impression",
                    "type": "open",
                    "text": "What's your overall impression of how the orchestration worked?",
                    "followups": [
                        {
                            "condition": "positive",
                            "question": "What specific aspects did you find most effective?"
                        },
                        {
                            "condition": "negative",
                            "question": "What specific issues did you encounter?"
                        },
                        {
                            "condition": "neutral",
                            "question": "What aspects were satisfactory and which ones could be improved?"
                        }
                    ]
                },
                {
                    "id": "expectations",
                    "type": "open",
                    "text": "Did the orchestration process meet your expectations? Why or why not?",
                    "required": True
                },
                {
                    "id": "key_improvements",
                    "type": "open",
                    "text": "If you could change one thing about the orchestration process, what would it be?",
                    "required": True
                }
            ],
            introduction="I'd like to have a conversation about your experience with our orchestration process. This will help us better understand your needs and improve our system.",
            conclusion="Thank you for sharing your thoughts. This kind of direct feedback is incredibly valuable for our improvement process.",
            estimated_completion_time=10,
            contextual_data_requirements={
                "user_role": True,
                "prior_interactions": True
            }
        )
        
        # Add templates to the system
        self.solicitation_templates = {**self.solicitation_templates, post_orchestration_template.template_id: post_orchestration_template}
        self.solicitation_templates = {**self.solicitation_templates, system_evaluation_template.template_id: system_evaluation_template}
        self.solicitation_templates = {**self.solicitation_templates, interactive_feedback_template.template_id: interactive_feedback_template}
    
    def _load_feedback_data(self) -> None:
        """Load feedback data from storage."""
        try:
            # Load feedback items
            feedback_items_path = os.path.join(self.feedback_storage_dir, "feedback_items.json")
            if os.path.exists(feedback_items_path):
                with open(feedback_items_path, "r") as f:
                    items_data = json.load(f)
                    for item_id, item_data in items_data.items():
                        self.feedback_items = {**self.feedback_items, item_id: FeedbackItem.from_dict(item_data)}
            
            # Load feedback collections
            collections_path = os.path.join(self.feedback_storage_dir, "feedback_collections.json")
            if os.path.exists(collections_path):
                with open(collections_path, "r") as f:
                    collections_data = json.load(f)
                    for collection_id, collection_data in collections_data.items():
                        self.feedback_collections = {**self.feedback_collections, collection_id: FeedbackCollection.from_dict(collection_data)}
            
            # Load solicitation templates
            templates_path = os.path.join(self.feedback_storage_dir, "solicitation_templates.json")
            if os.path.exists(templates_path):
                with open(templates_path, "r") as f:
                    templates_data = json.load(f)
                    for template_id, template_data in templates_data.items():
                        if template_id not in self.solicitation_templates:  # Don't overwrite defaults
                            self.solicitation_templates = {**self.solicitation_templates, template_id: FeedbackSolicitationTemplate.from_dict(template_data)}
            
            # Load solicitation campaigns
            campaigns_path = os.path.join(self.feedback_storage_dir, "solicitation_campaigns.json")
            if os.path.exists(campaigns_path):
                with open(campaigns_path, "r") as f:
                    campaigns_data = json.load(f)
                    for campaign_id, campaign_data in campaigns_data.items():
                        self.solicitation_campaigns = {**self.solicitation_campaigns, campaign_id: FeedbackSolicitationCampaign.from_dict(campaign_data)}
            
            # Load stakeholder profiles
            profiles_path = os.path.join(self.feedback_storage_dir, "stakeholder_profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, "r") as f:
                    profiles_data = json.load(f)
                    for stakeholder_id, profile_data in profiles_data.items():
                        self.stakeholder_profiles = {**self.stakeholder_profiles, stakeholder_id: StakeholderProfile.from_dict(profile_data)}
                        
            logger.info(f"Loaded feedback data: {len(self.feedback_items)} items, " +
                      f"{len(self.feedback_collections)} collections, " +
                      f"{len(self.solicitation_campaigns)} campaigns, " +
                      f"{len(self.stakeholder_profiles)} stakeholder profiles")
                      
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
