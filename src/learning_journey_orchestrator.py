"""
Learning Journey Orchestrator Extension

This module extends the ProjectOrchestrator with specialized capabilities for
coordinating educational experiences across multiple knowledge domains.

The LearningJourneyOrchestrator implements sophisticated learning path adaptation,
educational agent selection, consistent knowledge building, and learning outcome measurement.
It embodies the "Growth as a Shared Journey" principle throughout the learning process.
"""

import json
import uuid
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
import threading
import copy
from dataclasses import dataclass, field

# Import related modules
from project_orchestrator import (
    ProjectOrchestrator, MilestoneStatus, ResourceType,
    Resource, Milestone, Project, StatusUpdate
)
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
from learning_system import LearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LearningJourneyOrchestrator")


class LearningDomainStatus(Enum):
    """Status of learning domains."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    NEEDS_REVIEW = "needs_review"
    MASTERED = "mastered"


class LearningModuleStatus(Enum):
    """Status of learning modules."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    MASTERED = "mastered"


class LearningActivityStatus(Enum):
    """Status of learning activities."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    MASTERED = "mastered"
    SKIPPED = "skipped"


class LearningStyle(Enum):
    """Learning styles for adaptive education."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING_WRITING = "reading_writing"
    KINESTHETIC = "kinesthetic"
    MULTIMODAL = "multimodal"
    SOCIAL = "social"
    SOLITARY = "solitary"
    LOGICAL = "logical"
    VERBAL = "verbal"


class DifficultyLevel(Enum):
    """Difficulty levels for learning content."""
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CompetencyLevel(Enum):
    """Competency levels for learning assessment."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    EXPERT = "expert"
    MASTER = "master"


@dataclass
class LearnerProfile:
    """Profile of a learner with preferences, history, and progress."""
    learner_id: str
    name: str
    preferred_learning_styles: List[LearningStyle]
    current_competency_levels: Dict[str, CompetencyLevel]  # domain -> level
    learning_speed: Dict[str, float]  # domain -> relative speed (1.0 = average)
    interests: List[str]
    motivation_factors: List[str]
    previous_knowledge_domains: List[str]
    completed_modules: List[str]
    learning_history: List[Dict[str, Any]]
    current_goals: List[Dict[str, Any]]
    feedback_history: List[Dict[str, Any]]
    preference_settings: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "learner_id": self.learner_id,
            "name": self.name,
            "preferred_learning_styles": [style.value for style in self.preferred_learning_styles],
            "current_competency_levels": {
                domain: level.value for domain, level in self.current_competency_levels.items()
            },
            "learning_speed": self.learning_speed,
            "interests": self.interests,
            "motivation_factors": self.motivation_factors,
            "previous_knowledge_domains": self.previous_knowledge_domains,
            "completed_modules": self.completed_modules,
            "learning_history": self.learning_history,
            "current_goals": self.current_goals,
            "feedback_history": self.feedback_history,
            "preference_settings": self.preference_settings,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnerProfile':
        """Create from dictionary representation."""
        return cls(
            learner_id=data["learner_id"],
            name=data["name"],
            preferred_learning_styles=[
                LearningStyle(style) for style in data["preferred_learning_styles"]
            ],
            current_competency_levels={
                domain: CompetencyLevel(level)
                for domain, level in data["current_competency_levels"].items()
            },
            learning_speed=data["learning_speed"],
            interests=data["interests"],
            motivation_factors=data["motivation_factors"],
            previous_knowledge_domains=data["previous_knowledge_domains"],
            completed_modules=data["completed_modules"],
            learning_history=data["learning_history"],
            current_goals=data["current_goals"],
            feedback_history=data["feedback_history"],
            preference_settings=data["preference_settings"],
            metadata=data.get("metadata", {})
        )


@dataclass
class LearningObjective:
    """Learning objective for a module or activity."""
    objective_id: str
    description: str
    bloom_taxonomy_level: str  # remember, understand, apply, analyze, evaluate, create
    assessment_criteria: List[str]
    required_competency_level: CompetencyLevel
    target_competency_level: CompetencyLevel
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "objective_id": self.objective_id,
            "description": self.description,
            "bloom_taxonomy_level": self.bloom_taxonomy_level,
            "assessment_criteria": self.assessment_criteria,
            "required_competency_level": self.required_competency_level.value,
            "target_competency_level": self.target_competency_level.value,
            "keywords": self.keywords,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningObjective':
        """Create from dictionary representation."""
        return cls(
            objective_id=data["objective_id"],
            description=data["description"],
            bloom_taxonomy_level=data["bloom_taxonomy_level"],
            assessment_criteria=data["assessment_criteria"],
            required_competency_level=CompetencyLevel(data["required_competency_level"]),
            target_competency_level=CompetencyLevel(data["target_competency_level"]),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class LearningActivity:
    """Represents a specific learning activity within a module."""
    activity_id: str
    title: str
    description: str
    activity_type: str  # lecture, exercise, quiz, project, discussion, etc.
    learning_objectives: List[LearningObjective]
    estimated_duration: int  # minutes
    difficulty_level: DifficultyLevel
    suited_learning_styles: List[LearningStyle]
    prerequisites: List[str]  # IDs of prerequisite activities
    content_resources: List[Dict[str, Any]]
    assessment_method: Optional[str] = None
    status: LearningActivityStatus = LearningActivityStatus.NOT_STARTED
    completion_percentage: float = 0.0
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "activity_id": self.activity_id,
            "title": self.title,
            "description": self.description,
            "activity_type": self.activity_type,
            "learning_objectives": [obj.to_dict() for obj in self.learning_objectives],
            "estimated_duration": self.estimated_duration,
            "difficulty_level": self.difficulty_level.value,
            "suited_learning_styles": [style.value for style in self.suited_learning_styles],
            "prerequisites": self.prerequisites,
            "content_resources": self.content_resources,
            "assessment_method": self.assessment_method,
            "status": self.status.value,
            "completion_percentage": self.completion_percentage,
            "feedback": self.feedback,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningActivity':
        """Create from dictionary representation."""
        return cls(
            activity_id=data["activity_id"],
            title=data["title"],
            description=data["description"],
            activity_type=data["activity_type"],
            learning_objectives=[
                LearningObjective.from_dict(obj) for obj in data["learning_objectives"]
            ],
            estimated_duration=data["estimated_duration"],
            difficulty_level=DifficultyLevel(data["difficulty_level"]),
            suited_learning_styles=[
                LearningStyle(style) for style in data["suited_learning_styles"]
            ],
            prerequisites=data["prerequisites"],
            content_resources=data["content_resources"],
            assessment_method=data.get("assessment_method"),
            status=LearningActivityStatus(data["status"]),
            completion_percentage=data["completion_percentage"],
            feedback=data.get("feedback", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class LearningModule:
    """Represents a learning module with activities and objectives."""
    module_id: str
    name: str
    description: str
    knowledge_domain: str
    learning_objectives: List[LearningObjective]
    activities: Dict[str, LearningActivity]
    prerequisites: List[str]  # IDs of prerequisite modules
    difficulty_level: DifficultyLevel
    estimated_duration: int  # minutes
    recommended_sequence: List[str]  # IDs of activities in recommended order
    status: LearningModuleStatus = LearningModuleStatus.NOT_STARTED
    completion_percentage: float = 0.0
    mastery_threshold: float = 0.8  # percentage required for mastery
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "module_id": self.module_id,
            "name": self.name,
            "description": self.description,
            "knowledge_domain": self.knowledge_domain,
            "learning_objectives": [obj.to_dict() for obj in self.learning_objectives],
            "activities": {
                activity_id: activity.to_dict()
                for activity_id, activity in self.activities.items()
            },
            "prerequisites": self.prerequisites,
            "difficulty_level": self.difficulty_level.value,
            "estimated_duration": self.estimated_duration,
            "recommended_sequence": self.recommended_sequence,
            "status": self.status.value,
            "completion_percentage": self.completion_percentage,
            "mastery_threshold": self.mastery_threshold,
            "feedback": self.feedback,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningModule':
        """Create from dictionary representation."""
        return cls(
            module_id=data["module_id"],
            name=data["name"],
            description=data["description"],
            knowledge_domain=data["knowledge_domain"],
            learning_objectives=[
                LearningObjective.from_dict(obj) for obj in data["learning_objectives"]
            ],
            activities={
                activity_id: LearningActivity.from_dict(activity)
                for activity_id, activity in data["activities"].items()
            },
            prerequisites=data["prerequisites"],
            difficulty_level=DifficultyLevel(data["difficulty_level"]),
            estimated_duration=data["estimated_duration"],
            recommended_sequence=data["recommended_sequence"],
            status=LearningModuleStatus(data["status"]),
            completion_percentage=data["completion_percentage"],
            mastery_threshold=data.get("mastery_threshold", 0.8),
            feedback=data.get("feedback", []),
            metadata=data.get("metadata", {})
        )
    
    def update_progress(self) -> float:
        """Update module progress based on activity completion."""
        if not self.activities:
            self.completion_percentage = 0.0
            return self.completion_percentage
        
        total_progress = sum(
            activity.completion_percentage for activity in self.activities.values()
        )
        self.completion_percentage = total_progress / len(self.activities)
        
        # Update module status based on progress
        if self.completion_percentage >= self.mastery_threshold:
            # Check if individual activities need review
            if any(activity.status == LearningActivityStatus.NEEDS_REVIEW 
                   for activity in self.activities.values()):
                self.status = LearningModuleStatus.NEEDS_REVIEW
            else:
                self.status = LearningModuleStatus.MASTERED
        elif self.completion_percentage >= 1.0:
            self.status = LearningModuleStatus.COMPLETED
        elif self.completion_percentage > 0:
            self.status = LearningModuleStatus.IN_PROGRESS
        else:
            self.status = LearningModuleStatus.NOT_STARTED
            
        return self.completion_percentage


@dataclass
class LearningDomain:
    """Represents a knowledge domain with learning modules."""
    domain_id: str
    name: str
    description: str
    modules: Dict[str, LearningModule]
    prerequisites: List[str]  # IDs of prerequisite domains
    recommended_sequence: List[str]  # IDs of modules in recommended order
    taxonomy: Dict[str, List[str]]  # Taxonomy of concepts and subtopics
    competency_model: Dict[str, Dict[str, Any]]  # Competency levels and criteria
    status: LearningDomainStatus = LearningDomainStatus.NOT_STARTED
    completion_percentage: float = 0.0
    mastery_threshold: float = 0.8  # percentage required for mastery
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "description": self.description,
            "modules": {
                module_id: module.to_dict()
                for module_id, module in self.modules.items()
            },
            "prerequisites": self.prerequisites,
            "recommended_sequence": self.recommended_sequence,
            "taxonomy": self.taxonomy,
            "competency_model": self.competency_model,
            "status": self.status.value,
            "completion_percentage": self.completion_percentage,
            "mastery_threshold": self.mastery_threshold,
            "feedback": self.feedback,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningDomain':
        """Create from dictionary representation."""
        return cls(
            domain_id=data["domain_id"],
            name=data["name"],
            description=data["description"],
            modules={
                module_id: LearningModule.from_dict(module)
                for module_id, module in data["modules"].items()
            },
            prerequisites=data["prerequisites"],
            recommended_sequence=data["recommended_sequence"],
            taxonomy=data["taxonomy"],
            competency_model=data["competency_model"],
            status=LearningDomainStatus(data["status"]),
            completion_percentage=data["completion_percentage"],
            mastery_threshold=data.get("mastery_threshold", 0.8),
            feedback=data.get("feedback", []),
            metadata=data.get("metadata", {})
        )
    
    def update_progress(self) -> float:
        """Update domain progress based on module completion."""
        if not self.modules:
            self.completion_percentage = 0.0
            return self.completion_percentage
        
        total_progress = sum(
            module.completion_percentage for module in self.modules.values()
        )
        self.completion_percentage = total_progress / len(self.modules)
        
        # Update domain status based on progress
        if self.completion_percentage >= self.mastery_threshold:
            # Check if individual modules need review
            if any(module.status == LearningModuleStatus.NEEDS_REVIEW 
                   for module in self.modules.values()):
                self.status = LearningDomainStatus.NEEDS_REVIEW
            else:
                self.status = LearningDomainStatus.MASTERED
        elif self.completion_percentage >= 1.0:
            self.status = LearningDomainStatus.COMPLETED
        elif self.completion_percentage > 0:
            self.status = LearningDomainStatus.IN_PROGRESS
        else:
            self.status = LearningDomainStatus.NOT_STARTED
            
        return self.completion_percentage


@dataclass
class LearningJourney:
    """Represents a comprehensive learning journey across multiple domains."""
    journey_id: str
    name: str
    description: str
    learner_id: str
    domains: Dict[str, LearningDomain]
    recommended_sequence: List[str]  # IDs of domains in recommended order
    created_at: str  # ISO format date-time
    updated_at: str  # ISO format date-time
    start_date: str  # ISO format date
    end_date: Optional[str] = None  # ISO format date
    status: str = "active"  # active, paused, completed
    overall_progress: float = 0.0
    learning_path_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)  # Knowledge connections
    learning_outcomes: Dict[str, Any] = field(default_factory=dict)  # Measured outcomes
    feedback_summary: Dict[str, Any] = field(default_factory=dict)  # Aggregated feedback
    active_session: Optional[Dict[str, Any]] = None  # Current learning session
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "journey_id": self.journey_id,
            "name": self.name,
            "description": self.description,
            "learner_id": self.learner_id,
            "domains": {
                domain_id: domain.to_dict()
                for domain_id, domain in self.domains.items()
            },
            "recommended_sequence": self.recommended_sequence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "overall_progress": self.overall_progress,
            "learning_path_adjustments": self.learning_path_adjustments,
            "knowledge_graph": self.knowledge_graph,
            "learning_outcomes": self.learning_outcomes,
            "feedback_summary": self.feedback_summary,
            "active_session": self.active_session,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningJourney':
        """Create from dictionary representation."""
        return cls(
            journey_id=data["journey_id"],
            name=data["name"],
            description=data["description"],
            learner_id=data["learner_id"],
            domains={
                domain_id: LearningDomain.from_dict(domain)
                for domain_id, domain in data["domains"].items()
            },
            recommended_sequence=data["recommended_sequence"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            start_date=data["start_date"],
            end_date=data.get("end_date"),
            status=data["status"],
            overall_progress=data["overall_progress"],
            learning_path_adjustments=data.get("learning_path_adjustments", []),
            knowledge_graph=data.get("knowledge_graph", {}),
            learning_outcomes=data.get("learning_outcomes", {}),
            feedback_summary=data.get("feedback_summary", {}),
            active_session=data.get("active_session"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
    
    def update_progress(self) -> float:
        """Update journey progress based on domain completion."""
        if not self.domains:
            self.overall_progress = 0.0
            return self.overall_progress
        
        total_progress = sum(
            domain.completion_percentage for domain in self.domains.values()
        )
        self.overall_progress = total_progress / len(self.domains)
        self.updated_at = datetime.utcnow().isoformat()
        
        # Update status if complete
        if self.overall_progress >= 1.0:
            self.status = "completed"
            self.end_date = datetime.utcnow().isoformat()
            
        return self.overall_progress


@dataclass
class LearningSessionMetrics:
    """Metrics from a learning session."""
    session_id: str
    learner_id: str
    start_time: str  # ISO format date-time
    end_time: Optional[str] = None  # ISO format date-time
    duration: int = 0  # seconds
    domains_visited: List[str] = field(default_factory=list)
    modules_visited: List[str] = field(default_factory=list)
    activities_completed: List[str] = field(default_factory=list)
    assessments_taken: List[Dict[str, Any]] = field(default_factory=list)
    questions_asked: int = 0
    content_viewed: List[Dict[str, Any]] = field(default_factory=list)
    engagement_score: float = 0.0
    focus_metrics: Dict[str, Any] = field(default_factory=dict)
    comprehension_metrics: Dict[str, Any] = field(default_factory=dict)
    knowledge_gains: Dict[str, float] = field(default_factory=dict)
    feedback_provided: List[Dict[str, Any]] = field(default_factory=list)
    technical_issues: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "learner_id": self.learner_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "domains_visited": self.domains_visited,
            "modules_visited": self.modules_visited,
            "activities_completed": self.activities_completed,
            "assessments_taken": self.assessments_taken,
            "questions_asked": self.questions_asked,
            "content_viewed": self.content_viewed,
            "engagement_score": self.engagement_score,
            "focus_metrics": self.focus_metrics,
            "comprehension_metrics": self.comprehension_metrics,
            "knowledge_gains": self.knowledge_gains,
            "feedback_provided": self.feedback_provided,
            "technical_issues": self.technical_issues,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningSessionMetrics':
        """Create from dictionary representation."""
        return cls(
            session_id=data["session_id"],
            learner_id=data["learner_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            duration=data.get("duration", 0),
            domains_visited=data.get("domains_visited", []),
            modules_visited=data.get("modules_visited", []),
            activities_completed=data.get("activities_completed", []),
            assessments_taken=data.get("assessments_taken", []),
            questions_asked=data.get("questions_asked", 0),
            content_viewed=data.get("content_viewed", []),
            engagement_score=data.get("engagement_score", 0.0),
            focus_metrics=data.get("focus_metrics", {}),
            comprehension_metrics=data.get("comprehension_metrics", {}),
            knowledge_gains=data.get("knowledge_gains", {}),
            feedback_provided=data.get("feedback_provided", []),
            technical_issues=data.get("technical_issues", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class LearningOutcomeReport:
    """Report on learning outcomes for stakeholders."""
    report_id: str
    journey_id: str
    learner_id: str
    generated_at: str  # ISO format date-time
    reporting_period: Dict[str, str]  # start_date, end_date in ISO format
    overall_progress: float
    domain_progress: Dict[str, float]
    completed_modules: List[Dict[str, Any]]
    mastered_concepts: List[str]
    knowledge_gaps: List[Dict[str, Any]]
    learning_strengths: List[Dict[str, Any]]
    improvement_areas: List[Dict[str, Any]]
    time_investment: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    next_milestones: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "journey_id": self.journey_id,
            "learner_id": self.learner_id,
            "generated_at": self.generated_at,
            "reporting_period": self.reporting_period,
            "overall_progress": self.overall_progress,
            "domain_progress": self.domain_progress,
            "completed_modules": self.completed_modules,
            "mastered_concepts": self.mastered_concepts,
            "knowledge_gaps": self.knowledge_gaps,
            "learning_strengths": self.learning_strengths,
            "improvement_areas": self.improvement_areas,
            "time_investment": self.time_investment,
            "engagement_metrics": self.engagement_metrics,
            "recommendations": self.recommendations,
            "next_milestones": self.next_milestones,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningOutcomeReport':
        """Create from dictionary representation."""
        return cls(
            report_id=data["report_id"],
            journey_id=data["journey_id"],
            learner_id=data["learner_id"],
            generated_at=data["generated_at"],
            reporting_period=data["reporting_period"],
            overall_progress=data["overall_progress"],
            domain_progress=data["domain_progress"],
            completed_modules=data["completed_modules"],
            mastered_concepts=data["mastered_concepts"],
            knowledge_gaps=data["knowledge_gaps"],
            learning_strengths=data["learning_strengths"],
            improvement_areas=data["improvement_areas"],
            time_investment=data["time_investment"],
            engagement_metrics=data["engagement_metrics"],
            recommendations=data["recommendations"],
            next_milestones=data["next_milestones"],
            metadata=data.get("metadata", {})
        )


class EducationalAgentSpecialization(Enum):
    """Specializations for educational agents."""
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"
    TUTOR = "tutor"
    COACH = "coach"
    ASSESSMENT_SPECIALIST = "assessment_specialist"
    CONTENT_CREATOR = "content_creator"
    CURATOR = "curator"
    FACILITATOR = "facilitator"
    MOTIVATOR = "motivator"
    LEARNING_STRATEGIST = "learning_strategist"


class LearningJourneyOrchestrator(ProjectOrchestrator):
    """
    Advanced orchestrator for coordinating educational experiences across multiple knowledge domains.
    
    The LearningJourneyOrchestrator extends ProjectOrchestrator with specialized capabilities for
    managing personalized learning journeys, adapting learning paths based on feedback and progress,
    selecting appropriate educational agents for different subjects and learning styles, ensuring
    consistent knowledge building across sessions, and measuring learning outcomes.
    
    It embodies the "Growth as a Shared Journey" principle throughout the learning process.
    """
    
    def __init__(
        self,
        agent_id: str,
        orchestrator_engine: OrchestratorEngine,
        principle_engine: Optional[PrincipleEngine] = None,
        learning_system: Optional[LearningSystem] = None,
        communication_adapter: Optional[CommunicationAdapter] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        content_handler: Optional[ContentHandler] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        storage_dir: str = "data/learning_journey"
    ):
        """
        Initialize the learning journey orchestrator.
        
        Args:
            agent_id: ID of the agent
            orchestrator_engine: Engine for orchestration
            principle_engine: Engine for principle-based reasoning
            learning_system: System for learning
            communication_adapter: Adapter for communication
            relationship_tracker: Tracker for relationships
            content_handler: Handler for content
            emotional_intelligence: System for emotional intelligence
            storage_dir: Directory for storing journey data
        """
        # Initialize as ProjectOrchestrator
        super().__init__(
            agent_id=agent_id,
            orchestrator_engine=orchestrator_engine,
            principle_engine=principle_engine,
            communication_adapter=communication_adapter,
            relationship_tracker=relationship_tracker
        )
        
        # Learning journey specific components
        self.learning_system = learning_system
        self.content_handler = content_handler
        self.emotional_intelligence = emotional_intelligence
        
        # Storage directory
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Learning journey data
        self.learning_journeys: Dict[str, LearningJourney] = {}
        self.learner_profiles: Dict[str, LearnerProfile] = {}
        self.learning_domains: Dict[str, LearningDomain] = {}
        self.session_metrics: Dict[str, LearningSessionMetrics] = {}
        self.outcome_reports: Dict[str, LearningOutcomeReport] = {}
        
        # Educational agent profiles
        self.educational_agents: Dict[str, Dict[str, Any]] = {}
        
        # Active learning sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_data()
        
        logger.info(f"LearningJourneyOrchestrator initialized for agent {agent_id}")
    
    def _load_data(self) -> None:
        """Load learning journey data from storage."""
        try:
            # Load learning journeys
            journey_path = os.path.join(self.storage_dir, "learning_journeys.json")
            if os.path.exists(journey_path):
                with open(journey_path, "r") as f:
                    journeys_data = json.load(f)
                    for journey_id, journey_data in journeys_data.items():
                        self.learning_journeys[journey_id] = LearningJourney.from_dict(journey_data)
            
            # Load learner profiles
            profiles_path = os.path.join(self.storage_dir, "learner_profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, "r") as f:
                    profiles_data = json.load(f)
                    for learner_id, profile_data in profiles_data.items():
                        self.learner_profiles[learner_id] = LearnerProfile.from_dict(profile_data)
            
            # Load learning domains
            domains_path = os.path.join(self.storage_dir, "learning_domains.json")
            if os.path.exists(domains_path):
                with open(domains_path, "r") as f:
                    domains_data = json.load(f)
                    for domain_id, domain_data in domains_data.items():
                        self.learning_domains[domain_id] = LearningDomain.from_dict(domain_data)
            
            # Load session metrics
            metrics_path = os.path.join(self.storage_dir, "session_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics_data = json.load(f)
                    for session_id, session_data in metrics_data.items():
                        self.session_metrics[session_id] = LearningSessionMetrics.from_dict(session_data)
            
            # Load outcome reports
            reports_path = os.path.join(self.storage_dir, "outcome_reports.json")
            if os.path.exists(reports_path):
                with open(reports_path, "r") as f:
                    reports_data = json.load(f)
                    for report_id, report_data in reports_data.items():
                        self.outcome_reports[report_id] = LearningOutcomeReport.from_dict(report_data)
            
            # Load educational agent profiles
            agents_path = os.path.join(self.storage_dir, "educational_agents.json")
            if os.path.exists(agents_path):
                with open(agents_path, "r") as f:
                    self.educational_agents = json.load(f)
                        
            logger.info(f"Loaded learning journey data: {len(self.learning_journeys)} journeys, " +
                       f"{len(self.learner_profiles)} learner profiles, " +
                       f"{len(self.learning_domains)} learning domains")
                       
        except Exception as e:
            logger.error(f"Error loading learning journey data: {str(e)}")
    
    def _save_data(self) -> None:
        """Save learning journey data to storage."""
        try:
            # Save learning journeys
            journey_path = os.path.join(self.storage_dir, "learning_journeys.json")
            journeys_data = {
                journey_id: journey.to_dict()
                for journey_id, journey in self.learning_journeys.items()
            }
            with open(journey_path, "w") as f:
                json.dump(journeys_data, f, indent=2)
            
            # Save learner profiles
            profiles_path = os.path.join(self.storage_dir, "learner_profiles.json")
            profiles_data = {
                learner_id: profile.to_dict()
                for learner_id, profile in self.learner_profiles.items()
            }
            with open(profiles_path, "w") as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save learning domains
            domains_path = os.path.join(self.storage_dir, "learning_domains.json")
            domains_data = {
                domain_id: domain.to_dict()
                for domain_id, domain in self.learning_domains.items()
            }
            with open(domains_path, "w") as f:
                json.dump(domains_data, f, indent=2)
            
            # Save session metrics
            metrics_path = os.path.join(self.storage_dir, "session_metrics.json")
            metrics_data = {
                session_id: metrics.to_dict()
                for session_id, metrics in self.session_metrics.items()
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save outcome reports
            reports_path = os.path.join(self.storage_dir, "outcome_reports.json")
            reports_data = {
                report_id: report.to_dict()
                for report_id, report in self.outcome_reports.items()
            }
            with open(reports_path, "w") as f:
                json.dump(reports_data, f, indent=2)
            
            # Save educational agent profiles
            agents_path = os.path.join(self.storage_dir, "educational_agents.json")
            with open(agents_path, "w") as f:
                json.dump(self.educational_agents, f, indent=2)
                
            logger.info("Learning journey data saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving learning journey data: {str(e)}")
    
    def create_learner_profile(self, 
                              learner_id: str, 
                              name: str,
                              preferred_learning_styles: List[LearningStyle],
                              initial_competency_levels: Dict[str, CompetencyLevel],
                              interests: List[str],
                              motivation_factors: List[str],
                              previous_knowledge_domains: List[str] = None,
                              metadata: Dict[str, Any] = None) -> LearnerProfile:
        """
        Create a new learner profile.
        
        Args:
            learner_id: Unique identifier for the learner
            name: Name of the learner
            preferred_learning_styles: List of preferred learning styles
            initial_competency_levels: Initial competency levels for various domains
            interests: List of learner's interests
            motivation_factors: Factors that motivate the learner
            previous_knowledge_domains: Previously studied knowledge domains
            metadata: Additional metadata
            
        Returns:
            The created learner profile
        """
        if learner_id in self.learner_profiles:
            logger.warning(f"Learner profile {learner_id} already exists, returning existing profile")
            return self.learner_profiles[learner_id]
        
        previous_knowledge_domains = previous_knowledge_domains or []
        metadata = metadata or {}
        
        profile = LearnerProfile(
            learner_id=learner_id,
            name=name,
            preferred_learning_styles=preferred_learning_styles,
            current_competency_levels=initial_competency_levels,
            learning_speed={},  # Will be populated as learning progresses
            interests=interests,
            motivation_factors=motivation_factors,
            previous_knowledge_domains=previous_knowledge_domains,
            completed_modules=[],
            learning_history=[],
            current_goals=[],
            feedback_history=[],
            preference_settings={},
            metadata=metadata
        )
        
        self.learner_profiles[learner_id] = profile
        self._save_data()
        
        logger.info(f"Created learner profile for {name} (ID: {learner_id})")
        return profile
    
    def create_learning_journey(self,
                               name: str,
                               description: str,
                               learner_id: str,
                               domain_ids: List[str],
                               recommended_sequence: List[str] = None,
                               tags: List[str] = None,
                               metadata: Dict[str, Any] = None) -> LearningJourney:
        """
        Create a new learning journey for a learner.
        
        Args:
            name: Name of the learning journey
            description: Description of the learning journey
            learner_id: ID of the learner
            domain_ids: IDs of domains to include in the journey
            recommended_sequence: Recommended sequence of domains (defaults to domain_ids order)
            tags: Tags for the journey
            metadata: Additional metadata
            
        Returns:
            The created learning journey
        """
        # Validate learner exists
        if learner_id not in self.learner_profiles:
            raise ValueError(f"Learner profile {learner_id} does not exist")
        
        # Validate domains exist
        for domain_id in domain_ids:
            if domain_id not in self.learning_domains:
                raise ValueError(f"Learning domain {domain_id} does not exist")
        
        recommended_sequence = recommended_sequence or domain_ids
        tags = tags or []
        metadata = metadata or {}
        
        journey_id = f"journey-{uuid.uuid4()}"
        now = datetime.utcnow().isoformat()
        
        # Get the domains
        domains = {
            domain_id: self.learning_domains[domain_id]
            for domain_id in domain_ids
        }
        
        journey = LearningJourney(
            journey_id=journey_id,
            name=name,
            description=description,
            learner_id=learner_id,
            domains=domains,
            recommended_sequence=recommended_sequence,
            created_at=now,
            updated_at=now,
            start_date=now,
            end_date=None,
            status="active",
            overall_progress=0.0,
            tags=tags,
            metadata=metadata
        )
        
        self.learning_journeys[journey_id] = journey
        self._save_data()
        
        logger.info(f"Created learning journey '{name}' (ID: {journey_id}) for learner {learner_id}")
        return journey
    
    def start_learning_session(self, 
                              journey_id: str, 
                              domain_id: str = None,
                              module_id: str = None,
                              activity_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new learning session within a journey.
        
        Args:
            journey_id: ID of the learning journey
            domain_id: ID of the domain to focus on (optional)
            module_id: ID of the module to focus on (optional)
            activity_id: ID of the activity to focus on (optional)
            
        Returns:
            Tuple of (session_id, session_context)
        """
        if journey_id not in self.learning_journeys:
            raise ValueError(f"Learning journey {journey_id} does not exist")
        
        journey = self.learning_journeys[journey_id]
        learner_id = journey.learner_id
        
        # If domain not specified, use the first domain in recommended sequence
        if domain_id is None and journey.recommended_sequence:
            domain_id = journey.recommended_sequence[0]
        
        # If domain specified, validate it exists in the journey
        if domain_id and domain_id not in journey.domains:
            raise ValueError(f"Domain {domain_id} does not exist in journey {journey_id}")
        
        # Create session ID and start time
        session_id = f"session-{uuid.uuid4()}"
        start_time = datetime.utcnow().isoformat()
        
        # Create session metrics
        metrics = LearningSessionMetrics(
            session_id=session_id,
            learner_id=learner_id,
            start_time=start_time
        )
        self.session_metrics[session_id] = metrics
        
        # Create session context
        session_context = {
            "session_id": session_id,
            "journey_id": journey_id,
            "learner_id": learner_id,
            "start_time": start_time,
            "learner_profile": self.learner_profiles[learner_id].to_dict(),
            "active_domain_id": domain_id,
            "active_module_id": module_id,
            "active_activity_id": activity_id,
            "navigation_history": []
        }
        
        # Update active sessions
        self.active_sessions[session_id] = session_context
        
        # Update journey's active session
        journey.active_session = {
            "session_id": session_id,
            "start_time": start_time,
            "domain_id": domain_id,
            "module_id": module_id,
            "activity_id": activity_id
        }
        
        # Save the updated data
        self._save_data()
        
        logger.info(f"Started learning session {session_id} for journey {journey_id}")
        return session_id, session_context
    
    def end_learning_session(self, session_id: str) -> LearningSessionMetrics:
        """
        End a learning session and finalize metrics.
        
        Args:
            session_id: ID of the session to end
            
        Returns:
            The finalized session metrics
        """
        if session_id not in self.session_metrics:
            raise ValueError(f"Learning session {session_id} does not exist")
        
        # Get session metrics and context
        metrics = self.session_metrics[session_id]
        context = self.active_sessions.get(session_id)
        
        if not context:
            logger.warning(f"Session context for {session_id} not found")
        else:
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Update journey's active session
            journey_id = context.get("journey_id")
            if journey_id in self.learning_journeys:
                journey = self.learning_journeys[journey_id]
                if journey.active_session and journey.active_session.get("session_id") == session_id:
                    journey.active_session = None
        
        # Update metrics
        end_time = datetime.utcnow().isoformat()
        metrics.end_time = end_time
        
        # Calculate duration
        start_dt = datetime.fromisoformat(metrics.start_time)
        end_dt = datetime.fromisoformat(end_time)
        metrics.duration = int((end_dt - start_dt).total_seconds())
        
        # Save the updated data
        self._save_data()
        
        logger.info(f"Ended learning session {session_id}, duration: {metrics.duration} seconds")
        return metrics
    
    def update_learning_progress(self, 
                                journey_id: str, 
                                domain_id: str = None,
                                module_id: str = None,
                                activity_id: str = None,
                                completion_percentage: float = None) -> float:
        """
        Update learning progress for a specific component of a journey.
        
        Args:
            journey_id: ID of the learning journey
            domain_id: ID of the domain to update (optional)
            module_id: ID of the module to update (optional)
            activity_id: ID of the activity to update (optional)
            completion_percentage: New completion percentage (optional)
            
        Returns:
            The updated overall journey progress
        """
        if journey_id not in self.learning_journeys:
            raise ValueError(f"Learning journey {journey_id} does not exist")
        
        journey = self.learning_journeys[journey_id]
        
        # Update activity progress if specified
        if domain_id and module_id and activity_id and completion_percentage is not None:
            if domain_id not in journey.domains:
                raise ValueError(f"Domain {domain_id} does not exist in journey {journey_id}")
                
            domain = journey.domains[domain_id]
            
            if module_id not in domain.modules:
                raise ValueError(f"Module {module_id} does not exist in domain {domain_id}")
                
            module = domain.modules[module_id]
            
            if activity_id not in module.activities:
                raise ValueError(f"Activity {activity_id} does not exist in module {module_id}")
                
            activity = module.activities[activity_id]
            
            # Update activity progress
            activity.completion_percentage = min(1.0, max(0.0, completion_percentage))
            
            # Update activity status based on progress
            if activity.completion_percentage >= 1.0:
                activity.status = LearningActivityStatus.COMPLETED
            elif activity.completion_percentage > 0:
                activity.status = LearningActivityStatus.IN_PROGRESS
            
            # Update module progress
            module.update_progress()
            
            # Update domain progress
            domain.update_progress()
        
        # Update journey progress
        journey.update_progress()
        journey.updated_at = datetime.utcnow().isoformat()
        
        # Save the updated data
        self._save_data()
        
        logger.info(f"Updated learning progress for journey {journey_id}: {journey.overall_progress:.2%}")
        return journey.overall_progress
    
    def generate_learning_outcome_report(self, 
                                        journey_id: str,
                                        reporting_period: Dict[str, str] = None) -> LearningOutcomeReport:
        """
        Generate a comprehensive learning outcome report for a journey.
        
        Args:
            journey_id: ID of the learning journey
            reporting_period: Start and end dates for the report
            
        Returns:
            The generated learning outcome report
        """
        if journey_id not in self.learning_journeys:
            raise ValueError(f"Learning journey {journey_id} does not exist")
        
        journey = self.learning_journeys[journey_id]
        learner_id = journey.learner_id
        
        # Default reporting period is from journey start to now
        if reporting_period is None:
            reporting_period = {
                "start_date": journey.start_date,
                "end_date": datetime.utcnow().isoformat()
            }
        
        # Generate report ID
        report_id = f"report-{uuid.uuid4()}"
        generated_at = datetime.utcnow().isoformat()
        
        # Calculate domain progress
        domain_progress = {
            domain_id: domain.completion_percentage
            for domain_id, domain in journey.domains.items()
        }
        
        # Collect completed modules
        completed_modules = []
        for domain_id, domain in journey.domains.items():
            for module_id, module in domain.modules.items():
                if module.status in [LearningModuleStatus.COMPLETED, LearningModuleStatus.MASTERED]:
                    completed_modules.append({
                        "domain_id": domain_id,
                        "module_id": module_id,
                        "name": module.name,
                        "completion_date": journey.updated_at
                    })
        
        # Collect mastered concepts
        mastered_concepts = []
        
        # Identify knowledge gaps
        knowledge_gaps = []
        
        # Identify learning strengths
        learning_strengths = []
        
        # Calculate time investment
        time_investment = {
            "total_minutes": 0,
            "by_domain": {},
            "by_module": {}
        }
        
        # Gather engagement metrics
        engagement_metrics = {
            "average_session_duration": 0,
            "session_count": 0,
            "completion_rate": 0
        }
        
        # Generate recommendations
        recommendations = []
        
        # Identify next milestones
        next_milestones = []
        
        # Create the report
        report = LearningOutcomeReport(
            report_id=report_id,
            journey_id=journey_id,
            learner_id=learner_id,
            generated_at=generated_at,
            reporting_period=reporting_period,
            overall_progress=journey.overall_progress,
            domain_progress=domain_progress,
            completed_modules=completed_modules,
            mastered_concepts=mastered_concepts,
            knowledge_gaps=knowledge_gaps,
            learning_strengths=learning_strengths,
            improvement_areas=[],
            time_investment=time_investment,
            engagement_metrics=engagement_metrics,
            recommendations=recommendations,
            next_milestones=next_milestones
        )
        
        # Save the report
        self.outcome_reports[report_id] = report
        self._save_data()
        
        logger.info(f"Generated learning outcome report {report_id} for journey {journey_id}")
        return report
