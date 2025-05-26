#!/usr/bin/env python3
"""
Collaborative Growth Module

This module provides functionality to identify opportunities for collaborative learning
and growth between agents, facilitating mutual knowledge exchange and problem-solving.
"""

import logging
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from collections import defaultdict

from learning_system import LearningSystem, LearningDimension, OutcomeType
from feedback_integration_system import FeedbackIntegrationSystem
from relationship_tracker import RelationshipTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CollaborativeGrowth")

class GrowthOpportunityType(Enum):
    """Types of collaborative growth opportunities."""
    KNOWLEDGE_SHARING = auto()  # Sharing known information with others
    INFORMATION_SEEKING = auto()  # Jointly seeking unknown information
    PROBLEM_SOLVING = auto()  # Collaboratively solving a challenge
    SKILL_DEVELOPMENT = auto()  # Developing new capabilities together
    FEEDBACK_EXCHANGE = auto()  # Structured feedback exchange
    JOINT_EXPLORATION = auto()  # Exploring new domains together

@dataclass
class KnowledgeGap:
    """Represents an identified knowledge gap."""
    id: str  # Unique identifier for the gap
    topic: str  # The topic of the knowledge gap
    agent_id: str  # ID of the agent with the gap
    confidence: float  # Confidence in the gap assessment (0.0-1.0)
    severity: float  # How significant the gap is (0.0-1.0)
    context: Dict[str, Any]  # Contextual information about the gap
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "topic": self.topic,
            "agent_id": self.agent_id,
            "confidence": self.confidence,
            "severity": self.severity,
            "context": self.context,
            "timestamp": self.timestamp
        }

@dataclass
class CollaborativeOpportunity:
    """Represents a collaborative growth opportunity."""
    id: str  # Unique identifier for the opportunity
    type: GrowthOpportunityType  # Type of opportunity
    title: str  # Short descriptive title
    description: str  # Detailed description
    participants: List[str]  # IDs of potential participants
    knowledge_gaps: List[KnowledgeGap]  # Related knowledge gaps
    learning_dimensions: List[LearningDimension]  # Learning dimensions involved
    expected_outcomes: List[str]  # Expected outcomes of collaboration
    proposed_activities: List[str]  # Proposed collaborative activities
    resources_needed: Dict[str, Any]  # Resources needed for collaboration
    evaluation_criteria: List[str]  # Criteria to evaluate success
    priority: float  # Priority score (0.0-1.0)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.name,
            "title": self.title,
            "description": self.description,
            "participants": self.participants,
            "knowledge_gaps": [gap.to_dict() for gap in self.knowledge_gaps],
            "learning_dimensions": [dim.name for dim in self.learning_dimensions],
            "expected_outcomes": self.expected_outcomes,
            "proposed_activities": self.proposed_activities,
            "resources_needed": self.resources_needed,
            "evaluation_criteria": self.evaluation_criteria,
            "priority": self.priority,
            "timestamp": self.timestamp
        }

def identify_collaborative_growth_opportunity(
    interaction_history: List[Dict[str, Any]],
    learning_system: Optional[LearningSystem] = None,
    feedback_system: Optional[FeedbackIntegrationSystem] = None,
    relationship_tracker: Optional[RelationshipTracker] = None,
    agent_id: str = "self",
    participant_ids: Optional[List[str]] = None,
    focus_dimensions: Optional[List[LearningDimension]] = None
) -> List[CollaborativeOpportunity]:
    """
    Analyze interactions to identify opportunities for collaborative learning and growth.
    
    Args:
        interaction_history: History of interactions to analyze
        learning_system: Optional learning system for additional context
        feedback_system: Optional feedback system for additional context
        relationship_tracker: Optional relationship tracker for participant context
        agent_id: ID of the agent seeking opportunities (default: "self")
        participant_ids: Optional specific participant IDs to focus on
        focus_dimensions: Optional specific learning dimensions to focus on
        
    Returns:
        List of identified collaborative growth opportunities
    """
    # Initialize results list
    opportunities = []
    
    # Step 1: Identify knowledge gaps in the agent and other participants
    agent_gaps = _identify_knowledge_gaps(interaction_history, agent_id, learning_system)
    participant_gaps = {}
    
    if participant_ids:
        for participant_id in participant_ids:
            participant_gaps[participant_id] = _identify_knowledge_gaps(
                interaction_history, participant_id, learning_system
            )
    else:
        # Extract unique participant IDs from interaction history
        unique_participants = set()
        for interaction in interaction_history:
            if "sender" in interaction and interaction["sender"] != agent_id:
                unique_participants.add(interaction["sender"])
            if "receiver" in interaction and interaction["receiver"] != agent_id:
                unique_participants.add(interaction["receiver"])
        
        for participant_id in unique_participants:
            participant_gaps[participant_id] = _identify_knowledge_gaps(
                interaction_history, participant_id, learning_system
            )
    
    # Step 2: Analyze learning patterns from learning system if available
    learning_patterns = []
    if learning_system:
        # Get metrics and successful patterns
        metrics = learning_system.get_learning_metrics()
        
        # Get growth journal entries for insights
        journal_entries = learning_system.get_growth_journal(
            entry_types=["reflection", "milestone"],
            limit=20
        )
        
        # Find patterns with high success rates in focus dimensions
        if focus_dimensions:
            for dimension in focus_dimensions:
                dimension_name = dimension.name
                if dimension_name in learning_system.pattern_contexts:
                    successful_patterns = [
                        pattern for pattern_id in learning_system.pattern_contexts[dimension_name]
                        if pattern_id in learning_system.interaction_patterns
                        and learning_system.interaction_patterns[pattern_id].success_rate >= 0.7
                    ]
                    learning_patterns.extend(successful_patterns)
    
    # Step 3: Gather feedback insights if feedback system is available
    feedback_insights = []
    if feedback_system:
        # This would be implemented based on the feedback system's API
        # For now, just add a placeholder
        feedback_insights = []
    
    # Step 4: Analyze relationship context if available
    relationship_contexts = {}
    if relationship_tracker and participant_ids:
        for participant_id in participant_ids:
            # This would be implemented based on the relationship tracker's API
            # For now, just add placeholder data
            relationship_contexts[participant_id] = {
                "trust_level": 0.7,
                "interaction_frequency": "medium",
                "interaction_quality": "positive",
                "collaborative_history": []
            }
    
    # Step 5: Generate collaborative opportunities based on gathered data
    
    # 5.1: Knowledge sharing opportunities based on complementary gaps
    knowledge_sharing_opps = _generate_knowledge_sharing_opportunities(
        agent_gaps, participant_gaps, agent_id
    )
    opportunities.extend(knowledge_sharing_opps)
    
    # 5.2: Joint information seeking opportunities based on common gaps
    info_seeking_opps = _generate_information_seeking_opportunities(
        agent_gaps, participant_gaps, agent_id
    )
    opportunities.extend(info_seeking_opps)
    
    # 5.3: Collaborative problem-solving opportunities
    problem_solving_opps = _generate_problem_solving_opportunities(
        interaction_history, agent_gaps, participant_gaps, agent_id
    )
    opportunities.extend(problem_solving_opps)
    
    # 5.4: Skill development opportunities
    if learning_system:
        skill_dev_opps = _generate_skill_development_opportunities(
            learning_system, agent_gaps, participant_gaps, agent_id
        )
        opportunities.extend(skill_dev_opps)
    
    # 5.5: Feedback exchange opportunities
    if feedback_system:
        feedback_opps = _generate_feedback_opportunities(
            feedback_system, agent_id, participant_ids
        )
        opportunities.extend(feedback_opps)
    
    # Step 6: Prioritize opportunities
    prioritized_opportunities = _prioritize_opportunities(
        opportunities, agent_id, relationship_contexts
    )
    
    # Return the prioritized opportunities
    return prioritized_opportunities

def _identify_knowledge_gaps(
    interaction_history: List[Dict[str, Any]],
    agent_id: str,
    learning_system: Optional[LearningSystem] = None
) -> List[KnowledgeGap]:
    """
    Identify knowledge gaps for a specific agent based on interaction history.
    
    Args:
        interaction_history: History of interactions to analyze
        agent_id: ID of the agent to analyze
        learning_system: Optional learning system for additional context
        
    Returns:
        List of identified knowledge gaps
    """
    knowledge_gaps = []
    
    # Keywords indicating knowledge gaps
    uncertainty_indicators = [
        "I don't know", "I'm not sure", "I'm uncertain", 
        "I'm not familiar with", "I haven't heard of",
        "I don't understand", "I'm unclear about",
        "What is", "How does", "Can you explain"
    ]
    
    # Extract messages from the agent
    agent_messages = []
    for interaction in interaction_history:
        if interaction.get("sender") == agent_id and "content" in interaction:
            agent_messages.append(interaction["content"])
    
    # Analyze messages for uncertainty indicators
    topic_uncertainties = defaultdict(list)
    for message in agent_messages:
        for indicator in uncertainty_indicators:
            if indicator.lower() in message.lower():
                # Try to extract the topic of uncertainty
                # This is a simplified approach - a real implementation would use NLP
                topic = _extract_topic(message, indicator)
                if topic:
                    topic_uncertainties[topic].append(message)
    
    # Create knowledge gaps based on identified topics
    for topic, messages in topic_uncertainties.items():
        confidence = min(0.9, 0.5 + (len(messages) * 0.1))  # More mentions = higher confidence
        severity = 0.5  # Default severity - could be refined based on context
        
        knowledge_gap = KnowledgeGap(
            id=f"gap_{hash(topic) % 10000}_{agent_id}",
            topic=topic,
            agent_id=agent_id,
            confidence=confidence,
            severity=severity,
            context={
                "message_count": len(messages),
                "example_message": messages[0] if messages else "",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        knowledge_gaps.append(knowledge_gap)
    
    # If learning system is available, analyze for additional gaps
    if learning_system:
        # Look for unsuccessful patterns that might indicate knowledge gaps
        unsuccessful_patterns = []
        for pattern in learning_system.interaction_patterns.values():
            if pattern.success_rate < 0.4 and pattern.confidence > 0.5:
                # This could indicate a knowledge gap
                unsuccessful_patterns.append(pattern)
        
        for pattern in unsuccessful_patterns:
            # Extract possible topic from pattern description
            topic = pattern.description
            
            # Check if this topic is already covered
            if not any(gap.topic == topic for gap in knowledge_gaps):
                knowledge_gap = KnowledgeGap(
                    id=f"gap_{hash(topic) % 10000}_{agent_id}_pattern",
                    topic=topic,
                    agent_id=agent_id,
                    confidence=pattern.confidence,
                    severity=1.0 - pattern.success_rate,
                    context={
                        "pattern_id": pattern.pattern_id,
                        "success_rate": pattern.success_rate,
                        "occurrences": pattern.occurrences,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                knowledge_gaps.append(knowledge_gap)
    
    return knowledge_gaps

def _extract_topic(message: str, indicator: str) -> Optional[str]:
    """
    Extract the topic of uncertainty from a message.
    
    Args:
        message: The message to extract from
        indicator: The uncertainty indicator found
        
    Returns:
        Extracted topic or None if no clear topic
    """
    # Find where the indicator appears in the message
    start_idx = message.lower().find(indicator.lower())
    if start_idx == -1:
        return None
    
    # Look at the text after the indicator
    after_text = message[start_idx + len(indicator):].strip()
    
    # Extract the first sentence or phrase
    end_markers = ['.', '?', '!', ',', ';']
    end_idx = len(after_text)
    for marker in end_markers:
        marker_idx = after_text.find(marker)
        if marker_idx != -1 and marker_idx < end_idx:
            end_idx = marker_idx
    
    topic = after_text[:end_idx].strip()
    
    # If topic is too long, truncate it
    if len(topic) > 50:
        topic = topic[:47] + "..."
    
    # If topic is too short or empty, try to use context before the indicator
    if len(topic) < 3:
        before_text = message[:start_idx].strip()
        words = before_text.split()
        if words:
            topic = " ".join(words[-min(3, len(words)):])
    
    return topic if topic else None

def _generate_knowledge_sharing_opportunities(
    agent_gaps: List[KnowledgeGap],
    participant_gaps: Dict[str, List[KnowledgeGap]],
    agent_id: str
) -> List[CollaborativeOpportunity]:
    """
    Generate knowledge sharing opportunities based on complementary knowledge gaps.
    
    Args:
        agent_gaps: Knowledge gaps of the agent
        participant_gaps: Knowledge gaps of participants
        agent_id: ID of the agent
        
    Returns:
        List of knowledge sharing opportunities
    """
    opportunities = []
    
    # Find topics where agent has knowledge and participant has a gap
    agent_gap_topics = {gap.topic for gap in agent_gaps}
    
    for participant_id, gaps in participant_gaps.items():
        participant_gap_topics = {gap.topic for gap in gaps}
        
        # Look for topics the agent knows but participant doesn't
        for participant_gap in gaps:
            if participant_gap.topic not in agent_gap_topics:
                # Agent might have knowledge to share on this topic
                
                # Create the opportunity
                opportunity = CollaborativeOpportunity(
                    id=f"opp_share_{hash(participant_gap.topic) % 10000}_{participant_id}",
                    type=GrowthOpportunityType.KNOWLEDGE_SHARING,
                    title=f"Share knowledge about {participant_gap.topic}",
                    description=f"Share knowledge with {participant_id} about {participant_gap.topic}.",
                    participants=[agent_id, participant_id],
                    knowledge_gaps=[participant_gap],
                    learning_dimensions=[LearningDimension.COMMUNICATION_EFFECTIVENESS],
                    expected_outcomes=[
                        f"Participant {participant_id} gains knowledge about {participant_gap.topic}",
                        "Strengthen collaborative relationship"
                    ],
                    proposed_activities=[
                        "Prepare concise explanation or resources",
                        "Schedule knowledge sharing session",
                        "Follow up to ensure understanding"
                    ],
                    resources_needed={
                        "time": "15-30 minutes",
                        "materials": "Relevant documentation or resources"
                    },
                    evaluation_criteria=[
                        "Participant's demonstrated understanding",
                        "Participant's ability to apply knowledge",
                        "Improvement in related interactions"
                    ],
                    priority=0.5 * participant_gap.confidence * participant_gap.severity
                )
                
                opportunities.append(opportunity)
    
    # Find topics where participant has knowledge and agent has a gap
    for agent_gap in agent_gaps:
        for participant_id, gaps in participant_gaps.items():
            participant_gap_topics = {gap.topic for gap in gaps}
            
            if agent_gap.topic not in participant_gap_topics:
                # Participant might have knowledge to share on this topic
                
                # Create the opportunity
                opportunity = CollaborativeOpportunity(
                    id=f"opp_learn_{hash(agent_gap.topic) % 10000}_{participant_id}",
                    type=GrowthOpportunityType.KNOWLEDGE_SHARING,
                    title=f"Learn about {agent_gap.topic}",
                    description=f"Request knowledge from {participant_id} about {agent_gap.topic}.",
                    participants=[agent_id, participant_id],
                    knowledge_gaps=[agent_gap],
                    learning_dimensions=[LearningDimension.COMMUNICATION_EFFECTIVENESS],
                    expected_outcomes=[
                        f"Agent gains knowledge about {agent_gap.topic}",
                        "Strengthen collaborative relationship"
                    ],
                    proposed_activities=[
                        "Prepare specific questions",
                        "Request knowledge sharing session",
                        "Take notes and ask follow-up questions",
                        "Apply knowledge in relevant context"
                    ],
                    resources_needed={
                        "time": "15-30 minutes",
                        "materials": "Note-taking tools"
                    },
                    evaluation_criteria=[
                        "Improved understanding of the topic",
                        "Ability to apply new knowledge",
                        "Reduction in related uncertainties"
                    ],
                    priority=0.7 * agent_gap.confidence * agent_gap.severity
                )
                
                opportunities.append(opportunity)
    
    return opportunities

def _generate_information_seeking_opportunities(
    agent_gaps: List[KnowledgeGap],
    participant_gaps: Dict[str, List[KnowledgeGap]],
    agent_id: str
) -> List[CollaborativeOpportunity]:
    """
    Generate information seeking opportunities based on common knowledge gaps.
    
    Args:
        agent_gaps: Knowledge gaps of the agent
        participant_gaps: Knowledge gaps of participants
        agent_id: ID of the agent
        
    Returns:
        List of information seeking opportunities
    """
    opportunities = []
    
    # Find topics where both agent and participant have gaps
    agent_gap_topics = {gap.topic: gap for gap in agent_gaps}
    
    for participant_id, gaps in participant_gaps.items():
        participant_gap_topics = {gap.topic: gap for gap in gaps}
        
        # Find common gap topics
        common_topics = set(agent_gap_topics.keys()) & set(participant_gap_topics.keys())
        
        for topic in common_topics:
            agent_gap = agent_gap_topics[topic]
            participant_gap = participant_gap_topics[topic]
            
            # Create the opportunity
            opportunity = CollaborativeOpportunity(
                id=f"opp_seek_{hash(topic) % 10000}_{participant_id}",
                type=GrowthOpportunityType.INFORMATION_SEEKING,
                title=f"Jointly explore {topic}",
                description=f"Collaborate with {participant_id} to learn about {topic}, which is a knowledge gap for both of you.",
                participants=[agent_id, participant_id],
                knowledge_gaps=[agent_gap, participant_gap],
                learning_dimensions=[
                    LearningDimension.COMMUNICATION_EFFECTIVENESS,
                    LearningDimension.TASK_COLLABORATION
                ],
                expected_outcomes=[
                    f"Both gain knowledge about {topic}",
                    "Develop collaborative research skills",
                    "Strengthen working relationship"
                ],
                proposed_activities=[
                    "Define specific learning objectives",
                    "Divide research tasks",
                    "Share and synthesize findings",
                    "Document new knowledge"
                ],
                resources_needed={
                    "time": "1-3 hours",
                    "materials": "Research resources, shared document"
                },
                evaluation_criteria=[
                    "Quality and depth of acquired knowledge",
                    "Efficiency of collaborative process",
                    "Mutual satisfaction with outcomes"
                ],
                priority=0.8 * (agent_gap.confidence + participant_gap.confidence) * 
                         (agent_gap.severity + participant_gap.severity) / 4
            )
            
            opportunities.append(opportunity)
    
    return opportunities

def _generate_problem_solving_opportunities(
    interaction_history: List[Dict[str, Any]],
    agent_gaps: List[KnowledgeGap],
    participant_gaps: Dict[str, List[KnowledgeGap]],
    agent_id: str
) -> List[CollaborativeOpportunity]:
    """
    Generate collaborative problem-solving opportunities.
    
    Args:
        interaction_history: History of interactions
        agent_gaps: Knowledge gaps of the agent
        participant_gaps: Knowledge gaps of participants
        agent_id: ID of the agent
        
    Returns:
        List of problem-solving opportunities
    """
    opportunities = []
    
    # Extract problems/challenges from interaction history
    challenges = _extract_challenges(interaction_history)
    
    for challenge in challenges:
        # Find participants who might be good collaborators for this challenge
        potential_collaborators = []
        
        for participant_id, gaps in participant_gaps.items():
            # Check if participant has knowledge in related areas
            if not any(challenge["topic"] in gap.topic for gap in gaps):
                # Participant might have knowledge in this area
                potential_collaborators.append(participant_id)
        
        if potential_collaborators:
            # Create the opportunity with first potential collaborator
            participant_id = potential_collaborators[0]
            
            opportunity = CollaborativeOpportunity(
                id=f"opp_solve_{hash(challenge['topic']) % 10000}",
                type=GrowthOpportunityType.PROBLEM_SOLVING,
                title=f"Solve {challenge['topic']} together",
                description=f"Collaborate with {participant_id} to address: {challenge['description']}",
                participants=[agent_id, participant_id],
                knowledge_gaps=[],  # No specific gaps associated
                learning_dimensions=[
                    LearningDimension.TASK_COLLABORATION,
                    LearningDimension.ADAPTABILITY
                ],
                expected_outcomes=[
                    f"Develop solution for {challenge['topic']}",
                    "Improve collaborative problem-solving skills",
                    "Apply and test knowledge in practical scenario"
                ],
                proposed_activities=[
                    "Define problem scope and objectives",
                    "Brainstorm potential approaches",
                    "Develop and evaluate solution",
                    "Document process and outcomes"
                ],
                resources_needed={
                    "time": "2-4 hours",
                    "materials": "Problem-specific resources"
                },
                evaluation_criteria=[
                    "Quality and effectiveness of solution",
                    "Efficiency of collaborative process",
                    "New insights gained"
                ],
                priority=0.6 * challenge["severity"]
            )
            
            opportunities.append(opportunity)
    
    return opportunities

def _extract_challenges(interaction_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract challenges/problems from interaction history.
    
    Args:
        interaction_history: History of interactions
        
    Returns:
        List of identified challenges
    """
    challenges = []
    
    # Keywords indicating challenges
    challenge_indicators = [
        "problem", "challenge", "difficult", "struggle",
        "issue", "trouble", "can't figure out", "stuck",
        "need help with", "don't know how to"
    ]
    
    # Extract messages containing challenge indicators
    for interaction in interaction_history:
        if "content" in interaction:
            message = interaction["content"]
            for indicator in challenge_indicators:
                if indicator.lower() in message.lower():
                    # Extract challenge topic
                    topic = _extract_topic(message, indicator)
                    if topic:
                        # Check if this challenge is already in the list
                        if not any(c["topic"] == topic for c in challenges):
                            challenge = {
                                "topic": topic,
                                "description": message,
                                "severity": 0.7,  # Default severity
                                "timestamp": interaction.get("timestamp", datetime.now(timezone.utc).isoformat())
                            }
                            challenges.append(challenge)
    
    return challenges

def _generate_skill_development_opportunities(
    learning_system: LearningSystem,
    agent_gaps: List[KnowledgeGap],
    participant_gaps: Dict[str, List[KnowledgeGap]],
    agent_id: str
) -> List[CollaborativeOpportunity]:
    """
    Generate skill development opportunities.
    
    Args:
        learning_system: Learning system for pattern analysis
        agent_gaps: Knowledge gaps of the agent
        participant_gaps: Knowledge gaps of participants
        agent_id: ID of the agent
        
    Returns:
        List of skill development opportunities
    """
    opportunities = []
    
    # Get metrics to identify areas for improvement
    metrics = learning_system.get_learning_metrics()
    
    # Identify dimensions with lowest success rates
    dimension_success_rates = metrics.get("dimension_success_rates", {})
    sorted_dimensions = sorted(dimension_success_rates.items(), key=lambda x: x[1])
    
    # Focus on the lowest performing dimensions
    for dimension_name, success_rate in sorted_dimensions[:2]:
        # Find participants who might be strong in this dimension
        for participant_id, gaps in participant_gaps.items():
            # Assumption: fewer gaps in a dimension indicates strength
            dimension_gaps = [gap for gap in gaps if dimension_name.lower() in gap.topic.lower()]
            
            if len(dimension_gaps) <= 1:
                # Participant might be strong in this area
                
                # Create the opportunity
                dim_display_name = dimension_name.replace("_", " ").lower()
                
                opportunity = CollaborativeOpportunity(
                    id=f"opp_skill_{hash(dimension_name) % 10000}_{participant_id}",
                    type=GrowthOpportunityType.SKILL_DEVELOPMENT,
                    title=f"Develop {dim_display_name} skills",
                    description=f"Collaborate with {participant_id} to improve {dim_display_name} capabilities.",
                    participants=[agent_id, participant_id],
                    knowledge_gaps=[],  # No specific gaps associated
                    learning_dimensions=[LearningDimension[dimension_name]],
                    expected_outcomes=[
                        f"Improved performance in {dim_display_name}",
                        "Develop specific techniques and practices",
                        "Build mentoring relationship"
                    ],
                    proposed_activities=[
                        f"Observe {participant_id}'s approach",
                        "Practice specific techniques",
                        "Request feedback on performance",
                        "Develop improvement plan"
                    ],
                    resources_needed={
                        "time": "Ongoing, multiple sessions",
                        "materials": "Practice scenarios"
                    },
                    evaluation_criteria=[
                        f"Improvement in {dim_display_name} metrics",
                        "Successful application in real situations",
                        "Growth in confidence and capability"
                    ],
                    priority=0.9 * (1.0 - success_rate)  # Higher priority for lower success rates
                )
                
                opportunities.append(opportunity)
                break  # Just one participant per dimension
    
    return opportunities

def _generate_feedback_opportunities(
    feedback_system: FeedbackIntegrationSystem,
    agent_id: str,
    participant_ids: Optional[List[str]]
) -> List[CollaborativeOpportunity]:
    """
    Generate feedback exchange opportunities.
    
    Args:
        feedback_system: Feedback integration system
        agent_id: ID of the agent
        participant_ids: IDs of potential participants
        
    Returns:
        List of feedback exchange opportunities
    """
    opportunities = []
    
    # This would be implemented based on the feedback system's API
    # For now, just create a generic feedback opportunity with the first participant
    
    if participant_ids:
        participant_id = participant_ids[0]
        
        opportunity = CollaborativeOpportunity(
            id=f"opp_feedback_{participant_id}",
            type=GrowthOpportunityType.FEEDBACK_EXCHANGE,
            title="Structured feedback exchange",
            description=f"Engage in a structured feedback exchange with {participant_id} to identify growth areas.",
            participants=[agent_id, participant_id],
            knowledge_gaps=[],  # No specific gaps associated
            learning_dimensions=[
                LearningDimension.COMMUNICATION_EFFECTIVENESS,
                LearningDimension.ADAPTABILITY
            ],
            expected_outcomes=[
                "Increased self-awareness",
                "Specific improvement opportunities identified",
                "Strengthened trust and openness"
            ],
            proposed_activities=[
                "Prepare feedback preparation questionnaire",
                "Schedule feedback exchange session",
                "Follow structured feedback protocol",
                "Develop action plans based on insights"
            ],
            resources_needed={
                "time": "1-2 hours",
                "materials": "Feedback forms or prompts"
            },
            evaluation_criteria=[
                "Quality and specificity of feedback",
                "Actionability of insights",
                "Mutual satisfaction with process"
            ],
            priority=0.6  # Medium priority
        )
        
        opportunities.append(opportunity)
    
    return opportunities

def _prioritize_opportunities(
    opportunities: List[CollaborativeOpportunity],
    agent_id: str,
    relationship_contexts: Dict[str, Dict[str, Any]]
) -> List[CollaborativeOpportunity]:
    """
    Prioritize collaborative opportunities based on various factors.
    
    Args:
        opportunities: List of opportunities to prioritize
        agent_id: ID of the agent
        relationship_contexts: Relationship context for each participant
        
    Returns:
        Prioritized list of opportunities
    """
    # Apply additional prioritization factors
    for opportunity in opportunities:
        priority = opportunity.priority
        
        # Adjust based on relationship context if available
        for participant_id in opportunity.participants:
            if participant_id != agent_id and participant_id in relationship_contexts:
                context = relationship_contexts[participant_id]
                
                # Higher trust level increases priority
                trust_level = context.get("trust_level", 0.5)
                priority *= (0.5 + trust_level * 0.5)  # Scale from 0.5-1.0
                
                # Interaction frequency impacts priority
                frequency = context.get("interaction_frequency", "medium")
                if frequency == "high":
                    priority *= 1.2  # Boost for frequent collaborators