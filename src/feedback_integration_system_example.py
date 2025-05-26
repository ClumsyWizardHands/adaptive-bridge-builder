"""
Feedback Integration System Example

This example demonstrates the usage of the FeedbackIntegrationSystem to solicit,
process, and integrate human feedback into the agent's orchestration processes,
applying the "Fairness as a Fundamental Truth" principle.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from feedback_integration_system import (
    FeedbackIntegrationSystem,
    FeedbackItem,
    FeedbackType,
    FeedbackSource,
    FeedbackUrgency,
    FeedbackFormat,
    FeedbackStatus,
    FeedbackSolicitationTemplate,
    FeedbackSolicitationCampaign,
    StakeholderProfile
)
from orchestration_analytics import OrchestrationAnalytics
from continuous_evolution_system import ContinuousEvolutionSystem
from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligence
from human_interaction_styler import HumanInteractionStyler
from communication_adapter import CommunicationAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeedbackIntegrationSystemExample")

def run_example() -> None:
    """Run the example demonstration of FeedbackIntegrationSystem."""
    # Initialize needed components
    logger.info("Initializing components...")
    principle_engine = PrincipleEngine(agent_id="principle-engine-001")
    
    # Add the "Fairness as a Fundamental Truth" principle
    principle_engine.add_principle(
        name="Fairness as a Fundamental Truth",
        description="Equal treatment of all messages and feedback regardless of source, with transparent and balanced prioritization.",
        weight=1.0
    )
    
    # Create supporting components (simplified for example)
    orchestrator_engine = None  # Would be an actual OrchestratorEngine in real implementation
    analytics = OrchestrationAnalytics(agent_id="analytics-001")
    continuous_evolution = ContinuousEvolutionSystem(agent_id="evolution-001")
    emotional_intelligence = EmotionalIntelligence(agent_id="emotional-001")
    human_interaction_styler = HumanInteractionStyler(agent_id="interaction-001")
    communication_adapter = CommunicationAdapter(agent_id="communication-001")
    
    # Initialize the feedback integration system
    logger.info("Setting up FeedbackIntegrationSystem...")
    feedback_system = FeedbackIntegrationSystem(
        agent_id="feedback-system-001",
        principle_engine=principle_engine,
        orchestration_analytics=analytics,
        continuous_evolution_system=continuous_evolution,
        emotional_intelligence=emotional_intelligence,
        human_interaction_styler=human_interaction_styler,
        communication_adapter=communication_adapter,
        feedback_storage_dir="data/feedback"
    )
    
    # Step 1: Create stakeholder profiles
    logger.info("Creating stakeholder profiles...")
    end_user_profile = StakeholderProfile(
        stakeholder_id="stakeholder-001",
        name="Alex Johnson",
        roles=[FeedbackSource.END_USER],
        contact_info={
            "email": "alex.johnson@example.com",
            "phone": "+1-555-123-4567"
        },
        preferred_communication_channels=["email", "in-app"]
    )
    
    operator_profile = StakeholderProfile(
        stakeholder_id="stakeholder-002",
        name="Taylor Smith",
        roles=[FeedbackSource.OPERATOR, FeedbackSource.DEVELOPER],
        contact_info={
            "email": "taylor.smith@example.com",
            "slack": "@taylorsmith"
        },
        preferred_communication_channels=["slack", "email"]
    )
    
    business_profile = StakeholderProfile(
        stakeholder_id="stakeholder-003",
        name="Jordan Lee",
        roles=[FeedbackSource.BUSINESS_STAKEHOLDER, FeedbackSource.AGENT_OWNER],
        contact_info={
            "email": "jordan.lee@example.com",
            "phone": "+1-555-987-6543"
        },
        preferred_communication_channels=["email", "meeting"]
    )
    
    # Register stakeholder profiles with the system
    feedback_system.register_stakeholder(end_user_profile)
    feedback_system.register_stakeholder(operator_profile)
    feedback_system.register_stakeholder(business_profile)
    
    # Step 2: Define a custom feedback solicitation template
    logger.info("Creating a custom feedback template...")
    template = feedback_system.create_solicitation_template(
        name="Agent Selection Effectiveness",
        description="Evaluate the effectiveness of agent selection for specific task types",
        target_audience=[FeedbackSource.OPERATOR, FeedbackSource.AGENT_OWNER],
        format=FeedbackFormat.STRUCTURED_SURVEY,
        questions=[
            {
                "id": "agent_selection_accuracy",
                "type": "rating",
                "text": "How accurately are agents matched to tasks requiring their specialties?",
                "scale": "1-5",
                "required": True
            },
            {
                "id": "agent_capacity_management",
                "type": "rating",
                "text": "How effectively is agent capacity being managed?",
                "scale": "1-5",
                "required": True
            },
            {
                "id": "improvement_suggestions",
                "type": "text",
                "text": "What specific improvements to agent selection would you recommend?",
                "required": True
            }
        ],
        introduction="Your feedback on agent selection will help us optimize orchestration effectiveness.",
        conclusion="Thank you for helping us improve our agent selection algorithms.",
        estimated_completion_time=5
    )
    
    # Step 3: Create a feedback solicitation campaign
    logger.info("Creating a feedback solicitation campaign...")
    campaign = feedback_system.create_solicitation_campaign(
        name="Q2 Agent Selection Optimization",
        description="Campaign to gather insights for improving agent selection in Q2",
        template_id=template.template_id,
        start_date=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        end_date=(datetime.now(timezone.utc) + timedelta(days=8)).isoformat(),
        target_stakeholders=["stakeholder-002", "stakeholder-003"],
        distribution_channels=["email", "slack"],
        contextual_data={
            "system_version": "2.3.1",
            "target_improvement_area": "agent_selection",
            "previous_feedback_summary": "Agent selection shows 72% satisfaction in Q1 reports"
        }
    )
    
    logger.info(f"Campaign '{campaign.name}' created with ID: {campaign.campaign_id}")
    
    # Step 4: Simulate receiving feedback from different stakeholders
    logger.info("Simulating feedback from various stakeholders...")
    
    # End user feedback
    end_user_feedback = feedback_system.record_feedback(
        content="The orchestration system took too long to process my request. It seemed like it was bouncing between different agents unnecessarily.",
        feedback_type=FeedbackType.ORCHESTRATION_QUALITY,
        source=FeedbackSource.END_USER,
        urgency=FeedbackUrgency.MEDIUM,
        format=FeedbackFormat.FREE_TEXT,
        stakeholder_id="stakeholder-001",
        numerical_ratings={
            "satisfaction": 2.0,
            "speed": 1.5,
            "quality": 3.5
        },
        related_orchestration_id="orch-task-12345",
        metadata={
            "task_type": "data_analysis",
            "completion_time": 345.5  # seconds
        }
    )
    
    # Operator feedback
    operator_feedback = feedback_system.record_feedback(
        content="The system is consistently selecting Agent B for natural language processing tasks even though Agent C has better specialized capabilities in that area. We should update the agent selection criteria.",
        feedback_type=FeedbackType.AGENT_SELECTION,
        source=FeedbackSource.OPERATOR,
        urgency=FeedbackUrgency.HIGH,
        format=FeedbackFormat.FREE_TEXT,
        stakeholder_id="stakeholder-002",
        related_agent_ids=["agent-B", "agent-C"],
        metadata={
            "observed_tasks": ["nlp-analysis", "text-summarization"],
            "observed_period": "last 2 weeks"
        }
    )
    
    # Business stakeholder feedback
    business_feedback = feedback_system.record_feedback(
        content="We need to improve the alignment with our 'Fairness as Fundamental Truth' principle. Some feedback from junior team members isn't being prioritized equally to senior members.",
        feedback_type=FeedbackType.PRINCIPLE_ALIGNMENT,
        source=FeedbackSource.BUSINESS_STAKEHOLDER,
        urgency=FeedbackUrgency.HIGH,
        format=FeedbackFormat.FREE_TEXT,
        stakeholder_id="stakeholder-003",
        metadata={
            "principle_reference": "Fairness as Fundamental Truth",
            "observation_context": "Weekly orchestration review meeting"
        }
    )
    
    logger.info(f"Received feedback items: {end_user_feedback}, {operator_feedback}, {business_feedback}")
    
    # Step 5: Process and analyze feedback
    logger.info("Processing and analyzing feedback...")
    feedback_analysis = feedback_system.analyze_feedback([
        end_user_feedback, operator_feedback, business_feedback
    ])
    
    logger.info("Feedback analysis results:")
    logger.info(f"Common themes: {feedback_analysis['common_themes']}")
    logger.info(f"Sentiment: {feedback_analysis['sentiment']}")
    logger.info(f"Priority areas: {feedback_analysis['priority_areas']}")
    
    # Step 6: Create a feedback collection for related items
    collection = feedback_system.create_feedback_collection(
        name="Agent Selection Improvements May 2025",
        description="Feedback related to agent selection optimization",
        feedback_items=[operator_feedback],
        associated_topic="agent_selection_optimization"
    )
    
    logger.info(f"Created feedback collection: {collection.collection_id}")
    
    # Step 7: Prioritize improvements based on feedback
    logger.info("Prioritizing improvements based on feedback...")
    prioritized_improvements = feedback_system.prioritize_improvements(
        feedback_items=[end_user_feedback, operator_feedback, business_feedback],
        feedback_collections=[collection]
    )
    
    for i, improvement in enumerate(prioritized_improvements):
        logger.info(f"Priority {i+1}: {improvement['title']} (score: {improvement['priority_score']:.2f})")
        logger.info(f"Description: {improvement['description']}")
        logger.info(f"Impact areas: {', '.join(improvement['impact_areas'])}")
    
    # Step 8: Create action plans for high-priority items
    logger.info("Creating action plans for high-priority feedback...")
    fairness_improvement_plan = feedback_system.create_action_plan(
        title="Enhance Feedback Fairness Implementation",
        description="Ensure all feedback is treated equally regardless of source",
        feedback_items=[business_feedback],
        improvement_actions=[
            {
                "action": "Update source weighting in prioritization algorithm",
                "description": "Adjust weights to ensure junior and senior feedback is valued equally",
                "responsible": "system-admin",
                "timeline": "1 week"
            },
            {
                "action": "Implement blind feedback review process",
                "description": "Hide stakeholder identity during initial feedback processing",
                "responsible": "development-team",
                "timeline": "2 weeks"
            },
            {
                "action": "Create regular fairness audit report",
                "description": "Monthly analysis of feedback processing fairness",
                "responsible": "analytics-team",
                "timeline": "1 month, then ongoing"
            }
        ]
    )
    
    agent_selection_plan = feedback_system.create_action_plan(
        title="Optimize Agent Selection for NLP Tasks",
        description="Improve the selection of appropriate agents for NLP-related tasks",
        feedback_items=[operator_feedback],
        improvement_actions=[
            {
                "action": "Update agent capability profiles",
                "description": "Review and update capability ratings for all NLP-capable agents",
                "responsible": "agent-admin",
                "timeline": "3 days"
            },
            {
                "action": "Enhance selection algorithm",
                "description": "Add specialization weighting to selection criteria",
                "responsible": "orchestration-team",
                "timeline": "1 week"
            },
            {
                "action": "Implement A/B testing",
                "description": "Test old vs. new selection algorithm on similar tasks",
                "responsible": "analytics-team",
                "timeline": "2 weeks"
            }
        ]
    )
    
    # Step 9: Close the feedback loop with stakeholders
    logger.info("Closing the feedback loop with stakeholders...")
    feedback_system.send_feedback_response(
        feedback_id=business_feedback,
        response_type="action_plan",
        response_content={
            "message": "Thank you for your valuable feedback about our principle alignment. We've created an action plan to address this issue.",
            "action_plan": fairness_improvement_plan.to_dict(),
            "timeline": "Implementation begins immediately and will be completed within 2 weeks.",
            "follow_up": "We'll provide an update on our progress in 1 week."
        },
        communication_channel="email"
    )
    
    feedback_system.send_feedback_response(
        feedback_id=operator_feedback,
        response_type="action_plan",
        response_content={
            "message": "We've analyzed your feedback about agent selection for NLP tasks and created an action plan.",
            "action_plan": agent_selection_plan.to_dict(),
            "timeline": "Changes to agent selection will be implemented within 1 week.",
            "follow_up": "We'll schedule a review meeting to discuss the improvements once implemented."
        },
        communication_channel="slack"
    )
    
    # Step 10: Integrate improvements with ContinuousEvolutionSystem
    logger.info("Integrating improvements with ContinuousEvolutionSystem...")
    for improvement in prioritized_improvements[:2]:  # Top 2 priority improvements
        integration_result = feedback_system.integrate_with_evolution_system(
            improvement_item=improvement,
            evolution_system=continuous_evolution,
            integration_type="capability_evolution",
            integration_params={
                "capability_id": improvement.get("related_capability_id"),
                "development_focus": improvement["title"],
                "performance_metrics": {
                    "current": improvement.get("current_metrics", {}),
                    "target": improvement.get("target_metrics", {})
                }
            }
        )
        logger.info(f"Integration result: {integration_result['status']}")
    
    # Step 11: Update analytics with feedback insights
    logger.info("Updating analytics with feedback insights...")
    feedback_system.update_analytics_with_feedback(
        analytics_system=analytics,
        feedback_collections=[collection],
        update_params={
            "metric_adjustments": [
                {
                    "metric_id": "agent_selection_effectiveness",
                    "adjustment_factor": 0.85,  # Adjust downward based on feedback
                    "confidence": 0.75
                }
            ],
            "new_kpis": [
                {
                    "name": "Feedback-Driven Improvements",
                    "description": "Rate of improvements implemented based on feedback",
                    "unit": "percent",
                    "initial_value": 65.0
                }
            ]
        }
    )
    
    logger.info("Example complete!")
    return feedback_system

if __name__ == "__main__":
    run_example()