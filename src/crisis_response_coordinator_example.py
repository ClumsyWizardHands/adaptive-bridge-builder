"""
Crisis Response Coordinator Example

This module demonstrates how to use the CrisisResponseCoordinator to rapidly assess
and respond to urgent situations requiring multiple agent expertise. It provides a
concrete example of coordinating multiple specialized agents during a business crisis
requiring immediate, coordinated response.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

from crisis_response_coordinator import (
    CrisisResponseCoordinator, CrisisSeverity, ResponsePhase,
    InformationReliability, InformationPriority, CommunicationChannel,
    InformationSource, CrisisInformation, ResponseAction, DecisionPoint,
    CommunicationMessage, SituationReport, Crisis
)
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability,
    DependencyType, TaskDecompositionStrategy
)
from collaborative_task_handler import TaskStatus, TaskPriority
from project_orchestrator import ResourceType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CrisisResponseCoordinatorExample")


def setup_crisis_coordinator() -> CrisisResponseCoordinator:
    """
    Set up a CrisisResponseCoordinator with specialized agents for crisis response.
    
    Returns:
        Configured CrisisResponseCoordinator instance
    """
    # Create underlying orchestrator engine
    orchestrator_engine = OrchestratorEngine(
        agent_id="crisis-coordinator",
        storage_dir="data/crisis_orchestration"
    )
    
    # Register specialized crisis response agents
    
    # Cybersecurity specialist
    orchestrator_engine.register_agent(
        agent_id="cybersecurity-agent",
        roles=[AgentRole.SPECIALIST, AgentRole.ANALYZER],
        capabilities=["threat_analysis", "system_security", "vulnerability_assessment", "incident_response"],
        specialization={
            TaskType.ANALYSIS: 0.9,
            TaskType.EXECUTION: 0.8,
            TaskType.VALIDATION: 0.8,
        },
        max_load=3,
        metadata={
            "crisis_expertise": ["data_breach", "ransomware", "ddos_attack", "phishing"],
            "response_time": "immediate",
            "security_clearance": "high"
        }
    )
    
    # Public relations specialist
    orchestrator_engine.register_agent(
        agent_id="pr-agent",
        roles=[AgentRole.COMMUNICATOR],
        capabilities=["crisis_communication", "media_relations", "public_statements", "stakeholder_management"],
        specialization={
            TaskType.COMMUNICATION: 0.9,
            TaskType.GENERATION: 0.8,
            TaskType.MONITORING: 0.7,
        },
        max_load=4,
        metadata={
            "crisis_expertise": ["reputation_management", "media_response", "public_perception"],
            "communication_channels": ["press", "social_media", "internal_comms", "stakeholders"]
        }
    )
    
    # Legal advisor
    orchestrator_engine.register_agent(
        agent_id="legal-agent",
        roles=[AgentRole.SPECIALIST, AgentRole.ADVISOR],
        capabilities=["legal_compliance", "risk_assessment", "regulatory_affairs", "liability_analysis"],
        specialization={
            TaskType.ANALYSIS: 0.9,
            TaskType.VALIDATION: 0.8,
            TaskType.DECISION: 0.7,
        },
        max_load=3,
        metadata={
            "crisis_expertise": ["data_protection", "corporate_law", "regulatory_compliance"],
            "jurisdictions": ["US", "EU", "international"]
        }
    )
    
    # Operations manager
    orchestrator_engine.register_agent(
        agent_id="operations-agent",
        roles=[AgentRole.COORDINATOR, AgentRole.EXECUTOR],
        capabilities=["business_continuity", "resource_management", "operational_recovery", "process_management"],
        specialization={
            TaskType.ORCHESTRATION: 0.9,
            TaskType.EXECUTION: 0.8,
            TaskType.MONITORING: 0.8,
        },
        max_load=5,
        metadata={
            "crisis_expertise": ["service_restoration", "supply_chain", "operational_continuity"],
            "business_systems": ["customer_service", "fulfillment", "operations"]
        }
    )
    
    # Executive decision maker
    orchestrator_engine.register_agent(
        agent_id="executive-agent",
        roles=[AgentRole.DECISION_MAKER],
        capabilities=["strategic_assessment", "executive_decision", "stakeholder_management", "crisis_leadership"],
        specialization={
            TaskType.DECISION: 0.9,
            TaskType.ORCHESTRATION: 0.8,
            TaskType.VALIDATION: 0.7,
        },
        max_load=2,
        metadata={
            "crisis_expertise": ["corporate_governance", "strategic_leadership", "organizational_resilience"],
            "authority_level": "executive"
        }
    )
    
    # Customer support specialist
    orchestrator_engine.register_agent(
        agent_id="customer-support-agent",
        roles=[AgentRole.COMMUNICATOR, AgentRole.FRONT_LINE],
        capabilities=["customer_communication", "support_operations", "inquiries_management", "remediation_assistance"],
        specialization={
            TaskType.COMMUNICATION: 0.9,
            TaskType.EXECUTION: 0.8,
            TaskType.MONITORING: 0.7,
        },
        max_load=6,
        metadata={
            "crisis_expertise": ["customer_relations", "service_recovery", "complaint_handling"],
            "channels": ["phone", "email", "chat", "social_media"]
        }
    )
    
    # Create the Crisis Response Coordinator
    crisis_coordinator = CrisisResponseCoordinator(
        agent_id="crisis-coordinator",
        orchestrator_engine=orchestrator_engine,
        storage_dir="data/crisis_response"
    )
    
    # Register information sources
    crisis_coordinator.register_information_source(
        name="Internal Monitoring System",
        source_type="system",
        reliability=InformationReliability.HIGH,
        access_method="API",
        capabilities=["system_status", "network_traffic", "security_alerts"],
        refresh_rate=30  # 30 seconds
    )
    
    crisis_coordinator.register_information_source(
        name="Customer Service Channels",
        source_type="human",
        reliability=InformationReliability.MEDIUM,
        access_method="reports",
        capabilities=["customer_complaints", "service_disruptions", "feedback"],
        refresh_rate=300  # 5 minutes
    )
    
    crisis_coordinator.register_information_source(
        name="External Security Services",
        source_type="external",
        reliability=InformationReliability.HIGH,
        access_method="API",
        capabilities=["threat_intelligence", "vulnerability_data", "security_advisories"],
        refresh_rate=600  # 10 minutes
    )
    
    crisis_coordinator.register_information_source(
        name="Media Monitoring",
        source_type="service",
        reliability=InformationReliability.MEDIUM,
        access_method="API",
        capabilities=["news_coverage", "social_media_sentiment", "public_reaction"],
        refresh_rate=300  # 5 minutes
    )
    
    crisis_coordinator.register_information_source(
        name="Executive Team",
        source_type="human",
        reliability=InformationReliability.HIGH,
        access_method="direct",
        capabilities=["strategic_direction", "decision_making", "authorization"],
        refresh_rate=None  # On-demand
    )
    
    logger.info("Crisis Response Coordinator set up with 6 specialized agents and 5 information sources")
    return crisis_coordinator


def simulate_data_breach_crisis(coordinator: CrisisResponseCoordinator) -> Crisis:
    """
    Simulate a data breach crisis scenario at a company.
    
    Args:
        coordinator: The CrisisResponseCoordinator instance
        
    Returns:
        The created Crisis instance
    """
    # Initialize crisis
    current_time = datetime.utcnow()
    crisis = coordinator.register_crisis(
        name="Customer Database Security Breach",
        description=(
            "Unauthorized access to customer data detected in the main customer database. "
            "Potential exposure of personal information including names, email addresses, "
            "and encrypted payment information for up to 500,000 customers."
        ),
        crisis_type="data_breach",
        severity=CrisisSeverity.HIGH,
        affected_systems=["customer_database", "authentication_system", "payment_processing"],
        affected_stakeholders=[
            {"type": "customers", "count": 500000, "impact": "data_exposure"},
            {"type": "employees", "count": 150, "impact": "operational"},
            {"type": "shareholders", "count": 5000, "impact": "financial"},
            {"type": "partners", "count": 25, "impact": "reputational"}
        ],
        location={
            "physical": "North American Data Center",
            "digital": "Customer Relationship Management System"
        },
        tags=["data_breach", "security_incident", "customer_data", "compliance"]
    )
    
    # Add initial information
    initial_detection = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Initial Breach Detection",
        content=(
            "Security monitoring system detected unusual database query patterns at 02:15 AM. "
            "Abnormal data export of approximately 2TB from customer database observed. "
            "IP address traced to Eastern European region."
        ),
        source_id=list(coordinator.information_sources.keys())[0],  # Internal Monitoring System
        reliability=InformationReliability.HIGH,
        priority=InformationPriority.CRITICAL,
        categories=["detection", "technical", "security"]
    )
    
    # Add breach details information
    breach_details = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Breach Details Assessment",
        content=(
            "Preliminary analysis indicates breach originated through compromised admin credentials. "
            "Access maintained for approximately 3 hours before detection. "
            "Potentially exposed data includes customer names, emails, addresses, phone numbers, "
            "and encrypted payment information (no full card numbers)."
        ),
        source_id=list(coordinator.information_sources.keys())[0],  # Internal Monitoring System
        reliability=InformationReliability.MEDIUM,  # Still being verified
        priority=InformationPriority.URGENT,
        categories=["assessment", "technical", "impact"]
    )
    
    # Add compliance implications information
    compliance_info = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Regulatory Compliance Implications",
        content=(
            "Breach likely falls under GDPR, CCPA, and PCI-DSS reporting requirements. "
            "Mandatory reporting timelines: GDPR (72 hours), CCPA (72 hours), Various state laws (24-48 hours). "
            "Potential penalties could range from $100K to $20M depending on handling of notification and remediation."
        ),
        source_id=list(coordinator.information_sources.keys())[2],  # Legal department via External Security Services
        reliability=InformationReliability.HIGH,
        priority=InformationPriority.URGENT,
        categories=["legal", "compliance", "regulatory"]
    )
    
    # Simulate initial response and action creation
    logger.info("Crisis registered, performing initial response planning...")
    coordinator.assess_crisis_situation(crisis.crisis_id)
    
    # Start response with immediate actions
    coordinator.create_response_actions(crisis.crisis_id)
    
    return crisis


def demonstrate_crisis_response(coordinator: CrisisResponseCoordinator, crisis: Crisis):
    """
    Demonstrate the crisis response process, showing how the coordinator handles the data breach.
    
    Args:
        coordinator: The CrisisResponseCoordinator
        crisis: The active crisis
    """
    logger.info(f"Demonstrating crisis response for: {crisis.name}")
    
    # Get first situation report
    sit_rep = coordinator.generate_situation_report(crisis.crisis_id)
    logger.info(f"Initial situation report generated: {sit_rep.title}")
    logger.info(f"Current crisis severity: {sit_rep.severity.value}")
    logger.info(f"Current response phase: {sit_rep.current_phase.value}")
    
    # Display immediate actions being executed
    logger.info("Top priority actions being executed:")
    for action in coordinator.get_prioritized_actions(crisis.crisis_id, limit=5):
        logger.info(f"  • {action.title} (Priority: {action.priority.value}) - Assigned to: {action.assigned_agent_id}")
    
    # Simulate new information coming in
    logger.info("New information received from external source...")
    
    # Add new threat intelligence information
    threat_intel = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Threat Intelligence Update",
        content=(
            "Attack patterns match known cybercriminal group 'DarkShadow'. Group has history of data exfiltration "
            "followed by ransom demands. Typically contacts company within 24-48 hours of breach with proof of data "
            "and ransom demands. No evidence of data being sold yet on monitored dark web channels."
        ),
        source_id=list(coordinator.information_sources.keys())[2],  # External Security Services
        reliability=InformationReliability.MEDIUM,
        priority=InformationPriority.URGENT,
        categories=["threat_intelligence", "attacker_profile", "prediction"]
    )
    
    # Add customer impact information
    customer_impact = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Customer Service Impact",
        content=(
            "No significant increase in customer service inquiries yet, indicating breach not yet public. "
            "Support team reports 3 customers calling about suspicious emails, possibly related to breach. "
            "Social media monitoring shows no indication of public awareness."
        ),
        source_id=list(coordinator.information_sources.keys())[1],  # Customer Service Channels
        reliability=InformationReliability.MEDIUM,
        priority=InformationPriority.IMPORTANT,
        categories=["customer_impact", "public_awareness", "service"]
    )
    
    # Demonstrate adaptability as new information comes in
    logger.info("Adapting response plan based on new information...")
    
    # Reassess and update the response plan
    coordinator.update_response_plan(crisis.crisis_id)
    
    # Show decision points that need executive input
    logger.info("Critical decisions requiring executive input:")
    decision_point = coordinator.create_decision_point(
        crisis_id=crisis.crisis_id,
        title="Ransom Payment Policy Decision",
        description=(
            "Based on threat intelligence, a ransom demand is likely. Need executive decision on company policy "
            "regarding potential ransom payment if demanded."
        ),
        options=[
            {
                "option": "Refuse to pay ransom under any circumstances",
                "consequences": [
                    "Maintains company policy against encouraging criminal activity",
                    "May result in data publication and increased reputational damage",
                    "Legal and regulatory position is cleaner"
                ],
                "risks": "Data may be published, leading to greater customer impact"
            },
            {
                "option": "Evaluate ransom demands if received",
                "consequences": [
                    "Keeps options open based on situation severity",
                    "May prevent immediate data publication",
                    "Could require negotiation capability"
                ],
                "risks": "May encourage future attacks, potential legal complications"
            },
            {
                "option": "Prepare for payment if critical data is at risk",
                "consequences": [
                    "Could prevent most severe customer impacts",
                    "Requires cryptocurrency preparation",
                    "Would need strict confidentiality"
                ],
                "risks": "Legal, ethical, and precedent-setting concerns"
            }
        ],
        decision_maker="executive-agent",
        required_information=[threat_intel.info_id],
        priority=TaskPriority.HIGH
    )
    
    logger.info(f"  • {decision_point.title} - Options: {len(decision_point.options)}")
    
    # Simulate passage of time and crisis progression
    logger.info("Simulating crisis progression (8 hours later)...")
    
    # Add executive decision for ransom policy
    coordinator.record_decision(
        crisis_id=crisis.crisis_id,
        decision_id=decision_point.decision_id,
        selected_option=0,  # Option 0: Refuse to pay
        rationale=(
            "Executive team and board have decided to maintain our published security policy of "
            "not paying ransoms under any circumstances. This aligns with law enforcement "
            "recommendations and our ethical stance against funding criminal organizations. "
            "We will focus resources on containing damage, supporting affected customers, and "
            "improving security measures."
        )
    )
    
    # Add new information - ransom demand received
    ransom_demand = coordinator.add_crisis_information(
        crisis_id=crisis.crisis_id,
        title="Ransom Demand Received",
        content=(
            "Email received to CEO and Security Director containing samples of customer data and "
            "demand for $2M in cryptocurrency. Threat to publish all data within 48 hours if not paid. "
            "Email contains technical details confirming authenticity."
        ),
        source_id=list(coordinator.information_sources.keys())[4],  # Executive Team
        reliability=InformationReliability.CONFIRMED,
        priority=InformationPriority.CRITICAL,
        categories=["ransom", "threat", "escalation"]
    )
    
    # Demonstrate adaptability in action as situation escalates
    logger.info("Situation escalating - rapidly adapting response plan...")
    
    # Re-prioritize actions based on new development
    coordinator.reprioritize_actions(crisis.crisis_id)
    new_priority_actions = coordinator.get_prioritized_actions(crisis.crisis_id, limit=5)
    
    logger.info("Updated priority actions based on new developments:")
    for action in new_priority_actions:
        logger.info(f"  • {action.title} (Priority: {action.priority.value}) - Assigned to: {action.assigned_agent_id}")
    
    # Show multi-agent coordination during fast-developing situation
    logger.info("Coordinating multiple specialized agents to address escalation...")
    
    # Simultaneous actions from different specialized agents
    actions = [
        coordinator.create_response_action(
            crisis_id=crisis.crisis_id,
            title="Public Notification Preparation",
            description=(
                "Prepare customer notification, press release, and public FAQ regarding the data breach. "
                "Include information on what data was exposed, what customers should do, and what the "
                "company is doing to address the situation."
            ),
            assigned_agent_id="pr-agent",
            priority=TaskPriority.CRITICAL,
            phase=ResponsePhase.IMMEDIATE_RESPONSE,
            communication_channel=CommunicationChannel.PUBLIC
        ),
        coordinator.create_response_action(
            crisis_id=crisis.crisis_id,
            title="Law Enforcement Engagement",
            description=(
                "Contact FBI cyber division and relevant international authorities regarding the ransom "
                "demand and provide all evidence collected so far. Request guidance on criminal investigation."
            ),
            assigned_agent_id="legal-agent",
            priority=TaskPriority.HIGH,
            phase=ResponsePhase.IMMEDIATE_RESPONSE,
            communication_channel=CommunicationChannel.AUTHORITIES
        ),
        coordinator.create_response_action(
            crisis_id=crisis.crisis_id,
            title="Customer Support Surge Planning",
            description=(
                "Scale up customer support capacity by 300% to handle anticipated surge in inquiries. "
                "Deploy pre-authorized overtime, external support services, and chatbot enhancements. "
                "Conduct rapid training on breach response protocols."
            ),
            assigned_agent_id="operations-agent",
            priority=TaskPriority.HIGH,
            phase=ResponsePhase.IMMEDIATE_RESPONSE
        )
    ]
    
    # Generate updated situation report after developments
    updated_sit_rep = coordinator.generate_situation_report(crisis.crisis_id)
    
    logger.info(f"Updated situation report: {updated_sit_rep.title}")
    logger.info("Key metrics:")
    for metric, value in updated_sit_rep.key_metrics.items():
        logger.info(f"  • {metric}: {value}")
    
    # Show how the coordinator manages and synthesizes critical information for decision makers
    logger.info("Synthesizing critical information for executive briefing...")
    
    # Generate executive briefing with synthesized information
    executive_briefing = coordinator.generate_executive_briefing(crisis.crisis_id)
    
    logger.info(f"Executive briefing prepared with {len(executive_briefing['key_points'])} key points")
    logger.info("Top recommendations:")
    for rec in executive_briefing["recommendations"][:3]:
        logger.info(f"  • {rec}")
    
    # Show adaptability metrics being tracked
    logger.info("Adaptability metrics during this crisis response:")
    adaptability_metrics = coordinator.get_adaptability_metrics(crisis.crisis_id)
    for metric, value in adaptability_metrics.items():
        if isinstance(value, float):
            logger.info(f"  • {metric}: {value:.2f}")
        else:
            logger.info(f"  • {metric}: {value}")
    
    # Demonstrate how the principle of "Adaptability as a Form of Strength" is applied
    logger.info("Examples of 'Adaptability as a Form of Strength' principle in action:")
    logger.info("  • Rapid plan adjustment based on ransom demand (plan_adjustment_rate: high)")
    logger.info("  • Continuous reprioritization as new information emerged")
    logger.info("  • Resource reallocation to customer support in anticipation of needs")
    logger.info("  • Parallel processing of legal, technical, and communication workstreams")
    logger.info("  • Flexible decision-making process with multiple options maintained")


def main():
    """Run the crisis response coordinator example."""
    # Set up the crisis coordinator with specialized agents
    coordinator = setup_crisis_coordinator()
    
    # Simulate a data breach crisis
    crisis = simulate_data_breach_crisis(coordinator)
    
    # Demonstrate crisis response coordination
    demonstrate_crisis_response(coordinator, crisis)


if __name__ == "__main__":
    main()
