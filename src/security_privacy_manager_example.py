"""
SecurityPrivacyManager Example - Demonstrates secure information handling with the SecurityPrivacyManager.

This example shows how to implement end-to-end encryption for sensitive communications, provide
granular access controls for information sharing between agents, maintain comprehensive audit logs,
support data minimization and purpose limitation, include consent management for human interactions,
and embody the "Trust as the Foundation of Leadership" principle.
"""
import asyncio
import datetime
from typing import Dict, List, Any, Set
import uuid

# Import the security privacy manager and related components
from security_privacy_manager import (
    SecurityPrivacyManager,
    SecurityLevel,
    PurposeCategory,
    ConsentStatus,
    AccessDecision,
    IdentityVerificationLevel
)

# Import dependencies
from principle_engine import PrincipleEngine
from relationship_tracker import RelationshipTracker
from session_manager import SessionManager
from agent_registry import AgentRegistry
from orchestrator_engine import OrchestratorEngine


async def demonstrate_security_privacy_manager():
    """Demonstrate the key features of the SecurityPrivacyManager."""
    
    # Set up dependencies
    principle_engine = PrincipleEngine(principles_path="config/principles.json")
    relationship_tracker = RelationshipTracker()
    session_manager = SessionManager()
    agent_registry = AgentRegistry()
    
    # Initialize the SecurityPrivacyManager
    security_manager = SecurityPrivacyManager(
        principle_engine=principle_engine,
        relationship_tracker=relationship_tracker,
        session_manager=session_manager,
        agent_registry=agent_registry,
        system_id="system-001"
    )
    
    print("SecurityPrivacyManager initialized successfully.")
    
    # -------------------------------------------------------------------------------
    # Example 1: Create and assign security policies
    # -------------------------------------------------------------------------------
    print("\n=== Example 1: Create and assign security policies ===")
    
    # Create security policies for different levels
    public_policy_id = security_manager.create_security_policy(
        name="Public Data Policy",
        description="Policy for public information that can be shared freely",
        security_level=SecurityLevel.PUBLIC,
        requires_consent=False,
        requires_encryption=False
    )
    
    internal_policy_id = security_manager.create_security_policy(
        name="Internal Data Policy",
        description="Policy for internal information shared between agents",
        security_level=SecurityLevel.INTERNAL,
        allowed_agents={"agent-001", "agent-002", "agent-003"},
        allowed_purposes={
            PurposeCategory.TASK_COMPLETION,
            PurposeCategory.COMMUNICATION,
            PurposeCategory.ANALYTICS
        },
        requires_consent=False,
        requires_encryption=False
    )
    
    confidential_policy_id = security_manager.create_security_policy(
        name="Confidential Data Policy",
        description="Policy for sensitive information with restricted access",
        security_level=SecurityLevel.CONFIDENTIAL,
        allowed_agents={"agent-001", "agent-002"},
        allowed_purposes={PurposeCategory.TASK_COMPLETION},
        requires_consent=True,
        requires_encryption=True,
        auto_redact_fields=["email", "phone", "address"]
    )
    
    # Assign policies to resources
    security_manager.assign_security_policy("resource-public-001", public_policy_id)
    security_manager.assign_security_policy("resource-internal-001", internal_policy_id)
    security_manager.assign_security_policy("resource-confidential-001", confidential_policy_id)
    
    print(f"Created and assigned policies: public, internal, and confidential")
    
    # -------------------------------------------------------------------------------
    # Example 2: End-to-end encryption for sensitive communications
    # -------------------------------------------------------------------------------
    print("\n=== Example 2: End-to-end encryption for sensitive communications ===")
    
    # Sensitive data that needs to be encrypted
    sensitive_data = {
        "id": "user-12345",
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "ssn": "123-45-6789",
        "health_info": {
            "conditions": ["diabetes", "hypertension"],
            "medications": ["insulin", "lisinopril"]
        },
        "financial_info": {
            "account_number": "9876543210",
            "balance": 12345.67
        }
    }
    
    # List of agents that are allowed to access this data
    allowed_agents = ["agent-001", "agent-002"]
    
    # Encrypt the data for these agents
    encrypted_package = security_manager.encrypt_sensitive_data(
        data=sensitive_data,
        recipients=allowed_agents,
        resource_id="health-record-12345"
    )
    
    print("Encrypted sensitive data for specific agents:")
    print(f"- Metadata: {encrypted_package['metadata']}")
    print(f"- Has encrypted data: {'encrypted_data' in encrypted_package}")
    
    # Agent-001 decrypts the data
    try:
        decrypted_data = security_manager.decrypt_sensitive_data(
            encrypted_package=encrypted_package,
            agent_id="agent-001"
        )
        print("\nAgent-001 successfully decrypted the data.")
        print(f"- Decrypted name: {decrypted_data['name']}")
        print(f"- Decrypted health conditions: {decrypted_data['health_info']['conditions']}")
    except ValueError as e:
        print(f"Decryption failed: {e}")
    
    # Agent-004 tries to decrypt the data (should fail)
    try:
        decrypted_data = security_manager.decrypt_sensitive_data(
            encrypted_package=encrypted_package,
            agent_id="agent-004"
        )
        print("Agent-004 decrypted the data (this should not happen)")
    except ValueError as e:
        print(f"\nAgent-004 decryption failed as expected: {e}")
    
    # -------------------------------------------------------------------------------
    # Example 3: Granular access control for information sharing
    # -------------------------------------------------------------------------------
    print("\n=== Example 3: Granular access control for information sharing ===")
    
    # Agent-001 requests access to a confidential resource
    decision, reason = security_manager.request_access(
        requester_id="agent-001",
        requester_type="agent",
        resource_id="resource-confidential-001",
        purpose=PurposeCategory.TASK_COMPLETION,
        action="read"
    )
    
    print(f"Agent-001 access request to confidential resource:")
    print(f"- Decision: {decision}")
    print(f"- Reason: {reason}")
    
    # Agent-003 requests access to the same confidential resource (should be denied)
    decision, reason = security_manager.request_access(
        requester_id="agent-003",
        requester_type="agent",
        resource_id="resource-confidential-001",
        purpose=PurposeCategory.TASK_COMPLETION,
        action="read"
    )
    
    print(f"\nAgent-003 access request to confidential resource:")
    print(f"- Decision: {decision}")
    print(f"- Reason: {reason}")
    
    # Agent-002 requests access for an unauthorized purpose (should be denied)
    decision, reason = security_manager.request_access(
        requester_id="agent-002",
        requester_type="agent",
        resource_id="resource-confidential-001",
        purpose=PurposeCategory.ANALYTICS,
        action="read"
    )
    
    print(f"\nAgent-002 access request with unauthorized purpose:")
    print(f"- Decision: {decision}")
    print(f"- Reason: {reason}")
    
    # -------------------------------------------------------------------------------
    # Example 4: Data minimization and purpose limitation
    # -------------------------------------------------------------------------------
    print("\n=== Example 4: Data minimization and purpose limitation ===")
    
    # Original data with many fields
    full_user_data = {
        "id": "user-12345",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "555-123-4567",
        "address": "123 Main St, Anytown, USA",
        "birthdate": "1980-01-01",
        "gender": "male",
        "preferences": {
            "theme": "dark",
            "notifications": True,
            "language": "en-US"
        },
        "account": {
            "created_at": "2020-01-01",
            "last_login": "2023-05-15",
            "subscription": "premium"
        },
        "metrics": {
            "logins": 423,
            "tasks_completed": 157,
            "average_session_minutes": 34
        },
        "task": {
            "id": "task-789",
            "title": "Complete project report",
            "deadline": "2023-06-01",
            "status": "in_progress"
        }
    }
    
    # Minimize data for task completion
    minimized_task_data = security_manager.apply_data_minimization(
        data=full_user_data,
        purpose=PurposeCategory.TASK_COMPLETION
    )
    
    print("Original data fields:")
    print(", ".join(full_user_data.keys()))
    print("Nested fields include:", 
          ", ".join([f"{k}.{sk}" for k in ["preferences", "account", "metrics", "task"] 
                    for sk in full_user_data[k].keys()]))
    
    print("\nMinimized data for TASK_COMPLETION:")
    print(", ".join(minimized_task_data.keys()))
    
    # Minimize data for analytics
    minimized_analytics_data = security_manager.apply_data_minimization(
        data=full_user_data,
        purpose=PurposeCategory.ANALYTICS
    )
    
    print("\nMinimized data for ANALYTICS:")
    print(", ".join(minimized_analytics_data.keys()))
    
    # Process text with sensitive information
    sensitive_text = """
    Please contact John Smith at john.smith@example.com or call him at 555-123-4567.
    His social security number is 123-45-6789 and his credit card is 4111-1111-1111-1111.
    """
    
    redacted_text = security_manager.process_sensitive_text(
        text=sensitive_text,
        security_level=SecurityLevel.CONFIDENTIAL
    )
    
    print("\nOriginal text:")
    print(sensitive_text)
    print("\nRedacted text:")
    print(redacted_text)
    
    # -------------------------------------------------------------------------------
    # Example 5: Consent management for human interactions
    # -------------------------------------------------------------------------------
    print("\n=== Example 5: Consent management for human interactions ===")
    
    # Request consent from a human user
    human_id = "human-001"
    purposes = [
        PurposeCategory.TASK_COMPLETION,
        PurposeCategory.COMMUNICATION,
        PurposeCategory.PERSONALIZATION
    ]
    
    consent_id = security_manager.request_human_consent(
        human_id=human_id,
        purposes=purposes,
        expiry_days=90
    )
    
    print(f"Requested consent from human-001 for {len(purposes)} purposes.")
    print(f"Consent ID: {consent_id}")
    
    # Record the human's consent decision (granted)
    consent_granted = security_manager.consent_manager.record_consent_decision(
        consent_id=consent_id,
        status=ConsentStatus.GRANTED,
        proof="digital-signature-hash-123456"
    )
    
    print(f"\nHuman granted consent: {consent_granted}")
    
    # Check if consent has been granted for a specific purpose
    is_consent_granted, _ = security_manager.consent_manager.check_consent(
        subject_id=human_id,
        purpose=PurposeCategory.TASK_COMPLETION
    )
    
    print(f"Consent granted for TASK_COMPLETION: {is_consent_granted}")
    
    # Human requests access to a resource that requires consent
    decision, reason = security_manager.request_access(
        requester_id=human_id,
        requester_type="human",
        resource_id="resource-confidential-001",
        purpose=PurposeCategory.TASK_COMPLETION,
        action="read",
        context={"data_subject_id": human_id}
    )
    
    print(f"\nHuman access request with proper consent:")
    print(f"- Decision: {decision}")
    print(f"- Reason: {reason}")
    
    # Revoke consent
    revoked = security_manager.consent_manager.revoke_consent(
        consent_id=consent_id,
        actor_id=human_id
    )
    
    print(f"\nConsent revoked: {revoked}")
    
    # Check consent again
    is_consent_granted, _ = security_manager.consent_manager.check_consent(
        subject_id=human_id,
        purpose=PurposeCategory.TASK_COMPLETION
    )
    
    print(f"Consent granted after revocation: {is_consent_granted}")
    
    # -------------------------------------------------------------------------------
    # Example 6: Audit logging and reporting
    # -------------------------------------------------------------------------------
    print("\n=== Example 6: Audit logging and reporting ===")
    
    # Generate an audit report for the last day
    report = security_manager.generate_audit_report(days=1)
    
    print("Audit Report Summary:")
    print(f"- Time Period: {report['start_time']} to {report['end_time']}")
    print(f"- Total Events: {report['total_events']}")
    
    print("\nEvent Types:")
    for event_type, count in report.get('event_counts', {}).items():
        print(f"- {event_type}: {count}")
    
    print("\nTop Actors:")
    for actor, count in sorted(
        report.get('actor_counts', {}).items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]:
        print(f"- {actor}: {count}")
    
    # -------------------------------------------------------------------------------
    # Example 7: Principle validation
    # -------------------------------------------------------------------------------
    print("\n=== Example 7: Principle validation ===")
    
    # Validate a security-related action against principles
    is_valid, reason = await security_manager.validate_principles(
        action="share_sensitive_information",
        context={
            "resource_id": "resource-confidential-001",
            "recipient_id": "agent-002",
            "purpose": PurposeCategory.TASK_COMPLETION.value,
            "security_level": SecurityLevel.CONFIDENTIAL.value,
            "consent_obtained": True
        },
        actor_id="agent-001"
    )
    
    print(f"Principle validation for sharing sensitive information:")
    print(f"- Valid: {is_valid}")
    print(f"- Reason: {reason}")
    
    # Validate an action that violates principles
    is_valid, reason = await security_manager.validate_principles(
        action="share_sensitive_information",
        context={
            "resource_id": "resource-confidential-001",
            "recipient_id": "unknown-agent",
            "purpose": "undefined_purpose",
            "security_level": SecurityLevel.CONFIDENTIAL.value,
            "consent_obtained": False
        },
        actor_id="agent-001"
    )
    
    print(f"\nPrinciple validation for sharing with unknown agent:")
    print(f"- Valid: {is_valid}")
    print(f"- Reason: {reason}")
    
    # -------------------------------------------------------------------------------
    # Example 8: Multi-agent orchestration with security
    # -------------------------------------------------------------------------------
    print("\n=== Example 8: Multi-agent orchestration with security ===")
    
    # Set up a simple orchestrator
    orchestrator = OrchestratorEngine(
        security_manager=security_manager,
        agent_registry=agent_registry
    )
    
    # Simulate a secure multi-agent task
    async def secure_multi_agent_task():
        """Example of a secure multi-agent task with orchestration."""
        
        # Define task with sensitive information
        task_data = {
            "task_id": "task-123",
            "description": "Analyze customer financial data",
            "customer_info": {
                "id": "customer-456",
                "name": "Alice Johnson",
                "financial_data": {
                    "account_number": "9876543210",
                    "balance": 50000,
                    "transactions": [
                        {"date": "2023-05-01", "amount": 1500, "type": "deposit"},
                        {"date": "2023-05-10", "amount": -500, "type": "withdrawal"}
                    ]
                }
            }
        }
        
        # Step 1: Create a security policy for this task
        task_policy_id = security_manager.create_security_policy(
            name=f"Task-123 Security Policy",
            description="Policy for handling sensitive financial analysis task",
            security_level=SecurityLevel.CONFIDENTIAL,
            allowed_agents={"agent-001", "agent-002", "agent-005"},
            allowed_purposes={PurposeCategory.TASK_COMPLETION, PurposeCategory.ANALYTICS},
            requires_consent=True,
            requires_encryption=True
        )
        
        # Step the task resource
        security_manager.assign_security_policy(f"task-{task_data['task_id']}", task_policy_id)
        
        print(f"Created security policy for task: {task_data['task_id']}")
        
        # Step 2: Apply data minimization for different agents
        
        # Data for the analyst agent (needs detailed financial data)
        analyst_data = security_manager.apply_data_minimization(
            data=task_data,
            purpose=PurposeCategory.ANALYTICS
        )
        
        # Data for the reporting agent (needs less detail)
        reporting_data = security_manager.apply_data_minimization(
            data=task_data,
            purpose=PurposeCategory.COMMUNICATION
        )
        
        print(f"Created minimized data views for different agent roles")
        
        # Step 3: Encrypt sensitive parts for specific agents
        encrypted_financial_data = security_manager.encrypt_sensitive_data(
            data=task_data["customer_info"]["financial_data"],
            recipients=["agent-002"],  # Only the financial analyst agent
            resource_id=f"financial-data-{task_data['customer_info']['id']}"
        )
        
        print(f"Encrypted sensitive financial data for the analyst agent")
        
        # Step 4: Demonstrate access control with multi-agent flow
        access_results = {}
        
        # Check access for multiple agents
        for agent_id in ["agent-001", "agent-002", "agent-003", "agent-005"]:
            decision, reason = security_manager.request_access(
                requester_id=agent_id,
                requester_type="agent",
                resource_id=f"task-{task_data['task_id']}",
                purpose=PurposeCategory.TASK_COMPLETION,
                action="read"
            )
            access_results[agent_id] = (decision, reason)
        
        print("\nAccess control results for multiple agents:")
        for agent_id, (decision, reason) in access_results.items():
            print(f"- {agent_id}: {decision} ({reason or 'No reason provided'})")
        
        # Step 5: Log the completion of the secure multi-agent task
        security_manager.log_audit_event(
            event_type="secure_multi_agent_task_completed",
            actor_id="orchestrator",
            actor_type="system",
            resource_id=f"task-{task_data['task_id']}",
            action="complete_task",
            details={
                "agent_count": len(access_results),
                "security_level": SecurityLevel.CONFIDENTIAL.value,
                "task_type": "financial_analysis"
            }
        )
        
        print("\nSecure multi-agent task completed and logged")
    
    # Run the secure multi-agent task
    await secure_multi_agent_task()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_security_privacy_manager())
