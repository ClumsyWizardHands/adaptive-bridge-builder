"""
Session Manager Example

This module demonstrates how to use the SessionManager class to maintain
conversation context across multiple interactions, group related tasks,
store relevant history, implement forgetting mechanisms, and balance
immediate context with long-term relationship data.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from session_manager import SessionManager, Session, SessionStatus, MessageRelevance
from relationship_tracker import RelationshipTracker, RelationshipTrustLevel, InteractionType
from a2a_task_handler import A2ATaskHandler


def basic_session_management() -> None:
    """Example of basic session creation and management."""
    print("\n=== Basic Session Management ===\n")
    
    # Create a session manager for our agent
    manager = SessionManager(
        agent_id="my-agent-001",
        storage_dir="example_output/sessions",
        max_active_sessions=10
    )
    
    # Create a session with another agent
    session = manager.create_session(
        initiator_id="customer-agent-001",
        name="Support Session",
        metadata={"purpose": "technical support", "product": "WidgetPro"}
    )
    
    print(f"Created session with ID: {session.session_id}")
    print(f"Session name: {session.name}")
    print(f"Session status: {session.status.value}")
    
    # Add messages to the session
    message1 = {
        "sender_id": "customer-agent-001",
        "content": "I'm having trouble with the WidgetPro installation. It keeps failing at 75%.",
        "topics": ["installation", "error", "WidgetPro"]
    }
    
    message_id = session.add_message(
        message=message1, 
        intent="request",
        relevance=MessageRelevance.HIGH  # This is an important message about the main issue
    )
    
    print(f"Added message with ID: {message_id}")
    
    # Add a response
    message2 = {
        "sender_id": "my-agent-001",
        "content": "I understand you're having installation issues. What operating system are you using?",
        "topics": ["installation", "operating system", "troubleshooting"]
    }
    
    session.add_message(
        message=message2,
        intent="query",
        relevance=MessageRelevance.MEDIUM  # Standard follow-up question
    )
    
    # Add a less relevant message
    message3 = {
        "sender_id": "customer-agent-001",
        "content": "I'm using Windows 11. By the way, I really like the new UI design in WidgetPro.",
        "topics": ["Windows 11", "UI design"]
    }
    
    session.add_message(
        message=message3,
        intent="inform",
        relevance=MessageRelevance.LOW  # The UI feedback is not directly relevant to the issue
    )
    
    # Add a critical message that should never be forgotten
    message4 = {
        "sender_id": "my-agent-001",
        "content": "I'll need to collect some diagnostic information. Please run the diagnostics tool and share the output file with me.",
        "topics": ["diagnostics", "troubleshooting"]
    }
    
    session.add_message(
        message=message4,
        intent="instruct",
        relevance=MessageRelevance.CRITICAL  # This is critical for solving the issue
    )
    
    # Get session summary
    print("\nSession Summary:")
    print(session.get_summary())
    
    # Get only the most relevant messages
    print("\nRelevant Messages (MEDIUM and above):")
    relevant_messages = session.get_relevant_messages(
        min_relevance=MessageRelevance.MEDIUM,
        max_count=5
    )
    
    for msg in relevant_messages:
        print(f" - {msg.get('sender_id')}: {msg.get('content')[:50]}...")
        
    # Update session status
    session.idle()
    print(f"\nSession status after idle: {session.status.value}")
    
    # Reactivate session
    session.activate()
    print(f"Session status after activation: {session.status.value}")
    
    # Expire session
    session.expire()
    print(f"Session status after expiring: {session.status.value}")


def conversation_context_maintenance() -> None:
    """Example of maintaining conversation context across multiple interactions."""
    print("\n=== Conversation Context Maintenance ===\n")
    
    # Create a session manager
    manager = SessionManager(
        agent_id="support-agent-001",
        storage_dir="example_output/sessions"
    )
    
    # Create a new session or get an existing one
    session = manager.get_or_create_session(
        initiator_id="customer-agent-002",
        metadata={"customer_type": "premium"}
    )
    
    print(f"Session ID: {session.session_id}")
    print(f"Is new conversation: {session.is_new_conversation()}")
    
    # Add initial messages if this is a new conversation
    if session.is_new_conversation():
        # Initial message from customer
        manager.add_message_to_session(
            session_id=session.session_id,
            message={
                "sender_id": "customer-agent-002",
                "content": "I need to upgrade my subscription plan. What options do I have?",
                "topics": ["subscription", "upgrade", "plans"]
            },
            intent="query",
            relevance=MessageRelevance.HIGH
        )
        
        # Response from support agent
        manager.add_message_to_session(
            session_id=session.session_id,
            message={
                "sender_id": "support-agent-001",
                "content": "I'd be happy to help with your subscription upgrade. We have several plans available: Premium, Business, and Enterprise.",
                "topics": ["subscription", "plans", "options"]
            },
            intent="inform",
            relevance=MessageRelevance.HIGH
        )
        
        print("Added initial conversation messages")
    else:
        print("Continuing existing conversation")
        
    # Add a new message in the current interaction
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "customer-agent-002",
            "content": "Can you tell me more about the Enterprise plan? What features does it include?",
            "topics": ["enterprise plan", "features"]
        },
        intent="query",
        relevance=MessageRelevance.HIGH
    )
    
    # Get the relevant context for this ongoing conversation
    context = manager.get_relevant_context(
        session_id=session.session_id,
        max_messages=5
    )
    
    print("\nRelevant context for continuing the conversation:")
    print(f"Session summary: {context['session_summary'].split(chr(10))[0]}...")
    print(f"Topics: {[topic for topic, relevance in context['topics']]}")
    print(f"Recent messages: {len(context['recent_messages'])}")
    
    # Add a response based on the context
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "support-agent-001",
            "content": "The Enterprise plan includes unlimited users, 24/7 priority support, custom integrations, and advanced analytics. It's our most comprehensive offering.",
            "topics": ["enterprise plan", "features", "pricing"]
        },
        intent="inform",
        relevance=MessageRelevance.HIGH
    )
    
    print("\nConversation successfully continued with context maintenance")


def forgetting_mechanism_example() -> None:
    """Example demonstrating the forgetting mechanism to avoid context overflow."""
    print("\n=== Forgetting Mechanism Example ===\n")
    
    # Create a session with a small message limit
    session = Session(
        session_id=f"session-{datetime.now(timezone.utc).timestamp()}",
        agent_id="my-agent-001",
        initiator_id="verbose-agent-001",
        max_message_count=5  # Only keep 5 messages maximum
    )
    
    print(f"Created session with max_message_count={session.max_message_count}")
    
    # Add several messages with different relevance levels
    relevance_levels = [
        MessageRelevance.LOW,
        MessageRelevance.MEDIUM,
        MessageRelevance.LOW,
        MessageRelevance.HIGH,
        MessageRelevance.CRITICAL,
        MessageRelevance.LOW,
        MessageRelevance.MEDIUM
    ]
    
    for i, relevance in enumerate(relevance_levels):
        message = {
            "id": f"msg-{i}",
            "sender_id": "verbose-agent-001" if i % 2 == 0 else "my-agent-001",
            "content": f"Message {i+1} with {relevance.name} relevance: {relevance.value}",
            "topics": [f"topic-{i}"]
        }
        
        session.add_message(message, relevance=relevance)
        
    # Check how many messages we have
    print(f"Added {len(relevance_levels)} messages")
    print(f"Current message count: {len(session.messages)}")
    
    # Check which messages were forgotten
    print("\nRemaining messages after forgetting mechanism:")
    for msg in session.messages:
        msg_id = msg["id"]
        relevance = session.message_relevance.get(msg_id, MessageRelevance.LOW)
        print(f"  - ID: {msg_id}, Relevance: {relevance.name}, Content: {msg['content']}")
        
    print("\nNote how less relevant messages were forgotten first")


def long_term_relationship_integration() -> None:
    """Example showing integration of long-term relationship data with immediate context."""
    print("\n=== Long-term Relationship Integration ===\n")
    
    # Create relationship tracker
    relationship_tracker = RelationshipTracker(storage_dir="example_output/relationships")
    
    # Initialize a relationship with a partner agent
    partner_id = "partner-agent-001"
    relationship_tracker.initialize_relationship(
        agent_id=partner_id,
        name="Business Partner",
        trust_level=RelationshipTrustLevel.HIGH,
        notes="Long-term partner with established communication patterns."
    )
    
    # Record some historical interactions
    for i in range(5):
        relationship_tracker.record_interaction(
            agent_id=partner_id,
            interaction_type=InteractionType.COLLABORATION,
            content_summary=f"Previous collaboration {i+1}",
            quality=i % 2 == 0  # Alternate between positive and neutral
        )
    
    # Create session manager with relationship tracker integration
    manager = SessionManager(
        agent_id="my-agent-001",
        relationship_tracker=relationship_tracker,
        storage_dir="example_output/sessions"
    )
    
    # Create a session with the partner
    session = manager.create_session(
        initiator_id=partner_id,
        name="Joint Project Planning"
    )
    
    # Add some messages to the session
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": partner_id,
            "content": "Let's discuss the timeline for our joint AI project. I think we need to revise the milestones.",
            "topics": ["project timeline", "milestones", "planning"]
        },
        intent="suggest"
    )
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "my-agent-001",
            "content": "I agree. Based on our progress, we should adjust the delivery dates.",
            "topics": ["delivery dates", "planning"]
        },
        intent="confirm"
    )
    
    # Get context with relationship data
    context = manager.get_relevant_context(
        session_id=session.session_id,
        include_relationship_data=True
    )
    
    print("Context with integrated relationship data:")
    print(f"Session topics: {[topic for topic, _ in context['topics']]}")
    print(f"Relationship trust level: {context['relationship']['trust_level']}")
    print(f"Interaction count: {context['relationship']['interaction_count']}")
    print(f"Recent interactions: {len(context['recent_interactions'])}")
    
    print("\nThis shows how immediate context is enriched with long-term relationship data")


def topic_tracking_example() -> None:
    """Example demonstrating topic tracking and session retrieval by topic."""
    print("\n=== Topic Tracking Example ===\n")
    
    # Create session manager
    manager = SessionManager(
        agent_id="knowledge-agent-001",
        storage_dir="example_output/sessions"
    )
    
    # Create sessions with different topics
    session1 = manager.create_session(
        initiator_id="learner-agent-001",
        name="Machine Learning Discussion"
    )
    
    session2 = manager.create_session(
        initiator_id="learner-agent-002",
        name="Computer Vision Discussion" 
    )
    
    session3 = manager.create_session(
        initiator_id="learner-agent-003",
        name="Neural Networks Deep Dive"
    )
    
    # Add messages with topics to each session
    manager.add_message_to_session(
        session_id=session1.session_id,
        message={
            "sender_id": "learner-agent-001",
            "content": "I'm interested in learning about supervised learning algorithms.",
            "topics": ["machine learning", "supervised learning", "algorithms"]
        }
    )
    
    manager.add_message_to_session(
        session_id=session2.session_id,
        message={
            "sender_id": "learner-agent-002",
            "content": "What are the latest advances in object detection using neural networks?",
            "topics": ["computer vision", "object detection", "neural networks"]
        }
    )
    
    manager.add_message_to_session(
        session_id=session3.session_id,
        message={
            "sender_id": "learner-agent-003",
            "content": "How do transformers compare to CNNs for image processing tasks?",
            "topics": ["neural networks", "transformers", "CNNs", "image processing"]
        }
    )
    
    # Find sessions by topic
    neural_network_sessions = manager.find_sessions_by_topic("neural networks")
    print(f"Found {len(neural_network_sessions)} sessions related to 'neural networks':")
    for session in neural_network_sessions:
        print(f"  - {session.name} (ID: {session.session_id})")
        
    # Search across all sessions
    ml_results = manager.search_sessions("machine learning")
    print(f"\nFound {len(ml_results)} sessions with messages containing 'machine learning'")
    
    # Update topic relevance
    if neural_network_sessions:
        session = neural_network_sessions[0]
        old_relevance = session.topic_relevance.get("neural networks", 0.5)
        session.update_topic_relevance("neural networks", 0.3)  # Increase relevance
        new_relevance = session.topic_relevance.get("neural networks", 0.5)
        
        print(f"\nUpdated relevance for 'neural networks' topic from {old_relevance} to {new_relevance}")


def task_grouping_example() -> None:
    """Example demonstrating grouping related tasks within sessions."""
    print("\n=== Task Grouping Example ===\n")
    
    # Create session manager
    manager = SessionManager(
        agent_id="project-manager-001",
        storage_dir="example_output/sessions"
    )
    
    # Create a session for a project
    session = manager.create_session(
        initiator_id="developer-agent-001",
        name="Website Redesign Project"
    )
    
    print(f"Created project session: {session.name} (ID: {session.session_id})")
    
    # Add initial conversation
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "developer-agent-001",
            "content": "I'd like to discuss the website redesign project and break it down into tasks.",
            "topics": ["website redesign", "project planning", "tasks"]
        }
    )
    
    # Add tasks to the session
    tasks = [
        {
            "task_id": "task-001",
            "title": "Design mockups for homepage",
            "assignee": "designer-agent-001",
            "status": "in_progress",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
            "priority": "high"
        },
        {
            "task_id": "task-002",
            "title": "Frontend implementation of homepage",
            "assignee": "developer-agent-001",
            "status": "pending",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=14)).isoformat(),
            "priority": "medium"
        },
        {
            "task_id": "task-003",
            "title": "Backend API development",
            "assignee": "developer-agent-002",
            "status": "pending",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=21)).isoformat(),
            "priority": "medium"
        }
    ]
    
    for task in tasks:
        manager.add_task_to_session(
            session_id=session.session_id,
            task_id=task["task_id"],
            task_metadata=task
        )
        
    print(f"Added {len(tasks)} tasks to the session")
    
    # Add discussions about the tasks
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "project-manager-001",
            "content": "I've created the tasks for our website redesign project. Let's discuss the homepage design first.",
            "topics": ["tasks", "homepage design"],
            "related_task_id": "task-001"
        }
    )
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "developer-agent-001",
            "content": "I have some questions about the frontend implementation. Will we be using React or Vue?",
            "topics": ["frontend", "technology stack"],
            "related_task_id": "task-002"
        }
    )
    
    # Get all tasks for the session
    session_tasks = manager.get_session_tasks(session.session_id)
    print("\nTasks in the session:")
    for task in session_tasks:
        print(f"  - {task['title']} (Status: {task['status']}, Due: {task['due_date'].split('T')[0]})")
    
    # Get context containing tasks
    context = manager.get_relevant_context(session.session_id)
    print(f"\nRelevant context includes {len(context['tasks'])} tasks and {len(context['recent_messages'])} messages")


def session_persistence_example() -> None:
    """Example demonstrating session persistence and retrieval."""
    print("\n=== Session Persistence Example ===\n")
    
    storage_dir = "example_output/persistent_sessions"
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create session manager
    manager = SessionManager(
        agent_id="persistent-agent-001",
        storage_dir=storage_dir
    )
    
    # Create a session
    session = manager.create_session(
        initiator_id="client-agent-001",
        name="Persistent Session Test",
        metadata={"test_case": "persistence"}
    )
    
    print(f"Created session: {session.name} (ID: {session.session_id})")
    
    # Add messages
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "client-agent-001",
            "content": "This is a test message for persistence.",
            "topics": ["test", "persistence"]
        }
    )
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message={
            "sender_id": "persistent-agent-001",
            "content": "I'll store this conversation and we can continue it later.",
            "topics": ["persistence", "storage"]
        }
    )
    
    # Force save
    manager._save_session(session)
    
    print("Session saved to disk")
    
    # Create a new session manager instance (simulating restart)
    new_manager = SessionManager(
        agent_id="persistent-agent-001",
        storage_dir=storage_dir
    )
    
    # Try to retrieve the session
    retrieved_session = new_manager.get_session(session.session_id)
    
    if retrieved_session:
        print("\nSuccessfully retrieved session from disk")
        print(f"Session name: {retrieved_session.name}")
        print(f"Message count: {len(retrieved_session.messages)}")
        print(f"Topics: {retrieved_session.topics}")
        
        # Add a new message to the retrieved session
        new_manager.add_message_to_session(
            session_id=retrieved_session.session_id,
            message={
                "sender_id": "client-agent-001",
                "content": "Hello again! Continuing our conversation from before.",
                "topics": ["continuation"]
            }
        )
        
        print("\nAdded a new message to the retrieved session")
        print(f"Updated message count: {len(retrieved_session.messages)}")
    else:
        print("\nFailed to retrieve session from disk")


def integrated_example() -> None:
    """An integrated example showing all features working together."""
    print("\n=== Integrated Session Management Example ===\n")
    
    # Create relationship tracker
    relationship_tracker = RelationshipTracker(storage_dir="example_output/integrated/relationships")
    
    # Create session manager with relationship integration
    manager = SessionManager(
        agent_id="assistant-agent-001",
        relationship_tracker=relationship_tracker,
        storage_dir="example_output/integrated/sessions",
        default_session_timeout=1800,  # 30 minutes
        default_max_messages=50
    )
    
    # Create a relationship with a user
    user_id = "user-001"
    relationship_tracker.initialize_relationship(
        agent_id=user_id,
        name="Primary User",
        trust_level=RelationshipTrustLevel.HIGH
    )
    
    # Create or continue a session with the user
    session = manager.get_or_create_session(user_id)
    if session.is_new_conversation():
        print("Starting a new conversation")
    else:
        print("Continuing an existing conversation")
        
    print(f"Session ID: {session.session_id}")
    
    # Add user message
    user_message = {
        "sender_id": user_id,
        "content": "I need help setting up the integration with our CRM system. We use Salesforce.",
        "topics": ["integration", "CRM", "Salesforce"]
    }
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message=user_message,
        intent="request",
        relevance=MessageRelevance.HIGH
    )
    
    # Get context for responding
    context = manager.get_relevant_context(
        session_id=session.session_id,
        include_relationship_data=True
    )
    
    print("\nContext for response:")
    print(f"Topics: {[topic for topic, _ in context['topics']]}")
    if "relationship" in context:
        print(f"Trust level: {context['relationship']['trust_level']}")
        print(f"Interaction count: {context['relationship']['interaction_count']}")
    
    # Add assistant response
    assistant_message = {
        "sender_id": "assistant-agent-001",
        "content": "I'll help you set up the Salesforce integration. First, let's create an API key in your Salesforce account.",
        "topics": ["integration", "Salesforce", "API key"]
    }
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message=assistant_message,
        intent="inform",
        relevance=MessageRelevance.HIGH
    )
    
    # Create a task related to this conversation
    task_id = f"task-{datetime.now(timezone.utc).timestamp()}"
    task_metadata = {
        "task_id": task_id,
        "title": "Salesforce CRM Integration",
        "description": "Set up API integration between our system and Salesforce CRM",
        "status": "in_progress",
        "priority": "high",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    manager.add_task_to_session(
        session_id=session.session_id,
        task_id=task_id,
        task_metadata=task_metadata
    )
    
    print(f"\nCreated task '{task_metadata['title']}' in session")
    
    # Add task-related messages
    user_followup = {
        "sender_id": user_id,
        "content": "Where do I find the API key section in Salesforce?",
        "topics": ["Salesforce", "API key"]
    }
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message=user_followup,
        intent="query",
        relevance=MessageRelevance.MEDIUM
    )
    
    assistant_response = {
        "sender_id": "assistant-agent-001",
        "content": "You can find the API key section in Salesforce by going to Setup > API > API Keys. Let me know when you've accessed that page.",
        "topics": ["Salesforce", "API key", "setup"]
    }
    
    manager.add_message_to_session(
        session_id=session.session_id,
        message=assistant_response,
        intent="instruct",
        relevance=MessageRelevance.HIGH
    )
    
    # Get session summary
    print("\nSession Summary:")
    print(session.get_summary())
    
    # Show stats about all sessions
    stats = manager.get_sessions_statistics()
    print("\nSession Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Run cleanup
    cleanup_stats = manager.cleanup()
    print(f"\nCleanup stats: {cleanup_stats}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs("example_output", exist_ok=True)
    os.makedirs("example_output/sessions", exist_ok=True)
    os.makedirs("example_output/relationships", exist_ok=True)
    os.makedirs("example_output/integrated", exist_ok=True)
    os.makedirs("example_output/integrated/sessions", exist_ok=True)
    os.makedirs("example_output/integrated/relationships", exist_ok=True)
    
    # Run examples
    basic_session_management()
    conversation_context_maintenance()
    forgetting_mechanism_example()
    long_term_relationship_integration()
    topic_tracking_example()
    task_grouping_example()
    session_persistence_example()
    integrated_example()