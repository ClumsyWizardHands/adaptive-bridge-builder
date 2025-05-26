#!/usr/bin/env python3
"""
Example usage of CrossModalContextManager for Adaptive Bridge Builder

This example demonstrates how the CrossModalContextManager maintains continuous
conversation context across different communication channels, links related interactions,
and preserves context when switching between modalities.
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from communication_channel_manager import (
    CommunicationChannelManager, 
    ChannelType, 
    ChannelMessage,
    MessagePriority
)
from session_manager import SessionManager
from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligence
from human_interaction_styler import HumanInteractionStyler
from relationship_tracker import RelationshipTracker

from cross_modal_context_manager import (
    CrossModalContextManager,
    ContextSensitivity,
    ContextLink,
    IdentityLink,
    ModalityTransition
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CrossModalContextManagerExample")

# Create mock classes for demonstration
class MockSession:
    def __init__(self, session_id, participants) -> None:
        self.session_id = session_id
        self.participants = participants
        self.messages = []
        
    def add_message(self, message) -> None:
        self.messages = [*self.messages, message]
        
class MockSessionManager:
    def __init__(self) -> None:
        self.sessions = {}
        
    def create_session(self, participants) -> None:
        session_id = f"session-{len(self.sessions) + 1}"
        session = MockSession(session_id, participants)
        self.sessions = {**self.sessions, session_id: session}
        return session
        
    def get_session(self, session_id) -> None:
        return self.sessions.get(session_id)
        
    def get_sessions_by_participant(self, participant_id) -> None:
        return [
            session for session in self.sessions.values()
            if participant_id in session.participants
        ]
        
    def add_message_to_session(self, session_id, message_id, content, sender_id, metadata=None) -> int:
        session = self.sessions.get(session_id)
        if session:
            message = {
                "id": message_id,
                "content": content,
                "sender_id": sender_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            session.add_message(message)
            return True
        return False
        
    def get_session_messages(self, session_id) -> List[Any]:
        session = self.sessions.get(session_id)
        if session:
            return session.messages
        return []

class MockChannelManager:
    def __init__(self) -> None:
        self.messages = []
        
    async def send_message(self, recipient_id, content, channel_type, **kwargs) -> None:
        message_id = f"msg-{len(self.messages) + 1}"
        message = {
            "message_id": message_id,
            "recipient_id": recipient_id,
            "content": content,
            "channel_type": channel_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.messages = [*self.messages, message]
        logger.info(f"Sent {channel_type.value} message to {recipient_id}")
        return message_id

# Example usage functions
async def demonstrate_identity_linking() -> None:
    """Demonstrate linking identities across different channels."""
    print("\n=== IDENTITY LINKING DEMONSTRATION ===\n")
    
    # Initialize components
    session_manager = MockSessionManager()
    channel_manager = MockChannelManager()
    
    # Initialize CrossModalContextManager
    context_manager = CrossModalContextManager(
        agent_id="agent1",
        session_manager=session_manager,
        channel_manager=channel_manager
    )
    
    # Link identities for a user across different channels
    primary_id = "user123"  # Database ID or other primary identifier
    
    # Link email identity
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.EMAIL,
        channel_identity="jane.doe@example.com",
        verified=True,
        metadata={"name": "Jane Doe", "department": "Engineering"}
    )
    
    # Link chat identity
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.CHAT,
        channel_identity="janedoe",
        verified=False,
        metadata={"platform": "Slack"}
    )
    
    # Link API identity
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.API,
        channel_identity="jane_api_key",
        verified=True,
        metadata={"api_version": "v2"}
    )
    
    # Demonstrate identity resolution
    print("Identity resolution examples:")
    print(f"Primary ID for 'janedoe' on chat: {context_manager.get_primary_id(ChannelType.CHAT, 'janedoe')}")
    print(f"Primary ID for 'jane.doe@example.com' on email: {context_manager.get_primary_id(ChannelType.EMAIL, 'jane.doe@example.com')}")
    email_identity = context_manager.get_channel_identity(primary_id, ChannelType.EMAIL)
    chat_identity = context_manager.get_channel_identity(primary_id, ChannelType.CHAT)
    print(f"Email identity for user123: {email_identity}")
    print(f"Chat identity for user123: {chat_identity}")
    
    # Show all identities
    if primary_id in context_manager.identity_links:
        identity_link = context_manager.identity_links[primary_id]
        print("\nAll identities for this user:")
        for channel_type, identity in identity_link.channel_identities.items():
            print(f"- {channel_type.value}: {identity}")
        print(f"Verification level: {identity_link.verification_level}")
        if identity_link.last_verification:
            print(f"Last verified: {identity_link.last_verification.isoformat()}")
    
async def demonstrate_context_tracking() -> None:
    """Demonstrate tracking context across channels."""
    print("\n=== CONTEXT TRACKING DEMONSTRATION ===\n")
    
    # Initialize components
    session_manager = MockSessionManager()
    channel_manager = MockChannelManager()
    
    # Initialize CrossModalContextManager
    context_manager = CrossModalContextManager(
        agent_id="agent1",
        session_manager=session_manager,
        channel_manager=channel_manager
    )
    
    # Create a test user with multiple identities
    primary_id = "user456"
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.EMAIL,
        channel_identity="john.smith@example.com",
        verified=True
    )
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.CHAT,
        channel_identity="johnsmith",
        verified=True
    )
    
    # Create an initial context link for an email conversation
    email_session = session_manager.create_session(["agent1", "john.smith@example.com"])
    email_message_id = "email-msg-1"
    
    # Add an email message
    session_manager.add_message_to_session(
        email_session.session_id,
        email_message_id,
        "Can you help me with the quarterly budget report?",
        "john.smith@example.com"
    )
    
    # Create a context link for this conversation
    context_link_id = context_manager.create_context_link(
        primary_topic="quarterly budget report",
        entity_ids=[primary_id],
        session_ids=[email_session.session_id],
        message_ids=[email_message_id],
        sensitivity=ContextSensitivity.MEDIUM
    )
    
    print(f"Created context link {context_link_id} for 'quarterly budget report'")
    print("Initial context elements:")
    context_link = context_manager.context_links[context_link_id]
    print(f"- Topic: {context_link.primary_topic}")
    print(f"- Entity IDs: {context_link.entity_ids}")
    print(f"- Session IDs: {context_link.session_ids}")
    print(f"- Message IDs: {context_link.message_ids}")
    print(f"- Sensitivity: {context_link.sensitivity.name}")
    
    # Add more messages to the context
    for i in range(2, 4):
        message_id = f"email-msg-{i}"
        session_manager.add_message_to_session(
            email_session.session_id,
            message_id,
            f"Additional message {i} about the budget report",
            ("agent1" if i % 2 == 0 else "john.smith@example.com")
        )
        context_manager.add_to_context_link(
            link_id=context_link_id,
            message_ids=[message_id]
        )
    
    # Create a chat session and link it to the same context
    chat_session = session_manager.create_session(["agent1", "johnsmith"])
    chat_message_id = "chat-msg-1"
    
    session_manager.add_message_to_session(
        chat_session.session_id,
        chat_message_id,
        "Hi, following up on our email about the budget report.",
        "johnsmith"
    )
    
    # Add chat session and message to the existing context link
    context_manager.add_to_context_link(
        link_id=context_link_id,
        session_ids=[chat_session.session_id],
        message_ids=[chat_message_id]
    )
    
    print("\nUpdated context after chat message:")
    print(f"- Session IDs: {context_link.session_ids}")
    print(f"- Message IDs: {context_link.message_ids}")
    print(f"- Last updated: {context_link.last_updated.isoformat()}")

async def demonstrate_channel_transition() -> None:
    """Demonstrate transitioning between channels with context preservation."""
    print("\n=== CHANNEL TRANSITION DEMONSTRATION ===\n")
    
    # Initialize components
    session_manager = MockSessionManager()
    channel_manager = MockChannelManager()
    principle_engine = PrincipleEngine()
    
    # Initialize CrossModalContextManager
    context_manager = CrossModalContextManager(
        agent_id="agent1",
        session_manager=session_manager,
        channel_manager=channel_manager,
        principle_engine=principle_engine
    )
    
    # Create a test user with multiple identities
    primary_id = "user789"
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.EMAIL,
        channel_identity="alice.johnson@example.com",
        verified=True
    )
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.CHAT,
        channel_identity="alicej",
        verified=True
    )
    
    # Create an email conversation about a project proposal
    email_session = session_manager.create_session(["agent1", "alice.johnson@example.com"])
    
    # Create sequence of messages
    email_messages = [
        {
            "id": "email-msg-1",
            "content": "I'd like to discuss the new marketing project proposal. Can we schedule a meeting?",
            "sender_id": "alice.johnson@example.com"
        },
        {
            "id": "email-msg-2",
            "content": "Certainly, I'd be happy to discuss the marketing project. Would next Tuesday at 2pm work for you?",
            "sender_id": "agent1"
        },
        {
            "id": "email-msg-3",
            "content": "Tuesday works. Could you send me the preliminary budget estimates before then?",
            "sender_id": "alice.johnson@example.com"
        }
    ]
    
    # Add messages to session
    for msg in email_messages:
        session_manager.add_message_to_session(
            email_session.session_id,
            msg["id"],
            msg["content"],
            msg["sender_id"]
        )
    
    # Create a context link for the email conversation
    context_link_id = context_manager.create_context_link(
        primary_topic="marketing project proposal",
        entity_ids=[primary_id],
        session_ids=[email_session.session_id],
        message_ids=[msg["id"] for msg in email_messages],
        sensitivity=ContextSensitivity.MEDIUM
    )
    
    print("Email conversation context created.")
    
    # Now, simulate a transition to chat
    # Generate transition context
    transition_context = context_manager.generate_transition_context(
        entity_id=primary_id,
        from_channel=ChannelType.EMAIL,
        to_channel=ChannelType.CHAT,
        topic="marketing project proposal"
    )
    
    print("\nTransition context when switching from email to chat:")
    print(f"Transition type: {transition_context['transition_type']}")
    print(f"Transition message: \"{transition_context['transition_message']}\"")
    
    # Send a message on the new channel with context
    transition_message = transition_context['transition_message']
    
    # Create the initial chat message that references the previous email conversation
    chat_session = session_manager.create_session(["agent1", "alicej"])
    initial_chat_msg = {
        "id": "chat-msg-1",
        "content": f"{transition_message} I've prepared the preliminary budget estimates you requested.",
        "sender_id": "agent1"
    }
    
    session_manager.add_message_to_session(
        chat_session.session_id,
        initial_chat_msg["id"],
        initial_chat_msg["content"],
        initial_chat_msg["sender_id"]
    )
    
    # Add this chat session to the same context link
    context_manager.add_to_context_link(
        link_id=context_link_id,
        session_ids=[chat_session.session_id],
        message_ids=[initial_chat_msg["id"]]
    )
    
    print("\nFull context now spans across email and chat:")
    context_link = context_manager.context_links[context_link_id]
    print(f"- Topic: {context_link.primary_topic}")
    print(f"- Sessions: {context_link.session_ids}")
    print(f"- Message count: {len(context_link.message_ids)}")
    print(f"- Last updated: {context_link.last_updated.isoformat()}")
    
    # Demonstrate finding related context for a new message
    print("\nLooking up related context for a new chat message:")
    new_chat_message = ChannelMessage.create(
        channel_type=ChannelType.CHAT,
        sender_id="alicej",
        recipient_id="agent1",
        content="Can we discuss the marketing project timeline?"
    )
    
    related_contexts = context_manager.find_related_context(
        entity_id=primary_id,
        topic="marketing project", 
        channel_type=ChannelType.CHAT
    )
    
    if related_contexts:
        print(f"Found {len(related_contexts)} related contexts")
        for ctx in related_contexts:
            print(f"- Topic: {ctx['topic']}")
            print(f"- Last updated: {ctx['last_updated']}")
    else:
        print("No related contexts found")

async def demonstrate_trust_principle() -> None:
    """Demonstrate the 'Trust as the Foundation of Leadership' principle in action."""
    print("\n=== TRUST PRINCIPLE DEMONSTRATION ===\n")
    
    # Initialize components
    session_manager = MockSessionManager()
    channel_manager = MockChannelManager()
    principle_engine = PrincipleEngine()
    
    # Initialize CrossModalContextManager
    context_manager = CrossModalContextManager(
        agent_id="agent1",
        session_manager=session_manager,
        channel_manager=channel_manager,
        principle_engine=principle_engine
    )
    
    # Set up sensitivity overrides for different channels
    context_manager.channel_sensitivity_overrides = {
        ChannelType.EMAIL: ContextSensitivity.MEDIUM,
        ChannelType.CHAT: ContextSensitivity.LOW,
        ChannelType.API: ContextSensitivity.HIGH
    }
    
    # Create a test user with multiple identities
    primary_id = "exec101"
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.EMAIL,
        channel_identity="ceo@example.com",
        verified=True
    )
    context_manager.link_identity(
        primary_id=primary_id,
        channel_type=ChannelType.CHAT,
        channel_identity="ceo_chat",
        verified=True
    )
    
    # Create highly sensitive context about company restructuring
    confidential_session = session_manager.create_session(["agent1", "ceo@example.com"])
    confidential_messages = [
        {
            "id": "conf-email-1",
            "content": "I need you to prepare analysis for the upcoming company restructuring. This is highly confidential.",
            "sender_id": "ceo@example.com"
        },
        {
            "id": "conf-email-2",
            "content": "I understand the sensitivity. I'll prepare the analysis with full confidentiality.",
            "sender_id": "agent1"
        },
        {
            "id": "conf-email-3",
            "content": "We need to consider closing the West division and consolidating operations.",
            "sender_id": "ceo@example.com"
        }
    ]
    
    # Add messages to session
    for msg in confidential_messages:
        session_manager.add_message_to_session(
            confidential_session.session_id,
            msg["id"],
            msg["content"],
            msg["sender_id"]
        )
    
    # Create a high-sensitivity context link
    confidential_link_id = context_manager.create_context_link(
        primary_topic="company restructuring",
        entity_ids=[primary_id],
        session_ids=[confidential_session.session_id],
        message_ids=[msg["id"] for msg in confidential_messages],
        sensitivity=ContextSensitivity.HIGH,  # Mark as highly sensitive
        metadata={"confidential": True, "restricted_to": ["Executive Team"]}
    )
    
    print("Created confidential context about company restructuring")
    
    # Simulate switching to chat - demonstrate how sensitive info is handled
    print("\nSimulating transition from email to chat (lower security level):")
    
    # Generate transition context
    transition_context = context_manager.generate_transition_context(
        entity_id=primary_id,
        from_channel=ChannelType.EMAIL,
        to_channel=ChannelType.CHAT,  # Chat has MEDIUM sensitivity
        topic="company restructuring"
    )
    
    print(f"Transition message: \"{transition_context['transition_message']}\"")
    
    # Check if detailed context is shared
    if "recent_messages" in transition_context:
        print("WARNING: Sensitive messages were included in less secure channel!")
        print(f"Message count: {len(transition_context['recent_messages'])}")
    else:
        print("✓ Sensitive message details were correctly withheld from less secure channel")
        print("Only general context reference was provided, protecting confidential information")
    
    # Now simulate a transition to a secure API channel
    print("\nSimulating transition from email to API (equivalent security level):")
    
    api_transition = context_manager.generate_transition_context(
        entity_id=primary_id,
        from_channel=ChannelType.EMAIL,
        to_channel=ChannelType.API,  # API has HIGH sensitivity
        topic="company restructuring"
    )
    
    # Check if detailed context is shared
    if "recent_messages" in api_transition:
        print("✓ Detailed context was shared on equally secure channel")
        print(f"Message count: {len(api_transition['recent_messages'])}")
    else:
        print("ERROR: Context should have been shared on equally secure channel")
    
    print("\nThis demonstrates the 'Trust as the Foundation of Leadership' principle:")
    print("1. Sensitive information is only shared on appropriately secure channels")
    print("2. Context is preserved across channels but with appropriate security filtering")
    print("3. The system respects confidentiality while maintaining conversation continuity")
    print("4. Trust is maintained by ensuring sensitive information is handled appropriately")

async def main() -> None:
    """Run the example demonstrations."""
    print("\nCROSS-MODAL CONTEXT MANAGER EXAMPLE\n")
    print("This example demonstrates how the CrossModalContextManager:")
    print("1. Links identities across different communication channels")
    print("2. Tracks context across related interactions")
    print("3. Preserves context when switching between communication modes")
    print("4. Implements the 'Trust as the Foundation of Leadership' principle")
    print("   for handling sensitive information appropriately\n")
    
    await demonstrate_identity_linking()
    await demonstrate_context_tracking()
    await demonstrate_channel_transition()
    await demonstrate_trust_principle()
    
    print("\n=== EXAMPLE SUMMARY ===\n")
    print("These examples show how the CrossModalContextManager enables:")
    print("• Continuous conversation context regardless of the communication channel")
    print("• Linking of related interactions across different modalities")
    print("• Recognition of when new interactions relate to previous ones")
    print("• Providing relevant history when switching communication modes")
    print("• Respecting privacy by maintaining appropriate separation between contexts")
    print("• Implementing 'Trust as the Foundation of Leadership' in managing sensitive information")

if __name__ == "__main__":
    asyncio.run(main())