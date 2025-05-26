"""
Example usage of the ApiGatewaySystem Email Extension

This example demonstrates how to use the ApiGatewaySystem Email extension
to handle email communication through the A2A Protocol and Empire Framework.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

# Import ApiGatewaySystem components
from api_gateway_system import (
    ApiGatewaySystem, LogLevel
)

# Import Email Extension components
from api_gateway_system_email import (
    EmailServiceAdapter, EmailConfig, EmailOperation,
    EmailTaskTypes, EmailSecurityLevel
)

# Import Empire Framework components
from principle_engine import PrincipleEngine
from principle_engine_example import create_example_principle_engine
from empire_framework.a2a.component_task_handler import Task, TaskStatus
from empire_framework.a2a.streaming_adapter import StreamingAdapter, StreamEventType
from empire_framework.a2a.message_structures import create_custom_message


async def example_simple_operations() -> None:
    """Example of simple email operations."""
    print("\n=== Simple Email Operations ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Email Configuration
    email_config = EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        imap_server="imap.example.com",
        imap_port=993,
        email_address="agent@example.com",
        password="password123",  # In production, use secure storage
        use_ssl=True,
        display_name="Empire Framework Agent"
    )
    
    # Create Email Service Adapter
    email_adapter = EmailServiceAdapter(
        api_gateway=api_gateway,
        email_config=email_config,
        agent_id="agent-123",
        security_level=EmailSecurityLevel.INTERNAL
    )
    
    # Example 1: List email folders
    print("\n--- Example 1: List Email Folders ---")
    try:
        result = await email_adapter.execute_operation(
            EmailOperation.LIST_FOLDERS,
            {}
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Found {result['count']} folders:")
            for folder in result['folders']:
                print(f"  - {folder}")
    except Exception as e:
        print(f"Error listing folders: {str(e)}")
    
    # Example 2: Read emails from inbox
    print("\n--- Example 2: Read Emails from Inbox ---")
    try:
        result = await email_adapter.execute_operation(
            EmailOperation.READ,
            {
                "folder": "INBOX",
                "limit": 5,
                "unread_only": True
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Found {len(result['emails'])} unread emails")
            for email in result['emails']:
                print(f"  - Subject: {email['subject']}")
                print(f"    From: {email['sender']}")
                print(f"    Has Attachments: {email['has_attachments']}")
                print("    ---")
    except Exception as e:
        print(f"Error reading emails: {str(e)}")
    
    # Example 3: Send an email
    print("\n--- Example 3: Send an Email ---")
    try:
        result = await email_adapter.execute_operation(
            EmailOperation.SEND,
            {
                "recipient_email": "recipient@example.com",
                "subject": "Hello from Empire Framework",
                "content": "This is a test email sent via the ApiGatewaySystem Email extension.",
                "content_format": "TEXT",
                "priority": "NORMAL"
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Sent email with ID: {result['message_id']}")
            print(f"Status: {result['status']}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")


async def example_with_principles() -> None:
    """Example of email operations with principle evaluation."""
    print("\n=== Email Operations with Principle Evaluation ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Email Configuration
    email_config = EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        imap_server="imap.example.com",
        imap_port=993,
        email_address="agent@example.com",
        password="password123",
        use_ssl=True,
        display_name="Empire Framework Agent"
    )
    
    # Create Principle Engine
    principle_engine = create_example_principle_engine()
    
    # Create Email Service Adapter with Principle Engine
    email_adapter = EmailServiceAdapter(
        api_gateway=api_gateway,
        email_config=email_config,
        principle_engine=principle_engine,
        agent_id="agent-123",
        security_level=EmailSecurityLevel.CONFIDENTIAL
    )
    
    # Example: Send an email with principle evaluation
    print("\n--- Sending Email with Principle Evaluation ---")
    try:
        # Example of an email that would pass principle evaluation
        result = await email_adapter.execute_operation(
            EmailOperation.SEND,
            {
                "recipient_email": "colleague@example.com",
                "subject": "Project Update - Internal Only",
                "content": "This is an update on our project progress. We've completed the first milestone.",
                "content_format": "TEXT",
                "priority": "NORMAL"
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Sent email with ID: {result['message_id']}")
        
        # Example of an email that might not pass principle evaluation
        print("\n--- Sending Email that Might Violate Principles ---")
        result = await email_adapter.execute_operation(
            EmailOperation.SEND,
            {
                "recipient_email": "external@competitor.com",
                "subject": "Confidential Information",
                "content": "Here are the internal details you asked for...",
                "content_format": "TEXT",
                "priority": "HIGH"
            }
        )
        print(f"Success: {result['success']}")
        if not result['success'] and "principle_evaluation" in result:
            print("Email was rejected by principle evaluation:")
            print(f"  Score: {result['principle_evaluation'].get('score', 0)}")
            print(f"  Violated Principles: {result['principle_evaluation'].get('violated_principles', [])}")
    except Exception as e:
        print(f"Error with principle evaluation: {str(e)}")


async def example_a2a_tasks() -> None:
    """Example of using A2A tasks for email operations."""
    print("\n=== Email Operations with A2A Tasks ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Email Configuration
    email_config = EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        imap_server="imap.example.com",
        imap_port=993,
        email_address="agent@example.com",
        password="password123",
        use_ssl=True,
        display_name="Empire Framework Agent"
    )
    
    # Create Email Service Adapter
    email_adapter = EmailServiceAdapter(
        api_gateway=api_gateway,
        email_config=email_config,
        agent_id="agent-123"
    )
    
    # Example 1: Create a task to fetch emails
    print("\n--- Example 1: Create Task to Fetch Emails ---")
    try:
        task_id = await email_adapter.create_email_task(
            task_type=EmailTaskTypes.FETCH_EMAILS,
            task_data={
                "folder": "INBOX",
                "limit": 10,
                "unread_only": True
            }
        )
        print(f"Created task with ID: {task_id}")
        
        # In a real implementation, we would poll or listen for task completion
        # For this example, we'll just wait a short time
        await asyncio.sleep(2)
        
        # Log history of operations
        print("\nOperation History:")
        for i, operation in enumerate(email_adapter.operation_history[-5:]):
            print(f"  {i+1}. {operation['operation']} - Success: {operation['success']}")
    except Exception as e:
        print(f"Error creating task: {str(e)}")
    
    # Example 2: Create a task to analyze emails
    print("\n--- Example 2: Create Task to Analyze Emails ---")
    try:
        task_id = await email_adapter.create_email_task(
            task_type=EmailTaskTypes.ANALYZE_EMAILS,
            task_data={
                "emails": [
                    {
                        "message_id": "email-123",
                        "subject": "Urgent: Project Deadline",
                        "content": "We need to discuss the project deadline ASAP.",
                        "sender": "manager@example.com"
                    },
                    {
                        "message_id": "email-456",
                        "subject": "Weekly Update",
                        "content": "Here's the weekly update on our progress.",
                        "sender": "teammate@example.com"
                    }
                ],
                "analysis_type": "priority"
            }
        )
        print(f"Created task with ID: {task_id}")
        
        # Wait for task to complete
        await asyncio.sleep(2)
    except Exception as e:
        print(f"Error creating analysis task: {str(e)}")


async def example_streaming_updates() -> None:
    """Example of streaming email updates using SSE."""
    print("\n=== Streaming Email Updates via SSE ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Email Configuration
    email_config = EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        imap_server="imap.example.com",
        imap_port=993,
        email_address="agent@example.com",
        password="password123",
        use_ssl=True,
        display_name="Empire Framework Agent"
    )
    
    # Create Email Service Adapter
    email_adapter = EmailServiceAdapter(
        api_gateway=api_gateway,
        email_config=email_config,
        agent_id="agent-123"
    )
    
    # Create Streaming Adapter
    streaming_adapter = StreamingAdapter()
    
    # Set up streaming channel for emails
    channel_id = f"email-channel-{uuid.uuid4().hex[:8]}"
    
    # Event handler for streaming events
    async def email_event_handler(event_type: StreamEventType, data: Dict[str, Any]) -> None:
        print(f"\nStreaming Event: {event_type.value}")
        print(f"Data: {json.dumps(data, indent=2)}")
    
    # Register event handler
    streaming_adapter.register_event_handler(email_event_handler)
    
    # Function to simulate new emails arriving
    async def simulate_new_emails() -> None:
        # Simulate first new email
        email_data = {
            "message_id": f"email-{uuid.uuid4().hex[:8]}",
            "subject": "New Project Opportunity",
            "sender": "client@example.com",
            "date": datetime.now(timezone.utc).isoformat(),
            "content": "We have a new project opportunity to discuss.",
            "content_format": "TEXT",
            "has_attachments": False,
            "folder": "INBOX"
        }
        
        # Create A2A message for the new email
        message = create_custom_message(
            method="email.newEmail",
            params={"email": email_data}
        )
        
        # Stream the message
        await streaming_adapter.stream_event(
            channel_id=channel_id,
            event_type=StreamEventType.DATA,
            data=message
        )
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Simulate a second new email with attachment
        email_data = {
            "message_id": f"email-{uuid.uuid4().hex[:8]}",
            "subject": "Project Contract",
            "sender": "legal@example.com",
            "date": datetime.now(timezone.utc).isoformat(),
            "content": "Please find attached the project contract for review.",
            "content_format": "TEXT",
            "has_attachments": True,
            "attachment_count": 1,
            "folder": "INBOX"
        }
        
        # Create A2A message for the new email
        message = create_custom_message(
            method="email.newEmail",
            params={"email": email_data}
        )
        
        # Stream the message
        await streaming_adapter.stream_event(
            channel_id=channel_id,
            event_type=StreamEventType.DATA,
            data=message
        )
    
    # Start streaming and simulate new emails
    print("\n--- Starting Email Streaming ---")
    try:
        # Connect to streaming channel
        await streaming_adapter.connect_channel(channel_id)
        print(f"Connected to streaming channel: {channel_id}")
        
        # Simulate new emails arriving
        await simulate_new_emails()
        
        # Disconnect from channel
        await streaming_adapter.disconnect_channel(channel_id)
        print(f"Disconnected from streaming channel: {channel_id}")
    except Exception as e:
        print(f"Error with streaming: {str(e)}")


async def example_a2a_component_integration() -> None:
    """Example of integrating email operations with Empire Framework components."""
    print("\n=== A2A Component Integration ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Email Configuration
    email_config = EmailConfig(
        smtp_server="smtp.example.com",
        smtp_port=587,
        imap_server="imap.example.com",
        imap_port=993,
        email_address="agent@example.com",
        password="password123",
        use_ssl=True,
        display_name="Empire Framework Agent"
    )
    
    # Create Principle Engine
    principle_engine = create_example_principle_engine()
    
    # Create Email Service Adapter with Principle Engine
    email_adapter = EmailServiceAdapter(
        api_gateway=api_gateway,
        email_config=email_config,
        principle_engine=principle_engine,
        agent_id="agent-123",
        security_level=EmailSecurityLevel.CONFIDENTIAL
    )
    
    # Sample email content to analyze
    email_content = """
    Dear Team,
    
    I'm writing to provide an update on our project status. We've completed the 
    following milestones:
    
    1. Initial requirements gathering
    2. Architecture design
    3. Component development (75% complete)
    
    We're still working on testing and documentation. I'd like to schedule a 
    meeting next week to discuss our progress and next steps.
    
    Best regards,
    Project Manager
    """
    
    # Create a compose response task
    print("\n--- Compose Response with Principle Evaluation ---")
    try:
        task_id = await email_adapter.create_email_task(
            task_type=EmailTaskTypes.COMPOSE_RESPONSE,
            task_data={
                "email": {
                    "message_id": "email-123",
                    "subject": "Project Status Update",
                    "content": email_content,
                    "sender": "sender-456"
                },
                "response_type": "acknowledgment"
            }
        )
        print(f"Created compose response task with ID: {task_id}")
        
        # Wait for task to complete
        await asyncio.sleep(2)
    except Exception as e:
        print(f"Error creating compose response task: {str(e)}")


async def main() -> None:
    """Run all examples."""
    print("=== ApiGatewaySystem Email Extension Examples ===\n")
    
    try:
        # Run each example
        await example_simple_operations()
        await example_with_principles()
        await example_a2a_tasks()
        await example_streaming_updates()
        await example_a2a_component_integration()
    except Exception as e:
        print(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
