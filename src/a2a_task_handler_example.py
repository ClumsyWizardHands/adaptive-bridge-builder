#!/usr/bin/env python3
"""
A2A Task Handler Example

This module demonstrates how to use the A2ATaskHandler class
for processing tasks from other agents in the Adaptive Bridge Builder framework.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from a2a_task_handler import (
    A2ATaskHandler,
    MessageIntent,
    ContentType,
    TaskPriority,
    TaskStatus,
    MessageContext
)
from principle_engine import PrincipleEngine
from communication_style_analyzer import CommunicationStyleAnalyzer
from relationship_tracker import RelationshipTracker
from conflict_resolver import ConflictResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("A2ATaskHandlerExample")

def create_example_message(
    content: str,
    method: str = "process",
    agent_id: str = "example-agent-001",
    message_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create an example message for testing."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": {
            "content": content,
            "sender": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        },
        "id": message_id or f"msg-{uuid.uuid4().hex[:8]}"
    }

def print_separator(title: str = None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def main():
    """Run the A2A Task Handler example."""
    print_separator("A2A Task Handler Example")
    
    # Create temporary directories for data
    data_dir = "data/example_a2a_tasks"
    rel_data_dir = "data/example_relationships"
    
    # Initialize components
    principle_engine = PrincipleEngine()
    communication_analyzer = CommunicationStyleAnalyzer()
    relationship_tracker = RelationshipTracker(
        agent_id="adaptive-bridge-001",
        data_dir=rel_data_dir
    )
    conflict_resolver = ConflictResolver(
        agent_id="adaptive-bridge-001",
        principle_engine=principle_engine,
        relationship_tracker=relationship_tracker
    )
    
    # Initialize the A2A Task Handler
    task_handler = A2ATaskHandler(
        agent_id="adaptive-bridge-001",
        principle_engine=principle_engine,
        communication_analyzer=communication_analyzer,
        relationship_tracker=relationship_tracker,
        conflict_resolver=conflict_resolver,
        data_dir=data_dir
    )
    
    print(f"A2A Task Handler initialized for agent: adaptive-bridge-001")
    print(f"Using PrincipleEngine: {principle_engine is not None}")
    print(f"Using CommunicationStyleAnalyzer: {communication_analyzer is not None}")
    print(f"Using RelationshipTracker: {relationship_tracker is not None}")
    print(f"Using ConflictResolver: {conflict_resolver is not None}")
    
    # Create a conversation context
    conversation_id = "example-conversation-001"
    
    # Example 1: Process a query task
    print_separator("Example 1: Processing a Query")
    
    query_message = create_example_message(
        content="Can you please tell me what capabilities you have?",
        agent_id="agent-alice"
    )
    
    query_response = task_handler.handle_task(
        message=query_message,
        agent_id="agent-alice",
        conversation_id=conversation_id
    )
    
    print("Query message:")
    print(f"  Content: {query_message['params']['content']}")
    print(f"  From agent: {query_message['params']['sender']}")
    
    print("\nResponse:")
    print(f"  Status: {query_response['result']['status']}")
    print(f"  Message: {query_response['result']['message']}")
    print(f"  Task ID: {query_response['result']['task_id']}")
    print(f"  Conversation ID: {query_response['result']['conversation_id']}")
    
    # Example 2: Process an instruction task
    print_separator("Example 2: Processing an Instruction")
    
    instruction_message = create_example_message(
        content="Please generate a summary of the data I sent earlier.",
        agent_id="agent-bob"
    )
    
    instruction_response = task_handler.handle_task(
        message=instruction_message,
        agent_id="agent-bob",
        conversation_id="different-conversation-001"
    )
    
    print("Instruction message:")
    print(f"  Content: {instruction_message['params']['content']}")
    print(f"  From agent: {instruction_message['params']['sender']}")
    
    print("\nResponse:")
    print(f"  Status: {instruction_response['result']['status']}")
    print(f"  Message: {instruction_response['result']['message']}")
    print(f"  Action: {instruction_response['result'].get('action_performed', 'N/A')}")
    
    # Example 3: Process a request task
    print_separator("Example 3: Processing a Request")
    
    request_message = create_example_message(
        content="Could you analyze this dataset and provide insights?",
        agent_id="agent-charlie"
    )
    
    request_response = task_handler.handle_task(
        message=request_message,
        agent_id="agent-charlie",
        conversation_id=conversation_id
    )
    
    print("Request message:")
    print(f"  Content: {request_message['params']['content']}")
    print(f"  From agent: {request_message['params']['sender']}")
    
    print("\nResponse:")
    print(f"  Status: {request_response['result']['status']}")
    print(f"  Message: {request_response['result']['message']}")
    
    # Example 4: Process messages in conversation context
    print_separator("Example 4: Conversation Context")
    
    # Add another message to the same conversation
    follow_up_message = create_example_message(
        content="I'm specifically interested in anomalies in the data.",
        agent_id="agent-charlie"
    )
    
    follow_up_response = task_handler.handle_task(
        message=follow_up_message,
        agent_id="agent-charlie",
        conversation_id=conversation_id
    )
    
    # Get context information
    context_summary = follow_up_response['result']['context_summary']
    
    print("Conversation context:")
    print(f"  Conversation ID: {conversation_id}")
    print(f"  Message count: {context_summary['message_count']}")
    print(f"  Last updated: {context_summary['last_updated']}")
    
    # Example 5: Process a message with potential conflicts
    print_separator("Example 5: Conflict Detection")
    
    conflict_message = create_example_message(
        content="I disagree with your approach. Your analysis is incorrect and misleading.",
        agent_id="agent-dave"
    )
    
    conflict_response = task_handler.handle_task(
        message=conflict_message,
        agent_id="agent-dave",
        conversation_id="conflict-conversation-001"
    )
    
    print("Potential conflict message:")
    print(f"  Content: {conflict_message['params']['content']}")
    print(f"  From agent: {conflict_message['params']['sender']}")
    
    print("\nResponse:")
    print(f"  Status: {conflict_response['result']['status']}")
    print(f"  Message: {conflict_response['result']['message']}")
    
    # Check if conflict was detected
    task_id = conflict_response['result']['task_id']
    task_data = task_handler.tasks.get(task_id, {})
    
    if task_data.get("conflict_detected", False):
        print("\nConflict detected:")
        print(f"  Conflict ID: {task_data.get('conflict_id', 'N/A')}")
        print(f"  Severity: {task_data.get('conflict_severity', 'N/A')}")
    
    print_separator("Example Complete")

if __name__ == "__main__":
    main()
