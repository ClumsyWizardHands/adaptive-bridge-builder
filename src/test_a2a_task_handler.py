#!/usr/bin/env python3
"""
Tests for A2A Task Handler

This module contains tests for the A2ATaskHandler class to ensure
that it correctly processes tasks from other agents and maintains
conversation context across multiple exchanges.
"""

import unittest
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from a2a_task_handler import (
    A2ATaskHandler,
    MessageIntent,
    ContentType,
    TaskPriority,
    TaskStatus,
    MessageContext
)

class TestA2ATaskHandler(unittest.TestCase):
    """Tests for the A2ATaskHandler class."""
    
    def setUp(self) -> None:
        """Set up test cases."""
        # Create mock components
        self.mock_principle_engine = MagicMock()
        self.mock_principle_engine.evaluate_message.return_value = {
            "score": 0.8,
            "principles": ["principle1", "principle2"]
        }
        
        self.mock_communication_analyzer = MagicMock()
        self.mock_communication_analyzer.analyze_text.return_value = MagicMock()
        self.mock_communication_analyzer.adapt_response.return_value = "Adapted response"
        
        self.mock_relationship_tracker = MagicMock()
        self.mock_relationship_tracker.record_interaction.return_value = None
        
        self.mock_conflict_resolver = MagicMock()
        self.mock_conflict_resolver.detect_conflicts.return_value = []
        
        # Initialize the task handler
        self.task_handler = A2ATaskHandler(
            agent_id="test-agent-001",
            principle_engine=self.mock_principle_engine,
            communication_analyzer=self.mock_communication_analyzer,
            relationship_tracker=self.mock_relationship_tracker,
            conflict_resolver=self.mock_conflict_resolver,
            data_dir="test_data"
        )
        
        # Sample test messages
        self.query_message = {
            "jsonrpc": "2.0",
            "method": "process",
            "params": {
                "content": "Can you tell me about your capabilities?"
            },
            "id": "test-msg-001"
        }
        
        self.instruction_message = {
            "jsonrpc": "2.0",
            "method": "process",
            "params": {
                "content": "Please process this data file."
            },
            "id": "test-msg-002"
        }
        
        self.request_message = {
            "jsonrpc": "2.0",
            "method": "process",
            "params": {
                "content": "Could you analyze this dataset?"
            },
            "id": "test-msg-003"
        }
    
    def test_normalize_message(self) -> None:
        """Test normalizing message to JSON-RPC format."""
        # Test with non-JSON-RPC message
        non_jsonrpc = {
            "action": "query",
            "text": "What can you do?",
            "sender": "test-agent-002"
        }
        
        normalized = self.task_handler._normalize_message(non_jsonrpc, "test-id")
        
        self.assertEqual(normalized["jsonrpc"], "2.0")
        self.assertEqual(normalized["method"], "query")
        self.assertEqual(normalized["id"], "test-id")
        self.assertEqual(normalized["params"]["text"], "What can you do?")
        self.assertEqual(normalized["params"]["sender"], "test-agent-002")
    
    def test_extract_content(self) -> None:
        """Test content extraction and type determination."""
        # Text content
        text_message = {
            "params": {
                "content": "This is a text message"
            }
        }
        content, content_type = self.task_handler._extract_content(text_message)
        self.assertEqual(content, "This is a text message")
        self.assertEqual(content_type, ContentType.TEXT)
        
        # JSON content
        json_message = {
            "params": {
                "content": json.dumps({"key": "value"})
            }
        }
        content, content_type = self.task_handler._extract_content(json_message)
        self.assertEqual(content["key"], "value")
        self.assertEqual(content_type, ContentType.JSON)
    
    def test_determine_intent(self) -> None:
        """Test intent determination from content."""
        # Query intent
        self.assertEqual(
            self.task_handler._determine_intent("Can you tell me about your capabilities?"),
            MessageIntent.QUERY
        )
        
        # Instruct intent
        self.assertEqual(
            self.task_handler._determine_intent("Please process this data file."),
            MessageIntent.INSTRUCT
        )
        
        # Request intent
        self.assertEqual(
            self.task_handler._determine_intent("Could you analyze this dataset?"),
            MessageIntent.REQUEST
        )
    
    def test_handle_task(self) -> None:
        """Test the main handle_task method."""
        # Handle a query task
        query_response = self.task_handler.handle_task(
            message=self.query_message,
            agent_id="test-agent-002",
            conversation_id="test-conv-001"
        )
        
        self.assertEqual(query_response["jsonrpc"], "2.0")
        self.assertEqual(query_response["id"], "test-msg-001")
        self.assertTrue("result" in query_response)
        self.assertEqual(query_response["result"]["status"], "success")
        
        # Verify that the task was tracked
        self.assertTrue("test-msg-001" in self.task_handler.tasks)
        self.assertEqual(
            self.task_handler.tasks["test-msg-001"]["status"],
            TaskStatus.COMPLETED
        )
        
        # Verify that conversation context was updated
        context_key = f"test-agent-002:test-conv-001"
        self.assertTrue(context_key in self.task_handler.active_contexts)
        context = self.task_handler.active_contexts[context_key]
        self.assertEqual(len(context.messages), 1)
        self.assertEqual(context.intent_history[0], MessageIntent.QUERY)
    
    def test_conversation_context(self) -> None:
        """Test that conversation context is maintained across multiple tasks."""
        # First message in conversation
        self.task_handler.handle_task(
            message=self.query_message,
            agent_id="test-agent-002",
            conversation_id="test-conv-002"
        )
        
        # Second message in same conversation
        response = self.task_handler.handle_task(
            message=self.request_message,
            agent_id="test-agent-002",
            conversation_id="test-conv-002"
        )
        
        # Verify that context was maintained
        context_key = f"test-agent-002:test-conv-002"
        context = self.task_handler.active_contexts[context_key]
        self.assertEqual(len(context.messages), 2)
        self.assertEqual(context.intent_history[0], MessageIntent.QUERY)
        self.assertEqual(context.intent_history[1], MessageIntent.REQUEST)
        
        # Verify that context summary is in response
        self.assertEqual(response["result"]["context_summary"]["message_count"], 2)
    
    def test_error_handling(self) -> None:
        """Test error handling in task processing."""
        # Create a message that will cause an error
        error_message = {
            "jsonrpc": "2.0",
            "method": "invalid_method",
            "params": {},
            "id": "test-msg-error"
        }
        
        # Mock a function that raises an exception
        with patch.object(self.task_handler, '_process_task', side_effect=Exception("Test error")):
            response = self.task_handler.handle_task(
                message=error_message,
                agent_id="test-agent-002"
            )
        
        # Verify error response
        self.assertTrue("error" in response)
        self.assertEqual(response["error"]["code"], -32603)  # Internal error
        self.assertEqual(response["error"]["message"], "Test error")
        
        # Verify that task status was updated
        self.assertEqual(
            self.task_handler.tasks["test-msg-error"]["status"],
            TaskStatus.FAILED
        )
        self.assertEqual(
            self.task_handler.tasks["test-msg-error"]["error"],
            "Test error"
        )
    
    def test_principle_evaluation(self) -> None:
        """Test that principle evaluation is applied."""
        response = self.task_handler.handle_task(
            message=self.query_message,
            agent_id="test-agent-002"
        )
        
        # Verify that principle evaluation was called
        self.mock_principle_engine.evaluate_message.assert_called_once()
        
        # Verify that principle alignment is in response
        self.assertTrue("principle_alignment" in response["result"])
        self.assertEqual(response["result"]["principle_alignment"]["score"], 0.8)
    
    def test_conflict_detection(self) -> None:
        """Test that conflicts are detected and recorded."""
        # Mock conflict detection
        conflict_record = MagicMock()
        conflict_record.severity.value = "high"
        conflict_record.conflict_id = "test-conflict-001"
        
        self.mock_conflict_resolver.detect_conflicts.return_value = ["indicator1"]
        self.mock_conflict_resolver.create_conflict_record.return_value = conflict_record
        
        response = self.task_handler.handle_task(
            message=self.query_message,
            agent_id="test-agent-002"
        )
        
        # Verify that conflict detection was called
        self.mock_conflict_resolver.detect_conflicts.assert_called_once()
        self.mock_conflict_resolver.create_conflict_record.assert_called_once()
        
        # Verify that conflict was recorded in task metadata
        self.assertTrue(self.task_handler.tasks["test-msg-001"]["conflict_detected"])
        self.assertEqual(
            self.task_handler.tasks["test-msg-001"]["conflict_id"],
            "test-conflict-001"
        )

if __name__ == "__main__":
    unittest.main()
