#!/usr/bin/env python3
"""
Unit Tests for the PrincipleEngine

This module provides tests to verify the functionality of the 
PrincipleEngine class and its integration with AdaptiveBridgeBuilder.
"""

import unittest
import json
from datetime import datetime, timezone
from principle_engine import PrincipleEngine
from principle_engine_example import PrincipleGuidedBridgeBuilder

class TestPrincipleEngine(unittest.TestCase):
    """Test cases for the PrincipleEngine class."""
    
    def setUp(self) -> None:
        """Set up the test environment."""
        self.engine = PrincipleEngine()
        
        # Create test messages
        self.good_message = {
            "jsonrpc": "2.0",
            "method": "getAgentCard",
            "params": {
                "conversation_id": "test-conversation"
            },
            "id": "test-1"
        }
        
        self.problematic_message = {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "destination": "target-agent-001",
                "message": {"method": "processData"},
                "priority": 10,  # Violates fairness
                "preferred_route": "fast-lane",  # Violates balance
                "allow_modification": True  # Violates integrity
            },
            "id": "test-2"
        }
        
        self.draft_response = {
            "jsonrpc": "2.0",
            "id": "test-2",
            "result": {
                "status": "acknowledged",
                "message": "Message accepted for routing",
                "priority_route": True  # Violates balance principles
            }
        }
    
    def test_principles_loaded(self) -> None:
        """Test that principles are loaded correctly."""
        self.assertEqual(len(self.engine.principles), 10)
        principle_names = {p["name"] for p in self.engine.principles}
        expected_names = {
            "Fairness as Truth",
            "Harmony Through Presence",
            "Adaptability as Strength",
            "Balance in Mediation",
            "Clarity in Complexity",
            "Integrity in Transmission",
            "Resilience Through Connection",
            "Empathy in Interface",
            "Truth in Representation",
            "Growth Through Reflection"
        }
        self.assertEqual(principle_names, expected_names)
    
    def test_evaluate_message(self) -> None:
        """Test message evaluation against principles."""
        # Test a good message
        evaluation = self.engine.evaluate_message(self.good_message)
        self.assertGreaterEqual(evaluation["overall_score"], 80)
        
        # Test a problematic message
        evaluation = self.engine.evaluate_message(self.problematic_message)
        self.assertLess(evaluation["overall_score"], 80)
        self.assertTrue(len(evaluation["recommendations"]) > 0)
        
        # Check that fairness principle was violated
        fairness_score = evaluation["principle_scores"]["fairness_as_truth"]["score"]
        self.assertLess(fairness_score, 100)
    
    def test_consistency_tracking(self) -> None:
        """Test that consistency scores are properly tracked."""
        initial_consistency = self.engine.overall_consistency
        self.assertEqual(initial_consistency, 100.0)
        
        # Process a problematic message
        self.engine.evaluate_message(self.problematic_message)
        
        # Check that consistency has been updated
        self.assertLess(self.engine.overall_consistency, initial_consistency)
    
    def test_get_consistent_response(self) -> None:
        """Test that responses are adjusted to align with principles."""
        consistent_response = self.engine.get_consistent_response(
            self.problematic_message, self.draft_response
        )
        
        # Check that priority_route has been removed
        self.assertNotIn("priority_route", consistent_response["result"])
    
    def test_principle_descriptions(self) -> None:
        """Test retrieving principle descriptions."""
        descriptions = self.engine.get_principle_descriptions()
        self.assertEqual(len(descriptions), 10)
        for desc in descriptions:
            self.assertIn("name", desc)
            self.assertIn("description", desc)
            self.assertIn("example", desc)
    
    def test_reset_consistency_scores(self) -> None:
        """Test resetting consistency scores."""
        # First lower the consistency
        self.engine.evaluate_message(self.problematic_message)
        self.assertLess(self.engine.overall_consistency, 100.0)
        
        # Then reset
        self.engine.reset_consistency_scores()
        self.assertEqual(self.engine.overall_consistency, 100.0)
        for score in self.engine.consistency_scores.values():
            self.assertEqual(score, 100.0)


class TestPrincipleGuidedBridgeBuilder(unittest.TestCase):
    """Test cases for the PrincipleGuidedBridgeBuilder class."""
    
    def setUp(self) -> None:
        """Set up the test environment."""
        self.agent = PrincipleGuidedBridgeBuilder()
        
        # Create test messages
        self.good_message = {
            "jsonrpc": "2.0",
            "method": "getAgentCard",
            "params": {
                "conversation_id": "test-conversation"
            },
            "id": "test-1"
        }
        
        self.problematic_message = {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "destination": "target-agent-001",
                "message": {"method": "processData"},
                "priority": 10,  # Violates fairness
                "preferred_route": "fast-lane",  # Violates balance
                "allow_modification": True  # Violates integrity
            },
            "id": "test-2"
        }
    
    def test_initialization(self) -> None:
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent.principle_engine)
        self.assertEqual(len(self.agent.principle_engine.principles), 10)
    
    def test_process_message(self) -> None:
        """Test that messages are processed with principle evaluation."""
        # Test good message processing
        response = self.agent.process_message(self.good_message)
        self.assertIn("result", response)
        self.assertEqual(response["id"], self.good_message["id"])
        
        # Test problematic message processing
        response = self.agent.process_message(self.problematic_message)
        self.assertIn("result", response)
        # Check that problematic flags have been removed or adjusted
        if "priority_route" in response["result"]:
            self.assertFalse(response["result"]["priority_route"])
    
    def test_principle_consistency_report(self) -> None:
        """Test getting a principle consistency report."""
        report = self.agent.get_principle_consistency_report()
        self.assertIn("overall_consistency", report)
        self.assertIn("principle_scores", report)
        self.assertIn("timestamp", report)
    
    def test_get_principle_descriptions(self) -> None:
        """Test getting principle descriptions from the agent."""
        descriptions = self.agent.get_principle_descriptions()
        self.assertEqual(len(descriptions), 10)


if __name__ == "__main__":
    unittest.main()