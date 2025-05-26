#!/usr/bin/env python3
"""
Test file for the positive reinforcement functionality in the PrincipleEngine.

This test demonstrates how to:
1. Initialize the necessary components
2. Extend the PrincipleEngine with positive reinforcement capabilities
3. Test the functionality with different types of interactions
4. Verify the results
"""

import unittest
import json
from typing import Dict, Any
from datetime import datetime, timezone

from principle_engine import PrincipleEngine
from principle_engine_positive_reinforcement import (
    extend_principle_engine,
    prioritize_positive_reinforcement
)

# Import additional components if available
try:
    from emotional_intelligence import EmotionalIntelligence
    HAS_EMOTIONAL_INTELLIGENCE = True
except ImportError:
    HAS_EMOTIONAL_INTELLIGENCE = False

try:
    from learning_system import LearningSystem
    HAS_LEARNING_SYSTEM = True
except ImportError:
    HAS_LEARNING_SYSTEM = False


class TestPrincipleEnginePositiveReinforcement(unittest.TestCase):
    """Tests for the positive reinforcement functionality in PrincipleEngine."""

    def setUp(self) -> None:
        """Set up the test environment."""
        self.principle_engine = PrincipleEngine()
        
        # Initialize optional components
        if HAS_EMOTIONAL_INTELLIGENCE:
            self.emotional_intelligence = EmotionalIntelligence()
            setattr(self.principle_engine, '_emotional_intelligence', self.emotional_intelligence)
        
        if HAS_LEARNING_SYSTEM:
            self.learning_system = LearningSystem()
            setattr(self.principle_engine, '_learning_system', self.learning_system)
        
        # Extend principle engine with positive reinforcement functionality
        extend_principle_engine(self.principle_engine)
        
        # Create test interaction data
        self.negative_interaction = self._create_test_interaction(
            "I'm frustrated with the lack of progress on this project. Nothing seems to be working right."
        )
        
        self.positive_interaction = self._create_test_interaction(
            "I'm really pleased with how the team has come together. Thanks for your excellent work on this project!"
        )
        
        self.neutral_interaction = self._create_test_interaction(
            "Here is the project status update for the week. We completed 5 tasks and have 3 remaining."
        )

    def _create_test_interaction(self, content: str) -> Dict[str, Any]:
        """Create test interaction data with the given message content."""
        return {
            "id": f"interaction_{hash(content) % 10000}",
            "message": {
                "type": "text",
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "sender": {
                "id": "test_agent",
                "name": "Test Agent",
                "type": "agent"
            },
            "history_summary": {
                "interaction_count": 5,
                "sentiment_trend": 0.0,
                "topic": "project_status"
            }
        }

    def test_direct_function_call(self) -> None:
        """Test calling the function directly."""
        result = prioritize_positive_reinforcement(
            self.negative_interaction,
            "test_agent"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("generative_potential_score", result)
        self.assertIn("suggested_modifications", result)
        self.assertIn("identified_positive_elements", result)

    def test_engine_extension_method(self) -> None:
        """Test calling the function through the extended engine."""
        result = self.principle_engine.prioritize_positive_reinforcement(
            self.negative_interaction,
            "test_agent"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("generative_potential_score", result)
        self.assertIn("suggested_modifications", result)
        self.assertIn("identified_positive_elements", result)

    def test_negative_interaction(self) -> None:
        """Test analysis of a negative interaction."""
        result = self.principle_engine.prioritize_positive_reinforcement(
            self.negative_interaction,
            "test_agent"
        )
        
        # Negative interactions should have:
        # 1. Lower generative potential score
        # 2. More suggested modifications
        # 3. Fewer positive elements
        self.assertLess(result["generative_potential_score"], 0.5)
        self.assertGreater(len(result["suggested_modifications"]), 0)

    def test_positive_interaction(self) -> None:
        """Test analysis of a positive interaction."""
        result = self.principle_engine.prioritize_positive_reinforcement(
            self.positive_interaction,
            "test_agent"
        )
        
        # Positive interactions should have:
        # 1. Higher generative potential score
        # 2. More identified positive elements
        # 3. Fewer or no suggested modifications
        self.assertGreater(result["generative_potential_score"], 0.5)
        self.assertGreater(len(result["identified_positive_elements"]), 0)

    def test_neutral_interaction(self) -> None:
        """Test analysis of a neutral interaction."""
        result = self.principle_engine.prioritize_positive_reinforcement(
            self.neutral_interaction,
            "test_agent"
        )
        
        # Neutral interactions should have:
        # 1. Mid-range generative potential score
        # 2. Some suggested modifications for positive steering
        self.assertGreaterEqual(result["generative_potential_score"], 0)
        self.assertLessEqual(result["generative_potential_score"], 0.7)

    def test_log_contains_analysis_steps(self) -> None:
        """Test that the result log contains the analysis steps."""
        result = self.principle_engine.prioritize_positive_reinforcement(
            self.negative_interaction,
            "test_agent"
        )
        
        self.assertIn("log", result)
        self.assertIn("analysis_steps", result["log"])
        self.assertGreater(len(result["log"]["analysis_steps"]), 0)
        
        # Verify key analysis steps are present
        step_types = [step["step"] for step in result["log"]["analysis_steps"]]
        self.assertIn("emotional_assessment", step_types)
        self.assertIn("positive_element_identification", step_types)
        self.assertIn("generative_potential_calculation", step_types)

    def test_with_empty_message(self) -> None:
        """Test handling of an empty message."""
        empty_interaction = self._create_test_interaction("")
        result = self.principle_engine.prioritize_positive_reinforcement(
            empty_interaction,
            "test_agent"
        )
        
        # Should gracefully handle empty messages
        self.assertEqual(result["generative_potential_score"], 0.0)
        self.assertEqual(len(result["suggested_modifications"]), 0)
        self.assertEqual(len(result["identified_positive_elements"]), 0)
        
        # Check that input validation failed
        validation_steps = [
            step for step in result["log"]["analysis_steps"] 
            if step["step"] == "input_validation"
        ]
        self.assertEqual(len(validation_steps), 1)
        self.assertEqual(validation_steps[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()