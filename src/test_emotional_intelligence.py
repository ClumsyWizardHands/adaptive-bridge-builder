#!/usr/bin/env python3
"""
Integration tests for EmotionalIntelligence module with the testing framework.

This module tests the EmotionalIntelligence integration with other components
of the Adaptive Bridge Builder system, particularly focusing on:
1. Integration with PrincipleEngine
2. Integration with CommunicationStyleAnalyzer
3. Testing the "Emotional Distance as Preservation" principle
4. Validating emotional response generation across different scenarios
"""

import unittest
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from principle_engine import PrincipleEngine
from communication_style import CommunicationStyle, EmotionalTone, FormalityLevel
from communication_style_analyzer import CommunicationStyleAnalyzer, MessageDirection, Message, MessageHistory
from emotional_intelligence import (
    EmotionalIntelligence, 
    EmotionCategory, 
    EmotionIntensity,
    InteractionType,
    EmotionSignal,
    EmotionalResponse
)

class TestAgent:
    """Simplified test agent that produces messages with specific emotional content."""
    
    def __init__(self, agent_id: str, emotional_profile: Dict[str, float] = None) -> None:
        """
        Initialize a test agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            emotional_profile: Optional dictionary mapping emotion names to frequencies.
        """
        self.agent_id = agent_id
        self.emotional_profile = emotional_profile or {
            "JOY": 0.2,
            "TRUST": 0.2,
            "FEAR": 0.1,
            "ANGER": 0.1,
            "NEUTRAL": 0.4
        }
        self.message_history: List[str] = []
    
    def generate_message(self, emotion: str, intensity: str, topic: str) -> str:
        """
        Generate a message with specific emotional content.
        
        Args:
            emotion: The primary emotion to express.
            intensity: The intensity level of the emotion.
            topic: The topic to reference in the message.
            
        Returns:
            A message string with the specified emotional content.
        """
        # Template messages for different emotions and intensities
        templates = {
            "JOY": {
                "HIGH": [
                    "I'm absolutely thrilled about {topic}! This is fantastic news!",
                    "I'm so excited about {topic}! This exceeds all our expectations!"
                ],
                "MODERATE": [
                    "I'm happy with how {topic} is progressing. Good work!",
                    "I'm pleased about {topic}. This is working out well."
                ],
                "LOW": [
                    "I'm somewhat glad about {topic}.",
                    "There's some positive progress with {topic}."
                ]
            },
            "TRUST": {
                "HIGH": [
                    "I completely trust your approach to {topic}. You've proven extremely reliable.",
                    "Your handling of {topic} demonstrates exceptional dependability."
                ],
                "MODERATE": [
                    "I believe your assessment of {topic} is accurate.",
                    "Your work on {topic} shows good reliability."
                ],
                "LOW": [
                    "I'm starting to trust your approach to {topic}.",
                    "You seem to be handling {topic} reliably so far."
                ]
            },
            "FEAR": {
                "HIGH": [
                    "I'm extremely worried about {topic}! We need urgent intervention!",
                    "I'm deeply concerned that {topic} could lead to serious problems!"
                ],
                "MODERATE": [
                    "I'm concerned about {topic}. We should address this soon.",
                    "I'm worried that {topic} might cause issues if not addressed."
                ],
                "LOW": [
                    "I'm slightly nervous about {topic}.",
                    "There's some minor concern regarding {topic}."
                ]
            },
            "ANGER": {
                "HIGH": [
                    "I'm furious about {topic}! This is completely unacceptable!",
                    "I'm outraged by how {topic} was handled! This needs immediate correction!"
                ],
                "MODERATE": [
                    "I'm frustrated by {topic}. This isn't what we agreed on.",
                    "I'm annoyed by the approach to {topic}. It needs improvement."
                ],
                "LOW": [
                    "I'm a bit bothered by {topic}.",
                    "The handling of {topic} is somewhat irritating."
                ]
            },
            "NEUTRAL": {
                "MODERATE": [
                    "I'd like to discuss {topic} at your convenience.",
                    "Let's review {topic} when you have time.",
                    "I'm following up on {topic}."
                ]
            }
        }
        
        # Default to neutral if emotion not found
        if emotion not in templates:
            emotion = "NEUTRAL"
            intensity = "MODERATE"
        
        # Default to moderate if intensity not found
        if intensity not in templates[emotion]:
            intensity = "MODERATE"
        
        # Select a template randomly (for this test, we'll just use the first one)
        template = templates[emotion][intensity][0]
        
        # Fill in the template
        message = template.format(topic=topic)
        
        # Add to message history
        self.message_history.append(message)
        
        return message


class TestEmotionalIntelligence(unittest.TestCase):
    """Test cases for EmotionalIntelligence module integration."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        self.principle_engine = PrincipleEngine()
        self.style_analyzer = CommunicationStyleAnalyzer(principle_engine=self.principle_engine)
        self.emotional_intelligence = EmotionalIntelligence(principle_engine=self.principle_engine)
        
        # Create test agents with different emotional profiles
        self.joyful_agent = TestAgent("agent-joyful", {
            "JOY": 0.6, "TRUST": 0.2, "NEUTRAL": 0.2
        })
        
        self.angry_agent = TestAgent("agent-angry", {
            "ANGER": 0.5, "FEAR": 0.3, "NEUTRAL": 0.2
        })
        
        self.neutral_agent = TestAgent("agent-neutral", {
            "NEUTRAL": 0.8, "JOY": 0.1, "TRUST": 0.1
        })
    
    def test_emotion_detection_accuracy(self) -> None:
        """Test that emotions are correctly detected in messages."""
        # Test joy detection
        joy_message = self.joyful_agent.generate_message("JOY", "HIGH", "the new project results")
        joy_emotions = self.emotional_intelligence.detect_emotions(joy_message)
        
        # Verify that joy is detected
        self.assertTrue(any(e.category == EmotionCategory.JOY for e in joy_emotions))
        
        # Test anger detection
        anger_message = self.angry_agent.generate_message("ANGER", "HIGH", "the system failure")
        anger_emotions = self.emotional_intelligence.detect_emotions(anger_message)
        
        # Verify that anger is detected
        self.assertTrue(any(e.category == EmotionCategory.ANGER for e in anger_emotions))
        
        # Test neutral detection
        neutral_message = self.neutral_agent.generate_message("NEUTRAL", "MODERATE", "the project timeline")
        neutral_emotions = self.emotional_intelligence.detect_emotions(neutral_message)
        
        # Verify that neutral is detected
        self.assertTrue(any(e.category == EmotionCategory.NEUTRAL for e in neutral_emotions))
    
    def test_interaction_type_detection(self) -> None:
        """Test that interaction types are correctly categorized."""
        # Test conflict detection
        conflict_message = "I strongly disagree with your approach to this problem. Your solution is flawed."
        self.assertEqual(
            self.emotional_intelligence.detect_interaction_type(conflict_message),
            InteractionType.CONFLICT
        )
        
        # Test crisis detection
        crisis_message = "URGENT: The system is down and customers are affected. We need immediate assistance!"
        self.assertEqual(
            self.emotional_intelligence.detect_interaction_type(crisis_message),
            InteractionType.CRISIS
        )
        
        # Test celebration detection
        celebration_message = "Congratulations on the successful launch! This is a major achievement."
        self.assertEqual(
            self.emotional_intelligence.detect_interaction_type(celebration_message),
            InteractionType.CELEBRATION
        )
        
        # Test routine detection (default when no specific patterns match)
        routine_message = "I'm sending the updated documents as requested."
        self.assertEqual(
            self.emotional_intelligence.detect_interaction_type(routine_message),
            InteractionType.ROUTINE
        )
    
    def test_emotional_profile_building(self) -> None:
        """Test that emotional profiles are built correctly over multiple interactions."""
        agent_id = "profile-test-agent"
        
        # Messages with different emotions
        messages = [
            self.joyful_agent.generate_message("JOY", "HIGH", "our collaboration"),
            self.joyful_agent.generate_message("TRUST", "MODERATE", "your expertise"),
            self.angry_agent.generate_message("ANGER", "MODERATE", "the delay"),
            self.neutral_agent.generate_message("NEUTRAL", "MODERATE", "the next steps")
        ]
        
        # Process messages and build profile
        for message in messages:
            emotions = self.emotional_intelligence.detect_emotions(message)
            self.emotional_intelligence.update_emotional_profile(agent_id, emotions)
        
        # Check that the profile exists
        self.assertIn(agent_id, self.emotional_intelligence.emotion_profiles)
        
        # Check that multiple emotions are represented in the profile
        profile = self.emotional_intelligence.emotion_profiles[agent_id]
        self.assertTrue(len(profile.primary_emotions) >= 3)
        
        # Check sample count
        self.assertEqual(profile.sample_count, 4)
    
    def test_emotional_distance_principle(self) -> None:
        """Test that the 'Emotional Distance as Preservation' principle is applied correctly."""
        # Generate a high-intensity anger message
        anger_message = self.angry_agent.generate_message("ANGER", "HIGH", "the project failure")
        
        # Detect emotions and interaction type
        emotions = self.emotional_intelligence.detect_emotions(anger_message)
        interaction_type = self.emotional_intelligence.detect_interaction_type(anger_message)
        
        # Generate response for conflict interaction
        response = self.emotional_intelligence.get_appropriate_response(
            message=anger_message,
            detected_emotions=emotions,
            interaction_type=InteractionType.CONFLICT,
            agent_id=self.angry_agent.agent_id
        )
        
        # Verify that the response applies emotional distance
        self.assertIn("emotional distance", response.content_template)
        
        # Verify that the response uses formal style
        self.assertEqual(response.expression_style["formality"], FormalityLevel.FORMAL.name)
        
        # Verify that the response uses neutral tone
        self.assertEqual(response.expression_style["emotional_tone"], EmotionalTone.NEUTRAL.name)
    
    def test_integration_with_communication_style(self) -> None:
        """Test integration between EmotionalIntelligence and CommunicationStyleAnalyzer."""
        agent_id = "integration-test-agent"
        
        # Create a message history
        history = MessageHistory(agent_id=agent_id)
        
        # Add some received messages with varied emotional content
        messages = [
            self.joyful_agent.generate_message("JOY", "HIGH", "our progress"),
            self.joyful_agent.generate_message("TRUST", "MODERATE", "your work"),
            self.angry_agent.generate_message("ANGER", "LOW", "the minor issue")
        ]
        
        for content in messages:
            msg = Message(
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                direction=MessageDirection.RECEIVED
            )
            history.add_message(msg)
        
        # Analyze communication style
        style = self.style_analyzer.analyze_message_history(history)
        
        # Process emotions in these messages
        for message in messages:
            emotions = self.emotional_intelligence.detect_emotions(message)
            self.emotional_intelligence.update_emotional_profile(agent_id, emotions)
        
        # Get emotional profile
        emotional_profile = self.emotional_intelligence.emotion_profiles[agent_id]
        
        # Verify that emotional tone in communication style is consistent with emotional profile
        if EmotionCategory.JOY in emotional_profile.primary_emotions:
            joy_frequency = emotional_profile.primary_emotions[EmotionCategory.JOY]
            if joy_frequency > 0.3:
                self.assertIn(style.emotional_tone.name, ["POSITIVE", "VERY_POSITIVE"])
        
        if EmotionCategory.ANGER in emotional_profile.primary_emotions:
            anger_frequency = emotional_profile.primary_emotions[EmotionCategory.ANGER]
            if anger_frequency > 0.3:
                self.assertIn(style.emotional_tone.name, ["NEGATIVE", "VERY_NEGATIVE"])
    
    def test_principled_emotional_responses(self) -> None:
        """Test that emotional responses align with principles."""
        # Generate a crisis message
        crisis_message = "URGENT: Our system has been compromised! Customer data may be at risk!"
        
        # Detect emotions and interaction type
        emotions = self.emotional_intelligence.detect_emotions(crisis_message)
        
        # Generate response for crisis interaction
        response = self.emotional_intelligence.get_appropriate_response(
            message=crisis_message,
            detected_emotions=emotions,
            interaction_type=InteractionType.CRISIS,
            agent_id="crisis-agent"
        )
        
        # Define test context variables
        context = {
            "crisis_topic": "the security breach",
            "action_items": "1) Isolate affected systems, 2) Assess data impact, 3) Implement containment measures"
        }
        
        # Format the response
        formatted_response = self.emotional_intelligence.format_response(response, context)
        
        # Verify that the response includes principle-aligned content
        self.assertTrue(
            any(principle in formatted_response for principle in 
                ["principle", "Adaptability", "fairness", "harmony"])
        )


class TestEmotionalResponseScenarios(unittest.TestCase):
    """Test specific response scenarios for different emotional contexts."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        self.principle_engine = PrincipleEngine()
        self.emotional_intelligence = EmotionalIntelligence(principle_engine=self.principle_engine)
    
    def test_response_to_joy(self) -> None:
        """Test appropriate responses to joyful messages."""
        joy_message = "I'm thrilled about our new partnership! This is going to be amazing!"
        
        emotions = self.emotional_intelligence.detect_emotions(joy_message)
        interaction_type = self.emotional_intelligence.detect_interaction_type(joy_message)
        
        response = self.emotional_intelligence.get_appropriate_response(
            message=joy_message,
            detected_emotions=emotions,
            interaction_type=interaction_type,
            agent_id="joy-test-agent"
        )
        
        # Format the response
        context = {"positive_event": "our new partnership", "principle": "Harmony Through Presence"}
        formatted_response = self.emotional_intelligence.format_response(response, context)
        
        # Verify positive tone
        self.assertEqual(response.expression_style["emotional_tone"], EmotionalTone.POSITIVE.name)
        
        # Verify appropriate content
        self.assertIn("our new partnership", formatted_response)
    
    def test_response_to_anger_in_conflict(self) -> None:
        """Test appropriate responses to angry messages in conflict situations."""
        anger_message = "Your team completely messed up this implementation! This is totally unacceptable!"
        
        emotions = self.emotional_intelligence.detect_emotions(anger_message)
        
        response = self.emotional_intelligence.get_appropriate_response(
            message=anger_message,
            detected_emotions=emotions,
            interaction_type=InteractionType.CONFLICT,
            agent_id="anger-test-agent"
        )
        
        # Format the response
        context = {"anger_topic": "the implementation issues"}
        formatted_response = self.emotional_intelligence.format_response(response, context)
        
        # Verify emotional distance
        self.assertIn("emotional distance", formatted_response)
        
        # Verify neutral tone
        self.assertEqual(response.expression_style["emotional_tone"], EmotionalTone.NEUTRAL.name)
        
        # Verify formal style
        self.assertEqual(response.expression_style["formality"], FormalityLevel.FORMAL.name)
    
    def test_response_to_fear_in_crisis(self) -> None:
        """Test appropriate responses to fearful messages in crisis situations."""
        fear_message = "I'm extremely worried about the security breach! Our customer data might be exposed!"
        
        emotions = self.emotional_intelligence.detect_emotions(fear_message)
        
        response = self.emotional_intelligence.get_appropriate_response(
            message=fear_message,
            detected_emotions=emotions,
            interaction_type=InteractionType.CRISIS,
            agent_id="fear-test-agent"
        )
        
        # Format the response
        context = {
            "crisis_topic": "the security breach",
            "action_items": "1) Isolate affected systems, 2) Assess data impact, 3) Implement containment"
        }
        formatted_response = self.emotional_intelligence.format_response(response, context)
        
        # Verify action-focused response
        self.assertIn("action", formatted_response.lower())
        
        # Verify contains action items
        self.assertIn("Isolate affected systems", formatted_response)
    
    def test_response_to_sensitive_topic(self) -> None:
        """Test appropriate responses to sensitive topics."""
        sensitive_message = "We need to discuss the upcoming team restructuring and potential role changes."
        
        emotions = self.emotional_intelligence.detect_emotions(sensitive_message)
        
        response = self.emotional_intelligence.get_appropriate_response(
            message=sensitive_message,
            detected_emotions=emotions,
            interaction_type=InteractionType.SENSITIVE,
            agent_id="sensitive-test-agent"
        )
        
        # Format the response
        context = {"sensitive_topic": "the team restructuring"}
        formatted_response = self.emotional_intelligence.format_response(response, context)
        
        # Verify appropriate handling
        self.assertTrue(
            any(term in formatted_response.lower() for term in 
                ["sensitive", "discretion", "appropriate", "understand"])
        )
        
        # Verify formal style
        self.assertEqual(response.expression_style["formality"], FormalityLevel.FORMAL.name)


if __name__ == "__main__":
    unittest.main()