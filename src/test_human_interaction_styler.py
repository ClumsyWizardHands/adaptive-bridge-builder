#!/usr/bin/env python3
"""
Unit tests for HumanInteractionStyler module

These tests verify the functionality of the HumanInteractionStyler component,
which builds profiles of human communication preferences and adapts responses
accordingly while maintaining authenticity.
"""

import unittest
import os
import shutil
import tempfile
from typing import Dict, Any, List

from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligence
from human_interaction_styler import (
    HumanInteractionStyler,
    HumanProfile,
    CulturalContext,
    FormalityLevel,
    DetailLevel,
    DirectnessLevel,
    EmotionalTone
)

class TestHumanInteractionStyler(unittest.TestCase):
    """Test case for HumanInteractionStyler functionality."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        # Create a temporary directory for profile storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        self.emotional_intelligence = EmotionalIntelligence()
        self.principle_engine = PrincipleEngine()
        
        # Initialize the styler with test dependencies
        self.styler = HumanInteractionStyler(
            emotional_intelligence=self.emotional_intelligence,
            principle_engine=self.principle_engine,
            profiles_directory=self.temp_dir
        )
        
        # Test data
        self.test_message_formal = "Good morning. I would like to request a detailed analysis of the system architecture, if you would be so kind. Thank you for your assistance. Sincerely, Dr. Johnson"
        self.test_message_casual = "Hey! Can you give me a quick rundown of how this works? Thanks!"
        self.test_message_direct = "What's the performance impact? Give me the numbers right now."
        self.test_message_emotional = "I'm really excited about this project! It's going to be amazing! Can't wait to see it in action!"
        
        self.test_response = """
        The system architecture uses a microservice approach with these components:
        
        1. API Gateway for routing and authentication
        2. Service Registry for discovery
        3. Data Services for business logic
        4. Notification System for messaging
        
        Each service is containerized and uses modern security protocols.
        """
    
    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_profile_creation(self) -> None:
        """Test creating and retrieving human profiles."""
        # Create a profile
        human_id = "test_user"
        profile = self.styler.get_or_create_profile(human_id, "Test User")
        
        # Verify the profile was created correctly
        self.assertEqual(profile.human_id, human_id)
        self.assertEqual(profile.name, "Test User")
        self.assertEqual(profile.interaction_count, 0)
        
        # Save the profile
        self.styler.save_profile(human_id)
        
        # Verify the profile was saved to disk
        profile_path = os.path.join(self.temp_dir, f"{human_id}.json")
        self.assertTrue(os.path.exists(profile_path))
        
        # Create a new styler and load the profile
        new_styler = HumanInteractionStyler(profiles_directory=self.temp_dir)
        loaded_profile = new_styler.get_or_create_profile(human_id)
        
        # Verify the profile was loaded correctly
        self.assertEqual(loaded_profile.human_id, human_id)
        self.assertEqual(loaded_profile.name, "Test User")
    
    def test_formality_detection(self) -> None:
        """Test detection of formality preferences from messages."""
        # Process formal message
        human_id = "formal_user"
        self.styler.update_profile_from_message(self.test_message_formal, human_id)
        formal_profile = self.styler.human_profiles[human_id]
        
        # Process casual message
        human_id = "casual_user"
        self.styler.update_profile_from_message(self.test_message_casual, human_id)
        casual_profile = self.styler.human_profiles[human_id]
        
        # Verify formality detection
        self.assertGreater(formal_profile.communication_style.formality.value, 
                        casual_profile.communication_style.formality.value)
    
    def test_directness_detection(self) -> None:
        """Test detection of directness preferences from messages."""
        # Process direct message
        human_id = "direct_user"
        self.styler.update_profile_from_message(self.test_message_direct, human_id)
        direct_profile = self.styler.human_profiles[human_id]
        
        # Process indirect message (using formal message as example)
        human_id = "indirect_user"
        self.styler.update_profile_from_message(self.test_message_formal, human_id)
        indirect_profile = self.styler.human_profiles[human_id]
        
        # Verify directness detection - direct message should lead to higher directness
        self.assertGreaterEqual(direct_profile.communication_style.directness.value, 
                             indirect_profile.communication_style.directness.value)
    
    def test_emotional_state_detection(self) -> None:
        """Test detection of emotional states from messages."""
        # Detect emotions from emotional message
        emotions = self.styler.detect_emotional_state(self.test_message_emotional)
        
        # Verify emotions were detected
        self.assertTrue(emotions)
        # The primary emotion should be JOY for the excited message
        primary_emotion = max(emotions, key=lambda e: e.confidence)
        self.assertEqual(primary_emotion.category.name, "JOY")
    
    def test_profile_update_from_message(self) -> None:
        """Test updating a profile based on a message."""
        human_id = "test_profile_update"
        
        # Get initial profile
        initial_profile = self.styler.get_or_create_profile(human_id)
        initial_interaction_count = initial_profile.interaction_count
        
        # Update profile with message
        self.styler.update_profile_from_message(self.test_message_emotional, human_id)
        updated_profile = self.styler.human_profiles[human_id]
        
        # Verify profile was updated
        self.assertEqual(updated_profile.interaction_count, initial_interaction_count + 1)
        # Emotional message should lead to positive tone and higher expressiveness
        self.assertGreater(updated_profile.emotional_expressiveness, 0.5)
        self.assertEqual(updated_profile.primary_emotional_tone, EmotionalTone.POSITIVE)
    
    def test_response_adaptation_formality(self) -> None:
        """Test adaptation of responses based on formality."""
        # Create formal profile
        formal_id = "formal_adaptation_test"
        formal_profile = self.styler.get_or_create_profile(formal_id)
        formal_profile.communication_style.formality = FormalityLevel.VERY_FORMAL
        
        # Create casual profile
        casual_id = "casual_adaptation_test"
        casual_profile = self.styler.get_or_create_profile(casual_id)
        casual_profile.communication_style.formality = FormalityLevel.VERY_CASUAL
        
        # Adapt response for formal user
        formal_response = self.styler.adapt_response(self.test_response, formal_id)
        
        # Adapt response for casual user
        casual_response = self.styler.adapt_response(self.test_response, casual_id)
        
        # Verify formal response has formal elements
        self.assertIn("Hello,", formal_response)
        self.assertIn("Regards,", formal_response)
        
        # Verify casual response has casual elements
        self.assertIn("Hey,", casual_response)
        self.assertIn("Cheers!", casual_response)
    
    def test_response_adaptation_detail_level(self) -> None:
        """Test adaptation of responses based on detail level."""
        # Create detailed profile
        detailed_id = "detailed_adaptation_test"
        detailed_profile = self.styler.get_or_create_profile(detailed_id)
        detailed_profile.communication_style.detail_level = DetailLevel.VERY_DETAILED
        
        # Create concise profile
        concise_id = "concise_adaptation_test"
        concise_profile = self.styler.get_or_create_profile(concise_id)
        concise_profile.communication_style.detail_level = DetailLevel.VERY_CONCISE
        
        # Create a response with multiple paragraphs
        detailed_response = """
        The system architecture consists of multiple components:
        
        First, there's the API Gateway that handles routing and authentication.
        It manages all incoming requests and distributes them appropriately.
        
        Second, there's the Service Registry that manages service discovery.
        This component keeps track of all available services and their locations.
        
        Third, we have Data Services that encapsulate business logic.
        These services handle the core functionality of the system.
        
        Finally, there's the Notification System for asynchronous messaging.
        This component ensures that messages are delivered reliably.
        
        In conclusion, this architecture provides scalability and maintainability.
        """
        
        # Adapt response for detailed user
        adapted_detailed = self.styler.adapt_response(detailed_response, detailed_id)
        
        # Adapt response for concise user
        adapted_concise = self.styler.adapt_response(detailed_response, concise_id)
        
        # Verify detailed response contains all paragraphs
        self.assertGreater(len(adapted_detailed), len(adapted_concise))
        
        # Verify concise response is shorter and contains a note about omitted details
        self.assertIn("details have been omitted", adapted_concise)
    
    def test_response_adaptation_directness(self) -> None:
        """Test adaptation of responses based on directness."""
        # Create direct profile
        direct_id = "direct_adaptation_test"
        direct_profile = self.styler.get_or_create_profile(direct_id)
        direct_profile.communication_style.directness = DirectnessLevel.VERY_DIRECT
        
        # Create indirect profile
        indirect_id = "indirect_adaptation_test"
        indirect_profile = self.styler.get_or_create_profile(indirect_id)
        indirect_profile.communication_style.directness = DirectnessLevel.VERY_INDIRECT
        
        # Create a response with a conclusion
        response_with_conclusion = """
        The system performance has been analyzed.
        
        There are several factors affecting the response time:
        1. Database query optimization
        2. Network latency
        3. Cache utilization
        
        In conclusion, we need to focus on optimizing database queries first
        as they are the main bottleneck.
        """
        
        # Adapt response for direct user
        direct_response = self.styler.adapt_response(response_with_conclusion, direct_id)
        
        # Adapt response for indirect user
        indirect_response = self.styler.adapt_response(response_with_conclusion, indirect_id)
        
        # For direct user, conclusion should come first
        first_paragraph_direct = direct_response.strip().split('\n\n')[0]
        self.assertIn("conclusion", first_paragraph_direct)
        
        # For indirect user, should use softening language
        self.assertIn("might consider", indirect_response)
    
    def test_cultural_adaptation(self) -> None:
        """Test adaptation based on cultural context."""
        # Create profile with collectivist culture
        collectivist_id = "collectivist_test"
        collectivist_profile = self.styler.get_or_create_profile(collectivist_id)
        collectivist_profile.cultural_contexts = [CulturalContext.COLLECTIVIST]
        
        # Create profile with individualist culture
        individualist_id = "individualist_test"
        individualist_profile = self.styler.get_or_create_profile(individualist_id)
        individualist_profile.cultural_contexts = [CulturalContext.INDIVIDUALIST]
        
        # Test response with individual focus
        individual_response = "I think you should implement the new system architecture to improve performance."
        
        # Adapt for collectivist
        collectivist_adapted = self.styler.adapt_response(individual_response, collectivist_id)
        
        # Adapt for individualist
        individualist_adapted = self.styler.adapt_response(individual_response, individualist_id)
        
        # Verify collectivist response uses "we" language
        self.assertIn("We might find", collectivist_adapted)
        self.assertIn("We could", collectivist_adapted)
        
        # Verify individualist response maintains "I" and "you" focus
        self.assertIn("I think", individualist_adapted)
        self.assertIn("you should", individualist_adapted)
    
    def test_authenticity_principle(self) -> None:
        """Test the 'Authenticity Beyond Performance' principle."""
        # Create a profile with high confidence and many interactions
        authentic_id = "authenticity_test"
        authentic_profile = self.styler.get_or_create_profile(authentic_id)
        authentic_profile.confidence_level = 0.85
        authentic_profile.interaction_count = 25
        
        # Create a profile with low confidence and few interactions
        new_id = "new_user_test"
        new_profile = self.styler.get_or_create_profile(new_id)
        new_profile.confidence_level = 0.2
        new_profile.interaction_count = 2
        
        # Test response
        test_response = "The system architecture provides scalability and maintainability."
        
        # Adapt for long-term user with high confidence
        authentic_adapted = self.styler.adapt_response(test_response, authentic_id)
        
        # Adapt for new user with low confidence
        new_adapted = self.styler.adapt_response(test_response, new_id)
        
        # Verify authenticity note is present for long-term user
        self.assertIn("adapted my communication style", authentic_adapted)
        self.assertIn("authentic and genuine", authentic_adapted)
        
        # Verify authenticity note is not present for new user
        self.assertNotIn("adapted my communication style", new_adapted)
        self.assertNotIn("authentic and genuine", new_adapted)
    
    def test_profile_serialization(self) -> None:
        """Test profile serialization and deserialization."""
        # Create a profile with various attributes
        human_id = "serialization_test"
        profile = self.styler.get_or_create_profile(human_id)
        profile.communication_style.formality = FormalityLevel.FORMAL
        profile.communication_style.detail_level = DetailLevel.DETAILED
        profile.communication_style.directness = DirectnessLevel.DIRECT
        profile.primary_emotional_tone = EmotionalTone.POSITIVE
        profile.cultural_contexts = [CulturalContext.HIGH_CONTEXT, CulturalContext.COLLECTIVIST]
        profile.emotional_expressiveness = 0.7
        profile.topic_interests = {"technology": 0.9, "business": 0.6}
        profile.interaction_count = 10
        
        # Convert to dictionary
        profile_dict = profile.to_dict()
        
        # Create new profile from dictionary
        new_profile = HumanProfile.from_dict(profile_dict)
        
        # Verify attributes were preserved
        self.assertEqual(new_profile.human_id, human_id)
        self.assertEqual(new_profile.communication_style.formality, FormalityLevel.FORMAL)
        self.assertEqual(new_profile.communication_style.detail_level, DetailLevel.DETAILED)
        self.assertEqual(new_profile.communication_style.directness, DirectnessLevel.DIRECT)
        self.assertEqual(new_profile.primary_emotional_tone, EmotionalTone.POSITIVE)
        self.assertEqual(len(new_profile.cultural_contexts), 2)
        self.assertIn(CulturalContext.HIGH_CONTEXT, new_profile.cultural_contexts)
        self.assertIn(CulturalContext.COLLECTIVIST, new_profile.cultural_contexts)
        self.assertEqual(new_profile.emotional_expressiveness, 0.7)
        self.assertEqual(new_profile.topic_interests["technology"], 0.9)
        self.assertEqual(new_profile.topic_interests["business"], 0.6)
        self.assertEqual(new_profile.interaction_count, 10)

if __name__ == '__main__':
    unittest.main()