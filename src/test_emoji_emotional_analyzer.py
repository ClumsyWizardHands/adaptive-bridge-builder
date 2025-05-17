import unittest
from emoji_emotional_analyzer import (
    EmojiEmotionalAnalyzer,
    EmotionCategory,
    EmotionIntensity,
    CulturalContext,
    ResponseTone
)

class TestEmojiEmotionalAnalyzer(unittest.TestCase):
    """Test cases for the EmojiEmotionalAnalyzer."""
    
    def setUp(self):
        """Setup for each test case."""
        self.analyzer = EmojiEmotionalAnalyzer()
    
    def test_emotion_detection(self):
        """Test detecting emotions in emoji sequences."""
        # Test simple joy emotion
        emotion = self.analyzer.detect_emotion("😀")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.JOY)
        self.assertGreater(emotion.primary_probability, 0.8)
        
        # Test complex pattern
        emotion = self.analyzer.detect_emotion("😊👍")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.CONTENTMENT)
        
        # Test sadness with high intensity
        emotion = self.analyzer.detect_emotion("😭😭😭")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.SADNESS)
        self.assertEqual(emotion.intensity, EmotionIntensity.HIGH)
        
        # Test anger with modifier
        emotion = self.analyzer.detect_emotion("😡❗")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.ANGER)
        self.assertGreater(emotion.primary_probability, 0.9)
        
        # Test neutral/empty
        emotion = self.analyzer.detect_emotion("")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.NEUTRAL)
    
    def test_emotional_mapping(self):
        """Test mapping emoji to emotional state probabilities."""
        # Test mixed emotions
        emotion_map = self.analyzer.map_to_emotional_states("😊😔")
        self.assertIn(EmotionCategory.JOY, emotion_map)
        self.assertIn(EmotionCategory.SADNESS, emotion_map)
        
        # Test related emotions
        emotion_map = self.analyzer.map_to_emotional_states("😀")
        self.assertIn(EmotionCategory.JOY, emotion_map)
        self.assertIn(EmotionCategory.EXCITEMENT, emotion_map)  # Related to joy
    
    def test_emotional_shift_detection(self):
        """Test detecting emotional shifts in conversation."""
        # Setup a conversation with a shift
        self.analyzer.track_conversation("😊")  # User happy
        self.analyzer.track_conversation("👍", is_user_message=False)  # Agent response
        self.analyzer.track_conversation("😔")  # User sad - emotional shift
        
        # Detect the shift
        shift = self.analyzer.detect_emotional_shift()
        self.assertIsNotNone(shift)
        self.assertEqual(shift.from_state.primary_emotion, EmotionCategory.JOY)
        self.assertEqual(shift.to_state.primary_emotion, EmotionCategory.SADNESS)
        self.assertGreater(shift.magnitude, 0.5)
        
        # No shift when emotions are similar
        self.analyzer.track_conversation("😔")  # User still sad
        shift = self.analyzer.detect_emotional_shift()
        self.assertIsNone(shift)  # No significant shift
    
    def test_response_generation(self):
        """Test generating emotionally appropriate responses."""
        # Test response to sadness
        response = self.analyzer.generate_response("😭")
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("SADNESS", response.emotional_intent)
        self.assertIn("SUPPORTIVE", response.emotional_intent)
        
        # Test response to anger with calming tone
        response = self.analyzer.generate_response("😡😡❗")  # High intensity anger
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("ANGER", response.emotional_intent)
        self.assertIn("CALMING", response.emotional_intent)
        
        # Test response to joy
        response = self.analyzer.generate_response("😄")
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("JOY", response.emotional_intent)
        
        # Test specific tone
        response = self.analyzer.generate_response("😔", desired_tone=ResponseTone.ENCOURAGING)
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("ENCOURAGING", response.emotional_intent)
    
    def test_cultural_adaptation(self):
        """Test cultural adaptation in responses."""
        # First get Western response to joy
        response = self.analyzer.generate_response("😄")
        western_response = response.emoji_sequence
        
        # Switch to Eastern Asian context
        self.analyzer.adapt_to_cultural_context(CulturalContext.EASTERN_ASIAN)
        response = self.analyzer.generate_response("😄")
        eastern_response = response.emoji_sequence
        
        # Check if there are cultural adaptations available
        cultural_adaptations = response.cultural_adaptations
        self.assertTrue(any(context in cultural_adaptations for context in 
                          [CulturalContext.EASTERN_ASIAN, CulturalContext.WESTERN]))
        
        # Reset for other tests
        self.analyzer.adapt_to_cultural_context(CulturalContext.GLOBAL)


if __name__ == '__main__':
    unittest.main()
