import unittest
from unittest.mock import MagicMock, patch
import json
import logging
from typing import Any, Dict
import emoji
from emoji_emotional_analyzer import (
    EmojiEmotionalAnalyzer,
    EmotionCategory,
    EmotionIntensity,
    CulturalContext,
    ResponseTone,
    EmotionDetectionEngine,
    EmotionMappingSystem,
    EmotionalShiftTracker,
    EmojiResponseGenerator,
    EmotionalState,
    EmotionalShift,
    EmojiEmotionalResponse
)

class TestEmotionDetectionEngine(unittest.TestCase):
    """Test suite focused on the EmotionDetectionEngine component."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        # Create a mock emoji knowledge base
        self.mock_emoji_kb = MagicMock()
        self.engine = EmotionDetectionEngine(self.mock_emoji_kb)
    
    def test_initialize_emotion_mappings(self) -> None:
        """Test that emotion mappings are properly initialized."""
        self.assertIsNotNone(self.engine.primary_emotion_mappings)
        self.assertIsNotNone(self.engine.modifier_mappings)
        self.assertIsNotNone(self.engine.complex_patterns)
        
        # Verify essential mappings exist
        self.assertIn("😀", self.engine.primary_emotion_mappings)
        self.assertIn("😔", self.engine.primary_emotion_mappings)
        self.assertIn("❗", self.engine.modifier_mappings)
    
    def test_extract_emojis(self) -> None:
        """Test extracting emojis from mixed text."""
        # Test with pure emojis
        emojis = self.engine._extract_emojis("😀😔👍")
        self.assertEqual(len(emojis), 3)
        self.assertIn("😀", emojis)
        self.assertIn("😔", emojis)
        self.assertIn("👍", emojis)
        
        # Test with mixed text and emojis
        emojis = self.engine._extract_emojis("Hello 😀 World! 👍")
        self.assertEqual(len(emojis), 2)
        self.assertIn("😀", emojis)
        self.assertIn("👍", emojis)
        
        # Test with no emojis
        emojis = self.engine._extract_emojis("Hello World!")
        self.assertEqual(len(emojis), 0)
    
    def test_detect_emotion_with_empty_input(self) -> None:
        """Test emotion detection with empty input."""
        state = self.engine.detect_emotion("")
        self.assertEqual(state.primary_emotion, EmotionCategory.NEUTRAL)
        self.assertEqual(state.primary_probability, 1.0)
        self.assertEqual(state.confidence, 0.5)
        self.assertIsNone(state.secondary_emotion)
    
    def test_detect_primary_emotion(self) -> None:
        """Test detecting a primary emotion from an emoji sequence."""
        # Test joy
        state = self.engine.detect_emotion("😀")
        self.assertEqual(state.primary_emotion, EmotionCategory.JOY)
        self.assertGreater(state.primary_probability, 0.8)
        
        # Test sadness
        state = self.engine.detect_emotion("😢")
        self.assertEqual(state.primary_emotion, EmotionCategory.SADNESS)
        self.assertGreater(state.primary_probability, 0.8)
        
        # Test anger
        state = self.engine.detect_emotion("😡")
        self.assertEqual(state.primary_emotion, EmotionCategory.ANGER)
        self.assertGreater(state.primary_probability, 0.8)
    
    def test_detect_emotion_with_modifiers(self) -> None:
        """Test emotion detection with intensity modifiers."""
        # Base emotion
        base_state = self.engine.detect_emotion("😀")
        
        # With intensifier
        intensified_state = self.engine.detect_emotion("😀❗")
        self.assertEqual(base_state.primary_emotion, intensified_state.primary_emotion)
        self.assertGreater(intensified_state.primary_probability, base_state.primary_probability)
        
        # With double intensifier
        double_intensified_state = self.engine.detect_emotion("😀‼️")
        self.assertEqual(base_state.primary_emotion, double_intensified_state.primary_emotion)
        self.assertGreater(double_intensified_state.primary_probability, intensified_state.primary_probability)
    
    def test_detect_complex_emotion_patterns(self) -> None:
        """Test detection of complex emotion patterns."""
        # Test the complex patterns defined in the engine
        for pattern, emotion, _ in self.engine.complex_patterns:
            pattern_str = ''.join(pattern)
            state = self.engine.detect_emotion(pattern_str)
            self.assertEqual(state.primary_emotion, emotion)
    
    def test_calculate_intensity(self) -> None:
        """Test calculation of emotional intensity from emojis."""
        # Test with single emoji (baseline)
        emojis = self.engine._extract_emojis("😀")
        intensity = self.engine._calculate_intensity(emojis)
        self.assertEqual(intensity, EmotionIntensity.MEDIUM)
        
        # Test with repeated emojis (higher intensity)
        emojis = self.engine._extract_emojis("😀😀😀")
        intensity = self.engine._calculate_intensity(emojis)
        self.assertIn(intensity, [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH])
        
        # Test with emphasis
        emojis = self.engine._extract_emojis("😀❗")
        intensity = self.engine._calculate_intensity(emojis)
        self.assertGreaterEqual(intensity, EmotionIntensity.MEDIUM)
        
        # Test with strong emojis
        emojis = self.engine._extract_emojis("😡")
        intensity = self.engine._calculate_intensity(emojis)
        self.assertGreaterEqual(intensity, EmotionIntensity.MEDIUM)
    
    def test_calculate_confidence(self) -> None:
        """Test calculation of confidence scores for emotion detection."""
        # Test with single emotion (high confidence)
        scores = {EmotionCategory.JOY: 1.0}
        confidence = self.engine._calculate_confidence(scores, 1.0)
        self.assertGreaterEqual(confidence, 0.8)
        
        # Test with mixed emotions (lower confidence)
        scores = {
            EmotionCategory.JOY: 0.6,
            EmotionCategory.EXCITEMENT: 0.4
        }
        confidence = self.engine._calculate_confidence(scores, 0.6)
        self.assertLess(confidence, 0.9)


class TestEmotionMappingSystem(unittest.TestCase):
    """Test suite focused on the EmotionMappingSystem component."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        # Create a mock emoji knowledge base
        self.mock_emoji_kb = MagicMock()
        self.mapping_system = EmotionMappingSystem(self.mock_emoji_kb)
    
    def test_map_to_emotional_states_simple(self) -> None:
        """Test mapping a simple emoji to multiple emotional states."""
        # Test with a joy emoji
        emotion_map = self.mapping_system.map_to_emotional_states("😀")
        self.assertIn(EmotionCategory.JOY, emotion_map)
        self.assertGreater(emotion_map[EmotionCategory.JOY], 0.5)
        
        # Check that related emotions are included
        self.assertIn(EmotionCategory.EXCITEMENT, emotion_map)
        self.assertIn(EmotionCategory.CONTENTMENT, emotion_map)
    
    def test_map_to_emotional_states_complex(self) -> None:
        """Test mapping complex emoji sequences to emotional states."""
        # Test with mixed emotions
        emotion_map = self.mapping_system.map_to_emotional_states("😀😔")
        self.assertIn(EmotionCategory.JOY, emotion_map)
        self.assertIn(EmotionCategory.SADNESS, emotion_map)
        
        # Check normalization (probabilities should sum to 1)
        total_probability = sum(emotion_map.values())
        self.assertAlmostEqual(total_probability, 1.0, places=1)
    
    def test_get_related_emotions(self) -> None:
        """Test retrieval of related emotions."""
        # Test with known emotion
        related = self.mapping_system._get_related_emotions(EmotionCategory.JOY)
        self.assertIn(EmotionCategory.EXCITEMENT, related)
        self.assertIn(EmotionCategory.CONTENTMENT, related)
        
        # Test with emotion that has no predefined relations
        related = self.mapping_system._get_related_emotions(EmotionCategory.NEUTRAL)
        self.assertEqual(len(related), 0)
        
        # Verify relatedness scores are between 0 and 1
        for emotion, score in related.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestEmotionalShiftTracker(unittest.TestCase):
    """Test suite focused on the EmotionalShiftTracker component."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        self.tracker = EmotionalShiftTracker()
    
    def test_track_conversation(self) -> None:
        """Test tracking conversation messages."""
        # Track a message
        self.tracker.track_conversation("😀")
        
        # Verify it was added to history
        self.assertEqual(len(self.tracker.conversation_history), 1)
        self.assertEqual(len(self.tracker.emotion_history), 1)
        
        # Check record details
        message_record = self.tracker.conversation_history[0]
        self.assertEqual(message_record["sequence"], "😀")
        self.assertTrue(message_record["is_user_message"])
        
        emotion_record = self.tracker.emotion_history[0]
        self.assertEqual(emotion_record["state"].primary_emotion, EmotionCategory.JOY)
        self.assertTrue(emotion_record["is_user_emotion"])
        
        # Track agent message
        self.tracker.track_conversation("👍", is_user_message=False)
        self.assertEqual(len(self.tracker.conversation_history), 2)
        self.assertEqual(len(self.tracker.emotion_history), 2)
        self.assertFalse(self.tracker.conversation_history[1]["is_user_message"])
    
    def test_detect_emotional_shift_insufficient_data(self) -> None:
        """Test shift detection with insufficient data."""
        # No messages
        shift = self.tracker.detect_emotional_shift()
        self.assertIsNone(shift)
        
        # Only one user message
        self.tracker.track_conversation("😀")
        shift = self.tracker.detect_emotional_shift()
        self.assertIsNone(shift)
        
        # One user and one agent message
        self.tracker.track_conversation("👍", is_user_message=False)
        shift = self.tracker.detect_emotional_shift()
        self.assertIsNone(shift)
    
    def test_detect_emotional_shift_significant(self) -> None:
        """Test detection of significant emotional shifts."""
        # Setup a conversation with a significant shift
        self.tracker.track_conversation("😀")  # User happy
        self.tracker.track_conversation("👍", is_user_message=False)  # Agent responds
        self.tracker.track_conversation("😭")  # User now sad - significant shift
        
        # Detect the shift
        shift = self.tracker.detect_emotional_shift()
        self.assertIsNotNone(shift)
        self.assertEqual(shift.from_state.primary_emotion, EmotionCategory.JOY)
        self.assertEqual(shift.to_state.primary_emotion, EmotionCategory.SADNESS)
        self.assertGreater(shift.magnitude, 0.3)  # Threshold for significant shifts
    
    def test_detect_emotional_shift_not_significant(self) -> None:
        """Test detection with non-significant emotional shifts."""
        # Setup a conversation with subtle changes
        self.tracker.track_conversation("😀")  # User happy
        self.tracker.track_conversation("👍", is_user_message=False)  # Agent responds
        self.tracker.track_conversation("😊")  # User still happy - no significant shift
        
        # Detect the shift
        shift = self.tracker.detect_emotional_shift()
        self.assertIsNone(shift)
    
    def test_calculate_shift_magnitude(self) -> None:
        """Test calculation of emotional shift magnitude."""
        # Create emotional states for testing
        joy_low = EmotionalState(
            primary_emotion=EmotionCategory.JOY,
            primary_probability=0.8,
            intensity=EmotionIntensity.LOW,
            confidence=0.9
        )
        
        joy_high = EmotionalState(
            primary_emotion=EmotionCategory.JOY,
            primary_probability=0.9,
            intensity=EmotionIntensity.HIGH,
            confidence=0.9
        )
        
        sadness = EmotionalState(
            primary_emotion=EmotionCategory.SADNESS,
            primary_probability=0.8,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.8
        )
        
        # Test intensity change (same emotion)
        magnitude = self.tracker._calculate_shift_magnitude(joy_low, joy_high)
        self.assertGreater(magnitude, 0)
        self.assertLess(magnitude, 0.5)  # Intensity changes have lower magnitude
        
        # Test emotion change (different emotions)
        magnitude = self.tracker._calculate_shift_magnitude(joy_high, sadness)
        self.assertGreater(magnitude, 0.5)  # Different emotions have higher magnitude
    
    def test_detect_shift_trigger(self) -> None:
        """Test detection of emotional shift triggers."""
        # Create emotional states for testing
        neutral = EmotionalState(
            primary_emotion=EmotionCategory.NEUTRAL,
            primary_probability=0.9,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.9
        )
        
        joy = EmotionalState(
            primary_emotion=EmotionCategory.JOY,
            primary_probability=0.9,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.9
        )
        
        # Test known pattern
        trigger = self.tracker._detect_shift_trigger(neutral, joy)
        self.assertEqual(trigger, "positive_development")
        
        # Test unknown pattern
        custom = EmotionalState(
            primary_emotion=EmotionCategory.CURIOSITY,
            primary_probability=0.9,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.9
        )
        trigger = self.tracker._detect_shift_trigger(joy, custom)
        self.assertEqual(trigger, "unknown_trigger")
    
    def test_determine_temporal_pattern(self) -> Dict[str, Any]:
        """Test determination of temporal patterns in emotional changes."""
        # Not enough data
        pattern = self.tracker._determine_temporal_pattern([])
        self.assertEqual(pattern, "insufficient_data")
        
        # Create mock emotion records for testing
        def create_mock_record(intensity) -> Dict[str, Any]:
            state = EmotionalState(
                primary_emotion=EmotionCategory.JOY,
                primary_probability=0.9,
                intensity=intensity,
                confidence=0.9
            )
            return {"state": state}
        
        # Test steadily increasing
        records = [
            create_mock_record(EmotionIntensity.LOW),
            create_mock_record(EmotionIntensity.MEDIUM),
            create_mock_record(EmotionIntensity.HIGH)
        ]
        pattern = self.tracker._determine_temporal_pattern(records)
        self.assertEqual(pattern, "steadily_increasing")
        
        # Test steadily decreasing
        records = [
            create_mock_record(EmotionIntensity.HIGH),
            create_mock_record(EmotionIntensity.MEDIUM),
            create_mock_record(EmotionIntensity.LOW)
        ]
        pattern = self.tracker._determine_temporal_pattern(records)
        self.assertEqual(pattern, "steadily_decreasing")
        
        # Test fluctuating
        records = [
            create_mock_record(EmotionIntensity.LOW),
            create_mock_record(EmotionIntensity.HIGH),
            create_mock_record(EmotionIntensity.MEDIUM)
        ]
        pattern = self.tracker._determine_temporal_pattern(records)
        self.assertEqual(pattern, "fluctuating")


class TestEmojiResponseGenerator(unittest.TestCase):
    """Test suite focused on the EmojiResponseGenerator component."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        self.mock_principle_engine = MagicMock()
        # Mock principle alignment to always return 0.9
        self.mock_principle_engine.return_value = 0.9
        
        self.generator = EmojiResponseGenerator(
            principle_engine=self.mock_principle_engine,
            cultural_context=CulturalContext.GLOBAL
        )
    
    def test_initialize_response_templates(self) -> None:
        """Test that response templates are properly initialized."""
        self.assertIn(EmotionCategory.JOY, self.generator.response_templates)
        self.assertIn(EmotionCategory.SADNESS, self.generator.response_templates)
        
        # Check specific tones exist
        self.assertIn(ResponseTone.MATCHING, self.generator.response_templates[EmotionCategory.JOY])
        self.assertIn(ResponseTone.SUPPORTIVE, self.generator.response_templates[EmotionCategory.SADNESS])
    
    def test_generate_response_basic(self) -> None:
        """Test basic response generation."""
        # Generate response to happy emoji
        response = self.generator.generate_response("😀")
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("JOY", response.emotional_intent)
        self.assertIn("MATCHING", response.emotional_intent)
        self.assertGreaterEqual(response.principle_alignment_score, 0.7)
        
        # Generate response to sad emoji
        response = self.generator.generate_response("😢")
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("SADNESS", response.emotional_intent)
        self.assertIn("SUPPORTIVE", response.emotional_intent)
    
    def test_generate_response_with_specified_tone(self) -> None:
        """Test response generation with specified tone."""
        # Generate supportive response to happy emoji
        response = self.generator.generate_response("😀", desired_tone=ResponseTone.SUPPORTIVE)
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("JOY", response.emotional_intent)
        self.assertIn("SUPPORTIVE", response.emotional_intent)
        
        # Generate encouraging response to sad emoji
        response = self.generator.generate_response("😢", desired_tone=ResponseTone.ENCOURAGING)
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("SADNESS", response.emotional_intent)
        self.assertIn("ENCOURAGING", response.emotional_intent)
    
    def test_determine_response_tone(self) -> None:
        """Test determination of appropriate response tone."""
        # Create emotional states for testing
        joy_medium = EmotionalState(
            primary_emotion=EmotionCategory.JOY,
            primary_probability=0.9,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.9
        )
        
        anger_high = EmotionalState(
            primary_emotion=EmotionCategory.ANGER,
            primary_probability=0.9,
            intensity=EmotionIntensity.HIGH,
            confidence=0.8
        )
        
        sadness_low = EmotionalState(
            primary_emotion=EmotionCategory.SADNESS,
            primary_probability=0.8,
            intensity=EmotionIntensity.LOW,
            confidence=0.8
        )
        
        # Test joy response
        tone = self.generator._determine_response_tone(joy_medium)
        self.assertEqual(tone, ResponseTone.MATCHING)
        
        # Test high anger response (should be calming)
        tone = self.generator._determine_response_tone(anger_high)
        self.assertEqual(tone, ResponseTone.CALMING)
        
        # Test sadness response
        tone = self.generator._determine_response_tone(sadness_low)
        self.assertEqual(tone, ResponseTone.SUPPORTIVE)
    
    def test_cultural_adaptations(self) -> None:
        """Test creation of culturally adapted responses."""
        # Test global context (default)
        adaptations = self.generator._create_cultural_adaptations(
            EmotionCategory.JOY, ResponseTone.MATCHING
        )
        # May or may not have adaptations in global context
        
        # Change to specific cultural context
        self.generator.cultural_context = CulturalContext.EASTERN_ASIAN
        adaptations = self.generator._create_cultural_adaptations(
            EmotionCategory.JOY, ResponseTone.MATCHING
        )
        # Should have adaptations for this specific context if defined in templates
        if CulturalContext.EASTERN_ASIAN in self.generator.cultural_variations:
            if EmotionCategory.JOY in self.generator.cultural_variations[CulturalContext.EASTERN_ASIAN]:
                if ResponseTone.MATCHING in self.generator.cultural_variations[CulturalContext.EASTERN_ASIAN][EmotionCategory.JOY]:
                    self.assertGreaterEqual(len(adaptations), 1)
    
    @patch.object(EmojiResponseGenerator, '_check_principle_alignment')
    def test_generate_alternatives(self, mock_check_alignment) -> None:
        """Test generation of alternative responses."""
        # Mock alignment scores for alternatives
        mock_check_alignment.return_value = 0.85
        
        # Generate alternatives for joy
        alternatives = self.generator._generate_alternatives(
            EmotionCategory.JOY, ResponseTone.MATCHING
        )
        
        # Should have some alternatives if there are other tones defined
        other_tones_exist = False
        for tone in self.generator.response_templates.get(EmotionCategory.JOY, {}).keys():
            if tone != ResponseTone.MATCHING:
                other_tones_exist = True
                break
        
        if other_tones_exist:
            self.assertGreaterEqual(len(alternatives), 1)
            # Verify structure of alternatives
            for alt in alternatives:
                self.assertEqual(len(alt), 3)  # (sequence, intent, alignment_score)
                self.assertIsInstance(alt[0], str)  # sequence
                self.assertIsInstance(alt[1], str)  # intent
                self.assertIsInstance(alt[2], float)  # alignment_score


class TestEmojiEmotionalAnalyzer(unittest.TestCase):
    """Test suite focused on the main EmojiEmotionalAnalyzer component."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        self.analyzer = EmojiEmotionalAnalyzer()
    
    def test_component_initialization(self) -> None:
        """Test that all subcomponents are properly initialized."""
        self.assertIsInstance(self.analyzer.detection_engine, EmotionDetectionEngine)
        self.assertIsInstance(self.analyzer.mapping_system, EmotionMappingSystem)
        self.assertIsInstance(self.analyzer.shift_tracker, EmotionalShiftTracker)
        self.assertIsInstance(self.analyzer.response_generator, EmojiResponseGenerator)
    
    def test_detect_emotion(self) -> None:
        """Test the detect_emotion method."""
        emotion = self.analyzer.detect_emotion("😀")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.JOY)
        self.assertGreater(emotion.primary_probability, 0.8)
    
    def test_map_to_emotional_states(self) -> None:
        """Test the map_to_emotional_states method."""
        emotion_map = self.analyzer.map_to_emotional_states("😀😔")
        self.assertIn(EmotionCategory.JOY, emotion_map)
        self.assertIn(EmotionCategory.SADNESS, emotion_map)
    
    def test_conversation_tracking_and_shift_detection(self) -> None:
        """Test conversation tracking and shift detection."""
        # Setup a conversation with a shift
        self.analyzer.track_conversation("😊")  # User happy
        self.analyzer.track_conversation("👍", is_user_message=False)  # Agent response
        self.analyzer.track_conversation("😔")  # User sad - emotional shift
        
        # Detect the shift
        shift = self.analyzer.detect_emotional_shift()
        self.assertIsNotNone(shift)
        self.assertEqual(shift.from_state.primary_emotion, EmotionCategory.JOY)
        self.assertEqual(shift.to_state.primary_emotion, EmotionCategory.SADNESS)
    
    def test_response_generation(self) -> None:
        """Test the response generation method."""
        # Generate response to sadness
        response = self.analyzer.generate_response("😭")
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("SADNESS", response.emotional_intent)
        
        # Generate response with specific tone
        response = self.analyzer.generate_response("😔", desired_tone=ResponseTone.ENCOURAGING)
        self.assertIsNotNone(response.emoji_sequence)
        self.assertIn("ENCOURAGING", response.emotional_intent)
    
    def test_cultural_context_adaptation(self) -> None:
        """Test cultural context adaptation method."""
        # Get original cultural context
        original_context = self.analyzer.cultural_context
        
        # Change to new context
        new_context = CulturalContext.EASTERN_ASIAN
        self.analyzer.adapt_to_cultural_context(new_context)
        
        # Verify context was changed
        self.assertEqual(self.analyzer.cultural_context, new_context)
        self.assertEqual(self.analyzer.response_generator.cultural_context, new_context)
        
        # Reset for other tests
        self.analyzer.adapt_to_cultural_context(original_context)


class TestEdgeCases(unittest.TestCase):
    """Test suite focused on edge cases and error handling."""
    
    def setUp(self) -> None:
        """Set up test resources before each test."""
        self.analyzer = EmojiEmotionalAnalyzer()
    
    def test_empty_input(self) -> None:
        """Test handling empty input in various methods."""
        # Detect emotion with empty input
        emotion = self.analyzer.detect_emotion("")
        self.assertEqual(emotion.primary_emotion, EmotionCategory.NEUTRAL)
        
        # Map empty input to emotional states
        emotion_map = self.analyzer.map_to_emotional_states("")
        self.assertIn(EmotionCategory.NEUTRAL, emotion_map)
        
        # Generate response to empty input
        response = self.analyzer.generate_response("")
        self.assertIsNotNone(response.emoji_sequence)
    
    def test_invalid_emojis(self) -> None:
        """Test handling input with invalid or unrecognized emojis."""
        # Use a rare or uncommon emoji
        emotion = self.analyzer.detect_emotion("🧿")  # Evil eye amulet
        # Should default to neutral if not in mappings
        if "🧿" not in self.analyzer.detection_engine.primary_emotion_mappings:
            self.assertEqual(emotion.primary_emotion, EmotionCategory.NEUTRAL)
    
    def test_mixed_content(self) -> None:
        """Test handling input with mixed emoji and text content."""
        # Mixed emoji and text
        emotion = self.analyzer.detect_emotion("Hello 😀 World! 👍")
        # Should extract and analyze only the emojis
        self.assertEqual(emotion.primary_emotion, EmotionCategory.JOY)
    
    def test_conflicting_emotions(self) -> None:
        """Test handling input with strongly conflicting emotions."""
        # Mix of happy and sad emojis
        emotion = self.analyzer.detect_emotion("😀😭")
        # Should have lower confidence due to conflict
        self.assertLess(emotion.confidence, 0.9)
        # Should have a secondary emotion
        self.assertIsNotNone(emotion.secondary_emotion)


if __name__ == '__main__':
    unittest.main()
