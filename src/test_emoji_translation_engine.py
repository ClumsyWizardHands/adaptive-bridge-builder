"""
Unit tests for the EmojiTranslationEngine component.

This file contains tests for the key functionality of the EmojiTranslationEngine,
including text-to-emoji translation, emoji-to-text translation, abstract concept
handling, ambiguity resolution, and dictionary customization.
"""

import unittest
from emoji_translation_engine import (
    EmojiTranslationEngine,
    TranslationMode,
    AmbiguityResolutionStrategy,
    EmojiCategory,
    EmojiEntry
)


class TestEmojiTranslationEngine(unittest.TestCase):
    """Test cases for the EmojiTranslationEngine component."""

    def setUp(self):
        """Set up the test environment."""
        self.engine = EmojiTranslationEngine()
        
        # Add some test entries to the dictionary for consistent testing
        self.test_emoji = EmojiEntry(
            emoji="🧪",
            name="test tube",
            keywords=["test", "experiment", "science", "laboratory"],
            categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
            sentiment_score=0.1,
            common_contexts=["testing", "research", "development"],
            related_emojis=["🔬", "⚗️", "🔍"],
            abstract_concepts=["testing", "experimentation", "research"]
        )
        
        self.engine.add_emoji_to_dictionary(self.test_emoji)

    def test_text_to_emoji_literal_mode(self):
        """Test text to emoji translation in LITERAL mode."""
        text = "I am happy to test this system."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.LITERAL)
        
        # In literal mode, we expect direct word-to-emoji mapping
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Test emoji should appear for the word "test"
        if "test" in text.lower():
            self.assertIn("🧪", result)

    def test_text_to_emoji_semantic_mode(self):
        """Test text to emoji translation in SEMANTIC mode."""
        text = "I am considering the implications of this experiment."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        
        # In semantic mode, we expect meaning-based translation
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Test emoji should appear for words related to "experiment"
        self.assertIn("🧪", result)

    def test_text_to_emoji_emotional_mode(self):
        """Test text to emoji translation in EMOTIONAL mode."""
        # Test positive sentiment
        positive_text = "I am very happy with the test results."
        positive_result = self.engine.translate_text_to_emoji(positive_text, TranslationMode.EMOTIONAL)
        
        # Should include a positive sentiment emoji
        self.assertIn("😊", positive_result)
        self.assertIn("🧪", positive_result)  # Should still include the test emoji
        
        # Test negative sentiment
        negative_text = "I am disappointed with the test results."
        negative_result = self.engine.translate_text_to_emoji(negative_text, TranslationMode.EMOTIONAL)
        
        # Should include a negative sentiment emoji
        self.assertIn("😢", negative_result)
        self.assertIn("🧪", negative_result)  # Should still include the test emoji

    def test_text_to_emoji_summarized_mode(self):
        """Test text to emoji translation in SUMMARIZED mode."""
        text = "We need to conduct extensive tests on the system before deploying it to production."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.SUMMARIZED)
        
        # In summarized mode, we expect fewer emojis but still capturing key concepts
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Should be more concise than semantic mode
        semantic_result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        self.assertLessEqual(len(result), len(semantic_result))
        
        # Test emoji should still appear for the concept of testing
        self.assertIn("🧪", result)

    def test_text_to_emoji_expressive_mode(self):
        """Test text to emoji translation in EXPRESSIVE mode."""
        text = "The experimental results are amazing and exceeded our expectations!"
        result = self.engine.translate_text_to_emoji(text, TranslationMode.EXPRESSIVE)
        
        # In expressive mode, we expect more emojis for emphasis
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Should be more expressive than semantic mode
        semantic_result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        self.assertGreaterEqual(len(result), len(semantic_result))
        
        # Test emoji should appear for the concept of experimentation
        self.assertIn("🧪", result)
        
        # Should include emphasis emojis for positive sentiment
        emphasis_emojis = ["✨", "🔥", "💯"]
        self.assertTrue(any(emoji in result for emoji in emphasis_emojis))

    def test_emoji_to_text_most_common(self):
        """Test emoji to text translation with MOST_COMMON strategy."""
        emoji_sequence = "🧪😊👍"
        result = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            resolution_strategy=AmbiguityResolutionStrategy.MOST_COMMON
        )
        
        # Should provide a text translation
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Should include the first keyword for each emoji
        self.assertIn("test", result)
        self.assertIn("smile", result)
        self.assertIn("yes", result)

    def test_emoji_to_text_contextual(self):
        """Test emoji to text translation with CONTEXTUAL strategy."""
        emoji_sequence = "🧪🤔💡"
        
        # Without context
        result_no_context = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            resolution_strategy=AmbiguityResolutionStrategy.CONTEXTUAL
        )
        
        # With context
        context = ["research", "development", "idea"]
        result_with_context = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            context=context,
            resolution_strategy=AmbiguityResolutionStrategy.CONTEXTUAL
        )
        
        # Both should provide a text translation
        self.assertIsInstance(result_no_context, str)
        self.assertIsInstance(result_with_context, str)
        
        # The translations might be different based on context
        # This is hard to test deterministically, but we can check that keywords are included
        self.assertIn("test", result_no_context)
        self.assertIn("test", result_with_context)
        
        # If we provide research context, experiment might be preferred over test
        if "experiment" in self.test_emoji.keywords:
            self.assertIn("experiment", result_with_context)

    def test_emoji_to_text_multiple(self):
        """Test emoji to text translation with MULTIPLE strategy."""
        emoji_sequence = "🧪👍"
        result = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            resolution_strategy=AmbiguityResolutionStrategy.MULTIPLE
        )
        
        # Should provide a list of possible interpretations
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
        # The first interpretation should be the most common one
        self.assertIn("test", result[0])
        self.assertIn("yes", result[0])

    def test_emoji_to_text_clarify(self):
        """Test emoji to text translation with CLARIFY strategy."""
        emoji_sequence = "🧪🤔"
        result = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            resolution_strategy=AmbiguityResolutionStrategy.CLARIFY
        )
        
        # Should provide a dictionary with translation and ambiguities
        self.assertIsInstance(result, dict)
        self.assertIn('translation', result)
        self.assertIn('ambiguities', result)
        self.assertIn('clarification_needed', result)
        self.assertIn('options', result)
        
        # The translation should include keywords for both emojis
        self.assertIn("test", result['translation'])
        self.assertIn("think", result['translation'])
        
        # Both emojis should be ambiguous since they have multiple keywords
        self.assertIn("🧪", result['ambiguities'])
        self.assertIn("🤔", result['ambiguities'])
        
        # Options should include keywords for ambiguous emojis
        self.assertIn("🧪", result['options'])
        self.assertIn("🤔", result['options'])
        self.assertIn("test", result['options']["🧪"])
        self.assertIn("think", result['options']["🤔"])

    def test_emoji_to_text_confidence(self):
        """Test emoji to text translation with CONFIDENCE strategy."""
        emoji_sequence = "🧪👍"
        result = self.engine.translate_emoji_to_text(
            emoji_sequence, 
            resolution_strategy=AmbiguityResolutionStrategy.CONFIDENCE
        )
        
        # Should provide a dictionary with translation and confidence
        self.assertIsInstance(result, dict)
        self.assertIn('translation', result)
        self.assertIn('confidence', result)
        self.assertIn('alternatives', result)
        
        # The translation should include keywords for both emojis
        self.assertIn("test", result['translation'])
        self.assertIn("yes", result['translation'])
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Alternatives should be a list of tuples (translation, confidence)
        self.assertIsInstance(result['alternatives'], list)
        if result['alternatives']:
            alt, conf = result['alternatives'][0]
            self.assertIsInstance(alt, str)
            self.assertIsInstance(conf, float)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_abstract_concept_handling(self):
        """Test handling of abstract concepts."""
        # Test getting emojis for abstract concepts
        time_emojis = self.engine.get_emoji_for_abstract_concept("time")
        self.assertIsInstance(time_emojis, list)
        self.assertIn("⏳", time_emojis)
        
        # Test translating text with abstract concepts
        text = "We need to wait for the process to complete."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        
        # Should include time-related emoji
        self.assertIn("⏳", result)
        
        # Test with explicit abstract concept
        text = "Let's find a creative solution."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        
        # Should include idea-related emoji
        self.assertIn("💡", result)

    def test_ambiguity_resolution(self):
        """Test ambiguity resolution through user feedback."""
        emoji = "🤔"
        
        # Initial translation
        before = self.engine.translate_emoji_to_text(emoji)
        self.assertIn("think", before)
        
        # Resolve ambiguity
        selected_meaning = "ponder"
        context = ["deep thought", "consideration"]
        
        self.engine.resolve_ambiguity(emoji, selected_meaning, context)
        
        # After resolution in the specific context
        after = self.engine.translate_emoji_to_text(emoji, context=context)
        
        # Should prefer the selected meaning
        self.assertIn("ponder", after)
        
        # Verify that the context was updated
        self.assertIn("deep thought", self.engine.recent_context)
        self.assertIn("consideration", self.engine.recent_context)

    def test_dictionary_customization(self):
        """Test adding custom emojis to the dictionary."""
        # Create a custom emoji entry
        code_emoji = EmojiEntry(
            emoji="👨‍💻",
            name="man technologist",
            keywords=["programmer", "developer", "coder", "engineer"],
            categories=[EmojiCategory.PERSON, EmojiCategory.ABSTRACT],
            sentiment_score=0.3,
            common_contexts=["development", "programming", "technology"],
            related_emojis=["💻", "⌨️", "🖥️"],
            abstract_concepts=["coding", "development", "technology"]
        )
        
        # Add to dictionary
        self.engine.add_emoji_to_dictionary(code_emoji)
        
        # Test that it was added
        text = "The programmer is writing code."
        result = self.engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        
        # Should include the custom emoji
        self.assertIn("👨‍💻", result)
        
        # Test translating back
        emoji_sequence = "👨‍💻"
        text_result = self.engine.translate_emoji_to_text(emoji_sequence)
        
        # Should include the first keyword
        self.assertIn("programmer", text_result)

    def test_context_update_and_caching(self):
        """Test context updating and translation caching."""
        # Initial state
        self.assertEqual(len(self.engine.recent_context), 0)
        self.assertEqual(len(self.engine.translation_cache), 0)
        
        # Translate with context update
        text = "This is a test message."
        result1 = self.engine.translate_text_to_emoji(text, update_context=True)
        
        # Context should be updated
        self.assertEqual(len(self.engine.recent_context), 1)
        self.assertIn(text, self.engine.recent_context)
        
        # Cache should be updated
        self.assertEqual(len(self.engine.translation_cache), 1)
        
        # Translate the same text again
        result2 = self.engine.translate_text_to_emoji(text)
        
        # Should get the same result from cache
        self.assertEqual(result1, result2)
        
        # Translate without context update
        new_text = "Another test message."
        self.engine.translate_text_to_emoji(new_text, update_context=False)
        
        # Context should not include the new text
        self.assertNotIn(new_text, self.engine.recent_context)
        
        # Test context pruning
        # Fill the context to capacity
        for i in range(self.engine.context_history_size):
            self.engine.translate_text_to_emoji(f"Context message {i}")
        
        # Add one more to trigger pruning
        self.engine.translate_text_to_emoji("Overflow message")
        
        # Context size should be maintained
        self.assertEqual(len(self.engine.recent_context), self.engine.context_history_size)
        
        # The first message should be removed
        self.assertNotIn(text, self.engine.recent_context)


if __name__ == '__main__':
    unittest.main()
