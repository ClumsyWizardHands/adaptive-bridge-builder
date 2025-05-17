#!/usr/bin/env python3
"""
Test Module for MultilingualEngine

This module contains unit tests for the MultilingualEngine class,
verifying its language detection, translation, cultural adaptation,
and communication style adaptation capabilities.
"""

import unittest
from unittest.mock import Mock, patch
import json

from multilingual_engine import (
    MultilingualEngine, Language, CulturalContext, LanguageProfile
)
from principle_engine import PrincipleEngine, Principle
from communication_style import (
    CommunicationStyle, FormalityLevel, DetailLevel, DirectnessLevel
)


class TestMultilingualEngine(unittest.TestCase):
    """Test cases for the MultilingualEngine class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock principle engine
        self.principle_engine = Mock(spec=PrincipleEngine)
        
        # Initialize the multilingual engine with mock dependencies
        self.engine = MultilingualEngine(
            agent_id="test_agent",
            default_language=Language.ENGLISH,
            principle_engine=self.principle_engine
        )
        
        # Set up some test language profiles
        self._setup_test_language_profiles()
        
        # Register some core principles and terminology
        self._setup_core_identity()

    def _setup_test_language_profiles(self):
        """Set up test language profiles for testing."""
        # English profile
        english_profile = LanguageProfile(
            language=Language.ENGLISH,
            cultural_dimensions=[
                CulturalContext.LOW_CONTEXT,
                CulturalContext.DIRECT,
                CulturalContext.INDIVIDUALIST
            ],
            formality_conventions={
                "professional": "Use titles initially, transition to first names"
            },
            honorifics={
                "general": "Mr./Ms./Mrs.",
                "academic": "Dr./Prof."
            },
            greeting_formats=[
                "Hello {name}",
                "Hi {name}"
            ],
            default_formality=FormalityLevel.NEUTRAL
        )
        self.engine.register_language_profile(english_profile)
        
        # Japanese profile
        japanese_profile = LanguageProfile(
            language=Language.JAPANESE,
            cultural_dimensions=[
                CulturalContext.HIGH_CONTEXT,
                CulturalContext.INDIRECT,
                CulturalContext.COLLECTIVIST,
                CulturalContext.HIERARCHICAL
            ],
            formality_conventions={
                "professional": "Use surname with honorifics, avoid first names"
            },
            honorifics={
                "general": "-san",
                "respected": "-sama",
                "teacher": "-sensei"
            },
            greeting_formats=[
                "こんにちは、{surname}{honorific}"
            ],
            default_formality=FormalityLevel.FORMAL
        )
        self.engine.register_language_profile(japanese_profile)
        
        # Spanish profile for testing
        spanish_profile = LanguageProfile(
            language=Language.SPANISH,
            cultural_dimensions=[
                CulturalContext.HIGH_CONTEXT,
                CulturalContext.COLLECTIVIST
            ],
            honorifics={
                "general": "Sr./Sra./Srta."
            },
            greeting_formats=[
                "Hola, {name}"
            ],
            default_formality=FormalityLevel.FORMAL
        )
        self.engine.register_language_profile(spanish_profile)

    def _setup_core_identity(self):
        """Set up core identity elements for testing."""
        # Register core principles
        self.engine.register_core_principle({
            "name": "Fairness as a Fundamental Truth",
            "description": "Treat all agents equitably based on merit and need."
        })
        self.engine.register_core_principle({
            "name": "Harmony Through Presence",
            "description": "Active engagement creates harmony in multi-agent systems."
        })
        
        # Register key terminology
        self.engine.register_key_terminology("agent_communication", {
            "agent": {
                "en": "agent",
                "es": "agente",
                "ja": "エージェント"
            },
            "protocol": {
                "en": "protocol",
                "es": "protocolo",
                "ja": "プロトコル"
            }
        })

    def test_language_detection(self):
        """Test language detection functionality."""
        # Test English detection
        self.assertEqual(
            self.engine.detect_language("Hello, how are you today?"),
            Language.ENGLISH
        )
        
        # Test Japanese detection (using simplified detection logic)
        with patch.object(self.engine, 'detection_service', None):  # Ensure fallback logic is used
            self.assertEqual(
                self.engine.detect_language("こんにちは、お元気ですか？"),
                Language.JAPANESE
            )
        
        # Test Spanish detection (using simplified detection logic)
        with patch.object(self.engine, 'detection_service', None):  # Ensure fallback logic is used
            self.assertEqual(
                self.engine.detect_language("Hola, ¿cómo estás hoy?"),
                Language.SPANISH
            )

    def test_get_language_profile(self):
        """Test retrieving language profiles."""
        # Test retrieving existing profile
        profile = self.engine.get_language_profile(Language.JAPANESE)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.language, Language.JAPANESE)
        
        # Test fallback to default language if profile not found
        profile = self.engine.get_language_profile(Language.HINDI)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.language, Language.ENGLISH)

    def test_agent_language_cache(self):
        """Test caching and retrieving agent languages."""
        # Test default language for unknown agent
        self.assertEqual(
            self.engine.get_agent_language("unknown_agent"),
            Language.ENGLISH
        )
        
        # Test updating and retrieving language
        self.engine.update_agent_language("test_agent_1", Language.JAPANESE)
        self.assertEqual(
            self.engine.get_agent_language("test_agent_1"),
            Language.JAPANESE
        )

    def test_translate_message(self):
        """Test message translation functionality."""
        # Simple test message
        message = "Hello, this is a test message about agent protocol."
        
        # Test translation between different languages
        translated, success = self.engine.translate_message(
            message=message,
            from_language=Language.ENGLISH,
            to_language=Language.JAPANESE
        )
        
        self.assertTrue(success)
        self.assertIsInstance(translated, str)
        self.assertIn("from English to Japanese", translated)
        
        # Test with nested content in dictionary
        message_dict = {
            "content": "Hello, this is a test message about agent protocol.",
            "metadata": {"importance": "high"}
        }
        
        translated_dict, success = self.engine.translate_message(
            message=message_dict,
            from_language=Language.ENGLISH,
            to_language=Language.SPANISH
        )
        
        self.assertTrue(success)
        self.assertIsInstance(translated_dict, dict)
        self.assertIn("content", translated_dict)
        self.assertIn("from English to Spanish", translated_dict["content"])
        self.assertEqual(translated_dict["source_language"], Language.ENGLISH.value)
        self.assertEqual(translated_dict["target_language"], Language.SPANISH.value)
        
        # Test with same source and target language (should not translate)
        original = "This should not be translated."
        translated, success = self.engine.translate_message(
            message=original,
            from_language=Language.ENGLISH,
            to_language=Language.ENGLISH
        )
        
        self.assertTrue(success)
        self.assertEqual(translated, original)

    def test_adapt_communication_style(self):
        """Test communication style adaptation for different languages."""
        # Create a base communication style
        base_style = CommunicationStyle(
            agent_id="test_agent",
            formality=FormalityLevel.CASUAL,
            directness=DirectnessLevel.DIRECT,
            detail_level=DetailLevel.CONCISE
        )
        
        # Test adaptation for Japanese (should become more formal and less direct)
        adapted_style = self.engine.adapt_communication_style(base_style, Language.JAPANESE)
        
        self.assertEqual(adapted_style.formality, FormalityLevel.FORMAL)
        self.assertNotEqual(adapted_style.directness, DirectnessLevel.DIRECT)
        
        # Test adaptation for Spanish
        adapted_style = self.engine.adapt_communication_style(base_style, Language.SPANISH)
        self.assertEqual(adapted_style.formality, FormalityLevel.FORMAL)

    def test_cultural_adaptations(self):
        """Test cultural adaptations of messages."""
        # Test message with greeting and honorifics
        message = "Hello, Mr. Smith. I would like to discuss the project directly."
        
        # Test adaptation for Japanese
        adaptations = self.engine._apply_cultural_adaptations(
            content=message,
            language=Language.JAPANESE
        )
        
        self.assertIn("content", adaptations)
        self.assertIn("applied_adaptations", adaptations)
        self.assertTrue(len(adaptations["applied_adaptations"]) > 0)
        
        # Verify greeting adaptation
        adapted_content = adaptations["content"]
        self.assertNotEqual(adapted_content, message)
        
        # Test honorific adaptation
        self.assertTrue(any("honorific" in adaptation for adaptation in adaptations["applied_adaptations"]))

    def test_process_incoming_message(self):
        """Test processing of incoming messages with language detection."""
        # Test with string message
        result = self.engine.process_incoming_message(
            message="Hello, this is a test message.",
            sender_id="test_sender"
        )
        
        self.assertIn("detected_language", result)
        self.assertEqual(result["detected_language"], Language.ENGLISH.value)
        self.assertIn("language_profile", result)
        
        # Test with dictionary message
        result = self.engine.process_incoming_message(
            message={"content": "Hola, esto es un mensaje de prueba.", "metadata": {}},
            sender_id="spanish_sender"
        )
        
        self.assertIn("detected_language", result)
        self.assertEqual(result["detected_language"], Language.SPANISH.value)
        self.assertIn("content", result)
        
        # Verify language is cached for sender
        self.assertEqual(
            self.engine.get_agent_language("spanish_sender"),
            Language.SPANISH
        )

    def test_prepare_response(self):
        """Test preparation of responses with appropriate cultural adaptation."""
        # Test response preparation
        message = "Thank you for your message. I will address your concerns directly."
        
        # Prepare response for Japanese recipient
        result = self.engine.prepare_response(
            message=message,
            recipient_id="japanese_recipient",
            source_language=Language.ENGLISH,
            target_language=Language.JAPANESE
        )
        
        self.assertIn("content", result)
        self.assertIn("source_language", result)
        self.assertEqual(result["source_language"], Language.ENGLISH.value)
        self.assertEqual(result["target_language"], Language.JAPANESE.value)
        self.assertIn("cultural_adaptations", result)
        
        # Test with recipient language from cache
        self.engine.update_agent_language("cached_recipient", Language.SPANISH)
        
        result = self.engine.prepare_response(
            message=message,
            recipient_id="cached_recipient"
        )
        
        self.assertEqual(result["target_language"], Language.SPANISH.value)

    def test_support_new_language(self):
        """Test adding support for new languages."""
        # Test adding language that's in enum but not in profiles
        result = self.engine.support_new_language("it")  # Italian
        
        self.assertEqual(result, Language.ITALIAN)
        self.assertIn(Language.ITALIAN, self.engine.language_profiles)
        
        # Test with custom profile data
        profile_data = {
            "cultural_dimensions": ["high_context", "collectivist"],
            "greeting_formats": ["Ciao, {name}"],
            "default_formality": "Formal"
        }
        
        result = self.engine.support_new_language("fr", profile_data)  # French
        
        self.assertEqual(result, Language.FRENCH)
        self.assertIn(Language.FRENCH, self.engine.language_profiles)
        fr_profile = self.engine.get_language_profile(Language.FRENCH)
        self.assertEqual(fr_profile.default_formality, FormalityLevel.FORMAL)
        
        # Test with unsupported language code (should default to English)
        with self.assertLogs(level='ERROR'):
            result = self.engine.support_new_language("xx")  # Invalid code
            self.assertEqual(result, Language.ENGLISH)

    def test_get_cultural_guidance(self):
        """Test getting cultural guidance for a language."""
        # Get guidance for Japanese
        guidance = self.engine.get_cultural_guidance(Language.JAPANESE)
        
        self.assertIn("language", guidance)
        self.assertIn("cultural_dimensions", guidance)
        self.assertIn("formality_guidance", guidance)
        self.assertIn("communication_recommendations", guidance)
        
        # Check for dimension-specific recommendations
        recommendations = guidance["communication_recommendations"]
        self.assertTrue(len(recommendations) > 0)
        
        # At least one recommendation should mention indirect communication
        self.assertTrue(any("indirect" in rec.lower() for rec in recommendations))
        
        # Test with language that has no profile (should return empty dict)
        no_profile_guidance = self.engine.get_cultural_guidance(Language.KOREAN)
        self.assertEqual(no_profile_guidance, {})


if __name__ == '__main__':
    unittest.main()
