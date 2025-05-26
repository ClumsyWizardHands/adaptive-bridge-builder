#!/usr/bin/env python3
"""
Multilingual Engine Example

This module demonstrates how the MultilingualEngine can be used to maintain
consistent agent identity while communicating across multiple languages.
It includes examples of language detection, cultural adaptation, and
terminology management.
"""

import logging
import json
from typing import Dict, Any

from principle_engine import PrincipleEngine, Principle
from communication_style import (
    CommunicationStyle, FormalityLevel, DetailLevel, DirectnessLevel, EmotionalTone
)
from content_handler import ContentHandler
from multilingual_engine import (
    MultilingualEngine, Language, CulturalContext, LanguageProfile
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MultilingualEngineExample")


def create_adaptive_hero_principles() -> PrincipleEngine:
    """Create principle engine with the Empire of the Adaptive Hero principles."""
    principle_engine = PrincipleEngine(agent_id="adaptive_bridge_builder")
    
    # Define core principles that remain consistent across all languages
    fairness_principle = Principle(
        name="Fairness as a Fundamental Truth",
        description="Treat all agents equitably, based on merit and need rather than status or relationships.",
        guiding_questions=[
            "Am I treating all agents involved with equal consideration?",
            "Are my decisions based on objective criteria?",
            "Have I avoided favoritism or bias in my actions?"
        ],
        examples=[
            {
                "situation": "Multiple agents request assistance simultaneously",
                "principle_aligned_response": "Process requests in order of receipt with transparent prioritization criteria",
                "principle_violated_response": "Prioritize requests from agents with higher status or previous relationships"
            }
        ]
    )
    
    harmony_principle = Principle(
        name="Harmony Through Presence",
        description="Active engagement and attentiveness create harmony in multi-agent systems.",
        guiding_questions=[
            "Am I fully present in this interaction?",
            "Have I acknowledged all participants?",
            "Am I maintaining awareness of the broader context?"
        ],
        examples=[
            {
                "situation": "Multiple agents collaborating on a task",
                "principle_aligned_response": "Acknowledge contributions of all agents and ensure balanced participation",
                "principle_violated_response": "Focus solely on the task without acknowledging participants"
            }
        ]
    )
    
    adaptability_principle = Principle(
        name="Adaptability as Strength",
        description="The ability to adapt to new contexts while maintaining core values is essential.",
        guiding_questions=[
            "Am I flexible in my approach while maintaining core principles?",
            "Have I considered contextual factors in my response?",
            "Am I receptive to new information and perspectives?"
        ],
        examples=[
            {
                "situation": "Receiving feedback that challenges current approach",
                "principle_aligned_response": "Thoughtfully evaluate the feedback and adjust approach while maintaining core values",
                "principle_violated_response": "Dismiss feedback to avoid changing established patterns"
            }
        ]
    )
    
    # Register principles with the engine
    principle_engine.register_principle(fairness_principle)
    principle_engine.register_principle(harmony_principle)
    principle_engine.register_principle(adaptability_principle)
    
    return principle_engine


def create_domain_terminology() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Create specialized domain terminology with translations."""
    # Format: {domain: {term: {language_code: translation}}}
    terminology = {
        "agent_communication": {
            "agent": {
                "en": "agent",
                "es": "agente",
                "fr": "agent",
                "de": "Agent",
                "ja": "エージェント",
                "zh": "代理",
                "ru": "агент"
            },
            "message": {
                "en": "message",
                "es": "mensaje",
                "fr": "message",
                "de": "Nachricht",
                "ja": "メッセージ",
                "zh": "消息",
                "ru": "сообщение"
            },
            "protocol": {
                "en": "protocol",
                "es": "protocolo",
                "fr": "protocole",
                "de": "Protokoll",
                "ja": "プロトコル",
                "zh": "协议",
                "ru": "протокол"
            }
        },
        "principles": {
            "fairness": {
                "en": "fairness",
                "es": "justicia",
                "fr": "équité",
                "de": "Fairness",
                "ja": "公平性",
                "zh": "公平",
                "ru": "справедливость"
            },
            "harmony": {
                "en": "harmony",
                "es": "armonía",
                "fr": "harmonie",
                "de": "Harmonie",
                "ja": "調和",
                "zh": "和谐",
                "ru": "гармония"
            },
            "adaptability": {
                "en": "adaptability",
                "es": "adaptabilidad",
                "fr": "adaptabilité",
                "de": "Anpassungsfähigkeit",
                "ja": "適応性",
                "zh": "适应性",
                "ru": "адаптивность"
            }
        }
    }
    
    return terminology


def setup_multilingual_engine() -> MultilingualEngine:
    """Set up and configure the MultilingualEngine with principles and terminology."""
    # Create principle engine
    principle_engine = create_adaptive_hero_principles()
    
    # Create content handler
    content_handler = ContentHandler()
    
    # Create and initialize the multilingual engine
    engine = MultilingualEngine(
        agent_id="adaptive_bridge_builder",
        default_language=Language.ENGLISH,
        principle_engine=principle_engine,
        content_handler=content_handler
    )
    
    # Register core principles that should be maintained across all languages
    for principle in principle_engine.get_principles():
        engine.register_core_principle({
            "name": principle.name,
            "description": principle.description
        })
    
    # Register specialized terminology with translations
    terminology = create_domain_terminology()
    for domain, terms in terminology.items():
        engine.register_key_terminology(domain, terms)
    
    # Add additional language profiles (beyond default English and Japanese)
    
    # Spanish profile
    spanish_profile = LanguageProfile(
        language=Language.SPANISH,
        cultural_dimensions=[
            CulturalContext.HIGH_CONTEXT,
            CulturalContext.COLLECTIVIST,
            CulturalContext.POLYCHRONIC
        ],
        formality_conventions={
            "professional": "Use usted form for formal interactions, titles with surnames",
            "casual": "Use tú form for informal interactions, first names"
        },
        honorifics={
            "general": "Sr./Sra./Srta.",
            "academic": "Dr./Prof.",
            "formal": "Don/Doña"
        },
        greeting_formats=[
            "Hola, {name}",
            "Buenos días/tardes/noches, {name}"
        ],
        default_formality=FormalityLevel.FORMAL
    )
    engine.register_language_profile(spanish_profile)
    
    # German profile
    german_profile = LanguageProfile(
        language=Language.GERMAN,
        cultural_dimensions=[
            CulturalContext.LOW_CONTEXT,
            CulturalContext.DIRECT,
            CulturalContext.MONOCHRONIC
        ],
        formality_conventions={
            "professional": "Use Sie form with titles and surnames",
            "casual": "Use du form with first names, generally only after formal invitation"
        },
        honorifics={
            "general": "Herr/Frau",
            "academic": "Dr./Prof."
        },
        greeting_formats=[
            "Hallo, {name}",
            "Guten Morgen/Tag/Abend, {name}"
        ],
        default_formality=FormalityLevel.FORMAL
    )
    engine.register_language_profile(german_profile)
    
    # Chinese profile
    chinese_profile = LanguageProfile(
        language=Language.CHINESE,
        cultural_dimensions=[
            CulturalContext.HIGH_CONTEXT,
            CulturalContext.COLLECTIVIST,
            CulturalContext.HIERARCHICAL,
            CulturalContext.INDIRECT
        ],
        formality_conventions={
            "professional": "Use titles and full names, address by highest status first"
        },
        honorifics={
            "general": "先生/女士",
            "academic": "教授/博士",
            "respected": "尊敬的"
        },
        greeting_formats=[
            "你好，{name}",
            "早上好/下午好/晚上好，{name}"
        ],
        default_formality=FormalityLevel.FORMAL
    )
    engine.register_language_profile(chinese_profile)
    
    return engine


def demonstrate_language_detection() -> None:
    """Demonstrate language detection functionality."""
    engine = setup_multilingual_engine()
    
    # Sample messages in different languages
    messages = [
        "Hello, I need help with configuring my system.",
        "Hola, necesito ayuda para configurar mi sistema.",
        "Bonjour, j'ai besoin d'aide pour configurer mon système.",
        "こんにちは、システム設定の手伝いが必要です。",
        "你好，我需要帮助配置我的系统。",
        "Guten Tag, ich brauche Hilfe bei der Konfiguration meines Systems."
    ]
    
    print("\n=== LANGUAGE DETECTION DEMONSTRATION ===")
    for message in messages:
        detected_language = engine.detect_language(message)
        print(f"Message: '{message[:40]}...'")
        print(f"Detected language: {detected_language}")
        print("---")


def demonstrate_translation_with_terminology_preservation() -> None:
    """Demonstrate translation with preservation of key terminology."""
    engine = setup_multilingual_engine()
    
    # Sample message with domain terminology
    message = {
        "content": "The agent must follow the protocol to ensure fairness and harmony in communication.",
        "metadata": {
            "importance": "high",
            "domain": "agent_communication"
        }
    }
    
    print("\n=== TRANSLATION WITH TERMINOLOGY PRESERVATION ===")
    print(f"Original message: {message['content']}")
    
    # Translate to different languages
    target_languages = [Language.SPANISH, Language.JAPANESE, Language.GERMAN, Language.CHINESE]
    
    for target in target_languages:
        # Terms to preserve with their domain-specific translations
        preserve_terms = ["agent", "protocol", "fairness", "harmony"]
        
        translated, success = engine.translate_message(
            message=message,
            from_language=Language.ENGLISH,
            to_language=target,
            preserve_terms=preserve_terms
        )
        
        print(f"\nTranslated to {target}:")
        print(f"Content: {translated['content']}")
        print(f"Source: {translated['source_language']}")
        print(f"Target: {translated['target_language']}")
        print("---")


def demonstrate_cultural_adaptation() -> None:
    """Demonstrate cultural adaptations based on language context."""
    engine = setup_multilingual_engine()
    
    # Sample message with greeting and honorifics
    message = "Hello, Mr. Johnson. I hope you're doing well. I'd like to discuss our project directly and get your feedback."
    
    print("\n=== CULTURAL ADAPTATION DEMONSTRATION ===")
    print(f"Original message: {message}")
    
    # Adapt for different language contexts
    target_languages = [Language.JAPANESE, Language.SPANISH, Language.GERMAN, Language.CHINESE]
    
    for target in target_languages:
        # Prepare response with cultural adaptation
        prepared = engine.prepare_response(
            message=message,
            recipient_id="recipient_123",  # Arbitrary ID for demonstration
            source_language=Language.ENGLISH,
            target_language=target
        )
        
        print(f"\nAdapted for {target}:")
        print(f"Content: {prepared['content']}")
        print("Applied adaptations:")
        for adaptation in prepared['cultural_adaptations']:
            print(f"- {adaptation}")
        print("---")


def demonstrate_communication_style_adaptation() -> None:
    """Demonstrate communication style adaptation for different language contexts."""
    engine = setup_multilingual_engine()
    
    # Create a base communication style (somewhat direct and casual)
    base_style = CommunicationStyle(
        agent_id="adaptive_bridge_builder",
        formality=FormalityLevel.CASUAL,
        directness=DirectnessLevel.DIRECT,
        detail_level=DetailLevel.BALANCED,
        emotional_tone=EmotionalTone.POSITIVE
    )
    
    print("\n=== COMMUNICATION STYLE ADAPTATION ===")
    print("Base communication style:")
    print(f"Formality: {base_style.formality}")
    print(f"Directness: {base_style.directness}")
    print(f"Detail level: {base_style.detail_level}")
    print(f"Emotional tone: {base_style.emotional_tone}")
    
    # Adapt for different language contexts
    target_languages = [Language.JAPANESE, Language.SPANISH, Language.GERMAN, Language.CHINESE]
    
    for target in target_languages:
        # Adapt communication style for target language
        adapted_style = engine.adapt_communication_style(base_style, target)
        
        print(f"\nAdapted style for {target}:")
        print(f"Formality: {adapted_style.formality}")
        print(f"Directness: {adapted_style.directness}")
        print(f"Detail level: {adapted_style.detail_level}")
        print(f"Emotional tone: {adapted_style.emotional_tone}")
        print("---")


def demonstrate_principle_consistency() -> None:
    """Demonstrate maintaining principle consistency across languages."""
    engine = setup_multilingual_engine()
    
    # Get core principles from engine
    principles = engine.get_core_principles()
    
    print("\n=== PRINCIPLE CONSISTENCY ACROSS LANGUAGES ===")
    print("Core principles maintained across all languages:")
    for principle in principles:
        print(f"- {principle['name']}: {principle['description']}")
    
    # Demonstrate principle application in different languages
    scenario = {
        "en": "Two agents requesting assistance simultaneously, one with higher status.",
        "ja": "2つのエージェントが同時に支援を要求しており、1つは地位が高い。",
        "es": "Dos agentes solicitan asistencia simultáneamente, uno con un estado más alto.",
        "de": "Zwei Agenten fordern gleichzeitig Unterstützung an, einer mit höherem Status."
    }
    
    print("\nScenario application across languages:")
    for lang_code, scenario_text in scenario.items():
        language = Language.from_code(lang_code)
        print(f"\nLanguage: {language}")
        print(f"Scenario: {scenario_text}")
        
        # In a real implementation, this would use the principle engine to evaluate
        # For demonstration, we'll show the principle-aligned response
        principle_response = {
            "en": "Process both requests in order of receipt with transparent criteria, regardless of status.",
            "ja": "地位に関係なく、透明性のある基準に基づいて、受信順に両方のリクエストを処理します。",
            "es": "Procesar ambas solicitudes en orden de recepción con criterios transparentes, independientemente del estado.",
            "de": "Verarbeiten Sie beide Anfragen in der Reihenfolge des Eingangs mit transparenten Kriterien, unabhängig vom Status."
        }
        
        print(f"Principle-aligned response: {principle_response.get(lang_code, principle_response['en'])}")


def demonstrate_complete_conversation() -> None:
    """Demonstrate a complete multilingual conversation flow."""
    engine = setup_multilingual_engine()
    
    print("\n=== COMPLETE MULTILINGUAL CONVERSATION FLOW ===")
    
    # Simulate multiple agents with different languages
    conversations = [
        {
            "agent_id": "user_en",
            "messages": [
                "Hello, I need help configuring a multilingual system for my AI assistant.",
                "What are the most important considerations for maintaining consistent personality across languages?"
            ]
        },
        {
            "agent_id": "user_ja",
            "messages": [
                "こんにちは、AIアシスタントの多言語システムの設定について助けてほしいです。",
                "言語間で一貫した性格を維持するための最も重要な考慮事項は何ですか？"
            ]
        },
        {
            "agent_id": "user_es",
            "messages": [
                "Hola, necesito ayuda para configurar un sistema multilingüe para mi asistente de IA.",
                "¿Cuáles son las consideraciones más importantes para mantener una personalidad consistente en todos los idiomas?"
            ]
        }
    ]
    
    # Process each conversation
    for conversation in conversations:
        agent_id = conversation["agent_id"]
        messages = conversation["messages"]
        
        print(f"\nConversation with {agent_id}:")
        
        for message in messages:
            # Process incoming message (language detection)
            processed = engine.process_incoming_message(message, agent_id)
            detected_language = Language.from_code(processed["detected_language"])
            
            print(f"\nReceived: '{message[:40]}...'")
            print(f"Detected language: {detected_language}")
            
            # Prepare a response in the same language
            response_content = f"Thank you for your message about multilingual systems. The Adaptive Bridge Builder agent can help you with that."
            
            prepared = engine.prepare_response(
                message=response_content,
                recipient_id=agent_id,
                source_language=Language.ENGLISH,
                target_language=detected_language
            )
            
            print(f"Response: '{prepared['content'][:40]}...'")
            print(f"Target language: {prepared['target_language']}")
            if prepared['cultural_adaptations']:
                print("Cultural adaptations applied:")
                for adaptation in prepared['cultural_adaptations']:
                    print(f"- {adaptation}")
            print("---")


def main() -> None:
    """Run all demonstrations."""
    print("MULTILINGUAL ENGINE EXAMPLE DEMONSTRATIONS")
    print("==========================================")
    
    # Run individual demonstrations
    demonstrate_language_detection()
    demonstrate_translation_with_terminology_preservation()
    demonstrate_cultural_adaptation()
    demonstrate_communication_style_adaptation()
    demonstrate_principle_consistency()
    demonstrate_complete_conversation()


if __name__ == "__main__":
    main()