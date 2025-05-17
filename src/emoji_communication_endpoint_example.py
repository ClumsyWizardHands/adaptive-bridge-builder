"""
Example demonstrating the EmojiCommunicationEndpoint component functionality.

This file provides concrete examples of how to use the dedicated emoji-only interface
for both synchronous and asynchronous communication patterns.
"""

import asyncio
import json
import time
from pprint import pprint

from emoji_communication_endpoint import (
    EmojiCommunicationEndpoint,
    EmojiRequest,
    EmojiResponse,
    EmojiContentType,
    EmojiErrorCode,
    EmojiAuthMethod,
    EmojiMetadata,
    EmojiFallback
)

from emoji_knowledge_base import (
    EmojiKnowledgeBase,
    EmojiDomain,
    CulturalContext
)

from emoji_sequence_optimizer import (
    OptimizationProfile
)


def demonstrate_synchronous_communication():
    """Demonstrate basic synchronous emoji communication."""
    print("\n=== Synchronous Emoji Communication ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Simple emoji sequence request
    request = EmojiRequest(
        emoji_content="👋😊🌎",
        content_type=EmojiContentType.EMOJI_SEQUENCE,
        domain=EmojiDomain.GENERAL,
        cultural_context=CulturalContext.GLOBAL,
        require_fallback=True
    )
    
    print("Sending request:")
    print(f"  Content: {request.emoji_content}")
    print(f"  Content Type: {request.content_type.value}")
    print(f"  Domain: {request.domain.value}")
    print(f"  Cultural Context: {request.cultural_context.value}")
    print()
    
    # Process the request
    response = endpoint.process_request(request)
    
    print("Received response:")
    print(f"  Status: {response.status.name}")
    print(f"  Content: {response.emoji_content}")
    
    if response.metadata:
        print("  Metadata:")
        print(f"    Source Domain: {response.metadata.source_domain.value}")
        print(f"    Cultural Context: {response.metadata.cultural_context.value}")
        print(f"    Confidence Score: {response.metadata.confidence_score}")
        
    if response.fallback:
        print("  Fallback:")
        print(f"    Text: {response.fallback.text_representation}")
        print(f"    Confidence: {response.fallback.translation_confidence}")
    
    print()


def demonstrate_content_negotiation():
    """Demonstrate content negotiation for emoji-only preference."""
    print("\n=== Content Negotiation for Emoji-Only Preference ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Request with different content types
    content_types = [
        EmojiContentType.EMOJI_SEQUENCE,
        EmojiContentType.EMOJI_JSON,
        EmojiContentType.EMOJI_GRAMMAR,
        EmojiContentType.EMOJI_DIALOGUE,
        EmojiContentType.EMOJI_METADATA,
        EmojiContentType.EMOJI_FALLBACK
    ]
    
    for content_type in content_types:
        # Create request with this content type
        request = EmojiRequest(
            emoji_content="👨‍💻🔍🐛✅",
            content_type=content_type,
            domain=EmojiDomain.TECHNICAL
        )
        
        print(f"Requesting with {content_type.value}:")
        
        try:
            # Process the request
            response = endpoint.process_request(request)
            
            print(f"  Status: {response.status.name}")
            print(f"  Response Content Type: {response.content_type.value}")
            print(f"  Content: {response.emoji_content}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()


def demonstrate_error_handling():
    """Demonstrate specialized error handling using emoji error codes."""
    print("\n=== Emoji Error Handling ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Sample error scenarios
    error_scenarios = [
        ("Empty Content", EmojiRequest(emoji_content="")),
        ("Unsupported Content Type", EmojiRequest(
            emoji_content="🚫",
            content_type=EmojiContentType.EMOJI_GRAMMAR  # Assuming not implemented
        )),
        ("Authentication Failure", EmojiRequest(
            emoji_content="🔑🔒",
            authentication={"method": EmojiAuthMethod.EMOJI_KEY.value, "key": "🔑"}
        )),
        ("Invalid Emoji Sequence", EmojiRequest(
            emoji_content="not_an_emoji_sequence"
        )),
    ]
    
    # Process each error scenario
    for name, request in error_scenarios:
        print(f"Scenario: {name}")
        
        # Process the request
        response = endpoint.process_request(request)
        
        print(f"  Status: {response.status.name}")
        print(f"  Error Emoji: {response.emoji_content}")
        
        if response.fallback:
            print(f"  Error Message: {response.fallback.text_representation}")
            
            if response.fallback.reason:
                print(f"  Reason: {response.fallback.reason}")
        
        print()
    
    # Show all error codes
    print("Available Emoji Error Codes:")
    for error_code in EmojiErrorCode:
        print(f"  {error_code.name}: {error_code.value}")
    
    print()


def demonstrate_emoji_authentication():
    """Demonstrate emoji-based authentication methods."""
    print("\n=== Emoji-Based Authentication ===\n")
    
    # Initialize the endpoint with a mock authentication handler
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Define a custom emoji key authentication handler for demo purposes
    def custom_emoji_key_handler(request):
        # For demo purposes: accept keys with at least 3 emojis
        key = request.authentication.get("key", "")
        return len(key) >= 3 and "🔑" in key
    
    # Register the custom handler
    endpoint.register_auth_handler(EmojiAuthMethod.EMOJI_KEY, custom_emoji_key_handler)
    
    # Authentication scenarios
    auth_scenarios = [
        ("Valid Key", {"method": EmojiAuthMethod.EMOJI_KEY.value, "key": "🔑🔒🔐"}),
        ("Invalid Key", {"method": EmojiAuthMethod.EMOJI_KEY.value, "key": "🔒"}),
        ("Unknown Method", {"method": "emoji_unknown", "key": "🔑🔒🔐"}),
        ("No Authentication", None)
    ]
    
    # Test each scenario
    for name, auth_data in auth_scenarios:
        print(f"Scenario: {name}")
        
        # Create request with authentication
        request = EmojiRequest(
            emoji_content="🔍👀",
            authentication=auth_data
        )
        
        # Process the request
        response = endpoint.process_request(request)
        
        print(f"  Status: {response.status.name}")
        print(f"  Content: {response.emoji_content}")
        
        if response.fallback:
            print(f"  Message: {response.fallback.text_representation}")
        
        print()
    
    # Show available authentication methods
    print("Available Emoji Authentication Methods:")
    for auth_method in EmojiAuthMethod:
        print(f"  {auth_method.name}: {auth_method.value}")
    
    print()


def demonstrate_metadata_and_interpretation():
    """Demonstrate metadata to help interpret emoji sequences."""
    print("\n=== Emoji Metadata and Interpretation ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Create a request with different domain contexts
    emoji_sequence = "🚀🔥💻🐛✅"
    domains = [
        EmojiDomain.GENERAL,
        EmojiDomain.TECHNICAL,
        EmojiDomain.BUSINESS,
        EmojiDomain.SOCIAL
    ]
    
    for domain in domains:
        # Create request for this domain
        request = EmojiRequest(
            emoji_content=emoji_sequence,
            domain=domain,
            content_type=EmojiContentType.EMOJI_METADATA,  # Request with metadata
            require_fallback=True
        )
        
        print(f"Domain: {domain.value}")
        print(f"  Emoji: {emoji_sequence}")
        
        # Process the request
        response = endpoint.process_request(request)
        
        if response.fallback:
            print(f"  Interpretation: {response.fallback.text_representation}")
        
        if response.metadata:
            print("  Metadata:")
            print(f"    Confidence: {response.metadata.confidence_score}")
            print(f"    Ambiguity: {response.metadata.ambiguity_score}")
            if response.metadata.intended_sentiment:
                print(f"    Sentiment: {response.metadata.intended_sentiment}")
        
        print()
    
    # Example metadata for a complex technical emoji sequence
    complex_sequence = "🔒⚠️🔥💻🛠️👨‍💻🕒"
    
    # Create metadata for interpretation
    metadata = EmojiMetadata(
        source_domain=EmojiDomain.TECHNICAL,
        cultural_context=CulturalContext.GLOBAL,
        translation_mode="semantic",
        optimization_profile=OptimizationProfile.TECHNICAL.value,
        grammar_patterns=["alert_pattern", "action_required"],
        confidence_score=0.85,
        intended_sentiment="urgent",
        ambiguity_score=0.2,
        context_reference="security_incident"
    )
    
    print("Complex Technical Message:")
    print(f"  Emoji: {complex_sequence}")
    print("  Metadata-enhanced interpretation:")
    print("    Security alert: Critical server vulnerability detected.")
    print("    System is at risk. Engineer action required immediately.")
    print("    Time-sensitive response needed.")
    print()


def demonstrate_fallback_mechanisms():
    """Demonstrate fallback mechanisms when emoji communication fails."""
    print("\n=== Emoji Fallback Mechanisms ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Scenarios that might trigger fallbacks
    fallback_scenarios = [
        ("Ambiguous Sequence", "🤔💭❓"),
        ("Abstract Concept", "🧠💡🌀"),
        ("Domain Mismatch", "📊📈💹"),
        ("Cultural Specificity", "🧧🎏"),
        ("Unsupported Characters", "🚫emoji🚫")
    ]
    
    # Process each scenario
    for name, content in fallback_scenarios:
        print(f"Scenario: {name}")
        print(f"  Content: {content}")
        
        # Create request with fallback required
        request = EmojiRequest(
            emoji_content=content,
            require_fallback=True
        )
        
        # Process the request
        response = endpoint.process_request(request)
        
        if response.status != EmojiErrorCode.SUCCESS:
            print(f"  Status: {response.status.name}")
        
        if response.fallback:
            print(f"  Fallback Text: {response.fallback.text_representation}")
            print(f"  Confidence: {response.fallback.translation_confidence}")
            
            if response.fallback.reason:
                print(f"  Reason: {response.fallback.reason}")
            
            if response.fallback.alternative_emoji_sequences:
                alternatives = ", ".join(response.fallback.alternative_emoji_sequences)
                print(f"  Alternatives: {alternatives}")
            
            if response.fallback.guidance:
                print(f"  Guidance: {response.fallback.guidance}")
        
        print()
    
    # Example of a complete fallback with alternatives
    complete_fallback = EmojiFallback(
        text_representation="Critical security vulnerability requiring immediate attention",
        translation_confidence=0.6,
        alternative_emoji_sequences=["🔒⚠️🔥", "⚠️🔒❗"],
        reason="Original sequence contained domain-specific technical terms",
        guidance="Use more universal security-related emojis",
        retry_suggestions=["Try using standard alert emojis", "Include timing indicators"]
    )
    
    print("Complete Fallback Example:")
    print(f"  Text: {complete_fallback.text_representation}")
    print(f"  Confidence: {complete_fallback.translation_confidence}")
    print(f"  Alternatives: {', '.join(complete_fallback.alternative_emoji_sequences)}")
    print(f"  Reason: {complete_fallback.reason}")
    print(f"  Guidance: {complete_fallback.guidance}")
    print(f"  Suggestions: {', '.join(complete_fallback.retry_suggestions)}")
    print()


async def demonstrate_async_communication():
    """Demonstrate asynchronous emoji communication patterns."""
    print("\n=== Asynchronous Emoji Communication ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Create multiple requests
    requests = [
        EmojiRequest(emoji_content="👋", domain=EmojiDomain.GENERAL),
        EmojiRequest(emoji_content="🚀💻", domain=EmojiDomain.TECHNICAL),
        EmojiRequest(emoji_content="📈💰", domain=EmojiDomain.BUSINESS),
        EmojiRequest(emoji_content="🎮🎲", domain=EmojiDomain.SOCIAL)
    ]
    
    print("Sending multiple requests asynchronously...")
    
    # Process asynchronously
    async_tasks = [endpoint.process_request_async(req) for req in requests]
    responses = await asyncio.gather(*async_tasks)
    
    print("Received all responses:")
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}:")
        print(f"  Request: {requests[i].emoji_content}")
        print(f"  Domain: {requests[i].domain.value}")
        print(f"  Response: {response.emoji_content}")
        print(f"  Status: {response.status.name}")
        print()


async def demonstrate_emoji_dialogue_session():
    """Demonstrate a multi-turn emoji dialogue session."""
    print("\n=== Multi-turn Emoji Dialogue Session ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Create a new dialogue session
    session_id = endpoint.create_dialogue_session(
        domain=EmojiDomain.TECHNICAL,
        cultural_context=CulturalContext.GLOBAL,
        optimization_profile=OptimizationProfile.PRECISE
    )
    
    print(f"Created dialogue session: {session_id}")
    print("Simulating a technical support conversation...")
    print()
    
    # Simulate a multi-turn conversation
    conversation = [
        # User reports a problem
        ("User", "❓💻🐛"),
        # System asks for details
        ("System", "🔍📋❓"),
        # User provides error details
        ("User", "💻🔄💥❌"),
        # System suggests a solution
        ("System", "💡🔄💾👨‍💻"),
        # User confirms it worked
        ("User", "✅🎉")
    ]
    
    # Process the conversation
    for speaker, message in conversation:
        print(f"{speaker}: {message}")
        
        if speaker == "User":
            # Send the message
            response = await endpoint.send_dialogue_message_async(
                session_id,
                message,
                require_fallback=True
            )
            
            # In a real implementation, this would be handled by the dialogue manager
            # For this example, we just use the existing response mechanism
            
            if response.status != EmojiErrorCode.SUCCESS:
                print(f"  Error: {response.status.name}")
            
            if response.fallback:
                print(f"  Interpretation: {response.fallback.text_representation}")
        
        # Add some delay for readability
        await asyncio.sleep(0.5)
    
    print("\nConversation complete.")
    endpoint.close_dialogue_session(session_id)
    print(f"Session {session_id} closed.")
    
    
def demonstrate_text_to_emoji_translation():
    """Demonstrate text-to-emoji translation with the endpoint."""
    print("\n=== Text-to-Emoji Translation ===\n")
    
    # Initialize the endpoint
    kb = EmojiKnowledgeBase(load_default=True)
    endpoint = EmojiCommunicationEndpoint(knowledge_base=kb)
    
    # Sample messages to translate
    messages = [
        ("Hello world!", EmojiDomain.GENERAL),
        ("The server is down", EmojiDomain.TECHNICAL),
        ("Let's schedule a meeting tomorrow", EmojiDomain.BUSINESS),
        ("I'm so excited about the party!", EmojiDomain.SOCIAL)
    ]
    
    # Translation options
    optimization_profiles = [
        OptimizationProfile.PRECISE,
        OptimizationProfile.CONCISE,
        OptimizationProfile.EXPRESSIVE
    ]
    
    for text, domain in messages:
        print(f"Text: \"{text}\"")
        print(f"Domain: {domain.value}")
        
        for profile in optimization_profiles:
            # Translate with this optimization profile
            response = endpoint.translate_text_to_emoji(
                text,
                domain=domain,
                optimization_profile=profile,
                include_metadata=True,
                include_fallback=True
            )
            
            print(f"  {profile.value.capitalize()}: {response.emoji_content}")
            
            if response.fallback and response.fallback.text_representation != text:
                print(f"    Fallback: {response.fallback.text_representation}")
        
        print()


def demonstrate_real_world_api_usage():
    """Demonstrate real-world API usage scenarios."""
    print("\n=== Real-World API Usage Examples ===\n")
    
    # Example 1: JSON API for emoji-only chat
    emoji_chat_api = {
        "endpoint": "/api/emoji-chat",
        "method": "POST",
        "headers": {
            "Content-Type": "application/x-emoji-json",
            "Accept": "application/x-emoji-json",
            "X-Emoji-Only": "true",
            "X-Emoji-Domain": "social",
            "X-Emoji-Cultural-Context": "global"
        },
        "request_body": {
            "emoji_content": "👋😊",
            "session_id": "abc123",
            "optimization_profile": "expressive",
            "require_fallback": True
        },
        "response_example": {
            "status": "✅",
            "emoji_content": "👋😊👍",
            "metadata": {
                "source_domain": "social",
                "cultural_context": "global",
                "confidence_score": 0.95
            },
            "fallback": {
                "text_representation": "Hello! Nice to meet you!",
                "translation_confidence": 0.9
            }
        }
    }
    
    print("Example 1: Emoji Chat API")
    print("  Endpoint: " + emoji_chat_api["endpoint"])
    print("  Method: " + emoji_chat_api["method"])
    print("  Headers:")
    for key, value in emoji_chat_api["headers"].items():
        print(f"    {key}: {value}")
    print("  Request:")
    print(json.dumps(emoji_chat_api["request_body"], indent=4))
    print("  Response Example:")
    print(json.dumps(emoji_chat_api["response_example"], indent=4))
    print()
    
    # Example 2: RESTful API with emoji error codes
    emoji_error_api = {
        "endpoint": "/api/resources/{id}",
        "method": "GET",
        "headers": {
            "Accept": "application/json",
            "X-Error-Format": "emoji+text"
        },
        "error_responses": {
            "404": {
                "status": 404,
                "emoji_code": "🔍❌",
                "message": "Resource not found",
                "fallback_text": "The requested resource could not be found"
            },
            "401": {
                "status": 401,
                "emoji_code": "🔒🚫",
                "message": "Unauthorized",
                "fallback_text": "Authentication is required to access this resource"
            },
            "500": {
                "status": 500,
                "emoji_code": "🔥💻",
                "message": "Server Error",
                "fallback_text": "An internal server error occurred"
            }
        }
    }
    
    print("Example 2: RESTful API with Emoji Error Codes")
    print("  Endpoint: " + emoji_error_api["endpoint"])
    print("  Method: " + emoji_error_api["method"])
    print("  Headers:")
    for key, value in emoji_error_api["headers"].items():
        print(f"    {key}: {value}")
    print("  Error Responses:")
    for code, response in emoji_error_api["error_responses"].items():
        print(f"    {code}: {response['emoji_code']} - {response['message']}")
    print()
    
    # Example 3: WebSocket API for real-time emoji communication
    emoji_websocket_api = {
        "endpoint": "ws://api.example.com/emoji-socket",
        "connection_params": {
            "domain": "technical",
            "cultural_context": "global",
            "optimization_profile": "precise",
            "authentication": {
                "method": "emoji_token",
                "token": "🔑🔒🔐"
            }
        },
        "message_format": {
            "type": "emoji_message",
            "emoji_content": "🔍🐛💻",
            "require_fallback": True,
            "metadata": {
                "session_id": "xyz789",
                "timestamp": 1621234567890
            }
        },
        "events": {
            "connection_established": "🔗✅",
            "connection_closed": "🔗❌",
            "message_received": "📩✅",
            "error": "🚫"
        }
    }
    
    print("Example 3: WebSocket API for Real-time Emoji Communication")
    print("  Endpoint: " + emoji_websocket_api["endpoint"])
    print("  Connection Parameters:")
    print(json.dumps(emoji_websocket_api["connection_params"], indent=4))
    print("  Message Format:")
    print(json.dumps(emoji_websocket_api["message_format"], indent=4))
    print("  Events:")
    for event, emoji in emoji_websocket_api["events"].items():
        print(f"    {event}: {emoji}")
    print()


async def main():
    """Run all emoji communication endpoint demonstrations."""
    print("="*80)
    print("           EmojiCommunicationEndpoint Demonstrations")
    print("="*80)
    
    demonstrate_synchronous_communication()
    demonstrate_content_negotiation()
    demonstrate_error_handling()
    demonstrate_emoji_authentication()
    demonstrate_metadata_and_interpretation()
    demonstrate_fallback_mechanisms()
    demonstrate_text_to_emoji_translation()
    demonstrate_real_world_api_usage()
    
    # Asynchronous examples
    await demonstrate_async_communication()
    await demonstrate_emoji_dialogue_session()
    
    print("="*80)
    print("                     Demonstrations Complete")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
