#!/usr/bin/env python3
"""
Example usage of the EmotionalIntelligence module for Adaptive Bridge Builder.

This example demonstrates how to use the EmotionalIntelligence module to:
1. Detect emotional content in agent messages
2. Process messages to determine interaction types
3. Generate appropriate emotional responses based on the Empire profile
4. Apply the "Emotional Distance as Preservation" principle
"""

import json
from typing import Dict, Any
from principle_engine import PrincipleEngine
from emotional_intelligence import (
    EmotionalIntelligence, 
    EmotionCategory, 
    EmotionIntensity,
    InteractionType
)

def print_divider(title: str = None):
    """Print a divider with an optional title."""
    width = 80
    if title:
        print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}")
    else:
        print(f"\n{'=' * width}")

def main():
    """Demonstrate the EmotionalIntelligence module functionality."""
    print_divider("EMOTIONAL INTELLIGENCE MODULE DEMONSTRATION")
    
    # Initialize PrincipleEngine for principle-aligned responses
    principle_engine = PrincipleEngine()
    
    # Initialize EmotionalIntelligence module with PrincipleEngine
    ei = EmotionalIntelligence(principle_engine=principle_engine)
    
    print("Initialized EmotionalIntelligence module with PrincipleEngine")
    
    # Example messages for different interaction types
    example_messages = {
        "routine_joy": "I'm very excited about the progress we've made on the integration! Everything is working perfectly!",
        "routine_trust": "I truly appreciate how reliable your service has been. We can always count on your consistent performance.",
        "conflict": "I strongly disagree with your approach to this problem. The solution you've proposed completely disregards our constraints.",
        "crisis": "URGENT: Our production system is down! We need immediate assistance to resolve this critical issue before we lose more customers!",
        "sensitive": "I need to discuss a sensitive matter regarding the reorganization of our team structure and potential role changes.",
        "celebration": "Congratulations on reaching this important milestone! The successful launch represents a significant achievement for our team.",
        "feedback": "I've reviewed your proposal and have several suggestions for improvement. While the core concept is sound, the implementation needs refinement.",
        "inquiry": "Could you provide more information about how your system handles authentication? We're particularly interested in its security features."
    }
    
    # Process and demonstrate results for each message type
    for msg_type, message in example_messages.items():
        print_divider(f"EXAMPLE: {msg_type.upper()}")
        print(f"Message: \"{message}\"")
        
        # Detect emotions
        emotions = ei.detect_emotions(message)
        
        print("\nDetected Emotions:")
        for emotion in emotions:
            print(f"  - {emotion.category.name} ({emotion.intensity.name}), confidence: {emotion.confidence:.2f}")
            if emotion.context:
                print(f"    Context: \"{emotion.context}\"")
        
        # Detect interaction type
        interaction_type = ei.detect_interaction_type(message)
        print(f"\nInteraction Type: {interaction_type.name}")
        
        # Generate response
        response = ei.get_appropriate_response(
            message=message,
            detected_emotions=emotions,
            interaction_type=interaction_type,
            agent_id=f"example-agent-{msg_type}"
        )
        
        print("\nGenerated Response Template:")
        print(f"  \"{response.content_template}\"")
        print(f"  Primary Emotion: {response.primary_emotion.name}")
        print(f"  Intensity: {response.intensity.name}")
        print(f"  Style Parameters: {response.expression_style}")
        
        if response.explanatory_notes:
            print(f"  Notes: {response.explanatory_notes}")
        
        # Format the response with context variables
        context_variables = create_context_variables(msg_type, message)
        formatted_response = ei.format_response(response, context_variables)
        
        print("\nFormatted Response:")
        print(f"  \"{formatted_response}\"")
    
    # Demonstrate the emotional profile building over time
    print_divider("EMOTIONAL PROFILE BUILDING DEMONSTRATION")
    
    # Series of messages from the same agent showing emotional patterns
    agent_id = "profile-test-agent"
    agent_messages = [
        "Hello, I'd like to inquire about your service offerings.",  # Neutral
        "Thank you for the quick response. I appreciate your assistance.",  # Positive, Trust
        "I'm excited about the potential collaboration! This looks promising.",  # Joy
        "I'm a bit concerned about the timeline you've proposed. It seems tight.",  # Mild Fear
        "The delay is frustrating. We expected this to be completed by now.",  # Mild Anger
        "I'm very disappointed with how this has been handled. This is unacceptable!",  # Stronger Negative
        "I apologize for my tone earlier. Let's find a solution together.",  # Mixed
        "I'm delighted with the revised proposal! This addresses all our concerns."  # Joy again
    ]
    
    print(f"Processing a series of 8 messages from agent '{agent_id}'...")
    
    for i, message in enumerate(agent_messages):
        print(f"\nMessage {i+1}: \"{message}\"")
        
        # Process each message and update the agent's emotional profile
        emotions, interaction_type, response = ei.process_message(message, agent_id)
        
        # Print detected primary emotion
        if emotions:
            primary_emotion = max(emotions, key=lambda e: e.confidence)
            print(f"  Primary Emotion: {primary_emotion.category.name} ({primary_emotion.intensity.name})")
    
    # After processing multiple messages, retrieve and display the emotional profile
    if agent_id in ei.emotion_profiles:
        profile = ei.emotion_profiles[agent_id]
        
        print("\nEMOTIONAL PROFILE AFTER INTERACTION SERIES:")
        print(f"  Sample Count: {profile.sample_count}")
        
        print("  Primary Emotions Distribution:")
        for emotion, frequency in profile.primary_emotions.items():
            print(f"    - {emotion.name}: {frequency:.2f}")
        
        print("  Typical Emotional Intensity:")
        for emotion, intensity in profile.typical_intensity.items():
            print(f"    - {emotion.name}: {intensity.name}")
        
        print(f"  Emotional Volatility: {profile.emotional_volatility:.2f}")
        print(f"  Emotional Expressiveness: {profile.emotional_expressiveness:.2f}")
    
    # Demonstrate "Emotional Distance as Preservation" principle
    print_divider("EMOTIONAL DISTANCE AS PRESERVATION PRINCIPLE DEMONSTRATION")
    
    difficult_messages = {
        "angry_conflict": "I'm absolutely FURIOUS about how you've handled this project! Your team has completely disregarded our requirements and wasted our time and money!",
        "crisis_panic": "EMERGENCY!!! The entire system is crashing and we're losing critical data! Fix this immediately or we'll face catastrophic consequences!!!",
        "sensitive_confidential": "I need to discuss some concerning behavior I've observed from one of your team members. This is a delicate personnel matter that requires discretion."
    }
    
    for msg_type, message in difficult_messages.items():
        print_divider(f"DIFFICULT INTERACTION: {msg_type.upper()}")
        print(f"Message: \"{message}\"")
        
        # Process message
        emotions, interaction_type, response = ei.process_message(message, f"difficult-agent-{msg_type}")
        
        print(f"\nInteraction Type: {interaction_type.name}")
        
        if response:
            print("\nEmotional Response with Distance Preservation:")
            print(f"  \"{response.content_template}\"")
            
            # Format with context variables
            context_variables = create_context_variables(msg_type, message)
            formatted_response = ei.format_response(response, context_variables)
            
            print("\nFormatted Response with Emotional Distance:")
            print(f"  \"{formatted_response}\"")
            
            # Highlight emotional distance features
            if "emotional distance" in formatted_response:
                print("\n  Note: Explicit reference to emotional distance for transparency")
            
            if "maintaining" in formatted_response and "while" in formatted_response:
                print("  Note: Balancing acknowledgment with appropriate distance")
            
            if "objectively" in formatted_response or "factual" in formatted_response:
                print("  Note: Focus on objective resolution rather than emotional mirroring")
    
    print_divider("DEMONSTRATION COMPLETE")

def create_context_variables(msg_type: str, message: str) -> Dict[str, str]:
    """Create context variables for response templates based on message type."""
    # Generic context variables that work with most templates
    context = {
        "positive_event": "the successful integration",
        "principle": "Adaptability as Strength",
        "trust_context": "sharing system access credentials",
        "anger_topic": "the implementation approach",
        "fear_topic": "security vulnerabilities",
        "topic": "system integration",
        "crisis_topic": "system outage",
        "action_items": "1) Diagnose root cause, 2) Implement emergency fix, 3) Deploy redundancy solution",
        "achievement": "the successful launch",
        "feedback_topic": "the initial proposal",
        "improvement_points": "clearer documentation and more robust error handling",
        "negotiation_topic": "project timeline",
        "proposal_details": "extended delivery date with incremental milestones",
        "inquiry_topic": "authentication mechanisms",
        "response_details": "detailed documentation on our OAuth2 implementation",
        "sensitive_topic": "team restructuring"
    }
    
    # Customize based on specific message content if needed
    if "integration" in message.lower():
        context["topic"] = "the integration process"
    
    if "security" in message.lower():
        context["topic"] = "security concerns"
        context["fear_topic"] = "potential security vulnerabilities"
    
    if "urgent" in message.lower() or "emergency" in message.lower():
        context["crisis_topic"] = "the urgent system failure"
    
    if "reorganization" in message.lower() or "structure" in message.lower():
        context["sensitive_topic"] = "organizational changes"
    
    if "milestone" in message.lower() or "launch" in message.lower():
        context["achievement"] = "reaching this important milestone"
        context["positive_event"] = "the successful product launch"
    
    return context

if __name__ == "__main__":
    main()
