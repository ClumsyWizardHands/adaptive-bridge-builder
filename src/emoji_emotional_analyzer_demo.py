"""
Demonstration of the EmojiEmotionalAnalyzer functionality.

This script provides real-world examples of how to use the EmojiEmotionalAnalyzer
to detect emotions in emoji sequences, track conversations, analyze emotional shifts,
and generate culturally-appropriate responses.
"""

from emoji_emotional_analyzer import (
    EmojiEmotionalAnalyzer,
    EmotionCategory,
    EmotionIntensity,
    CulturalContext,
    ResponseTone
)

def run_demo():
    """Run a comprehensive demonstration of the EmojiEmotionalAnalyzer."""
    print("=" * 80)
    print("EmojiEmotionalAnalyzer Demonstration")
    print("=" * 80)
    
    # Create an analyzer instance
    analyzer = EmojiEmotionalAnalyzer()
    
    # 1. Basic Emotion Detection
    print("\n1. EMOTION DETECTION EXAMPLES")
    print("-" * 40)
    
    examples = [
        "😊",                  # Simple joy
        "😊👍",                # Contentment pattern
        "😭💔",                # Sadness pattern
        "😡💢",                # Anger pattern
        "😨😨😨",              # Fear with intensity
        "🤔❓",                # Curiosity with uncertainty
        "😀😭",                # Mixed emotions
        "😀😀😀❗",            # Joy with high intensity
    ]
    
    for emoji_seq in examples:
        emotion = analyzer.detect_emotion(emoji_seq)
        print(f"\nEmoji: {emoji_seq}")
        print(f"  Primary emotion: {emotion.primary_emotion.name} ({emotion.primary_probability:.2f})")
        if emotion.secondary_emotion:
            print(f"  Secondary emotion: {emotion.secondary_emotion.name} ({emotion.secondary_probability:.2f})")
        print(f"  Intensity: {emotion.intensity.name}")
        print(f"  Confidence: {emotion.confidence:.2f}")
    
    # 2. Emotional State Mapping
    print("\n\n2. EMOTIONAL STATE MAPPING")
    print("-" * 40)
    
    emoji_seq = "😀😭"  # Mixed joy and sadness
    print(f"\nMapping emotional states for: {emoji_seq}")
    emotion_map = analyzer.map_to_emotional_states(emoji_seq)
    
    print("\nEmotional state probabilities:")
    # Sort by probability
    sorted_emotions = sorted(emotion_map.items(), key=lambda x: x[1], reverse=True)
    for emotion, probability in sorted_emotions:
        print(f"  {emotion.name}: {probability:.2f}")
    
    # 3. Conversation Tracking and Emotional Shift Detection
    print("\n\n3. CONVERSATION TRACKING & EMOTIONAL SHIFTS")
    print("-" * 40)
    
    # Reset conversation history for the demo
    analyzer.shift_tracker.conversation_history = []
    analyzer.shift_tracker.emotion_history = []
    
    # Sample conversation
    conversation = [
        ("😊👍", True),           # User is happy
        ("🎉😄", False),          # Agent responds positively
        ("😊", True),             # User still happy
        ("👍✨", False),          # Agent responds positively
        ("😔", True),             # User becomes sad - emotional shift
        ("🫂❤️", False),          # Agent responds with support
        ("😢💔", True),           # User more sad
        ("🌈✨", False),          # Agent responds with encouragement
        ("😊", True),             # User becomes happy again - another shift
    ]
    
    print("\nTracking conversation:")
    for i, (emoji_seq, is_user) in enumerate(conversation):
        analyzer.track_conversation(emoji_seq, is_user)
        sender = "User" if is_user else "Agent"
        print(f"{i+1}. {sender}: {emoji_seq}")
        
        if is_user and i > 0:  # Check for shifts after each user message (except the first)
            shift = analyzer.detect_emotional_shift()
            if shift:
                print(f"   [SHIFT DETECTED] {shift.from_state.primary_emotion.name} → {shift.to_state.primary_emotion.name}")
                print(f"   Magnitude: {shift.magnitude:.2f}, Trigger: {shift.detected_trigger}")
                print(f"   Pattern: {shift.temporal_pattern}")
    
    # 4. Response Generation
    print("\n\n4. RESPONSE GENERATION")
    print("-" * 40)
    
    emotion_examples = [
        ("😊", None),                              # Joy with default tone
        ("😭", None),                              # Sadness with default tone
        ("😡", None),                              # Anger with default tone
        ("😨", None),                              # Fear with default tone
        ("😔", ResponseTone.ENCOURAGING),          # Sadness with encouraging tone
        ("😡😡❗", ResponseTone.CALMING),          # Intense anger with calming tone
    ]
    
    print("\nGenerating responses to different emotional states:")
    for emoji_seq, tone in emotion_examples:
        response = analyzer.generate_response(emoji_seq, tone)
        
        tone_str = f" with {tone.name} tone" if tone else ""
        print(f"\nInput emoji: {emoji_seq} ({response.emotional_intent.split(' with ')[0]}){tone_str}")
        print(f"  Response: {response.emoji_sequence}")
        print(f"  Intent: {response.emotional_intent}")
        print(f"  Principle alignment: {response.principle_alignment_score:.2f}")
        
        if response.alternative_sequences:
            print("  Alternative responses:")
            for alt_seq, alt_intent, alt_score in response.alternative_sequences:
                print(f"    {alt_seq} - {alt_intent} (alignment: {alt_score:.2f})")
    
    # 5. Cultural Adaptation
    print("\n\n5. CULTURAL ADAPTATION")
    print("-" * 40)
    
    cultural_contexts = [
        CulturalContext.GLOBAL,
        CulturalContext.WESTERN,
        CulturalContext.EASTERN_ASIAN,
    ]
    
    print("\nGenerating culturally adapted responses to joy emoji:")
    emoji_seq = "😊"  # Joy
    
    for context in cultural_contexts:
        analyzer.adapt_to_cultural_context(context)
        response = analyzer.generate_response(emoji_seq)
        
        print(f"\n{context.name} context:")
        print(f"  Response: {response.emoji_sequence}")
        
        if response.cultural_adaptations:
            print("  Cultural adaptations:")
            for ctx, adaptation in response.cultural_adaptations.items():
                print(f"    {ctx.name}: {adaptation}")

if __name__ == "__main__":
    run_demo()
