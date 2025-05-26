# Emoji Emotional Analyzer

A sophisticated component for analyzing emotional content in emoji sequences, tracking emotional shifts in conversations, and generating emotionally appropriate responses.

## Overview

The `EmojiEmotionalAnalyzer` is a core component of the Adaptive Bridge Builder's emoji communication system. It enables the agent to understand emotional nuances in emoji-based communication, track changes in emotional states during conversations, and respond with emotionally appropriate emoji sequences.

This component embodies the Empire Framework principles of empathy, adaptability, fairness, harmony, and resilience by:
- Accurately detecting emotions in emoji messages
- Adapting responses based on cultural context
- Providing equitable emotional communication across diverse backgrounds
- Ensuring response alignment with agent principles
- Recognizing and adapting to changes in user emotional states

## Key Features

- **Emotion Detection**: Analyzes emoji sequences to detect primary and secondary emotions with probability scores
- **Emotion Mapping**: Maps emojis to multiple emotional states simultaneously for nuanced understanding
- **Emotional Shift Detection**: Tracks conversations to identify significant emotional shifts and their patterns
- **Response Generation**: Creates emotionally appropriate responses with different tones
- **Cultural Adaptation**: Adapts interpretation and responses based on cultural context
- **Principle Alignment**: Ensures generated responses align with agent principles
- **Performance Optimization**: Efficiently processes emoji sequences with minimal latency

## Installation

```bash
# Install the package
pip install adaptive-bridge-builder

# Or for development
git clone https://github.com/adaptive-bridge-builder/adaptive-bridge-builder.git
cd adaptive-bridge-builder
pip install -e .
```

## Quick Start

```python
from emoji_emotional_analyzer import EmojiEmotionalAnalyzer

# Create an analyzer instance
analyzer = EmojiEmotionalAnalyzer()

# Detect emotion in an emoji sequence
emotion = analyzer.detect_emotion("😊👍")
print(f"Primary emotion: {emotion.primary_emotion.name}")
print(f"Probability: {emotion.primary_probability:.2f}")
print(f"Intensity: {emotion.intensity.name}")

# Generate an appropriate response
response = analyzer.generate_response("😭")
print(f"Response: {response.emoji_sequence}")
print(f"Emotional intent: {response.emotional_intent}")
```

## Core Components

The `EmojiEmotionalAnalyzer` consists of four main subcomponents:

1. **EmotionDetectionEngine**: Parses emoji sequences and detects primary and secondary emotions with their intensities
2. **EmotionMappingSystem**: Maps emoji combinations to multiple emotional states with probability scores
3. **EmotionalShiftTracker**: Tracks emotional changes across conversations to detect significant shifts
4. **EmojiResponseGenerator**: Generates emotionally appropriate emoji responses based on detected emotions

## Advanced Usage

### Tracking Emotional Shifts in Conversations

```python
# Track a conversation
analyzer.track_conversation("😀")  # User is happy
analyzer.track_conversation("👍", is_user_message=False)  # Agent response
analyzer.track_conversation("😔")  # User becomes sad

# Detect emotional shifts
shift = analyzer.detect_emotional_shift()
if shift:
    print(f"Detected shift from {shift.from_state.primary_emotion.name} to {shift.to_state.primary_emotion.name}")
    print(f"Magnitude: {shift.magnitude:.2f}")
    print(f"Trigger: {shift.detected_trigger}")
```

### Cultural Adaptation

```python
# Adapt to a different cultural context
analyzer.adapt_to_cultural_context(CulturalContext.EASTERN_ASIAN)

# Generate a culturally adapted response
response = analyzer.generate_response("😀")
print(f"Eastern Asian response: {response.emoji_sequence}")

# Display all cultural adaptations
for context, adaptation in response.cultural_adaptations.items():
    print(f"{context.name}: {adaptation}")
```

### Custom Response Tones

```python
# Generate response with a specific tone
response = analyzer.generate_response("😔", desired_tone=ResponseTone.ENCOURAGING)
print(f"Encouraging response: {response.emoji_sequence}")
print(f"Intent: {response.emotional_intent}")
```

## Integration with Other Components

The `EmojiEmotionalAnalyzer` integrates with several other components of the Adaptive Bridge Builder framework:

- **PrincipleEngine**: For checking alignment of emoji responses with agent principles
- **OrchestrationAnalytics**: For tracking emotional patterns and response effectiveness
- **GrowthJournal**: For recording learning about emotional patterns and successful strategies
- **EmojiKnowledgeBase**: For accessing emoji meanings and cultural variations
- **EmojiGrammarSystem**: For structuring emotionally appropriate responses with proper grammar
- **EmojiDialogueManager**: For managing emotionally-aware multi-turn conversations

## Configuration

The analyzer can be configured through:

1. **JSON Configuration File**:
   ```json
   {
     "emoji_kb_path": "path/to/knowledge_base.json",
     "default_cultural_context": "GLOBAL",
     "confidence_threshold": 0.3,
     "shift_magnitude_threshold": 0.3,
     "logging_level": "INFO"
   }
   ```

2. **Environment Variables**:
   ```bash
   export EMOJI_ANALYZER_EMOJI_KB_PATH=path/to/knowledge_base.json
   export EMOJI_ANALYZER_DEFAULT_CULTURAL_CONTEXT=GLOBAL
   export EMOJI_ANALYZER_CONFIDENCE_THRESHOLD=0.3
   export EMOJI_ANALYZER_SHIFT_MAGNITUDE_THRESHOLD=0.3
   export EMOJI_ANALYZER_LOGGING_LEVEL=INFO
   ```

3. **Direct Constructor Parameters**:
   ```python
   analyzer = EmojiEmotionalAnalyzer(
       emoji_kb_path="path/to/knowledge_base.json",
       cultural_context=CulturalContext.GLOBAL,
       confidence_threshold=0.3,
       shift_magnitude_threshold=0.3,
       logging_level=LogLevel.INFO
   )
   ```

## Documentation and Examples

For comprehensive documentation and examples, see:

- [API Documentation](./emoji_emotional_analyzer_documentation.md)
- [Basic Demo](./emoji_emotional_analyzer_demo.py)
- [Enhanced Demo](./emoji_emotional_analyzer_enhanced_demo.py)
- [Unit Tests](./test_emoji_emotional_analyzer.py)
- [Enhanced Tests](./test_emoji_emotional_analyzer_enhanced.py)

## Performance Considerations

The `EmojiEmotionalAnalyzer` is designed for real-time analysis and response generation:

- Typical emotion detection takes < 50ms per sequence
- Response generation typically takes < 100ms per response
- Memory usage is typically < 50MB

For high-volume applications, consider enabling the cache for frequently analyzed emoji sequences.

## Contributing

Contributions to improve the `EmojiEmotionalAnalyzer` are welcome. Areas for potential improvement include:

- Additional emotion categories for more nuanced detection
- More sophisticated cultural adaptation algorithms
- Enhanced emotional shift detection with better pattern recognition
- Optimization for very large emoji sequences
- Integration with additional analytics systems

## License

This component is part of the Adaptive Bridge Builder framework and is licensed under the same terms as the parent project.

## Related Components

- [EmojiTranslationEngine](./emoji_translation_engine_README.md)
- [EmojiGrammarSystem](./emoji_grammar_system_README.md)
- [EmojiDialogueManager](./emoji_dialogue_manager_README.md)
- [EmojiKnowledgeBase](./emoji_knowledge_base_README.md)
- [EmojiSequenceOptimizer](./emoji_sequence_optimizer_README.md)
- [EmojiCommunicationEndpoint](./emoji_communication_endpoint_README.md)
