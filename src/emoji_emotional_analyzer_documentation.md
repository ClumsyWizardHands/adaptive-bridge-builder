# EmojiEmotionalAnalyzer Documentation

## Overview

The EmojiEmotionalAnalyzer is a comprehensive system for analyzing emotional content in emoji sequences, tracking emotional shifts in conversations, and generating emotionally appropriate responses that can be culturally adapted. This system is designed to bring emotional intelligence to emoji-only or emoji-rich communications.

## Core Components

### 1. EmotionDetectionEngine

The foundational component that detects emotional content in emoji sequences.

- **Capabilities:**
  - Identify primary and secondary emotions in emoji sequences
  - Calculate emotional intensity based on patterns and repetition
  - Recognize complex emotional patterns (combinations of emojis)
  - Apply modifiers that adjust emotional intensity
  - Calculate confidence scores for detected emotions

- **Example:**
  ```python
  engine = EmotionDetectionEngine()
  emotion = engine.detect_emotion("😊👍")
  # Returns an EmotionalState with primary_emotion=CONTENTMENT
  ```

### 2. EmotionMappingSystem

Maps emoji combinations to a full spectrum of emotional states with probability scores.

- **Capabilities:**
  - Convert detected emotions to a probability distribution across all emotions
  - Include related emotions with appropriate probability scores
  - Normalize the emotional state probabilities

- **Example:**
  ```python
  mapping = EmotionMappingSystem()
  emotion_map = mapping.map_to_emotional_states("😊😔")
  # Returns a dict mapping EmotionCategory to probabilities
  # {EmotionCategory.JOY: 0.6, EmotionCategory.SADNESS: 0.3, ...}
  ```

### 3. EmotionalShiftTracker

Tracks and analyzes emotional shifts in conversations over time.

- **Capabilities:**
  - Record conversation history with timestamped entries
  - Detect significant emotional shifts between messages
  - Calculate the magnitude of emotional shifts
  - Identify potential triggers for emotional changes
  - Recognize temporal patterns in emotional changes

- **Example:**
  ```python
  tracker = EmotionalShiftTracker()
  tracker.track_conversation("😊")  # Happy
  tracker.track_conversation("😔")  # Sad
  shift = tracker.detect_emotional_shift()
  # Returns an EmotionalShift object with details about the change
  ```

### 4. EmojiResponseGenerator

Generates emotionally appropriate emoji responses with cultural adaptations.

- **Capabilities:**
  - Generate responses appropriate to the emotional context
  - Adapt response tones based on emotional intensity
  - Apply cultural variations to responses
  - Ensure responses align with agent principles
  - Provide alternative responses when needed

- **Example:**
  ```python
  generator = EmojiResponseGenerator()
  response = generator.generate_response("😭")
  # Returns supportive response like "🫂❤️"
  ```

### 5. EmojiEmotionalAnalyzer

The main class that unifies all components into a cohesive system.

- **Capabilities:**
  - Detect emotions in emoji sequences
  - Map emoji combinations to emotional states
  - Track conversations and detect emotional shifts
  - Generate culturally appropriate emotional responses
  - Adapt to different cultural contexts

- **Example:**
  ```python
  analyzer = EmojiEmotionalAnalyzer()
  analyzer.track_conversation("😊")
  analyzer.track_conversation("😔")
  shift = analyzer.detect_emotional_shift()
  response = analyzer.generate_response("😔")
  ```

## Data Models

### EmotionalState

Represents a detected emotional state with probability scores.

```python
@dataclass
class EmotionalState:
    primary_emotion: EmotionCategory
    primary_probability: float
    secondary_emotion: Optional[EmotionCategory] = None
    secondary_probability: float = 0.0
    intensity: EmotionIntensity = EmotionIntensity.MEDIUM
    confidence: float = 1.0
    context_notes: List[str] = field(default_factory=list)
```

### EmotionalShift

Represents a shift in emotional state during a conversation.

```python
@dataclass
class EmotionalShift:
    from_state: EmotionalState
    to_state: EmotionalState
    magnitude: float  # 0.0 to 1.0
    detected_trigger: Optional[str] = None
    temporal_pattern: Optional[str] = None
    confidence: float = 1.0
```

### EmojiEmotionalResponse

Represents an emoji response with emotional intent information.

```python
@dataclass
class EmojiEmotionalResponse:
    emoji_sequence: str
    emotional_intent: str
    principle_alignment_score: float
    alternative_sequences: List[Tuple[str, str, float]] = field(default_factory=list)
    cultural_adaptations: Dict[CulturalContext, str] = field(default_factory=dict)
    confidence: float = 1.0
```

## Enumerations

### EmotionCategory

Categories of emotions that can be detected in emoji sequences:

- Primary: JOY, SADNESS, ANGER, FEAR, SURPRISE, DISGUST, TRUST, ANTICIPATION, NEUTRAL, MIXED
- Secondary: EXCITEMENT, CONTENTMENT, PRIDE, LOVE, JEALOUSY, ENVY, SHAME, GUILT, ANXIETY, HOPE, DISAPPOINTMENT, CONFUSION, CURIOSITY, EMPATHY, GRATITUDE, RELIEF, EMBARRASSMENT

### EmotionIntensity

Intensity levels for emotions: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH

### CulturalContext

Cultural contexts for adapting emoji emotional interpretations:
WESTERN, EASTERN_ASIAN, SOUTH_ASIAN, MIDDLE_EASTERN, LATIN_AMERICAN, AFRICAN, OCEANIAN, GLOBAL

### ResponseTone

Possible emotional tones for responses:
MATCHING, SUPPORTIVE, NEUTRAL, REDIRECTING, VALIDATING, CALMING, ENCOURAGING, EMPATHETIC, PROFESSIONAL, CURIOUS, CELEBRATORY

## Usage Examples

### Basic Emotion Detection

```python
analyzer = EmojiEmotionalAnalyzer()
emotion = analyzer.detect_emotion("😊👍")
print(f"Detected emotion: {emotion.primary_emotion.name}")
print(f"Probability: {emotion.primary_probability:.2f}")
print(f"Intensity: {emotion.intensity.name}")
```

### Tracking Conversations and Detecting Shifts

```python
analyzer = EmojiEmotionalAnalyzer()
# Track a conversation
analyzer.track_conversation("😊")  # User is happy
analyzer.track_conversation("👍", is_user_message=False)  # Agent response
analyzer.track_conversation("😔")  # User is sad - emotional shift

# Check for emotional shifts
shift = analyzer.detect_emotional_shift()
if shift:
    print(f"Detected shift from {shift.from_state.primary_emotion.name} to {shift.to_state.primary_emotion.name}")
    print(f"Magnitude: {shift.magnitude:.2f}")
```

### Generating Appropriate Responses

```python
analyzer = EmojiEmotionalAnalyzer()
# Generate response to sadness
response = analyzer.generate_response("😭")
print(f"Response to sadness: {response.emoji_sequence}")

# Generate response with specific tone
response = analyzer.generate_response("😔", desired_tone=ResponseTone.ENCOURAGING)
print(f"Encouraging response: {response.emoji_sequence}")
```

### Cultural Adaptation

```python
analyzer = EmojiEmotionalAnalyzer()
# Get default response
default_response = analyzer.generate_response("😊")
print(f"Default response: {default_response.emoji_sequence}")

# Adapt to Eastern Asian context
analyzer.adapt_to_cultural_context(CulturalContext.EASTERN_ASIAN)
asian_response = analyzer.generate_response("😊")
print(f"Eastern Asian response: {asian_response.emoji_sequence}")
```

## Integration Guide

1. **Import the necessary components:**
   ```python
   from emoji_emotional_analyzer import (
       EmojiEmotionalAnalyzer,
       EmotionCategory,
       EmotionIntensity,
       CulturalContext,
       ResponseTone
   )
   ```

2. **Create an analyzer instance:**
   ```python
   analyzer = EmojiEmotionalAnalyzer()
   ```

3. **For emotion detection in messages:**
   ```python
   def process_message(emoji_message):
       emotion = analyzer.detect_emotion(emoji_message)
       # Use the detected emotion
   ```

4. **For conversation tracking:**
   ```python
   def on_user_message(emoji_message):
       analyzer.track_conversation(emoji_message, is_user_message=True)
       shift = analyzer.detect_emotional_shift()
       if shift:
           handle_emotional_shift(shift)
   ```

5. **For response generation:**
   ```python
   def generate_agent_response(user_emoji):
       response = analyzer.generate_response(user_emoji)
       analyzer.track_conversation(response.emoji_sequence, is_user_message=False)
       return response.emoji_sequence
   ```

6. **For cultural adaptation:**
   ```python
   def adapt_to_user_culture(culture_code):
       cultural_context = map_culture_code_to_context(culture_code)
       analyzer.adapt_to_cultural_context(cultural_context)
   ```

## Best Practices

1. **Emotion Detection:**
   - Use the complete emoji sequence for best results
   - Consider intensity when interpreting emotions
   - Pay attention to confidence scores

2. **Conversation Tracking:**
   - Track both user and agent messages
   - Check for emotional shifts after each user message
   - Consider the magnitude of shifts when responding

3. **Response Generation:**
   - Match response tone to the emotional context
   - For high-intensity negative emotions, use calming tones
   - For low-intensity negative emotions, use supportive tones

4. **Cultural Adaptation:**
   - Adapt to the user's cultural context when known
   - Default to GLOBAL context when cultural background is uncertain
   - Consider regional or demographic variations in emoji use

## Performance Considerations

- Emotion detection is a relatively lightweight operation
- Conversation tracking requires keeping history in memory
- Response generation includes randomization to avoid repetitive responses

## Future Extensions

1. **Custom Emotion Mappings:**
   - Add support for custom emoji-to-emotion mappings
   - Allow for personal emotional profiles

2. **Temporal Analysis:**
   - Enhance shift detection with more sophisticated temporal analysis
   - Identify emotional patterns over longer conversations

3. **Integration with Text Analysis:**
   - Combine emoji and text analysis for richer emotional understanding
   - Support mixed text-emoji messages
