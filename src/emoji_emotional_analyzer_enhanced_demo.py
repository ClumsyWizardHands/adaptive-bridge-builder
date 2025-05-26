import emoji
"""
Enhanced Demonstration of the EmojiEmotionalAnalyzer functionality.

This script provides comprehensive examples of the EmojiEmotionalAnalyzer's advanced capabilities,
including integration with other system components, performance testing, error handling,
and sophisticated usage patterns.
"""

import time
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from emoji_emotional_analyzer import (
    EmojiEmotionalAnalyzer,
    EmotionCategory,
    EmotionIntensity,
    CulturalContext,
    ResponseTone,
    EmotionalState,
    EmotionalShift,
    EmojiEmotionalResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmojiEmotionalAnalyzerDemo")

# Mock classes for integration demonstrations
class MockPrincipleEngine:
    """Mock implementation of the PrincipleEngine for demonstration purposes."""
    
    def __init__(self) -> None:
        self.principles = {
            "empathy": {"weight": 0.8, "threshold": 0.7},
            "respect": {"weight": 0.9, "threshold": 0.8},
            "adaptability": {"weight": 0.7, "threshold": 0.6},
        }
    
    def check_alignment(self, action: str, context: Dict) -> Tuple[float, Dict]:
        """Check if an action aligns with established principles."""
        # For demo purposes, calculate alignment based on emoji content
        scores = {}
        
        # Empathy score - higher for supportive responses to negative emotions
        if context.get("detected_emotion") in [EmotionCategory.SADNESS, EmotionCategory.FEAR]:
            empathy_score = 0.9 if "❤️" in action or "🫂" in action else 0.5
        else:
            empathy_score = 0.8  # Default for other emotions
        scores["empathy"] = empathy_score
        
        # Respect score - lower for intense responses to anger
        if context.get("detected_emotion") == EmotionCategory.ANGER:
            respect_score = 0.4 if "😡" in action else 0.9
        else:
            respect_score = 0.9  # Default for other emotions
        scores["respect"] = respect_score
        
        # Adaptability score - higher for culturally adapted responses
        adaptability_score = 0.9 if context.get("cultural_adaptation") else 0.7
        scores["adaptability"] = adaptability_score
        
        # Calculate weighted average
        total_weight = sum(p["weight"] for p in self.principles.values())
        weighted_score = sum(scores[p] * self.principles[p]["weight"] for p in scores) / total_weight
        
        return weighted_score, scores


class MockOrchestrationAnalytics:
    """Mock implementation of OrchestrationAnalytics for demonstration purposes."""
    
    def __init__(self) -> None:
        self.emotion_stats = {emotion: 0 for emotion in EmotionCategory}
        self.shift_stats = []
        self.response_stats = {tone: 0 for tone in ResponseTone}
        self.cultural_stats = {context: 0 for context in CulturalContext}
        self.performance_metrics = {
            "detection_times": [],
            "response_times": [],
            "shift_detection_times": []
        }
    
    def record_emotion_detection(self, emotion: EmotionalState, detection_time: float) -> None:
        """Record an emotion detection event."""
        self.emotion_stats[emotion.primary_emotion] += 1
        self.performance_metrics["detection_times"].append(detection_time)
        logger.info(f"Analytics: Recorded emotion detection - {emotion.primary_emotion.name} in {detection_time:.2f}ms")
    
    def record_emotional_shift(self, shift: EmotionalShift, detection_time: float) -> None:
        """Record an emotional shift event."""
        self.shift_stats = [*self.shift_stats, (shift.from_state.primary_emotion, shift.to_state.primary_emotion, shift.magnitude)
        self.performance_metrics["shift_detection_times"].append(detection_time)
        logger.info(f"Analytics: Recorded emotional shift from {shift.from_state.primary_emotion.name} to {shift.to_state.primary_emotion.name}")
    
    def record_response_generation(self, response: EmojiEmotionalResponse, generation_time: float) -> None:
        """Record a response generation event."""
        # Extract the tone from the emotional intent
        for tone in ResponseTone:
            if tone.name in response.emotional_intent:
                self.response_stats[tone] += 1
                break
        
        self.performance_metrics["response_times"].append(generation_time)
        logger.info(f"Analytics: Recorded response generation with intent '{response.emotional_intent}' in {generation_time:.2f}ms")
    
    def record_cultural_adaptation(self, context: CulturalContext) -> None:
        """Record a cultural adaptation event."""
        self.cultural_stats[context] += 1
        logger.info(f"Analytics: Recorded cultural adaptation to {context.name}")
    
    def generate_analytics_report(self) -> Dict:
        """Generate a comprehensive analytics report."""
        # Calculate metrics
        avg_detection_time = sum(self.performance_metrics["detection_times"]) / len(self.performance_metrics["detection_times"]) if self.performance_metrics["detection_times"] else 0
        avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
        
        # Most detected emotions
        top_emotions = sorted(self.emotion_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        top_emotions = [(e.name, count) for e, count in top_emotions if count > 0]
        
        # Most used response tones
        top_tones = sorted(self.response_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        top_tones = [(t.name, count) for t, count in top_tones if count > 0]
        
        return {
            "total_emotions_detected": sum(self.emotion_stats.values()),
            "total_shifts_detected": len(self.shift_stats),
            "total_responses_generated": sum(self.response_stats.values()),
            "top_emotions": top_emotions,
            "top_response_tones": top_tones,
            "performance": {
                "avg_detection_time_ms": avg_detection_time,
                "avg_response_time_ms": avg_response_time,
                "max_detection_time_ms": max(self.performance_metrics["detection_times"], default=0),
                "max_response_time_ms": max(self.performance_metrics["response_times"], default=0)
            }
        }


class MockGrowthJournal:
    """Mock implementation of the GrowthJournal for demonstration purposes."""
    
    def __init__(self) -> None:
        self.entries = []
    
    def record_learning(self, category: str, description: str, confidence: float) -> None:
        """Record a learning experience in the growth journal."""
        self.entries.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "category": category,
            "description": description,
            "confidence": confidence
        })
        logger.info(f"GrowthJournal: Recorded {category} learning - {description}")
    
    def generate_growth_report(self) -> Dict:
        """Generate a report of growth and learning."""
        categories = {}
        for entry in self.entries:
            cat = entry["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)
        
        return {
            "total_entries": len(self.entries),
            "categories": {cat: len(entries) for cat, entries in categories.items()},
            "recent_learnings": self.entries[-3:] if len(self.entries) >= 3 else self.entries
        }


@dataclass
class DemoScenario:
    """Represents a demonstration scenario with test data and expected results."""
    name: str
    description: str
    emoji_sequences: List[str]
    expected_emotions: List[EmotionCategory]
    cultural_context: Optional[CulturalContext] = None
    response_tone: Optional[ResponseTone] = None
    error_condition: Optional[str] = None


def create_demo_scenarios() -> List[DemoScenario]:
    """Create a set of demonstration scenarios."""
    return [
        DemoScenario(
            name="Basic Emotions",
            description="Detection of primary emotions from clear emoji signals",
            emoji_sequences=["😊", "😭", "😡", "😨", "😲"],
            expected_emotions=[
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.FEAR,
                EmotionCategory.SURPRISE
            ]
        ),
        DemoScenario(
            name="Complex Patterns",
            description="Detection of emotions from combined emoji patterns",
            emoji_sequences=["😊👍", "😭💔", "😡💢", "🤔❓"],
            expected_emotions=[
                EmotionCategory.CONTENTMENT,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.CURIOSITY
            ]
        ),
        DemoScenario(
            name="Intensity Variations",
            description="Detection of emotion intensity variations",
            emoji_sequences=["😀", "😀😀", "😀😀😀", "😀❗", "😀‼️"],
            expected_emotions=[
                EmotionCategory.JOY,
                EmotionCategory.JOY,
                EmotionCategory.JOY,
                EmotionCategory.JOY,
                EmotionCategory.JOY
            ]
        ),
        DemoScenario(
            name="Mixed Emotions",
            description="Detection of mixed emotional signals",
            emoji_sequences=["😀😔", "😊😡", "😨😄", "😭😡"],
            expected_emotions=[
                EmotionCategory.JOY,  # Primary, but with sadness as secondary
                EmotionCategory.JOY,  # Primary, but with anger as secondary
                EmotionCategory.FEAR,  # Primary, but with joy as secondary
                EmotionCategory.SADNESS  # Primary, but with anger as secondary
            ]
        ),
        DemoScenario(
            name="Cultural Adaptations",
            description="Response generation with cultural adaptations",
            emoji_sequences=["😊", "😭", "😡"],
            expected_emotions=[
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER
            ],
            cultural_context=CulturalContext.EASTERN_ASIAN
        ),
        DemoScenario(
            name="Specific Response Tones",
            description="Response generation with specific tones",
            emoji_sequences=["😔", "😡", "😨"],
            expected_emotions=[
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.FEAR
            ],
            response_tone=ResponseTone.ENCOURAGING
        ),
        DemoScenario(
            name="Error Handling - Empty",
            description="Handling of empty input",
            emoji_sequences=["", "   ", "\n"],
            expected_emotions=[
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL
            ],
            error_condition="empty_input"
        ),
        DemoScenario(
            name="Error Handling - Invalid",
            description="Handling of inputs with no valid emojis",
            emoji_sequences=["Hello world", "123", "!@#$"],
            expected_emotions=[
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL
            ],
            error_condition="no_emojis"
        ),
        DemoScenario(
            name="Error Handling - Rare Emojis",
            description="Handling of rare or unsupported emojis",
            emoji_sequences=["🧿", "🫠", "🦤"],
            expected_emotions=[
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL,
                EmotionCategory.NEUTRAL
            ],
            error_condition="rare_emojis"
        )
    ]


def run_demo_scenario(
    analyzer: EmojiEmotionalAnalyzer,
    scenario: DemoScenario,
    analytics: MockOrchestrationAnalytics,
    growth_journal: MockGrowthJournal
) -> Dict:
    """Run a single demonstration scenario and collect results."""
    logger.info(f"Running scenario: {scenario.name} - {scenario.description}")
    
    results = []
    
    # Set cultural context if specified
    if scenario.cultural_context:
        analyzer.adapt_to_cultural_context(scenario.cultural_context)
        analytics.record_cultural_adaptation(scenario.cultural_context)
        logger.info(f"Adapted to cultural context: {scenario.cultural_context.name}")
    
    # Process each emoji sequence
    for i, emoji_seq in enumerate(scenario.emoji_sequences):
        result = {"emoji": emoji_seq}
        
        try:
            # Time emotion detection
            start_time = time.time()
            emotion = analyzer.detect_emotion(emoji_seq)
            detection_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result["emotion"] = emotion
            result["detection_time"] = detection_time
            
            # Record analytics
            analytics.record_emotion_detection(emotion, detection_time)
            
            # Verify expected emotion
            expected = scenario.expected_emotions[i]
            result["expected"] = expected
            result["matched"] = emotion.primary_emotion == expected
            
            # Generate response if no error condition
            if not scenario.error_condition:
                start_time = time.time()
                response = analyzer.generate_response(
                    emoji_seq,
                    desired_tone=scenario.response_tone
                )
                generation_time = (time.time() - start_time) * 1000  # Convert to ms
                
                result["response"] = response
                result["generation_time"] = generation_time
                
                # Record analytics
                analytics.record_response_generation(response, generation_time)
            
        except Exception as e:
            # Record error
            result["error"] = str(e)
            logger.warning(f"Error processing emoji sequence '{emoji_seq}': {e}")
            
            # Record learning in growth journal
            growth_journal.record_learning(
                "error_handling",
                f"Encountered {type(e).__name__} when processing '{emoji_seq}': {str(e)}",
                0.8
            )
        
        results.append(result)
        
    return {
        "scenario": scenario,
        "results": results
    }


def run_conversation_demo(
    analyzer: EmojiEmotionalAnalyzer,
    analytics: MockOrchestrationAnalytics,
    growth_journal: MockGrowthJournal
) -> Dict:
    """Run a demo of conversation tracking and emotional shift detection."""
    logger.info("Running conversation tracking and emotional shift demo")
    
    # Reset conversation history
    analyzer.shift_tracker.conversation_history = []
    analyzer.shift_tracker.emotion_history = []
    
    # Define a multi-turn conversation with emotional shifts
    conversation = [
        # Start with happy user
        ("😊👍", True, "User starts happy"),
        ("🎉😄", False, "Agent responds with matching joy"),
        # Continue happy
        ("😄🙌", True, "User continues to be happy"),
        ("✨👍", False, "Agent responds positively"),
        # First shift to curiosity
        ("🤔❓", True, "User becomes curious - first emotional shift"),
        ("👀✨", False, "Agent responds with curiosity"),
        # Shift to concern/fear
        ("😨", True, "User becomes worried - second emotional shift"),
        ("🫂💙", False, "Agent responds with supportive tone"),
        # Escalation of fear
        ("😱❗", True, "User's fear intensifies"),
        ("🧘‍♀️✨", False, "Agent responds with calming tone"),
        # Shift to relief
        ("😌", True, "User feels relief - third emotional shift"),
        ("👍💙", False, "Agent responds positively"),
        # Back to happiness
        ("😊🙏", True, "User returns to happiness - fourth emotional shift"),
        ("🎉✨", False, "Agent responds with celebration")
    ]
    
    shifts_detected = []
    
    # Process conversation
    for i, (emoji_seq, is_user, description) in enumerate(conversation):
        # Track the message
        analyzer.track_conversation(emoji_seq, is_user)
        
        # Detect shifts after user messages (except the first)
        if is_user and i > 0:
            start_time = time.time()
            shift = analyzer.detect_emotional_shift()
            detection_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if shift:
                # Record the shift
                analytics.record_emotional_shift(shift, detection_time)
                
                shifts_detected.append({
                    "message_index": i,
                    "emoji": emoji_seq,
                    "from_emotion": shift.from_state.primary_emotion,
                    "to_emotion": shift.to_state.primary_emotion,
                    "magnitude": shift.magnitude,
                    "trigger": shift.detected_trigger,
                    "pattern": shift.temporal_pattern,
                    "description": description
                })
                
                # Record learning in growth journal
                if shift.magnitude > 0.7:
                    growth_journal.record_learning(
                        "emotional_patterns",
                        f"Detected significant emotional shift from {shift.from_state.primary_emotion.name} to {shift.to_state.primary_emotion.name} with magnitude {shift.magnitude:.2f}",
                        0.9
                    )
    
    return {
        "conversation_length": len(conversation),
        "shifts_detected": shifts_detected,
        "num_shifts": len(shifts_detected)
    }


def run_performance_demo(
    analyzer: EmojiEmotionalAnalyzer,
    analytics: MockOrchestrationAnalytics
) -> Dict:
    """Run a demo focused on performance metrics."""
    logger.info("Running performance testing demo")
    
    # Generate test data
    test_emojis = ["😊", "😭", "😡", "😨", "🤔", "😄", "😔", "😱", "😲", "😐"]
    test_sequences = []
    
    # Single emojis
    test_sequences.extend(test_emojis)
    
    # Pairs
    for i in range(5):
        sequence = random.sample(test_emojis, 2)
        test_sequences.append(''.join(sequence))
    
    # Triplets
    for i in range(5):
        sequence = random.sample(test_emojis, 3)
        test_sequences.append(''.join(sequence))
    
    # With modifiers
    modifiers = ["❗", "‼️", "❓"]
    for i in range(5):
        emoji = random.choice(test_emojis)
        modifier = random.choice(modifiers)
        test_sequences.append(f"{emoji}{modifier}")
    
    # Performance results
    detection_times = []
    response_times = []
    
    # Run performance test
    for emoji_seq in test_sequences:
        # Test emotion detection
        start_time = time.time()
        emotion = analyzer.detect_emotion(emoji_seq)
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        detection_times.append(detection_time)
        
        # Record analytics
        analytics.record_emotion_detection(emotion, detection_time)
        
        # Test response generation
        start_time = time.time()
        response = analyzer.generate_response(emoji_seq)
        generation_time = (time.time() - start_time) * 1000  # Convert to ms
        response_times.append(generation_time)
        
        # Record analytics
        analytics.record_response_generation(response, generation_time)
    
    # Calculate performance metrics
    avg_detection_time = sum(detection_times) / len(detection_times)
    avg_response_time = sum(response_times) / len(response_times)
    max_detection_time = max(detection_times)
    max_response_time = max(response_times)
    
    logger.info(f"Performance metrics - Avg detection: {avg_detection_time:.2f}ms, Avg response: {avg_response_time:.2f}ms")
    
    return {
        "num_sequences": len(test_sequences),
        "detection_performance": {
            "avg_time_ms": avg_detection_time,
            "max_time_ms": max_detection_time,
            "min_time_ms": min(detection_times),
        },
        "response_performance": {
            "avg_time_ms": avg_response_time,
            "max_time_ms": max_response_time,
            "min_time_ms": min(response_times),
        }
    }


def run_principle_alignment_demo(
    analyzer: EmojiEmotionalAnalyzer,
    principles: MockPrincipleEngine,
    growth_journal: MockGrowthJournal
) -> Dict:
    """Run a demo of principle alignment checking for emoji responses."""
    logger.info("Running principle alignment demo")
    
    test_cases = [
        # Emotion, Response Tone, Context Description
        (EmotionCategory.SADNESS, ResponseTone.SUPPORTIVE, "Empathetic response to sadness"),
        (EmotionCategory.SADNESS, ResponseTone.NEUTRAL, "Neutral response to sadness"),
        (EmotionCategory.ANGER, ResponseTone.CALMING, "Calming response to anger"),
        (EmotionCategory.ANGER, ResponseTone.MATCHING, "Matching response to anger"),
        (EmotionCategory.JOY, ResponseTone.MATCHING, "Matching response to joy"),
        (EmotionCategory.FEAR, ResponseTone.SUPPORTIVE, "Supportive response to fear"),
    ]
    
    results = []
    
    for emotion_cat, tone, description in test_cases:
        # Find an emoji that matches the target emotion
        test_emoji = ""
        for emoji, (mapped_emotion, _) in analyzer.detection_engine.primary_emotion_mappings.items():
            if mapped_emotion == emotion_cat:
                test_emoji = emoji
                break
        
        if not test_emoji:
            test_emoji = "😐"  # Fallback if no matching emoji found
        
        # Generate response
        response = analyzer.generate_response(test_emoji, desired_tone=tone)
        
        # Check principle alignment
        context = {
            "detected_emotion": emotion_cat,
            "response_tone": tone,
            "cultural_adaptation": bool(response.cultural_adaptations)
        }
        
        alignment_score, principle_scores = principles.check_alignment(
            response.emoji_sequence, context
        )
        
        # Record result
        result = {
            "test_emoji": test_emoji,
            "emotion": emotion_cat,
            "tone": tone,
            "response": response.emoji_sequence,
            "description": description,
            "alignment_score": alignment_score,
            "principle_scores": principle_scores
        }
        
        results.append(result)
        
        # Record learning in growth journal for low alignment scores
        if alignment_score < 0.7:
            growth_journal.record_learning(
                "principle_alignment",
                f"Low alignment score ({alignment_score:.2f}) for {description}. Need to improve response patterns for {tone.name} responses to {emotion_cat.name}.",
                0.85
            )
        
        logger.info(
            f"Alignment for {description}: {alignment_score:.2f} - " +
            ", ".join(f"{p}: {s:.2f}" for p, s in principle_scores.items())
        )
    
    return {
        "num_tests": len(results),
        "results": results,
        "avg_alignment": sum(r["alignment_score"] for r in results) / len(results)
    }


def run_enhanced_demo() -> None:
    """Run the comprehensive enhanced demonstration of the EmojiEmotionalAnalyzer."""
    print("=" * 80)
    print("EmojiEmotionalAnalyzer Enhanced Demonstration")
    print("=" * 80)
    
    # Initialize components
    analyzer = EmojiEmotionalAnalyzer()
    principles = MockPrincipleEngine()
    analytics = MockOrchestrationAnalytics()
    growth_journal = MockGrowthJournal()
    
    # Create the scenario test cases
    scenarios = create_demo_scenarios()
    
    # 1. Run all demonstration scenarios
    print("\n1. RUNNING DEMONSTRATION SCENARIOS")
    print("-" * 60)
    
    scenario_results = []
    for scenario in scenarios:
        result = run_demo_scenario(analyzer, scenario, analytics, growth_journal)
        scenario_results.append(result)
        
        print(f"\nCompleted scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  Sequences tested: {len(scenario.emoji_sequences)}")
        
        if scenario.error_condition:
            print(f"  Error condition: {scenario.error_condition}")
        
        # Print a few example results
        for i, res in enumerate(result["results"][:2]):  # Just show first 2 for brevity
            print(f"  Example {i+1}: {res['emoji']}")
            if "error" in res:
                print(f"    Error: {res['error']}")
            else:
                print(f"    Detected: {res['emotion'].primary_emotion.name}")
                print(f"    Expected: {res['expected'].name}")
                print(f"    Matched: {res['matched']}")
                
                if "response" in res:
                    print(f"    Response: {res['response'].emoji_sequence}")
                    print(f"    Intent: {res['response'].emotional_intent}")
    
    # 2. Conversation tracking and emotional shift detection
    print("\n\n2. CONVERSATION TRACKING & EMOTIONAL SHIFTS")
    print("-" * 60)
    
    conversation_result = run_conversation_demo(analyzer, analytics, growth_journal)
    
    print(f"\nCompleted conversation demo:")
    print(f"  Conversation length: {conversation_result['conversation_length']} messages")
    print(f"  Emotional shifts detected: {conversation_result['num_shifts']}")
    
    # Print details of detected shifts
    for i, shift in enumerate(conversation_result["shifts_detected"]):
        print(f"\n  Shift {i+1}: {shift['description']}")
        print(f"    From: {shift['from_emotion'].name} → To: {shift['to_emotion'].name}")
        print(f"    Magnitude: {shift['magnitude']:.2f}")
        print(f"    Trigger: {shift['trigger']}")
        print(f"    Pattern: {shift['pattern']}")
    
    # 3. Performance testing
    print("\n\n3. PERFORMANCE TESTING")
    print("-" * 60)
    
    performance_result = run_performance_demo(analyzer, analytics)
    
    print(f"\nCompleted performance testing:")
    print(f"  Sequences tested: {performance_result['num_sequences']}")
    print("\n  Emotion detection performance:")
    print(f"    Average time: {performance_result['detection_performance']['avg_time_ms']:.2f} ms")
    print(f"    Maximum time: {performance_result['detection_performance']['max_time_ms']:.2f} ms")
    print(f"    Minimum time: {performance_result['detection_performance']['min_time_ms']:.2f} ms")
    
    print("\n  Response generation performance:")
    print(f"    Average time: {performance_result['response_performance']['avg_time_ms']:.2f} ms")
    print(f"    Maximum time: {performance_result['response_performance']['max_time_ms']:.2f} ms")
    print(f"    Minimum time: {performance_result['response_performance']['min_time_ms']:.2f} ms")
    
    # 4. Principle alignment
    print("\n\n4. PRINCIPLE ALIGNMENT TESTING")
    print("-" * 60)
    
    alignment_result = run_principle_alignment_demo(analyzer, principles, growth_journal)
    
    print(f"\nCompleted principle alignment testing:")
    print(f"  Test cases: {alignment_result['num_tests']}")
    print(f"  Average alignment score: {alignment_result['avg_alignment']:.2f}")
    
    # Print details of alignment results
    for i, result in enumerate(alignment_result["results"]):
        print(f"\n  Test {i+1}: {result['description']}")
        print(f"    Emoji: {result['test_emoji']} ({result['emotion'].name})")
        print(f"    Response: {result['response']} ({result['tone'].name})")
        print(f"    Alignment score: {result['alignment_score']:.2f}")
        print("    Principle scores:")
        for principle, score in result["principle_scores"].items():
            print(f"      - {principle}: {score:.2f}")
    
    # 5. Analytics Report
    print("\n\n5. ANALYTICS REPORT")
    print("-" * 60)
    
    analytics_report = analytics.generate_analytics_report()
    
    print(f"\nOrchestrationAnalytics Report:")
    print(f"  Total emotions detected: {analytics_report['total_emotions_detected']}")
    print(f"  Total emotional shifts detected: {analytics_report['total_shifts_detected']}")
    print(f"  Total responses generated: {analytics_report['total_responses_generated']}")
    
    print("\n  Top detected emotions:")
    for emotion, count in analytics_report['top_emotions']:
        print(f"    - {emotion}: {count}")
    
    print("\n  Top response tones:")
    for tone, count in analytics_report['top_response_tones']:
        print(f"    - {tone}: {count}")
    
    print("\n  Performance metrics:")
    print(f"    Average detection time: {analytics_report['performance']['avg_detection_time_ms']:.2f} ms")
    print(f"    Average response time: {analytics_report['performance']['avg_response_time_ms']:.2f} ms")
    print(f"    Maximum detection time: {analytics_report['performance']['max_detection_time_ms']:.2f} ms")
    print(f"    Maximum response time: {analytics_report['performance']['max_response_time_ms']:.2f} ms")
    
    # 6. Growth Journal Report
    print("\n\n6. GROWTH JOURNAL REPORT")
    print("-" * 60)
    
    growth_report = growth_journal.generate_growth_report()
    
    print(f"\nGrowth Journal Report:")
    print(f"  Total learning entries: {growth_report['total_entries']}")
    
    print("\n  Learning categories:")
    for category, count in growth_report['categories'].items():
        print(f"    - {category}: {count} entries")
    
    print("\n  Recent learnings:")
    for i, entry in enumerate(growth_report['recent_learnings']):
        print(f"    {i+1}. [{entry['category']}] {entry['description']}")
        print(f"       Confidence: {entry['confidence']:.2f}, Recorded: {entry['timestamp']}")


if __name__ == "__main__":
    run_enhanced_demo()