import emoji
"""
Emoji Tutorial System for teaching emoji-based communication.

This system provides a structured learning approach for users to become proficient
in emoji-only communication with intelligent agents.
"""

import random
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
import datetime
from datetime import timezone
import copy

from emoji_emotional_analyzer import (
    EmojiEmotionalAnalyzer,
    EmotionCategory,
    CulturalContext
)

from domain_specific_emoji_sets import (
    DomainSpecificEmojiSet,
    TechnicalSupportEmojiSet,
    ProjectManagementEmojiSet,
    EducationalEmojiSet,
    FinancialEmojiSet
)


class DifficultyLevel(Enum):
    """Difficulty levels for emoji tutorials."""
    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()
    FLUENT = auto()


class TutorialCategory(Enum):
    """Categories of emoji tutorials."""
    BASICS = auto()
    EMOTIONS = auto()
    CONVERSATION = auto()
    DOMAIN_SPECIFIC = auto()
    GRAMMAR_SYNTAX = auto()
    CULTURAL_VARIATIONS = auto()
    ADVANCED_SEQUENCES = auto()


@dataclass
class TutorialLesson:
    """A lesson in the emoji tutorial system."""
    title: str
    description: str
    difficulty: DifficultyLevel
    category: TutorialCategory
    example_sequences: List[Tuple[str, str]] = field(default_factory=list)  # (emoji_sequence, explanation)
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)  # List of lesson titles that should be completed first


@dataclass
class TutorialExercise:
    """An exercise in the emoji tutorial system."""
    exercise_id: str
    prompt: str
    difficulty: DifficultyLevel
    exercise_type: str  # "multiple_choice", "free_response", "matching", "sequence_completion"
    options: List[Any] = field(default_factory=list)  # For multiple choice or matching
    correct_answer: Any = None
    hints: List[str] = field(default_factory=list)
    feedback_templates: Dict[str, str] = field(default_factory=dict)  # Different feedback for different response types


@dataclass
class UserProfile:
    """User profile for the emoji tutorial system."""
    user_id: str
    name: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    completed_lessons: List[str] = field(default_factory=list)
    current_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    personal_emoji_dictionary: Dict[str, str] = field(default_factory=dict)  # emoji -> meaning
    frequently_used_emojis: Dict[str, int] = field(default_factory=dict)  # emoji -> count
    communication_patterns: Dict[str, Any] = field(default_factory=dict)
    preferred_domains: List[str] = field(default_factory=list)
    lesson_scores: Dict[str, float] = field(default_factory=dict)  # lesson_id -> score (0.0 to 1.0)
    session_history: List[Dict[str, Any]] = field(default_factory=list)


class EmojiTutorialSystem:
    """System for teaching users to communicate with emoji."""
    
    def __init__(self) -> None:
        """Initialize the emoji tutorial system."""
        self.lessons: Dict[str, TutorialLesson] = {}
        self.exercises: Dict[str, TutorialExercise] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.emoji_analyzer = EmojiEmotionalAnalyzer()
        self.domain_sets = {
            "technical_support": TechnicalSupportEmojiSet(),
            "project_management": ProjectManagementEmojiSet(),
            "educational": EducationalEmojiSet(),
            "financial": FinancialEmojiSet()
        }
        
        # Initialize lesson catalog
        self._initialize_tutorials()
    
    def _initialize_tutorials(self) -> None:
        """Initialize the tutorial lessons and exercises."""
        self._create_basic_emoji_lessons()
        self._create_emotional_expression_lessons()
        self._create_conversation_structure_lessons()
        self._create_domain_specific_lessons()
        self._create_cultural_adaptation_lessons()
        self._create_advanced_sequence_lessons()
    
    def _create_basic_emoji_lessons(self) -> None:
        """Create lessons for basic emoji usage."""
        # Lesson 1: Introduction to Emoji Communication
        lesson = TutorialLesson(
            title="Introduction to Emoji Communication",
            description="Learn the basics of using emojis to communicate with an agent.",
            difficulty=DifficultyLevel.BEGINNER,
            category=TutorialCategory.BASICS,
            example_sequences=[
                ("👋", "A simple greeting (Hello)"),
                ("👍", "Acknowledgment or agreement (Yes, OK)"),
                ("❓", "A question or confusion"),
                ("👋❓", "Hello, how are you?"),
                ("👍😊", "Yes, I'm happy with that")
            ]
        )
        
        # Create exercises for this lesson
        exercises = [
            TutorialExercise(
                exercise_id="basic_1_1",
                prompt="What emoji would you use to say hello?",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="multiple_choice",
                options=["👋", "👍", "❓", "🔄"],
                correct_answer="👋",
                hints=["This is a common greeting gesture."],
                feedback_templates={
                    "correct": "Great job! 👋 is commonly used as a greeting, similar to saying 'hello' or 'hi'.",
                    "incorrect": "Not quite. 👋 is the emoji that represents a waving hand, commonly used as a greeting."
                }
            ),
            TutorialExercise(
                exercise_id="basic_1_2",
                prompt="Respond with an emoji that means 'yes' or 'I agree'",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="free_response",
                correct_answer="👍",
                hints=["Think about a gesture that shows approval."],
                feedback_templates={
                    "correct": "Perfect! 👍 is widely understood to mean 'yes', 'ok', or 'I agree'.",
                    "similar": "That works too! While 👍 is the most common, your response also conveys agreement.",
                    "incorrect": "Not quite. 👍 (thumbs up) is the most common emoji for showing agreement or saying yes."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        lesson.completion_criteria = {"min_correct_exercises": 2}
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
            
        # Lesson 2: Basic Emoji Combinations
        lesson = TutorialLesson(
            title="Basic Emoji Combinations",
            description="Learn how to combine emojis to create more complex messages.",
            difficulty=DifficultyLevel.BEGINNER,
            category=TutorialCategory.BASICS,
            prerequisites=["Introduction to Emoji Communication"],
            example_sequences=[
                ("👍👀", "I'll look into it"),
                ("❓🕐", "When will this happen?"),
                ("🙏👍", "Please approve this"),
                ("👍🔜", "Yes, coming soon"),
                ("❓💭", "What do you think?")
            ]
        )
        
        # Create exercises for this lesson
        exercises = [
            TutorialExercise(
                exercise_id="basic_2_1",
                prompt="How would you ask 'When is the meeting?'",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="multiple_choice",
                options=["🕐❓", "👍🕐", "🙏🕐", "🔄🕐"],
                correct_answer="🕐❓",
                hints=["You need to combine a time-related emoji with something indicating a question."],
                feedback_templates={
                    "correct": "Excellent! 🕐❓ combines the clock (time) with a question mark to ask when something will happen.",
                    "incorrect": "Not quite. 🕐❓ is the best choice here, combining the clock (time) with a question mark to ask when."
                }
            ),
            TutorialExercise(
                exercise_id="basic_2_2",
                prompt="Create an emoji sequence that means 'I need help'",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="free_response",
                correct_answer=["🆘", "🙏❓", "❓🙏", "🆘🙏"],
                hints=["Think about combining a request with a question, or using a universal symbol for help."],
                feedback_templates={
                    "correct": "Great job! Your emoji sequence clearly communicates the need for help.",
                    "similar": "That works too! Your emoji sequence gets the message across.",
                    "incorrect": "Your sequence is a bit unclear for 'I need help'. Consider using 🆘 or a combination like 🙏❓"
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        lesson.completion_criteria = {"min_correct_exercises": 2}
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def _create_emotional_expression_lessons(self) -> None:
        """Create lessons for expressing emotions with emoji."""
        # Lesson: Basic Emotional Expression
        lesson = TutorialLesson(
            title="Basic Emotional Expression",
            description="Learn to express and recognize basic emotions using emojis.",
            difficulty=DifficultyLevel.BEGINNER,
            category=TutorialCategory.EMOTIONS,
            prerequisites=["Basic Emoji Combinations"],
            example_sequences=[
                ("😊", "Happiness/Joy"),
                ("😢", "Sadness"),
                ("😡", "Anger"),
                ("😮", "Surprise"),
                ("😊👍", "Happy and approving"),
                ("😢❓", "Why am I sad?/Are you sad?")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="emotion_1_1",
                prompt="Match each emoji with the emotion it primarily expresses:",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="matching",
                options=[
                    {"emoji": "😊", "emotion": "Happiness"},
                    {"emoji": "😢", "emotion": "Sadness"},
                    {"emoji": "😡", "emotion": "Anger"},
                    {"emoji": "😮", "emotion": "Surprise"}
                ],
                correct_answer=[
                    {"emoji": "😊", "emotion": "Happiness"},
                    {"emoji": "😢", "emotion": "Sadness"},
                    {"emoji": "😡", "emotion": "Anger"},
                    {"emoji": "😮", "emotion": "Surprise"}
                ],
                feedback_templates={
                    "correct": "Perfect! You've correctly matched the emojis to their primary emotions.",
                    "partial": "Some matches are correct, but not all. Review the emotional expressions and try again.",
                    "incorrect": "None of the matches are correct. Study the example emotional expressions and try again."
                }
            ),
            TutorialExercise(
                exercise_id="emotion_1_2",
                prompt="How would you express 'I'm excited about this new project'?",
                difficulty=DifficultyLevel.BEGINNER,
                exercise_type="free_response",
                correct_answer=["😀🎉", "🎉😀", "😊🎉", "🎉😊", "😄🎉", "🎉😄"],
                hints=["Combine an emoji for happiness with one that represents celebration or excitement."],
                feedback_templates={
                    "correct": "Excellent! Your sequence clearly expresses excitement about the project.",
                    "similar": "Good attempt! Your sequence conveys the excitement, though there might be even clearer ways to express it.",
                    "incorrect": "Your response doesn't clearly communicate excitement. Try combining a happy face with celebration emojis like 🎉"
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
            
        # Lesson: Emotional Intensity
        lesson = TutorialLesson(
            title="Emotional Intensity",
            description="Learn to express different levels of emotional intensity using emojis.",
            difficulty=DifficultyLevel.INTERMEDIATE,
            category=TutorialCategory.EMOTIONS,
            prerequisites=["Basic Emotional Expression"],
            example_sequences=[
                ("🙂", "Mild happiness/contentment"),
                ("😊", "Moderate happiness/joy"),
                ("😄", "Strong happiness/delight"),
                ("😁", "Very strong happiness/elation"),
                ("😆😂🤣", "Extreme happiness/hilarity"),
                ("😐", "Neutral/mild disappointment"),
                ("😔", "Moderate sadness"),
                ("😢", "Strong sadness"),
                ("😭", "Very strong sadness/grief")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="emotion_2_1",
                prompt="Arrange these happiness emojis from lowest to highest intensity:",
                difficulty=DifficultyLevel.INTERMEDIATE,
                exercise_type="sequence_completion",
                options=["😁", "🙂", "😄", "😊"],
                correct_answer=["🙂", "😊", "😄", "😁"],
                hints=["Consider the facial expressions and how pronounced they are."],
                feedback_templates={
                    "correct": "Perfect! You've correctly arranged the happiness emojis from lowest to highest intensity.",
                    "incorrect": "Not quite. The correct order from lowest to highest intensity is: 🙂 (slight smile), 😊 (smiling), 😄 (grinning), 😁 (beaming)."
                }
            ),
            TutorialExercise(
                exercise_id="emotion_2_2",
                prompt="Express 'I'm extremely upset about this situation'",
                difficulty=DifficultyLevel.INTERMEDIATE,
                exercise_type="free_response",
                correct_answer=["😡😡😡", "😡🤬", "🤬😡", "😡‼️", "😤😡", "😡😤"],
                hints=["Use multiple instances of the same emoji to increase intensity, or combine with emphasis symbols."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates extreme upset.",
                    "similar": "Good attempt! Your sequence conveys being upset, though the intensity could be clearer.",
                    "incorrect": "Your response doesn't clearly communicate being extremely upset. Try using anger emojis with intensity modifiers."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def _create_conversation_structure_lessons(self) -> None:
        """Create lessons for conversation structure using emojis."""
        # Lesson: Emoji Conversation Flow
        lesson = TutorialLesson(
            title="Emoji Conversation Flow",
            description="Learn to structure conversation flows using emojis.",
            difficulty=DifficultyLevel.INTERMEDIATE,
            category=TutorialCategory.CONVERSATION,
            prerequisites=["Emotional Intensity"],
            example_sequences=[
                ("👋 → 👋", "Greeting exchange"),
                ("❓ → 💭 → 👍", "Question → Thinking → Answer"),
                ("🙏 → 👍/👎", "Request → Approval/Rejection"),
                ("😢 → 🫂 → 😊", "Sadness → Comfort → Happiness"),
                ("🤔 → 💡 → 🎉", "Confusion → Realization → Celebration")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="convo_1_1",
                prompt="Complete this conversation flow: I suggest an idea (💡) → You express interest (?) → I provide details (?) → You approve (?)",
                difficulty=DifficultyLevel.INTERMEDIATE,
                exercise_type="sequence_completion",
                options=["👍", "👀", "📝", "🤔"],
                correct_answer=["👀", "📝", "👍"],
                hints=["Think about how you would express interest, then how details would be provided, then approval."],
                feedback_templates={
                    "correct": "Perfect! 💡 → 👀 → 📝 → 👍 creates a clear conversation flow from idea to interest to details to approval.",
                    "incorrect": "Not quite. A good flow would be: 💡 (idea) → 👀 (interest) → 📝 (details) → 👍 (approval)."
                }
            ),
            TutorialExercise(
                exercise_id="convo_1_2",
                prompt="Create an emoji conversation where you ask for help, receive acknowledgment, then express gratitude.",
                difficulty=DifficultyLevel.INTERMEDIATE,
                exercise_type="free_response",
                correct_answer=["🆘❓→👍→🙏", "❓🙏→👍→😊🙏", "🙏❓→👀👍→🙏😊"],
                hints=["Start with a request for help, followed by a response, and end with thanks."],
                feedback_templates={
                    "correct": "Excellent! Your conversation flow clearly shows asking for help, getting acknowledgment, and expressing gratitude.",
                    "similar": "Good attempt! Your sequence shows the general idea, though it could be more precise.",
                    "incorrect": "Your response doesn't create a clear help → acknowledgment → gratitude flow. Try something like 🆘❓→👍→🙏"
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
            
        # Lesson: Complex Discussions
        lesson = TutorialLesson(
            title="Complex Discussions with Emojis",
            description="Learn to have detailed, nuanced discussions using emoji sequences.",
            difficulty=DifficultyLevel.ADVANCED,
            category=TutorialCategory.CONVERSATION,
            prerequisites=["Emoji Conversation Flow"],
            example_sequences=[
                ("🤔💡❓", "I have an idea, what do you think?"),
                ("👍🔄⏰", "Yes, but we need to do it again later."),
                ("📊📉❓→🔍→📈🎯", "Why are metrics down? → Let's investigate → Found solution, hitting targets now"),
                ("🙅‍♂️👎+👍👀🔄", "I disagree with option A + let's look at alternative options"),
                ("❓💼🔄📅→👥📝→👍🚀", "When to reschedule meeting? → Team decides → Approved, launching")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="convo_2_1",
                prompt="Translate this conversation to emoji sequences: 'I notice a problem with the project. Can you investigate and suggest solutions?'",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=["👀⚠️💼❓→🔍→💡❓", "⚠️💼→🙏🔍→💡❓", "👀⚠️💼→🙏🔍💡"],
                hints=["Break it down into 'noticing a problem', 'investigation request', and 'solutions request'."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates the problem observation, investigation request, and solution inquiry.",
                    "similar": "Good effort! Your sequence covers the main points, though it could be more specific in certain areas.",
                    "incorrect": "Your sequence doesn't clearly communicate all three elements: problem identification, investigation request, and solution inquiry."
                }
            ),
            TutorialExercise(
                exercise_id="convo_2_2",
                prompt="Create an emoji conversation discussing project progress, identifying a delay, and proposing a new timeline.",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=[
                    "📊💼❓→⏰⚠️→📅🔄❓→👍",
                    "💼📊→⚠️⏰📉→📅🆕❓→👍",
                    "📊💼→⏰⚠️→🔄📅→👍"
                ],
                hints=["Include elements for 'project', 'progress/status', 'delay/problem', 'timeline/schedule', and 'approval'."],
                feedback_templates={
                    "correct": "Excellent! Your emoji conversation clearly discusses project status, identifies a delay, proposes a new timeline, and reaches agreement.",
                    "similar": "Good attempt! Your sequence covers the main points of the discussion, though some elements could be clearer.",
                    "incorrect": "Your conversation doesn't clearly address all required elements: project progress check, delay identification, new timeline proposal, and resolution."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def _create_domain_specific_lessons(self) -> None:
        """Create lessons for domain-specific emoji communication."""
        # Technical Support Domain Lesson
        lesson = TutorialLesson(
            title="Technical Support Emoji Communication",
            description="Learn specialized emoji patterns for technical support contexts.",
            difficulty=DifficultyLevel.ADVANCED,
            category=TutorialCategory.DOMAIN_SPECIFIC,
            prerequisites=["Complex Discussions with Emojis"],
            example_sequences=[
                ("🛑🖥️", "Critical server error"),
                ("⚠️💾", "Storage warning"),
                ("❓🔌", "Network connectivity issue"),
                ("🔄🖥️✅", "Server restart successful"),
                ("💾⏱️⚠️", "Database performance warning")
            ]
        )
        
        # Create exercises
        tech_support_set = self.domain_sets["technical_support"]
        exercises = [
            TutorialExercise(
                exercise_id="domain_tech_1",
                prompt="Create an emoji sequence to report: 'The database is down and needs immediate attention'",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=["🛑💾", "💾🛑", "💾⚠️🛑", "🛑💾❗"],
                hints=["Combine critical error indicator with database representation."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates a critical database issue.",
                    "similar": "Good attempt! Your sequence indicates a database problem, but could be more specific about the severity.",
                    "incorrect": "Your response doesn't clearly communicate a critical database issue. Try using 🛑 (critical) with 💾 (database)."
                }
            ),
            TutorialExercise(
                exercise_id="domain_tech_2",
                prompt="Translate this technical support conversation: 'Is the network down? → Yes, investigating now → Found the issue, fixing → Network restored'",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=[
                    "❓🔌→👍🔍→💡🔧→🔌✅",
                    "🔌❓→🔍👀→🔧🔌→✅",
                    "🔌⚠️❓→👍🔍→🔧→🔌✅"
                ],
                hints=["Use emojis for 'network', 'question', 'investigation', 'fix/repair', and 'success/completion'."],
                feedback_templates={
                    "correct": "Excellent! Your emoji conversation clearly follows the network issue investigation and resolution process.",
                    "similar": "Good attempt! Your sequence covers the main flow, though some elements could be clearer.",
                    "incorrect": "Your conversation doesn't clearly address all steps in the technical support process: question, confirmation/investigation, solution identification, and resolution."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
        
        # Project Management Domain Lesson
        lesson = TutorialLesson(
            title="Project Management Emoji Communication",
            description="Learn specialized emoji patterns for project management contexts.",
            difficulty=DifficultyLevel.ADVANCED,
            category=TutorialCategory.DOMAIN_SPECIFIC,
            prerequisites=["Complex Discussions with Emojis"],
            example_sequences=[
                ("🆕🔥", "New high-priority task"),
                ("⏰⚠️", "Deadline warning"),
                ("📊📈", "Metrics improving"),
                ("👥🔄", "Team iteration/sprint"),
                ("✅🎉", "Task completion celebration")
            ]
        )
        
        # Create exercises
        project_mgmt_set = self.domain_sets["project_management"]
        exercises = [
            TutorialExercise(
                exercise_id="domain_pm_1",
                prompt="Create an emoji sequence to communicate: 'Sprint planning meeting tomorrow, high priority'",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=["🔄📅👥🔥", "👥🔄📅🔥", "🔄👥⏰🔥", "🔄📝👥⏰🔥"],
                hints=["Combine sprint/iteration, meeting/team, schedule/time, and priority indicators."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates a high-priority sprint planning meeting scheduled for tomorrow.",
                    "similar": "Good attempt! Your sequence indicates a sprint meeting, but could be more specific about the timing or priority.",
                    "incorrect": "Your response doesn't clearly communicate a high-priority sprint planning meeting. Try combining emojis for sprint (🔄), team (👥), schedule (📅), and priority (🔥)."
                }
            ),
            TutorialExercise(
                exercise_id="domain_pm_2",
                prompt="Translate this project management update: 'Two high-priority tasks are blocked. Team is investigating. Need approval to extend deadline.'",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=[
                    "2️⃣🔥⏸️→👥🔍→🙏⏰🔄",
                    "🔥2️⃣⏸️→👥🔍→⏰🔄❓",
                    "⏸️🔥2️⃣→🔍👥→🙏⏰➕"
                ],
                hints=["Include elements for 'number two', 'high priority', 'blocked', 'team', 'investigation', 'request', and 'deadline extension'."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates the blocked tasks, investigation, and deadline extension request.",
                    "similar": "Good attempt! Your sequence covers the main points, though some elements could be clearer.",
                    "incorrect": "Your response doesn't address all elements of the project management update: two high-priority blocked tasks, team investigation, and deadline extension request."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def _create_cultural_adaptation_lessons(self) -> None:
        """Create lessons for cultural adaptation in emoji usage."""
        lesson = TutorialLesson(
            title="Cultural Variations in Emoji Usage",
            description="Learn how emoji interpretations vary across cultures and how to adapt your communication.",
            difficulty=DifficultyLevel.ADVANCED,
            category=TutorialCategory.CULTURAL_VARIATIONS,
            prerequisites=["Emotional Intensity"],
            example_sequences=[
                ("🙏", "In Western contexts: please/prayer; In Eastern contexts: gratitude/thank you"),
                ("👍", "Positive in most contexts, but potentially offensive in some Middle Eastern cultures"),
                ("🙂", "Friendly smile in Western contexts; possibly sarcastic or passive-aggressive in some online communities"),
                ("🔥", "In Western contexts: popular/trending/exciting; In business contexts: urgent/high priority"),
                ("🍒", "Fruit in most contexts; has suggestive connotations in some contexts which should be avoided in professional communication")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="cultural_1",
                prompt="For each emoji, identify which interpretation would be most appropriate in a global business context:",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="matching",
                options=[
                    {"emoji": "🙏", "interpretations": ["Please approve this", "Thank you", "I'm praying"]},
                    {"emoji": "👊", "interpretations": ["I'll fight you", "Solidarity/agreement", "Fist bump greeting"]},
                    {"emoji": "😊", "interpretations": ["I'm being friendly", "I'm flirting with you", "I'm secretly annoyed"]}
                ],
                correct_answer=[
                    {"emoji": "🙏", "interpretation": "Thank you"},
                    {"emoji": "👊", "interpretation": "Solidarity/agreement"},
                    {"emoji": "😊", "interpretation": "I'm being friendly"}
                ],
                feedback_templates={
                    "correct": "Excellent! You've chosen the most universally appropriate interpretations for a global business context.",
                    "partial": "Some of your choices are appropriate, but others might cause misunderstandings in global contexts.",
                    "incorrect": "Your choices could lead to misunderstandings. In global business contexts, it's best to use the most neutral and widely-accepted interpretations."
                }
            ),
            TutorialExercise(
                exercise_id="cultural_2",
                prompt="You're communicating with a team that spans Western, Eastern Asian, and Middle Eastern cultures. Create an emoji sequence to say 'Thank you for your help' that would be universally understood.",
                difficulty=DifficultyLevel.ADVANCED,
                exercise_type="free_response",
                correct_answer=["🙏", "😊🙏", "🙏😊", "❤️🙏", "🙏❤️"],
                hints=["Choose emojis that are universally understood across cultures for expressing gratitude."],
                feedback_templates={
                    "correct": "Excellent! Your emoji choice is universally understood as gratitude across different cultures.",
                    "similar": "Good attempt! Your sequence conveys gratitude but might be interpreted slightly differently across cultures.",
                    "incorrect": "Your response might lead to misunderstandings across cultures. Consider using 🙏 which is widely recognized for expressing thanks."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def _create_advanced_sequence_lessons(self) -> None:
        """Create lessons for advanced emoji sequences."""
        lesson = TutorialLesson(
            title="Advanced Emoji Sequence Construction",
            description="Learn to create complex, nuanced emoji sequences for sophisticated communication.",
            difficulty=DifficultyLevel.EXPERT,
            category=TutorialCategory.ADVANCED_SEQUENCES,
            prerequisites=["Complex Discussions with Emojis", "Cultural Variations in Emoji Usage"],
            example_sequences=[
                ("📊📉→🔍→🧠💡→📈", "Problem analysis, investigation, solution, and improvement"),
                ("💡→👀🤔→🔄📝→👥💬→👍", "Idea proposal, consideration, revision, discussion, approval"),
                ("⚠️🔄📆🔚→📊⏰→👥🗣️→📝🆕📆", "Sprint ending warning, status review, team discussion, new planning"),
                ("📱❓→📖💭→📝→✅", "Question about product, research, documentation, resolution"),
                ("💼📉⚠️→👥🧩→🔍→👨‍💻🛠️→📈✅", "Business problem, team collaboration, investigation, technical fix, improvement")
            ]
        )
        
        # Create exercises
        exercises = [
            TutorialExercise(
                exercise_id="advanced_1",
                prompt="Create an emoji sequence for the following complex scenario: 'We've identified security vulnerabilities in our system. We need to prioritize fixes, implement them quickly, and then verify the system is secure.'",
                difficulty=DifficultyLevel.EXPERT,
                exercise_type="free_response",
                correct_answer=[
                    "🔒⚠️→🔍→🔥🗂️→👨‍💻🛠️→🔒✅",
                    "⚠️🔒→👥🔍→🗂️🔥→🛠️→✅🔒",
                    "🔒⚠️🔍→🗂️🔥→👨‍💻→🔒✅"
                ],
                hints=["Include elements for security/lock, warning/vulnerability, investigation, prioritization, implementation, and verification."],
                feedback_templates={
                    "correct": "Excellent! Your emoji sequence clearly communicates the security vulnerability workflow from identification to resolution.",
                    "similar": "Good attempt! Your sequence covers the main elements of the security scenario, though some transitions could be clearer.",
                    "incorrect": "Your response doesn't address all elements of the security scenario: vulnerability identification, prioritization, implementation, and verification."
                }
            ),
            TutorialExercise(
                exercise_id="advanced_2",
                prompt="Create a complex emoji conversation about launching a new product, addressing customer feedback, making improvements, and celebrating success.",
                difficulty=DifficultyLevel.EXPERT,
                exercise_type="free_response",
                correct_answer=[
                    "🚀📱→👥💬⚠️→🔍📝→🛠️📱→📈🎉",
                    "📱🆕🚀→👥💬⚠️→🧠🛠️→📱📈→🎉",
                    "🚀📱→👂👥⚠️→📝🔄→📱✅→📈🎉"
                ],
                hints=["Include elements for product launch, customer feedback (including issues), analysis, improvements, metrics, and celebration."],
                feedback_templates={
                    "correct": "Excellent! Your emoji conversation clearly communicates the entire product lifecycle from launch through improvement to success.",
                    "similar": "Good attempt! Your sequence covers most of the product journey, though some elements could be more specific.",
                    "incorrect": "Your conversation doesn't address all required elements of the product lifecycle: launch, feedback collection, issue identification, improvements, and success measurement."
                }
            )
        ]
        
        lesson.exercises = [ex.exercise_id for ex in exercises]
        
        self.lessons = {**self.lessons, lesson.title: lesson}
        for exercise in exercises:
            self.exercises = {**self.exercises, exercise.exercise_id: exercise}
    
    def create_user_profile(self, user_id, name) -> int:
        """Create a new user profile."""
        if user_id in self.user_profiles:
            return False
        
        self.user_profiles = {**self.user_profiles, user_id: UserProfile(user_id=user_id, name=name)}
        return True
    
    def get_lesson(self, lesson_title) -> None:
        """Get a lesson by title."""
        return self.lessons.get(lesson_title)
    
    def get_exercise(self, exercise_id) -> None:
        """Get an exercise by ID."""
        return self.exercises.get(exercise_id)
    
    def get_user_profile(self, user_id) -> None:
        """Get a user profile by ID."""
        return self.user_profiles.get(user_id)
    
    def get_lesson_progress(self, user_id, lesson_title) -> Optional[Dict[str, Any]]:
        """Get a user's progress on a specific lesson."""
        user = self.get_user_profile(user_id)
        if not user:
            return None
        
        return {
            "completed": lesson_title in user.completed_lessons,
            "score": user.lesson_scores.get(lesson_title, 0.0)
        }
    
    def evaluate_exercise_response(self, exercise_id, user_response) -> Dict[str, Any]:
        """Evaluate a user's response to an exercise."""
        exercise = self.get_exercise(exercise_id)
        if not exercise:
            return {"success": False, "message": "Exercise not found"}
        
        if exercise.exercise_type == "multiple_choice":
            correct = user_response == exercise.correct_answer
            feedback_key = "correct" if correct else "incorrect"
            return {
                "success": True,
                "correct": correct,
                "feedback": exercise.feedback_templates.get(feedback_key, "")
            }
        elif exercise.exercise_type == "free_response":
            if isinstance(exercise.correct_answer, list):
                if user_response in exercise.correct_answer:
                    return {
                        "success": True,
                        "correct": True,
                        "feedback": exercise.feedback_templates.get("correct", "")
                    }
                else:
                    # Check for similarity - this would be more sophisticated in a real implementation
                    similar = any(r.split()[0] in user_response for r in exercise.correct_answer)
                    feedback_key = "similar" if similar else "incorrect"
                    return {
                        "success": True,
                        "correct": False,
                        "similar": similar,
                        "feedback": exercise.feedback_templates.get(feedback_key, "")
                    }
            else:
                correct = user_response == exercise.correct_answer
                feedback_key = "correct" if correct else "incorrect"
                return {
                    "success": True,
                    "correct": correct,
                    "feedback": exercise.feedback_templates.get(feedback_key, "")
                }
        elif exercise.exercise_type == "matching":
            # Simplified matching evaluation
            correct_count = 0
            total = len(exercise.correct_answer)
            
            for i, item in enumerate(user_response):
                if i < total and item == exercise.correct_answer[i]:
                    correct_count += 1
            
            if correct_count == total:
                feedback_key = "correct"
                correct = True
            elif correct_count > 0:
                feedback_key = "partial"
                correct = False
            else:
                feedback_key = "incorrect"
                correct = False
                
            return {
                "success": True,
                "correct": correct,
                "score": correct_count / total,
                "feedback": exercise.feedback_templates.get(feedback_key, "")
            }
        elif exercise.exercise_type == "sequence_completion":
            correct = user_response == exercise.correct_answer
            feedback_key = "correct" if correct else "incorrect"
            return {
                "success": True,
                "correct": correct,
                "feedback": exercise.feedback_templates.get(feedback_key, "")
            }
        else:
            return {"success": False, "message": "Unsupported exercise type"}
    
    def complete_lesson(self, user_id, lesson_title, score=1.0) -> int:
        """Mark a lesson as completed for a user."""
        user = self.get_user_profile(user_id)
        if not user:
            return False
        
        if lesson_title not in self.lessons:
            return False
        
        if lesson_title not in user.completed_lessons:
            user.completed_lessons.append(lesson_title)
        
        user.lesson_scores[lesson_title] = score
        
        # Update user's difficulty level based on progress
        self._update_user_difficulty(user)
        
        return True
    
    def _update_user_difficulty(self, user) -> None:
        """Update a user's difficulty level based on their progress."""
        completed_count = len(user.completed_lessons)
        avg_score = sum(user.lesson_scores.values()) / max(1, len(user.lesson_scores))
        
        beginner_lessons = [l for l in self.lessons.values() if l.difficulty == DifficultyLevel.BEGINNER]
        intermediate_lessons = [l for l in self.lessons.values() if l.difficulty == DifficultyLevel.INTERMEDIATE]
        advanced_lessons = [l for l in self.lessons.values() if l.difficulty == DifficultyLevel.ADVANCED]
        expert_lessons = [l for l in self.lessons.values() if l.difficulty == DifficultyLevel.EXPERT]
        
        # Simple progression logic
        if all(l.title in user.completed_lessons for l in expert_lessons) and avg_score > 0.8:
            user.current_difficulty = DifficultyLevel.FLUENT
        elif all(l.title in user.completed_lessons for l in advanced_lessons) and avg_score > 0.7:
            user.current_difficulty = DifficultyLevel.EXPERT
        elif all(l.title in user.completed_lessons for l in intermediate_lessons) and avg_score > 0.6:
            user.current_difficulty = DifficultyLevel.ADVANCED
        elif all(l.title in user.completed_lessons for l in beginner_lessons) and avg_score > 0.5:
            user.current_difficulty = DifficultyLevel.INTERMEDIATE
        else:
            user.current_difficulty = DifficultyLevel.BEGINNER
    
    def get_recommended_lessons(self, user_id, count=3) -> List[Any]:
        """Get recommended lessons for a user based on their progress and difficulty level."""
        user = self.get_user_profile(user_id)
        if not user:
            return []
        
        # Filter out completed lessons
        available_lessons = [l for l in self.lessons.values() if l.title not in user.completed_lessons]
        
        # Filter by difficulty - prefer current level, but include some from next level
        same_level = [l for l in available_lessons if l.difficulty == user.current_difficulty]
        next_level = []
        
        if user.current_difficulty == DifficultyLevel.BEGINNER:
            next_level = [l for l in available_lessons if l.difficulty == DifficultyLevel.INTERMEDIATE]
        elif user.current_difficulty == DifficultyLevel.INTERMEDIATE:
            next_level = [l for l in available_lessons if l.difficulty == DifficultyLevel.ADVANCED]
        elif user.current_difficulty == DifficultyLevel.ADVANCED:
            next_level = [l for l in available_lessons if l.difficulty == DifficultyLevel.EXPERT]
        
        # Filter by prerequisites
        eligible_lessons = []
        for lesson in same_level + next_level:
            prerequisites_met = all(prereq in user.completed_lessons for prereq in lesson.prerequisites)
            if prerequisites_met:
                eligible_lessons.append(lesson)
        
        # Sort by relevance (simple implementation)
        # In a real system, this would use more sophisticated relevance metrics
        if user.preferred_domains:
            domain_lessons = [l for l in eligible_lessons if l.category == TutorialCategory.DOMAIN_SPECIFIC]
            other_lessons = [l for l in eligible_lessons if l.category != TutorialCategory.DOMAIN_SPECIFIC]
            sorted_lessons = domain_lessons + other_lessons
        else:
            sorted_lessons = eligible_lessons
        
        # Return top N recommendations
        return sorted_lessons[:count]
    
    def record_emoji_usage(self, user_id, emoji) -> int:
        """Record emoji usage for a user to track patterns."""
        user = self.get_user_profile(user_id)
        if not user:
            return False
        
        if emoji in user.frequently_used_emojis:
            user.frequently_used_emojis[emoji] += 1
        else:
            user.frequently_used_emojis[emoji] = 1
        
        return True
    
    def add_to_personal_dictionary(self, user_id, emoji, meaning) -> int:
        """Add an emoji to a user's personal dictionary."""
        user = self.get_user_profile(user_id)
        if not user:
            return False
        
        user.personal_emoji_dictionary[emoji] = meaning
        return True
    
    def get_personal_dictionary(self, user_id) -> Dict[str, Any]:
        """Get a user's personal emoji dictionary."""
        user = self.get_user_profile(user_id)
        if not user:
            return {}
        
        return user.personal_emoji_dictionary
    
    def start_tutorial_session(self, user_id, lesson_title) -> Dict[str, Any]:
        """Start a tutorial session for a user."""
        user = self.get_user_profile(user_id)
        lesson = self.get_lesson(lesson_title)
        
        if not user or not lesson:
            return {"success": False, "message": "User or lesson not found"}
        
        # Check prerequisites
        prerequisites_met = all(prereq in user.completed_lessons for prereq in lesson.prerequisites)
        if not prerequisites_met:
            return {
                "success": False, 
                "message": "Prerequisites not met", 
                "missing_prerequisites": [p for p in lesson.prerequisites if p not in user.completed_lessons]
            }
        
        # Create session record
        session = {
            "session_id": f"{user_id}_{lesson_title}_{datetime.datetime.now().isoformat()}",
            "user_id": user_id,
            "lesson_title": lesson_title,
            "start_time": datetime.datetime.now(),
            "status": "in_progress",
            "exercise_results": {}
        }
        
        user.session_history.append(session)
        
        return {
            "success": True,
            "session": session,
            "lesson": {
                "title": lesson.title,
                "description": lesson.description,
                "difficulty": lesson.difficulty.name,
                "category": lesson.category.name,
                "example_sequences": lesson.example_sequences,
                "exercises": [self.get_exercise(ex_id) for ex_id in lesson.exercises]
            }
        }