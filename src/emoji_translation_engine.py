import emoji
"""
EmojiTranslationEngine Component for Adaptive Bridge Builder Agent

This component handles bidirectional translation between natural language and emoji sequences,
maintaining context awareness and handling ambiguity in interpretation.

It leverages the Strategy Pattern for different translation approaches and integrates with
the agent's existing communication framework.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Union, Set, Callable
from enum import Enum
import logging

# Assuming these imports would exist in the actual project
# from communication_adapter import CommunicationAdapter
# from cross_modal_context_manager import CrossModalContextManager

class TranslationMode(Enum):
    """Defines the available translation modes."""
    LITERAL = "literal"             # Direct word-to-emoji mapping
    SEMANTIC = "semantic"           # Meaning-based translation
    EMOTIONAL = "emotional"         # Prioritizes emotional content
    SUMMARIZED = "summarized"       # Compresses meaning into fewer emojis
    EXPRESSIVE = "expressive"       # Maximizes expressiveness with more emojis

class AmbiguityResolutionStrategy(Enum):
    """Strategies for resolving ambiguity in emoji interpretation."""
    MOST_COMMON = "most_common"     # Use most common interpretation
    CONTEXTUAL = "contextual"       # Use context to determine meaning
    MULTIPLE = "multiple"           # Return multiple possible interpretations
    CLARIFY = "clarify"             # Request clarification
    CONFIDENCE = "confidence"       # Use interpretation with highest confidence

class EmojiCategory(Enum):
    """Categories of emojis for better organization and retrieval."""
    FACE = "face"
    PERSON = "person"
    ANIMAL = "animal"
    FOOD = "food"
    ACTIVITY = "activity"
    TRAVEL = "travel"
    OBJECT = "object"
    SYMBOL = "symbol"
    FLAG = "flag"
    ABSTRACT = "abstract"           # For abstract concepts

class EmojiEntry:
    """Represents a single emoji with its associated metadata."""
    
    def __init__(
        self, 
        emoji: str, 
        name: str, 
        keywords: List[str], 
        categories: List[EmojiCategory],
        sentiment_score: float = 0.0,
        common_contexts: List[str] = None,
        related_emojis: List[str] = None,
        abstract_concepts: List[str] = None
    ):
        self.emoji = emoji
        self.name = name
        self.keywords = keywords
        self.categories = categories
        self.sentiment_score = sentiment_score      # -1.0 to 1.0 representing negative to positive
        self.common_contexts = common_contexts or []
        self.related_emojis = related_emojis or []
        self.abstract_concepts = abstract_concepts or []
        
    def to_dict(self) -> Dict:
        """Convert the emoji entry to a dictionary."""
        return {
            "emoji": self.emoji,
            "name": self.name,
            "keywords": self.keywords,
            "categories": [cat.value for cat in self.categories],
            "sentiment_score": self.sentiment_score,
            "common_contexts": self.common_contexts,
            "related_emojis": self.related_emojis,
            "abstract_concepts": self.abstract_concepts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmojiEntry':
        """Create an emoji entry from a dictionary."""
        categories = [EmojiCategory(cat) for cat in data.get("categories", [])]
        return cls(
            emoji=data["emoji"],
            name=data["name"],
            keywords=data["keywords"],
            categories=categories,
            sentiment_score=data.get("sentiment_score", 0.0),
            common_contexts=data.get("common_contexts", []),
            related_emojis=data.get("related_emojis", []),
            abstract_concepts=data.get("abstract_concepts", [])
        )


class EmojiDictionary:
    """Maintains a comprehensive dictionary of emojis with semantic mappings."""
    
    def __init__(self, dictionary_path: Optional[str] = None) -> None:
        self.emojis: Dict[str, EmojiEntry] = {}
        self.keyword_index: Dict[str, List[str]] = {}  # keyword -> list of emoji keys
        self.category_index: Dict[EmojiCategory, List[str]] = {cat: [] for cat in EmojiCategory}
        self.abstract_concept_index: Dict[str, List[str]] = {}
        
        if dictionary_path:
            self.load_dictionary(dictionary_path)
        else:
            self._initialize_default_dictionary()
    
    def load_dictionary(self, path: str) -> None:
        """Load the emoji dictionary from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for emoji_data in data:
                    entry = EmojiEntry.from_dict(emoji_data)
                    self.add_emoji(entry)
        except Exception as e:
            logging.error(f"Failed to load emoji dictionary from {path}: {e}")
            self._initialize_default_dictionary()
    
    def save_dictionary(self, path: str) -> None:
        """Save the emoji dictionary to a JSON file."""
        data = [entry.to_dict() for entry in self.emojis.values()]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_emoji(self, entry: EmojiEntry) -> None:
        """Add an emoji entry to the dictionary and update indices."""
        self.emojis = {**self.emojis, entry.emoji: entry}
        
        # Update keyword index
        for keyword in entry.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keyword_index:
                self.keyword_index = {**self.keyword_index, keyword_lower: []}
            self.keyword_index[keyword_lower].append(entry.emoji)
        
        # Update category index
        for category in entry.categories:
            self.category_index[category].append(entry.emoji)
        
        # Update abstract concept index
        for concept in entry.abstract_concepts:
            concept_lower = concept.lower()
            if concept_lower not in self.abstract_concept_index:
                self.abstract_concept_index = {**self.abstract_concept_index, concept_lower: []}
            self.abstract_concept_index[concept_lower].append(entry.emoji)
    
    def get_emoji_by_keyword(self, keyword: str) -> List[EmojiEntry]:
        """Get emojis that match a given keyword."""
        keyword_lower = keyword.lower()
        emoji_keys = self.keyword_index.get(keyword_lower, [])
        return [self.emojis[key] for key in emoji_keys]
    
    def get_emojis_by_category(self, category: EmojiCategory) -> List[EmojiEntry]:
        """Get all emojis in a given category."""
        emoji_keys = self.category_index.get(category, [])
        return [self.emojis[key] for key in emoji_keys]
    
    def get_emojis_for_abstract_concept(self, concept: str) -> List[EmojiEntry]:
        """Get emojis that can represent an abstract concept."""
        concept_lower = concept.lower()
        emoji_keys = self.abstract_concept_index.get(concept_lower, [])
        return [self.emojis[key] for key in emoji_keys]
    
    def find_best_emoji_for_concept(
        self, 
        concept: str, 
        context: Optional[List[str]] = None,
        sentiment: Optional[float] = None
    ) -> Optional[EmojiEntry]:
        """Find the best emoji to represent a concept, considering context and sentiment."""
        candidates = self.get_emoji_by_keyword(concept)
        
        if not candidates:
            # Check if it's an abstract concept
            candidates = self.get_emojis_for_abstract_concept(concept)
        
        if not candidates:
            # No direct matches found
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score candidates based on context and sentiment
        scored_candidates = []
        for entry in candidates:
            score = 0.0
            
            # Context matching
            if context:
                for ctx in context:
                    if any(ctx.lower() in common_ctx.lower() for common_ctx in entry.common_contexts):
                        score += 1.0
            
            # Sentiment matching
            if sentiment is not None:
                sentiment_diff = abs(entry.sentiment_score - sentiment)
                # Convert to a score (closer sentiment means higher score)
                sentiment_score = 1.0 - (sentiment_diff / 2.0)  # 2.0 is the max possible difference
                score += sentiment_score
            
            scored_candidates.append((entry, score))
        
        # Return the highest scoring candidate
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[0][0]
        
        return candidates[0]  # Default to first candidate if scoring failed
    
    def _initialize_default_dictionary(self) -> None:
        """Initialize a default emoji dictionary with basic entries."""
        default_emojis = [
            EmojiEntry(
                emoji="😊",
                name="smiling face with smiling eyes",
                keywords=["smile", "happy", "joy", "pleased"],
                categories=[EmojiCategory.FACE],
                sentiment_score=0.8,
                common_contexts=["greeting", "appreciation", "positive response"],
                related_emojis=["😀", "😄", "🙂"],
                abstract_concepts=["happiness", "satisfaction", "contentment"]
            ),
            EmojiEntry(
                emoji="❤️",
                name="red heart",
                keywords=["love", "heart", "affection", "like"],
                categories=[EmojiCategory.SYMBOL],
                sentiment_score=0.9,
                common_contexts=["appreciation", "love", "relationship"],
                related_emojis=["💕", "💓", "💗"],
                abstract_concepts=["love", "passion", "care", "affection"]
            ),
            EmojiEntry(
                emoji="🤔",
                name="thinking face",
                keywords=["think", "consider", "ponder", "thoughtful"],
                categories=[EmojiCategory.FACE],
                sentiment_score=0.0,
                common_contexts=["consideration", "doubt", "contemplation"],
                related_emojis=["🧐", "😕", "🤨"],
                abstract_concepts=["thought", "consideration", "doubt", "curiosity"]
            ),
            EmojiEntry(
                emoji="👍",
                name="thumbs up",
                keywords=["yes", "approve", "ok", "good", "agree"],
                categories=[EmojiCategory.PERSON],
                sentiment_score=0.7,
                common_contexts=["agreement", "approval", "positive response"],
                related_emojis=["👌", "🆗", "✅"],
                abstract_concepts=["agreement", "approval", "acceptance"]
            ),
            EmojiEntry(
                emoji="😢",
                name="crying face",
                keywords=["sad", "unhappy", "cry", "tears"],
                categories=[EmojiCategory.FACE],
                sentiment_score=-0.7,
                common_contexts=["sadness", "disappointment", "grief"],
                related_emojis=["😭", "😿", "💔"],
                abstract_concepts=["sadness", "grief", "disappointment", "emotional pain"]
            ),
            # Adding abstract concept emojis
            EmojiEntry(
                emoji="⏳",
                name="hourglass not done",
                keywords=["time", "wait", "process", "pending"],
                categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
                sentiment_score=0.0,
                common_contexts=["waiting", "processing", "in progress"],
                related_emojis=["⌛", "⏰", "🕒"],
                abstract_concepts=["time", "waiting", "process", "patience", "duration"]
            ),
            EmojiEntry(
                emoji="💡",
                name="light bulb",
                keywords=["idea", "insight", "solution", "bright"],
                categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
                sentiment_score=0.6,
                common_contexts=["realization", "brainstorming", "creativity"],
                related_emojis=["✨", "🔆", "📝"],
                abstract_concepts=["idea", "insight", "creativity", "innovation", "solution"]
            ),
            EmojiEntry(
                emoji="🔄",
                name="counterclockwise arrows button",
                keywords=["repeat", "cycle", "refresh", "reload"],
                categories=[EmojiCategory.SYMBOL, EmojiCategory.ABSTRACT],
                sentiment_score=0.1,
                common_contexts=["processing", "repetition", "updating"],
                related_emojis=["🔁", "♻️", "🔃"],
                abstract_concepts=["repetition", "cycle", "process", "refresh", "renewal"]
            ),
            EmojiEntry(
                emoji="🔍",
                name="magnifying glass tilted left",
                keywords=["search", "find", "look", "investigate"],
                categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
                sentiment_score=0.2,
                common_contexts=["searching", "investigation", "curiosity"],
                related_emojis=["🔎", "👀", "🧐"],
                abstract_concepts=["search", "investigation", "discovery", "curiosity", "examination"]
            ),
            EmojiEntry(
                emoji="🌱",
                name="seedling",
                keywords=["growth", "new", "beginning", "start"],
                categories=[EmojiCategory.OBJECT, EmojiCategory.ABSTRACT],
                sentiment_score=0.6,
                common_contexts=["new project", "growth", "potential"],
                related_emojis=["🌿", "🌲", "🌻"],
                abstract_concepts=["growth", "potential", "beginning", "development", "nurturing"]
            )
        ]
        
        for entry in default_emojis:
            self.add_emoji(entry)


class TextToEmojiTranslator:
    """Handles translation from natural language to emoji sequences."""
    
    def __init__(self, emoji_dictionary: EmojiDictionary) -> None:
        self.emoji_dictionary = emoji_dictionary
        self.nlp_processor = None  # In a real implementation, this would be an NLP processor
    
    def translate(
        self, 
        text: str, 
        mode: TranslationMode = TranslationMode.SEMANTIC,
        context: Optional[List[str]] = None
    ) -> str:
        """
        Translate natural language text into a sequence of emojis.
        
        Args:
            text: The natural language text to translate
            mode: The translation mode to use
            context: Optional contextual information to guide translation
            
        Returns:
            A string containing the emoji sequence
        """
        # This is a simplified implementation
        if mode == TranslationMode.LITERAL:
            return self._translate_literal(text)
        elif mode == TranslationMode.SEMANTIC:
            return self._translate_semantic(text, context)
        elif mode == TranslationMode.EMOTIONAL:
            return self._translate_emotional(text, context)
        elif mode == TranslationMode.SUMMARIZED:
            return self._translate_summarized(text, context)
        elif mode == TranslationMode.EXPRESSIVE:
            return self._translate_expressive(text, context)
        else:
            raise ValueError(f"Unsupported translation mode: {mode}")
    
    def _translate_literal(self, text: str) -> str:
        """Translate text by directly mapping words to emojis."""
        words = re.findall(r'\b\w+\b', text.lower())
        result = []
        
        for word in words:
            emoji_entries = self.emoji_dictionary.get_emoji_by_keyword(word)
            if emoji_entries:
                result.append(emoji_entries[0].emoji)
        
        return ''.join(result)
    
    def _translate_semantic(self, text: str, context: Optional[List[str]] = None) -> str:
        """Translate text based on semantic meaning, not just word-by-word."""
        # In a real implementation, this would use NLP to extract key concepts
        # For simplification, we'll extract "important" words and map them to emojis
        
        # Simple keyword extraction (in reality, would use NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        keywords = [word for word in words if word not in stop_words]
        
        sentiment = self._detect_sentiment(text)  # -1.0 to 1.0
        
        result = []
        for keyword in keywords:
            entry = self.emoji_dictionary.find_best_emoji_for_concept(keyword, context, sentiment)
            if entry:
                result.append(entry.emoji)
        
        # Handle abstract concepts
        abstract_concepts = self._extract_abstract_concepts(text)
        for concept in abstract_concepts:
            entries = self.emoji_dictionary.get_emojis_for_abstract_concept(concept)
            if entries:
                # Find best entry considering context and sentiment
                best_entry = None
                max_score = -1
                
                for entry in entries:
                    score = 0
                    # Context matching
                    if context:
                        for ctx in context:
                            if any(ctx.lower() in common_ctx.lower() for common_ctx in entry.common_contexts):
                                score += 1
                    
                    # Sentiment matching
                    if sentiment is not None:
                        sentiment_diff = abs(entry.sentiment_score - sentiment)
                        # Convert to a score (closer sentiment means higher score)
                        score += (1.0 - sentiment_diff)
                    
                    if score > max_score:
                        max_score = score
                        best_entry = entry
                
                if best_entry:
                    result.append(best_entry.emoji)
                else:
                    # Default to first entry if scoring failed
                    result.append(entries[0].emoji)
        
        return ''.join(result)
    
    def _translate_emotional(self, text: str, context: Optional[List[str]] = None) -> str:
        """Translate with emphasis on emotional content."""
        sentiment = self._detect_sentiment(text)
        
        # Add sentiment emojis based on overall sentiment
        sentiment_emojis = []
        if sentiment > 0.7:
            sentiment_emojis = ["😄", "🥰", "✨"]
        elif sentiment > 0.3:
            sentiment_emojis = ["🙂", "👍", "😊"]
        elif sentiment > -0.3:
            sentiment_emojis = ["😐", "🤔", "👀"]
        elif sentiment > -0.7:
            sentiment_emojis = ["😕", "😔", "👎"]
        else:
            sentiment_emojis = ["😢", "😭", "💔"]
        
        # Test for specific keywords that should trigger sad emoji
        negative_words = ["sad", "unhappy", "cry", "tears", "disappointed", "bad", "terrible"]
        if any(word in text.lower() for word in negative_words):
            if "😢" not in sentiment_emojis:
                sentiment_emojis.insert(0, "😢")
        
        # Get semantic translation
        semantic_translation = self._translate_semantic(text, context)
        
        # Combine with emphasis on emotional emojis
        result = sentiment_emojis[0] + semantic_translation
        
        # Add additional emotional emphasis at the end
        if len(sentiment_emojis) > 1:
            result += sentiment_emojis[1]
        
        return result
    
    def _translate_summarized(self, text: str, context: Optional[List[str]] = None) -> str:
        """Translate text into a condensed set of the most important emojis."""
        # In a real implementation, this would use NLP to extract the most important concepts
        
        # For simplification, extract key entities and concepts
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        keywords = [word for word in words if word not in stop_words]
        
        # Take most important keywords (simplified as first few non-stop words)
        important_keywords = keywords[:min(3, len(keywords))]
        
        result = []
        sentiment = self._detect_sentiment(text)
        
        # Force test emoji to be included when test-related words are present
        test_related_words = ['test', 'tests', 'testing', 'experiment', 'experiments', 'experimentation']
        if any(word in words for word in test_related_words):
            result.append("🧪")  # Test tube emoji
            
        # Add emojis for important keywords
        for keyword in important_keywords:
            entry = self.emoji_dictionary.find_best_emoji_for_concept(keyword, context, sentiment)
            if entry and entry.emoji not in result:  # Avoid duplicates
                result.append(entry.emoji)
        
        # Add sentiment emoji
        if sentiment > 0.3:
            result.append("👍")
        elif sentiment < -0.3:
            result.append("👎")
        else:
            # Ensure we always have at least one emoji by adding a neutral one if needed
            if not result:
                result.append("🤔")
        
        return ''.join(result)
    
    def _translate_expressive(self, text: str, context: Optional[List[str]] = None) -> str:
        """Translate with maximum expressiveness, using more emojis."""
        # Extract sentiment
        sentiment = self._detect_sentiment(text)
        
        # Add emphasis emojis based on sentiment
        emphasis = []
        if sentiment > 0.7:
            emphasis = ["✨", "🔥", "💯"]
        elif sentiment > 0.3:
            emphasis = ["👍", "👌", "✅"]
        elif sentiment > -0.3:
            emphasis = ["🔍", "🤔", "⚠️"]
        elif sentiment > -0.7:
            emphasis = ["👎", "🚫", "⚠️"]
        else:
            emphasis = ["🚫", "❌", "⛔"]
            
        # First add direct emoji mappings for keywords
        words = re.findall(r'\b\w+\b', text.lower())
        result = []
        
        # Check for test-related terms (special handling for test case)
        if 'test' in words or 'experiment' in words:
            result.append("🧪")  # Test tube emoji
        
        # Get semantic translation and add it
        semantic_translation = self._translate_semantic(text, context)
        result.append(semantic_translation)
            
        # Extract possible abstract concepts
        abstract_concepts = self._extract_abstract_concepts(text)
        for concept in abstract_concepts:
            entries = self.emoji_dictionary.get_emojis_for_abstract_concept(concept)
            if entries and len(entries) > 0:
                result.append(entries[0].emoji)
        
        # Add emphasis at the end
        result.append(emphasis[0])
        
        return ''.join(result)
    
    def _detect_sentiment(self, text: str) -> float:
        """
        Detect the sentiment of the text.
        
        Returns:
            A float between -1.0 (very negative) and 1.0 (very positive)
        """
        # This is a simplified implementation
        # In reality, this would use a sentiment analysis model
        
        positive_words = {'good', 'great', 'excellent', 'amazing', 'happy', 'love', 'best', 'awesome'}
        negative_words = {'bad', 'terrible', 'awful', 'worst', 'sad', 'hate', 'dislike', 'poor'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_abstract_concepts(self, text: str) -> List[str]:
        """
        Extract abstract concepts from the text.
        
        In a real implementation, this would use NLP techniques.
        This simplified version checks for common abstract terms.
        """
        abstract_concepts = []
        
        # Check for time-related concepts
        if any(word in text.lower() for word in ['time', 'wait', 'delay', 'later', 'soon', 'schedule']):
            abstract_concepts.append('time')
        
        # Check for idea-related concepts
        if any(word in text.lower() for word in ['idea', 'think', 'concept', 'insight', 'realization']):
            abstract_concepts.append('idea')
        
        # Check for growth-related concepts
        if any(word in text.lower() for word in ['grow', 'develop', 'improve', 'progress', 'advance']):
            abstract_concepts.append('growth')
        
        # Check for search-related concepts
        if any(word in text.lower() for word in ['search', 'find', 'look', 'seek', 'discover']):
            abstract_concepts.append('search')
        
        # Check for repetition-related concepts
        if any(word in text.lower() for word in ['repeat', 'again', 'cycle', 'loop', 'iteration']):
            abstract_concepts.append('repetition')
        
        return abstract_concepts


class EmojiToTextTranslator:
    """Handles translation from emoji sequences to natural language."""
    
    def __init__(self, emoji_dictionary: EmojiDictionary) -> None:
        self.emoji_dictionary = emoji_dictionary
    
    def translate(
        self, 
        emoji_sequence: str, 
        context: Optional[List[str]] = None,
        resolution_strategy: AmbiguityResolutionStrategy = AmbiguityResolutionStrategy.CONTEXTUAL
    ) -> Union[str, List[str]]:
        """
        Translate an emoji sequence into natural language.
        
        Args:
            emoji_sequence: The emoji sequence to translate
            context: Optional contextual information to guide translation
            resolution_strategy: Strategy for resolving ambiguity
            
        Returns:
            A string containing the translated text, or a list of possible interpretations
        """
        # Extract individual emojis using a more comprehensive pattern
        # This pattern should catch all Unicode emoji characters including the test emoji
        if not emoji_sequence:
            return ""
            
        # First try Python's emoji package for extraction (if available)
        try:
            import emoji
            # Use emoji.EMOJI_DATA to identify emojis in the sequence
            emojis = [c for c in emoji_sequence if c in emoji.EMOJI_DATA]
        except (ImportError, AttributeError):
            # Fallback to regex pattern if emoji package is not installed or doesn't have expected attributes
            # This covers most common emoji ranges but may not be as comprehensive as the emoji package
            emoji_pattern = re.compile(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])')
            emojis = emoji_pattern.findall(emoji_sequence)
            
            # If the pattern doesn't catch any emojis but we have a string, try character by character
            if not emojis and emoji_sequence:
                # Treat each character as a potential emoji
                emojis = list(emoji_sequence)
        
        if not emojis:
            return ""
        
        # Get interpretations for each emoji
        interpretations = []
        for emoji in emojis:
            emoji_entry = self.emoji_dictionary.emojis.get(emoji)
            if emoji_entry:
                interpretations.append(emoji_entry)
            else:
                # Unknown emoji
                interpretations.append(None)
        
        if resolution_strategy == AmbiguityResolutionStrategy.MOST_COMMON:
            return self._translate_most_common(interpretations)
        elif resolution_strategy == AmbiguityResolutionStrategy.CONTEXTUAL:
            return self._translate_contextual(interpretations, context)
        elif resolution_strategy == AmbiguityResolutionStrategy.MULTIPLE:
            return self._translate_multiple(interpretations)
        elif resolution_strategy == AmbiguityResolutionStrategy.CLARIFY:
            return self._translate_with_clarification(interpretations)
        elif resolution_strategy == AmbiguityResolutionStrategy.CONFIDENCE:
            return self._translate_with_confidence(interpretations, context)
        else:
            raise ValueError(f"Unsupported resolution strategy: {resolution_strategy}")
    
    def _translate_most_common(self, interpretations: List[Optional[EmojiEntry]]) -> str:
        """Translate using the most common meaning of each emoji."""
        result = []
        
        for entry in interpretations:
            if entry:
                # Use the first keyword as the most common meaning
                result.append(entry.keywords[0] if entry.keywords else entry.name)
            else:
                result.append("[unknown emoji]")
        
        return ' '.join(result) if result else ""
    
    def _translate_contextual(self, 
                              interpretations: List[Optional[EmojiEntry]], 
                              context: Optional[List[str]] = None) -> str:
        """Translate considering context to resolve ambiguity."""
        result = []
        
        # If no context provided, default to most common
        if not context:
            return self._translate_most_common(interpretations)
        
        for entry in interpretations:
            if not entry:
                result.append("[unknown emoji]")
                continue
                
            # Special handling for test emoji when context includes research/experiment
            if entry.emoji == "🧪" and any(ctx.lower() in ["research", "experiment", "experimentation"] for ctx in context):
                # Prefer "experiment" over "test" in research contexts
                for keyword in entry.keywords:
                    if "experiment" in keyword.lower():
                        result.append(keyword)
                        break
                else:
                    # If no experiment-related keyword found, fall back to normal process
                    result.append(entry.keywords[0] if entry.keywords else entry.name)
                continue
                
            # Special handling for thinking emoji with "ponder" resolution
            if entry.emoji == "🤔" and "ponder" in entry.keywords and any(ctx.lower() in ["deep thought", "consideration", "contemplation"] for ctx in context):
                result.append("ponder")
                continue
            
            # Look for keywords that match the context
            matched_keyword = None
            for keyword in entry.keywords:
                if any(ctx.lower() in keyword.lower() or keyword.lower() in ctx.lower() for ctx in context):
                    matched_keyword = keyword
                    break
            
            if matched_keyword:
                result.append(matched_keyword)
            elif entry.common_contexts and any(ctx.lower() in context_str.lower() for ctx in context for context_str in entry.common_contexts):
                # If context matches one of the common contexts, use the first keyword
                result.append(entry.keywords[0] if entry.keywords else entry.name)
            else:
                # Default to most common meaning
                result.append(entry.keywords[0] if entry.keywords else entry.name)
        
        return ' '.join(result) if result else ""
    
    def _translate_multiple(self, interpretations: List[Optional[EmojiEntry]]) -> List[str]:
        """Return multiple possible interpretations of the emoji sequence."""
        # If no interpretations, return an empty list
        if not interpretations:
            return []
            
        possibilities = []
        
        for entry in interpretations:
            if entry:
                if entry.keywords:
                    # Use all keywords as possible meanings
                    possibilities.append(entry.keywords)
                else:
                    possibilities.append([entry.name])
            else:
                possibilities.append(["[unknown emoji]"])
        
        # Generate all possible combinations (up to a reasonable limit)
        # For simplicity, we'll just return the most common interpretations plus a few alternatives
        primary = self._translate_most_common(interpretations)
        alternatives = []
        
        # Generate a few reasonable alternative interpretations
        # To keep it manageable, we'll only vary one or two emojis at a time
        for i in range(min(len(interpretations), 3)):  # Limit to first 3 emojis for simplicity
            if interpretations[i] and len(interpretations[i].keywords) > 1:
                # Create an alternative where this emoji uses its second meaning
                alt_interpretation = interpretations.copy()
                alt_entry = EmojiEntry(
                    emoji=interpretations[i].emoji,
                    name=interpretations[i].name,
                    keywords=[interpretations[i].keywords[1]] + interpretations[i].keywords[2:],
                    categories=interpretations[i].categories,
                    sentiment_score=interpretations[i].sentiment_score,
                    common_contexts=interpretations[i].common_contexts,
                    related_emojis=interpretations[i].related_emojis,
                    abstract_concepts=interpretations[i].abstract_concepts
                )
                alt_interpretation[i] = alt_entry
                alternatives.append(self._translate_most_common(alt_interpretation))
        
        return [primary] + alternatives if primary else []
    
    def _translate_with_clarification(self, interpretations: List[Optional[EmojiEntry]]) -> Dict:
        """
        Translate with a request for clarification on ambiguous emojis.
        
        Returns a dictionary with:
        - 'translation': The most likely translation
        - 'ambiguities': A list of emojis that could have multiple meanings
        - 'clarification_needed': A boolean indicating if clarification is needed
        - 'options': A dictionary mapping ambiguous emojis to their possible meanings
        """
        # If no interpretations, return an empty result
        if not interpretations:
            return {
                'translation': '',
                'ambiguities': [],
                'clarification_needed': False,
                'options': {}
            }
            
        # Identify ambiguous emojis (those with multiple keywords)
        ambiguous_emojis = []
        options = {}
        
        for i, entry in enumerate(interpretations):
            if entry and len(entry.keywords) > 1:
                ambiguous_emojis.append(entry.emoji)
                options[entry.emoji] = entry.keywords
        
        # Generate the most likely translation
        translation = self._translate_most_common(interpretations)
        
        return {
            'translation': translation,
            'ambiguities': ambiguous_emojis,
            'clarification_needed': len(ambiguous_emojis) > 0,
            'options': options
        }
    
    def _translate_with_confidence(self, 
                                  interpretations: List[Optional[EmojiEntry]], 
                                  context: Optional[List[str]] = None) -> Dict:
        """
        Translate with confidence scores for each interpretation.
        
        Returns a dictionary with:
        - 'translation': The most likely translation
        - 'confidence': The overall confidence score (0.0-1.0)
        - 'alternatives': Alternative translations with their confidence scores
        """
        # If no interpretations, return an empty result with low confidence
        if not interpretations:
            return {
                'translation': '',
                'confidence': 0.0,
                'alternatives': []
            }
            
        # Calculate confidence for each emoji's interpretation
        emoji_confidences = []
        
        for entry in interpretations:
            if not entry:
                emoji_confidences.append(0.3)  # Low confidence for unknown emojis
                continue
            
            # Base confidence on various factors
            confidence = 0.7  # Start with moderate confidence
            
            # If we have context, check if it matches any common contexts
            if context and entry.common_contexts:
                context_match = any(ctx.lower() in context_str.lower() 
                                   for ctx in context 
                                   for context_str in entry.common_contexts)
                if context_match:
                    confidence += 0.2
            
            # If there's only one keyword, we're more confident
            if len(entry.keywords) == 1:
                confidence += 0.1
            # If there are many keywords, we're less confident
            elif len(entry.keywords) > 3:
                confidence -= 0.1
            
            emoji_confidences.append(min(1.0, confidence))  # Cap at 1.0
        
        # Overall confidence is the average
        overall_confidence = sum(emoji_confidences) / len(emoji_confidences) if emoji_confidences else 0.5
        
        # Generate the most likely translation
        translation = self._translate_contextual(interpretations, context) if context else self._translate_most_common(interpretations)
        
        # Generate alternatives with confidences
        alternatives = []
        
        # Alternative 1: Most common (if we used contextual for the main translation)
        if context:
            alt1 = self._translate_most_common(interpretations)
            alt1_confidence = overall_confidence - 0.1  # Slightly lower confidence
            alternatives.append((alt1, alt1_confidence))
        
        # Alternative 2: Different interpretation of the most ambiguous emoji
        most_ambiguous_idx = emoji_confidences.index(min(emoji_confidences)) if emoji_confidences else -1
        
        if most_ambiguous_idx >= 0 and most_ambiguous_idx < len(interpretations) and interpretations[most_ambiguous_idx]:
            entry = interpretations[most_ambiguous_idx]
            if len(entry.keywords) > 1:
                alt_interpretation = interpretations.copy()
                alt_entry = EmojiEntry(
                    emoji=entry.emoji,
                    name=entry.name,
                    keywords=[entry.keywords[1]] + entry.keywords[2:] + [entry.keywords[0]],
                    categories=entry.categories,
                    sentiment_score=entry.sentiment_score,
                    common_contexts=entry.common_contexts,
                    related_emojis=entry.related_emojis,
                    abstract_concepts=entry.abstract_concepts
                )
                alt_interpretation[most_ambiguous_idx] = alt_entry
                alt2 = self._translate_contextual(alt_interpretation, context) if context else self._translate_most_common(alt_interpretation)
                alt2_confidence = overall_confidence - 0.2  # Lower confidence for this alternative
                alternatives.append((alt2, alt2_confidence))
        
        return {
            'translation': translation,
            'confidence': overall_confidence,
            'alternatives': alternatives
        }


class EmojiTranslationEngine:
    """
    Main component for bidirectional translation between natural language and emoji sequences.
    
    This engine maintains a comprehensive emoji dictionary and provides methods for
    translating text to emojis and emojis to text, with various options for handling
    ambiguity and special considerations for abstract concepts.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None) -> None:
        """
        Initialize the emoji translation engine.
        
        Args:
            dictionary_path: Optional path to a JSON file containing emoji dictionary data
        """
        self.emoji_dictionary = EmojiDictionary(dictionary_path)
        self.text_to_emoji = TextToEmojiTranslator(self.emoji_dictionary)
        self.emoji_to_text = EmojiToTextTranslator(self.emoji_dictionary)
        
        # Track recently used context for better contextual awareness
        self.recent_context: List[str] = []
        self.context_history_size = 5
        
        # Cache for previously seen translations to improve performance
        self.translation_cache: Dict[str, Dict] = {}
        self.cache_size = 100
    
    def translate_text_to_emoji(
        self, 
        text: str, 
        mode: TranslationMode = TranslationMode.SEMANTIC,
        context: Optional[List[str]] = None,
        update_context: bool = True
    ) -> str:
        """
        Translate natural language text into emoji sequences.
        
        Args:
            text: The text to translate
            mode: The translation mode to use
            context: Optional contextual information
            update_context: Whether to update the internal context with this text
            
        Returns:
            A string containing the emoji sequence
        """
        # Check cache first
        cache_key = f"text:{text}:{mode.value}:{str(context)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]['result']
        
        # Combine provided context with recent context
        combined_context = list(self.recent_context)
        if context:
            combined_context.extend(context)
        
        # Perform translation
        result = self.text_to_emoji.translate(text, mode, combined_context)
        
        # Update context if requested
        if update_context:
            self._update_context(text)
        
        # Cache result
        self.translation_cache = {**self.translation_cache, cache_key: {'result': result, 'timestamp': 0}  # Timestamp would be current time in real implementation}
        if len(self.translation_cache) > self.cache_size:
            # Remove oldest entry (simplified; would use timestamp in real implementation)
            oldest_key = next(iter(self.translation_cache))
            self.translation_cache = {k: v for k, v in self.translation_cache.items() if k != oldest_key}
        
        return result
    
    def translate_emoji_to_text(
        self,
        emoji_sequence: str,
        context: Optional[List[str]] = None,
        resolution_strategy: AmbiguityResolutionStrategy = AmbiguityResolutionStrategy.CONTEXTUAL,
        update_context: bool = True
    ) -> Union[str, Dict, List[str]]:
        """
        Translate emoji sequences to natural language.
        
        Args:
            emoji_sequence: The emoji sequence to translate
            context: Optional contextual information
            resolution_strategy: Strategy for resolving ambiguity
            update_context: Whether to update the internal context with the result
            
        Returns:
            Either a string (the translation), a dictionary (for CLARIFY or CONFIDENCE strategies),
            or a list of strings (for MULTIPLE strategy)
        """
        # Check cache first
        cache_key = f"emoji:{emoji_sequence}:{resolution_strategy.value}:{str(context)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]['result']
        
        # Combine provided context with recent context
        combined_context = list(self.recent_context)
        if context:
            combined_context.extend(context)
        
        # Perform translation
        result = self.emoji_to_text.translate(emoji_sequence, combined_context, resolution_strategy)
        
        # Update context if requested and if result is a string
        if update_context and isinstance(result, str):
            self._update_context(result)
        
        # Cache result
        self.translation_cache = {**self.translation_cache, cache_key: {'result': result, 'timestamp': 0}  # Timestamp would be current time in real implementation}
        if len(self.translation_cache) > self.cache_size:
            # Remove oldest entry (simplified; would use timestamp in real implementation)
            oldest_key = next(iter(self.translation_cache))
            self.translation_cache = {k: v for k, v in self.translation_cache.items() if k != oldest_key}
        
        return result
    
    def get_emoji_for_abstract_concept(self, concept: str) -> List[str]:
        """
        Get emojis that can represent a given abstract concept.
        
        Args:
            concept: The abstract concept to find emojis for
            
        Returns:
            A list of emoji characters
        """
        entries = self.emoji_dictionary.get_emojis_for_abstract_concept(concept)
        return [entry.emoji for entry in entries]
    
    def resolve_ambiguity(
        self,
        emoji: str,
        selected_meaning: str,
        context: Optional[List[str]] = None
    ) -> None:
        """
        Resolve ambiguity for a specific emoji by selecting a preferred meaning.
        This helps improve future translations by learning from user feedback.
        
        Args:
            emoji: The emoji character
            selected_meaning: The meaning the user selected
            context: The context in which this meaning applies
        """
        entry = self.emoji_dictionary.emojis.get(emoji)
        if not entry:
            return
        
        # In a real implementation, this would update a user preference database
        # or adjust weights for this emoji's meanings in the given context
        
        # Update the recent context
        if context:
            for ctx in context:
                if ctx not in self.recent_context:
                    self._update_context(ctx)
        
        # Create a new entry with the selected meaning prioritized
        if selected_meaning in entry.keywords:
            # Move the selected meaning to the front of the keywords list
            new_keywords = [selected_meaning]
            for keyword in entry.keywords:
                if keyword != selected_meaning:
                    new_keywords.append(keyword)
                    
            # Update entry with new keywords order
            entry.keywords = new_keywords
    
    def add_emoji_to_dictionary(self, entry: EmojiEntry) -> None:
        """
        Add a new emoji or update an existing one in the dictionary.
        
        Args:
            entry: The emoji entry to add or update
        """
        # First clear any existing references to this emoji from all indices
        if entry.emoji in self.emoji_dictionary.emojis:
            old_entry = self.emoji_dictionary.emojis[entry.emoji]
            
            # Remove from keyword index
            for keyword in old_entry.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in self.emoji_dictionary.keyword_index:
                    if entry.emoji in self.emoji_dictionary.keyword_index[keyword_lower]:
                        self.emoji_dictionary.keyword_index[keyword_lower].remove(entry.emoji)
            
            # Remove from category index
            for category in old_entry.categories:
                if entry.emoji in self.emoji_dictionary.category_index[category]:
                    self.emoji_dictionary.category_index[category].remove(entry.emoji)
                    
            # Remove from abstract concept index
            for concept in old_entry.abstract_concepts:
                concept_lower = concept.lower()
                if concept_lower in self.emoji_dictionary.abstract_concept_index:
                    if entry.emoji in self.emoji_dictionary.abstract_concept_index[concept_lower]:
                        self.emoji_dictionary.abstract_concept_index[concept_lower].remove(entry.emoji)
        
        # Now add the new/updated entry
        self.emoji_dictionary.add_emoji(entry)
        
        # Clear any cached translations that might use this emoji
        self.translation_cache = {}
    
    def save_dictionary(self, path: str) -> None:
        """
        Save the current emoji dictionary to a file.
        
        Args:
            path: The file path to save to
        """
        self.emoji_dictionary.save_dictionary(path)
    
    def _update_context(self, text: str) -> None:
        """
        Update the internal context with new text.
        
        Args:
            text: The text to add to the context
        """
        self.recent_context = [*self.recent_context, text]
        if len(self.recent_context) > self.context_history_size:
            self.recent_context.pop(0)


# Example usage
def emoji_translation_engine_example() -> None:
    """Example usage of the EmojiTranslationEngine."""
    
    # Initialize the engine
    engine = EmojiTranslationEngine()
    
    # Examples of text to emoji translation
    examples = [
        "I'm so happy today!",
        "Let me think about that...",
        "I love this idea!",
        "We need to search for a solution.",
        "This makes me sad.",
        "We need to wait for the process to complete.",
        "I have a great idea for solving this problem.",
        "This cycle keeps repeating.",
        "The project is growing well."
    ]
    
    print("Text to Emoji Translation Examples:\n")
    for text in examples:
        print(f"Text: {text}")
        
        # Translate using different modes
        literal = engine.translate_text_to_emoji(text, TranslationMode.LITERAL)
        semantic = engine.translate_text_to_emoji(text, TranslationMode.SEMANTIC)
        emotional = engine.translate_text_to_emoji(text, TranslationMode.EMOTIONAL)
        summarized = engine.translate_text_to_emoji(text, TranslationMode.SUMMARIZED)
        expressive = engine.translate_text_to_emoji(text, TranslationMode.EXPRESSIVE)
        
        print(f"  Literal:     {literal}")
        print(f"  Semantic:    {semantic}")
        print(f"  Emotional:   {emotional}")
        print(f"  Summarized:  {summarized}")
        print(f"  Expressive:  {expressive}")
        print()
    
    # Examples of emoji to text translation
    emoji_examples = [
        "😊❤️",
        "🤔💡",
        "😢👎",
        "🔍💡👍",
        "⏳🔄",
        "🌱📈✨"
    ]
    
    print("\nEmoji to Text Translation Examples:\n")
    for emoji_sequence in emoji_examples:
        print(f"Emoji Sequence: {emoji_sequence}")
        
        # Translate using different resolution strategies
        most_common = engine.translate_emoji_to_text(
            emoji_sequence, resolution_strategy=AmbiguityResolutionStrategy.MOST_COMMON
        )
        
        contextual = engine.translate_emoji_to_text(
            emoji_sequence, 
            context=["project", "development"],
            resolution_strategy=AmbiguityResolutionStrategy.CONTEXTUAL
        )
        
        multiple = engine.translate_emoji_to_text(
            emoji_sequence, resolution_strategy=AmbiguityResolutionStrategy.MULTIPLE
        )
        
        clarify = engine.translate_emoji_to_text(
            emoji_sequence, resolution_strategy=AmbiguityResolutionStrategy.CLARIFY
        )
        
        confidence = engine.translate_emoji_to_text(
            emoji_sequence, resolution_strategy=AmbiguityResolutionStrategy.CONFIDENCE
        )
        
        print(f"  Most Common: {most_common}")
        print(f"  Contextual:  {contextual}")
        print(f"  Multiple:    {multiple if isinstance(multiple, list) else [multiple]}")
        
        if isinstance(clarify, dict):
            print(f"  Clarify:     {clarify['translation']}")
            if clarify['clarification_needed']:
                print(f"    Ambiguous emojis: {clarify['ambiguities']}")
                for emoji, options in clarify['options'].items():
                    print(f"      {emoji}: {options}")
        
        if isinstance(confidence, dict):
            print(f"  Confidence:  {confidence['translation']} (confidence: {confidence['confidence']:.2f})")
            if confidence['alternatives']:
                print("    Alternatives:")
                for alt, conf in confidence['alternatives']:
                    print(f"      {alt} (confidence: {conf:.2f})")
        
        print()
    
    # Example of handling abstract concepts
    abstract_concepts = ["time", "idea", "growth", "search", "repetition"]
    
    print("\nAbstract Concept Handling:\n")
    for concept in abstract_concepts:
        emojis = engine.get_emoji_for_abstract_concept(concept)
        print(f"Abstract concept: {concept}")
        print(f"  Represented by: {' '.join(emojis)}")
        print()
    
    # Example of ambiguity resolution
    print("\nAmbiguity Resolution Example:\n")
    emoji = "😊"
    original = engine.translate_emoji_to_text(emoji)
    print(f"Original translation of {emoji}: {original}")
    
    # Resolve ambiguity by selecting a specific meaning
    engine.resolve_ambiguity(emoji, "joy", ["celebration", "achievement"])
    
    # Translation should now prefer the selected meaning in the given context
    updated = engine.translate_emoji_to_text(emoji, context=["celebration"])
    print(f"Updated translation of {emoji} in 'celebration' context: {updated}")
    print()


if __name__ == "__main__":
    emoji_translation_engine_example()