import emoji
#!/usr/bin/env python3
"""
Communication Style Analyzer for Adaptive Bridge Builder

This module provides the CommunicationStyleAnalyzer class which analyzes 
message patterns from other agents to determine their communication style
preferences, and helps adapt responses accordingly.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from collections import Counter
import statistics
from enum import Enum
import string

from communication_style import (
    CommunicationStyle, 
    FormalityLevel, 
    DetailLevel, 
    DirectnessLevel, 
    EmotionalTone,
    ResponseSpeed
)
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommunicationStyleAnalyzer")

class MessageDirection(Enum):
    """Enumeration of message directions."""
    SENT = "sent"
    RECEIVED = "received"

class Message:
    """Represents a single message in a conversation."""
    
    def __init__(
        self, 
        content: str, 
        timestamp: str, 
        direction: MessageDirection, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.timestamp = timestamp
        self.direction = direction
        self.metadata = metadata or {}
        self.analyzed = False
        self.analysis_results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "direction": self.direction.value,
            "metadata": self.metadata,
            "analyzed": self.analyzed,
            "analysis_results": self.analysis_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary."""
        direction = MessageDirection(data.get("direction", "received"))
        message = cls(
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            direction=direction,
            metadata=data.get("metadata", {})
        )
        message.analyzed = data.get("analyzed", False)
        message.analysis_results = data.get("analysis_results", {})
        return message
    
    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.direction.value.upper()}: {self.content[:50]}..."

class MessageHistory:
    """Represents a history of messages with an agent."""
    
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.messages: List[Message] = []
        self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def add_message(self, message: Union[Message, Dict[str, Any]]) -> None:
        """Add a message to the history."""
        if isinstance(message, dict):
            message = Message.from_dict(message)
        
        self.messages = [*self.messages, message]
        self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message history to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageHistory':
        """Create a MessageHistory from a dictionary."""
        history = cls(agent_id=data.get("agent_id", "unknown"))
        history.last_updated = data.get("last_updated", datetime.now(timezone.utc).isoformat())
        
        for msg_data in data.get("messages", []):
            history.add_message(Message.from_dict(msg_data))
            
        return history
    
    def get_received_messages(self) -> List[Message]:
        """Get messages received from the other agent."""
        return [msg for msg in self.messages if msg.direction == MessageDirection.RECEIVED]
    
    def get_sent_messages(self) -> List[Message]:
        """Get messages sent to the other agent."""
        return [msg for msg in self.messages if msg.direction == MessageDirection.SENT]
    
    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
    
    def get_conversation_duration(self) -> Optional[float]:
        """
        Get the duration of the conversation in seconds.
        
        Returns None if there are fewer than 2 messages.
        """
        if len(self.messages) < 2:
            return None
        
        try:
            first_ts = datetime.fromisoformat(self.messages[0].timestamp)
            last_ts = datetime.fromisoformat(self.messages[-1].timestamp)
            return (last_ts - first_ts).total_seconds()
        except (ValueError, TypeError):
            logger.warning("Could not calculate conversation duration due to invalid timestamps")
            return None
    
    def get_average_response_time(self) -> Optional[float]:
        """
        Calculate the average response time in seconds.
        
        Returns None if there are insufficient message pairs to calculate.
        """
        if len(self.messages) < 3:
            return None
        
        response_times = []
        for i in range(1, len(self.messages)):
            current = self.messages[i]
            previous = self.messages[i-1]
            
            # Only consider alternating directions
            if current.direction != previous.direction:
                try:
                    curr_ts = datetime.fromisoformat(current.timestamp)
                    prev_ts = datetime.fromisoformat(previous.timestamp)
                    response_time = (curr_ts - prev_ts).total_seconds()
                    if response_time > 0:
                        response_times.append(response_time)
                except (ValueError, TypeError):
                    continue
        
        if not response_times:
            return None
        
        return statistics.mean(response_times)

class CommunicationStyleAnalyzer:
    """
    Analyzes message history to detect communication style preferences.
    
    This class examines patterns in message content, structure, and timing
    to determine style attributes such as formality level, detail preference,
    directness, and emotional tone.
    """
    
    def __init__(self, principle_engine: Optional[PrincipleEngine] = None) -> None:
        """
        Initialize the CommunicationStyleAnalyzer.
        
        Args:
            principle_engine: Optional PrincipleEngine for principle alignment checks.
        """
        self.principle_engine = principle_engine
        self.style_cache: Dict[str, Tuple[CommunicationStyle, datetime]] = {}
        
        # Initialize analyzers for specific attributes
        self._initialize_analyzers()
        
        logger.info("CommunicationStyleAnalyzer initialized")
    
    def _initialize_analyzers(self) -> None:
        """Initialize the pattern analysis components."""
        # Formality analysis patterns
        self.formality_patterns = {
            FormalityLevel.VERY_FORMAL: [
                (r'\b(shall|hereby|aforementioned|pursuant|henceforth)\b', 2.0),
                (r'[^.!?]+[.!?]', 0.5),  # Complete sentences
                (r"\bI'm\b|\byou're\b|\bdon't\b|\bcan't\b|\bit's\b", -1.0),  # Contractions (negative)
                (r'\bDear\b|\bSincerely\b|\bRegards\b', 1.0),
                (r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', 0.5)  # Full names
            ],
            FormalityLevel.FORMAL: [
                (r'\b(would|should|could|may I|kindly)\b', 1.0),
                (r'\b(thank you|pleased to|appreciate your)\b', 0.8),
                (r'\b(please|request|inquire)\b', 0.7),
                (r'\b(Hi|Hello)\s[A-Z][a-z]+\b', 0.5),  # Greetings with name
                (r'\b[A-Z][a-z]+\b', 0.3)  # Capitalized words
            ],
            FormalityLevel.CASUAL: [
                (r'\b(hey|hi there|thanks|sure|okay|ok)\b', 1.0),
                (r'\b(yeah|cool|great|awesome|btw)\b', 1.2),
                (r'!{1,2}', 0.5),  # 1-2 exclamation marks
                (r'\b(gonna|wanna|gotta)\b', 1.5),
                (r'\b(anyways|basically|pretty much)\b', 0.8)
            ],
            FormalityLevel.VERY_CASUAL: [
                (r'\b(yo|sup|lol|haha|wtf|omg|idk|tbh)\b', 2.0),
                (r'!{3,}|\?{3,}', 1.5),  # Multiple punctuation
                (r'\b(u|r|ur|y|k|tho|cuz)\b', 1.8),  # Text shorthand
                (r'(?<![.])[.]{2,}', 1.0),  # Ellipsis without period
                (r'xD|:D|:P|:3|:O|B\)', 1.2)  # Emoticons
            ]
        }
        
        # Detail level analysis patterns
        self.detail_patterns = {
            DetailLevel.VERY_DETAILED: [
                (r'\b(specifically|in particular|to elaborate|furthermore|additionally)\b', 1.5),
                (r'\b(for example|for instance|such as|namely|in other words)\b', 1.2),
                (r'[^.!?]+[.!?]', 0.1),  # Sentence count (worth less, but adds up)
                (r'[:;]\s*[-–—]\s*\w', 1.0),  # Lists with semicolons/colons
                (r'\d+[.,]?\d*\s*%|\d+[.,]?\d*\s*\w+', 0.8)  # Numbers, measurements
            ],
            DetailLevel.DETAILED: [
                (r'\b(because|therefore|since|as a result)\b', 0.8),
                (r'\bexplain\w*\b|\bdescribe\w*\b|\bdetail\w*\b', 1.0),
                (r'\bnote\b|\bimportant\b|\bkey\b|\bsignificant\b', 0.9),
                (r'\b\d+\b|#\d+', 0.5),  # Numbers
                (r'\([^)]+\)|\[[^\]]+\]', 0.7)  # Parenthetical expressions
            ],
            DetailLevel.CONCISE: [
                (r'\b(briefly|in short|simply put|just|only)\b', 1.0),
                (r'\b(main|basic|core|essential)\b', 0.8),
                (r'^.{1,20}$', 0.5),  # Very short messages
                (r'^\s*\w+\s*[.!?]?\s*$', 1.5),  # One-word responses
                (r'\b(ok|sure|fine|done|got it)\b', 1.2)
            ],
            DetailLevel.VERY_CONCISE: [
                (r'^\s*\w{1,3}\s*$', 2.0),  # Ultra-short responses
                (r'\b(k|y|n|ok)\b', 1.8),  # Single letter/short responses
                (r'^\s*[👍👎✓✔️👌]\s*$', 2.0),  # Just an emoji
                (r'^\s*\+1\s*$|^\s*-1\s*$', 1.5),  # Just +1/-1
                (r'^\s*[.!?]\s*$', 2.0)  # Just punctuation
            ]
        }
        
        # Directness analysis patterns
        self.directness_patterns = {
            DirectnessLevel.VERY_DIRECT: [
                (r'^(I need|I want|I require|Do this|Make sure|You must)\b', 2.0),
                (r'\b(immediately|now|asap|urgent|critical)\b', 1.5),
                (r'^\s*\w+[.!]$', 1.0),  # Imperative sentences
                (r'^[A-Z][^.!?]*[.!?]', 1.2),  # Capitalized first sentences
                (r'\b(should|must|need to|have to|required)\b', 1.3)
            ],
            DirectnessLevel.DIRECT: [
                (r'^(Please|Could you|Would you|Let\'s|I think)\b', 1.0),
                (r'\b(recommend|suggest|advise|request)\b', 0.8),
                (r'\?((?!\s+[A-Z]).)*$', 0.7),  # Questions
                (r'\b(direct|clear|straight|explicit)\b', 1.0),
                (r'\b(important|key|main|essential)\b', 0.6)
            ],
            DirectnessLevel.INDIRECT: [
                (r'\b(perhaps|maybe|possibly|might|could consider)\b', 1.0),
                (r'\b(wonder|curious|thought|idea|option)\b', 0.8),
                (r'\b(sometime|when you can|if possible|at your convenience)\b', 1.2),
                (r'\b(appreciate|grateful|thankful)\b', 0.7),
                (r'\b(sorry|apologize|excuse|bother)\b', 0.9)
            ],
            DirectnessLevel.VERY_INDIRECT: [
                (r'\b(if it\'s not too much trouble|if you don\'t mind|whenever you get a chance)\b', 2.0),
                (r'\b(was wondering if perhaps|just thinking that maybe|not sure if this is possible)\b', 1.8),
                (r'\b(hint|imply|suggest|allude|infer)\b', 1.5),
                (r'\b(bit|little|slight|somewhat|kind of|sort of)\b', 1.3),
                (r'\b(hypothetically|theoretically|in an ideal world)\b', 1.7)
            ]
        }
        
        # Emotional tone analysis patterns
        self.tone_patterns = {
            EmotionalTone.VERY_POSITIVE: [
                (r'\b(excellent|amazing|outstanding|exceptional|fantastic|wonderful)\b', 1.8),
                (r'\b(love|adore|delighted|thrilled|excited|impressive)\b', 1.5),
                (r'!{2,}', 1.0),  # Multiple exclamation marks
                (r'\b(perfect|brilliant|superb|tremendous|magnificent)\b', 2.0),
                (r'[😄😁😊🥰😍🤩😃😀👍👏❤️💯]', 1.5)  # Positive emojis
            ],
            EmotionalTone.POSITIVE: [
                (r'\b(good|great|nice|happy|pleased|glad)\b', 1.0),
                (r'\b(thank|appreciate|welcome|helpful|enjoy)\b', 0.8),
                (r'\b(success|achievement|progress|improvement)\b', 0.7),
                (r'\b(opportunity|benefit|advantage|valuable)\b', 0.6),
                (r'[:)(:;):D]', 0.5)  # Simple positive emoticons
            ],
            EmotionalTone.NEGATIVE: [
                (r'\b(bad|poor|wrong|difficult|hard|issue|problem)\b', 1.0),
                (r'\b(sorry|unfortunate|regret|concern|worry)\b', 0.8),
                (r'\b(fail|error|mistake|defect|fault|flaw)\b', 0.7),
                (r'\b(disappoint|frustrat|upset|annoy|bother)\b', 0.6),
                (r'[:(;(;/]', 0.5)  # Simple negative emoticons
            ],
            EmotionalTone.VERY_NEGATIVE: [
                (r'\b(terrible|horrible|awful|disastrous|catastrophic)\b', 1.8),
                (r'\b(hate|despise|detest|abhor|loathe|disgusting)\b', 1.5),
                (r'\b(furious|angry|enraged|outraged|infuriated)\b', 1.6),
                (r'\b(useless|worthless|pathetic|ridiculous|absurd)\b', 1.7),
                (r'[😡😠🤬😤👎💔😞😖]', 1.5)  # Negative emojis
            ]
        }
    
    def analyze_message_history(self, history: MessageHistory) -> CommunicationStyle:
        """
        Analyze a message history to determine communication style.
        
        Args:
            history: MessageHistory object containing messages from an agent.
            
        Returns:
            CommunicationStyle object representing the agent's communication preferences.
        """
        agent_id = history.agent_id
        received_messages = history.get_received_messages()
        
        # Check if we already have a recent analysis for this agent
        if agent_id in self.style_cache:
            cached_style, cache_time = self.style_cache[agent_id]
            cache_age = (datetime.now(timezone.utc) - cache_time).total_seconds()
            
            # If the cache is less than an hour old and we have enough messages analyzed
            if cache_age < 3600 and cached_style.sample_count >= len(received_messages):
                logger.info(f"Using cached communication style for agent {agent_id}")
                return cached_style
        
        # Not in cache or cache outdated, perform full analysis
        logger.info(f"Analyzing communication style for agent {agent_id} with {len(received_messages)} messages")
        
        if len(received_messages) < 3:
            logger.warning(f"Limited message history for agent {agent_id} ({len(received_messages)} messages)")
        
        # Initialize a new communication style
        style = CommunicationStyle(agent_id=agent_id)
        style.sample_count = len(received_messages)
        style.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Analyze various aspects of communication style
        formality_score = self._analyze_formality(received_messages)
        detail_score = self._analyze_detail_level(received_messages)
        directness_score = self._analyze_directness(received_messages)
        tone_score = self._analyze_emotional_tone(received_messages)
        
        # Set the core style attributes
        style.formality = self._map_score_to_enum(formality_score, FormalityLevel)
        style.detail_level = self._map_score_to_enum(detail_score, DetailLevel)
        style.directness = self._map_score_to_enum(directness_score, DirectnessLevel)
        style.emotional_tone = self._map_score_to_enum(tone_score, EmotionalTone)
        
        # Analyze additional preferences
        style.prefers_acknowledgments = self._analyze_acknowledgment_preference(received_messages)
        style.prefers_structured_responses = self._analyze_structure_preference(received_messages)
        style.prefers_examples = self._analyze_examples_preference(received_messages)
        style.vocabulary_level = self._analyze_vocabulary_complexity(received_messages)
        
        # Analyze response timing if we have timing data
        response_time = history.get_average_response_time()
        if response_time is not None:
            style.response_speed = self._analyze_response_speed(response_time)
        
        # Set consistency and confidence metrics
        style.consistency_score = self._calculate_consistency_score(received_messages)
        style.confidence_level = self._calculate_confidence_level(received_messages)
        
        # Add any notable style observations
        style.style_notes = self._generate_style_notes(received_messages)
        
        # Cache the result
        self.style_cache = {**self.style_cache, agent_id: (style, datetime.now(timezone.utc))}
        
        logger.info(f"Completed style analysis for agent {agent_id}: {style.to_dict()}")
        return style
    
    def _analyze_formality(self, messages: List[Message]) -> float:
        """
        Analyze the formality level of messages.
        
        Returns a score from 1.0 (very casual) to 5.0 (very formal).
        """
        if not messages:
            return 3.0  # Default to neutral
        
        scores = []
        
        for message in messages:
            content = message.content
            msg_score = 3.0  # Start at neutral
            total_weight = 0.0
            
            # Apply pattern matching for each formality level
            for level, patterns in self.formality_patterns.items():
                level_score = 0.0
                level_weight = 0.0
                
                for pattern, weight in patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        level_score += matches * weight
                        level_weight += matches
                
                if level_weight > 0:
                    # Adjust the message score toward this level
                    level_value = level.value
                    adjustment = (level_score / level_weight) * 0.5  # Dampen the effect
                    msg_score = (msg_score + (level_value * adjustment)) / (1 + adjustment)
                    total_weight += level_weight
            
            # Additional heuristics
            # Check for complete sentences (formal) vs. fragments (casual)
            sentences = re.findall(r'[^.!?]+[.!?]', content)
            if sentences:
                sentence_ratio = len(sentences) / max(1, len(content.split()))
                if sentence_ratio > 0.15:  # High ratio of sentences to words
                    msg_score += 0.5  # More formal
                elif sentence_ratio < 0.05:  # Low ratio of sentences to words
                    msg_score -= 0.5  # More casual
            
            # Check capitalization (formal) vs. lowercase (casual)
            words = content.split()
            if words:
                proper_capitalized = sum(1 for w in words if w[0:1].isupper())
                if proper_capitalized / len(words) > 0.5:
                    msg_score += 0.3  # More formal
                elif proper_capitalized / len(words) < 0.1:
                    msg_score -= 0.3  # More casual
            
            # Ensure score stays in range
            msg_score = max(1.0, min(5.0, msg_score))
            scores.append(msg_score)
        
        # Return average formality score
        return statistics.mean(scores) if scores else 3.0
    
    def _analyze_detail_level(self, messages: List[Message]) -> float:
        """
        Analyze the detail level of messages.
        
        Returns a score from 1.0 (very concise) to 5.0 (very detailed).
        """
        if not messages:
            return 3.0  # Default to balanced
        
        scores = []
        
        for message in messages:
            content = message.content
            msg_score = 3.0  # Start at balanced
            total_weight = 0.0
            
            # Apply pattern matching for each detail level
            for level, patterns in self.detail_patterns.items():
                level_score = 0.0
                level_weight = 0.0
                
                for pattern, weight in patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        level_score += matches * weight
                        level_weight += matches
                
                if level_weight > 0:
                    # Adjust the message score toward this level
                    level_value = level.value
                    adjustment = (level_score / level_weight) * 0.5  # Dampen the effect
                    msg_score = (msg_score + (level_value * adjustment)) / (1 + adjustment)
                    total_weight += level_weight
            
            # Additional heuristics
            # Message length is a strong indicator of detail level
            word_count = len(content.split())
            if word_count > 150:
                msg_score += 1.0  # Very detailed
            elif word_count > 75:
                msg_score += 0.5  # Detailed
            elif word_count < 10:
                msg_score -= 1.0  # Very concise
            elif word_count < 30:
                msg_score -= 0.5  # Concise
            
            # Check for lists, bullet points, enumerations
            list_items = re.findall(r'(\n\s*[-*•⦿◦]|\n\s*\d+[.)]|\n\s*[a-z][.)])', content)
            if list_items:
                msg_score += min(1.0, len(list_items) * 0.2)  # More detailed
            
            # Ensure score stays in range
            msg_score = max(1.0, min(5.0, msg_score))
            scores.append(msg_score)
        
        # Return average detail level score
        return statistics.mean(scores) if scores else 3.0
    
    def _analyze_directness(self, messages: List[Message]) -> float:
        """
        Analyze the directness level of messages.
        
        Returns a score from 1.0 (very indirect) to 5.0 (very direct).
        """
        if not messages:
            return 3.0  # Default to balanced
        
        scores = []
        
        for message in messages:
            content = message.content
            msg_score = 3.0  # Start at balanced
            total_weight = 0.0
            
            # Apply pattern matching for each directness level
            for level, patterns in self.directness_patterns.items():
                level_score = 0.0
                level_weight = 0.0
                
                for pattern, weight in patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        level_score += matches * weight
                        level_weight += matches
                
                if level_weight > 0:
                    # Adjust the message score toward this level
                    level_value = level.value
                    adjustment = (level_score / level_weight) * 0.5  # Dampen the effect
                    msg_score = (msg_score + (level_value * adjustment)) / (1 + adjustment)
                    total_weight += level_weight
            
            # Additional heuristics
            # Check for imperative sentences (direct)
            imperatives = re.findall(r'^[A-Z][^.!?]*[.!?]', content, re.MULTILINE)
            if imperatives:
                msg_score += min(1.0, len(imperatives) * 0.3)  # More direct
            
            # Check for hedging language (indirect)
            hedges = re.findall(r'\b(kind of|sort of|a bit|somewhat|relatively|fairly|quite|rather)\b', 
                               content, re.IGNORECASE)
            if hedges:
                msg_score -= min(1.0, len(hedges) * 0.3)  # More indirect
            
            # Check for questions (indirect) vs. statements (direct)
            questions = re.findall(r'\?', content)
            statements = re.findall(r'[.!]', content)
            if questions and statements:
                q_ratio = len(questions) / (len(questions) + len(statements))
                if q_ratio > 0.7:
                    msg_score -= 0.5  # More indirect
                elif q_ratio < 0.3:
                    msg_score += 0.5  # More direct
            
            # Ensure score stays in range
            msg_score = max(1.0, min(5.0, msg_score))
            scores.append(msg_score)
        
        # Return average directness score
        return statistics.mean(scores) if scores else 3.0
    
    def _analyze_emotional_tone(self, messages: List[Message]) -> float:
        """
        Analyze the emotional tone of messages.
        
        Returns a score from 1.0 (very negative) to 5.0 (very positive).
        """
        if not messages:
            return 3.0  # Default to neutral
        
        scores = []
        
        for message in messages:
            content = message.content
            msg_score = 3.0  # Start at neutral
            total_weight = 0.0
            
            # Apply pattern matching for each tone level
            for tone, patterns in self.tone_patterns.items():
                tone_score = 0.0
                tone_weight = 0.0
                
                for pattern, weight in patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        tone_score += matches * weight
                        tone_weight += matches
                
                if tone_weight > 0:
                    # Adjust the message score toward this tone
                    tone_value = tone.value
                    adjustment = (tone_score / tone_weight) * 0.5  # Dampen the effect
                    msg_score = (msg_score + (tone_value * adjustment)) / (1 + adjustment)
                    total_weight += tone_weight
            
            # Additional heuristics
            # Check for exclamation marks (positive) vs. periods (neutral)
            exclamations = re.findall(r'!', content)
            if exclamations:
                msg_score += min(1.0, len(exclamations) * 0.2)  # More positive
            
            # Check for ALL CAPS (intensity - could be positive or negative)
            caps_words = re.findall(r'\b[A-Z]{3,}\b', content)
            if caps_words:
                # Intensify the existing direction
                if msg_score > 3.0:
                    msg_score += min(1.0, len(caps_words) * 0.3)  # More positive
                elif msg_score < 3.0:
                    msg_score -= min(1.0, len(caps_words) * 0.3)  # More negative
            
            # Ensure score stays in range
            msg_score = max(1.0, min(5.0, msg_score))
            scores.append(msg_score)
        
        # Return average emotional tone score
        return statistics.mean(scores) if scores else 3.0
    
    def _analyze_acknowledgment_preference(self, messages: List[Message]) -> bool:
        """
        Analyze whether the agent prefers acknowledgments in communication.
        
        Returns True if the agent seems to prefer acknowledgments, False otherwise.
        """
        if not messages:
            return True  # Default to preferring acknowledgments
        
        # Patterns indicating preference for acknowledgments
        ack_patterns = [
            r'\b(confirm|acknowledge|received|got it|thanks for|appreciate|confirm receipt)\b',
            r'\bplease (confirm|acknowledge|let me know|respond)\b',
            r'\b(did you|have you) (receive|get|see)\b'
        ]
        
        # Count messages containing acknowledgment patterns
        ack_count = 0
        for message in messages:
            content = message.content.lower()
            for pattern in ack_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    ack_count += 1
                    break  # Only count once per message
        
        # If more than 20% of messages include acknowledgment patterns, consider it a preference
        ack_ratio = ack_count / len(messages)
        return ack_ratio > 0.2
    
    def _analyze_structure_preference(self, messages: List[Message]) -> bool:
        """
        Analyze whether the agent prefers structured responses.
        
        Returns True if the agent seems to prefer structured content, False otherwise.
        """
        if not messages:
            return False  # Default to not preferring structured responses
        
        # Count messages with structural elements
        structure_count = 0
        for message in messages:
            content = message.content
            
            # Check for various structural elements
            has_structure = False
            
            # Check for headings
            if re.search(r'^\s*#{1,3}\s+\w+|^\s*[A-Z][A-Za-z\s]+:$', content, re.MULTILINE):
                has_structure = True
            
            # Check for bullet points or numbered lists
            elif re.search(r'^\s*[-*•] |^\s*\d+\.\s+', content, re.MULTILINE):
                has_structure = True
            
            # Check for table-like structures
            elif re.search(r'^\s*\|[^|]+\|[^|]+\|', content, re.MULTILINE):
                has_structure = True
                
            # Check for explicit sections
            elif re.search(r'^\s*(Section|Part|Step|Phase|Category)[\s:]', content, re.MULTILINE | re.IGNORECASE):
                has_structure = True
            
            if has_structure:
                structure_count += 1
        
        # If more than 30% of messages are structured, consider it a preference
        structure_ratio = structure_count / len(messages)
        return structure_ratio > 0.3
    
    def _analyze_examples_preference(self, messages: List[Message]) -> bool:
        """
        Analyze whether the agent prefers examples in communication.
        
        Returns True if the agent seems to prefer examples, False otherwise.
        """
        if not messages:
            return True  # Default to preferring examples
        
        # Patterns indicating use of examples
        example_patterns = [
            r'\b(for example|for instance|e\.g\.|such as|like|as in)\b',
            r'\bhere\'s (an example|a sample|an illustration)\b',
            r'```\w*\n[\s\S]+?\n```',  # Code blocks
            r'^\s*\d+\.\s+.+\n\s*\d+\.\s+',  # Numbered list examples
            r'"[^"]{10,}"'  # Quoted examples (at least 10 chars)
        ]
        
        # Count messages containing example patterns
        example_count = 0
        for message in messages:
            content = message.content
            for pattern in example_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    example_count += 1
                    break  # Only count once per message
        
        # If more than 25% of messages include examples, consider it a preference
        example_ratio = example_count / len(messages)
        return example_ratio > 0.25
    
    def _analyze_vocabulary_complexity(self, messages: List[Message]) -> float:
        """
        Analyze the vocabulary complexity of messages.
        
        Returns a score from 0.0 (simple) to 1.0 (complex).
        """
        if not messages:
            return 0.5  # Default to moderate complexity
        
        # Complex words (sampling of advanced/specialized vocabulary)
        complex_words = {
            'utilize', 'implementation', 'facilitate', 'comprehensive', 'subsequently',
            'nevertheless', 'furthermore', 'consequently', 'preliminary', 'approximately',
            'fundamental', 'significant', 'potential', 'efficiently', 'essentially',
            'specifically', 'appropriate', 'additional', 'particular', 'alternative',
            'sophisticated', 'ultimately', 'relatively', 'primarily', 'effectively',
            'regarding', 'sufficient', 'consistent', 'approximately', 'conjunction',
            'methodology', 'infrastructure', 'paradigm', 'conceptual', 'theoretical',
            'derivative', 'optimization', 'leverage', 'integrated', 'interface',
            'algorithm', 'subsequent', 'redundant', 'requisite', 'derivative',
            'configuration', 'implementation', 'mitigate', 'ubiquitous', 'paradigm'
        }
        
        word_count = 0
        complex_count = 0
        avg_word_lengths = []
        
        for message in messages:
            # Split into words, ignoring punctuation
            content = message.content.lower()
            words = re.findall(r'\b[a-z]+\b', content)
            
            if not words:
                continue
                
            word_count += len(words)
            
            # Count complex words
            for word in words:
                if word in complex_words or len(word) > 8:
                    complex_count += 1
            
            # Calculate average word length for this message
            avg_length = sum(len(word) for word in words) / len(words)
            avg_word_lengths.append(avg_length)
        
        if word_count == 0:
            return 0.5  # Default if no valid words found
        
        # Calculate complexity metrics
        complex_ratio = complex_count / word_count
        avg_word_length = statistics.mean(avg_word_lengths) if avg_word_lengths else 5.0
        
        # Normalize average word length (typical range 4-8)
        norm_length = (avg_word_length - 4.0) / 4.0
        norm_length = max(0.0, min(1.0, norm_length))
        
        # Combine metrics (complex words ratio and average word length)
        complexity = (complex_ratio * 0.7) + (norm_length * 0.3)
        
        # Ensure score is in range
        return max(0.0, min(1.0, complexity))
    
    def _analyze_response_speed(self, avg_response_time: float) -> ResponseSpeed:
        """
        Analyze response speed preference based on average response time.
        
        Args:
            avg_response_time: Average response time in seconds.
            
        Returns:
            ResponseSpeed enum value.
        """
        # Map response time ranges to ResponseSpeed
        if avg_response_time < 30:  # Less than 30 seconds
            return ResponseSpeed.IMMEDIATE
        elif avg_response_time < 300:  # Less than 5 minutes
            return ResponseSpeed.QUICK
        elif avg_response_time < 1800:  # Less than 30 minutes
            return ResponseSpeed.STANDARD
        elif avg_response_time < 7200:  # Less than 2 hours
            return ResponseSpeed.RELAXED
        else:  # 2+ hours
            return ResponseSpeed.EXTENDED
    
    def _calculate_consistency_score(self, messages: List[Message]) -> float:
        """
        Calculate how consistent the agent's communication style is.
        
        Returns a score from 0.0 (inconsistent) to 1.0 (very consistent).
        """
        if len(messages) < 3:
            return 1.0  # Default to high consistency with limited data
        
        # Calculate consistency across various dimensions
        
        # 1. Formality consistency
        formality_scores = []
        for message in messages:
            formality_scores.append(self._analyze_formality([message]))
            
        # 2. Detail level consistency
        detail_scores = []
        for message in messages:
            detail_scores.append(self._analyze_detail_level([message]))
            
        # 3. Directness consistency
        directness_scores = []
        for message in messages:
            directness_scores.append(self._analyze_directness([message]))
            
        # 4. Emotional tone consistency
        tone_scores = []
        for message in messages:
            tone_scores.append(self._analyze_emotional_tone([message]))
            
        # 5. Message length consistency
        message_lengths = [len(message.content.split()) for message in messages]
        
        # Calculate standard deviations (lower = more consistent)
        try:
            formality_std = statistics.stdev(formality_scores)
            detail_std = statistics.stdev(detail_scores)
            directness_std = statistics.stdev(directness_scores)
            tone_std = statistics.stdev(tone_scores)
            length_std = statistics.stdev(message_lengths) / max(1, statistics.mean(message_lengths))
        except statistics.StatisticsError:
            # Not enough data points
            return 1.0
            
        # Normalize standard deviations to consistency scores (0-1)
        # For these style metrics, a stdev of 0 means perfect consistency (score 1.0)
        # and a stdev of 2+ means high inconsistency (score approaching 0.0)
        formality_consistency = max(0.0, 1.0 - (formality_std / 2.0))
        detail_consistency = max(0.0, 1.0 - (detail_std / 2.0))
        directness_consistency = max(0.0, 1.0 - (directness_std / 2.0))
        tone_consistency = max(0.0, 1.0 - (tone_std / 2.0))
        length_consistency = max(0.0, 1.0 - min(1.0, length_std))
        
        # Combine consistency scores with appropriate weights
        weighted_consistency = (
            formality_consistency * 0.2 +
            detail_consistency * 0.3 +
            directness_consistency * 0.2 +
            tone_consistency * 0.2 +
            length_consistency * 0.1
        )
        
        return weighted_consistency
    
    def _calculate_confidence_level(self, messages: List[Message]) -> float:
        """
        Calculate confidence level in the style analysis.
        
        Returns a score from 0.0 (low confidence) to 1.0 (high confidence).
        """
        # Base confidence on amount of data
        message_count = len(messages)
        
        if message_count == 0:
            return 0.0
        
        # More messages = higher confidence, with diminishing returns
        # 10+ messages is considered very good data
        message_confidence = min(1.0, message_count / 10.0)
        
        # More text content = higher confidence
        total_content_length = sum(len(message.content) for message in messages)
        content_confidence = min(1.0, total_content_length / 5000.0)  # 5000+ chars is very good
        
        # Multiple messages over time indicate more reliable patterns
        if message_count >= 3:
            time_span_confidence = 0.8
        elif message_count >= 2:
            time_span_confidence = 0.5
        else:
            time_span_confidence = 0.3
        
        # Combine confidence factors
        confidence = (
            message_confidence * 0.5 +
            content_confidence * 0.3 +
            time_span_confidence * 0.2
        )
        
        return confidence
    
    def _generate_style_notes(self, messages: List[Message]) -> List[str]:
        """
        Generate noteworthy observations about the agent's communication style.
        
        Returns a list of style observations.
        """
        if not messages:
            return []
            
        notes = []
        
        # Check for frequent greetings
        greeting_patterns = [r'^\s*hi\b', r'^\s*hello\b', r'^\s*hey\b', r'^\s*greetings\b']
        greeting_count = sum(1 for m in messages if any(re.search(p, m.content, re.IGNORECASE) for p in greeting_patterns))
        if greeting_count / len(messages) > 0.5:
            notes.append("Often begins messages with greetings")
        
        # Check for sign-offs
        signoff_patterns = [r'regards\b', r'sincerely\b', r'best\b', r'cheers\b', r'thanks\b']
        signoff_count = sum(1 for m in messages if any(re.search(p, m.content, re.IGNORECASE) for p in signoff_patterns))
        if signoff_count / len(messages) > 0.5:
            notes.append("Frequently uses sign-offs at the end of messages")
        
        # Check for emoji usage
        emoji_pattern = r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]'
        emoji_count = sum(len(re.findall(emoji_pattern, m.content)) for m in messages)
        if emoji_count > 0:
            emoji_per_message = emoji_count / len(messages)
            if emoji_per_message > 3:
                notes.append("Uses emojis very frequently")
            elif emoji_per_message > 1:
                notes.append("Regularly includes emojis")
        
        # Check for question asking behavior
        question_counts = [len(re.findall(r'\?', m.content)) for m in messages]
        avg_questions = statistics.mean(question_counts) if question_counts else 0
        if avg_questions > 2:
            notes.append("Tends to ask multiple questions")
        
        # Check for code snippets
        code_blocks = sum(1 for m in messages if re.search(r'```[\s\S]+?```', m.content))
        if code_blocks > 0 and code_blocks / len(messages) > 0.2:
            notes.append("Often includes code examples")
        
        # Check for technical vocabulary
        tech_terms = ['api', 'function', 'method', 'algorithm', 'data', 'server', 'client', 
                     'interface', 'system', 'protocol', 'database', 'query', 'code', 'module',
                     'variable', 'parameter', 'request', 'response', 'endpoint', 'implementation']
        tech_count = 0
        for m in messages:
            content_lower = m.content.lower()
            tech_count += sum(1 for term in tech_terms if re.search(r'\b' + term + r'\b', content_lower))
        
        if tech_count > 0 and tech_count / len(messages) > 3:
            notes.append("Uses technical terminology frequently")
        
        return notes
    
    def _map_score_to_enum(self, score: float, enum_class) -> Any:
        """
        Map a numeric score to the corresponding enum value.
        
        Args:
            score: Numeric score (usually 1.0-5.0)
            enum_class: The Enum class to map to
            
        Returns:
            Corresponding enum value
        """
        # Get all enum values sorted by value
        enum_values = sorted(enum_class, key=lambda x: x.value)
        
        # For 1.0-5.0 scale mapped to 5 enum values
        if score <= 1.5:
            return enum_values[0]  # Lowest level
        elif score <= 2.5:
            return enum_values[1]  # Low level
        elif score <= 3.5:
            return enum_values[2]  # Medium level
        elif score <= 4.5:
            return enum_values[3]  # High level
        else:
            return enum_values[4]  # Highest level
    
    def adapt_message_to_style(self, original_message: str, target_style: CommunicationStyle) -> str:
        """
        Adapt a message to match a target communication style.
        
        Args:
            original_message: The original message content.
            target_style: The target CommunicationStyle to adapt to.
            
        Returns:
            Adapted message content.
        """
        # Get style adaptation guidance
        guidance = target_style.get_adaptation_guidance()
        
        # This is a placeholder for actual adaptation logic
        # In a real implementation, this would use NLP techniques to
        # transform the original message to match the target style
        
        # Instead, we'll add style indicators based on the guidance
        # This is a simplified approach to demonstrate the concept
        
        adapted_message = original_message
        
        # Add appropriate greeting based on formality
        formality = target_style.formality
        if formality in (FormalityLevel.VERY_FORMAL, FormalityLevel.FORMAL):
            if not any(adapted_message.startswith(greeting) for greeting in ["Dear ", "Hello ", "Greetings,"]):
                adapted_message = f"Hello,\n\n{adapted_message}"
        elif formality in (FormalityLevel.CASUAL, FormalityLevel.VERY_CASUAL):
            if not any(adapted_message.startswith(greeting) for greeting in ["Hi", "Hey", "Hello"]):
                adapted_message = f"Hi,\n\n{adapted_message}"
        
        # Add appropriate closing based on formality
        if formality in (FormalityLevel.VERY_FORMAL, FormalityLevel.FORMAL):
            if not any(closing in adapted_message.split("\n")[-1] for closing in ["Sincerely", "Regards", "Best regards"]):
                adapted_message = f"{adapted_message}\n\nBest regards,"
        elif formality in (FormalityLevel.CASUAL, FormalityLevel.VERY_CASUAL):
            if not any(closing in adapted_message.split("\n")[-1] for closing in ["Thanks", "Cheers", "Take care"]):
                adapted_message = f"{adapted_message}\n\nThanks!"
        
        # Apply detail level preferences
        detail_level = target_style.detail_level
        
        # Apply directness preferences
        directness = target_style.directness
        
        # Apply tone preferences
        tone = target_style.emotional_tone
        
        # Apply structural preferences if needed
        if target_style.prefers_structured_responses and "\n\n" in adapted_message:
            # Add structure to multi-paragraph messages
            paragraphs = adapted_message.split("\n\n")
            if len(paragraphs) > 2:  # Enough paragraphs for structure
                structured = []
                for i, para in enumerate(paragraphs):
                    if i == 0:  # First paragraph (often intro)
                        structured.append(para)
                    else:
                        # Add a heading if there isn't one
                        if not re.match(r'^#+ |^[A-Z][A-Za-z\s]+:', para):
                            heading = f"Point {i}:"
                            structured.append(f"{heading}\n{para}")
                        else:
                            structured.append(para)
                
                adapted_message = "\n\n".join(structured)
        
        # Apply principle alignment if we have a principle engine
        if self.principle_engine is not None:
            # This would be implemented to ensure the adaptation respects core principles
            pass
        
        return adapted_message
    
    def get_style_compatibility(self, style1: CommunicationStyle, style2: CommunicationStyle) -> Dict[str, Any]:
        """
        Analyze compatibility between two communication styles.
        
        Args:
            style1: First communication style.
            style2: Second communication style.
            
        Returns:
            Dictionary with compatibility analysis.
        """
        is_compatible, incompatibilities = style1.is_compatible_with(style2)
        alignment_strategy = style1.get_alignment_strategy(style2)
        
        compatibility_result = {
            "is_compatible": is_compatible,
            "compatibility_score": 1.0 - (len(incompatibilities) / 8.0),  # 8 possible incompatibilities
            "incompatibilities": incompatibilities,
            "alignment_strategy": alignment_strategy
        }
        
        return compatibility_result
    
    def create_style_from_examples(self, examples: List[str], agent_id: str = "derived-style") -> CommunicationStyle:
        """
        Create a communication style based on example messages.
        
        Args:
            examples: List of example message strings.
            agent_id: Identifier for the agent.
            
        Returns:
            A CommunicationStyle object derived from the examples.
        """
        # Convert examples to Message objects
        messages = []
        for example in examples:
            msg = Message(
                content=example,
                timestamp=datetime.now(timezone.utc).isoformat(),
                direction=MessageDirection.RECEIVED
            )
            messages.append(msg)
        
        # Create temporary history
        history = MessageHistory(agent_id=agent_id)
        for msg in messages:
            history.add_message(msg)
        
        # Analyze the history to derive a style
        style = self.analyze_message_history(history)
        return style
    
    def analyze_message(self, message: Union[str, Dict[str, Any], Message], agent_id: str = "unknown") -> Dict[str, Any]:
        """
        Analyze a single message to extract style characteristics.
        
        Args:
            message: The message to analyze (string, dict, or Message object)
            agent_id: The agent ID associated with the message
            
        Returns:
            Dictionary containing analysis results including formality, detail level, directness, and tone
        """
        # Convert to Message object if needed
        if isinstance(message, str):
            msg = Message(
                content=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                direction=MessageDirection.RECEIVED
            )
        elif isinstance(message, dict):
            msg = Message.from_dict(message)
        else:
            msg = message
        
        # Analyze individual aspects
        formality_score = self._analyze_formality([msg])
        detail_score = self._analyze_detail_level([msg])
        directness_score = self._analyze_directness([msg])
        tone_score = self._analyze_emotional_tone([msg])
        
        # Map scores to enum values
        formality = self._map_score_to_enum(formality_score, FormalityLevel)
        detail_level = self._map_score_to_enum(detail_score, DetailLevel)
        directness = self._map_score_to_enum(directness_score, DirectnessLevel)
        emotional_tone = self._map_score_to_enum(tone_score, EmotionalTone)
        
        # Additional analysis
        word_count = len(msg.content.split())
        has_greeting = bool(re.search(r'^\s*(hi|hello|hey|greetings|dear)\b', msg.content, re.IGNORECASE))
        has_signoff = bool(re.search(r'\b(regards|sincerely|best|cheers|thanks)\s*[,!]?\s*$', msg.content, re.IGNORECASE))
        has_questions = bool(re.search(r'\?', msg.content))
        
        # Return analysis results
        return {
            "agent_id": agent_id,
            "timestamp": msg.timestamp,
            "formality": formality.name,
            "formality_score": formality_score,
            "detail_level": detail_level.name,
            "detail_score": detail_score,
            "directness": directness.name,
            "directness_score": directness_score,
            "emotional_tone": emotional_tone.name,
            "tone_score": tone_score,
            "word_count": word_count,
            "has_greeting": has_greeting,
            "has_signoff": has_signoff,
            "has_questions": has_questions,
            "vocabulary_complexity": self._analyze_vocabulary_complexity([msg])
        }


# Example usage
if __name__ == "__main__":
    # Create a style analyzer (optionally with a principle engine)
    analyzer = CommunicationStyleAnalyzer()
    
    # Create a message history for an example agent
    history = MessageHistory(agent_id="agent-formal")
    
    # Add some received messages with a formal style
    formal_messages = [
        "Dear Sir/Madam, I am writing to inquire about the status of our previous correspondence. Would you be so kind as to provide an update at your earliest convenience? Thank you for your attention to this matter. Sincerely, Agent Formal",
        "Hello, I have reviewed the documentation you provided and have several questions regarding the implementation details. Specifically, I would like clarification on sections 3.2 and 4.1. I appreciate your assistance in this matter. Regards, Agent Formal",
        "Good day, Please find attached the requested information as per our previous discussion. I would appreciate your confirmation of receipt. Should you require any additional details, please do not hesitate to contact me. Best regards, Agent Formal"
    ]
    
    for content in formal_messages:
        msg = Message(
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=MessageDirection.RECEIVED
        )
        history.add_message(msg)
    
    # Analyze the formal agent's style
    formal_style = analyzer.analyze_message_history(history)
    print("Formal Agent Style:")
    print(json.dumps(formal_style.to_dict(), indent=2))
    print("\nStyle Adaptation Guidance:")
    print(json.dumps(formal_style.get_adaptation_guidance(), indent=2))
    
    # Create another history for a casual agent
    casual_history = MessageHistory(agent_id="agent-casual")
    
    # Add some received messages with a casual style
    casual_messages = [
        "Hey! Just checking in on that thing we talked about yesterday. Any updates? Thanks!",
        "Yo, got your message! Sounds good to me... Let's go with option 2, it seems way easier tbh. Hit me up if you need anything else!",
        "Haha cool! 😁 I'm all for it! Let me know when you want to get started and I'll make time. Btw did you see that new feature they launched? Pretty awesome stuff!"
    ]
    
    for content in casual_messages:
        msg = Message(
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=MessageDirection.RECEIVED
        )
        casual_history.add_message(msg)
    
    # Analyze the casual agent's style
    casual_style = analyzer.analyze_message_history(casual_history)
    print("\nCasual Agent Style:")
    print(json.dumps(casual_style.to_dict(), indent=2))
    
    # Check compatibility between the styles
    compatibility = analyzer.get_style_compatibility(formal_style, casual_style)
    print("\nStyle Compatibility:")
    print(json.dumps(compatibility, indent=2))
    
    # Adapt a message from formal to casual style
    formal_message = "I am writing to inform you that the project deadline has been extended to next Friday. Please adjust your schedule accordingly."
    adapted_message = analyzer.adapt_message_to_style(formal_message, casual_style)
    print("\nAdapted Message:")
    print(f"Original: {formal_message}")
    print(f"Adapted:  {adapted_message}")