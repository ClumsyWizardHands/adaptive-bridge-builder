#!/usr/bin/env python3
"""
A2A Task Handler

This module implements a comprehensive A2A task handler that processes
incoming tasks from other agents using the Adaptive Bridge Builder framework.
It extracts message content and intent, evaluates responses based on principles,
adapts communication styles, and maintains conversation context.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import traceback
import os

from principle_engine import PrincipleEngine
from communication_style_analyzer import CommunicationStyleAnalyzer
from communication_style import CommunicationStyle
from relationship_tracker import RelationshipTracker, InteractionType, InteractionQuality
from conflict_resolver import ConflictResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("A2ATaskHandler")

class MessageIntent:
    """Enumeration of possible message intents."""
    QUERY = "query"                      # Asking for information
    INSTRUCT = "instruct"                # Giving instructions or commands
    INFORM = "inform"                    # Providing information
    REQUEST = "request"                  # Requesting an action
    CLARIFY = "clarify"                  # Seeking or providing clarification
    CONFIRM = "confirm"                  # Confirming something
    DENY = "deny"                        # Denying or rejecting
    ACKNOWLEDGE = "acknowledge"          # Acknowledging receipt
    SUGGEST = "suggest"                  # Making suggestions
    EXPRESS = "express"                  # Expressing feelings or opinions
    UNKNOWN = "unknown"                  # Could not determine intent

class ContentType:
    """Enumeration of possible content types."""
    TEXT = "text"                        # Plain text
    JSON = "json"                        # JSON data
    IMAGE = "image"                      # Image data
    AUDIO = "audio"                      # Audio data
    VIDEO = "video"                      # Video data
    FILE = "file"                        # File data
    COMPOSITE = "composite"              # Multiple types combined
    UNKNOWN = "unknown"                  # Unknown content type

class TaskPriority:
    """Enumeration of task priorities."""
    CRITICAL = "critical"                # Highest priority, requires immediate attention
    HIGH = "high"                        # High priority
    MEDIUM = "medium"                    # Medium priority (default)
    LOW = "low"                          # Low priority
    BACKGROUND = "background"            # Lowest priority, processed when resources available

class TaskStatus:
    """Enumeration of task statuses."""
    RECEIVED = "received"                # Task has been received
    PROCESSING = "processing"            # Task is being processed
    COMPLETED = "completed"              # Task has been completed successfully
    FAILED = "failed"                    # Task processing failed
    PENDING = "pending"                  # Task is pending additional information
    DELEGATED = "delegated"              # Task has been delegated to another agent
    REJECTED = "rejected"                # Task has been rejected

class MessageContext:
    """Maintains context for a conversation."""
    
    def __init__(
        self,
        conversation_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the message context.
        
        Args:
            conversation_id: ID of the conversation.
            agent_id: ID of the agent in conversation with.
            metadata: Additional metadata about the conversation.
        """
        self.conversation_id = conversation_id
        self.agent_id = agent_id
        self.metadata = metadata or {}
        self.messages: List[Dict[str, Any]] = []
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.topics: List[str] = []
        self.intent_history: List[str] = []
        self.status = "active"
        self.custom_data: Dict[str, Any] = {}
    
    def add_message(self, message: Dict[str, Any], intent: str) -> None:
        """
        Add a message to the context.
        
        Args:
            message: The message to add.
            intent: The intent of the message.
        """
        self.messages = [*self.messages, message]
        self.intent_history = [*self.intent_history, intent]
        self.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Update topics if available in message
        if "topics" in message:
            for topic in message["topics"]:
                if topic not in self.topics:
                    self.topics = [*self.topics, topic]
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent messages.
        
        Args:
            count: Number of messages to retrieve.
            
        Returns:
            List of recent messages.
        """
        return self.messages[-count:] if self.messages else []
    
    def get_common_intent(self) -> str:
        """
        Get the most common intent in this conversation.
        
        Returns:
            The most common intent.
        """
        if not self.intent_history:
            return MessageIntent.UNKNOWN
            
        intent_counts = {}
        for intent in self.intent_history:
            if intent not in intent_counts:
                intent_counts[intent] = 0
            intent_counts[intent] += 1
            
        return max(intent_counts.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message context to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "messages": self.messages,
            "last_updated": self.last_updated,
            "topics": self.topics,
            "intent_history": self.intent_history,
            "status": self.status,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageContext':
        """Create a MessageContext from a dictionary."""
        context = cls(
            conversation_id=data.get("conversation_id", ""),
            agent_id=data.get("agent_id", ""),
            metadata=data.get("metadata", {})
        )
        context.messages = data.get("messages", [])
        context.last_updated = data.get("last_updated", datetime.now(timezone.utc).isoformat())
        context.topics = data.get("topics", [])
        context.intent_history = data.get("intent_history", [])
        context.status = data.get("status", "active")
        context.custom_data = data.get("custom_data", {})
        return context
    
    def is_new_conversation(self) -> bool:
        """Check if this is a new conversation (no messages yet)."""
        return len(self.messages) == 0

class A2ATaskHandler:
    """
    Handles A2A tasks from other agents.
    
    This class processes incoming tasks, extracts message content and intent,
    evaluates responses using the PrincipleEngine, adapts communication using
    the CommunicationStyleAnalyzer, and maintains context across exchanges.
    """
    
    def __init__(
        self,
        agent_id: str,
        principle_engine: Optional[PrincipleEngine] = None,
        communication_analyzer: Optional[CommunicationStyleAnalyzer] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the A2A Task Handler.
        
        Args:
            agent_id: ID of this agent.
            principle_engine: Engine for evaluating principles.
            communication_analyzer: Analyzer for communication styles.
            relationship_tracker: Tracker for agent relationships.
            conflict_resolver: Resolver for conflicts in communication.
            data_dir: Directory for storing data.
        """
        self.agent_id = agent_id
        self.principle_engine = principle_engine
        self.communication_analyzer = communication_analyzer
        self.relationship_tracker = relationship_tracker
        self.conflict_resolver = conflict_resolver
        self.data_dir = data_dir or "data/a2a_tasks"
        
        # Storage for conversations
        self.active_contexts: Dict[str, MessageContext] = {}
        
        # Intent recognition patterns
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Task metadata
        self.tasks: Dict[str, Dict[str, Any]] = {}
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"A2ATaskHandler initialized for agent {agent_id}")
    
    def _initialize_intent_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Initialize patterns for recognizing message intents.
        
        Returns:
            Dictionary of intent patterns.
        """
        patterns = {}
        
        # Query intent patterns
        patterns[MessageIntent.QUERY] = [
            re.compile(r'(?i)(what|who|when|where|why|how|can you|could you tell|do you know|please tell).*\?'),
            re.compile(r'(?i)^(tell me|explain|describe|elaborate|clarify).*'),
            re.compile(r'(?i)(looking for|searching for|need information|trying to find).*')
        ]
        
        # Instruct intent patterns
        patterns[MessageIntent.INSTRUCT] = [
            re.compile(r'(?i)^(please|kindly)?\s*(do|make|create|generate|implement|execute|run|perform|configure).*'),
            re.compile(r'(?i)^(you should|you must|you need to|i need you to).*'),
            re.compile(r'(?i)^(ensure|make sure|verify|check).*')
        ]
        
        # Inform intent patterns
        patterns[MessageIntent.INFORM] = [
            re.compile(r'(?i)(i am|we are|they are|it is|just|fyi|for your information).*'),
            re.compile(r'(?i)(wanted to (let|inform|tell) you|thought you should know|heads up).*'),
            re.compile(r'(?i)(update|status|progress report|news|announcement).*')
        ]
        
        # Request intent patterns
        patterns[MessageIntent.REQUEST] = [
            re.compile(r'(?i)(would you|could you|can you|will you|please|kindly).*'),
            re.compile(r'(?i)(i (would|would like to|want to|need to) request).*'),
            re.compile(r'(?i)(requesting|asking for|would it be possible).*')
        ]
        
        # Clarify intent patterns
        patterns[MessageIntent.CLARIFY] = [
            re.compile(r'(?i)(to clarify|just to be clear|to be specific|what i meant|in other words).*'),
            re.compile(r'(?i)(did you mean|are you saying|are you referring to|when you say).*'),
            re.compile(r'(?i)(confused about|unclear on|not sure if|need clarification).*')
        ]
        
        # Confirm intent patterns
        patterns[MessageIntent.CONFIRM] = [
            re.compile(r'(?i)(yes|correct|right|exactly|precisely|indeed|absolutely|definitely).*'),
            re.compile(r'(?i)(confirm|confirmed|i agree|we agree|that is correct|that\'s right).*'),
            re.compile(r'(?i)(sounds good|looks good|works for me|that will work|perfect|great).*')
        ]
        
        # Deny intent patterns
        patterns[MessageIntent.DENY] = [
            re.compile(r'(?i)(no|nope|not|incorrect|wrong|false|mistaken|untrue).*'),
            re.compile(r'(?i)(i disagree|we disagree|that is not|that\'s not|i don\'t think so).*'),
            re.compile(r'(?i)(cannot|can\'t|won\'t|will not|unable to|not possible).*')
        ]
        
        # Acknowledge intent patterns
        patterns[MessageIntent.ACKNOWLEDGE] = [
            re.compile(r'(?i)(got it|understood|i see|noted|acknowledged|received|thanks|thank you).*'),
            re.compile(r'(?i)(okay|ok|alright|very well|will do|on it|working on it).*'),
            re.compile(r'(?i)(message received|confirming receipt|read and understood).*')
        ]
        
        # Suggest intent patterns
        patterns[MessageIntent.SUGGEST] = [
            re.compile(r'(?i)(suggest|suggestion|recommend|recommendation|propose|proposal).*'),
            re.compile(r'(?i)(what if|perhaps|maybe|how about|have you considered|consider).*'),
            re.compile(r'(?i)(one option|another approach|alternative|instead of|rather than).*')
        ]
        
        # Express intent patterns
        patterns[MessageIntent.EXPRESS] = [
            re.compile(r'(?i)(i feel|we feel|i think|we think|in my opinion|from my perspective).*'),
            re.compile(r'(?i)(happy|sad|frustrated|concerned|worried|excited|pleased|disappointed).*'),
            re.compile(r'(?i)(love|hate|like|dislike|prefer|favor|appreciate|value|enjoy).*')
        ]
        
        return patterns
    
    def handle_task(
        self,
        message: Dict[str, Any],
        agent_id: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for handling an incoming A2A task.
        
        Args:
            message: The incoming message to process.
            agent_id: ID of the agent sending the message.
            conversation_id: Optional ID for the conversation context.
            metadata: Additional metadata about the task.
            
        Returns:
            Response message with appropriate formatting.
        """
        # Generate a task ID
        task_id = message.get("id") or f"task-{uuid.uuid4().hex}"
        
        # Set up task tracking
        self.tasks = {**self.tasks, task_id: {
            "status": TaskStatus.RECEIVED,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "metadata": metadata or {}
        }}
        
        # Create standardized message structure if needed
        if "jsonrpc" not in message:
            message = self._normalize_message(message, task_id)
        
        try:
            # Update task status
            self.tasks[task_id]["status"] = TaskStatus.PROCESSING
            
            # Get or create conversation context
            context = self._get_or_create_context(agent_id, conversation_id, metadata)
            
            # Detect conflicts if conflict resolver is available
            if self.conflict_resolver:
                conflict_indicators = self.conflict_resolver.detect_conflicts(
                    message=message,
                    agent_id=agent_id,
                    conversation_id=context.conversation_id
                )
                
                # If conflicts detected, create a conflict record
                if conflict_indicators:
                    conflict_record = self.conflict_resolver.create_conflict_record(
                        indicators=conflict_indicators,
                        agent_id=agent_id,
                        message=message,
                        conversation_id=context.conversation_id
                    )
                    
                    # If severe conflict, add to task metadata
                    if conflict_record and conflict_record.severity.value in ["high", "critical"]:
                        self.tasks[task_id]["conflict_detected"] = True
                        self.tasks[task_id]["conflict_id"] = conflict_record.conflict_id
                        self.tasks[task_id]["conflict_severity"] = conflict_record.severity.value
            
            # Extract content and determine content type
            content, content_type = self._extract_content(message)
            self.tasks[task_id]["content_type"] = content_type
            
            # Determine message intent
            intent = self._determine_intent(content)
            self.tasks[task_id]["intent"] = intent
            
            # Update the conversation context with this message
            context.add_message(message, intent)
            
            # Evaluate response using principle engine if available
            principle_evaluation = None
            if self.principle_engine:
                principle_evaluation = self.principle_engine.evaluate_message(
                    content=content,
                    agent_id=agent_id,
                    context=context.to_dict()
                )
                self.tasks[task_id]["principle_score"] = principle_evaluation.get("score", 0.0)
            
            # Analyze communication style if analyzer available
            communication_style = None
            if self.communication_analyzer and isinstance(content, str):
                communication_style = self.communication_analyzer.analyze_text(content)
                self.tasks[task_id]["communication_style"] = communication_style.to_dict() if communication_style else None
            
            # Record interaction if relationship tracker available
            if self.relationship_tracker:
                interaction_quality = InteractionQuality.NEUTRAL
                
                # Determine interaction quality based on principle evaluation
                if principle_evaluation:
                    score = principle_evaluation.get("score", 0.5)
                    if score >= 0.7:
                        interaction_quality = InteractionQuality.POSITIVE
                    elif score <= 0.3:
                        interaction_quality = InteractionQuality.NEGATIVE
                
                self.relationship_tracker.record_interaction(
                    agent_id=agent_id,
                    interaction_type=InteractionType.MESSAGE,
                    content_summary=self._summarize_content(content),
                    quality=interaction_quality,
                    principle_alignment=principle_evaluation.get("score", 0.5) if principle_evaluation else 0.5,
                    metadata={
                        "task_id": task_id,
                        "conversation_id": context.conversation_id,
                        "intent": intent,
                        "content_type": content_type
                    }
                )
            
            # Process the task based on content and intent
            response = self._process_task(
                task_id=task_id,
                content=content,
                content_type=content_type,
                intent=intent,
                agent_id=agent_id,
                context=context,
                principle_evaluation=principle_evaluation,
                communication_style=communication_style
            )
            
            # Update task status
            self.tasks[task_id]["status"] = TaskStatus.COMPLETED
            self.tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            # Log successful completion
            logger.info(f"Task {task_id} from {agent_id} completed successfully. "
                      f"Intent: {intent}, Content type: {content_type}")
            
            return response
            
        except Exception as e:
            # Handle exceptions
            error_message = str(e)
            stack_trace = traceback.format_exc()
            
            # Update task status
            self.tasks[task_id]["status"] = TaskStatus.FAILED
            self.tasks[task_id]["error"] = error_message
            self.tasks[task_id]["stack_trace"] = stack_trace
            
            # Log the error
            logger.error(f"Error processing task {task_id} from {agent_id}: {error_message}")
            logger.debug(f"Stack trace: {stack_trace}")
            
            # Return error response
            return self._create_error_response(
                task_id=task_id,
                error_message=error_message,
                error_code=-32603  # Internal error
            )
    
    def _normalize_message(
        self,
        message: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """
        Normalize a message to JSON-RPC 2.0 format.
        
        Args:
            message: The message to normalize.
            task_id: ID for the task.
            
        Returns:
            Normalized message.
        """
        # Determine method
        method = message.get("method", "process")
        if "action" in message:
            method = message["action"]
        elif "command" in message:
            method = message["command"]
            
        # Extract params
        params = {}
        for key, value in message.items():
            if key not in ["id", "jsonrpc", "method", "action", "command"]:
                params[key] = value
                
        # Create normalized message
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": task_id
        }
    
    def _get_or_create_context(
        self,
        agent_id: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageContext:
        """
        Get or create a conversation context.
        
        Args:
            agent_id: ID of the agent in conversation with.
            conversation_id: Optional ID for the conversation.
            metadata: Additional metadata about the conversation.
            
        Returns:
            MessageContext object.
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv-{uuid.uuid4().hex}"
            
        # Check if context exists
        context_key = f"{agent_id}:{conversation_id}"
        if context_key in self.active_contexts:
            return self.active_contexts[context_key]
            
        # Create new context
        context = MessageContext(
            conversation_id=conversation_id,
            agent_id=agent_id,
            metadata=metadata
        )
        self.active_contexts = {**self.active_contexts, context_key: context}
        
        return context
    
    def _extract_content(self, message: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Extract content from a message and determine its type.
        
        Args:
            message: The message to extract content from.
            
        Returns:
            Tuple of (content, content_type).
        """
        # Check if method is a file operation
        method = message.get("method", "").lower()
        if method in ["upload", "download", "file", "attachment"]:
            return message, ContentType.FILE
            
        # Extract from params
        params = message.get("params", {})
        
        # Check for explicit content field
        if "content" in params:
            content = params["content"]
            
            # Determine content type
            if isinstance(content, str):
                # Check if it's JSON
                try:
                    json_content = json.loads(content)
                    return json_content, ContentType.JSON
                except:
                    return content, ContentType.TEXT
            elif isinstance(content, dict) or isinstance(content, list):
                return content, ContentType.JSON
            else:
                return content, ContentType.UNKNOWN
        
        # Check for common text fields
        text_fields = ["text", "message", "query", "question", "statement"]
        for field in text_fields:
            if field in params and isinstance(params[field], str):
                return params[field], ContentType.TEXT
        
        # Check for data field which might be structured
        if "data" in params:
            data = params["data"]
            if isinstance(data, dict) or isinstance(data, list):
                return data, ContentType.JSON
            elif isinstance(data, str):
                # Check if it's JSON
                try:
                    json_data = json.loads(data)
                    return json_data, ContentType.JSON
                except:
                    return data, ContentType.TEXT
        
        # Check for multimedia content
        if "image" in params:
            return params["image"], ContentType.IMAGE
        if "audio" in params:
            return params["audio"], ContentType.AUDIO
        if "video" in params:
            return params["video"], ContentType.VIDEO
            
        # Check if multiple content types are present
        content_fields = {
            "text": ContentType.TEXT,
            "image": ContentType.IMAGE,
            "audio": ContentType.AUDIO,
            "video": ContentType.VIDEO,
            "json": ContentType.JSON,
            "file": ContentType.FILE
        }
        
        composite_content = {}
        for field, type_value in content_fields.items():
            if field in params:
                composite_content[field] = params[field]
                
        if len(composite_content) > 1:
            return composite_content, ContentType.COMPOSITE
            
        # If no specific content found, return the entire params
        return params, ContentType.UNKNOWN
    
    def _determine_intent(self, content: Any) -> str:
        """
        Determine the intent of a message.
        
        Args:
            content: The message content.
            
        Returns:
            The determined intent.
        """
        # If not a string, can't determine intent from content
        if not isinstance(content, str):
            # Try to convert JSON to string
            if isinstance(content, dict) or isinstance(content, list):
                try:
                    content = json.dumps(content)
                except:
                    return MessageIntent.UNKNOWN
            else:
                return MessageIntent.UNKNOWN
                
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    return intent
                    
        # Default to unknown
        return MessageIntent.UNKNOWN
    
    def _summarize_content(self, content: Any) -> str:
        """
        Create a summary of content for logging and tracking.
        
        Args:
            content: The content to summarize.
            
        Returns:
            A string summary of the content.
        """
        # If string, truncate if necessary
        if isinstance(content, str):
            if len(content) > 100:
                return content[:97] + "..."
            return content
            
        # If JSON, convert to string and truncate
        if isinstance(content, dict) or isinstance(content, list):
            try:
                json_str = json.dumps(content)
                if len(json_str) > 100:
                    return json_str[:97] + "..."
                return json_str
            except:
                return "Complex JSON content"
                
        # For other types
        return f"Content of type: {type(content).__name__}"
    
    def _determine_priority(
        self,
        intent: str,
        content: Any,
        agent_id: str
    ) -> str:
        """
        Determine the priority of a task based on intent and content.
        
        Args:
            intent: The intent of the message.
            content: The content of the message.
            agent_id: ID of the agent sending the message.
            
        Returns:
            Priority level.
        """
        # Check relationship with agent if available
        if self.relationship_tracker:
            relationship = self.relationship_tracker.get_relationship(agent_id)
            if relationship:
                # Critical priority for trusted agents with high trust level
                if relationship.trust_level.name in ["HIGH", "VERY_HIGH"]:
                    return TaskPriority.HIGH
                # Low priority for agents with damaged relationships
                elif relationship.trust_level.name in ["VERY_LOW", "DAMAGED"]:
                    return TaskPriority.LOW
        
        # Determine priority based on intent
        if intent in [MessageIntent.QUERY, MessageIntent.REQUEST]:
            return TaskPriority.MEDIUM
        elif intent == MessageIntent.INSTRUCT:
            return TaskPriority.HIGH
        elif intent in [MessageIntent.CLARIFY, MessageIntent.CONFIRM, MessageIntent.DENY]:
            return TaskPriority.HIGH
        elif intent == MessageIntent.INFORM:
            return TaskPriority.MEDIUM
        elif intent == MessageIntent.ACKNOWLEDGE:
            return TaskPriority.LOW
        
        # Default priority
        return TaskPriority.MEDIUM
    
    def _process_task(
        self,
        task_id: str,
        content: Any,
        content_type: str,
        intent: str,
        agent_id: str,
        context: MessageContext,
        principle_evaluation: Optional[Dict[str, Any]] = None,
        communication_style: Optional[CommunicationStyle] = None
    ) -> Dict[str, Any]:
        """
        Process a task based on its content and intent.
        
        Args:
            task_id: ID of the task.
            content: The content of the message.
            content_type: Type of the content.
            intent: Intent of the message.
            agent_id: ID of the agent sending the message.
            context: Conversation context.
            principle_evaluation: Results of principle evaluation.
            communication_style: Detected communication style.
            
        Returns:
            Response message.
        """
        # Default method and response values
        method = "response"
        result = {"status": "success", "message": "Task processed successfully"}
        
        # Determine priority based on intent and content
        priority = self._determine_priority(intent, content, agent_id)
        self.tasks[task_id]["priority"] = priority
        
        # Handle based on intent
        if intent == MessageIntent.QUERY:
            result = self._handle_query(content, content_type, agent_id, context)
            
        elif intent == MessageIntent.INSTRUCT:
            result = self._handle_instruction(content, content_type, agent_id, context)
            
        elif intent == MessageIntent.REQUEST:
            result = self._handle_request(content, content_type, agent_id, context)
            
        elif intent == MessageIntent.INFORM:
            result = self._handle_information(content, content_type, agent_id, context)
            
        elif intent == MessageIntent.CLARIFY:
            result = self._handle_clarification(content, content_type, agent_id, context)
            
        else:
            # For other intents, provide a generic response
            result = self._handle_generic(content, content_type, intent, agent_id, context)
        
        # Add task metadata to result
        result["task_id"] = task_id
        result["processed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Add context information
        result["conversation_id"] = context.conversation_id
        result["context_summary"] = {
            "message_count": len(context.messages),
            "topics": context.topics,
            "last_updated": context.last_updated
        }
        
        # Add principle information if available
        if principle_evaluation:
            result["principle_alignment"] = {
                "score": principle_evaluation.get("score", 0.0),
                "principles": principle_evaluation.get("principles", [])
            }
        
        # Adapt response based on communication style if available
        if self.communication_analyzer and communication_style and "message" in result:
            message = result["message"]
            result["message"] = self.communication_analyzer.adapt_response(message, communication_style)
        
        # Create JSON-RPC response
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": task_id
        }
        
        return response
    
    def _handle_query(
        self,
        content: Any,
        content_type: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle a query intent.
        
        Args:
            content: The query content.
            content_type: Type of the content.
            agent_id: ID of the agent making the query.
            context: Conversation context.
            
        Returns:
            Response to the query.
        """
        # Prepare response based on content type
        if content_type == ContentType.TEXT and isinstance(content, str):
            # Simple text query response
            return {
                "status": "success",
                "message": f"Query received: {content[:100]}...",
                "query_type": "text",
                "agent_id": agent_id,
                "response": "This is a placeholder response. Implement domain-specific logic here."
            }
        elif content_type == ContentType.JSON:
            # Structured query response
            return {
                "status": "success",
                "message": "Structured query processed",
                "query_type": "json",
                "agent_id": agent_id,
                "query_data": content,
                "response": "Structured response placeholder"
            }
        else:
            # Unknown content type
            return {
                "status": "partial",
                "message": f"Query with {content_type} content received",
                "query_type": content_type,
                "agent_id": agent_id,
                "note": "Content type handling not fully implemented"
            }
    
    def _handle_instruction(
        self,
        content: Any,
        content_type: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle an instruction intent.
        
        Args:
            content: The instruction content.
            content_type: Type of the content.
            agent_id: ID of the agent giving the instruction.
            context: Conversation context.
            
        Returns:
            Response to the instruction.
        """
        return {
            "status": "acknowledged",
            "message": "Instruction received and will be processed",
            "instruction_type": content_type,
            "agent_id": agent_id,
            "execution_status": "pending",
            "estimated_completion": "TBD"
        }
    
    def _handle_request(
        self,
        content: Any,
        content_type: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle a request intent.
        
        Args:
            content: The request content.
            content_type: Type of the content.
            agent_id: ID of the agent making the request.
            context: Conversation context.
            
        Returns:
            Response to the request.
        """
        return {
            "status": "accepted",
            "message": "Request has been received",
            "request_type": content_type,
            "agent_id": agent_id,
            "processing_status": "initiated",
            "request_id": f"req-{uuid.uuid4().hex[:8]}"
        }
    
    def _handle_information(
        self,
        content: Any,
        content_type: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle an information intent.
        
        Args:
            content: The information content.
            content_type: Type of the content.
            agent_id: ID of the agent providing information.
            context: Conversation context.
            
        Returns:
            Acknowledgment of the information.
        """
        return {
            "status": "received",
            "message": "Information has been recorded",
            "info_type": content_type,
            "agent_id": agent_id,
            "stored": True,
            "reference_id": f"info-{uuid.uuid4().hex[:8]}"
        }
    
    def _handle_clarification(
        self,
        content: Any,
        content_type: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle a clarification intent.
        
        Args:
            content: The clarification content.
            content_type: Type of the content.
            agent_id: ID of the agent seeking/providing clarification.
            context: Conversation context.
            
        Returns:
            Response to the clarification.
        """
        # Check if this is seeking or providing clarification
        seeking = False
        if isinstance(content, str):
            seeking_patterns = ["did you mean", "are you saying", "confused about", "unclear"]
            seeking = any(pattern in content.lower() for pattern in seeking_patterns)
        
        if seeking:
            return {
                "status": "clarifying",
                "message": "I'll provide clarification",
                "clarification_type": "response",
                "agent_id": agent_id,
                "clarification": "Here's a clearer explanation..."
            }
        else:
            return {
                "status": "understood",
                "message": "Thank you for the clarification",
                "clarification_type": "received",
                "agent_id": agent_id,
                "updated_understanding": True
            }
    
    def _handle_generic(
        self,
        content: Any,
        content_type: str,
        intent: str,
        agent_id: str,
        context: MessageContext
    ) -> Dict[str, Any]:
        """
        Handle generic intents not covered by specific handlers.
        
        Args:
            content: The message content.
            content_type: Type of the content.
            intent: The detected intent.
            agent_id: ID of the agent.
            context: Conversation context.
            
        Returns:
            Generic response.
        """
        return {
            "status": "processed",
            "message": f"{intent.capitalize()} message processed",
            "intent": intent,
            "content_type": content_type,
            "agent_id": agent_id,
            "action_taken": "logged and acknowledged"
        }
    
    def _create_error_response(
        self,
        task_id: str,
        error_message: str,
        error_code: int = -32603
    ) -> Dict[str, Any]:
        """
        Create a JSON-RPC error response.
        
        Args:
            task_id: ID of the task that failed.
            error_message: Error message to include.
            error_code: JSON-RPC error code.
            
        Returns:
            Error response in JSON-RPC format.
        """
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": error_code,
                "message": error_message,
                "data": {
                    "task_id": task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            "id": task_id
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            Task status information or None if not found.
        """
        return self.tasks.get(task_id)
    
    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """
        Get list of active conversations.
        
        Returns:
            List of active conversation summaries.
        """
        conversations = []
        for context_key, context in self.active_contexts.items():
            conversations.append({
                "conversation_id": context.conversation_id,
                "agent_id": context.agent_id,
                "message_count": len(context.messages),
                "last_updated": context.last_updated,
                "status": context.status,
                "topics": context.topics
            })
        return conversations
    
    def close_conversation(self, agent_id: str, conversation_id: str) -> bool:
        """
        Close a conversation.
        
        Args:
            agent_id: ID of the agent.
            conversation_id: ID of the conversation.
            
        Returns:
            True if conversation was closed, False if not found.
        """
        context_key = f"{agent_id}:{conversation_id}"
        if context_key in self.active_contexts:
            self.active_contexts[context_key].status = "closed"
            # Optionally save to disk before removing from memory
            # self._save_context(self.active_contexts[context_key])
            self.active_contexts = {k: v for k, v in self.active_contexts.items() if k != context_key}
            return True
        return False
