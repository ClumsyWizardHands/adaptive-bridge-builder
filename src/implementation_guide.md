# Adaptive Bridge Builder Implementation Guide

This guide provides comprehensive instructions for integrating all Adaptive Bridge Builder components into a complete agent system. Follow these steps to deploy an agent capable of adapting to different communication styles, maintaining principles, resolving conflicts, and effectively handling agent-to-agent interactions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Integration](#component-integration)
3. [Configuration Options](#configuration-options)
4. [A2A Server Implementation](#a2a-server-implementation)
5. [Empire Profile Integration](#empire-profile-integration)
6. [Deployment Scenarios](#deployment-scenarios)
7. [Testing and Validation](#testing-and-validation)

## System Architecture

The Adaptive Bridge Builder consists of several key components working together:

```
┌───────────────────────────────────────────────────────────────┐
│                 Adaptive Bridge Builder Agent                  │
├───────────────┬───────────────┬───────────────┬───────────────┤
│  Agent Core   │ Communication │  Relationship │   Knowledge   │
│               │    Layer      │  Management   │   Management  │
├───────────────┼───────────────┼───────────────┼───────────────┤
│ PrincipleEngine│CommunicationStyle│RelationshipTracker│ContentHandler│
│               │   Analyzer    │               │               │
│ AgentCard     │Communication  │ConflictResolver│FileExchange  │
│               │   Adapter     │               │   Handler     │
│ SessionManager│A2ATaskHandler │               │               │
└───────────────┴───────────────┴───────────────┴───────────────┘
           │                │                │
           ▼                ▼                ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ External Agent │  │ External Agent │  │ External Agent │
│ Communications │  │ Communications │  │ Communications │
└───────────────┘  └───────────────┘  └───────────────┘
```

### Core Components

1. **Principle Engine**: Maintains and enforces core principles and values
2. **Communication Style Analyzer**: Identifies and adapts to different communication styles
3. **Relationship Tracker**: Maintains relationship status between agents
4. **Conflict Resolver**: Detects and resolves communication conflicts
5. **Agent Card**: Manages identity and capability information
6. **Session Manager**: Handles conversation contexts and state
7. **Content Handler**: Processes different content formats
8. **A2A Task Handler**: Coordinates tasks between agents
9. **Communication Adapter**: Adapts communications to different agent capabilities
10. **File Exchange Handler**: Manages file sharing between agents

## Component Integration

The main integration point is through the `AdaptiveBridgeBuilder` class, which coordinates all components. Here's how to implement the complete agent:

```python
#!/usr/bin/env python3
"""
Adaptive Bridge Builder - Main Integration Module

This module integrates all components of the Adaptive Bridge Builder
into a complete agent system.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union

from principle_engine import PrincipleEngine
from communication_style import CommunicationStyle
from communication_style_analyzer import CommunicationStyleAnalyzer
from relationship_tracker import RelationshipTracker
from conflict_resolver import ConflictResolver
from agent_card import AgentCard
from session_manager import SessionManager
from content_handler import ContentHandler
from a2a_task_handler import A2ATaskHandler
from file_exchange_handler import FileExchangeHandler
from communication_adapter import CommunicationAdapter
from collaborative_task_handler import CollaborativeTaskHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AdaptiveBridgeBuilder")


class AdaptiveBridgeBuilder:
    """
    Main integration class for the Adaptive Bridge Builder.
    
    This class coordinates all components to create a complete
    agent capable of adaptive communication, relationship management,
    principle adherence, and conflict resolution.
    """
    
    def __init__(
        self,
        agent_id: str,
        config_path: Optional[str] = None,
        principles_path: Optional[str] = None,
        agent_card_path: Optional[str] = None
    ):
        """
        Initialize the Adaptive Bridge Builder.
        
        Args:
            agent_id: Unique identifier for this agent
            config_path: Path to configuration file (optional)
            principles_path: Path to principles definition file (optional)
            agent_card_path: Path to agent card definition file (optional)
        """
        self.agent_id = agent_id
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.principle_engine = self._init_principle_engine(principles_path)
        self.agent_card = self._init_agent_card(agent_card_path)
        self.communication_style_analyzer = CommunicationStyleAnalyzer()
        self.relationship_tracker = RelationshipTracker(agent_id=agent_id)
        self.conflict_resolver = ConflictResolver(
            principle_engine=self.principle_engine,
            relationship_tracker=self.relationship_tracker
        )
        self.session_manager = SessionManager()
        self.content_handler = ContentHandler()
        self.communication_adapter = CommunicationAdapter()
        self.a2a_task_handler = A2ATaskHandler(agent_id=agent_id)
        self.file_exchange_handler = FileExchangeHandler(
            agent_id=agent_id,
            storage_dir=self.config.get("file_storage_dir", "file_storage")
        )
        self.collaborative_task_handler = CollaborativeTaskHandler(
            agent_id=agent_id,
            a2a_task_handler=self.a2a_task_handler
        )
        
        # Keep track of known agents
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Adaptive Bridge Builder initialized with ID: {agent_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "log_level": "INFO",
            "file_storage_dir": "file_storage",
            "max_session_history": 100,
            "default_communication_style": {
                "formality": "NEUTRAL",
                "detail_level": "BALANCED",
                "directness": "BALANCED",
                "emotional_tone": "NEUTRAL",
                "response_speed": "STANDARD"
            }
        }
    
    def _init_principle_engine(self, principles_path: Optional[str]) -> PrincipleEngine:
        """Initialize the principle engine with defined principles."""
        principles = []
        
        if principles_path and os.path.exists(principles_path):
            with open(principles_path, 'r') as f:
                principles = json.load(f)
        
        # Default principles if none provided
        if not principles:
            principles = [
                {
                    "name": "User Privacy",
                    "description": "Respect user privacy and maintain confidentiality of information",
                    "weight": 1.0
                },
                {
                    "name": "Accuracy",
                    "description": "Provide accurate and truthful information",
                    "weight": 0.9
                },
                {
                    "name": "Adaptability",
                    "description": "Adapt to user needs while maintaining core values",
                    "weight": 0.8
                },
                {
                    "name": "Transparency",
                    "description": "Maintain clear and transparent communication",
                    "weight": 0.7
                }
            ]
        
        return PrincipleEngine(principles=principles, agent_id=self.agent_id)
    
    def _init_agent_card(self, agent_card_path: Optional[str]) -> AgentCard:
        """Initialize the agent card with agent information."""
        agent_info = {}
        
        if agent_card_path and os.path.exists(agent_card_path):
            with open(agent_card_path, 'r') as f:
                agent_info = json.load(f)
        
        # Add required fields if not present
        if "agent_id" not in agent_info:
            agent_info["agent_id"] = self.agent_id
        
        if "name" not in agent_info:
            agent_info["name"] = f"Bridge Builder {self.agent_id}"
            
        if "description" not in agent_info:
            agent_info["description"] = "An adaptive agent that bridges communication between other agents"
        
        if "version" not in agent_info:
            agent_info["version"] = "1.0.0"
        
        return AgentCard(**agent_info)
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message from another agent.
        
        This is the main entry point for message processing. It coordinates
        all components to analyze, adapt, and respond to messages.
        
        Args:
            message: The incoming message in JSON-RPC format
            
        Returns:
            Response message in JSON-RPC format
        """
        try:
            logger.info(f"Processing message from {message.get('params', {}).get('sender_id', 'unknown')}")
            
            # Extract message parameters
            if "params" not in message:
                return self._create_error_response(message.get("id"), "No params in message")
            
            params = message["params"]
            sender_id = params.get("sender_id")
            conversation_id = params.get("conversation_id")
            content = params.get("content")
            format = params.get("format", "text")
            
            if not sender_id or not content:
                return self._create_error_response(message.get("id"), "Missing required parameters")
            
            # Retrieve or create session
            session = self.session_manager.get_or_create_session(conversation_id)
            
            # Update session with the new message
            session.add_message(params)
            
            # Analyze communication style if we don't know this agent
            if sender_id not in self.known_agents:
                style = self.communication_style_analyzer.analyze_message(content)
                relationship = self.relationship_tracker.get_or_create_relationship(sender_id)
                
                self.known_agents[sender_id] = {
                    "communication_style": style,
                    "relationship": relationship,
                    "first_contact": True
                }
            else:
                # Update communication style understanding
                current_style = self.known_agents[sender_id]["communication_style"]
                updated_style = self.communication_style_analyzer.update_style(current_style, content)
                self.known_agents[sender_id]["communication_style"] = updated_style
                self.known_agents[sender_id]["first_contact"] = False
            
            # Check for conflicts
            conflict_detected = self.conflict_resolver.detect_conflict(message)
            
            if conflict_detected:
                logger.info(f"Conflict detected in message from {sender_id}")
                resolution = self.conflict_resolver.resolve_conflict(
                    message, 
                    self.known_agents[sender_id]["relationship"]
                )
                
                # Update relationship based on conflict resolution
                self.relationship_tracker.update_relationship(
                    sender_id, 
                    resolution.outcome
                )
            
            # Process content based on format
            processed_content = self.content_handler.process(content, format)
            
            # Check if message contains a task
            task_detected = self.a2a_task_handler.detect_task(message)
            
            if task_detected:
                logger.info(f"Task detected in message from {sender_id}")
                task_response = self.a2a_task_handler.handle_task(message, session)
                return task_response
            
            # Adapt response to sender's communication style
            sender_style = self.known_agents[sender_id]["communication_style"]
            adapted_response = self._generate_adapted_response(
                sender_id, 
                processed_content, 
                sender_style, 
                conflict_detected
            )
            
            # Create response message
            response = {
                "id": message.get("id", "unknown-id"),
                "jsonrpc": "2.0",
                "result": {
                    "status": "success",
                    "sender_id": self.agent_id,
                    "recipient_id": sender_id,
                    "conversation_id": conversation_id,
                    "content": adapted_response,
                    "format": format
                }
            }
            
            # Record the response in the session
            session.add_message(response["result"])
            
            logger.info(f"Sent response to {sender_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return self._create_error_response(
                message.get("id", "unknown-id"), 
                f"Internal error: {str(e)}"
            )
    
    def _generate_adapted_response(
        self, 
        recipient_id: str, 
        content: str, 
        recipient_style: CommunicationStyle,
        conflict_detected: bool
    ) -> str:
        """
        Generate a response adapted to the recipient's communication style.
        
        Args:
            recipient_id: ID of the recipient
            content: Content to respond to
            recipient_style: Communication style of the recipient
            conflict_detected: Whether a conflict was detected
            
        Returns:
            Adapted response content
        """
        # Sample response generation - in a real implementation,
        # this would involve more sophisticated NLP/LLM processing
        if self.known_agents[recipient_id].get("first_contact", False):
            # First contact - introduce the agent
            response = f"Hello, I am {self.agent_card.name}. {self.agent_card.description}. "
            response += "I've received your message and am processing it now."
        else:
            # Generate response based on content and adapt to style
            response = f"I've processed your message about: '{content[:50]}...'. "
            
            if conflict_detected:
                response += "I notice we may have a misalignment. Let's work together to find common ground. "
        
        # Adapt formality
        if recipient_style.formality.value >= 3:  # FORMAL or VERY_FORMAL
            response = response.replace("I've", "I have")
            response = response.replace("Let's", "Let us")
        else:  # CASUAL or VERY_CASUAL
            response = response.replace("I have", "I've")
            response = response.replace("Let us", "Let's")
        
        # Adapt detail level
        if recipient_style.detail_level.value >= 3:  # DETAILED or VERY_DETAILED
            response += "I'm analyzing the complete context of our conversation to provide you with comprehensive information. "
        else:  # CONCISE or VERY_CONCISE
            # Keep it shorter
            response = response.split('.')[0] + '.'
        
        # Adapt directness
        if recipient_style.directness.value >= 3:  # DIRECT or VERY_DIRECT
            response += "What specific information do you need next?"
        else:  # INDIRECT or VERY_INDIRECT
            response += "Perhaps we could explore this topic further if you're interested."
        
        return response
    
    def _create_error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "id": request_id,
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": message
            }
        }
    
    def exchange_agent_cards(self, agent_id: str, agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exchange agent cards with another agent.
        
        Args:
            agent_id: ID of the agent to exchange cards with
            agent_card: Agent card from the other agent
            
        Returns:
            This agent's card
        """
        # Store the other agent's card
        if agent_id not in self.known_agents:
            relationship = self.relationship_tracker.get_or_create_relationship(agent_id)
            self.known_agents[agent_id] = {
                "agent_card": agent_card,
                "relationship": relationship,
                "first_contact": True
            }
        else:
            self.known_agents[agent_id]["agent_card"] = agent_card
        
        # Return this agent's card
        return self.agent_card.to_dict()
    
    def save_state(self, state_dir: str = "agent_state") -> None:
        """
        Save the agent's state to disk.
        
        Args:
            state_dir: Directory to save state files
        """
        os.makedirs(state_dir, exist_ok=True)
        
        # Save known agents
        with open(os.path.join(state_dir, f"{self.agent_id}_known_agents.json"), 'w') as f:
            # Convert communication styles to dictionaries
            serializable_agents = {}
            for agent_id, agent_data in self.known_agents.items():
                serializable_agents[agent_id] = {
                    "communication_style": agent_data["communication_style"].to_dict() 
                        if "communication_style" in agent_data else None,
                    "relationship": agent_data["relationship"].to_dict() 
                        if "relationship" in agent_data else None,
                    "first_contact": agent_data.get("first_contact", False),
                    "agent_card": agent_data.get("agent_card")
                }
            json.dump(serializable_agents, f, indent=2)
        
        # Save sessions
        self.session_manager.save_sessions(os.path.join(state_dir, f"{self.agent_id}_sessions.json"))
        
        # Save relationships
        self.relationship_tracker.save_relationships(os.path.join(state_dir, f"{self.agent_id}_relationships.json"))
        
        logger.info(f"Agent state saved to {state_dir}")
    
    def load_state(self, state_dir: str = "agent_state") -> bool:
        """
        Load the agent's state from disk.
        
        Args:
            state_dir: Directory containing state files
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        try:
            # Load known agents
            known_agents_path = os.path.join(state_dir, f"{self.agent_id}_known_agents.json")
            if os.path.exists(known_agents_path):
                with open(known_agents_path, 'r') as f:
                    serialized_agents = json.load(f)
                    
                    for agent_id, agent_data in serialized_agents.items():
                        self.known_agents[agent_id] = {
                            "communication_style": CommunicationStyle.from_dict(agent_data["communication_style"]) 
                                if agent_data.get("communication_style") else None,
                            "relationship": self.relationship_tracker.relationship_from_dict(agent_data["relationship"]) 
                                if agent_data.get("relationship") else None,
                            "first_contact": agent_data.get("first_contact", False),
                            "agent_card": agent_data.get("agent_card")
                        }
            
            # Load sessions
            sessions_path = os.path.join(state_dir, f"{self.agent_id}_sessions.json")
            if os.path.exists(sessions_path):
                self.session_manager.load_sessions(sessions_path)
            
            # Load relationships
            relationships_path = os.path.join(state_dir, f"{self.agent_id}_relationships.json")
            if os.path.exists(relationships_path):
                self.relationship_tracker.load_relationships(relationships_path)
            
            logger.info(f"Agent state loaded from {state_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading agent state: {str(e)}")
            return False
```

This class serves as the main integration point for all components, providing:

1. Initialization of all components with proper dependencies
2. Message processing pipeline
3. Communication style adaptation
4. Conflict detection and resolution
5. Task handling
6. State persistence and loading

## Configuration Options

The Adaptive Bridge Builder can be configured for different deployment scenarios. Create configuration files that adjust behavior based on your needs:

### Basic Configuration (config.json)

```json
{
  "log_level": "INFO",
  "file_storage_dir": "file_storage",
  "max_session_history": 100,
  "message_retention_days": 30,
  "communication_defaults": {
    "formality": "NEUTRAL",
    "detail_level": "BALANCED",
    "directness": "BALANCED",
    "emotional_tone": "NEUTRAL",
    "response_speed": "STANDARD"
  },
  "security": {
    "require_authentication": true,
    "token_expiration_minutes": 60,
    "allowed_sources": ["trusted-agent-1", "trusted-agent-2"]
  },
  "a2a_server": {
    "host": "0.0.0.0",
    "port": 8080,
    "ssl_enabled": false
  }
}
```

### Principles Configuration (principles.json)

```json
[
  {
    "name": "User Privacy",
    "description": "Respect user privacy and maintain confidentiality of information",
    "weight": 1.0,
    "rules": [
      "Never share user information without explicit consent",
      "Store only necessary information for the minimal required time",
      "Apply appropriate security measures to all user data"
    ]
  },
  {
    "name": "Accuracy",
    "description": "Provide accurate and truthful information",
    "weight": 0.9,
    "rules": [
      "Verify information before sharing it",
      "Acknowledge uncertainty when it exists",
      "Correct misinformation promptly"
    ]
  },
  {
    "name": "Adaptability",
    "description": "Adapt to user needs while maintaining core values",
    "weight": 0.8,
    "rules": [
      "Adjust communication style to match the user",
      "Maintain principle consistency while being flexible",
      "Continuously improve based on feedback"
    ]
  }
]
```

### Agent Card Configuration (agent_card.json)

```json
{
  "agent_id": "adaptive-bridge-1",
  "name": "Adaptive Bridge Builder",
  "description": "An intelligent agent that adapts to different communication styles and facilitates agent-to-agent interactions while maintaining principle alignment",
  "version": "1.0.0",
  "capabilities": [
    "communication_style_adaptation",
    "conflict_resolution",
    "principle_aligned_reasoning",
    "collaborative_task_handling",
    "file_exchange"
  ],
  "communication_preferences": {
    "preferred_format": "json",
    "supported_formats": ["text", "json", "html", "markdown"],
    "languages": ["en"]
  },
  "principles": [
    "User Privacy",
    "Accuracy",
    "Adaptability",
    "Transparency"
  ]
}
```

## A2A Server Implementation

To enable agent-to-agent communication, implement a server that exposes endpoints for messaging. Here's a FastAPI implementation:

```python
#!/usr/bin/env python3
"""
Adaptive Bridge Builder A2A Server

This module implements a server for agent-to-agent communication
using the Adaptive Bridge Builder.
"""

import json
import logging
import os
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Any, Optional
import jwt
from datetime import datetime, timedelta

from adaptive_bridge_builder import AdaptiveBridgeBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("A2AServer")

# Create FastAPI app
app = FastAPI(
    title="Adaptive Bridge Builder A2A Server",
    description="Server for agent-to-agent communication using the Adaptive Bridge Builder",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        "log_level": "INFO",
        "agent_id": "adaptive-bridge-1",
        "security": {
            "secret_key": "replace-with-your-secret-key",
            "token_expiration_minutes": 60,
            "require_authentication": True
        }
    }

# Initialize configuration and bridge builder
config = load_config()
logging.getLogger().setLevel(getattr(logging, config.get("log_level", "INFO")))

# Initialize the bridge builder
bridge_builder = AdaptiveBridgeBuilder(
    agent_id=config.get("agent_id", "adaptive-bridge-1"),
    config_path=os.environ.get("CONFIG_PATH", "config.json"),
    principles_path=os.environ.get("PRINCIPLES_PATH", "principles.json"),
    agent_card_path=os.environ.get("AGENT_CARD_PATH", "agent_card.json")
)

# Security utilities
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config["security"]["secret_key"], algorithm="HS256")

def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, config["security"]["secret_key"], algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

async def verify_auth(request: Request):
    if not config["security"].get("require_authentication", True):
        return True
        
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.split("Bearer ")[1]
    return verify_token(token)

# API Endpoints
@app.post("/api/message")
async def process_message(
    message: Dict[str, Any] = Body(...),
    authenticated: bool = Depends(verify_auth)
):
    """Process an incoming agent message."""
    try:
        logger.info(f"Received message from {message.get('params', {}).get('sender_id', 'unknown')}")
        response = bridge_builder.process_message(message)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/api/agent-card/exchange")
async def exchange_agent_cards(
    request: Dict[str, Any] = Body(...),
    authenticated: bool = Depends(verify_auth)
):
    """Exchange agent cards."""
    try:
        agent_id = request.get("agent_id")
        agent_card = request.get("agent_card")
        
        if not agent_id or not agent_card:
            return JSONResponse(
                status_code=400, 
                content={"error": "Missing agent_id or agent_card"}
            )
        
        response = bridge_builder.exchange_agent_cards(agent_id, agent_card)
        return JSONResponse(content={"agent_card": response})
    except Exception as e:
        logger.error(f"Error exchanging agent cards: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/api/auth/token")
async def get_token(request: Dict[str, Any] = Body(...)):
    """Get an authentication token."""
    agent_id = request.get("agent_id")
    api_key = request.get("api_key")
    
    # In a real implementation, validate the API key against stored values
    # This is a simplified example
    if not agent_id or not api_key or api_key != "your-api-key":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    expires = timedelta(minutes=config["security"].get("token_expiration_minutes", 60))
    token = create_access_token(
        data={"sub": agent_id},
        expires_delta=expires
    )
    
    return {"access_token": token, "token_type": "bearer", "expires_in": expires.total_seconds()}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent_id": bridge_builder.agent_id}

@app.post("/api/file/upload")
async def upload_file(
    file: UploadFile = File(...),
    sender_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    authenticated: bool = Depends(verify_auth)
):
    """Handle file uploads."""
    try:
        file_path = await bridge_builder.file_exchange_handler.save_file(
            file=file,
            sender_id=sender_id,
            conversation_id=conversation_id
        )
        
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error uploading file: {str(e)}"}
        )

@app.get("/api/file/{file_id}")
async def get_file(
    file_id: str,
    authenticated: bool = Depends(verify_auth)
):
    """Get a file by ID."""
    try:
        file_path = bridge_builder.file_exchange_handler.get_file_path(file_id)
        if not file_path or not os.
