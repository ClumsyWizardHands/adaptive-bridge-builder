# Adaptive Bridge Builder - Google ADK Integration Guide

This guide outlines how to integrate the Adaptive Bridge Builder with Google's Agent Development Kit (ADK). It provides a comprehensive roadmap for mapping existing A2A concepts to ADK components, implementing the PrincipleEngine within ADK's framework, leveraging ADK's session and memory services, and handling events properly.

## Table of Contents

1. [Mapping A2A to ADK](#mapping-a2a-to-adk)
2. [Implementing PrincipleEngine with ADK](#implementing-principleengine-with-adk)
3. [Using ADK Session and Memory Services](#using-adk-session-and-memory-services)
4. [Handling Events in ADK](#handling-events-in-adk)
5. [Migration Path from Custom Implementation](#migration-path-from-custom-implementation)

## Mapping A2A to ADK

The following table maps Adaptive Bridge Builder components to their corresponding ADK equivalents:

| Adaptive Bridge Builder | Google ADK | Notes |
|-------------------------|------------|-------|
| `AdaptiveBridgeBuilder` | `AgentApp` | Top-level agent application |
| `AgentCard` | `AgentManifest` | Agent identity and capabilities |
| `SessionManager` | `SessionManager` | Conversation context management |
| `ContentHandler` | `ContentParser` | Process different content formats |
| `RelationshipTracker` | `MemoryStore` | Store agent relationships |
| `A2ATaskHandler` | `ToolExecutor` | Execute agent-to-agent tasks |
| `PrincipleEngine` | Custom component using ADK's `RulesEngine` | Principles enforcement |
| `ConflictResolver` | Custom component using ADK's `ConflictHandlers` | Conflict detection and resolution |
| `FileExchangeHandler` | `AssetService` | File sharing between agents |

### Code Example: Basic ADK Agent Setup

```python
from google.adk import AgentApp, AgentManifest, SessionManager
from google.adk.tools import ToolExecutor, AssetService
from google.adk.memory import MemoryStore

# Define the agent manifest with metadata
agent_manifest = AgentManifest(
    agent_id="adaptive-bridge-1",
    name="Adaptive Bridge Builder",
    description="An agent that adapts to communication styles and facilitates agent-to-agent interactions",
    version="1.0.0",
    capabilities=["communication_style_adaptation", "conflict_resolution", "principle_aligned_reasoning"]
)

# Initialize the ADK agent application
app = AgentApp(
    manifest=agent_manifest,
    session_manager=SessionManager(),
    memory_store=MemoryStore(persistence_path="./memory"),
    tool_executor=ToolExecutor(),
    asset_service=AssetService(storage_path="./assets")
)

# Register handlers for different events
@app.on_message
def handle_message(message, session):
    # Message handling logic
    pass

@app.on_agent_interaction
def handle_agent_interaction(agent_id, interaction_type, payload, session):
    # Agent-to-agent interaction logic
    pass

@app.on_startup
def initialize_components():
    # Initialize any custom components
    pass

# Start the agent application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

## Implementing PrincipleEngine with ADK

The PrincipleEngine can be implemented using ADK's `RulesEngine` and custom components. This integration allows the agent to maintain principle alignment while using ADK's infrastructure.

### Implementation Strategy

1. Define principles as ADK rules
2. Use ADK's rule evaluation framework
3. Implement principle checking as middleware in the request processing pipeline
4. Store principle state in ADK's memory service

### Code Example: PrincipleEngine Integration

```python
from google.adk import RulesEngine, Rule, RuleContext
from google.adk.memory import MemoryStore
from typing import List, Dict, Any, Optional

class ADKPrincipleEngine:
    """PrincipleEngine implementation using ADK's RulesEngine."""
    
    def __init__(
        self, 
        memory_store: MemoryStore,
        agent_id: str,
        principles_config: Optional[str] = None
    ):
        self.memory_store = memory_store
        self.agent_id = agent_id
        self.rules_engine = RulesEngine()
        
        # Load principles from config or use defaults
        self.principles = self._load_principles(principles_config)
        
        # Register principles as rules
        self._register_principles()
    
    def _load_principles(self, config_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load principles from configuration or use defaults."""
        # Implementation similar to original PrincipleEngine._load_principles
        # but leveraging ADK's configuration loading mechanisms
        if config_path:
            # Load from ADK config service
            return self.memory_store.get("principles_config") or []
        
        # Default principles
        return [
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
            # More default principles...
        ]
    
    def _register_principles(self):
        """Register principles as rules in the ADK RulesEngine."""
        for principle in self.principles:
            rule = Rule(
                name=principle["name"],
                description=principle["description"],
                weight=principle["weight"],
                evaluation_function=self._create_principle_evaluator(principle)
            )
            self.rules_engine.register_rule(rule)
    
    def _create_principle_evaluator(self, principle: Dict[str, Any]):
        """Create an evaluation function for a principle."""
        def evaluate_principle(context: RuleContext) -> float:
            # Extract relevant data from the context
            message = context.get("message")
            
            # Placeholder for actual principle evaluation logic
            # In a real implementation, this would use LLMs or other methods
            # to evaluate if the message adheres to the principle
            return 1.0  # Fully compliant by default
        
        return evaluate_principle
    
    def evaluate_message(self, message: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a message against all principles."""
        context = RuleContext({"message": message})
        evaluation_results = self.rules_engine.evaluate_all(context)
        
        # Store evaluation results in memory
        self.memory_store.set(
            f"principle_evaluation:{message.get('id', 'unknown')}",
            evaluation_results
        )
        
        return evaluation_results
    
    def get_principle_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a principle by name."""
        for principle in self.principles:
            if principle["name"] == name:
                return principle
        return None

# Register the PrincipleEngine with the ADK agent
@app.on_startup
def initialize_principle_engine():
    principle_engine = ADKPrincipleEngine(
        memory_store=app.memory_store,
        agent_id=app.manifest.agent_id,
        principles_config="principles.json"
    )
    
    # Store in the app's context for later use
    app.context["principle_engine"] = principle_engine

# Use the PrincipleEngine in message handling
@app.on_message
def handle_message(message, session):
    principle_engine = app.context["principle_engine"]
    
    # Evaluate the message against principles
    evaluation_results = principle_engine.evaluate_message(message)
    
    # Use evaluation results to guide response
    if all(score > 0.7 for score in evaluation_results.values()):
        # All principles satisfied, proceed normally
        return {"status": "success", "response": "Message processed successfully"}
    else:
        # Some principles violated, handle accordingly
        violated_principles = [name for name, score in evaluation_results.items() if score <= 0.7]
        return {
            "status": "warning",
            "response": f"Message violates principles: {', '.join(violated_principles)}"
        }
```

## Using ADK Session and Memory Services

ADK provides robust session and memory services that can be leveraged for relationship tracking and state persistence.

### ADK Memory Service for Relationship Tracking

The RelationshipTracker can be implemented using ADK's MemoryStore:

```python
from google.adk.memory import MemoryStore
from typing import Dict, Any, Optional
import time

class ADKRelationshipTracker:
    """Relationship tracking implementation using ADK's MemoryStore."""
    
    def __init__(self, memory_store: MemoryStore, agent_id: str):
        self.memory_store = memory_store
        self.agent_id = agent_id
    
    def get_relationship(self, agent_id: str) -> Dict[str, Any]:
        """Get the relationship with a specific agent."""
        relationship_key = f"relationship:{agent_id}"
        
        # Try to get existing relationship from memory
        relationship = self.memory_store.get(relationship_key)
        
        if relationship:
            return relationship
        
        # If no relationship exists, create a new one
        return self.create_relationship(agent_id)
    
    def create_relationship(self, agent_id: str) -> Dict[str, Any]:
        """Create a new relationship with an agent."""
        relationship = {
            "agent_id": agent_id,
            "trust_level": 0.5,  # Neutral trust initially
            "interaction_count": 0,
            "last_interaction": time.time(),
            "history": [],
            "status": "new"
        }
        
        # Store the new relationship
        self.memory_store.set(f"relationship:{agent_id}", relationship)
        
        return relationship
    
    def update_relationship(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing relationship."""
        relationship = self.get_relationship(agent_id)
        
        # Apply updates
        for key, value in updates.items():
            if key in relationship:
                relationship[key] = value
        
        # Update interaction metadata
        relationship["interaction_count"] += 1
        relationship["last_interaction"] = time.time()
        
        # Store the updated relationship
        self.memory_store.set(f"relationship:{agent_id}", relationship)
        
        return relationship
    
    def record_interaction(self, agent_id: str, interaction_type: str, outcome: str) -> None:
        """Record an interaction with an agent."""
        relationship = self.get_relationship(agent_id)
        
        # Create interaction record
        interaction = {
            "type": interaction_type,
            "outcome": outcome,
            "timestamp": time.time()
        }
        
        # Add to history
        relationship["history"].append(interaction)
        
        # Limit history size
        if len(relationship["history"]) > 100:
            relationship["history"] = relationship["history"][-100:]
        
        # Update trust level based on outcome
        if outcome == "positive":
            relationship["trust_level"] = min(1.0, relationship["trust_level"] + 0.05)
        elif outcome == "negative":
            relationship["trust_level"] = max(0.0, relationship["trust_level"] - 0.05)
        
        # Store the updated relationship
        self.memory_store.set(f"relationship:{agent_id}", relationship)
    
    def get_all_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked relationships."""
        # Get all keys with the relationship prefix
        relationship_keys = self.memory_store.scan_keys("relationship:*")
        
        relationships = {}
        for key in relationship_keys:
            agent_id = key.split(":", 1)[1]
            relationships[agent_id] = self.memory_store.get(key)
        
        return relationships

# Register the RelationshipTracker with the ADK agent
@app.on_startup
def initialize_relationship_tracker():
    relationship_tracker = ADKRelationshipTracker(
        memory_store=app.memory_store,
        agent_id=app.manifest.agent_id
    )
    
    # Store in the app's context for later use
    app.context["relationship_tracker"] = relationship_tracker
```

### Using ADK Session Management

ADK's session management can be used to maintain conversation context:

```python
from google.adk import Session
from typing import Dict, Any, List, Optional
import uuid

# Use ADK's session in message handling
@app.on_message
def handle_message(message: Dict[str, Any], session: Session):
    # Add the message to the session history
    if "history" not in session:
        session["history"] = []
    
    session["history"].append({
        "role": "user" if message.get("sender_type") == "user" else "agent",
        "content": message.get("content", ""),
        "timestamp": message.get("timestamp", time.time())
    })
    
    # Get the communication style for the sender
    sender_id = message.get("sender_id")
    communication_style = get_or_analyze_communication_style(sender_id, message, session)
    
    # Generate response adapted to the sender's style
    response = generate_adapted_response(message, communication_style, session)
    
    # Add the response to the session history
    session["history"].append({
        "role": "agent",
        "content": response,
        "timestamp": time.time()
    })
    
    # Return the response
    return {
        "content": response,
        "sender_id": app.manifest.agent_id,
        "recipient_id": sender_id,
        "conversation_id": session.id
    }

def get_or_analyze_communication_style(sender_id: str, message: Dict[str, Any], session: Session) -> Dict[str, Any]:
    """Get existing or analyze new communication style for a sender."""
    # Check if we have a stored style in the session
    if "communication_styles" not in session:
        session["communication_styles"] = {}
    
    if sender_id in session["communication_styles"]:
        return session["communication_styles"][sender_id]
    
    # Otherwise analyze the message to determine style
    # This would call the CommunicationStyleAnalyzer implementation
    style = analyze_communication_style(message.get("content", ""))
    
    # Store for future use
    session["communication_styles"][sender_id] = style
    
    return style

def analyze_communication_style(content: str) -> Dict[str, Any]:
    """Analyze content to determine communication style."""
    # Placeholder implementation
    # In a real system, this would use NLP or other analysis methods
    return {
        "formality": "neutral",
        "detail_level": "balanced",
        "directness": "balanced",
        "emotional_tone": "neutral"
    }

def generate_adapted_response(message: Dict[str, Any], style: Dict[str, Any], session: Session) -> str:
    """Generate a response adapted to the sender's style."""
    # Placeholder implementation
    # In a real system, this would use more sophisticated generation
    content = message.get("content", "")
    
    if style["formality"] == "formal":
        response = f"I have received your message regarding '{content[:30]}...'. "
    else:
        response = f"Got your message about '{content[:30]}...'. "
    
    if style["detail_level"] == "detailed":
        response += "I'll provide a comprehensive response with all relevant details. "
    else:
        response += "Here's a quick response. "
    
    # Add conversation context awareness
    history_len = len(session.get("history", []))
    if history_len > 2:
        response += f"This is our {history_len // 2}th exchange. "
    
    return response
```

## Handling Events in ADK

ADK uses an event-driven architecture. Here's how to implement event handling for different components:

```python
from google.adk import Event, EventType

# Handle agent-to-agent interaction events
@app.on_event(EventType.AGENT_INTERACTION)
def handle_agent_interaction(event: Event):
    """Handle interactions with other agents."""
    interaction_data = event.data
    agent_id = interaction_data.get("agent_id")
    interaction_type = interaction_data.get("type")
    
    # Get the relationship tracker
    relationship_tracker = app.context["relationship_tracker"]
    
    # Record the interaction
    relationship_tracker.record_interaction(
        agent_id=agent_id,
        interaction_type=interaction_type,
        outcome=interaction_data.get("outcome", "neutral")
    )
    
    # Get current relationship
    relationship = relationship_tracker.get_relationship(agent_id)
    
    # Handle based on interaction type
    if interaction_type == "message":
        # Handle message exchange
        pass
    elif interaction_type == "task":
        # Handle task request/response
        pass
    elif interaction_type == "conflict":
        # Handle conflict resolution
        handle_conflict(agent_id, interaction_data, relationship)

# Handle conflict events
def handle_conflict(agent_id: str, interaction_data: Dict[str, Any], relationship: Dict[str, Any]):
    """Handle conflicts with other agents."""
    # Get the conflict resolver
    conflict_resolver = app.context.get("conflict_resolver")
    if not conflict_resolver:
        # Initialize if not available
        conflict_resolver = initialize_conflict_resolver()
        app.context["conflict_resolver"] = conflict_resolver
    
    # Resolve the conflict
    resolution = conflict_resolver.resolve_conflict(
        agent_id=agent_id,
        conflict_data=interaction_data.get("conflict_data", {}),
        relationship=relationship
    )
    
    # Update the relationship based on resolution
    relationship_tracker = app.context["relationship_tracker"]
    relationship_tracker.update_relationship(
        agent_id=agent_id,
        updates={
            "status": resolution.get("status", "active"),
            "trust_level": resolution.get("trust_level", relationship["trust_level"])
        }
    )
    
    # Return the resolution
    return resolution

# Handle file exchange events
@app.on_event(EventType.FILE_EXCHANGE)
def handle_file_exchange(event: Event):
    """Handle file sharing between agents."""
    file_data = event.data
    operation = file_data.get("operation")
    sender_id = file_data.get("sender_id")
    file_id = file_data.get("file_id")
    
    # Get the asset service
    asset_service = app.asset_service
    
    if operation == "upload":
        # Handle file upload
        file_content = file_data.get("content")
        file_metadata = file_data.get("metadata", {})
        
        # Store the file
        asset_id = asset_service.store_asset(
            content=file_content,
            metadata={
                "sender_id": sender_id,
                "filename": file_metadata.get("filename", "unknown"),
                "mime_type": file_metadata.get("mime_type", "application/octet-stream"),
                "timestamp": time.time()
            }
        )
        
        return {"status": "success", "asset_id": asset_id}
    
    elif operation == "download":
        # Handle file download
        asset = asset_service.get_asset(file_id)
        if not asset:
            return {"status": "error", "message": "File not found"}
        
        return {
            "status": "success",
            "content": asset.content,
            "metadata": asset.metadata
        }
```

## Migration Path from Custom Implementation

If you have an existing custom implementation, follow these steps to port it to ADK:

### 1. Set Up the ADK Project Structure

```bash
# Create a new ADK project
mkdir adk-bridge-builder
cd adk-bridge-builder

# Initialize ADK project
adk init --name "Adaptive Bridge Builder" --id "adaptive-bridge-1"

# Install required dependencies
pip install google-adk
```

### 2. Map Existing Components

Create adapter classes that wrap your existing implementation with ADK interfaces:

```python
# adapter.py
from google.adk import AgentApp, Session
from typing import Dict, Any

# Import your existing implementation
from your_implementation.adaptive_bridge_builder import AdaptiveBridgeBuilder
from your_implementation.principle_engine import PrincipleEngine
from your_implementation.relationship_tracker import RelationshipTracker

class AdaptiveBridgeBuilderAdapter:
    """Adapter to wrap the existing AdaptiveBridgeBuilder with ADK interfaces."""
    
    def __init__(self, app: AgentApp):
        self.app = app
        
        # Initialize your existing implementation
        self.bridge_builder = AdaptiveBridgeBuilder(
            agent_id=app.manifest.agent_id,
            # Map other parameters as needed
        )
    
    def process_message(self, message: Dict[str, Any], session: Session) -> Dict[str, Any]:
        """Process a message using your existing implementation, but with ADK session."""
        # Convert ADK session to your session format
        your_session = self._convert_session(session)
        
        # Process using your implementation
        result = self.bridge_builder.process_message({
            "jsonrpc": "2.0",
            "id": message.get("id", "unknown"),
            "params": {
                "sender_id": message.get("sender_id"),
                "conversation_id": session.id,
                "content": message.get("content"),
                "format": message.get("format", "text")
            }
        })
        
        # Convert your session back to ADK session
        self._update_adk_session(session, your_session)
        
        return result
    
    def _convert_session(self, adk_session: Session):
        """Convert ADK session to your session format."""
        # Implementation depends on your session structure
        your_session_data = {
            "id": adk_session.id,
            "messages": adk_session.get("history", []),
            # Other session data...
        }
        
        return your_session_data
    
    def _update_adk_session(self, adk_session: Session, your_session: Dict[str, Any]):
        """Update ADK session with data from your session."""
        # Implementation depends on your session structure
        adk_session["history"] = your_session.get("messages", [])
        # Other session data...

# Register the adapter with ADK
@app.on_startup
def initialize_adapter():
    adapter = AdaptiveBridgeBuilderAdapter(app)
    app.context["adapter"] = adapter

# Use the adapter in event handlers
@app.on_message
def handle_message(message: Dict[str, Any], session: Session):
    adapter = app.context["adapter"]
    return adapter.process_message(message, session)
```

### 3. Convert Key Components

Gradually replace adapter components with native ADK implementations:

#### PrincipleEngine Conversion

```python
# principle_engine_adk.py
from google.adk import RulesEngine, Rule
from your_implementation.principle_engine import PrincipleEngine

class ADKPrincipleEngine:
    """Native ADK implementation of PrincipleEngine."""
    
    def __init__(self, adk_rules_engine: RulesEngine, original_engine: PrincipleEngine):
        self.adk_rules_engine = adk_rules_engine
        self.original_engine = original_engine
        
        # Convert original principles to ADK rules
        self._convert_principles_to_rules()
    
    def _convert_principles_to_rules(self):
        """Convert original principles to ADK rules."""
        for principle in self.original_engine.principles:
            # Create ADK rule from principle
            rule = Rule(
                name=principle["name"],
                description=principle["description"],
                # Other rule properties...
            )
            
            # Register with ADK rules engine
            self.adk_rules_engine.register_rule(rule)
    
    # Implement other methods...
```

### 4. Update Configuration

Update configuration to use ADK formats:

```python
# config/agent_manifest.json
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
  "apis": [
    {
      "name": "process_message",
      "description": "Process an incoming message",
      "parameters": {
        "sender_id": {
          "type": "string",
          "description": "ID of the sender"
        },
        "content": {
          "type": "string",
          "description": "Message content"
        },
        "format": {
          "type": "string",
          "description": "Content format"
        }
      }
    },
    {
      "name": "exchange_agent_cards",
      "description": "Exchange agent cards with another agent",
      "parameters": {
        "agent_id": {
          "type": "string",
          "description": "ID of the agent to exchange cards with"
        },
        "agent_card": {
          "type": "object",
          "description": "Agent card from the other agent"
        }
      }
    }
  ]
}
```

### 5. Gradual Migration Strategy

1. **Phase 1**: Use adapters to wrap existing components
2. **Phase 2**: Replace individual components with native ADK implementations
3. **Phase 3**: Refactor to use ADK patterns and services fully
4. **Phase 4**: Optimize and leverage ADK-specific features

This approach allows for incremental migration while maintaining functionality.

### Code Example: Complete Migration Main File

```python
#!/usr/bin/env python3
"""
Adaptive Bridge Builder - ADK Implementation

This module provides the entry point for the ADK version of the
Adaptive Bridge Builder agent.
"""

from google.adk import AgentApp, Session
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AdaptiveBridgeBuilderADK")

# Initialize the ADK agent app
app = AgentApp()

# Import event handlers
from handlers.message_handler import handle_message
from handlers.agent_interaction_handler import handle_agent_interaction
from handlers.file_exchange_handler import handle_file_exchange

# Register event handlers
app.on_message(handle_message)
app.on_event("agent_interaction", handle_agent_interaction)
app.on_event("file_exchange", handle_file_exchange)

# Initialize components on startup
@app.on_startup
def initialize_components():
    from components.principle_engine_adk import ADKPrincipleEngine
    from components.relationship_tracker_adk import ADKRelationshipTracker
    from components.conflict_resolver_adk import ADKConflictResolver
    
    # Initialize ADK-native components
    principle_engine = ADKPrincipleEngine(
        memory_store=app.memory_store,
        agent_id=app.manifest.agent_id,
        principles_config=os.environ.get("PRINCIPLES_CONFIG", "config/principles.json")
    )
    
    relationship_tracker = ADKRelationshipTracker(
        memory_store=app.memory_store,
        agent_id=app.manifest.agent_id
    )
    
    conflict_resolver = ADKConflictResolver(
        principle_engine=principle_engine,
        relationship_tracker=relationship_tracker
    )
    
    # Store in app context
    app.context["principle_engine"] = principle_engine
    app.context["relationship_tracker"] = relationship_tracker
    app.context["conflict_resolver"] = conflict_resolver
    
    logger.info("Adaptive Bridge Builder ADK components initialized")

# Start the agent application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
```

This comprehensive migration plan provides a clear path from a custom implementation to a fully ADK-integrated Adaptive Bridge Builder agent, leveraging ADK's robust services while maintaining the core functionality of the original system.
