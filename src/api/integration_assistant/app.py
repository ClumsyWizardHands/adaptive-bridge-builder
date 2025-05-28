"""
FastAPI Integration Assistant Backend

Provides API endpoints for agent registration, WebSocket connections,
integration code generation, and connection testing.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Coroutine, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    AgentRegistration,
    AgentRegistrationResponse,
    ConnectionStatus,
    IntegrationCodeRequest,
    IntegrationCodeResponse,
    ConnectionTestRequest,
    ConnectionTestResponse,
    ErrorResponse,
    AgentFramework
)
from .websocket_manager import ConnectionManager
from .code_generator import IntegrationCodeGenerator

# Import from existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent_registry import AgentRegistry
from ai_framework_detector import AIFrameworkDetector
from universal_agent_connector import UniversalAgentConnector


# Global instances
connection_manager = ConnectionManager()
code_generator = IntegrationCodeGenerator()
agent_registry = AgentRegistry()
framework_detector = AIFrameworkDetector()


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    Manage application lifecycle - startup and shutdown
    """
    # Startup
    print("Starting Integration Assistant API...")
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    print("Shutting down Integration Assistant API...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# Create FastAPI app
app = FastAPI(
    title="Adaptive Bridge Builder Integration Assistant",
    description="API for agent registration, integration, and real-time communication",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": connection_manager.get_active_connections_count()
    }


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time updates
    
    Query params:
        token: Optional connection token for authentication
    """
    # Connect the WebSocket
    client_id = await connection_manager.connect(websocket)
    
    try:
        # TODO: Add cancellation check or break condition
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "associate_agent":
                # Associate client with an agent
                agent_id = data.get("agent_id")
                if agent_id:
                    await connection_manager.associate_agent(client_id, agent_id)
                    
            elif message_type == "ping":
                # Respond to ping
                await connection_manager.send_personal_message(
                    client_id,
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                )
                
            elif message_type == "broadcast":
                # Broadcast to all clients of an agent
                agent_id = data.get("agent_id")
                message = data.get("message", {})
                if agent_id:
                    await connection_manager.broadcast_to_agent(agent_id, message)
                    
            else:
                # Echo unknown messages back
                await connection_manager.send_personal_message(
                    client_id,
                    {
                        "type": "echo",
                        "original_type": message_type,
                        "data": data
                    }
                )
                
    except WebSocketDisconnect:
        # Handle disconnection
        await connection_manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")
    except Exception as e:
        # Handle other errors
        print(f"WebSocket error for client {client_id}: {e}")
        await connection_manager.disconnect(client_id)


# Agent registration endpoint
@app.post("/api/agents/register", response_model=AgentRegistrationResponse)
async def register_agent(registration: AgentRegistration) -> None:
    """
    Register a new agent with the system
    """
    try:
        # Generate agent ID
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        
        # Register with agent registry
        agent_info = {
            "id": agent_id,
            "name": registration.name,
            "framework": registration.framework.value,
            "version": registration.version,
            "description": registration.description,
            "capabilities": [cap.dict() for cap in registration.capabilities],
            "endpoint_url": registration.endpoint_url,
            "metadata": registration.metadata
        }
        
        agent_registry.register_agent(agent_id, agent_info)
        
        # Generate connection token
        connection_token = f"token-{uuid.uuid4().hex}"
        
        # Generate initial integration code if framework is known
        integration_code = None
        if registration.framework != AgentFramework.CUSTOM:
            code_request = IntegrationCodeRequest(
                agent_id=agent_id,
                framework=registration.framework,
                include_examples=False
            )
            code_response = code_generator.generate_code(code_request)
            integration_code = code_response.code
            
        # Notify connected clients
        await connection_manager.broadcast_all({
            "type": "agent_registered",
            "agent_id": agent_id,
            "name": registration.name,
            "framework": registration.framework.value
        })
        
        return AgentRegistrationResponse(
            agent_id=agent_id,
            registration_time=datetime.utcnow(),
            status="registered",
            connection_token=connection_token,
            integration_code=integration_code
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get registered agents
@app.get("/api/agents")
async def get_agents(
    framework: Optional[AgentFramework] = None,
    capability: Optional[str] = None
):
    """
    Get list of registered agents
    
    Query params:
        framework: Filter by framework type
        capability: Filter by capability
    """
    agents = agent_registry.list_agents()
    
    # Apply filters
    if framework:
        agents = [a for a in agents if a.get("framework") == framework.value]
        
    if capability:
        agents = [
            a for a in agents 
            if any(cap.get("name") == capability for cap in a.get("capabilities", []))
        ]
        
    return {"agents": agents, "count": len(agents)}


# Get specific agent
@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str) -> None:
    """Get information about a specific agent"""
    agent = agent_registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


# Generate integration code
@app.post("/api/integration/code", response_model=IntegrationCodeResponse)
async def generate_integration_code(request: IntegrationCodeRequest) -> None:
    """Generate integration code for an agent"""
    try:
        # Verify agent exists
        agent = agent_registry.get_agent(request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        # Generate code
        response = code_generator.generate_code(request)
        
        # Notify via WebSocket
        await connection_manager.broadcast_to_agent(
            request.agent_id,
            {
                "type": "integration_code_generated",
                "framework": request.framework.value,
                "language": request.language
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Test connection endpoint
@app.post("/api/agents/{agent_id}/test", response_model=ConnectionTestResponse)
async def test_connection(agent_id: str, request: ConnectionTestRequest) -> None:
    """Test connection to an agent"""
    try:
        # Get agent info
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        # Update connection state
        await connection_manager.send_update(
            agent_id,
            "connection_state",
            {"state": "testing"}
        )
        
        # Perform connection test
        start_time = datetime.utcnow()
        
        # Try to connect using UniversalAgentConnector
        connector = UniversalAgentConnector()
        test_result = await connector.test_agent_connection(
            agent_id,
            agent.get("endpoint_url"),
            request.test_payload or {}
        )
        
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create response
        response = ConnectionTestResponse(
            agent_id=agent_id,
            success=test_result.get("success", False),
            response_time_ms=response_time_ms,
            test_results=test_result,
            error_message=test_result.get("error")
        )
        
        # Update connection state
        await connection_manager.send_update(
            agent_id,
            "connection_state",
            {
                "state": "connected" if response.success else "error",
                "test_result": response.dict()
            }
        )
        
        return response
        
    except Exception as e:
        # Update connection state on error
        await connection_manager.send_update(
            agent_id,
            "connection_state",
            {"state": "error", "error": str(e)}
        )
        raise HTTPException(status_code=400, detail=str(e))


# Detect framework from code
@app.post("/api/detect-framework")
async def detect_framework(code_snippet: str) -> Dict[str, Any]:
    """Detect AI framework from code snippet"""
    try:
        detection_result = framework_detector.detect_framework(code_snippet)
        
        # Convert to our framework enum if possible
        detected_framework = detection_result.get("framework", "").lower()
        framework_enum = None
        
        for framework in AgentFramework:
            if framework.value == detected_framework:
                framework_enum = framework
                break
                
        return {
            "detected_framework": detected_framework,
            "framework_enum": framework_enum.value if framework_enum else None,
            "confidence": detection_result.get("confidence", 0),
            "suggested_adapter": detection_result.get("suggested_adapter")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get connection status
@app.get("/api/connections/{client_id}/status", response_model=ConnectionStatus)
async def get_connection_status(client_id: str) -> None:
    """Get status of a WebSocket connection"""
    status = await connection_manager.get_connection_status(client_id)
    if not status:
        raise HTTPException(status_code=404, detail="Connection not found")
    return status


# Get all active connections
@app.get("/api/connections")
async def get_connections() -> Dict[str, Any]:
    """Get all active connections"""
    connections = []
    for client_id, status in connection_manager.connection_metadata.items():
        connections.append(status.dict())
    return {
        "connections": connections,
        "total": len(connections)
    }


# A2A Protocol endpoint (integration with existing A2A handler)
@app.post("/api/a2a/send")
async def send_a2a_message(message: Dict) -> None:
    """Send A2A protocol message"""
    try:
        # Import A2A handler
        from src.api.a2a.handlers import handle_a2a_message
        
        # Process the message
        result = await handle_a2a_message(message)
        
        # Notify via WebSocket if agent is connected
        agent_id = message.get("params", {}).get("from")
        if agent_id:
            await connection_manager.broadcast_to_agent(
                agent_id,
                {
                    "type": "a2a_message_sent",
                    "message_id": message.get("id"),
                    "method": message.get("method")
                }
            )
            
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": str(e)
                },
                "id": message.get("id")
            }
        )


# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> None:
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


# Periodic cleanup task
async def periodic_cleanup() -> None:
    """Periodically clean up stale connections"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            await connection_manager.cleanup_stale_connections()
            print(f"Cleaned up stale connections. Active: {connection_manager.get_active_connections_count()}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
