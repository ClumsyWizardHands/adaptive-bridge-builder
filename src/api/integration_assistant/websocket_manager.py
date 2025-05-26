"""
WebSocket Connection Manager

Manages WebSocket connections for real-time updates between agents and the integration assistant.
"""

import asyncio
import json
import uuid
from typing import Dict, Set, Optional
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
from .models import WebSocketMessage, ConnectionState, ConnectionStatus


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self) -> None:
        # Active connections by client_id
        self.active_connections: Dict[str, WebSocket] = {}
        # Agent associations by client_id
        self.agent_associations: Dict[str, str] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, ConnectionStatus] = {}
        # Subscriptions: agent_id -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection
        
        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID, generates one if not provided
            
        Returns:
            The client ID for this connection
        """
        await websocket.accept()
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
            
        async with self._lock:
            # Store connection
            self.active_connections = {**self.active_connections, client_id: websocket}
            
            # Initialize connection metadata
            self.connection_metadata = {**self.connection_metadata, client_id: ConnectionStatus(
                client_id=client_id,
                state=ConnectionState.CONNECTED,
                connected_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )}
        
        # Send connection confirmation
        await self.send_personal_message(
            client_id,
            {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return client_id
        
    async def disconnect(self, client_id: str) -> None:
        """
        Handle WebSocket disconnection
        
        Args:
            client_id: The client ID to disconnect
        """
        async with self._lock:
            # Remove from active connections
            if client_id in self.active_connections:
                self.active_connections = {k: v for k, v in self.active_connections.items() if k != client_id}
                
            # Update connection metadata
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id].state = ConnectionState.DISCONNECTED
                
            # Remove from agent associations
            if client_id in self.agent_associations:
                agent_id = self.agent_associations[client_id]
                self.agent_associations = {k: v for k, v in self.agent_associations.items() if k != client_id}
                
                # Remove from subscriptions
                if agent_id in self.subscriptions:
                    self.subscriptions[agent_id].discard(client_id)
                    if not self.subscriptions[agent_id]:
                        self.subscriptions = {k: v for k, v in self.subscriptions.items() if k != agent_id}
                    
    async def associate_agent(self, client_id: str, agent_id: str) -> None:
        """
        Associate a client with an agent
        
        Args:
            client_id: The client ID
            agent_id: The agent ID to associate
        """
        async with self._lock:
            self.agent_associations = {**self.agent_associations, client_id: agent_id}
            
            # Add to subscriptions
            if agent_id not in self.subscriptions:
                self.subscriptions = {**self.subscriptions, agent_id: set()}
            self.subscriptions[agent_id].add(client_id)
            
            # Update connection metadata
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id].agent_id = agent_id
                self.connection_metadata[client_id].last_activity = datetime.utcnow()
            
        # Notify client of association
        await self.send_personal_message(
            client_id,
            {
                "type": "agent_associated",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    async def send_personal_message(self, client_id: str, message: Dict) -> None:
        """
        Send a message to a specific client
        
        Args:
            client_id: The client ID to send to
            message: The message data to send
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                # Create WebSocket message
                ws_message = WebSocketMessage(
                    type=message.get("type", "message"),
                    client_id=client_id,
                    agent_id=self.agent_associations.get(client_id),
                    data=message,
                    timestamp=datetime.utcnow()
                )
                
                # Send message
                await websocket.send_json(ws_message.dict())
                
                # Update last activity
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id].last_activity = datetime.utcnow()
                    
            except Exception as e:
                # Handle send errors
                print(f"Error sending message to {client_id}: {e}")
                await self.handle_connection_error(client_id, str(e))
                
    async def broadcast_to_agent(self, agent_id: str, message: Dict) -> None:
        """
        Broadcast a message to all clients associated with an agent
        
        Args:
            agent_id: The agent ID to broadcast to
            message: The message data to broadcast
        """
        if agent_id in self.subscriptions:
            # Send to all subscribed clients
            for client_id in self.subscriptions[agent_id]:
                await self.send_personal_message(client_id, message)
                
    async def broadcast_all(self, message: Dict) -> None:
        """
        Broadcast a message to all connected clients
        
        Args:
            message: The message data to broadcast
        """
        # Create tasks for all sends
        tasks = []
        for client_id in self.active_connections:
            tasks.append(self.send_personal_message(client_id, message))
            
        # Execute all sends concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def handle_connection_error(self, client_id: str, error_message: str) -> None:
        """
        Handle connection errors
        
        Args:
            client_id: The client ID with the error
            error_message: The error message
        """
        # Update connection metadata
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id].state = ConnectionState.ERROR
            self.connection_metadata[client_id].error_message = error_message
            
        # Disconnect the client
        await self.disconnect(client_id)
        
    async def send_update(self, client_id: str, update_type: str, data: Dict) -> None:
        """
        Send a specific type of update to a client
        
        Args:
            client_id: The client ID to send to
            update_type: The type of update
            data: The update data
        """
        message = {
            "type": f"update_{update_type}",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_personal_message(client_id, message)
        
    async def ping_client(self, client_id: str) -> bool:
        """
        Ping a client to check if connection is alive
        
        Args:
            client_id: The client ID to ping
            
        Returns:
            True if client responded, False otherwise
        """
        if client_id not in self.active_connections:
            return False
            
        try:
            await self.send_personal_message(
                client_id,
                {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
            )
            return True
        except:
            return False
            
    async def get_connection_status(self, client_id: str) -> Optional[ConnectionStatus]:
        """
        Get the connection status for a client
        
        Args:
            client_id: The client ID
            
        Returns:
            The connection status or None if not found
        """
        return self.connection_metadata.get(client_id)
        
    async def cleanup_stale_connections(self, timeout_seconds: int = 300) -> None:
        """
        Clean up stale connections that haven't been active
        
        Args:
            timeout_seconds: Inactivity timeout in seconds
        """
        current_time = datetime.utcnow()
        stale_clients = []
        
        for client_id, status in self.connection_metadata.items():
            if status.last_activity:
                time_diff = (current_time - status.last_activity).total_seconds()
                if time_diff > timeout_seconds:
                    stale_clients.append(client_id)
                    
        # Disconnect stale clients
        for client_id in stale_clients:
            await self.disconnect(client_id)
            
    def get_active_connections_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
        
    def get_agent_connections_count(self, agent_id: str) -> int:
        """Get the number of connections for a specific agent"""
        return len(self.subscriptions.get(agent_id, set()))
