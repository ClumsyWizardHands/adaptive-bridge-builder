"""
Streaming A2A Adapter for Empire Framework

This module provides streaming capabilities for Empire Framework components,
enabling real-time component updates and changes to be streamed over the A2A Protocol
using Server-Sent Events (SSE).
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union
from datetime import datetime, timezone

# Import the standard A2A adapter
from .a2a_adapter import A2AAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StreamingA2AAdapter")

class StreamingA2AAdapter:
    """
    Extends the A2A Adapter with streaming capabilities.
    
    This adapter enables Empire Framework components to be streamed over the
    A2A Protocol using Server-Sent Events (SSE), facilitating real-time updates
    and changes to components.
    """
    
    def __init__(self, base_adapter: Optional[A2AAdapter] = None) -> None:
        """
        Initialize the streaming adapter.
        
        Args:
            base_adapter: Optional base A2A adapter to delegate non-streaming operations to
        """
        self.base_adapter = base_adapter or A2AAdapter(validation_enabled=True)
        logger.info("Streaming A2A Adapter initialized")
    
    async def stream_component_updates(
        self, 
        component_id: str,
        update_callback: Callable[[], Optional[Dict[str, Any]]],
        interval: float = 1.0,
        max_updates: Optional[int] = None,
        include_metadata: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream updates to a component over time.
        
        Args:
            component_id: ID of the component to stream updates for
            update_callback: Function that returns the updated component or None if no update
            interval: Time between update checks in seconds
            max_updates: Maximum number of updates to stream (None for unlimited)
            include_metadata: Whether to include metadata in streamed events
            
        Yields:
            SSE-formatted messages containing component updates
        """
        update_count = 0
        last_update = None
        
        while max_updates is None or update_count < max_updates:
            # Get the current component state
            current_update = update_callback()
            
            # If no update or same as last update, wait and continue
            if current_update is None or current_update == last_update:
                await asyncio.sleep(interval)
                continue
                
            # Convert component to A2A message
            a2a_message = self.base_adapter.component_to_a2a_message(
                current_update, 
                include_metadata=include_metadata
            )
            
            # Add streaming metadata
            a2a_message["streaming_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "update_number": update_count + 1,
                "component_id": component_id
            }
            
            # Format as SSE event
            event_data = self._format_sse_event(
                event="component_update",
                data=a2a_message
            )
            
            # Update tracking variables
            last_update = current_update
            update_count += 1
            
            # Yield the event
            yield event_data
            
            # Wait for next interval
            await asyncio.sleep(interval)
    
    async def stream_component_batch(
        self,
        components: List[Dict[str, Any]],
        chunk_size: int = 5,
        delay: float = 0.5,
        include_metadata: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a batch of components in chunks.
        
        This is useful for efficiently transferring large sets of components.
        
        Args:
            components: List of components to stream
            chunk_size: Number of components per chunk
            delay: Delay between chunks in seconds
            include_metadata: Whether to include metadata in streamed events
            
        Yields:
            SSE-formatted messages containing component chunks
        """
        total_components = len(components)
        total_chunks = (total_components + chunk_size - 1) // chunk_size
        
        for chunk_index in range(total_chunks):
            # Get current chunk
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, total_components)
            current_chunk = components[start_idx:end_idx]
            
            # Convert to A2A batch
            batch_metadata = {
                "source": "empire_framework",
                "streaming": True,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            a2a_batch = self.base_adapter.components_to_a2a_batch(
                current_chunk, 
                batch_metadata
            )
            
            # Format as SSE event
            event_data = self._format_sse_event(
                event="component_batch",
                data=a2a_batch
            )
            
            # Yield the event
            yield event_data
            
            # Wait before next chunk (except last chunk)
            if chunk_index < total_chunks - 1:
                await asyncio.sleep(delay)
    
    async def stream_component_changes(
        self,
        component_id: str,
        get_component: Callable[[], Dict[str, Any]],
        interval: float = 1.0,
        max_duration: Optional[float] = 300.0,  # 5 minutes default
        include_metadata: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream only the changes to a component.
        
        Detects changes in a component and streams only the differences,
        reducing bandwidth usage.
        
        Args:
            component_id: ID of the component to stream changes for
            get_component: Function that returns the current component state
            interval: Time between change checks in seconds
            max_duration: Maximum duration to stream changes (None for unlimited)
            include_metadata: Whether to include metadata in streamed events
            
        Yields:
            SSE-formatted messages containing component changes
        """
        start_time = time.time()
        previous_state = get_component()
        
        while max_duration is None or (time.time() - start_time) < max_duration:
            # Get current state
            current_state = get_component()
            
            # Calculate diff
            changes = self._calculate_component_diff(previous_state, current_state)
            
            # If changes detected
            if changes:
                # Format diff as A2A message
                diff_message = {
                    "a2a_version": "1.0",
                    "message_type": "empire.component_diff",
                    "content": {
                        "component_id": component_id,
                        "component_type": current_state.get("type", "unknown"),
                        "changes": changes
                    },
                    "metadata": {
                        "source": "empire_framework",
                        "timestamp": datetime.utcnow().isoformat(),
                        "previous_hash": self._calculate_state_hash(previous_state),
                        "current_hash": self._calculate_state_hash(current_state)
                    }
                }
                
                # Format as SSE event
                event_data = self._format_sse_event(
                    event="component_diff",
                    data=diff_message
                )
                
                # Yield the event
                yield event_data
                
                # Update previous state
                previous_state = current_state
                
            # Wait for next interval
            await asyncio.sleep(interval)
    
    def _format_sse_event(self, event: str, data: Any) -> Dict[str, str]:
        """
        Format data as a Server-Sent Event.
        
        Args:
            event: Event name
            data: Event data
            
        Returns:
            Formatted SSE event
        """
        # Convert data to JSON string
        data_str = json.dumps(data)
        
        # Create SSE event
        return {
            "event": event,
            "data": data_str,
            "id": f"event-{time.time_ns()}",
            "retry": "5000"  # 5 second retry interval
        }
    
    def _calculate_component_diff(
        self, 
        old_component: Dict[str, Any], 
        new_component: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate changes between component versions.
        
        Args:
            old_component: Previous component state
            new_component: Current component state
            
        Returns:
            Dictionary of changes
        """
        changes = {}
        
        # Check all fields in new component
        for key, value in new_component.items():
            # If key not in old component or value changed
            if key not in old_component or old_component[key] != value:
                changes[key] = {
                    "new": value,
                    "old": old_component.get(key, None) if key in old_component else None
                }
                
        # Check for deleted fields
        for key in old_component:
            if key not in new_component:
                changes[key] = {
                    "new": None,
                    "old": old_component[key],
                    "deleted": True
                }
                
        return changes
    
    def _calculate_state_hash(self, component: Dict[str, Any]) -> str:
        """
        Calculate a hash for a component state.
        
        Args:
            component: Component to hash
            
        Returns:
            Hash string
        """
        import hashlib
        
        # Convert to ordered JSON string
        json_str = json.dumps(component, sort_keys=True)
        
        # Calculate hash
        return hashlib.sha256(json_str.encode()).hexdigest()


class StreamingEvent:
    """
    Represents a streaming event in the A2A Protocol.
    
    This class models the structure of a streaming event and provides
    methods for serialization and deserialization.
    """
    
    def __init__(
        self,
        event_type: str,
        data: Any,
        event_id: Optional[str] = None,
        retry: Optional[int] = None
    ):
        """
        Initialize a streaming event.
        
        Args:
            event_type: Type of the event
            data: Event data
            event_id: Optional unique ID for the event
            retry: Optional retry interval in milliseconds
        """
        self.event_type = event_type
        self.data = data
        self.event_id = event_id or f"event-{time.time_ns()}"
        self.retry = retry or 5000  # Default 5 second retry
    
    def to_sse_format(self) -> str:
        """
        Convert to Server-Sent Events format.
        
        Returns:
            String in SSE format
        """
        # Format data as JSON
        data_json = json.dumps(self.data)
        
        # Build SSE message
        sse_parts = [
            f"event: {self.event_type}",
            f"id: {self.event_id}",
            f"retry: {self.retry}",
            f"data: {data_json}",
            "\n"  # End with blank line
        ]
        
        return "\n".join(sse_parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamingEvent':
        """
        Create a StreamingEvent from a dictionary.
        
        Args:
            data: Dictionary with event data
            
        Returns:
            StreamingEvent instance
        """
        return cls(
            event_type=data.get("event", "message"),
            data=data.get("data"),
            event_id=data.get("id"),
            retry=data.get("retry")
        )
