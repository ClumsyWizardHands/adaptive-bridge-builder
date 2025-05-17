"""
Agent Card

This module implements a comprehensive Agent Card system that follows the A2A Protocol
standard for agent capabilities, principles, and interaction preferences.
"""

import json
import os
import uuid
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
import logging
from enum import Enum
import semver
import hashlib

# Import communication modules
from communication_adapter import CommunicationAdapter, AgentCapability
from content_handler import ContentHandler, ContentFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentCard")


class CompatibilityLevel(Enum):
    """Levels of compatibility between agents."""
    FULL = "full"              # All capabilities are compatible
    HIGH = "high"              # Most capabilities are compatible
    MEDIUM = "medium"          # Some capabilities are compatible
    LOW = "low"                # Few capabilities are compatible
    INCOMPATIBLE = "incompatible"  # No meaningful compatibility


class AgentCard:
    """
    Represents an agent's capabilities, principles, and interaction preferences.
    
    This class provides methods for:
    1. Loading and managing agent card data
    2. Validating card structure and content
    3. Checking version compatibility
    4. Exposing agent capabilities and principles
    5. Generating card representations in different formats
    """
    
    # A2A Protocol Agent Card schema version
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(
        self, 
        card_path: Optional[str] = None,
        card_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an agent card from a file or data.
        
        Args:
            card_path: Path to the agent card JSON file
            card_data: Direct agent card data as a dictionary
        """
        if card_path and card_data:
            raise ValueError("Provide either card_path or card_data, not both")
            
        self.card_data: Dict[str, Any] = {}
        self.card_hash: str = ""
        self._last_updated: datetime = datetime.utcnow()
        
        if card_path:
            self.load_from_file(card_path)
        elif card_data:
            self.load_from_data(card_data)
        else:
            # Initialize with minimal data
            self.card_data = {
                "agent_id": f"agent-{uuid.uuid4().hex[:8]}",
                "name": "Unnamed Agent",
                "description": "No description provided",
                "version": "0.1.0",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "principles": [],
                "capabilities": [],
                "communication": {
                    "protocols": [],
                    "formats": ["json"],
                    "endpoints": []
                },
                "identity": {
                    "profile": "Unspecified",
                    "key_traits": [],
                    "ethical_boundaries": []
                },
                "performance": {},
                "metadata": {}
            }
        
        # Generate card hash
        self._update_hash()
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load agent card data from a JSON file.
        
        Args:
            file_path: Path to the agent card JSON file
        """
        try:
            with open(file_path, 'r') as f:
                self.card_data = json.load(f)
            self._validate_card()
            self._last_updated = datetime.utcnow()
            self._update_hash()
            logger.info(f"Loaded agent card for {self.card_data.get('name', 'unknown agent')}")
        except Exception as e:
            logger.error(f"Error loading agent card from {file_path}: {str(e)}")
            raise
    
    def load_from_data(self, card_data: Dict[str, Any]) -> None:
        """
        Load agent card data from a dictionary.
        
        Args:
            card_data: Agent card data as a dictionary
        """
        self.card_data = card_data
        self._validate_card()
        self._last_updated = datetime.utcnow()
        self._update_hash()
        logger.info(f"Loaded agent card for {self.card_data.get('name', 'unknown agent')}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save agent card data to a JSON file.
        
        Args:
            file_path: Path to save the agent card JSON file
        """
        try:
            # Update the updated_at timestamp
            self.card_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Make sure parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.card_data, f, indent=2)
                
            logger.info(f"Saved agent card to {file_path}")
        except Exception as e:
            logger.error(f"Error saving agent card to {file_path}: {str(e)}")
            raise
    
    def _validate_card(self) -> None:
        """
        Validate the agent card structure and content.
        
        Raises:
            ValueError: If the card is invalid
        """
        required_fields = ["agent_id", "name", "version"]
        for field in required_fields:
            if field not in self.card_data:
                raise ValueError(f"Agent card missing required field: {field}")
        
        # Ensure created_at and updated_at are present
        if "created_at" not in self.card_data:
            self.card_data["created_at"] = datetime.utcnow().isoformat()
        if "updated_at" not in self.card_data:
            self.card_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Ensure capabilities is a list
        if "capabilities" not in self.card_data:
            self.card_data["capabilities"] = []
        elif not isinstance(self.card_data["capabilities"], list):
            raise ValueError("Agent card capabilities must be a list")
        
        # Ensure communication section exists
        if "communication" not in self.card_data:
            self.card_data["communication"] = {
                "protocols": [],
                "formats": ["json"],
                "endpoints": []
            }
    
    def _update_hash(self) -> None:
        """
        Update the hash of the card data for integrity checking.
        """
        card_json = json.dumps(self.card_data, sort_keys=True)
        self.card_hash = hashlib.sha256(card_json.encode()).hexdigest()
    
    def get_card_hash(self) -> str:
        """
        Get the hash of the card data.
        
        Returns:
            Hash string of the card data
        """
        return self.card_hash
    
    def update_version(
        self, 
        new_version: Optional[str] = None,
        increment: str = "patch"
    ) -> str:
        """
        Update the agent card version.
        
        Args:
            new_version: New version string (semver)
            increment: Increment type if no version provided ('major', 'minor', 'patch')
            
        Returns:
            The new version string
        """
        current_version = self.card_data.get("version", "0.1.0")
        
        if new_version:
            # Validate semver
            if not semver.VersionInfo.isvalid(new_version):
                raise ValueError(f"Invalid version format: {new_version}")
            self.card_data["version"] = new_version
        else:
            # Increment based on specified level
            parsed_version = semver.VersionInfo.parse(current_version)
            if increment == "major":
                new_ver = parsed_version.bump_major()
            elif increment == "minor":
                new_ver = parsed_version.bump_minor()
            else:  # Default to patch
                new_ver = parsed_version.bump_patch()
                
            self.card_data["version"] = str(new_ver)
        
        # Update updated_at timestamp
        self.card_data["updated_at"] = datetime.utcnow().isoformat()
        self._update_hash()
        
        return self.card_data["version"]
    
    def is_compatible_with(self, other_card: 'AgentCard') -> Tuple[bool, CompatibilityLevel, Dict[str, Any]]:
        """
        Check if this agent is compatible with another agent.
        
        Args:
            other_card: The other agent's card to check compatibility with
            
        Returns:
            Tuple of (is_compatible, compatibility_level, details)
        """
        # Check communication protocol compatibility
        my_protocols = set(self.card_data.get("communication", {}).get("protocols", []))
        other_protocols = set(other_card.card_data.get("communication", {}).get("protocols", []))
        
        protocol_overlap = my_protocols.intersection(other_protocols)
        
        # Check communication format compatibility
        my_formats = set(self.card_data.get("communication", {}).get("formats", []))
        other_formats = set(other_card.card_data.get("communication", {}).get("formats", []))
        
        format_overlap = my_formats.intersection(other_formats)
        
        # Check capability compatibility
        my_capabilities = {cap["name"] for cap in self.card_data.get("capabilities", [])}
        other_capabilities = {cap["name"] for cap in other_card.card_data.get("capabilities", [])}
        
        capability_overlap = my_capabilities.intersection(other_capabilities)
        
        # Determine compatibility level
        compatibility_details = {
            "protocol_compatibility": {
                "my_protocols": list(my_protocols),
                "other_protocols": list(other_protocols),
                "compatible_protocols": list(protocol_overlap)
            },
            "format_compatibility": {
                "my_formats": list(my_formats),
                "other_formats": list(other_formats),
                "compatible_formats": list(format_overlap)
            },
            "capability_compatibility": {
                "my_capabilities": list(my_capabilities),
                "other_capabilities": list(other_capabilities),
                "compatible_capabilities": list(capability_overlap)
            }
        }
        
        # Basic compatibility requires at least one common format
        is_compatible = len(format_overlap) > 0
        
        # Determine compatibility level
        if not is_compatible:
            compatibility_level = CompatibilityLevel.INCOMPATIBLE
        else:
            # Calculate percentages
            protocol_percent = len(protocol_overlap) / max(1, len(my_protocols.union(other_protocols)))
            format_percent = len(format_overlap) / max(1, len(my_formats.union(other_formats)))
            capability_percent = len(capability_overlap) / max(1, len(my_capabilities.union(other_capabilities)))
            
            # Overall compatibility score (weighted)
            overall_score = (
                protocol_percent * 0.3 + 
                format_percent * 0.4 + 
                capability_percent * 0.3
            )
            
            if overall_score >= 0.8:
                compatibility_level = CompatibilityLevel.FULL
            elif overall_score >= 0.6:
                compatibility_level = CompatibilityLevel.HIGH
            elif overall_score >= 0.4:
                compatibility_level = CompatibilityLevel.MEDIUM
            else:
                compatibility_level = CompatibilityLevel.LOW
        
        return is_compatible, compatibility_level, compatibility_details
    
    def add_capability(
        self, 
        name: str, 
        description: str, 
        methods: List[str],
        parameters: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a capability to the agent card.
        
        Args:
            name: Name of the capability
            description: Description of the capability
            methods: List of methods implemented by the capability
            parameters: Dictionary of parameter descriptions
        """
        new_capability = {
            "name": name,
            "description": description,
            "methods": methods,
            "parameters": parameters or {}
        }
        
        # Check if capability with this name already exists
        for i, cap in enumerate(self.card_data.get("capabilities", [])):
            if cap["name"] == name:
                # Update existing capability
                self.card_data["capabilities"][i] = new_capability
                logger.info(f"Updated capability: {name}")
                self.card_data["updated_at"] = datetime.utcnow().isoformat()
                self._update_hash()
                return
        
        # Add new capability
        if "capabilities" not in self.card_data:
            self.card_data["capabilities"] = []
            
        self.card_data["capabilities"].append(new_capability)
        logger.info(f"Added new capability: {name}")
        
        # Update timestamp
        self.card_data["updated_at"] = datetime.utcnow().isoformat()
        self._update_hash()
    
    def remove_capability(self, name: str) -> bool:
        """
        Remove a capability from the agent card.
        
        Args:
            name: Name of the capability to remove
            
        Returns:
            True if the capability was removed, False if not found
        """
        if "capabilities" not in self.card_data:
            return False
            
        initial_count = len(self.card_data["capabilities"])
        self.card_data["capabilities"] = [
            cap for cap in self.card_data["capabilities"] 
            if cap["name"] != name
        ]
        
        removed = len(self.card_data["capabilities"]) < initial_count
        
        if removed:
            logger.info(f"Removed capability: {name}")
            self.card_data["updated_at"] = datetime.utcnow().isoformat()
            self._update_hash()
            
        return removed
    
    def add_principle(
        self, 
        name: str, 
        description: str,
        implementation: Optional[str] = None
    ) -> None:
        """
        Add a principle to the agent card.
        
        Args:
            name: Name of the principle
            description: Description of the principle
            implementation: How the principle is implemented
        """
        new_principle = {
            "name": name,
            "description": description
        }
        
        if implementation:
            new_principle["implementation"] = implementation
        
        # Check if principle with this name already exists
        if "principles" in self.card_data:
            for i, princ in enumerate(self.card_data["principles"]):
                if princ["name"] == name:
                    # Update existing principle
                    self.card_data["principles"][i] = new_principle
                    logger.info(f"Updated principle: {name}")
                    self.card_data["updated_at"] = datetime.utcnow().isoformat()
                    self._update_hash()
                    return
        else:
            self.card_data["principles"] = []
            
        # Add new principle
        self.card_data["principles"].append(new_principle)
        logger.info(f"Added new principle: {name}")
        
        # Update timestamp
        self.card_data["updated_at"] = datetime.utcnow().isoformat()
        self._update_hash()
    
    def get_capability(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific capability by name.
        
        Args:
            name: Name of the capability
            
        Returns:
            Capability data or None if not found
        """
        for cap in self.card_data.get("capabilities", []):
            if cap["name"] == name:
                return cap
        return None
    
    def get_principle(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific principle by name.
        
        Args:
            name: Name of the principle
            
        Returns:
            Principle data or None if not found
        """
        for princ in self.card_data.get("principles", []):
            if princ["name"] == name:
                return princ
        return None
    
    def get_communication_config(self) -> Dict[str, Any]:
        """
        Get the agent's communication configuration.
        
        Returns:
            Communication configuration data
        """
        return self.card_data.get("communication", {
            "protocols": [],
            "formats": ["json"],
            "endpoints": []
        })
    
    def update_communication_config(
        self,
        protocols: Optional[List[str]] = None,
        formats: Optional[List[str]] = None,
        endpoints: Optional[List[Dict[str, Any]]] = None,
        authentication: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the agent's communication configuration.
        
        Args:
            protocols: List of supported protocols
            formats: List of supported formats
            endpoints: List of endpoint configurations
            authentication: Authentication configuration
        """
        if "communication" not in self.card_data:
            self.card_data["communication"] = {}
            
        comm = self.card_data["communication"]
        
        if protocols is not None:
            comm["protocols"] = protocols
            
        if formats is not None:
            comm["formats"] = formats
            
        if endpoints is not None:
            comm["endpoints"] = endpoints
            
        if authentication is not None:
            comm["authentication"] = authentication
            
        # Update timestamp
        self.card_data["updated_at"] = datetime.utcnow().isoformat()
        self._update_hash()
        
        logger.info(f"Updated communication configuration")
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert agent card to a JSON string.
        
        Args:
            pretty: Whether to format the JSON with indentation
            
        Returns:
            JSON string representation of the agent card
        """
        if pretty:
            return json.dumps(self.card_data, indent=2)
        return json.dumps(self.card_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get agent card as a dictionary.
        
        Returns:
            Dictionary representation of the agent card
        """
        return self.card_data.copy()
    
    def generate_capabilities_map(self) -> Dict[str, List[str]]:
        """
        Generate a map of capabilities to methods.
        
        Returns:
            Dictionary mapping capability names to method lists
        """
        cap_map = {}
        for cap in self.card_data.get("capabilities", []):
            cap_map[cap["name"]] = cap.get("methods", [])
        return cap_map
    
    def check_supports_capability(self, capability_name: str, method: Optional[str] = None) -> bool:
        """
        Check if the agent supports a specific capability and optionally a method.
        
        Args:
            capability_name: Name of the capability
            method: Optional method name to check
            
        Returns:
            True if the capability and method are supported
        """
        for cap in self.card_data.get("capabilities", []):
            if cap["name"] == capability_name:
                if method is None:
                    return True
                return method in cap.get("methods", [])
        return False
    
    def check_supports_format(self, format_name: str) -> bool:
        """
        Check if the agent supports a specific communication format.
        
        Args:
            format_name: Name of the format
            
        Returns:
            True if the format is supported
        """
        formats = self.card_data.get("communication", {}).get("formats", [])
        return format_name in formats
    
    def get_agent_id(self) -> str:
        """Get the agent ID."""
        return self.card_data.get("agent_id", "")
    
    def get_name(self) -> str:
        """Get the agent name."""
        return self.card_data.get("name", "")
    
    def get_version(self) -> str:
        """Get the agent version."""
        return self.card_data.get("version", "")
    
    def get_description(self) -> str:
        """Get the agent description."""
        return self.card_data.get("description", "")
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get all agent capabilities."""
        return self.card_data.get("capabilities", [])


class AgentRegistry:
    """
    Registry for discovering and tracking other agents.
    
    This class provides methods for:
    1. Discovering other agents through various mechanisms
    2. Storing and retrieving agent cards
    3. Monitoring agent status and availability
    4. Finding compatible agents for specific tasks
    5. Tracking compatibility and communication history
    """
    
    def __init__(self, storage_dir: str = "agent_registry"):
        """
        Initialize the agent registry.
        
        Args:
            storage_dir: Directory for storing agent cards
        """
        self.storage_dir = storage_dir
        self.agents: Dict[str, AgentCard] = {}
        self.compatibility_cache: Dict[Tuple[str, str], Tuple[bool, CompatibilityLevel, Dict[str, Any]]] = {}
        self.communication_adapter = CommunicationAdapter()
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load any existing agent cards
        self._load_existing_cards()
        
        logger.info(f"Agent registry initialized with {len(self.agents)} agents")
    
    def _load_existing_cards(self) -> None:
        """Load existing agent cards from storage directory."""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        agent_card = AgentCard(card_path=file_path)
                        agent_id = agent_card.get_agent_id()
                        if agent_id:
                            self.agents[agent_id] = agent_card
                    except Exception as e:
                        logger.warning(f"Failed to load agent card from {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading existing agent cards: {str(e)}")
    
    def register_agent(self, agent_card: AgentCard) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent_card: Agent card to register
        """
        agent_id = agent_card.get_agent_id()
        if not agent_id:
            raise ValueError("Agent card missing agent_id")
            
        # Store in memory
        self.agents[agent_id] = agent_card
        
        # Store to disk
        file_path = os.path.join(self.storage_dir, f"{agent_id}.json")
        agent_card.save_to_file(file_path)
        
        # Register with communication adapter if available
        try:
            self._register_communication_capabilities(agent_card)
        except Exception as e:
            logger.warning(f"Failed to register communication capabilities: {str(e)}")
        
        logger.info(f"Registered agent: {agent_card.get_name()} (ID: {agent_id})")
    
    def _register_communication_capabilities(self, agent_card: AgentCard) -> None:
        """
        Register agent's communication capabilities with the adapter.
        
        Args:
            agent_card: Agent card to register
        """
        agent_id = agent_card.get_agent_id()
        
        # Map communication formats to capabilities
        capabilities = []
        
        # Get communication formats
        formats = agent_card.card_data.get("communication", {}).get("formats", [])
        
        # Add base capability
        capabilities.append(AgentCapability.TEXT_ONLY)
        
        # Map formats to capabilities
        for fmt in formats:
            if fmt.lower() == "json":
                capabilities.append(AgentCapability.JSON)
            elif fmt.lower() == "markdown" or fmt.lower() == "md":
                capabilities.append(AgentCapability.MARKDOWN)
            elif fmt.lower() == "html":
                capabilities.append(AgentCapability.HTML)
            
        # Check for binary capabilities
        binary_capabilities = []
        for cap in agent_card.get_capabilities():
            if "file" in cap["name"].lower() or "binary" in cap["name"].lower():
                binary_capabilities.append(cap["name"])
                
        if binary_capabilities:
            capabilities.append(AgentCapability.BINARY)
            
        # Check for media capabilities
        for cap in agent_card.get_capabilities():
            if "image" in cap["name"].lower():
                capabilities.append(AgentCapability.IMAGES)
            if "audio" in cap["name"].lower():
                capabilities.append(AgentCapability.AUDIO)
            if "video" in cap["name"].lower():
                capabilities.append(AgentCapability.VIDEO)
        
        # Register with communication adapter
        self.communication_adapter.register_agent(agent_id, capabilities)
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if the agent was unregistered
        """
        if agent_id not in self.agents:
            return False
            
        # Remove from memory
        del self.agents[agent_id]
        
        # Remove from disk
        file_path = os.path.join(self.storage_dir, f"{agent_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Clear from compatibility cache
        keys_to_remove = []
        for key in self.compatibility_cache:
            if agent_id in key:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.compatibility_cache[key]
            
        logger.info(f"Unregistered agent: {agent_id}")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentCard]:
        """
        Get an agent card by ID.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent card or None if not found
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, str]]:
        """
        List all registered agents.
        
        Returns:
            List of agent summaries
        """
        return [
            {
                "agent_id": agent.get_agent_id(),
                "name": agent.get_name(),
                "version": agent.get_version(),
                "description": agent.get_description()
            }
            for agent in self.agents.values()
        ]
    
    def discover_agents_http(
        self, 
        discovery_endpoint: str,
        headers: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Discover agents through an HTTP discovery endpoint.
        
        Args:
            discovery_endpoint: URL of the discovery endpoint
            headers: Optional HTTP headers
            
        Returns:
            List of discovered agent IDs
        """
        try:
            response = requests.get(discovery_endpoint, headers=headers or {})
            response.raise_for_status()
            
            data = response.json()
            discovered_ids = []
            
            # Process discovered agent data
            if isinstance(data, list):
                # List of agent data
                for agent_data in data:
                    if isinstance(agent_data, dict) and "agent_id" in agent_data:
                        agent_card = AgentCard(card_data=agent_data)
                        self.register_agent(agent_card)
                        discovered_ids.append(agent_card.get_agent_id())
            elif isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
                # Container with agents list
                for agent_data in data["agents"]:
                    if isinstance(agent_data, dict) and "agent_id" in agent_data:
                        agent_card = AgentCard(card_data=agent_data)
                        self.register_agent(agent_card)
                        discovered_ids.append(agent_card.get_agent_id())
            
            logger.info(f"Discovered {len(discovered_ids)} agents from {discovery_endpoint}")
            return discovered_ids
            
        except Exception as e:
            logger.error(f"Error discovering agents from {discovery_endpoint}: {str(e)}")
            return []
    
    def discover_agents_file(self, directory: str) -> List[str]:
        """
        Discover agents from a directory of agent card files.
        
        Args:
            directory: Directory containing agent card files
            
        Returns:
            List of discovered agent IDs
        """
        discovered_ids = []
        
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                return []
                
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    file_path = os.path.join(directory, filename)
                    try:
                        agent_card = AgentCard(card_path=file_path)
                        self.register_agent(agent_card)
                        discovered_ids.append(agent_card.get_agent_id())
                    except Exception as e:
                        logger.warning(f"Failed to load agent card from {file_path}: {str(e)}")
            
            logger.info(f"Discovered {len(discovered_ids)} agents from {directory}")
            return discovered_ids
            
        except Exception as e:
            logger.error(f"Error discovering agents from {directory}: {str(e)}")
            return []
    
    def find_compatible_agents(
        self, 
        my_agent: AgentCard,
        required_capability: Optional[str] = None,
        min_compatibility: CompatibilityLevel = CompatibilityLevel.MEDIUM
    ) -> List[Dict[str, Any]]:
        """
        Find agents compatible with a given agent.
        
        Args:
            my_agent: The agent to find compatibility with
            required_capability: Optional capability that must be supported
            min_compatibility: Minimum compatibility level
            
        Returns:
            List of compatible agent summaries with compatibility details
        """
        compatible_agents = []
        my_agent_id = my_agent.get_agent_id()
        
        for agent_id, agent_card in self.agents.items():
            # Skip self
            if agent_id == my_agent_id:
                continue
                
            # Check if this agent has the required capability if specified
            if required_capability and not agent_card.check_supports_capability(required_capability):
                continue
                
            # Check compatibility
            cache_key = (my_agent_id, agent_id)
            if cache_key in self.compatibility_cache:
                is_compatible, compatibility_level, details = self.compatibility_cache[cache_key]
            else:
                is_compatible, compatibility_level, details = my_agent.is_compatible_with(agent_card)
                self.compatibility_cache[cache_key] = (is_compatible, compatibility_level, details)
                
            # If not compatible or below minimum compatibility level, skip
            if not is_compatible or compatibility_level.value < min_compatibility.value:
                continue
                
            # Add to compatible agents
            compatible_agents.append({
                "agent_id": agent_id,
                "name": agent_card.get_name(),
                "version": agent_card.get_version(),
                "description": agent_card.get_description(),
                "compatibility_level": compatibility_level.value,
                "compatibility_details": details
            })
            
        # Sort by compatibility level (highest first)
        compatible_agents.sort(
            key=lambda a: {
                "full": 0,
                "high": 1,
                "medium": 2,
                "low": 3
            }.get(a["compatibility_level"], 4)
        )
        
        return compatible_agents
        
    def find_agent_by_capability(self, capability_name: str) -> List[Dict[str, Any]]:
        """
        Find agents that support a specific capability.
        
        Args:
            capability_name: Name of the capability to look for
            
        Returns:
            List of agent summaries with the capability
        """
        matching_agents = []
        
        for agent_id, agent_card in self.agents.items():
            if agent_card.check_supports_capability(capability_name):
                matching_agents.append({
                    "agent_id": agent_id,
                    "name": agent_card.get_name(),
                    "version": agent_card.get_version(),
                    "description": agent_card.get_description(),
                    "capability": agent_card.get_capability(capability_name)
                })
                
        return matching_agents
        
    def get_communication_protocol(
        self, 
        sender_id: str, 
        recipient_id: str
    ) -> Dict[str, Any]:
        """
        Determine the optimal communication protocol between two agents.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            
        Returns:
            Protocol settings dictionary
        """
        sender = self.get_agent(sender_id)
        recipient = self.get_agent(recipient_id)
        
        if not sender or not recipient:
            raise ValueError("Sender or recipient agent not found")
            
        # Use communication adapter to determine protocol
        return self.communication_adapter.determine_protocol(sender_id, recipient_id)
        
    def update_agent_capabilities(
        self, 
        agent_id: str, 
        capabilities: List[Dict[str, Any]]
    ) -> bool:
        """
        Update an agent's capabilities in the registry.
        
        Args:
            agent_id: ID of the agent
            capabilities: List of capability dictionaries
            
        Returns:
            True if successful
        """
        if agent_id not in self.agents:
            return False
            
        agent_card = self.agents[agent_id]
        
        # Clear existing capabilities
        agent_card.card_data["capabilities"] = []
        
        # Add new capabilities
        for cap in capabilities:
            agent_card.add_capability(
                name=cap["name"],
                description=cap.get("description", ""),
                methods=cap.get("methods", []),
                parameters=cap.get("parameters", {})
            )
            
        # Update cache
        for key in list(self.compatibility_cache.keys()):
            if agent_id in key:
                del self.compatibility_cache[key]
                
        # Save to disk
        file_path = os.path.join(self.storage_dir, f"{agent_id}.json")
        agent_card.save_to_file(file_path)
        
        # Update communication adapter
        self._register_communication_capabilities(agent_card)
        
        return True
