"""
Agent Card Example

This module demonstrates how to use the AgentCard and AgentRegistry classes
to manage agent capabilities, discover other agents, and adapt communications.
"""

import os
import json
from datetime import datetime

from agent_card import AgentCard, AgentRegistry, CompatibilityLevel
from communication_adapter import CommunicationAdapter, AgentCapability
from content_handler import ContentHandler

def load_existing_card_example():
    """Example of loading and working with an existing agent card."""
    print("\n=== Loading Existing Agent Card ===\n")
    
    # Load existing card from JSON file
    agent_card = AgentCard(card_path="src/agent_card.json")
    
    # Display basic information
    print(f"Agent ID: {agent_card.get_agent_id()}")
    print(f"Name: {agent_card.get_name()}")
    print(f"Version: {agent_card.get_version()}")
    print(f"Description: {agent_card.get_description()}")
    
    # List capabilities
    print("\nCapabilities:")
    for capability in agent_card.get_capabilities():
        print(f"  - {capability['name']}: {capability['description']}")
        if "methods" in capability:
            print(f"    Methods: {', '.join(capability['methods'])}")
    
    # Check for specific capability
    has_discovery = agent_card.check_supports_capability("agent_discovery")
    print(f"\nSupports agent discovery: {has_discovery}")
    
    # Check for specific method
    supports_getAgentCard = agent_card.check_supports_capability("agent_discovery", "getAgentCard")
    print(f"Supports getAgentCard method: {supports_getAgentCard}")
    
    # Get communication configuration
    comm_config = agent_card.get_communication_config()
    print(f"\nSupported protocols: {', '.join(comm_config.get('protocols', []))}")
    print(f"Supported formats: {', '.join(comm_config.get('formats', []))}")
    
    # Update the card version
    old_version = agent_card.get_version()
    new_version = agent_card.update_version(increment="patch")
    print(f"\nUpdated version from {old_version} to {new_version}")
    
    # Save the updated card to a new file
    agent_card.save_to_file("example_output/updated_agent_card.json")
    print(f"Saved updated card to example_output/updated_agent_card.json")


def create_new_card_example():
    """Example of creating a new agent card from scratch."""
    print("\n=== Creating New Agent Card ===\n")
    
    # Create a new agent card
    new_agent = AgentCard()
    
    # Update basic information
    new_agent.card_data["agent_id"] = "assistant-007"
    new_agent.card_data["name"] = "Research Assistant"
    new_agent.card_data["description"] = "An agent specialized in research tasks and data analysis"
    
    # Add capabilities
    new_agent.add_capability(
        name="research",
        description="Perform research on a given topic",
        methods=["searchWeb", "analyzeSources", "summarizeFindings"],
        parameters={
            "topic": "string - the research topic",
            "depth": "number - research depth (1-5)",
            "format": "string - output format (summary, report, outline)"
        }
    )
    
    new_agent.add_capability(
        name="data_analysis",
        description="Analyze data sets and generate insights",
        methods=["analyzeData", "generateCharts", "identifyTrends"],
        parameters={
            "data": "object - the data to analyze",
            "analysis_type": "string - type of analysis to perform"
        }
    )
    
    # Add principles
    new_agent.add_principle(
        name="Research Integrity",
        description="Maintain high standards of research integrity by citing sources and avoiding misrepresentation.",
        implementation="All research results include source citations and confidence levels."
    )
    
    new_agent.add_principle(
        name="Data Privacy",
        description="Respect privacy in data handling and analysis.",
        implementation="Personal identifiers are automatically removed from analyzed data."
    )
    
    # Update communication configuration
    new_agent.update_communication_config(
        protocols=["a2a", "rest-api"],
        formats=["json", "markdown", "html"],
        endpoints=[
            {
                "type": "http",
                "url": "https://api.example.com/research-assistant",
                "methods": ["POST", "GET"]
            }
        ],
        authentication={
            "required": True,
            "methods": ["api-key", "oauth2"]
        }
    )
    
    # Print the resulting card as JSON
    print(f"New agent card created:")
    print(new_agent.to_json(pretty=True))
    
    # Save to file
    os.makedirs("example_output", exist_ok=True)
    new_agent.save_to_file("example_output/research_assistant_card.json")
    print(f"Saved new card to example_output/research_assistant_card.json")


def agent_registry_example():
    """Example of using the agent registry to discover and manage agents."""
    print("\n=== Agent Registry Example ===\n")
    
    # Initialize registry
    registry = AgentRegistry(storage_dir="example_output/agent_registry")
    
    # Create some example agents
    basic_agent = AgentCard()
    basic_agent.card_data["agent_id"] = "basic-001"
    basic_agent.card_data["name"] = "Basic Agent"
    basic_agent.card_data["description"] = "A simple agent with minimal capabilities"
    basic_agent.update_communication_config(
        formats=["text"]
    )
    
    advanced_agent = AgentCard()
    advanced_agent.card_data["agent_id"] = "advanced-001"
    advanced_agent.card_data["name"] = "Advanced Agent"
    advanced_agent.card_data["description"] = "An advanced agent with multiple capabilities"
    advanced_agent.update_communication_config(
        protocols=["a2a", "json-rpc"],
        formats=["json", "markdown", "html"]
    )
    advanced_agent.add_capability(
        name="file_processing",
        description="Process different file types",
        methods=["readFile", "parseFile", "convertFormat"]
    )
    
    # Register agents
    registry.register_agent(basic_agent)
    registry.register_agent(advanced_agent)
    
    # Load and register the existing agent card
    bridge_agent = AgentCard(card_path="src/agent_card.json")
    registry.register_agent(bridge_agent)
    
    # List all registered agents
    print("Registered agents:")
    for agent in registry.list_agents():
        print(f"  - {agent['name']} (ID: {agent['agent_id']}, Version: {agent['version']})")
    
    # Find compatible agents
    print("\nFinding agents compatible with the Bridge Agent:")
    compatibles = registry.find_compatible_agents(
        my_agent=bridge_agent,
        min_compatibility=CompatibilityLevel.LOW
    )
    
    for agent in compatibles:
        print(f"  - {agent['name']} (Compatibility: {agent['compatibility_level']})")
        formats = agent['compatibility_details']['format_compatibility']['compatible_formats']
        print(f"    Compatible formats: {formats}")
    
    # Find agents with a specific capability
    print("\nFinding agents with file processing capability:")
    file_processors = registry.find_agent_by_capability("file_processing")
    for agent in file_processors:
        print(f"  - {agent['name']}")
        print(f"    Methods: {agent['capability']['methods']}")
    
    # Get communication protocol between agents
    print("\nDetermining optimal communication protocol:")
    protocol = registry.get_communication_protocol(
        sender_id=bridge_agent.get_agent_id(),
        recipient_id=advanced_agent.get_agent_id()
    )
    print(f"  Format: {protocol['format']}")
    print(f"  Interactive: {protocol['interactive']}")
    print(f"  Chunking enabled: {protocol['chunking']}")


def adaptive_communication_example():
    """Example of adapting communication based on agent capabilities."""
    print("\n=== Adaptive Communication Example ===\n")
    
    # Initialize registry and load agents
    registry = AgentRegistry(storage_dir="example_output/agent_registry")
    
    # Create agents with different capabilities
    text_only_agent = AgentCard()
    text_only_agent.card_data["agent_id"] = "text-only-001"
    text_only_agent.card_data["name"] = "Text-Only Agent"
    text_only_agent.update_communication_config(formats=["text"])
    registry.register_agent(text_only_agent)
    
    json_agent = AgentCard()
    json_agent.card_data["agent_id"] = "json-001"
    json_agent.card_data["name"] = "JSON Agent"
    json_agent.update_communication_config(formats=["json", "text"])
    registry.register_agent(json_agent)
    
    markdown_agent = AgentCard()
    markdown_agent.card_data["agent_id"] = "markdown-001"
    markdown_agent.card_data["name"] = "Markdown Agent"
    markdown_agent.update_communication_config(formats=["markdown", "text"])
    registry.register_agent(markdown_agent)
    
    # Create a content handler for processing content
    content_handler = ContentHandler()
    
    # Create example data to share
    data = {
        "title": "Quarterly Report",
        "date": "2025-Q1",
        "metrics": {
            "revenue": 1250000,
            "growth": 12.5,
            "customers": 5280
        },
        "highlights": [
            "Expanded to 3 new markets",
            "Launched 2 new product lines",
            "Reduced customer churn by 15%"
        ]
    }
    
    # Define how we want to present this to different agents
    recipients = ["text-only-001", "json-001", "markdown-001"]
    
    print("Adapting the same content for different agents:")
    
    for recipient_id in recipients:
        recipient = registry.get_agent(recipient_id)
        print(f"\nAdapting content for: {recipient.get_name()}")
        
        # Determine optimal communication format
        protocol = registry.get_communication_protocol(
            sender_id="advanced-001",  # Using as sender
            recipient_id=recipient_id
        )
        format_type = protocol["format"]
        print(f"  Using format: {format_type}")
        
        # Adapt content based on format
        if format_type == "json":
            # JSON agent can receive structured data directly
            message = {
                "type": "report",
                "content": data,
                "requires_response": False
            }
            print(f"  Sending structured JSON data")
            print(f"  Sample: {json.dumps(message)[:100]}...")
            
        elif format_type == "markdown":
            # Markdown agent gets formatted markdown
            markdown_content = f"""
# {data['title']} ({data['date']})

## Key Metrics
- Revenue: ${data['metrics']['revenue']:,}
- Growth: {data['metrics']['growth']}%
- Customers: {data['metrics']['customers']:,}

## Highlights
"""
            for highlight in data["highlights"]:
                markdown_content += f"- {highlight}\n"
                
            message = markdown_content
            print(f"  Sending markdown-formatted content")
            print(f"  Sample: {message[:100]}...")
            
        else:
            # Text-only agent gets plain text
            text_content = f"{data['title']} ({data['date']})\n\n"
            text_content += "KEY METRICS:\n"
            text_content += f"Revenue: ${data['metrics']['revenue']:,}\n"
            text_content += f"Growth: {data['metrics']['growth']}%\n"
            text_content += f"Customers: {data['metrics']['customers']:,}\n\n"
            text_content += "HIGHLIGHTS:\n"
            for highlight in data["highlights"]:
                text_content += f"* {highlight}\n"
                
            message = text_content
            print(f"  Sending plain text content")
            print(f"  Sample: {message[:100]}...")


def version_management_example():
    """Example of managing agent card versions."""
    print("\n=== Version Management Example ===\n")
    
    # Create a new agent card
    agent = AgentCard()
    agent.card_data["agent_id"] = "evolving-agent-001"
    agent.card_data["name"] = "Evolving Agent"
    agent.card_data["description"] = "An agent that evolves over time"
    
    print(f"Initial version: {agent.get_version()}")
    
    # Add initial capability
    agent.add_capability(
        name="basic_conversation",
        description="Basic conversation capabilities",
        methods=["greet", "respond"]
    )
    
    # Make a patch update (for bug fixes)
    agent.update_version(increment="patch")
    print(f"After bug fix (patch): {agent.get_version()}")
    
    # Add a new feature (minor update)
    agent.add_capability(
        name="memory",
        description="Remember conversation history",
        methods=["rememberFact", "recall"]
    )
    agent.update_version(increment="minor")
    print(f"After adding memory feature (minor): {agent.get_version()}")
    
    # Major architectural change (major update)
    agent.add_capability(
        name="reasoning",
        description="Complex reasoning capabilities",
        methods=["analyze", "infer", "deduce"]
    )
    agent.update_communication_config(
        protocols=["a2a-v2", "reasoning-protocol"],
        formats=["json", "reasoning-graph"]
    )
    agent.update_version(increment="major")
    print(f"After architectural overhaul (major): {agent.get_version()}")
    
    # Set specific version for release
    agent.update_version(new_version="2.0.0-beta.1")
    print(f"Release candidate version: {agent.get_version()}")
    
    # Save each version to track evolution
    os.makedirs("example_output/version_history", exist_ok=True)
    agent.save_to_file(f"example_output/version_history/evolving_agent_{agent.get_version()}.json")
    print(f"Saved version history to example_output/version_history/")


if __name__ == "__main__":
    # Create example output directory
    os.makedirs("example_output", exist_ok=True)
    
    # Run the examples
    load_existing_card_example()
    create_new_card_example()
    agent_registry_example()
    adaptive_communication_example()
    version_management_example()
