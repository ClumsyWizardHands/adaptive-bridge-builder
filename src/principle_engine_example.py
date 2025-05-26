#!/usr/bin/env python3
"""
Example Integration of PrincipleEngine with AdaptiveBridgeBuilder

This example demonstrates how to integrate the PrincipleEngine with
the AdaptiveBridgeBuilder agent to evaluate messages against principles
and ensure responses align with the core values.
"""

import json
import uuid
from adaptive_bridge_builder import AdaptiveBridgeBuilder
from principle_engine import PrincipleEngine

def print_header(title) -> None:
    """Print a formatted section header."""
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"{'-' * 80}")

def print_json(title, data) -> None:
    """Print formatted JSON data."""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2))
    print()

class PrincipleGuidedBridgeBuilder(AdaptiveBridgeBuilder):
    """
    Enhanced version of AdaptiveBridgeBuilder that integrates the PrincipleEngine
    to ensure all communications adhere to core principles.
    """
    
    def __init__(self, agent_id=None, agent_card_path="src/agent_card.json", principles_file=None) -> None:
        """
        Initialize the PrincipleGuidedBridgeBuilder.
        
        Args:
            agent_id: Unique identifier for this agent instance.
            agent_card_path: Path to the agent card JSON file.
            principles_file: Path to principles JSON file (optional).
        """
        # Initialize the base AdaptiveBridgeBuilder
        super().__init__(agent_id, agent_card_path)
        
        # Initialize the PrincipleEngine
        self.principle_engine = PrincipleEngine(principles_file)
        
        print(f"PrincipleGuidedBridgeBuilder initialized with {len(self.principle_engine.principles)} principles")
    
    def process_message(self, message) -> None:
        """
        Process an incoming message with principle evaluation and alignment.
        
        Args:
            message: The incoming message in JSON-RPC 2.0 format.
            
        Returns:
            A principle-aligned response message in JSON-RPC 2.0 format.
        """
        # Step 1: Evaluate the incoming message against principles
        print("Evaluating message against principles...")
        evaluation = self.principle_engine.evaluate_message(message)
        
        # Get recommendations if score is below threshold
        if evaluation["overall_score"] < 80 and evaluation["recommendations"]:
            print("Principle recommendations for this message:")
            for i, rec in enumerate(evaluation["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        # Step 2: Process the message with the base implementation
        print("Processing message with AdaptiveBridgeBuilder...")
        draft_response = super().process_message(message)
        
        # Step 3: Adjust the response to align with principles
        print("Adjusting response to align with principles...")
        consistent_response = self.principle_engine.get_consistent_response(message, draft_response)
        
        # Step 4: Update the consistency report
        print(f"Current principle consistency score: {self.principle_engine.overall_consistency:.2f}")
        
        return consistent_response
    
    def get_principle_consistency_report(self) -> None:
        """Get a detailed report on principle consistency."""
        return self.principle_engine.get_consistency_report()
    
    def get_principle_descriptions(self) -> None:
        """Get descriptions of all principles."""
        return self.principle_engine.get_principle_descriptions()



    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
def demonstrate_principle_effects() -> None:
    """Demonstrate how each principle affects communication decisions."""
    print_header("PRINCIPLE EFFECTS DEMONSTRATION")
    
    # Create a principle engine
    engine = PrincipleEngine()
    principles = engine.get_principle_descriptions()
    
    # Example messages demonstrating each principle
    examples = [
        {
            "principle": "Fairness as Truth",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "priority": 10  # High priority (violates fairness)
                },
                "id": "example-1"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"}
                    # No priority field (respects fairness)
                },
                "id": "example-1"
            },
            "explanation": "Removing priority flags ensures all messages are treated equally regardless of source."
        },
        {
            "principle": "Harmony Through Presence",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "require_ack": False  # No acknowledgment (violates harmony)
                },
                "id": "example-2"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "conversation_id": "conv-123",
                    "require_ack": True  # Explicit acknowledgment (respects harmony)
                },
                "id": "example-2"
            },
            "explanation": "Adding conversation tracking and acknowledgment requests ensures continuous communication."
        },
        {
            "principle": "Adaptability as Strength",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "translateProtocol",
                "params": {
                    "source_protocol": "json-rpc-2.0",
                    "target_protocol": "a2a",
                    "message": {"method": "getData"},
                    "strict_format": True,  # Strict format (violates adaptability)
                    "allow_interpretation": False
                },
                "id": "example-3"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "translateProtocol",
                "params": {
                    "source_protocol": "json-rpc-2.0",
                    "target_protocol": "a2a",
                    "message": {"method": "getData"},
                    "allow_interpretation": True  # Flexible interpretation (respects adaptability)
                },
                "id": "example-3"
            },
            "explanation": "Allowing flexible interpretation enables adaptive handling of unexpected formats."
        },
        {
            "principle": "Balance in Mediation",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "preferred_route": "fast-lane"  # Preferred route (violates balance)
                },
                "id": "example-4"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"}
                    # No preferred route (respects balance)
                },
                "id": "example-4"
            },
            "explanation": "Removing preferred route specifications ensures equal treatment in routing decisions."
        },
        {
            "principle": "Clarity in Complexity",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "echo",
                "params": {
                    "data": {
                        "nested": {
                            "deeply": {
                                "complex": {
                                    "structure": "with many levels",
                                    "that": "is hard to understand",
                                    "and": "unnecessarily verbose",
                                    "for": "a simple operation"
                                }
                            }
                        }
                    }
                },
                "id": "example-5"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "echo",
                "params": {
                    "data": "Simple, clear message structure",
                    "details": {
                        "additional_info": "Organized in a flat, accessible structure"
                    }
                },
                "id": "example-5"
            },
            "explanation": "Simplifying message structure improves clarity while preserving essential information."
        },
        {
            "principle": "Integrity in Transmission",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "allow_modification": True  # Allow modifications (violates integrity)
                },
                "id": "example-6"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "allow_modification": False,  # Preserve content (respects integrity)
                    "verify_delivery": True
                },
                "id": "example-6"
            },
            "explanation": "Disabling content modification and adding verification ensures message integrity."
        },
        {
            "principle": "Resilience Through Connection",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"}
                    # No fallback routes (violates resilience)
                },
                "id": "example-7"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "fallback_routes": [  # Multiple routes (respects resilience)
                        "backup-route-1",
                        "backup-route-2"
                    ],
                    "retry_strategy": {
                        "max_retries": 3,
                        "backoff_factor": 1.5
                    }
                },
                "id": "example-7"
            },
            "explanation": "Adding fallback routes and retry strategies improves communication resilience."
        },
        {
            "principle": "Empathy in Interface",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"}
                    # No recipient preferences (violates empathy)
                },
                "id": "example-8"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "recipient_preferences": {  # Recipient preferences (respects empathy)
                        "format": "compact",
                        "response_style": "detailed",
                        "language": "en-US"
                    }
                },
                "id": "example-8"
            },
            "explanation": "Considering recipient preferences adapts communication to their needs and constraints."
        },
        {
            "principle": "Truth in Representation",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "nonExistentMethod",  # Unsupported method (violates truth)
                "params": {},
                "id": "example-9"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "getAgentCard",  # Supported method (respects truth)
                "params": {},
                "id": "example-9"
            },
            "explanation": "Using supported methods ensures accurate representation of capabilities."
        },
        {
            "principle": "Growth Through Reflection",
            "original_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "collect_feedback": False  # No feedback (violates growth)
                },
                "id": "example-10"
            },
            "improved_message": {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "destination": "target-agent-001",
                    "message": {"method": "processData"},
                    "collect_feedback": True,  # Enable feedback (respects growth)
                    "learning_mode": "active"
                },
                "id": "example-10"
            },
            "explanation": "Enabling feedback collection and learning mode supports continuous improvement."
        }
    ]
    
    # Display each principle example
    for example in examples:
        principle_name = example["principle"]
        print(f"\n{principle_name}:")
        print(f"  Original Message (Violates Principle):")
        print(f"    {json.dumps(example['original_message'], indent=2)}")
        print(f"  Improved Message (Respects Principle):")
        print(f"    {json.dumps(example['improved_message'], indent=2)}")
        print(f"  Effect: {example['explanation']}")
        
        # Demonstrate evaluation scores
        original_score = engine.evaluate_message(example["original_message"])
        improved_score = engine.evaluate_message(example["improved_message"])
        
        principle_id = next((p["id"] for p in engine.principles if p["name"] == principle_name), None)
        if principle_id:
            original_principle_score = original_score["principle_scores"][principle_id]["score"]
            improved_principle_score = improved_score["principle_scores"][principle_id]["score"]
            
            print(f"  Principle Score Improvement: {original_principle_score:.1f} → {improved_principle_score:.1f}")
            print(f"  Overall Score Improvement: {original_score['overall_score']:.1f} → {improved_score['overall_score']:.1f}")


def main() -> None:
    """Main demonstration function."""
    print_header("PRINCIPLE GUIDED BRIDGE BUILDER DEMONSTRATION")
    
    # Create a PrincipleGuidedBridgeBuilder instance
    agent = PrincipleGuidedBridgeBuilder()
    
    # Generate a unique conversation ID
    conversation_id = str(uuid.uuid4())
    print(f"Conversation ID: {conversation_id}")
    
    # Display principle descriptions
    principles = agent.get_principle_descriptions()
    print(f"\nCore Principles ({len(principles)}):")
    for i, principle in enumerate(principles, 1):
        print(f"{i}. {principle['name']}: {principle['description']}")
        print(f"   Example: {principle['example']}")
    
    # Example 1: Well-formed message respecting principles
    print_header("EXAMPLE 1: WELL-FORMED MESSAGE")
    good_message = {
        "jsonrpc": "2.0",
        "method": "getAgentCard",
        "params": {
            "conversation_id": conversation_id
        },
        "id": "request-1"
    }
    print_json("Request", good_message)
    response = agent.process_message(good_message)
    print_json("Response", response)
    
    # Example 2: Message that violates some principles
    print_header("EXAMPLE 2: MESSAGE VIOLATING PRINCIPLES")
    problematic_message = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": conversation_id,
            "destination": "target-agent-001",
            "message": {
                "jsonrpc": "2.0",
                "method": "processData",
                "params": {"data": {"type": "sensor_reading", "value": 23.5}}
            },
            "priority": 10,  # Violates fairness
            "preferred_route": "fast-lane",  # Violates balance
            "allow_modification": True  # Violates integrity
        },
        "id": "request-2"
    }
    print_json("Request", problematic_message)
    response = agent.process_message(problematic_message)
    print_json("Response", response)
    
    # Example 3: Method not found with principle-guided suggestions
    print_header("EXAMPLE 3: METHOD NOT FOUND WITH SUGGESTIONS")
    unknown_method_message = {
        "jsonrpc": "2.0",
        "method": "unknownMethod",
        "params": {
            "conversation_id": conversation_id
        },
        "id": "request-3"
    }
    print_json("Request", unknown_method_message)
    response = agent.process_message(unknown_method_message)
    print_json("Response", response)
    
    # Get final consistency report
    print_header("PRINCIPLE CONSISTENCY REPORT")
    report = agent.get_principle_consistency_report()
    print(f"Overall consistency: {report['overall_consistency']:.2f}")
    for principle_id, score in report['principle_scores'].items():
        principle_name = next((p["name"] for p in agent.principle_engine.principles if p["id"] == principle_id), principle_id)
        print(f"{principle_name}: {score:.2f}")
    
    # Demonstrate principle effects
    demonstrate_principle_effects()

if __name__ == "__main__":
    main()