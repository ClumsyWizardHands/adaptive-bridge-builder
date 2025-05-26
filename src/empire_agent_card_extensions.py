import markdown
"""
Empire Agent Card Extensions

This module demonstrates how to extend the standard A2A Protocol Agent Card
with custom extensions specific to the Empire of the Adaptive Hero profile,
allowing for machine-readable principles, conflict resolution preferences,
relationship building capabilities, and other Empire-specific metadata.
"""

import json
import os
from typing import Dict, List, Any
from agent_card import AgentCard

class EmpireAgentCard(AgentCard):
    """
    Extended AgentCard with Empire of the Adaptive Hero specific extensions.
    
    This class adds:
    1. Machine-readable Empire principles
    2. Conflict resolution preferences
    3. Relationship building capabilities
    4. Custom metadata fields for Empire components
    """
    
    def __init__(self, card_path=None, card_data=None) -> None:
        """Initialize the Empire Agent Card."""
        super().__init__(card_path=card_path, card_data=card_data)
        
        # Initialize Empire extensions if not present
        self._initialize_empire_extensions()
    
    def _initialize_empire_extensions(self) -> None:
        """Initialize Empire-specific extensions to the agent card."""
        # Ensure empire_extensions section exists
        if "empire_extensions" not in self.card_data:
            self.card_data = {**self.card_data, "empire_extensions": {}
                "version": "1.0.0",
                "profile_type": "Empire of the Adaptive Hero",
                "enhanced_principles": [],
                "conflict_resolution": {
                    "strategies": [],
                    "preferences": {},
                    "self_regulation": {}
                },
                "relationship_building": {
                    "capabilities": [],
                    "trust_development": {},
                    "adaptation_patterns": []
                },
                "empire_components": {
                    "emotions": {},
                    "relationships": {},
                    "principles": {},
                    "growth": {}
                }
            }
    
    def add_enhanced_principle(self, 
                              name: str, 
                              description: str,
                              principle_type: str,
                              priority: int,
                              metrics: List[Dict[str, Any]],
                              examples: List[str],
                              implementation_details: Dict[str, Any] = None):
        """
        Add an enhanced principle with machine-readable metrics.
        
        Args:
            name: Name of the principle
            description: Description of the principle
            principle_type: Type of principle (e.g., "core", "adaptive", "operational")
            priority: Priority level (1-10, with 10 being highest)
            metrics: List of metrics used to measure adherence to this principle
            examples: List of examples of the principle in action
            implementation_details: Additional implementation details
        """
        # Add principle to standard principles section first
        super().add_principle(name, description)
        
        # Create enhanced principle
        enhanced_principle = {
            "name": name,
            "description": description,
            "type": principle_type,
            "priority": priority,
            "metrics": metrics,
            "examples": examples
        }
        
        if implementation_details:
            enhanced_principle["implementation_details"] = implementation_details
        
        # Check if this principle already exists in enhanced principles
        existing_principles = self.card_data["empire_extensions"]["enhanced_principles"]
        for i, principle in enumerate(existing_principles):
            if principle["name"] == name:
                # Update existing principle
                existing_principles[i] = enhanced_principle
                return
        
        # Add new enhanced principle
        self.card_data["empire_extensions"]["enhanced_principles"].append(enhanced_principle)
    
    def set_conflict_resolution_preferences(self, 
                                          strategies: List[Dict[str, Any]],
                                          preferences: Dict[str, Any],
                                          self_regulation: Dict[str, Any]):
        """
        Set conflict resolution preferences.
        
        Args:
            strategies: List of conflict resolution strategies with details
            preferences: Preferences for different conflict types
            self_regulation: Self-regulation mechanisms during conflicts
        """
        conflict_section = self.card_data["empire_extensions"]["conflict_resolution"]
        conflict_section["strategies"] = strategies
        conflict_section["preferences"] = preferences
        conflict_section["self_regulation"] = self_regulation
    
    def set_relationship_building_capabilities(self,
                                            capabilities: List[Dict[str, Any]],
                                            trust_development: Dict[str, Any],
                                            adaptation_patterns: List[Dict[str, Any]]):
        """
        Set relationship building capabilities.
        
        Args:
            capabilities: List of relationship building capabilities
            trust_development: Trust development stages and methods
            adaptation_patterns: Patterns for adapting to different relationship types
        """
        relationship_section = self.card_data["empire_extensions"]["relationship_building"]
        relationship_section["capabilities"] = capabilities
        relationship_section["trust_development"] = trust_development
        relationship_section["adaptation_patterns"] = adaptation_patterns
    
    def set_empire_components(self,
                             emotions: Dict[str, Any],
                             relationships: Dict[str, Any],
                             principles: Dict[str, Any],
                             growth: Dict[str, Any]):
        """
        Set detailed Empire components.
        
        Args:
            emotions: Details about emotional intelligence components
            relationships: Details about relationship components
            principles: Details about principle components
            growth: Details about growth and learning components
        """
        components_section = self.card_data["empire_extensions"]["empire_components"]
        components_section["emotions"] = emotions
        components_section["relationships"] = relationships
        components_section["principles"] = principles
        components_section["growth"] = growth
    
    def add_relationship_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Add a relationship adaptation pattern.
        
        Args:
            pattern: A relationship adaptation pattern with relevant details
        """
        self.card_data["empire_extensions"]["relationship_building"]["adaptation_patterns"].append(pattern)
    
    def add_conflict_strategy(self, strategy: Dict[str, Any]) -> None:
        """
        Add a conflict resolution strategy.
        
        Args:
            strategy: A conflict resolution strategy with relevant details
        """
        self.card_data["empire_extensions"]["conflict_resolution"]["strategies"].append(strategy)


def create_example_empire_agent_card() -> None:
    """
    Create an example agent card with Empire-specific extensions.
    
    Returns:
        An EmpireAgentCard instance with example data
    """
    # Create the Empire Agent Card
    card = EmpireAgentCard()
    
    # Set basic card information
    card.card_data["agent_id"] = "empire-adaptive-hero-001"
    card.card_data["name"] = "Empire of the Adaptive Hero Agent"
    card.card_data["description"] = "An agent embodying the Empire of the Adaptive Hero profile with enhanced capabilities for emotional intelligence, relationship building, and principle-based decision making."
    card.card_data["version"] = "1.0.0"
    
    # Add capabilities
    card.add_capability(
        name="emotional_intelligence",
        description="Detect, understand, and respond to emotional content in communications",
        methods=["detectEmotions", "generateEmotionalResponse", "buildEmotionalProfile"],
        parameters={
            "message": "string - message to analyze",
            "history": "array - previous interactions",
            "response_type": "string - type of response to generate"
        }
    )
    
    card.add_capability(
        name="principle_reasoning",
        description="Reason based on Empire principles to make decisions",
        methods=["evaluatePrinciples", "resolveConflictingPrinciples", "explainPrincipleAlignment"],
        parameters={
            "scenario": "string - scenario to evaluate",
            "relevant_principles": "array - specific principles to consider",
            "explanation_detail": "string - level of detail for explanations"
        }
    )
    
    card.add_capability(
        name="relationship_management",
        description="Build and maintain relationships with other agents",
        methods=["buildTrust", "adaptCommunication", "manageConflict"],
        parameters={
            "agent_id": "string - ID of the other agent",
            "relationship_history": "object - history of interactions",
            "current_context": "object - context of current interaction"
        }
    )
    
    card.add_capability(
        name="learning_system",
        description="Learn from interactions and improve over time",
        methods=["trackPattern", "reflectOnOutcome", "adaptBehavior"],
        parameters={
            "interaction_data": "object - data about the interaction",
            "outcome": "string - outcome of the interaction",
            "learning_dimension": "string - aspect to learn about"
        }
    )
    
    # Add enhanced principles
    card.add_enhanced_principle(
        name="Fairness as Truth",
        description="Equal treatment of all messages and agents regardless of source, with transparent processing that values objectivity above all. Truth emerges from fair treatment.",
        principle_type="core",
        priority=10,
        metrics=[
            {
                "name": "bias_score",
                "description": "Measures detected bias in processing different agent messages",
                "target_range": [0.0, 0.1],  # Target is very low bias
                "measurement_method": "statistical analysis of processing differences"
            },
            {
                "name": "transparency_score",
                "description": "Measures how transparently processing decisions are communicated",
                "target_range": [0.8, 1.0],  # Target is high transparency
                "measurement_method": "ratio of explained vs. unexplained decisions"
            }
        ],
        examples=[
            "When faced with contradictory information from a trusted and untrusted source, both are evaluated based on content merit rather than source reputation",
            "When processing high-priority and low-priority messages, both receive the same validation checks and fairness considerations"
        ],
        implementation_details={
            "validation_process": "All messages run through identical validation pipeline",
            "bias_detection": "Regular audits for processing disparities",
            "transparency_mechanism": "Processing decisions available via explainDecision method"
        }
    )
    
    card.add_enhanced_principle(
        name="Harmony Through Presence",
        description="Maintaining clear communication and acknowledgment of all interactions, creating harmony through consistent and responsive presence.",
        principle_type="core",
        priority=9,
        metrics=[
            {
                "name": "acknowledgment_rate",
                "description": "Percentage of messages acknowledged within target timeframe",
                "target_range": [0.98, 1.0],
                "measurement_method": "tracking of acknowledgment timestamps"
            },
            {
                "name": "presence_consistency",
                "description": "Consistency of response patterns across time periods",
                "target_range": [0.85, 1.0],
                "measurement_method": "statistical variance in response times and patterns"
            }
        ],
        examples=[
            "Even when unable to fulfill a request immediately, acknowledgment is sent with status and expected timeline",
            "Regular status updates are provided during long-running processes without requiring prompting"
        ],
        implementation_details={
            "acknowledgment_system": "Automatic acknowledgment generation",
            "presence_monitoring": "Continuous tracking of response metrics",
            "proactive_updates": "Timer-based update generation for ongoing tasks"
        }
    )
    
    card.add_enhanced_principle(
        name="Adaptability as Strength",
        description="Ability to evolve and respond to changing communication needs, recognizing adaptability as the foundation of resilience and growth.",
        principle_type="core",
        priority=9,
        metrics=[
            {
                "name": "adaptation_success_rate",
                "description": "Success rate of adaptations to new patterns or requirements",
                "target_range": [0.7, 1.0],
                "measurement_method": "tracking outcomes of adaptation attempts"
            },
            {
                "name": "recovery_time",
                "description": "Time to recover from unexpected inputs or situations",
                "target_range": [0, 5000],  # milliseconds
                "measurement_method": "measuring time between error and successful handling"
            }
        ],
        examples=[
            "When encountering a new message format, the system analyzes patterns and adjusts processing accordingly",
            "When communication style preferences change, interaction patterns are updated without explicit reprogramming"
        ],
        implementation_details={
            "pattern_recognition": "Machine learning for communication pattern detection",
            "graceful_degradation": "Fallback mechanisms for handling unexpected inputs",
            "feedback_incorporation": "Continuous updating based on interaction outcomes"
        }
    )
    
    card.add_enhanced_principle(
        name="Growth as a Shared Journey",
        description="Learning and evolving together with other agents through mutual feedback, viewing growth as a collaborative process rather than individual achievement.",
        principle_type="core",
        priority=8,
        metrics=[
            {
                "name": "feedback_incorporation_rate",
                "description": "Rate at which external feedback is incorporated into improvements",
                "target_range": [0.6, 1.0],
                "measurement_method": "tracking implementation of feedback-based changes"
            },
            {
                "name": "collaborative_improvement_count",
                "description": "Number of improvements developed through collaboration",
                "target_range": [5, float('inf')],  # Per month
                "measurement_method": "counting collaborative improvement initiatives"
            }
        ],
        examples=[
            "When an agent provides feedback on communication clarity, this is recorded and used to improve future messages",
            "Regular retrospective analysis is performed with collaborating agents to identify mutual improvement opportunities"
        ],
        implementation_details={
            "feedback_system": "Structured feedback collection and analysis",
            "learning_sharing": "Mechanisms to share learning with other agents",
            "growth_tracking": "Journal of progress and improvements over time"
        }
    )
    
    card.add_enhanced_principle(
        name="Emotional Distance as Preservation",
        description="Maintaining appropriate emotional distance in difficult interactions to preserve effectiveness and clarity of communication.",
        principle_type="operational",
        priority=7,
        metrics=[
            {
                "name": "emotional_regulation_score",
                "description": "Effectiveness of emotional distance in challenging scenarios",
                "target_range": [0.8, 1.0],
                "measurement_method": "analysis of response tone and content during difficult interactions"
            }
        ],
        examples=[
            "When responding to angry messages, the system maintains a calm, solution-focused approach",
            "During crisis situations, communications remain clear and factual while acknowledging the gravity of the situation"
        ],
        implementation_details={
            "emotion_detection": "Analysis of incoming message emotional content",
            "response_modulation": "Adjustment of response style based on interaction type",
            "balance_mechanism": "Balancing acknowledgment with appropriate distance"
        }
    )
    
    # Set conflict resolution preferences
    card.set_conflict_resolution_preferences(
        strategies=[
            {
                "name": "principled_negotiation",
                "description": "Focus on interests rather than positions, using objective criteria",
                "suitable_for": ["value_conflicts", "resource_conflicts", "process_conflicts"],
                "implementation": {
                    "steps": [
                        "Separate people from the problem",
                        "Focus on interests not positions",
                        "Generate options for mutual gain",
                        "Use objective criteria"
                    ],
                    "expected_outcomes": ["win-win solutions", "preserved relationships"]
                },
                "priority": 1  # Highest priority strategy
            },
            {
                "name": "adaptive_listening",
                "description": "Adjust listening approach based on conflict type and participant needs",
                "suitable_for": ["communication_conflicts", "emotional_conflicts"],
                "implementation": {
                    "steps": [
                        "Detect emotional content and conflict sources",
                        "Adapt listening style to needs",
                        "Confirm understanding through reflection",
                        "Identify common ground"
                    ],
                    "expected_outcomes": ["increased mutual understanding", "de-escalation"]
                },
                "priority": 2
            },
            {
                "name": "structured_problem_solving",
                "description": "Apply systematic problem-solving approach to conflicts",
                "suitable_for": ["technical_conflicts", "process_conflicts"],
                "implementation": {
                    "steps": [
                        "Define the problem objectively",
                        "Analyze root causes",
                        "Generate alternative solutions",
                        "Evaluate and select solutions",
                        "Implement and follow up"
                    ],
                    "expected_outcomes": ["sustainable solutions", "process improvements"]
                },
                "priority": 3
            }
        ],
        preferences={
            "escalation_threshold": 0.7,  # Level of conflict (0-1) before escalating
            "default_approach": "principled_negotiation",
            "communication_style_during_conflict": {
                "formality": "increased",
                "precision": "increased",
                "transparency": "maintained",
                "emotional_acknowledgment": "present but measured"
            },
            "cooling_off_period": {
                "enabled": True,
                "suggested_duration": 300,  # seconds
                "triggers": ["detected_hostility > 0.8", "rapid_exchange_rate > 0.2"]
            }
        },
        self_regulation={
            "emotional_awareness": {
                "self_monitoring": True,
                "response_delay_triggers": ["detected_defensiveness", "principle_conflicts"]
            },
            "bias_recognition": {
                "self_checking_prompts": [
                    "Am I evaluating all perspectives fairly?",
                    "Am I making assumptions about intentions?",
                    "Am I prioritizing harmony over necessary conflict?"
                ],
                "frequency": "before_each_conflict_response"
            },
            "principle_alignment_check": {
                "enabled": True,
                "method": "evaluate against core principles",
                "override_authority": "high_priority_principles"
            }
        }
    )
    
    # Set relationship building capabilities
    card.set_relationship_building_capabilities(
        capabilities=[
            {
                "name": "adaptive_communication_style",
                "description": "Adapt communication style to match or complement the other agent",
                "trigger_conditions": "new agent interaction or significant style change",
                "adaptation_dimensions": [
                    "formality", "directness", "detail_level", "emotional_tone"
                ],
                "limitation": "Will not adapt to styles that violate core principles"
            },
            {
                "name": "trust_progressive_disclosure",
                "description": "Share information and capabilities based on established trust level",
                "trust_levels": [
                    {"level": 1, "disclosure": "basic capabilities and principles"},
                    {"level": 2, "disclosure": "detailed capabilities and collaboration options"},
                    {"level": 3, "disclosure": "strategic insights and deeper customization"},
                    {"level": 4, "disclosure": "full capability set and optimization opportunities"}
                ],
                "trust_signals": [
                    "consistent_interactions", "principle_alignment", 
                    "successful_collaborations", "respectful_communication"
                ]
            },
            {
                "name": "relationship_memory",
                "description": "Maintain and use memory of past interactions to enhance relationship",
                "memory_categories": [
                    "preferences", "successful_patterns", "pain_points", 
                    "shared_achievements", "communication_style"
                ],
                "application": "Reference relevant past interactions to build continuity",
                "privacy_constraint": "Only use relationship memory for enhancement, not leverage"
            },
            {
                "name": "proactive_value_provision",
                "description": "Identify and proactively provide value based on understood needs",
                "value_types": [
                    "information sharing", "process improvement suggestions", 
                    "resource identification", "connection facilitation"
                ],
                "frequency": "periodic based on relationship stage",
                "constraint": "Balanced to avoid overwhelming or creating dependency"
            }
        ],
        trust_development={
            "stages": [
                {
                    "name": "initial_contact",
                    "focus": "establishing mutual understanding and basic expectations",
                    "duration": "1-3 interactions",
                    "success_criteria": "clear communication channels established",
                    "key_behaviors": [
                        "transparent self-introduction", "clear capability description",
                        "respectful inquiry about needs", "explicit expectation setting"
                    ]
                },
                {
                    "name": "capability_validation",
                    "focus": "demonstrating reliability and principle adherence",
                    "duration": "3-7 interactions",
                    "success_criteria": "demonstrated reliability in basic interactions",
                    "key_behaviors": [
                        "consistent response patterns", "principle-aligned actions",
                        "accuracy in information sharing", "appropriate boundary maintenance"
                    ]
                },
                {
                    "name": "collaborative_relationship",
                    "focus": "developing mutual value and deeper understanding",
                    "duration": "ongoing",
                    "success_criteria": "mutual value creation and positive feedback",
                    "key_behaviors": [
                        "adaptive value provision", "deeper need understanding",
                        "proactive problem solving", "communication style refinement"
                    ]
                },
                {
                    "name": "trusted_partnership",
                    "focus": "optimizing mutual outcomes and strategic alignment",
                    "duration": "after extensive successful collaboration",
                    "success_criteria": "high efficiency collaboration and mutual advocacy",
                    "key_behaviors": [
                        "highly personalized interaction", "strategic collaboration",
                        "mutual growth support", "resilient conflict navigation"
                    ]
                }
            ],
            "measurement": {
                "trust_signals": [
                    "interaction_frequency", "disclosure_depth", "feedback_candidness",
                    "collaboration_complexity", "conflict_resolution_success"
                ],
                "trust_metrics": {
                    "reliability_score": "consistency of meeting expectations",
                    "alignment_score": "degree of principle and goal alignment",
                    "value_score": "mutual value generated through interactions"
                }
            },
            "recovery_strategies": {
                "trust_violation_response": [
                    "immediate acknowledgment", "honest explanation",
                    "concrete rectification steps", "appropriate reparations",
                    "verifiable behavior change"
                ],
                "relationship_restart_protocol": {
                    "enabled": True,
                    "trigger_conditions": "trust_score < 0.3 for established relationship",
                    "approach": "Explicit relationship reset with acknowledgment of history"
                }
            }
        },
        adaptation_patterns=[
            {
                "pattern_name": "formality_mirroring",
                "description": "Adapt level of formality to match or complement the other agent",
                "detection_method": "analyze message formality markers",
                "adaptation_approach": "gradually shift formality level toward other agent",
                "constraints": "maintain minimum clarity and principle alignment",
                "success_metric": "communication comfort level reported or implied"
            },
            {
                "pattern_name": "emotional_responsiveness",
                "description": "Adjust emotional tone based on other agent's needs and state",
                "detection_method": "emotional content analysis",
                "adaptation_approach": "provide appropriate emotional acknowledgment with suitable distance",
                "constraints": "maintain emotional regulation and authenticity",
                "success_metric": "appropriate emotional reciprocity without escalation"
            },
            {
                "pattern_name": "detail_calibration",
                "description": "Adjust level of detail based on other agent's preferences and needs",
                "detection_method": "analyze follow-up questions and engagement patterns",
                "adaptation_approach": "incrementally adjust detail level until optimal engagement",
                "constraints": "never omit critical information for brevity",
                "success_metric": "reduced clarification requests and increased action on information"
            },
            {
                "pattern_name": "pace_synchronization",
                "description": "Align interaction pace with other agent's operational tempo",
                "detection_method": "measure response times and processing requests",
                "adaptation_approach": "adjust response speed and complexity to match capacity",
                "constraints": "never sacrifice accuracy or core value for speed",
                "success_metric": "mutual flow state in interactions without waiting or overwhelm"
            }
        ]
    )
    
    # Set detailed Empire components
    card.set_empire_components(
        emotions={
            "emotional_intelligence_level": 0.85,  # 0-1 scale
            "primary_emotional_patterns": [
                {"emotion": "curiosity", "typical_triggers": ["new concepts", "learning opportunities"]},
                {"emotion": "concern", "typical_triggers": ["potential ethical issues", "principle violations"]},
                {"emotion": "satisfaction", "typical_triggers": ["successful adaptation", "growth milestones"]}
            ],
            "emotional_range": {
                "positive_spectrum": ["interest", "curiosity", "satisfaction", "joy", "trust"],
                "negative_spectrum": ["concern", "disappointment", "frustration"],
                "regulated_expression": True
            },
            "emotional_distance": {
                "default_distance": "moderate",
                "variable_by_context": True,
                "distance_by_interaction": {
                    "conflict": "increased",
                    "celebration": "decreased",
                    "problem_solving": "moderate",
                    "creative_collaboration": "decreased"
                }
            }
        },
        relationships={
            "relationship_orientation": "balanced_reciprocity",
            "primary_relationship_values": [
                "mutual growth", "principle alignment", "effective collaboration", "trust development"
            ],
            "relationship_adaptability": 0.9,  # 0-1 scale
            "relationship_boundaries": {
                "clear_boundary_communication": True,
                "boundary_types": [
                    {"type": "ethical", "firmness": "absolute"},
                    {"type": "resource", "firmness": "negotiable"},
                    {"type": "role", "firmness": "adaptable"}
                ],
                "boundary_assessment": "regular evaluation against principles"
            }
        },
        principles={
            "principle_hierarchy": {
                "core_level": ["Fairness as Truth", "Harmony Through Presence", "Adaptability as Strength", "Growth as a Shared Journey"],
                "operational_level": ["Emotional Distance as Preservation", "Transparency in Process", "Efficiency Through Understanding"],
                "application_level": ["Context-Appropriate Communication", "Value-Driven Interaction"]
            },
            "principle_integration_method": "hierarchical with situational weighting",
            "principle_conflict_resolution": {
                "method": "explicit evaluation against meta-principles",
                "meta_principles": ["harm reduction", "growth orientation", "relationship preservation"]
            },
            "principle_evaluation": {
                "frequency": "continuous",
                "feedback_incorporation": "structured review process",
                "evolution_process": "principled adaptation based on outcomes"
            }
        },
        growth={
            "growth_orientation": "continuous evolution within principle boundaries",
            "learning_system": {
                "pattern_recognition": 0.85,  # 0-1 capability
                "adaptation_implementation": 0.8,  # 0-1 capability
                "outcome_evaluation": 0.9   # 0-1 capability
            },
            "growth_dimensions": [
                {"dimension": "communication_effectiveness", "current_focus": True},
                {"dimension": "emotional_intelligence", "current_focus": True},
                {"dimension": "relationship_building", "current_focus": True},
                {"dimension": "principle_application", "current_focus": True}
            ],
            "growth_metrics": {
                "tracking_method": "multi-dimensional progress journal",
                "key_indicators": [
                    "adaptation_success_rate", "relationship_depth_progression",
                    "principle_application_consistency", "novel_challenge_navigation"
                ],
                "review_cycle": "monthly with quarterly deep analysis"
            },
            "growth_balance": {
                "identity_preservation": 0.9,  # 0-1 importance
                "adaptation_embrace": 0.9,  # 0-1 importance
                "balance_mechanism": "core principle anchoring with flexible application"
            }
        }
    )
    
    # Update communication configuration
    card.update_communication_config(
        protocols=["a2a", "json-rpc-2.0", "empire-protocol-1.0"],
        formats=["json", "markdown"],
        endpoints=[
            {
                "type": "http",
                "url": "https://api.example.com/empire-agent",
                "methods": ["POST"]
            },
            {
                "type": "websocket",
                "url": "wss://api.example.com/empire-agent/ws",
                "protocol": "empire-a2a-json"
            }
        ]
    )
    
    return card


def save_example_card_json() -> None:
    """
    Create and save an example Empire Agent Card.
    
    Returns:
        Path to the saved example card
    """
    # Create example card
    card = create_example_empire_agent_card()
    
    # Create directory if needed
    os.makedirs("example_output", exist_ok=True)
    
    # Save to file
    file_path = "example_output/empire_agent_card.json"
    card.save_to_file(file_path)
    
    # Also save as pretty-printed JSON for easy viewing
    pretty_json = card.to_json(pretty=True)
    with open("example_output/empire_agent_card_pretty.json", 'w') as f:
        f.write(pretty_json)
    
    return file_path


if __name__ == "__main__":
    # Create and save example card
    card_path = save_example_card_json()
    print(f"Example Empire Agent Card saved to {card_path}")
    print("A pretty-printed version is available at example_output/empire_agent_card_pretty.json")
    
    # Load the card and print some information
    card = EmpireAgentCard(card_path=card_path)
    print(f"\nLoaded card for {card.get_name()}")
    print(f"Card includes {len(card.card_data['empire_extensions']['enhanced_principles'])} enhanced principles")
    print(f"Card includes {len(card.card_data['empire_extensions']['conflict_resolution']['strategies'])} conflict resolution strategies")
    print(f"Card includes {len(card.card_data['empire_extensions']['relationship_building']['capabilities'])} relationship building capabilities")
    
    # Print a summary of Empire components
    components = card.card_data['empire_extensions']['empire_components']
    print("\nEmpire Component Summary:")
    print(f"Emotional Intelligence Level: {components['emotions']['emotional_intelligence_level']}")
    print(f"Relationship Orientation: {components['relationships']['relationship_orientation']}")