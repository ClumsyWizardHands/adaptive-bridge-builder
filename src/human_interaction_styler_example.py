#!/usr/bin/env python3
"""
Example usage of HumanInteractionStyler for Adaptive Bridge Builder

This example demonstrates how the HumanInteractionStyler builds profiles of human 
communication preferences, detects emotional states, respects cultural differences,
and adapts responses appropriately while maintaining authenticity.
"""

import os
from typing import Dict, Any, List
from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligence
from human_interaction_styler import (
    HumanInteractionStyler,
    HumanProfile,
    CulturalContext
)

def simulate_conversations() -> None:
    """Simulate conversations with different humans over time to show adaptation."""
    # Create a directory for profile storage
    profiles_dir = "profiles"
    if not os.path.exists(profiles_dir):
        os.makedirs(profiles_dir)
    
    # Initialize components
    principle_engine = PrincipleEngine()
    emotional_intelligence = EmotionalIntelligence()
    
    # Initialize HumanInteractionStyler
    styler = HumanInteractionStyler(
        emotional_intelligence=emotional_intelligence,
        principle_engine=principle_engine,
        profiles_directory=profiles_dir
    )
    
    # Create human profiles for our examples
    humans = {
        "formal_executive": {
            "name": "Dr. Sarah Williams",
            "messages": [
                "Good morning. Could you please provide a detailed analysis of the system architecture?",
                "Thank you for the information. I would like to request additional specifics regarding the security protocols.",
                "I appreciate your thorough responses. Please forward the documentation as discussed. Regards, Dr. Williams"
            ],
            "cultural_context": CulturalContext.HIERARCHICAL
        },
        "casual_developer": {
            "name": "Jake",
            "messages": [
                "Hey there! Just checking what's up with the new API? Got a sec to explain?",
                "Cool, thx! Can you give me the quick version of how to implement it?",
                "Awesome explanation! You rock! 🙌"
            ],
            "cultural_context": CulturalContext.EGALITARIAN
        },
        "team_leader": {
            "name": "Maria Chen",
            "messages": [
                "Hello team, we need to discuss our approach to the upcoming project. How should we proceed?",
                "I think we should consider how this affects everyone in our department. What are the implications for our group?",
                "Let's make sure we're all on the same page before moving forward. Does anyone have concerns?"
            ],
            "cultural_context": CulturalContext.COLLECTIVIST
        },
        "direct_analyzer": {
            "name": "Alex",
            "messages": [
                "What's the performance impact? Give me the numbers.",
                "This approach has issues. First, it's too slow. Second, it's not scalable. We need alternatives.",
                "Let's not waste time. What's the bottom line here?"
            ],
            "cultural_context": CulturalContext.LOW_CONTEXT
        },
        "nuanced_consultant": {
            "name": "Yuki Tanaka",
            "messages": [
                "It might be beneficial to consider the various perspectives on this matter before proceeding.",
                "Perhaps we should take into account the historical context of similar implementations.",
                "I believe there may be additional factors at play that we have not yet fully explored."
            ],
            "cultural_context": CulturalContext.HIGH_CONTEXT
        }
    }
    
    # Sample responses to adapt based on human profiles
    sample_responses = {
        "technical_explanation": """
        The system implements a microservice architecture with RESTful APIs for service communication. 
        The main components include:
        
        1. API Gateway: Handles authentication, routing, and rate limiting
        2. Service Registry: Manages service discovery and health monitoring
        3. Data Services: Encapsulate database operations and business logic
        4. Notification System: Manages asynchronous messaging between services
        
        Each service is containerized using Docker and orchestrated with Kubernetes.
        The system uses PostgreSQL for structured data and Redis for caching.
        
        In conclusion, this architecture provides scalability, fault tolerance, and maintainability.
        """,
        
        "project_status": """
        The project is currently at 75% completion with several key milestones achieved:
        
        - Backend API development is complete
        - Frontend framework is implemented
        - Database schema design is finalized
        - 80% of unit tests are passing
        
        However, we've encountered some challenges with third-party integrations and performance
        optimization that may impact the timeline. The integration with the payment gateway
        has been particularly problematic.
        
        We should consider allocating additional resources to address these issues before
        they affect the release date.
        """,
        
        "technical_recommendation": """
        I recommend implementing a caching layer to improve API response times.
        You should use Redis for this purpose due to its performance characteristics
        and built-in expiration policies.
        
        The implementation would require:
        1. Setting up a Redis instance
        2. Modifying API endpoints to check cache before database queries
        3. Implementing a cache invalidation strategy
        4. Adding monitoring for cache hit/miss ratios
        
        This approach will reduce database load and improve user experience
        by decreasing response times for frequently accessed data.
        """
    }
    
    # Simulate conversations over time to demonstrate profile building
    simulate_progressive_adaptation(styler, humans, sample_responses)
    
    # Demonstrate emotional state detection and adaptation
    simulate_emotional_adaptation(styler)
    
    # Demonstrate cultural adaptation
    simulate_cultural_adaptation(styler)
    
    # Demonstrate authenticity principle in action
    simulate_authenticity_principle(styler)

def simulate_progressive_adaptation(
    styler: HumanInteractionStyler, 
    humans: Dict[str, Dict[str, Any]], 
    sample_responses: Dict[str, str]
) -> None:
    """Simulate conversations over time to show how profiles evolve."""
    print("\n=== PROGRESSIVE ADAPTATION DEMONSTRATION ===\n")
    
    for human_id, human_data in humans.items():
        print(f"\n--- INTERACTIONS WITH {human_data['name']} ({human_id}) ---\n")
        
        # Set up profile with initial cultural context
        profile = styler.get_or_create_profile(human_id, human_data['name'])
        profile.cultural_contexts.append(human_data['cultural_context'])
        
        # Process each message and show adapted responses
        for i, message in enumerate(human_data['messages']):
            print(f"MESSAGE {i+1}: \"{message}\"")
            
            # Update profile based on message
            styler.update_profile_from_message(message, human_id)
            
            # Select a sample response to adapt
            response_key = list(sample_responses.keys())[i % len(sample_responses.keys())]
            original_response = sample_responses[response_key]
            
            # Adapt the response
            adapted_response = styler.adapt_response(original_response, human_id)
            
            # Show adaptation results
            print("\nADAPTED RESPONSE:")
            print(f"{adapted_response}")
            
            # Show profile evolution
            if i == len(human_data['messages']) - 1:
                profile = styler.human_profiles[human_id]
                print("\nFINAL PROFILE:")
                print(f"- Formality: {profile.communication_style.formality}")
                print(f"- Detail Level: {profile.communication_style.detail_level}")
                print(f"- Directness: {profile.communication_style.directness}")
                print(f"- Emotional Tone: {profile.primary_emotional_tone}")
                print(f"- Cultural Contexts: {[str(ctx) for ctx in profile.cultural_contexts]}")
                print(f"- Confidence Level: {profile.confidence_level:.2f}")
                print(f"- Interaction Count: {profile.interaction_count}")
                print("")
            
            print("-" * 50)

def simulate_emotional_adaptation(styler: HumanInteractionStyler) -> None:
    """Demonstrate emotional state detection and adaptation."""
    print("\n=== EMOTIONAL STATE ADAPTATION DEMONSTRATION ===\n")
    
    emotional_scenarios = [
        {
            "human_id": "excited_user",
            "name": "Excited User",
            "message": "I'm so thrilled about the new features! This is amazing work! Can't wait to try everything out!",
            "response": "The new features include enhanced data visualization, customizable dashboards, and integrated reporting. Each component has been designed for optimal user experience."
        },
        {
            "human_id": "frustrated_user",
            "name": "Frustrated User",
            "message": "I've been trying to make this work for hours and nothing is happening. This is incredibly frustrating and I'm about to give up.",
            "response": "To resolve this issue, you'll need to clear your cache, restart the application, and ensure all dependencies are updated to the latest version."
        },
        {
            "human_id": "anxious_user",
            "name": "Anxious User",
            "message": "I'm worried that we won't meet the deadline. There are so many things that could go wrong. What if we fail?",
            "response": "Project timelines often face challenges. The best approach is to prioritize critical features, maintain clear communication, and adjust scope if necessary."
        }
    ]
    
    for scenario in emotional_scenarios:
        human_id = scenario["human_id"]
        print(f"\n--- EMOTIONAL ADAPTATION FOR {scenario['name']} ---\n")
        
        # Create profile
        styler.get_or_create_profile(human_id, scenario["name"])
        
        # Process message
        print(f"MESSAGE: \"{scenario['message']}\"")
        
        # Detect emotions
        emotions = styler.detect_emotional_state(scenario["message"])
        print("\nDETECTED EMOTIONS:")
        for emotion in emotions:
            print(f"- {emotion.category.name} ({emotion.intensity.name}), confidence: {emotion.confidence:.2f}")
        
        # Update profile based on message
        styler.update_profile_from_message(scenario["message"], human_id)
        
        # Adapt response
        original_response = scenario["response"]
        adapted_response = styler.adapt_response(original_response, human_id)
        
        # Show adaptation results
        print("\nORIGINAL RESPONSE:")
        print(f"{original_response}")
        print("\nADAPTED RESPONSE:")
        print(f"{adapted_response}")
        print("-" * 50)

def simulate_cultural_adaptation(styler: HumanInteractionStyler) -> None:
    """Demonstrate cultural adaptation in communication."""
    print("\n=== CULTURAL ADAPTATION DEMONSTRATION ===\n")
    
    cultural_scenarios = [
        {
            "human_id": "high_context_japanese",
            "name": "Takahiro Yamada",
            "cultural_context": CulturalContext.HIGH_CONTEXT,
            "message": "I've reviewed the proposal. There are certain aspects that might benefit from further consideration.",
            "response": "We need to revise the timeline and budget allocation. The technical approach should be changed to use a different framework."
        },
        {
            "human_id": "low_context_american",
            "name": "John Smith",
            "cultural_context": CulturalContext.LOW_CONTEXT,
            "message": "I've reviewed the proposal and found several problems. The timeline is unrealistic, the budget is insufficient, and the technical approach won't work.",
            "response": "We will need to make some changes to ensure project success."
        },
        {
            "human_id": "collectivist_korean",
            "name": "Min-ji Kim",
            "cultural_context": CulturalContext.COLLECTIVIST,
            "message": "Our team has been discussing the project direction. We believe we should consider how these changes will affect everyone involved.",
            "response": "I recommend implementing the new system architecture which will improve performance by 30%."
        },
        {
            "human_id": "hierarchical_indian",
            "name": "Dr. Rajesh Patel",
            "cultural_context": CulturalContext.HIERARCHICAL,
            "message": "Please review the attached document and provide your insights at your earliest convenience. Your expertise is highly valued.",
            "response": "Here's what you need to do: First, update the configuration. Second, restart the service. Third, verify the logs show no errors."
        }
    ]
    
    for scenario in cultural_scenarios:
        human_id = scenario["human_id"]
        print(f"\n--- CULTURAL ADAPTATION FOR {scenario['name']} ---\n")
        
        # Create profile with cultural context
        profile = styler.get_or_create_profile(human_id, scenario["name"])
        profile.cultural_contexts = [scenario["cultural_context"]]
        
        # Process message
        print(f"MESSAGE: \"{scenario['message']}\"")
        print(f"CULTURAL CONTEXT: {scenario['cultural_context']}")
        
        # Update profile based on message
        styler.update_profile_from_message(scenario["message"], human_id)
        
        # Adapt response
        original_response = scenario["response"]
        adapted_response = styler.adapt_response(original_response, human_id)
        
        # Show adaptation results
        print("\nORIGINAL RESPONSE:")
        print(f"{original_response}")
        print("\nCULTURALLY ADAPTED RESPONSE:")
        print(f"{adapted_response}")
        print("-" * 50)

def simulate_authenticity_principle(styler: HumanInteractionStyler) -> None:
    """Demonstrate the 'Authenticity Beyond Performance' principle in action."""
    print("\n=== AUTHENTICITY PRINCIPLE DEMONSTRATION ===\n")
    
    # Create a profile with high confidence and many interactions
    human_id = "long_term_user"
    profile = styler.get_or_create_profile(human_id, "Long-term User")
    
    # Set high confidence and interaction count to trigger authenticity principle
    profile.confidence_level = 0.85
    profile.interaction_count = 25
    
    # Adapt with authenticity principle
    response = """
    Based on the latest market analysis, there are three key trends to consider:
    
    1. Growing demand for AI-assisted tools in the manufacturing sector
    2. Increasing regulatory focus on data privacy and security
    3. Shift toward cloud-based solutions for enterprise applications
    
    These trends represent significant opportunities for strategic positioning.
    """
    
    print("SCENARIO: Long-term user with established communication preferences\n")
    print("ORIGINAL RESPONSE:")
    print(response)
    
    # Adapt response
    adapted_response = styler.adapt_response(response, human_id)
    
    print("\nADAPTED RESPONSE WITH AUTHENTICITY PRINCIPLE:")
    print(adapted_response)
    
    # Explain the principle
    print("\nEXPLANATION:")
    print("The 'Authenticity Beyond Performance' principle ensures that while communication")
    print("is adapted to human preferences, the agent maintains its authentic voice and values.")
    print("This prevents the adaptation from becoming mere performance and preserves genuine")
    print("communication, especially in long-term relationships where trust has been established.")

def main() -> None:
    """Run all simulations to demonstrate HumanInteractionStyler capabilities."""
    print("\nHUMAN INTERACTION STYLER DEMONSTRATION\n")
    print("This example demonstrates how the HumanInteractionStyler:")
    print("1. Builds profiles of human communication preferences")
    print("2. Detects emotional states and adapts responses appropriately")
    print("3. Respects cultural differences in communication")
    print("4. Adjusts formality, detail level, and tone based on preferences")
    print("5. Remembers preferences across conversations")
    print("6. Applies the 'Authenticity Beyond Performance' principle")
    
    simulate_conversations()

if __name__ == "__main__":
    main()