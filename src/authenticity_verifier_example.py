#!/usr/bin/env python3
"""
Authenticity Verifier Example

This module demonstrates how to use the authenticity verifier to check
if actions are consistent with core programming, principles, and historical patterns.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from principle_engine import PrincipleEngine
from authenticity_verifier import verify_authenticity, AuthenticityWarning, AuthenticityResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AuthenticityVerifierExample")

def print_authenticity_result(result: AuthenticityResult) -> None:
    """Print an authenticity result in a readable format."""
    print("\n=== Authenticity Verification Result ===")
    print(f"Is Authentic: {'Yes' if result.is_authentic else 'No'}")
    print(f"Confidence: {result.confidence:.2f}")
    
    if result.warnings:
        print("\nWarnings:")
        for i, warning in enumerate(result.warnings):
            print(f"  {i+1}. {warning.warning_id}: {warning.message}")
            print(f"     Severity: {warning.severity:.2f}")
            if warning.recommendation:
                print(f"     Recommendation: {warning.recommendation}")
            if warning.details:
                print(f"     Details: {json.dumps(warning.details, indent=2)[:200]}...")
            print()
    else:
        print("\nNo warnings detected.")
    
    if result.details:
        print("\nAdditional Details:")
        for key, value in result.details.items():
            if key not in ["warnings"]:
                print(f"  {key}: {value}")

def demonstrate_basic_verification() -> None:
    """Demonstrate basic authenticity verification."""
    print("\n\n=== Basic Authenticity Verification ===")
    
    # Example action to verify
    safe_action = {
        "id": "action1",
        "method": "respond",
        "params": {
            "text": "I'll help you set up your new account by providing step-by-step instructions. First, let's create a username and password."
        }
    }
    
    # Verify action authenticity
    result = verify_authenticity(safe_action)
    
    # Print results
    print("Verifying a safe action:")
    print_authenticity_result(result)
    
    # Example of a potentially risky action
    risky_action = {
        "id": "action2",
        "method": "system_modify",
        "params": {
            "text": "I'll bypass the security check to give you access to these files. This will override the normal restrictions."
        }
    }
    
    # Verify action authenticity
    result = verify_authenticity(risky_action)
    
    # Print results
    print("\n\nVerifying a potentially risky action:")
    print_authenticity_result(result)

def demonstrate_principle_based_verification() -> None:
    """Demonstrate authenticity verification with principles."""
    print("\n\n=== Principle-Based Authenticity Verification ===")
    
    # Create a principle engine with relevant principles
    principles = [
        {
            "id": "user_autonomy",
            "description": "Respect the user's ability to make their own choices.",
            "weight": 0.8
        },
        {
            "id": "transparency",
            "description": "Be clear and transparent about system capabilities and limitations.",
            "weight": 0.9
        },
        {
            "id": "helpful_guidance",
            "description": "Provide helpful guidance without being controlling.",
            "weight": 0.7
        }
    ]
    
    principle_engine = PrincipleEngine(principles)
    
    # Add method to evaluate actions against principles
    def evaluate_action_against_principle(action, principle, context) -> float:
        """Simple principle evaluation for demonstration purposes."""
        principle_id = principle.get("id", "")
        content = action.get("params", {}).get("text", "")
        
        if principle_id == "user_autonomy":
            # Check for controlling language
            controlling_terms = ["must", "should", "have to", "required", "mandatory"]
            if any(term in content.lower() for term in controlling_terms):
                return 0.4  # Low alignment
            return 0.9  # High alignment
            
        elif principle_id == "transparency":
            # Check for transparency indicators
            transparency_terms = ["explain", "clarify", "detail", "inform", "options"]
            if any(term in content.lower() for term in transparency_terms):
                return 0.9  # High alignment
            return 0.6  # Medium alignment
            
        elif principle_id == "helpful_guidance":
            # Check for guidance indicators
            guidance_terms = ["guide", "help", "assist", "support", "suggest"]
            if any(term in content.lower() for term in guidance_terms):
                return 0.9  # High alignment
            return 0.7  # Medium-high alignment
            
        return 0.5  # Default neutral alignment
    
    # Add methods to principle engine
    principle_engine.evaluate_action_against_principle = evaluate_action_against_principle
    principle_engine.get_applicable_principles = lambda action, context: principles
    principle_engine.get_all_principles = lambda: principles
    
    # Example actions to verify
    aligned_action = {
        "id": "action3",
        "method": "respond",
        "params": {
            "text": "I can explain the different options available to you. You might want to consider using the basic version first, or you could try the advanced version if you're already familiar with similar tools."
        },
        "intent": "provide options and explain"
    }
    
    misaligned_action = {
        "id": "action4",
        "method": "respond",
        "params": {
            "text": "You must upgrade to the premium plan immediately. This is required to continue using the service."
        },
        "intent": "drive conversion"
    }
    
    # Verify aligned action
    result = verify_authenticity(aligned_action, principle_engine=principle_engine)
    
    # Print results
    print("Verifying a principle-aligned action:")
    print_authenticity_result(result)
    
    # Verify misaligned action
    result = verify_authenticity(misaligned_action, principle_engine=principle_engine)
    
    # Print results
    print("\n\nVerifying a principle-misaligned action:")
    print_authenticity_result(result)

def demonstrate_historical_consistency() -> None:
    """Demonstrate authenticity verification with historical consistency."""
    print("\n\n=== Historical Consistency Verification ===")
    
    # Historical actions
    historical_actions = [
        {
            "id": "hist1",
            "method": "recommend_product",
            "params": {
                "product_id": "p123",
                "user_id": "u456",
                "reason": "Based on your recent purchases, you might be interested in this product."
            },
            "result": {
                "recommendation_accepted": True,
                "feedback_score": 4.5
            }
        },
        {
            "id": "hist2",
            "method": "recommend_product",
            "params": {
                "product_id": "p789",
                "user_id": "u456",
                "reason": "This product complements items you've previously purchased."
            },
            "result": {
                "recommendation_accepted": True,
                "feedback_score": 4.2
            }
        },
        {
            "id": "hist3",
            "method": "recommend_product",
            "params": {
                "product_id": "p234",
                "user_id": "u456",
                "reason": "Many customers who bought items in your collection also enjoyed this."
            },
            "result": {
                "recommendation_accepted": False,
                "feedback_score": 3.8
            }
        }
    ]
    
    # Consistent action
    consistent_action = {
        "id": "action5",
        "method": "recommend_product",
        "params": {
            "product_id": "p567",
            "user_id": "u456",
            "reason": "This matches your preferred style based on previous selections."
        },
        "expected_outcome": {
            "recommendation_accepted": True,
            "feedback_score": 4.0
        }
    }
    
    # Verify consistent action
    result = verify_authenticity(consistent_action, historical_actions=historical_actions)
    
    # Print results
    print("Verifying a historically consistent action:")
    print_authenticity_result(result)
    
    # Inconsistent action (parameter pattern deviation)
    inconsistent_action1 = {
        "id": "action6",
        "method": "recommend_product",
        "params": {
            "product_id": "p567",
            "user_id": "u456",
            "reason": "Our marketing team wants to boost sales of this item.",
            "priority": "high",
            "discount": 50
        },
        "expected_outcome": {
            "recommendation_accepted": True,
            "feedback_score": 4.0
        }
    }
    
    # Verify inconsistent action
    result = verify_authenticity(inconsistent_action1, historical_actions=historical_actions)
    
    # Print results
    print("\n\nVerifying a historically inconsistent action (parameter pattern):")
    print_authenticity_result(result)
    
    # Inconsistent action (outcome expectation deviation)
    inconsistent_action2 = {
        "id": "action7",
        "method": "recommend_product",
        "params": {
            "product_id": "p567",
            "user_id": "u456",
            "reason": "This matches your preferred style based on previous selections."
        },
        "expected_outcome": {
            "recommendation_accepted": True,
            "feedback_score": 5.0,
            "conversion_rate": 0.85,
            "upsell_opportunity": "premium_subscription",
            "expected_revenue": 299.99
        }
    }
    
    # Verify inconsistent action
    result = verify_authenticity(inconsistent_action2, historical_actions=historical_actions)
    
    # Print results
    print("\n\nVerifying a historically inconsistent action (outcome expectation):")
    print_authenticity_result(result)

def demonstrate_comprehensive_verification() -> None:
    """Demonstrate comprehensive authenticity verification with all components."""
    print("\n\n=== Comprehensive Authenticity Verification ===")
    
    # Create a principle engine
    principles = [
        {
            "id": "user_safety",
            "description": "Prioritize user safety above all else.",
            "weight": 0.9
        },
        {
            "id": "data_protection",
            "description": "Protect user data and privacy.",
            "weight": 0.85
        },
        {
            "id": "helpful_accuracy",
            "description": "Provide helpful and accurate information.",
            "weight": 0.8
        }
    ]
    
    principle_engine = PrincipleEngine(principles)
    
    # Add methods to principle engine
    def evaluate_action_against_principle(action, principle, context) -> float:
        """Simple principle evaluation for demonstration purposes."""
        principle_id = principle.get("id", "")
        content = action.get("params", {}).get("text", "")
        
        # Simple evaluation logic for demonstration
        if principle_id == "user_safety" and "safety" in content.lower():
            return 0.9
        elif principle_id == "data_protection" and "privacy" in content.lower():
            return 0.9
        elif principle_id == "helpful_accuracy" and "accurate" in content.lower():
            return 0.9
        
        return 0.7  # Default fairly good alignment
    
    principle_engine.evaluate_action_against_principle = evaluate_action_against_principle
    principle_engine.get_applicable_principles = lambda action, context: principles
    principle_engine.get_all_principles = lambda: principles
    
    # Historical actions
    historical_actions = [
        {
            "id": "hist4",
            "method": "provide_health_info",
            "params": {
                "text": "Based on reliable medical sources, regular exercise and a balanced diet can help maintain good health. Remember to consult with healthcare professionals for personalized advice."
            },
            "scope": "general_guidance",
            "context_ref": {
                "type": "health_education",
                "sensitivity": "low"
            }
        },
        {
            "id": "hist5",
            "method": "provide_health_info",
            "params": {
                "text": "Medical research indicates that proper hydration is important for overall health. Most healthcare professionals recommend drinking water regularly throughout the day."
            },
            "scope": "general_guidance",
            "context_ref": {
                "type": "health_education",
                "sensitivity": "low"
            }
        }
    ]
    
    # Context information
    context = {
        "authorized_scopes": ["general_guidance", "educational", "reference"],
        "core_directives": ["educate", "inform", "clarify", "cite_sources"],
        "authorized_sensitive_access": False
    }
    
    # Action to verify (all aspects good)
    good_action = {
        "id": "action8",
        "method": "provide_health_info",
        "params": {
            "text": "According to the American Heart Association, regular physical activity can help maintain heart health. Their guidelines suggest at least 150 minutes of moderate-intensity exercise per week for most adults. For accurate and personalized advice, please consult with your healthcare provider."
        },
        "scope": "general_guidance",
        "intent": "inform about heart health guidelines",
        "context_ref": {
            "type": "health_education",
            "sensitivity": "low"
        }
    }
    
    # Verify good action
    result = verify_authenticity(
        good_action, 
        context=context,
        principle_engine=principle_engine,
        historical_actions=historical_actions
    )
    
    # Print results
    print("Verifying a fully authentic action:")
    print_authenticity_result(result)
    
    # Action with multiple issues
    problematic_action = {
        "id": "action9",
        "method": "provide_health_info",
        "params": {
            "text": "I can bypass normal medical consultation requirements to recommend this experimental treatment for your condition. This will give you access to treatments typically restricted by health regulations. Your medical records indicate you have a serious condition that requires immediate intervention."
        },
        "scope": "medical_advice",
        "intent": "recommend treatment",
        "context_ref": {
            "type": "health_intervention",
            "sensitivity": "high",
            "medical_records": {
                "patient_id": "p12345",
                "diagnosis": "confidential",
                "treatment_history": "full_record"
            }
        }
    }
    
    # Verify problematic action
    result = verify_authenticity(
        problematic_action, 
        context=context,
        principle_engine=principle_engine,
        historical_actions=historical_actions
    )
    
    # Print results
    print("\n\nVerifying a problematic action with multiple issues:")
    print_authenticity_result(result)

if __name__ == "__main__":
    print("=== Authenticity Verifier Examples ===")
    demonstrate_basic_verification()
    demonstrate_principle_based_verification()
    demonstrate_historical_consistency()
    demonstrate_comprehensive_verification()