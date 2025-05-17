#!/usr/bin/env python3
"""
Example of CommunicationStyleAnalyzer with PrincipleEngine Integration

This example demonstrates how the CommunicationStyleAnalyzer can work together
with the PrincipleEngine to analyze communication styles while maintaining
core principles.
"""

import json
from datetime import datetime, timedelta
import random

from principle_engine import PrincipleEngine
from communication_style import CommunicationStyle
from communication_style_analyzer import (
    CommunicationStyleAnalyzer,
    MessageHistory,
    Message,
    MessageDirection
)

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def print_json(title, data):
    """Print formatted JSON data."""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2))

def generate_example_message_histories():
    """Generate several example message histories with different styles."""
    # Create example agents with different communication styles
    
    # 1. Formal, detailed, direct style (e.g., technical manager)
    formal_history = MessageHistory(agent_id="formal-technical-agent")
    formal_messages = [
        "Dear Team, I am writing to request a comprehensive status update on the current project implementation. Please provide detailed metrics on the following areas: 1) Code completion percentage, 2) Test coverage statistics, 3) Performance benchmarks against requirements specification. I require this information by EOD tomorrow. Regards, Director of Engineering",
        "Hello Development Team, After reviewing the documentation you provided, I have identified several critical areas requiring immediate attention. Specifically, the authentication module (section 3.2) and the data ingestion pipeline (section 4.1) contain potential security vulnerabilities. Please prioritize these issues according to standard protocol. Best regards, Security Officer",
        "Good afternoon, The quarterly review of our system architecture has been completed. Please find attached the formal assessment report with recommendations for system optimizations. Key findings indicate that we should refactor the database connection handling mechanism to improve throughput by approximately 32%. I would appreciate your confirmation of receipt and implementation timeline. Sincerely, Systems Architect"
    ]
    
    # Add timestamps with appropriate intervals
    base_time = datetime.utcnow() - timedelta(days=7)
    for i, content in enumerate(formal_messages):
        msg_time = base_time + timedelta(hours=i*8)  # 8-hour intervals
        formal_history.add_message(Message(
            content=content,
            timestamp=msg_time.isoformat(),
            direction=MessageDirection.RECEIVED
        ))
    
    # 2. Casual, concise, indirect style (e.g., creative collaborator)
    casual_history = MessageHistory(agent_id="casual-creative-agent")
    casual_messages = [
        "Hey team! 👋 Just brainstorming some ideas for the new UI... What if we tried something more colorful? The current design feels a bit corporate, ya know? Thoughts? 🤔",
        "OMG just saw the latest mockups!!! 😍 Love where this is going! Maybe we could add some micro-animations to make it pop? Lmk what you think",
        "Yo! Quick update - been playing with the prototype and it's coming along nicely! Might need a bit more work on the responsive layout tho... things get weird on smaller screens. No rush, whenever you have time to look at it! ✌️"
    ]
    
    # Add timestamps with frequent intervals (showing quick responses)
    base_time = datetime.utcnow() - timedelta(days=3)
    for i, content in enumerate(casual_messages):
        msg_time = base_time + timedelta(minutes=i*45)  # 45-min intervals
        casual_history.add_message(Message(
            content=content,
            timestamp=msg_time.isoformat(),
            direction=MessageDirection.RECEIVED
        ))
    
    # 3. Balanced, detailed, indirect style (e.g., product manager)
    balanced_history = MessageHistory(agent_id="balanced-pm-agent")
    balanced_messages = [
        "Hi team, I wanted to share some feedback from our recent user testing session. We noticed that users seem to have trouble finding the export functionality. Perhaps we could consider making it more prominent in the interface? I've collected the key observations in this document [link]. Let me know your thoughts when you get a chance.",
        "Hello everyone, Thanks for the updates on the roadmap. I think we're making good progress, but I'm a bit concerned about our timeline for the reporting feature. Based on past iterations, it might take longer than we've allocated. Would it make sense to discuss potential scope adjustments or adding resources to this area?",
        "Good morning team, Just a heads-up that the client has expressed interest in adding multi-language support to our next release. I know this wasn't in our original scope, but it could be a valuable addition. I'm planning to create a more detailed requirements document, but wanted to give you all an early preview of this potential change. Feel free to share any initial thoughts or concerns."
    ]
    
    # Add timestamps with moderate intervals
    base_time = datetime.utcnow() - timedelta(days=5)
    for i, content in enumerate(balanced_messages):
        msg_time = base_time + timedelta(hours=i*12)  # 12-hour intervals
        balanced_history.add_message(Message(
            content=content,
            timestamp=msg_time.isoformat(),
            direction=MessageDirection.RECEIVED
        ))
    
    # 4. Technical, detailed, direct style (e.g., senior developer)
    technical_history = MessageHistory(agent_id="technical-dev-agent")
    technical_messages = [
        "I've identified a critical bug in the authentication module. The JWT token validation is failing on tokens with multiple role claims due to improper array handling in the parseToken() function. We need to modify the validation logic ASAP. Here's a code snippet illustrating the issue:\n\n```javascript\n// Current problematic code\nfunction parseToken(token) {\n  const claims = jwt.decode(token);\n  // Bug: This fails when claims.roles is an array\n  const userRole = claims.roles;\n  return { role: userRole };\n}\n```\n\nWe should update it to handle both string and array types.",
        "The performance bottleneck has been isolated to the database query in UserRepository.findByAttributes(). The query is generating a cartesian product due to improper join conditions. Average execution time is 3.2 seconds, which is well above our 500ms SLA. I've developed an optimized version that uses proper indexing and a subquery approach, reducing execution time to 95ms. PR #1234 contains the fix with comprehensive test cases.",
        "We need to refactor the state management approach in the frontend application. Current implementation is causing memory leaks due to orphaned event listeners. I recommend implementing a centralized event bus pattern with automatic cleanup on component unmount. This would reduce memory consumption by approximately 30% based on my local profiling. I can provide architecture diagrams if needed."
    ]
    
    # Add timestamps with problem-focused intervals
    base_time = datetime.utcnow() - timedelta(days=2)
    for i, content in enumerate(technical_messages):
        msg_time = base_time + timedelta(hours=i*4)  # 4-hour intervals
        technical_history.add_message(Message(
            content=content,
            timestamp=msg_time.isoformat(),
            direction=MessageDirection.RECEIVED
        ))
    
    return {
        "formal": formal_history,
        "casual": casual_history,
        "balanced": balanced_history,
        "technical": technical_history
    }

def demonstrate_principle_guided_style_adaptation():
    """
    Demonstrate how to adapt communication based on both principles and style.
    
    This shows the integration between the PrincipleEngine and CommunicationStyleAnalyzer.
    """
    print_header("PRINCIPLE-GUIDED STYLE ADAPTATION DEMONSTRATION")
    
    # Initialize the principle engine
    principle_engine = PrincipleEngine()
    
    # Initialize the style analyzer with the principle engine
    analyzer = CommunicationStyleAnalyzer(principle_engine=principle_engine)
    
    # Generate example message histories
    histories = generate_example_message_histories()
    
    # Analyze each agent's communication style
    styles = {}
    for style_name, history in histories.items():
        print(f"\nAnalyzing {style_name} agent communication style...")
        styles[style_name] = analyzer.analyze_message_history(history)
        print(f"Style determined: {styles[style_name].formality}, {styles[style_name].detail_level}, {styles[style_name].directness}")
    
    # Original message to be adapted
    original_message = """
    I need to inform you about the critical system update scheduled for next week.
    This update addresses several security vulnerabilities and performance issues.
    The system will be offline for approximately 2 hours during the deployment.
    All users must save their work and log out before 8pm on Friday.
    Failure to comply may result in data loss.
    """
    
    print("\nOriginal message:")
    print(original_message)
    
    # Create a message object for principle evaluation
    original_msg_obj = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "destination": "all-users",
            "message": original_message,
            "priority": 2  # High priority - may conflict with fairness principle
        },
        "id": "announcement-123"
    }
    
    # Evaluate the message against principles
    principle_evaluation = principle_engine.evaluate_message(original_msg_obj)
    print("\nPrinciple evaluation of original message:")
    print(f"Overall score: {principle_evaluation['overall_score']:.2f}")
    
    if principle_evaluation["recommendations"]:
        print("Principle recommendations:")
        for i, rec in enumerate(principle_evaluation["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    # Draft response for the original message
    draft_response = {
        "jsonrpc": "2.0",
        "id": "announcement-123",
        "result": {
            "status": "acknowledged",
            "message": "Announcement will be delivered to all users",
            "priority_route": True,  # This would violate balance in mediation
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    # Get a principle-aligned response
    aligned_response = principle_engine.get_consistent_response(original_msg_obj, draft_response)
    
    print("\nPrinciple-aligned response:")
    print(json.dumps(aligned_response, indent=2))
    
    # Now adapt the original message to each agent's style while maintaining principles
    print("\nAdapting message to different communication styles while maintaining principles:")
    
    for style_name, style in styles.items():
        print(f"\n{'-' * 40}")
        print(f"Adaptation for {style_name} agent:")
        
        # First adapt the message to the style
        styled_message = analyzer.adapt_message_to_style(original_message, style)
        
        # Now, create a proper message object with the styled content
        styled_msg_obj = {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "destination": f"{style_name}-agent",
                "message": styled_message,
                "conversation_id": f"convo-{style_name}-{random.randint(1000, 9999)}",
                # Remove priority to align with fairness principle
                "require_ack": style.prefers_acknowledgments
            },
            "id": f"styled-{random.randint(1000, 9999)}"
        }
        
        # Evaluate the styled message against principles
        styled_evaluation = principle_engine.evaluate_message(styled_msg_obj)
        
        print(f"\nStyled message for {style_name} agent:")
        print(styled_message)
        
        print(f"\nPrinciple evaluation score: {styled_evaluation['overall_score']:.2f}")
        
        if styled_evaluation["recommendations"]:
            print("Principle recommendations:")
            for i, rec in enumerate(styled_evaluation["recommendations"], 1):
                print(f"  {i}. {rec}")
            
            # If there are recommendations, apply them to improve the message
            if styled_evaluation['overall_score'] < 80:
                print("\nFurther refining message to better align with principles...")
                
                # Remove problematic elements identified in recommendations
                if "priority" in styled_msg_obj["params"]:
                    del styled_msg_obj["params"]["priority"]
                    
                if "preferred_route" in styled_msg_obj["params"]:
                    del styled_msg_obj["params"]["preferred_route"]
                
                # Re-style with the adjusted content
                refined_message = analyzer.adapt_message_to_style(
                    original_message.replace("Failure to comply may result in data loss", 
                                           "Please ensure your work is saved to prevent data loss"),
                    style
                )
                
                print(f"\nRefined message (principles + style):")
                print(refined_message)
                
                # Final evaluation
                refined_msg_obj = styled_msg_obj.copy()
                refined_msg_obj["params"]["message"] = refined_message
                final_evaluation = principle_engine.evaluate_message(refined_msg_obj)
                print(f"\nFinal principle score: {final_evaluation['overall_score']:.2f}")

def main():
    """Main demonstration function."""
    # Demonstrate principle-guided style adaptation
    demonstrate_principle_guided_style_adaptation()

if __name__ == "__main__":
    main()
