#!/usr/bin/env python3
"""
Integration file for adding the positive reinforcement functionality to the PrincipleEngine.

This module provides code for integrating the prioritize_positive_reinforcement function
directly into the PrincipleEngine class. It includes:

1. A patch function for adding the method to an existing PrincipleEngine instance
2. Instructions for permanently integrating the function in the core PrincipleEngine class
3. Utility functions for managing related dependencies
"""

import logging
import importlib.util
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineIntegration")

def integrate_positive_reinforcement() -> int:
    """
    Integrate the positive reinforcement capability directly into the PrincipleEngine class.
    
    This function should be called once during module initialization to add the
    prioritize_positive_reinforcement method to the PrincipleEngine class.
    """
    try:
        # First, check if the principle_engine module is available
        if not importlib.util.find_spec("principle_engine"):
            logger.error("principle_engine module not found. Unable to integrate positive reinforcement.")
            return False
        
        # Next, check if positive reinforcement module is available
        if not importlib.util.find_spec("principle_engine_positive_reinforcement"):
            logger.error("principle_engine_positive_reinforcement module not found. Unable to integrate.")
            return False
        
        # Import modules
        from principle_engine import PrincipleEngine
        from principle_engine_positive_reinforcement import prioritize_positive_reinforcement
        
        # Check if the function is already integrated
        if hasattr(PrincipleEngine, "prioritize_positive_reinforcement"):
            logger.info("prioritize_positive_reinforcement is already integrated with PrincipleEngine.")
            return True
        
        # Add the method to the PrincipleEngine class
        def _method(self, interaction_data, agent_id) -> None:
            """
            Analyze interaction data to identify opportunities for positive reinforcement.
            
            This method embodies the 'Love as a Generative Force' principle by identifying
            opportunities to steer interactions toward positive, constructive outcomes.
            
            Args:
                interaction_data: Dictionary containing message content, sender info, and
                                context from the SessionManager
                agent_id: The ID of the agent for which to prioritize positive reinforcement
                
            Returns:
                Dictionary containing:
                - generative_potential_score: Float from -1.0 to 1.0 indicating potential for positive steering
                - suggested_modifications: Optional list of suggested response modifications
                - identified_positive_elements: Optional list of positive elements identified in the input
            """
            # Create references to related systems if available
            emotional_intelligence = getattr(self, '_emotional_intelligence', None)
            learning_system = getattr(self, '_learning_system', None)
            
            # Call the implementation function
            return prioritize_positive_reinforcement(
                interaction_data=interaction_data,
                agent_id=agent_id,
                principle_engine=self,
                emotional_intelligence=emotional_intelligence,
                learning_system=learning_system
            )
        
        # Add the method to the class
        setattr(PrincipleEngine, "prioritize_positive_reinforcement", _method)
        logger.info("Successfully integrated prioritize_positive_reinforcement with PrincipleEngine.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate positive reinforcement: {e}")
        return False

def setup_dependencies(principle_engine_instance) -> None:
    """
    Set up dependencies for optimal positive reinforcement functionality.
    
    Args:
        principle_engine_instance: The PrincipleEngine instance to set up
        
    Returns:
        Dictionary with information about which dependencies were successfully set up
    """
    results = {
        "emotional_intelligence": False,
        "learning_system": False
    }
    
    # Set up EmotionalIntelligence
    try:
        from emotional_intelligence import EmotionalIntelligence
        emotional_intelligence = EmotionalIntelligence()
        setattr(principle_engine_instance, '_emotional_intelligence', emotional_intelligence)
        results["emotional_intelligence"] = True
        logger.info("EmotionalIntelligence successfully set up as dependency")
    except ImportError:
        logger.warning("EmotionalIntelligence module not available. Using simplified emotion analysis.")
    
    # Set up LearningSystem
    try:
        from learning_system import LearningSystem
        learning_system = LearningSystem()
        setattr(principle_engine_instance, '_learning_system', learning_system)
        results["learning_system"] = True
        logger.info("LearningSystem successfully set up as dependency")
    except ImportError:
        logger.warning("LearningSystem module not available. Learning from interactions disabled.")
    
    return results

def is_positive_reinforcement_available() -> bool:
    """
    Check if the positive reinforcement functionality is available.
    
    Returns:
        Boolean indicating whether the functionality can be used
    """
    # Check if required modules are available
    pe_spec = importlib.util.find_spec("principle_engine")
    pr_spec = importlib.util.find_spec("principle_engine_positive_reinforcement")
    
    if not pe_spec or not pr_spec:
        return False
    
    # Check if PrincipleEngine has the method
    try:
        from principle_engine import PrincipleEngine
        return hasattr(PrincipleEngine, "prioritize_positive_reinforcement")
    except ImportError:
        return False


# Integration instructions (as regular comments to avoid Python parser issues)
# ============================================================================
#
# To permanently integrate this functionality into the PrincipleEngine class:
#
# 1. Add the following import to principle_engine.py:
#    from principle_engine_positive_reinforcement import prioritize_positive_reinforcement
#
# 2. Add the prioritize_positive_reinforcement method to the PrincipleEngine class in principle_engine.py:
#    def prioritize_positive_reinforcement(self, interaction_data: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
#        """
#        Analyze interaction data to identify opportunities for positive reinforcement.
#        
#        This method embodies the 'Love as a Generative Force' principle by identifying
#        opportunities to steer interactions toward positive, constructive outcomes.
#        
#        Args:
#            interaction_data: Dictionary containing message content, sender info, and
#                            context from the SessionManager
#            agent_id: The ID of the agent for which to prioritize positive reinforcement
#            
#        Returns:
#            Dictionary containing:
#            - generative_potential_score: Float from -1.0 to 1.0 indicating potential for positive steering
#            - suggested_modifications: Optional list of suggested response modifications
#            - identified_positive_elements: Optional list of positive elements identified in the input
#        """
#        # Create references to related systems if available
#        emotional_intelligence = getattr(self, '_emotional_intelligence', None)
#        learning_system = getattr(self, '_learning_system', None)
#        
#        # Call the implementation function
#        return prioritize_positive_reinforcement(
#            interaction_data=interaction_data,
#            agent_id=agent_id,
#            principle_engine=self,
#            emotional_intelligence=emotional_intelligence,
#            learning_system=learning_system
#        )
#
# 3. Alternatively, use runtime integration by calling this at module initialization:
#    from principle_engine_integration import integrate_positive_reinforcement
#    integrate_positive_reinforcement()


if __name__ == "__main__":
    # Demonstrate the integration
    result = integrate_positive_reinforcement()
    if result:
        print("Successfully integrated positive reinforcement with PrincipleEngine")
    else:
        print("Failed to integrate positive reinforcement")
    
    # Check availability
    if is_positive_reinforcement_available():
        print("Positive reinforcement functionality is available")
    else:
        print("Positive reinforcement functionality is not available")