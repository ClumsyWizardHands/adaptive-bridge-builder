#!/usr/bin/env python3
"""
Principle Engine DB

This module provides a version of the PrincipleEngine that loads principles from
a database using the PrincipleRepository. It maintains the same interfaces as the
original PrincipleEngine class, allowing it to be used as a drop-in replacement.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from principle_repository import PrincipleRepository
from principle_engine import PrincipleEngine, Principle, PrincipleEvalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineDB")


class PrincipleEngineDB(PrincipleEngine):
    """
    Principle engine that loads principles from a database.
    
    This class extends the base PrincipleEngine but loads principles from a
    database using the PrincipleRepository. It maintains the same interfaces
    as the original PrincipleEngine class.
    """
    
    def __init__(self, repository: PrincipleRepository, refresh_on_init: bool = True) -> None:
        """
        Initialize the principle engine with a repository.
        
        Args:
            repository: The PrincipleRepository to use for loading principles
            refresh_on_init: Whether to refresh principles from the database on initialization
        """
        super().__init__(principles=[])  # Initialize with empty principles list
        self.repository = repository
        
        if refresh_on_init:
            self.refresh_principles()
    
    def refresh_principles(self) -> int:
        """
        Refresh principles from the database.
        
        Returns:
            The number of principles loaded
        """
        # Get active principles from the repository
        db_principles = self.repository.get_principles(is_active=True)
        
        # Convert DB principles to Principle objects
        principles = []
        for db_principle in db_principles:
            # Get full principle details
            full_principle = self.repository.get_principle(db_principle["id"])
            
            # Extract evaluation criteria
            evaluation_criteria = []
            for ec in full_principle.get("evaluation_criteria", []):
                if ec.get("is_active"):
                    criteria = {
                        "id": ec["id"],
                        "type": ec["type_name"],
                        "content": ec["content"],
                        "requires_llm": ec["requires_llm"]
                    }
                    
                    # Add parameters if available
                    if ec.get("parameters"):
                        try:
                            params = json.loads(ec["parameters"]) if isinstance(ec["parameters"], str) else ec["parameters"]
                            criteria["parameters"] = params
                        except Exception as e:
                            logger.warning(f"Failed to parse parameters for criteria {ec['id']}: {str(e)}")
                    
                    evaluation_criteria.append(criteria)
            
            # Create Principle object
            principle = Principle(
                name=db_principle["name"],
                short_name=db_principle["short_name"] or db_principle["name"].lower().replace(" ", "_"),
                description=db_principle["description"],
                evaluation_criteria=evaluation_criteria,
                category=db_principle["category_name"],
                importance_level=db_principle["importance_level_name"],
                tags=[tag["name"] for tag in db_principle.get("tags", [])]
            )
            
            principles.append(principle)
        
        # Replace existing principles with the new ones
        self.principles = principles
        logger.info(f"Loaded {len(principles)} active principles from the database")
        
        return len(principles)
    
    def get_principle_by_name(self, name: str) -> Optional[Principle]:
        """
        Get a principle by name or short name.
        
        Args:
            name: Name or short name of the principle
            
        Returns:
            Principle if found, None otherwise
        """
        # First try to get the principle from the in-memory list
        principle = super().get_principle_by_name(name)
        
        # If not found, try to refresh from the database
        if not principle:
            db_principles = self.repository.get_principles()
            for db_principle in db_principles:
                if db_principle["name"] == name or db_principle["short_name"] == name:
                    # Refresh principles and search again
                    self.refresh_principles()
                    return super().get_principle_by_name(name)
        
        return principle
    
    def get_principles_by_decision_point(self, decision_point_name: str) -> List[Principle]:
        """
        Get principles associated with a specific decision point.
        
        Args:
            decision_point_name: Name of the decision point
            
        Returns:
            List of principles for the decision point
        """
        # Get decision points from the repository
        decision_points = self.repository.get_decision_points()
        
        # Find the decision point by name
        decision_point = next((dp for dp in decision_points if dp["name"] == decision_point_name), None)
        if not decision_point:
            logger.warning(f"Decision point '{decision_point_name}' not found")
            return []
        
        # Load principles for this decision point
        principles = []
        for principle in self.principles:
            # Check if principle is associated with this decision point
            db_principle = self.repository.get_principle_by_short_name(principle.short_name)
            if db_principle and any(dp["name"] == decision_point_name for dp in db_principle.get("decision_points", [])):
                principles.append(principle)
        
        return principles
    
    def evaluate_action_for_decision_point(
        self,
        action_description: str,
        context: Dict[str, Any],
        decision_point_name: str,
        alignment_threshold: float = 0.7
    ) -> PrincipleEvalResult:
        """
        Evaluate an action against principles for a specific decision point.
        
        Args:
            action_description: Description of the action to evaluate
            context: Context for the evaluation
            decision_point_name: Name of the decision point
            alignment_threshold: Threshold for determining if an action is aligned
            
        Returns:
            Evaluation result
        """
        # Get principles for this decision point
        principles = self.get_principles_by_decision_point(decision_point_name)
        
        # If no principles found, return default result
        if not principles:
            logger.warning(f"No principles found for decision point '{decision_point_name}'")
            return PrincipleEvalResult(
                aligned=True,
                overall_score=1.0,
                principle_scores={},
                reasoning="No relevant principles to evaluate against",
                recommendations=[]
            )
        
        # Get decision point details to get alignment threshold
        decision_points = self.repository.get_decision_points()
        decision_point = next((dp for dp in decision_points if dp["name"] == decision_point_name), None)
        
        if decision_point:
            # For each principle, check if it has a custom threshold for this decision point
            for principle in principles:
                db_principle = self.repository.get_principle_by_short_name(principle.short_name)
                if db_principle:
                    for dp in db_principle.get("decision_points", []):
                        if dp["name"] == decision_point_name:
                            # Use the custom threshold if available
                            alignment_threshold = dp.get("alignment_threshold", alignment_threshold)
                            break
        
        # Evaluate against the selected principles
        return self.evaluate_action(
            action_description=action_description,
            context=context,
            principles=[p.name for p in principles],
            alignment_threshold=alignment_threshold
        )


# Add methods to PrincipleRepository for convenience
def get_principle_by_short_name(self, short_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a principle by short name.
    
    Args:
        short_name: Short name of the principle
        
    Returns:
        Principle dictionary or None if not found
    """
    principles = self.get_principles()
    principle = next((p for p in principles if p["short_name"] == short_name), None)
    
    if principle:
        return self.get_principle(principle["id"])
    
    return None

# Monkey patch the PrincipleRepository class
PrincipleRepository.get_principle_by_short_name = get_principle_by_short_name
