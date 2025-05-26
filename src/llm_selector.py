"""
LLM Selector

This module provides a model selection system that intelligently chooses
the most appropriate LLM for a given task based on various criteria.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter

logger = logging.getLogger(__name__)

class LLMSelector:
    """
    Selects the most appropriate LLM for a given task based on various criteria.
    
    This class evaluates available models against task requirements to choose
    the best model for the job, considering factors like:
    - Task complexity
    - Privacy requirements
    - Cost vs. performance trade-offs
    - Latency requirements
    """
    
    # Constants for task complexity
    COMPLEXITY_SIMPLE = "simple"
    COMPLEXITY_MODERATE = "moderate"
    COMPLEXITY_COMPLEX = "complex"
    
    # Constants for cost preference
    COST_LOWEST = "lowest"
    COST_BALANCED = "balanced"
    COST_PERFORMANCE = "performance"
    
    # Constants for privacy requirement
    PRIVACY_STANDARD = "standard"
    PRIVACY_HIGH = "high"
    PRIVACY_MAXIMUM = "maximum"
    
    # Constants for latency requirement
    LATENCY_LOW = "low"         # Fast response required
    LATENCY_MEDIUM = "medium"   # Balanced response time
    LATENCY_HIGH = "high"       # Can wait for high quality
    
    # Model metadata (default values)
    DEFAULT_MODEL_METADATA = {
        # Format: provider/model_name
        "openai/gpt-3.5-turbo": {
            "complexity_score": 0.7,     # Good for moderate tasks
            "privacy_score": 0.5,        # Standard privacy
            "cost_score": 0.8,           # Low cost
            "latency_score": 0.9,        # Low latency (fast)
            "strengths": ["general", "coding", "conversation"],
            "weaknesses": ["reasoning", "complex_tasks"]
        },
        "openai/gpt-4-turbo": {
            "complexity_score": 0.9,     # Great for complex tasks
            "privacy_score": 0.5,        # Standard privacy
            "cost_score": 0.4,           # Higher cost
            "latency_score": 0.6,        # Medium latency
            "strengths": ["reasoning", "coding", "complex_tasks"],
            "weaknesses": ["cost"]
        },
        "anthropic/claude-3-sonnet-20240229": {
            "complexity_score": 0.85,    # Good for complex tasks
            "privacy_score": 0.6,        # Standard privacy+
            "cost_score": 0.5,           # Medium cost
            "latency_score": 0.7,        # Medium latency
            "strengths": ["reasoning", "ethics", "long_context"],
            "weaknesses": ["cost"]
        },
        "anthropic/claude-3-opus-20240229": {
            "complexity_score": 0.95,    # Excellent for complex tasks
            "privacy_score": 0.6,        # Standard privacy+
            "cost_score": 0.3,           # High cost
            "latency_score": 0.5,        # Higher latency
            "strengths": ["reasoning", "ethics", "complex_tasks", "long_context"],
            "weaknesses": ["cost", "latency"]
        },
        "mistral/mistral-tiny": {
            "complexity_score": 0.6,     # OK for simple tasks
            "privacy_score": 0.5,        # Standard privacy
            "cost_score": 0.9,           # Very low cost
            "latency_score": 0.9,        # Low latency (fast)
            "strengths": ["general", "cost", "speed"],
            "weaknesses": ["complex_tasks", "reasoning"]
        },
        "mistral/mistral-small": {
            "complexity_score": 0.7,     # Good for moderate tasks
            "privacy_score": 0.5,        # Standard privacy
            "cost_score": 0.7,           # Low-medium cost
            "latency_score": 0.8,        # Good latency
            "strengths": ["general", "balanced"],
            "weaknesses": ["complex_tasks"]
        },
        "mistral/mistral-medium": {
            "complexity_score": 0.8,     # Good for moderate-complex tasks
            "privacy_score": 0.5,        # Standard privacy
            "cost_score": 0.6,           # Medium cost
            "latency_score": 0.7,        # Medium latency
            "strengths": ["reasoning", "general"],
            "weaknesses": []
        },
        "mistral-local/local-mistral": {
            "complexity_score": 0.6,     # OK for moderate tasks (depends on model)
            "privacy_score": 1.0,        # Maximum privacy (local)
            "cost_score": 1.0,           # No API cost
            "latency_score": 0.6,        # Medium latency (depends on hardware)
            "strengths": ["privacy", "cost", "offline"],
            "weaknesses": ["complex_tasks"]
        }
    }
    
    def __init__(
        self,
        registry: LLMAdapterRegistry,
        model_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        default_provider: Optional[str] = None
    ):
        """
        Initialize the LLM selector.
        
        Args:
            registry: LLM adapter registry to use for model selection
            model_metadata: Custom metadata for models
            default_provider: Default provider to use if no suitable model is found
        """
        self.registry = registry
        self.model_metadata = model_metadata or self.DEFAULT_MODEL_METADATA
        self.default_provider = default_provider
        
        # Add any available models that aren't in the metadata
        self._add_missing_models()
    
    def _add_missing_models(self) -> None:
        """Add any available models that aren't in the metadata."""
        for provider in self.registry.list_adapters():
            adapter = self.registry.get_adapter(provider)
            if not adapter:
                continue
                
            # Add models to metadata if they aren't there
            try:
                available_models = getattr(adapter, "get_available_models", lambda: [])()
                if not available_models and hasattr(adapter, "model_name"):
                    available_models = [adapter.model_name]
                
                for model in available_models:
                    model_key = f"{provider}/{model}"
                    if model_key not in self.model_metadata:
                        # Add with default values based on provider
                        if "mistral-local" in provider or "local" in model.lower():
                            # Local model
                            self.model_metadata = {**self.model_metadata, model_key: {}
                                "complexity_score": 0.6,
                                "privacy_score": 1.0,
                                "cost_score": 1.0,
                                "latency_score": 0.6,
                                "strengths": ["privacy", "cost", "offline"],
                                "weaknesses": ["complex_tasks"]
                            }
                        else:
                            # Generic cloud model
                            self.model_metadata = {**self.model_metadata, model_key: {}
                                "complexity_score": 0.7,
                                "privacy_score": 0.5,
                                "cost_score": 0.7,
                                "latency_score": 0.7,
                                "strengths": ["general"],
                                "weaknesses": []
                            }
                        
                        logger.debug(f"Added missing model to metadata: {model_key}")
            except Exception as e:
                logger.warning(f"Error getting available models for provider {provider}: {str(e)}")
    
    def select_model(
        self,
        task_complexity: str = COMPLEXITY_MODERATE,
        privacy_requirement: str = PRIVACY_STANDARD,
        cost_preference: str = COST_BALANCED,
        latency_requirement: str = LATENCY_MEDIUM,
        required_capabilities: Optional[List[str]] = None,
        excluded_providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select the most appropriate model for the given requirements.
        
        Args:
            task_complexity: Complexity of the task
            privacy_requirement: Privacy requirement level
            cost_preference: Cost preference
            latency_requirement: Latency requirement
            required_capabilities: List of required capabilities
            excluded_providers: List of providers to exclude
            
        Returns:
            Dictionary with selected model information
        """
        # Default values
        required_capabilities = required_capabilities or []
        excluded_providers = excluded_providers or []
        
        # Get the list of available providers
        available_providers = self.registry.list_adapters()
        
        # Filter out excluded providers
        available_providers = [p for p in available_providers if p not in excluded_providers]
        
        if not available_providers:
            logger.warning("No available providers after filtering")
            return {
                "provider": None,
                "model": None,
                "adapter": None,
                "score": 0.0
            }
        
        # Determine the weights for each criterion based on the requirements
        weights = self._get_weights(
            task_complexity,
            privacy_requirement,
            cost_preference,
            latency_requirement
        )
        
        # Get minimum scores based on requirements
        min_scores = self._get_minimum_scores(
            task_complexity,
            privacy_requirement,
            cost_preference,
            latency_requirement
        )
        
        # Score each available model
        model_scores = []
        
        for provider in available_providers:
            adapter = self.registry.get_adapter(provider)
            if not adapter:
                continue
                
            # Get available models for this provider
            try:
                available_models = getattr(adapter, "get_available_models", lambda: [])()
                if not available_models and hasattr(adapter, "model_name"):
                    available_models = [adapter.model_name]
                
                for model_name in available_models:
                    model_key = f"{provider}/{model_name}"
                    
                    # Skip if model metadata is not available
                    if model_key not in self.model_metadata:
                        logger.debug(f"Skipping model with no metadata: {model_key}")
                        continue
                    
                    # Get model metadata
                    metadata = self.model_metadata[model_key]
                    
                    # Check if model meets minimum requirements
                    if self._meets_minimum_requirements(metadata, min_scores, required_capabilities):
                        # Calculate weighted score
                        score = self._calculate_score(metadata, weights)
                        
                        # Add to scores list
                        model_scores.append({
                            "provider": provider,
                            "model": model_name,
                            "adapter": adapter,
                            "score": score,
                            "metadata": metadata,
                            "reasons": self._get_selection_reasons(
                                metadata,
                                task_complexity,
                                privacy_requirement,
                                cost_preference,
                                latency_requirement
                            )
                        })
            except Exception as e:
                logger.warning(f"Error evaluating models for provider {provider}: {str(e)}")
        
        # Sort by score (descending)
        model_scores.sort(key=lambda x: x["score"], reverse=True)
        
        if not model_scores:
            logger.warning("No suitable models found")
            
            # Fall back to default provider if specified
            if self.default_provider:
                adapter = self.registry.get_adapter(self.default_provider)
                if adapter:
                    model_name = getattr(adapter, "model_name", "default")
                    return {
                        "provider": self.default_provider,
                        "model": model_name,
                        "adapter": adapter,
                        "score": 0.0,
                        "reasons": ["Fallback to default provider (no suitable models found)"]
                    }
            
            # No suitable models
            return {
                "provider": None,
                "model": None,
                "adapter": None,
                "score": 0.0
            }
        
        # Return the highest scoring model
        return model_scores[0]
    
    def _get_weights(
        self,
        task_complexity: str,
        privacy_requirement: str,
        cost_preference: str,
        latency_requirement: str
    ) -> Dict[str, float]:
        """
        Get the weights for each criterion based on the requirements.
        
        Args:
            task_complexity: Complexity of the task
            privacy_requirement: Privacy requirement level
            cost_preference: Cost preference
            latency_requirement: Latency requirement
            
        Returns:
            Dictionary with weights for each criterion
        """
        weights = {
            "complexity_score": 1.0,
            "privacy_score": 1.0,
            "cost_score": 1.0,
            "latency_score": 1.0
        }
        
        # Adjust weights based on task complexity
        if task_complexity == self.COMPLEXITY_COMPLEX:
            weights["complexity_score"] = 2.0  # Double importance for complex tasks
            weights["cost_score"] = 0.7        # Less emphasis on cost
        elif task_complexity == self.COMPLEXITY_SIMPLE:
            weights["complexity_score"] = 0.7  # Less emphasis on complexity
            weights["cost_score"] = 1.5        # More emphasis on cost
        
        # Adjust weights based on privacy requirement
        if privacy_requirement == self.PRIVACY_MAXIMUM:
            weights["privacy_score"] = 3.0     # Triple importance for maximum privacy
        elif privacy_requirement == self.PRIVACY_HIGH:
            weights["privacy_score"] = 2.0     # Double importance for high privacy
        
        # Adjust weights based on cost preference
        if cost_preference == self.COST_LOWEST:
            weights["cost_score"] = 2.0        # Double importance for lowest cost
        elif cost_preference == self.COST_PERFORMANCE:
            weights["cost_score"] = 0.5        # Half importance for performance focus
            weights["complexity_score"] = 1.5  # More emphasis on complexity
        
        # Adjust weights based on latency requirement
        if latency_requirement == self.LATENCY_LOW:
            weights["latency_score"] = 2.0     # Double importance for low latency
        elif latency_requirement == self.LATENCY_HIGH:
            weights["latency_score"] = 0.5     # Half importance for high latency
        
        return weights
    
    def _get_minimum_scores(
        self,
        task_complexity: str,
        privacy_requirement: str,
        cost_preference: str,
        latency_requirement: str
    ) -> Dict[str, float]:
        """
        Get the minimum scores for each criterion based on the requirements.
        
        Args:
            task_complexity: Complexity of the task
            privacy_requirement: Privacy requirement level
            cost_preference: Cost preference
            latency_requirement: Latency requirement
            
        Returns:
            Dictionary with minimum scores for each criterion
        """
        min_scores = {
            "complexity_score": 0.0,
            "privacy_score": 0.0,
            "cost_score": 0.0,
            "latency_score": 0.0
        }
        
        # Set minimum scores based on task complexity
        if task_complexity == self.COMPLEXITY_COMPLEX:
            min_scores["complexity_score"] = 0.8
        elif task_complexity == self.COMPLEXITY_MODERATE:
            min_scores["complexity_score"] = 0.6
        
        # Set minimum scores based on privacy requirement
        if privacy_requirement == self.PRIVACY_MAXIMUM:
            min_scores["privacy_score"] = 0.9
        elif privacy_requirement == self.PRIVACY_HIGH:
            min_scores["privacy_score"] = 0.7
        
        # Set minimum scores based on latency requirement
        if latency_requirement == self.LATENCY_LOW:
            min_scores["latency_score"] = 0.8
        elif latency_requirement == self.LATENCY_MEDIUM:
            min_scores["latency_score"] = 0.6
        
        return min_scores
    
    def _meets_minimum_requirements(
        self,
        metadata: Dict[str, Any],
        min_scores: Dict[str, float],
        required_capabilities: List[str]
    ) -> bool:
        """
        Check if a model meets the minimum requirements.
        
        Args:
            metadata: Model metadata
            min_scores: Minimum scores for each criterion
            required_capabilities: List of required capabilities
            
        Returns:
            True if the model meets the minimum requirements, False otherwise
        """
        # Check minimum scores
        for criterion, min_score in min_scores.items():
            if metadata.get(criterion, 0) < min_score:
                return False
        
        # Check required capabilities
        if required_capabilities:
            strengths = set(metadata.get("strengths", []))
            for capability in required_capabilities:
                if capability not in strengths:
                    return False
        
        return True
    
    def _calculate_score(
        self,
        metadata: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate a weighted score for a model.
        
        Args:
            metadata: Model metadata
            weights: Weights for each criterion
            
        Returns:
            Weighted score
        """
        total_weight = sum(weights.values())
        score = 0.0
        
        for criterion, weight in weights.items():
            score += metadata.get(criterion, 0) * weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _get_selection_reasons(
        self,
        metadata: Dict[str, Any],
        task_complexity: str,
        privacy_requirement: str,
        cost_preference: str,
        latency_requirement: str
    ) -> List[str]:
        """
        Get the reasons for selecting a model.
        
        Args:
            metadata: Model metadata
            task_complexity: Complexity of the task
            privacy_requirement: Privacy requirement level
            cost_preference: Cost preference
            latency_requirement: Latency requirement
            
        Returns:
            List of reasons for selecting the model
        """
        reasons = []
        
        # Task complexity
        if task_complexity == self.COMPLEXITY_COMPLEX:
            if metadata.get("complexity_score", 0) >= 0.8:
                reasons.append("Well-suited for complex complexity tasks")
            elif metadata.get("complexity_score", 0) >= 0.6:
                reasons.append("Capable of handling complex tasks")
        elif task_complexity == self.COMPLEXITY_MODERATE:
            if metadata.get("complexity_score", 0) >= 0.6:
                reasons.append("Well-suited for moderate complexity tasks")
        else:  # COMPLEXITY_SIMPLE
            if metadata.get("complexity_score", 0) >= 0.5:
                reasons.append("Well-suited for simple complexity tasks")
        
        # Privacy
        if privacy_requirement == self.PRIVACY_MAXIMUM:
            if metadata.get("privacy_score", 0) >= 0.9:
                reasons.append("Provides maximum privacy protection")
        elif privacy_requirement == self.PRIVACY_HIGH:
            if metadata.get("privacy_score", 0) >= 0.7:
                reasons.append("Provides high privacy protection")
        
        # Cost
        if cost_preference == self.COST_LOWEST:
            if metadata.get("cost_score", 0) >= 0.8:
                reasons.append("Aligns with lowest cost preference")
        elif cost_preference == self.COST_BALANCED:
            if metadata.get("cost_score", 0) >= 0.6:
                reasons.append("Aligns with balanced cost preference")
        else:  # COST_PERFORMANCE
            if metadata.get("cost_score", 0) <= 0.5 and metadata.get("complexity_score", 0) >= 0.8:
                reasons.append("Provides high performance while balancing cost")
        
        # Latency
        if latency_requirement == self.LATENCY_LOW:
            if metadata.get("latency_score", 0) >= 0.8:
                reasons.append("Meets low latency requirements")
        elif latency_requirement == self.LATENCY_MEDIUM:
            if metadata.get("latency_score", 0) >= 0.6:
                reasons.append("Meets medium latency requirements")
        else:  # LATENCY_HIGH
            if metadata.get("latency_score", 0) >= 0.5:
                reasons.append("Meets high latency requirements")
        
        # Add strengths
        strengths = metadata.get("strengths", [])
        if strengths:
            # Max 2 strengths
            if len(strengths) > 2:
                reasons.append(f"Strengths include {strengths[0]} and {strengths[1]}")
            else:
                reasons.append(f"Strengths include {', '.join(strengths)}")
        
        return reasons
