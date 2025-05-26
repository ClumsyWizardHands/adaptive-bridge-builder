"""
Enhanced Principle Engine with Multi-LLM Support

This module extends the principle engine with multi-model support,
allowing different principles to be evaluated by different LLMs
based on their strengths and specializations.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union

from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from llm_selector import LLMSelector
from principle_engine import PrincipleEngine, PrincipleEvaluator
from principle_engine_llm import LLMPrincipleEvaluator

logger = logging.getLogger(__name__)

class MultiModelPrincipleEvaluator(PrincipleEvaluator):
    """
    Principle evaluator that uses multiple LLMs for different types of principles.
    
    This evaluator routes different principle evaluations to different LLMs based on
    their strengths and specializations, providing more accurate and efficient evaluations.
    """
    
    def __init__(
        self,
        registry: LLMAdapterRegistry,
        selector: Optional[LLMSelector] = None,
        principle_model_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        default_model_selector_criteria: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-model principle evaluator.
        
        Args:
            registry: LLM adapter registry to use for model access
            selector: LLM selector for intelligent model selection (created if None)
            principle_model_mapping: Mapping of principle types to model selection criteria
            default_model_selector_criteria: Default criteria for model selection
        """
        self.registry = registry
        self.selector = selector or LLMSelector(registry)
        
        # Default mapping of principle types to model selection criteria
        self.principle_model_mapping = principle_model_mapping or {
            "privacy": {
                "privacy_requirement": LLMSelector.PRIVACY_MAXIMUM,
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE
            },
            "security": {
                "privacy_requirement": LLMSelector.PRIVACY_HIGH,
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX
            },
            "ethical": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "cost_preference": LLMSelector.COST_PERFORMANCE,
                "required_capabilities": ["ethics", "reasoning"]
            },
            "fairness": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "required_capabilities": ["reasoning"]
            },
            "default": default_model_selector_criteria or {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "cost_preference": LLMSelector.COST_BALANCED
            }
        }
        
        # Cache for evaluators to avoid creating duplicates
        self._evaluator_cache = {}
    
    async def evaluate_principle(
        self,
        principle: Dict[str, Any],
        context: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a principle using the appropriate LLM.
        
        Args:
            principle: The principle to evaluate
            context: The context for evaluation
            options: Additional options for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        options = options or {}
        
        # Get the principle type (default to "default")
        principle_type = principle.get("type", "default").lower()
        
        # Check if we have a specific mapping for this principle type
        if principle_type not in self.principle_model_mapping:
            principle_type = "default"
        
        # Get the model selection criteria for this principle type
        selection_criteria = self.principle_model_mapping[principle_type].copy()
        
        # Apply any overrides from options
        if "model_selection_criteria" in options:
            selection_criteria.update(options["model_selection_criteria"])
        
        # Select the appropriate model
        model_info = self.selector.select_model(**selection_criteria)
        
        if not model_info["adapter"]:
            logger.warning(f"No suitable model found for principle type '{principle_type}'")
            # Return a default evaluation result
            return {
                "principle_id": principle.get("id", "unknown"),
                "evaluation": "unknown",
                "confidence": 0.0,
                "reasons": ["No suitable model available for evaluation"],
                "metadata": {
                    "evaluated": False,
                    "error": "No suitable model available"
                }
            }
        
        # Get or create an evaluator for this model
        evaluator = await self._get_evaluator(model_info)
        
        # Evaluate the principle
        try:
            result = await evaluator.evaluate_principle(principle, context, options)
            
            # Add model information to the result metadata
            result.setdefault("metadata", {})
            result["metadata"]["model"] = {
                "provider": model_info["provider"],
                "model": model_info["model"],
                "score": model_info["score"],
                "reasons": model_info.get("reasons", [])
            }
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating principle with model {model_info['provider']}/{model_info['model']}: {str(e)}")
            
            # Try with a fallback model if available
            if "fallback" in options:
                try:
                    fallback_model_info = self.selector.select_model(**options["fallback"])
                    if fallback_model_info["adapter"]:
                        logger.info(f"Using fallback model {fallback_model_info['provider']}/{fallback_model_info['model']}")
                        fallback_evaluator = await self._get_evaluator(fallback_model_info)
                        result = await fallback_evaluator.evaluate_principle(principle, context, options)
                        
                        # Add fallback model information to the result metadata
                        result.setdefault("metadata", {})
                        result["metadata"]["model"] = {
                            "provider": fallback_model_info["provider"],
                            "model": fallback_model_info["model"],
                            "score": fallback_model_info["score"],
                            "reasons": fallback_model_info.get("reasons", []),
                            "fallback": True
                        }
                        
                        return result
                except Exception as fallback_error:
                    logger.error(f"Error using fallback model: {str(fallback_error)}")
            
            # Return an error result
            return {
                "principle_id": principle.get("id", "unknown"),
                "evaluation": "unknown",
                "confidence": 0.0,
                "reasons": [f"Error during evaluation: {str(e)}"],
                "metadata": {
                    "evaluated": False,
                    "error": str(e),
                    "model": {
                        "provider": model_info["provider"],
                        "model": model_info["model"]
                    }
                }
            }
    
    async def evaluate_principles(
        self,
        principles: List[Dict[str, Any]],
        context: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple principles, potentially in parallel.
        
        Args:
            principles: List of principles to evaluate
            context: The context for evaluation
            options: Additional options for evaluation
            
        Returns:
            List of evaluation results
        """
        options = options or {}
        
        # Create tasks for each principle evaluation
        tasks = []
        for principle in principles:
            tasks.append(self.evaluate_principle(principle, context, options))
        
        # Run evaluations in parallel (if enabled), otherwise sequentially
        if options.get("parallel", True):
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for task in tasks:
                results.append(await task)
        
        return results
    
    async def _get_evaluator(self, model_info: Dict[str, Any]) -> PrincipleEvaluator:
        """
        Get or create a principle evaluator for a specific model.
        
        Args:
            model_info: Model information from LLMSelector
            
        Returns:
            Principle evaluator for the model
        """
        # Create a key for the cache
        cache_key = f"{model_info['provider']}/{model_info['model']}"
        
        # Check if we already have an evaluator for this model
        if cache_key in self._evaluator_cache:
            return self._evaluator_cache[cache_key]
        
        # Create a new evaluator
        adapter = model_info["adapter"]
        evaluator = LLMPrincipleEvaluator(adapter, model=model_info["model"])
        
        # Cache the evaluator
        self._evaluator_cache = {**self._evaluator_cache, cache_key: evaluator}
        
        return evaluator


class EnhancedPrincipleEngine(PrincipleEngine):
    """
    Enhanced principle engine with multi-model support.
    
    This engine extends the base principle engine with support for
    using multiple LLMs for different types of principles.
    """
    
    def __init__(
        self,
        registry: LLMAdapterRegistry,
        selector: Optional[LLMSelector] = None,
        principle_model_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        default_model_selector_criteria: Optional[Dict[str, Any]] = None,
        principles: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced principle engine.
        
        Args:
            registry: LLM adapter registry to use for model access
            selector: LLM selector for intelligent model selection (created if None)
            principle_model_mapping: Mapping of principle types to model selection criteria
            default_model_selector_criteria: Default criteria for model selection
            principles: List of principles to use
            options: Additional options for the engine
        """
        # Create the multi-model evaluator
        evaluator = MultiModelPrincipleEvaluator(
            registry=registry,
            selector=selector,
            principle_model_mapping=principle_model_mapping,
            default_model_selector_criteria=default_model_selector_criteria
        )
        
        # Initialize the base engine
        super().__init__(evaluator=evaluator, principles=principles, options=options)
    
    async def evaluate_action(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        principles: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an action against principles.
        
        Args:
            action: The action to evaluate
            context: Additional context for evaluation
            principles: Specific principles to evaluate against (uses all if None)
            options: Additional options for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        # Use the base implementation
        return await super().evaluate_action(action, context, principles, options)
