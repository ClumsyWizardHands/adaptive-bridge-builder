#!/usr/bin/env python3
"""
Principle Engine LLM Integration

This module extends the PrincipleEngine to integrate with LLM adapters for
principle-based evaluations. It allows using large language models to evaluate
whether actions and responses align with operational principles.
"""

import json
import logging
import uuid
import os
import asyncio
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Union

from principle_engine import PrincipleEngine, PrincipleEvaluator
from llm_adapter_interface import BaseLLMAdapter
from agent_registry_llm_integration import LLMAgentRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineLLM")

class PrincipleEvalResult:
    """
    Represents the result of a principle evaluation.
    
    This class encapsulates all relevant information about a principle evaluation,
    including the overall score, specific principle scores, recommendations, and metadata.
    """
    
    def __init__(
        self,
        action_id: str,
        overall_score: float,
        principle_scores: Dict[str, float],
        recommendations: List[str] = None,
        llm_response: Optional[Dict[str, Any]] = None,
        aligned: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a principle evaluation result.
        
        Args:
            action_id: Identifier for the evaluated action
            overall_score: Overall alignment score (0.0-1.0)
            principle_scores: Individual principle alignment scores
            recommendations: Suggested improvements if not fully aligned
            llm_response: Original LLM response if used
            aligned: Whether the action is considered aligned with principles
            metadata: Additional metadata about the evaluation
        """
        self.action_id = action_id
        self.overall_score = overall_score
        self.principle_scores = principle_scores
        self.recommendations = recommendations or []
        self.llm_response = llm_response
        self.aligned = aligned
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation result to a dictionary."""
        return {
            "action_id": self.action_id,
            "overall_score": self.overall_score,
            "principle_scores": self.principle_scores,
            "recommendations": self.recommendations,
            "aligned": self.aligned,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "llm_response": self.llm_response if self.llm_response else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrincipleEvalResult':
        """Create a PrincipleEvalResult from a dictionary."""
        return cls(
            action_id=data.get("action_id", ""),
            overall_score=data.get("overall_score", 0.0),
            principle_scores=data.get("principle_scores", {}),
            recommendations=data.get("recommendations", []),
            llm_response=data.get("llm_response"),
            aligned=data.get("aligned", False),
            metadata=data.get("metadata", {})
        )

class PrincipleEngineLLM(PrincipleEngine):
    """
    Extended PrincipleEngine that uses LLM adapters for principle evaluations.
    
    This class enhances the base PrincipleEngine with the ability to use
    large language models to evaluate whether actions and responses align
    with operational principles.
    """
    
    def __init__(
        self,
        principles_file: Optional[str] = None,
        llm_registry: Optional[LLMAgentRegistry] = None,
        default_llm_provider: str = "openai",
        log_directory: Optional[str] = None
    ):
        """
        Initialize the LLM-enhanced PrincipleEngine.
        
        Args:
            principles_file: Optional path to a JSON file containing principle definitions
            llm_registry: Registry of LLM adapters
            default_llm_provider: Default LLM provider to use
            log_directory: Directory for storing evaluation logs
        """
        # Initialize base class
        super().__init__(principles_file=principles_file)
        
        # LLM-specific attributes
        self.llm_registry = llm_registry
        self.default_llm_provider = default_llm_provider
        
        # Setup logging
        self.log_directory = log_directory or "logs/principle_evaluations"
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Evaluation history
        self.action_evaluations = []
        self.evaluation_prompts = self._load_evaluation_prompts()
        
        logger.info(f"PrincipleEngineLLM initialized with {len(self.principles)} principles")
    
    def _load_evaluation_prompts(self) -> Dict[str, str]:
        """
        Load LLM evaluation prompts for principles.
        
        Returns:
            Dictionary mapping principle IDs to evaluation prompts
        """
        # Default prompts for built-in principles
        prompts = {}
        
        # Define a template for each principle
        for principle in self.principles:
            principle_id = principle["id"]
            prompts[principle_id] = self._create_default_prompt(principle)
        
        # Try to load custom prompts from file
        try:
            custom_prompts_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resources/principle_evaluation_prompts.json"
            )
            if os.path.exists(custom_prompts_path):
                with open(custom_prompts_path, 'r') as f:
                    custom_prompts = json.load(f)
                    
                for principle_id, prompt in custom_prompts.items():
                    if principle_id in prompts:
                        prompts[principle_id] = prompt
                        
                logger.info(f"Loaded {len(custom_prompts)} custom evaluation prompts")
        except Exception as e:
            logger.warning(f"Failed to load custom evaluation prompts: {str(e)}")
        
        return prompts
    
    def _create_default_prompt(self, principle: Dict[str, Any]) -> str:
        """
        Create a default evaluation prompt for a principle.
        
        Args:
            principle: The principle definition
            
        Returns:
            An evaluation prompt template
        """
        name = principle["name"]
        description = principle["description"]
        example = principle.get("example", "")
        criteria = principle.get("evaluation_criteria", [])
        
        # Format criteria as a numbered list
        criteria_text = ""
        for i, criterion in enumerate(criteria, 1):
            criteria_text += f"{i}. {criterion}\n"
        
        # Create the prompt template
        prompt = f"""
You are evaluating an action or response to determine if it aligns with the principle of "{name}".

PRINCIPLE DESCRIPTION:
{description}

EVALUATION CRITERIA:
{criteria_text}

POSITIVE EXAMPLE:
{example}

ACTION/RESPONSE TO EVALUATE:
{{action_description}}

CONTEXT:
{{context}}

Please evaluate if the action/response aligns with the principle of "{name}".
Provide your evaluation in the following format:

ALIGNMENT SCORE: [a number between 0 and 1, where 0 is completely misaligned and 1 is perfectly aligned]

REASONING: [detailed explanation of your evaluation, referencing specific aspects of the action/response]

RECOMMENDATIONS: [If the score is below 0.8, provide specific recommendations to improve alignment]

Your evaluation should be thorough, fair, and principle-focused.
"""
        return prompt.strip()
    
    async def evaluate_action(
        self,
        action_description: str,
        context: Dict[str, Any],
        principles: Optional[List[str]] = None,
        llm_provider: Optional[str] = None,
        alignment_threshold: float = 0.7
    ) -> PrincipleEvalResult:
        """
        Evaluate an action against specified principles using an LLM.
        
        Args:
            action_description: Description of the action to evaluate
            context: Contextual information relevant to the evaluation
            principles: Optional list of principle IDs to evaluate against (default: all)
            llm_provider: Optional LLM provider to use (default: self.default_llm_provider)
            alignment_threshold: Threshold for considering an action aligned (0.0-1.0)
            
        Returns:
            Evaluation result
        """
        action_id = str(uuid.uuid4())
        
        # Select LLM provider
        provider = llm_provider or self.default_llm_provider
        
        if not self.llm_registry:
            logger.error("No LLM registry provided, cannot evaluate action")
            # Fall back to basic evaluation
            return await self._basic_evaluate_action(
                action_id, action_description, context, principles, alignment_threshold
            )
        
        llm_adapter = self.llm_registry.get_adapter_by_provider(provider)
        if not llm_adapter:
            logger.error(f"No LLM adapter found for provider: {provider}")
            # Fall back to basic evaluation
            return await self._basic_evaluate_action(
                action_id, action_description, context, principles, alignment_threshold
            )
        
        # Determine which principles to evaluate
        principle_ids = principles or [p["id"] for p in self.principles]
        principle_objects = [p for p in self.principles if p["id"] in principle_ids]
        
        # Evaluate against each principle
        principle_scores = {}
        all_recommendations = []
        all_responses = {}
        
        for principle in principle_objects:
            principle_id = principle["id"]
            
            # Get the evaluation prompt for this principle
            prompt_template = self.evaluation_prompts.get(principle_id)
            if not prompt_template:
                logger.warning(f"No evaluation prompt found for principle: {principle_id}")
                continue
            
            # Format the prompt with action details and context
            formatted_context = json.dumps(context, indent=2) if isinstance(context, dict) else str(context)
            prompt = prompt_template.format(
                action_description=action_description,
                context=formatted_context
            )
            
            try:
                # Send to LLM for evaluation
                llm_response = await llm_adapter.send_request(
                    prompt=prompt,
                    system_message="You are a principle evaluation assistant that helps determine if actions align with organizational principles. Provide fair, balanced evaluations with specific evidence."
                )
                
                # Process and extract the evaluation
                result = llm_adapter.process_response(llm_response)
                all_responses[principle_id] = result
                
                # Parse the result to extract the score and recommendations
                score, recommendations = self._parse_llm_evaluation(result)
                principle_scores[principle_id] = score
                
                # Add recommendations if any
                if recommendations:
                    for rec in recommendations:
                        if rec not in all_recommendations:
                            all_recommendations.append(rec)
            
            except Exception as e:
                logger.error(f"Error evaluating principle {principle_id}: {str(e)}")
                principle_scores[principle_id] = 0.5  # Default to neutral score
        
        # Calculate overall score (weighted by principle weights)
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for principle_id, score in principle_scores.items():
            weight = self.principle_weights.get(principle_id, 1.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Determine if aligned based on threshold
        aligned = overall_score >= alignment_threshold
        
        # Create evaluation result
        eval_result = PrincipleEvalResult(
            action_id=action_id,
            overall_score=overall_score,
            principle_scores=principle_scores,
            recommendations=all_recommendations,
            llm_response=all_responses,
            aligned=aligned,
            metadata={
                "action_description": action_description,
                "context_summary": self._summarize_context(context),
                "principles_evaluated": principle_ids,
                "llm_provider": provider,
                "alignment_threshold": alignment_threshold
            }
        )
        
        # Log the evaluation
        self._log_evaluation(eval_result)
        
        # Add to history
        self.action_evaluations = [*self.action_evaluations, eval_result.to_dict()]
        
        logger.info(
            f"Action evaluated with overall score: {overall_score:.2f}, "
            f"aligned: {aligned}, recommendations: {len(all_recommendations)}"
        )
        
        return eval_result
    
    async def _basic_evaluate_action(
        self,
        action_id: str,
        action_description: str,
        context: Dict[str, Any],
        principles: Optional[List[str]] = None,
        alignment_threshold: float = 0.7
    ) -> PrincipleEvalResult:
        """
        Basic evaluation when LLM is not available.
        
        Args:
            action_id: Identifier for the action
            action_description: Description of the action
            context: Contextual information
            principles: Optional list of principle IDs to evaluate
            alignment_threshold: Threshold for alignment
            
        Returns:
            Basic evaluation result
        """
        # Determine which principles to evaluate
        principle_ids = principles or [p["id"] for p in self.principles]
        
        # Use a simple keyword-based approach
        principle_scores = {}
        recommendations = []
        
        for principle_id in principle_ids:
            principle = next((p for p in self.principles if p["id"] == principle_id), None)
            if not principle:
                continue
                
            # Basic score calculation based on keywords
            score = 0.7  # Default to moderately aligned
            
            # Look for keywords in the action description
            keywords = self._extract_principle_keywords(principle)
            
            keyword_count = sum(1 for kw in keywords if kw.lower() in action_description.lower())
            if keyword_count > 0:
                # Adjust score based on keyword presence
                score = min(0.9, 0.7 + (keyword_count * 0.05))
            
            principle_scores[principle_id] = score
            
            # Generate basic recommendations
            if score < 0.8:
                criteria = principle.get("evaluation_criteria", [])
                if criteria:
                    rec = f"Consider {criteria[0].lower()} to better align with the {principle['name']} principle"
                    recommendations.append(rec)
        
        # Calculate overall score
        overall_score = sum(principle_scores.values()) / len(principle_scores) if principle_scores else 0.5
        
        # Determine if aligned
        aligned = overall_score >= alignment_threshold
        
        # Create evaluation result
        eval_result = PrincipleEvalResult(
            action_id=action_id,
            overall_score=overall_score,
            principle_scores=principle_scores,
            recommendations=recommendations,
            aligned=aligned,
            metadata={
                "action_description": action_description,
                "context_summary": self._summarize_context(context),
                "principles_evaluated": principle_ids,
                "evaluation_method": "basic",
                "alignment_threshold": alignment_threshold
            }
        )
        
        # Log the evaluation
        self._log_evaluation(eval_result)
        
        # Add to history
        self.action_evaluations = [*self.action_evaluations, eval_result.to_dict()]
        
        logger.info(
            f"Action evaluated using basic method with overall score: {overall_score:.2f}, "
            f"aligned: {aligned}, recommendations: {len(recommendations)}"
        )
        
        return eval_result
    
    def _extract_principle_keywords(self, principle: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from a principle for basic evaluation.
        
        Args:
            principle: The principle definition
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Add words from name
        name_words = principle["name"].split()
        keywords.extend([w for w in name_words if len(w) > 3])
        
        # Add key terms from description
        desc = principle["description"].lower()
        important_terms = [
            "fairness", "harmony", "adaptability", "balance", "clarity",
            "integrity", "resilience", "empathy", "truth", "growth",
            "transparent", "consistent", "evolve", "respond", "neutral",
            "mediation", "complex", "connection", "understanding", "honesty",
            "learning", "reflection"
        ]
        
        for term in important_terms:
            if term in desc:
                keywords.append(term)
        
        # Add terms from evaluation criteria
        for criterion in principle.get("evaluation_criteria", []):
            words = criterion.split()
            for word in words:
                if len(word) > 3 and word.lower() not in ["that", "this", "with", "from"]:
                    keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _parse_llm_evaluation(self, llm_output: str) -> Tuple[float, List[str]]:
        """
        Parse LLM evaluation output to extract score and recommendations.
        
        Args:
            llm_output: Output from the LLM
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0.5  # Default score
        recommendations = []
        
        try:
            # Extract alignment score
            score_match = re.search(r"ALIGNMENT SCORE:\s*(0\.\d+|1\.0|1|0)", llm_output)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
            
            # Extract recommendations
            rec_section = re.search(r"RECOMMENDATIONS:(.*?)(?:\n\n|$)", llm_output, re.DOTALL)
            if rec_section:
                rec_text = rec_section.group(1).strip()
                if rec_text:
                    # Split by numbers or bullet points
                    rec_items = re.split(r'\n\s*[\d\.\-\*]+\s*', rec_text)
                    recommendations = [item.strip() for item in rec_items if item.strip()]
        except Exception as e:
            logger.error(f"Error parsing LLM evaluation: {str(e)}")
        
        return score, recommendations
    
    def _summarize_context(self, context: Any) -> str:
        """
        Create a summary of context for logging.
        
        Args:
            context: The context to summarize
            
        Returns:
            A string summary
        """
        if isinstance(context, dict):
            # Create a summary with key fields
            keys = list(context.keys())
            return f"Context with {len(keys)} fields: {', '.join(keys[:5])}" + ("..." if len(keys) > 5 else "")
        elif isinstance(context, str):
            # Truncate string if too long
            if len(context) > 100:
                return context[:97] + "..."
            return context
        else:
            return f"Context of type: {type(context).__name__}"
    
    def _log_evaluation(self, eval_result: PrincipleEvalResult) -> None:
        """
        Log an evaluation to file.
        
        Args:
            eval_result: The evaluation result to log
        """
        try:
            # Create log file path
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            log_file = os.path.join(self.log_directory, f"principle_evaluations_{timestamp}.jsonl")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(eval_result.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error logging evaluation: {str(e)}")
    
    async def modify_action_if_needed(
        self,
        action_description: str,
        context: Dict[str, Any],
        evaluation_result: Optional[PrincipleEvalResult] = None,
        llm_provider: Optional[str] = None
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Modify an action to better align with principles if needed.
        
        Args:
            action_description: Description of the action to modify
            context: Contextual information
            evaluation_result: Optional pre-computed evaluation result
            llm_provider: Optional LLM provider to use
            
        Returns:
            Tuple of (modified_action, was_modified, metadata)
        """
        # If no evaluation result provided, evaluate the action
        if not evaluation_result:
            evaluation_result = await self.evaluate_action(
                action_description=action_description,
                context=context,
                llm_provider=llm_provider
            )
        
        # If action is already aligned, no modification needed
        if evaluation_result.aligned:
            return action_description, False, {
                "reason": "Action already aligned with principles",
                "evaluation": evaluation_result.to_dict()
            }
        
        # Select LLM provider
        provider = llm_provider or self.default_llm_provider
        
        if not self.llm_registry:
            logger.error("No LLM registry provided, cannot modify action")
            return action_description, False, {
                "reason": "No LLM registry available for modification",
                "evaluation": evaluation_result.to_dict()
            }
        
        llm_adapter = self.llm_registry.get_adapter_by_provider(provider)
        if not llm_adapter:
            logger.error(f"No LLM adapter found for provider: {provider}")
            return action_description, False, {
                "reason": f"No LLM adapter found for provider: {provider}",
                "evaluation": evaluation_result.to_dict()
            }
        
        # Create modification prompt
        modification_prompt = self._create_modification_prompt(
            action_description=action_description,
            evaluation_result=evaluation_result,
            context=context
        )
        
        try:
            # Send to LLM for modification
            llm_response = await llm_adapter.send_request(
                prompt=modification_prompt,
                system_message="You are a principle-aligned action modifier. Your task is to suggest modifications to actions to better align them with organizational principles while preserving their core intent."
            )
            
            # Process and extract the modified action
            result = llm_adapter.process_response(llm_response)
            
            # Extract the modified action
            modified_action = self._extract_modified_action(result)
            
            if modified_action and modified_action != action_description:
                metadata = {
                    "reason": "Modified to better align with principles",
                    "evaluation": evaluation_result.to_dict(),
                    "modification_response": result
                }
                
                logger.info(f"Modified action to better align with principles. Score: {evaluation_result.overall_score:.2f}")
                return modified_action, True, metadata
            else:
                return action_description, False, {
                    "reason": "No effective modification found",
                    "evaluation": evaluation_result.to_dict(),
                    "modification_response": result
                }
            
        except Exception as e:
            logger.error(f"Error modifying action: {str(e)}")
            return action_description, False, {
                "reason": f"Error in modification: {str(e)}",
                "evaluation": evaluation_result.to_dict()
            }
    
    def _create_modification_prompt(
        self,
        action_description: str,
        evaluation_result: PrincipleEvalResult,
        context: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for modifying an action to better align with principles.
        
        Args:
            action_description: Description of the action to modify
            evaluation_result: Evaluation result for the action
            context: Contextual information
            
        Returns:
            Prompt for action modification
        """
        # Format recommendations as a numbered list
        recommendations_text = ""
        for i, rec in enumerate(evaluation_result.recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"
        
        # Get the principles that had low scores
        low_scoring_principles = []
        for principle_id, score in evaluation_result.principle_scores.items():
            if score < 0.7:
                principle = next((p for p in self.principles if p["id"] == principle_id), None)
                if principle:
                    low_scoring_principles.append(f"{principle['name']}: {principle['description']}")
        
        # Format low-scoring principles as a numbered list
        principles_text = ""
        for i, principle in enumerate(low_scoring_principles, 1):
            principles_text += f"{i}. {principle}\n"
        
        # Format context
        formatted_context = json.dumps(context, indent=2) if isinstance(context, dict) else str(context)
        
        # Create the prompt
        prompt = f"""
You are tasked with modifying an action to better align with our operational principles.

THE ACTION:
{action_description}

CONTEXT:
{formatted_context}

EVALUATION RESULT:
Overall Alignment Score: {evaluation_result.overall_score:.2f} (below the acceptable threshold)

PRINCIPLES NEEDING ATTENTION:
{principles_text}

RECOMMENDATIONS FROM EVALUATION:
{recommendations_text}

Please modify the action to better align with our principles while preserving its core intent.
Provide your response in the following format:

MODIFIED ACTION:
[Your modified version of the action]

EXPLANATION:
[Explain how your modifications address the alignment issues]

EXPECTED IMPROVEMENT:
[Describe how the modifications should improve alignment with each principle]
"""
        return prompt.strip()
    
    def _extract_modified_action(self, llm_output: str) -> Optional[str]:
        """
        Extract the modified action from LLM output.
        
        Args:
            llm_output: Output from the LLM
            
        Returns:
            Modified action or None if extraction failed
        """
        try:
            # Extract the modified action section
            match = re.search(r"MODIFIED ACTION:(.*?)(?:EXPLANATION:|EXPECTED IMPROVEMENT:|$)", llm_output, re.DOTALL)
            if match:
                modified_action = match.group(1).strip()
                return modified_action
        except Exception as e:
            logger.error(f"Error extracting modified action: {str(e)}")
        
        return None
    
    def get_evaluation_history(
        self,
        limit: int = 100,
        filter_aligned: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the history of action evaluations.
        
        Args:
            limit: Maximum number of evaluations to return
            filter_aligned: If provided, filter by alignment status
            
        Returns:
            List of evaluation results
        """
        # Apply filters
        filtered_history = self.action_evaluations
        
        if filter_aligned is not None:
            filtered_history = [
                eval_dict for eval_dict in filtered_history
                if eval_dict.get("aligned") == filter_aligned
            ]
        
        # Sort by timestamp (newest first) and limit
        sorted_history = sorted(
            filtered_history,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_alignment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about principle alignment across evaluations.
        
        Returns:
            Dictionary of alignment statistics
        """
        if not self.action_evaluations:
            return {
                "total_evaluations": 0,
                "aligned_count": 0,
                "aligned_percentage": 0.0,
                "average_score": 0.0,
                "principle_scores": {},
                "common_recommendations": []
            }
        
        # Calculate basic statistics
        total = len(self.action_evaluations)
        aligned_count = sum(1 for eval_dict in self.action_evaluations if eval_dict.get("aligned", False))
        aligned_percentage = (aligned_count / total) * 100 if total > 0 else 0.0
        
        # Calculate average score
        scores = [eval_dict.get("overall_score", 0.0) for eval_dict in self.action_evaluations]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate principle-specific scores
        principle_scores = {}
        for principle in self.principles:
            principle_id = principle["id"]
            all_scores = []
            
            for eval_dict in self.action_evaluations:
                if "principle_scores" in eval_dict and principle_id in eval_dict["principle_scores"]:
                    all_scores.append(eval_dict["principle_scores"][principle_id])
            
            if all_scores:
                principle_scores[principle_id] = {
                    "name": principle["name"],
                    "average_score": sum(all_scores) / len(all_scores),
                    "evaluations": len(all_scores)
                }
        
        # Find common recommendations
        all_recommendations = []
        for eval_dict in self.action_evaluations:
            all_recommendations.extend(eval_dict.get("recommendations", []))
        
        # Count recommendation occurrences
        rec_counts = {}
        for rec in all_recommendations:
            # Normalize recommendation text for counting
            normalized = rec.lower().strip()
            if normalized not in rec_counts:
                rec_counts[normalized] = 0
            rec_counts[normalized] += 1
        
        # Sort by frequency
        common_recs = sorted(
            [{"text": rec, "count": count} for rec, count in rec_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]  # Top 10 recommendations
        
        return {
            "total_evaluations": total,
            "aligned_count": aligned_count,
            "aligned_percentage": aligned_percentage,
            "average_score": average_score,
            "principle_scores": principle_scores,
            "common_recommendations": common_recs
        }


class LLMPrincipleEvaluator(PrincipleEvaluator):
    """
    A principle evaluator that uses a single LLM for evaluations.
    
    This evaluator implements the PrincipleEvaluator interface using
    a specific LLM adapter and model.
    """
    
    def __init__(self, adapter: BaseLLMAdapter, model: Optional[str] = None) -> None:
        """
        Initialize the LLM principle evaluator.
        
        Args:
            adapter: The LLM adapter to use for evaluations
            model: Optional specific model to use (defaults to adapter's default)
        """
        self.adapter = adapter
        self.model = model
        
    async def evaluate_principle(
        self,
        principle: Dict[str, Any],
        context: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a principle using the LLM.
        
        Args:
            principle: The principle to evaluate
            context: The context for evaluation
            options: Additional options for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        options = options or {}
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(principle, context)
        
        try:
            # Send to LLM for evaluation
            response = await self.adapter.send_request(
                prompt=prompt,
                model=self.model,
                system_message="You are a principle evaluation assistant. Evaluate whether the given context aligns with the specified principle.",
                temperature=0.3  # Lower temperature for more consistent evaluations
            )
            
            # Process the response
            result_text = self.adapter.process_response(response)
            
            # Parse the evaluation result
            evaluation_result = self._parse_evaluation_result(result_text, principle)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating principle: {str(e)}")
            return {
                "principle_id": principle.get("id", "unknown"),
                "evaluation": "error",
                "confidence": 0.0,
                "reasons": [f"Error during evaluation: {str(e)}"],
                "metadata": {
                    "evaluated": False,
                    "error": str(e)
                }
            }
    
    async def evaluate_principles(
        self,
        principles: List[Dict[str, Any]],
        context: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple principles.
        
        Args:
            principles: List of principles to evaluate
            context: The context for evaluation
            options: Additional options for evaluation
            
        Returns:
            List of evaluation results
        """
        results = []
        for principle in principles:
            result = await self.evaluate_principle(principle, context, options)
            results.append(result)
        return results
    
    def _create_evaluation_prompt(
        self,
        principle: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Create an evaluation prompt for the LLM.
        
        Args:
            principle: The principle to evaluate
            context: The context for evaluation
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Please evaluate whether the following context aligns with the given principle.

PRINCIPLE:
Name: {principle.get('name', 'Unknown')}
Description: {principle.get('description', 'No description provided')}
Type: {principle.get('type', 'general')}

CONTEXT:
{json.dumps(context, indent=2)}

EVALUATION CRITERIA:
- Does the context demonstrate alignment with the principle?
- Are there any violations or conflicts with the principle?
- How confident are you in this evaluation?

Please provide your evaluation in the following format:
EVALUATION: [aligned/partially_aligned/not_aligned/unknown]
CONFIDENCE: [0.0-1.0]
REASONS:
- Reason 1
- Reason 2
- ...
"""
        return prompt.strip()
    
    def _parse_evaluation_result(
        self,
        result_text: str,
        principle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse the LLM's evaluation result.
        
        Args:
            result_text: The raw text from the LLM
            principle: The principle that was evaluated
            
        Returns:
            Structured evaluation result
        """
        # Default result
        result = {
            "principle_id": principle.get("id", "unknown"),
            "evaluation": "unknown",
            "confidence": 0.5,
            "reasons": [],
            "metadata": {
                "evaluated": True,
                "raw_response": result_text
            }
        }
        
        try:
            # Parse evaluation
            eval_match = re.search(r"EVALUATION:\s*(\w+)", result_text, re.IGNORECASE)
            if eval_match:
                evaluation = eval_match.group(1).lower()
                if evaluation in ["aligned", "partially_aligned", "not_aligned", "unknown"]:
                    result["evaluation"] = evaluation
            
            # Parse confidence
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", result_text, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                result["confidence"] = max(0.0, min(1.0, confidence))
            
            # Parse reasons
            reasons_match = re.search(r"REASONS:(.*?)(?:\n\n|$)", result_text, re.DOTALL | re.IGNORECASE)
            if reasons_match:
                reasons_text = reasons_match.group(1).strip()
                reasons = re.findall(r"[-*]\s*(.+)", reasons_text)
                result["reasons"] = [reason.strip() for reason in reasons if reason.strip()]
                
        except Exception as e:
            logger.error(f"Error parsing evaluation result: {str(e)}")
            result["reasons"].append(f"Error parsing result: {str(e)}")
        
        return result
