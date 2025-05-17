"""
EmojiSequenceOptimizer Component for Adaptive Bridge Builder Agent

This component optimizes emoji sequences for readability, clarity, and effectiveness,
balancing expressiveness with conciseness and reducing ambiguity based on different
optimization profiles and communication contexts.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from dataclasses import dataclass, field

from emoji_knowledge_base import (
    EmojiKnowledgeBase,
    EmojiDomain,
    CulturalContext,
    FamiliarityLevel,
    SentimentValue,
    EmojiCategory
)


class OptimizationProfile(Enum):
    """Optimization profiles for different communication needs."""
    PRECISE = "precise"        # Maximum clarity, minimal ambiguity
    CONCISE = "concise"        # Brevity, fewer emojis
    EXPRESSIVE = "expressive"  # Rich expression, more emotional
    UNIVERSAL = "universal"    # Maximally recognizable across cultures
    TECHNICAL = "technical"    # Optimized for technical communication
    BUSINESS = "business"      # Optimized for business communication
    SOCIAL = "social"          # Optimized for social/casual communication
    CULTURAL = "cultural"      # Culturally adaptive


class OptimizationWeight(Enum):
    """Weights for different optimization factors."""
    EXPRESSIVENESS = "expressiveness"    # How expressive the sequence is
    CONCISENESS = "conciseness"          # How brief the sequence is
    CLARITY = "clarity"                  # How clear/unambiguous the sequence is
    UNIVERSALITY = "universality"        # How universal/recognized the sequence is
    EMOTIONALITY = "emotionality"        # How emotionally rich the sequence is
    PRECISION = "precision"              # How precisely it captures the meaning
    CREATIVITY = "creativity"            # How creative/novel the sequence is
    CONSISTENCY = "consistency"          # How consistent with past usage


class GroupingStrategy(Enum):
    """Strategies for grouping emojis for readability."""
    SEMANTIC = "semantic"      # Group by meaning
    SYNTACTIC = "syntactic"    # Group by grammatical function
    VISUAL = "visual"          # Group by visual similarity
    NONE = "none"              # No grouping


@dataclass
class OptimizationContext:
    """Context for emoji sequence optimization."""
    domain: EmojiDomain = EmojiDomain.GENERAL
    cultural_context: CulturalContext = CulturalContext.GLOBAL
    profile: OptimizationProfile = OptimizationProfile.PRECISE
    min_familiarity: FamiliarityLevel = FamiliarityLevel.COMMON
    grouping_strategy: GroupingStrategy = GroupingStrategy.SEMANTIC
    max_sequence_length: Optional[int] = None
    min_sequence_length: Optional[int] = None
    space_between_groups: bool = True
    weights: Dict[OptimizationWeight, float] = field(default_factory=dict)
    allowed_emojis: Optional[Set[str]] = None
    forbidden_emojis: Optional[Set[str]] = None
    required_emojis: Optional[Set[str]] = None
    prior_sequences: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default weights if not provided."""
        if not self.weights:
            # Default weights for each profile
            if self.profile == OptimizationProfile.PRECISE:
                self.weights = {
                    OptimizationWeight.CLARITY: 1.0,
                    OptimizationWeight.PRECISION: 0.9,
                    OptimizationWeight.UNIVERSALITY: 0.7,
                    OptimizationWeight.CONCISENESS: 0.5,
                    OptimizationWeight.EXPRESSIVENESS: 0.3,
                    OptimizationWeight.EMOTIONALITY: 0.2,
                    OptimizationWeight.CREATIVITY: 0.1,
                    OptimizationWeight.CONSISTENCY: 0.6
                }
            elif self.profile == OptimizationProfile.CONCISE:
                self.weights = {
                    OptimizationWeight.CONCISENESS: 1.0,
                    OptimizationWeight.CLARITY: 0.8,
                    OptimizationWeight.UNIVERSALITY: 0.7,
                    OptimizationWeight.PRECISION: 0.6,
                    OptimizationWeight.EXPRESSIVENESS: 0.4,
                    OptimizationWeight.EMOTIONALITY: 0.3,
                    OptimizationWeight.CREATIVITY: 0.2,
                    OptimizationWeight.CONSISTENCY: 0.5
                }
            elif self.profile == OptimizationProfile.EXPRESSIVE:
                self.weights = {
                    OptimizationWeight.EXPRESSIVENESS: 1.0,
                    OptimizationWeight.EMOTIONALITY: 0.9,
                    OptimizationWeight.CREATIVITY: 0.8,
                    OptimizationWeight.CLARITY: 0.6,
                    OptimizationWeight.PRECISION: 0.5,
                    OptimizationWeight.UNIVERSALITY: 0.4,
                    OptimizationWeight.CONCISENESS: 0.2,
                    OptimizationWeight.CONSISTENCY: 0.7
                }
            elif self.profile == OptimizationProfile.UNIVERSAL:
                self.weights = {
                    OptimizationWeight.UNIVERSALITY: 1.0,
                    OptimizationWeight.CLARITY: 0.8,
                    OptimizationWeight.CONCISENESS: 0.6,
                    OptimizationWeight.PRECISION: 0.7,
                    OptimizationWeight.EXPRESSIVENESS: 0.4,
                    OptimizationWeight.EMOTIONALITY: 0.3,
                    OptimizationWeight.CREATIVITY: 0.2,
                    OptimizationWeight.CONSISTENCY: 0.5
                }
            elif self.profile == OptimizationProfile.TECHNICAL:
                self.weights = {
                    OptimizationWeight.PRECISION: 1.0,
                    OptimizationWeight.CLARITY: 0.9,
                    OptimizationWeight.CONCISENESS: 0.7,
                    OptimizationWeight.UNIVERSALITY: 0.6,
                    OptimizationWeight.EXPRESSIVENESS: 0.3,
                    OptimizationWeight.EMOTIONALITY: 0.1,
                    OptimizationWeight.CREATIVITY: 0.2,
                    OptimizationWeight.CONSISTENCY: 0.8
                }
            elif self.profile == OptimizationProfile.BUSINESS:
                self.weights = {
                    OptimizationWeight.CLARITY: 1.0,
                    OptimizationWeight.CONCISENESS: 0.9,
                    OptimizationWeight.PRECISION: 0.8,
                    OptimizationWeight.UNIVERSALITY: 0.7,
                    OptimizationWeight.EXPRESSIVENESS: 0.4,
                    OptimizationWeight.EMOTIONALITY: 0.2,
                    OptimizationWeight.CREATIVITY: 0.1,
                    OptimizationWeight.CONSISTENCY: 0.6
                }
            elif self.profile == OptimizationProfile.SOCIAL:
                self.weights = {
                    OptimizationWeight.EXPRESSIVENESS: 0.9,
                    OptimizationWeight.EMOTIONALITY: 0.8,
                    OptimizationWeight.CREATIVITY: 0.7,
                    OptimizationWeight.CLARITY: 0.6,
                    OptimizationWeight.UNIVERSALITY: 0.5,
                    OptimizationWeight.CONCISENESS: 0.4,
                    OptimizationWeight.PRECISION: 0.3,
                    OptimizationWeight.CONSISTENCY: 0.6
                }
            elif self.profile == OptimizationProfile.CULTURAL:
                self.weights = {
                    OptimizationWeight.CLARITY: 0.9,
                    OptimizationWeight.UNIVERSALITY: 0.8,
                    OptimizationWeight.PRECISION: 0.7,
                    OptimizationWeight.EXPRESSIVENESS: 0.6,
                    OptimizationWeight.EMOTIONALITY: 0.5,
                    OptimizationWeight.CONCISENESS: 0.4,
                    OptimizationWeight.CREATIVITY: 0.3,
                    OptimizationWeight.CONSISTENCY: 0.6
                }
            else:
                # Default balanced weights
                self.weights = {
                    OptimizationWeight.CLARITY: 0.7,
                    OptimizationWeight.CONCISENESS: 0.7,
                    OptimizationWeight.PRECISION: 0.7,
                    OptimizationWeight.UNIVERSALITY: 0.7,
                    OptimizationWeight.EXPRESSIVENESS: 0.7,
                    OptimizationWeight.EMOTIONALITY: 0.7,
                    OptimizationWeight.CREATIVITY: 0.7,
                    OptimizationWeight.CONSISTENCY: 0.7
                }


@dataclass
class OptimizationResult:
    """Result of emoji sequence optimization."""
    original_sequence: str
    optimized_sequence: str
    optimization_score: float
    scores: Dict[OptimizationWeight, float]
    substitutions: List[Tuple[str, str, str]]  # (original_emoji, new_emoji, reason)
    removals: List[Tuple[str, str]]  # (removed_emoji, reason)
    additions: List[Tuple[str, str]]  # (added_emoji, reason)
    rearrangements: List[Tuple[str, str, str]]  # (from_position, to_position, reason)
    groups: List[str]  # Groups of emojis in the final sequence
    profile_used: OptimizationProfile
    context: OptimizationContext


class EmojiSequenceOptimizer:
    """
    Optimizes emoji sequences for readability, clarity, and effectiveness.
    
    This component balances expressiveness with conciseness, reduces ambiguity,
    prioritizes universally recognized emoji, and implements readability patterns.
    """
    
    def __init__(
        self,
        knowledge_base: Optional[EmojiKnowledgeBase] = None
    ):
        """
        Initialize the EmojiSequenceOptimizer.
        
        Args:
            knowledge_base: Optional EmojiKnowledgeBase instance.
                If None, a new instance will be created.
        """
        self.knowledge_base = knowledge_base or EmojiKnowledgeBase()
        
        # Cache for frequently used data
        self._familiarity_cache: Dict[str, FamiliarityLevel] = {}
        self._ambiguity_cache: Dict[str, float] = {}
        self._frequency_cache: Dict[str, float] = {}
        
        # Pattern for extracting individual emojis
        self.emoji_pattern = re.compile(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])')
    
    def optimize_sequence(
        self,
        emoji_sequence: str,
        context: Optional[OptimizationContext] = None
    ) -> OptimizationResult:
        """
        Optimize an emoji sequence based on the provided context.
        
        Args:
            emoji_sequence: The emoji sequence to optimize
            context: Optimization context with preferences
            
        Returns:
            OptimizationResult with the optimized sequence and metadata
        """
        # Create default context if not provided
        if context is None:
            context = OptimizationContext()
        
        # Extract individual emojis
        emojis = self.emoji_pattern.findall(emoji_sequence)
        
        if not emojis:
            return OptimizationResult(
                original_sequence=emoji_sequence,
                optimized_sequence=emoji_sequence,
                optimization_score=1.0,
                scores={},
                substitutions=[],
                removals=[],
                additions=[],
                rearrangements=[],
                groups=[emoji_sequence] if emoji_sequence else [],
                profile_used=context.profile,
                context=context
            )
        
        # Apply optimization algorithms
        result = self._optimize_sequence(emojis, context)
        
        return result
    
    def optimize_sequence_batch(
        self,
        sequences: List[str],
        context: Optional[OptimizationContext] = None
    ) -> List[OptimizationResult]:
        """
        Optimize multiple emoji sequences with the same context.
        
        Args:
            sequences: List of emoji sequences to optimize
            context: Optimization context with preferences
            
        Returns:
            List of OptimizationResult objects
        """
        results = []
        
        for sequence in sequences:
            results.append(self.optimize_sequence(sequence, context))
        
        return results
    
    def get_optimization_profiles(self) -> Dict[OptimizationProfile, Dict[OptimizationWeight, float]]:
        """
        Get the default optimization profiles with their weight configurations.
        
        Returns:
            Dictionary mapping profiles to their weight configurations
        """
        # Create temporary context objects to get default weights
        profiles = {}
        
        for profile in OptimizationProfile:
            context = OptimizationContext(profile=profile)
            profiles[profile] = dict(context.weights)
        
        return profiles
    
    def create_custom_profile(
        self,
        weights: Dict[OptimizationWeight, float]
    ) -> OptimizationContext:
        """
        Create a custom optimization profile with specified weights.
        
        Args:
            weights: Dictionary mapping weights to their values (0.0-1.0)
            
        Returns:
            OptimizationContext with the custom weights
        """
        context = OptimizationContext()
        
        # Validate and normalize weights
        for weight, value in weights.items():
            if not isinstance(weight, OptimizationWeight):
                raise ValueError(f"Invalid weight: {weight}")
            
            # Ensure weight is between 0 and 1
            normalized_value = max(0.0, min(1.0, value))
            context.weights[weight] = normalized_value
        
        return context
    
    def analyze_sequence(
        self,
        emoji_sequence: str,
        context: Optional[OptimizationContext] = None
    ) -> Dict[str, Any]:
        """
        Analyze an emoji sequence without optimizing it.
        
        Args:
            emoji_sequence: The emoji sequence to analyze
            context: Optional context with analysis preferences
            
        Returns:
            Dictionary with analysis metrics
        """
        if context is None:
            context = OptimizationContext()
        
        # Extract individual emojis
        emojis = self.emoji_pattern.findall(emoji_sequence)
        
        if not emojis:
            return {
                "emojis": [],
                "length": 0,
                "familiarity": 1.0,
                "ambiguity": 0.0,
                "universality": 1.0,
                "cultural_specificity": 0.0,
                "domain_specificity": 0.0,
                "most_ambiguous": None,
                "least_familiar": None,
            }
        
        # Calculate metrics
        analysis = {
            "emojis": emojis,
            "length": len(emojis),
            "familiarity": self._calculate_familiarity_score(emojis, context),
            "ambiguity": self._calculate_ambiguity_score(emojis, context),
            "universality": self._calculate_universality_score(emojis, context),
        }
        
        # Find most ambiguous emoji
        most_ambiguous = None
        max_ambiguity = -1
        
        for emoji in emojis:
            ambiguity = self._get_emoji_ambiguity(emoji)
            if ambiguity > max_ambiguity:
                max_ambiguity = ambiguity
                most_ambiguous = emoji
        
        analysis["most_ambiguous"] = most_ambiguous
        
        # Find least familiar emoji
        least_familiar = None
        min_familiarity = float('inf')
        
        for emoji in emojis:
            familiarity = self._get_emoji_familiarity_score(emoji)
            if familiarity < min_familiarity:
                min_familiarity = familiarity
                least_familiar = emoji
        
        analysis["least_familiar"] = least_familiar
        
        # Calculate cultural and domain specificity
        analysis["cultural_specificity"] = self._calculate_cultural_specificity(emojis, context.cultural_context)
        analysis["domain_specificity"] = self._calculate_domain_specificity(emojis, context.domain)
        
        return analysis
    
    def _optimize_sequence(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> OptimizationResult:
        """
        Core optimization algorithm for emoji sequences.
        
        Args:
            emojis: List of individual emojis
            context: Optimization context
            
        Returns:
            OptimizationResult with the optimized sequence
        """
        original_sequence = "".join(emojis)
        substitutions = []
        removals = []
        additions = []
        rearrangements = []
        
        # Step 1: Remove redundant and low-value emojis
        optimized_emojis = self._remove_redundant_emojis(emojis, context)
        
        for emoji in emojis:
            if emoji not in optimized_emojis:
                removals.append((emoji, "Redundant or low value"))
        
        # Step 2: Replace ambiguous emojis with clearer alternatives
        optimized_emojis, sub_map = self._reduce_ambiguity(optimized_emojis, context)
        substitutions.extend(sub_map)
        
        # Step 3: Ensure required emojis are present
        optimized_emojis, added = self._ensure_required_emojis(optimized_emojis, context)
        additions.extend(added)
        
        # Step 4: Apply length constraints
        optimized_emojis, removed, added = self._apply_length_constraints(optimized_emojis, context)
        removals.extend(removed)
        additions.extend(added)
        
        # Step 5: Rearrange emojis for better readability
        optimized_emojis, reordering = self._optimize_order(optimized_emojis, context)
        rearrangements.extend(reordering)
        
        # Step 6: Apply grouping strategy
        groups = self._apply_grouping(optimized_emojis, context)
        
        # Build final sequence with groups
        separator = " " if context.space_between_groups else ""
        optimized_sequence = separator.join(groups)
        
        # Calculate scores
        scores = self._calculate_scores(optimized_emojis, context)
        optimization_score = self._calculate_weighted_score(scores, context.weights)
        
        return OptimizationResult(
            original_sequence=original_sequence,
            optimized_sequence=optimized_sequence,
            optimization_score=optimization_score,
            scores=scores,
            substitutions=substitutions,
            removals=removals,
            additions=additions,
            rearrangements=rearrangements,
            groups=groups,
            profile_used=context.profile,
            context=context
        )
    
    def _remove_redundant_emojis(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> List[str]:
        """
        Remove redundant and low-value emojis.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            List of emojis with redundancies removed
        """
        # If conciseness is weighted highly, we should remove more aggressively
        conciseness_weight = context.weights.get(OptimizationWeight.CONCISENESS, 0.5)
        
        if conciseness_weight < 0.3:
            # Low conciseness weight, don't remove much
            return emojis
        
        result = []
        seen_meanings = set()
        
        for emoji in emojis:
            # Skip if it's in the forbidden list
            if context.forbidden_emojis and emoji in context.forbidden_emojis:
                continue
                
            # Always keep if it's in the required list
            if context.required_emojis and emoji in context.required_emojis:
                result.append(emoji)
                continue
            
            # Get associated meanings
            emoji_metadata = self.knowledge_base.get_emoji(emoji)
            if not emoji_metadata:
                # Unknown emoji, keep it for now
                result.append(emoji)
                continue
            
            # Get primary meaning
            primary_meaning = emoji_metadata.primary_meaning
            
            # Get domain-specific meaning if available
            domain_meaning = None
            if emoji_metadata.domain_meanings:
                domain_meaning = emoji_metadata.domain_meanings.get(context.domain)
            
            # Use domain meaning if available, otherwise primary meaning
            meaning = domain_meaning if domain_meaning else primary_meaning
            
            # Skip if we've already seen this meaning and conciseness is important
            if meaning and meaning in seen_meanings and conciseness_weight > 0.7:
                continue
            
            if meaning:
                seen_meanings.add(meaning)
            
            # Add to result
            result.append(emoji)
        
        return result
    
    def _reduce_ambiguity(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Replace ambiguous emojis with clearer alternatives.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Tuple of (optimized emojis, list of substitutions)
        """
        clarity_weight = context.weights.get(OptimizationWeight.CLARITY, 0.5)
        
        if clarity_weight < 0.3:
            # Low clarity weight, don't substitute much
            return emojis, []
        
        result = []
        substitutions = []
        
        for emoji in emojis:
            # Skip if it's in the required list
            if context.required_emojis and emoji in context.required_emojis:
                result.append(emoji)
                continue
            
            # Get ambiguity
            ambiguity = self._get_emoji_ambiguity(emoji)
            
            # If ambiguity is high and clarity is important, try to find a clearer alternative
            if ambiguity > 0.5 and clarity_weight > 0.5:
                # Find concepts associated with this emoji
                concepts = self.knowledge_base.find_concept_for_emoji(emoji)
                
                if concepts:
                    # For the first concept, find a clearer alternative
                    concept = concepts[0]
                    alternatives = self.knowledge_base.find_emojis_for_concept(
                        concept, 
                        domain=context.domain,
                        cultural_context=context.cultural_context
                    )
                    
                    # Get primary and alternative emojis
                    primary_emojis = alternatives.get("primary", [])
                    alt_emojis = alternatives.get("alternatives", [])
                    
                    # Combine and filter
                    candidates = []
                    for candidate in primary_emojis + alt_emojis:
                        # Skip the original emoji and forbidden emojis
                        if candidate == emoji or (context.forbidden_emojis and candidate in context.forbidden_emojis):
                            continue
                            
                        # Check if candidate is clearer
                        candidate_ambiguity = self._get_emoji_ambiguity(candidate)
                        
                        if candidate_ambiguity < ambiguity:
                            candidates.append((candidate, candidate_ambiguity))
                    
                    # Sort by ambiguity (lower is better)
                    candidates.sort(key=lambda x: x[1])
                    
                    if candidates:
                        # Use the clearest alternative
                        clearer_emoji = candidates[0][0]
                        substitutions.append((emoji, clearer_emoji, f"Reduced ambiguity for concept: {concept}"))
                        result.append(clearer_emoji)
                        continue
            
            # If we didn't substitute, use the original
            result.append(emoji)
        
        return result, substitutions
    
    def _ensure_required_emojis(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Ensure all required emojis are present.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Tuple of (optimized emojis, list of additions)
        """
        if not context.required_emojis:
            return emojis, []
        
        result = list(emojis)  # Copy the list
        additions = []
        
        for required in context.required_emojis:
            if required not in result:
                result.append(required)
                additions.append((required, "Required emoji"))
        
        return result, additions
    
    def _apply_length_constraints(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Apply length constraints to the emoji sequence.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Tuple of (constrained emojis, list of removals, list of additions)
        """
        result = list(emojis)  # Copy the list
        removals = []
        additions = []
        
        # Check if we need to trim
        if context.max_sequence_length and len(result) > context.max_sequence_length:
            # Calculate how many to remove
            to_remove = len(result) - context.max_sequence_length
            
            # Sort by importance (using a combination of frequency, ambiguity, and familiarity)
            importance = []
            for i, emoji in enumerate(result):
                freq = self._get_emoji_frequency(emoji)
                ambig = self._get_emoji_ambiguity(emoji)
                famil = self._get_emoji_familiarity_score(emoji)
                
                # Higher score = more important
                score = freq * 0.4 + (1 - ambig) * 0.3 + famil * 0.3
                
                # Required emojis get highest importance
                if context.required_emojis and emoji in context.required_emojis:
                    score = float('inf')
                    
                importance.append((i, emoji, score))
            
            # Sort by importance (ascending, so least important first)
            importance.sort(key=lambda x: x[2])
            
            # Remove least important
            for i in range(to_remove):
                if i < len(importance):
                    idx, emoji, _ = importance[i]
                    removals.append((emoji, "Length constraint"))
            
            # Keep only the most important
            important_indices = [idx for idx, _, _ in importance[to_remove:]]
            important_indices.sort()  # Preserve original order
            result = [result[i] for i in important_indices]
        
        # Check if we need to add more
        if context.min_sequence_length and len(result) < context.min_sequence_length:
            # Calculate how many to add
            to_add = context.min_sequence_length - len(result)
            
            # Find concepts in the current emojis
            all_concepts = []
            for emoji in result:
                concepts = self.knowledge_base.find_concept_for_emoji(emoji)
                all_concepts.extend(concepts)
            
            # Get the most frequent concepts
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Sort by frequency
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Add emojis for the most frequent concepts
            added = 0
            for concept, _ in sorted_concepts:
                if added >= to_add:
                    break
                    
                # Find emojis for this concept
                alternatives = self.knowledge_base.find_emojis_for_concept(
                    concept, 
                    domain=context.domain,
                    cultural_context=context.cultural_context
                )
                
                # Get candidates
                candidates = []
                for emoji_list in [alternatives.get("primary", []), alternatives.get("alternatives", [])]:
                    for emoji in emoji_list:
                        # Skip if already in the result or forbidden
                        if emoji in result or (context.forbidden_emojis and emoji in context.forbidden_emojis):
                            continue
                            
                        # Skip if ambiguity is too high
                        ambiguity = self._get_emoji_ambiguity(emoji)
                        if ambiguity > 0.7:
                            continue
                            
                        candidates.append(emoji)
                
                # Add one candidate
                if candidates:
                    emoji = candidates[0]
                    result.append(emoji)
                    additions.append((emoji, f"Length constraint, concept: {concept}"))
                    added += 1
        
        return result, removals, additions
    
    def _optimize_order(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Optimize the order of emojis for better readability.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Tuple of (reordered emojis, list of rearrangements)
        """
        if len(emojis) <= 1:
            return emojis, []
        
        # Start with the original order
        result = list(emojis)
        rearrangements = []
        
        # Define order based on context
        if context.profile == OptimizationProfile.PRECISE or context.profile == OptimizationProfile.TECHNICAL:
            # For precise/technical: order by specificity (most specific first)
            order = []
            for i, emoji in enumerate(result):
                # Get specificity (opposite of ambiguity)
                specificity = 1.0 - self._get_emoji_ambiguity(emoji)
                order.append((i, emoji, specificity))
            
            # Sort by specificity (descending)
            order.sort(key=lambda x: x[2], reverse=True)
            
            # Build new order
            new_order = [emoji for _, emoji, _ in order]
            
            # Record rearrangements
            for i, (old_idx, emoji, _) in enumerate(order):
                if i != old_idx:
                    rearrangements.append((str(old_idx), str(i), f"Reordered by specificity"))
            
            return new_order, rearrangements
        
        elif context.profile == OptimizationProfile.SOCIAL or context.profile == OptimizationProfile.EXPRESSIVE:
            # For social/expressive: order by emotional impact (most impactful first)
            order = []
            for i, emoji in enumerate(result):
                # Get emotional impact (combination of sentiment intensity and expressiveness)
                metadata = self.knowledge_base.get_emoji(emoji)
                if metadata:
                    intensity = metadata.intensity_score
                else:
                    intensity = 0.5
                
                order.append((i, emoji, intensity))
            
            # Sort by intensity (descending)
            order.sort(key=lambda x: x[2], reverse=True)
            
            # Build new order
            new_order = [emoji for _, emoji, _ in order]
            
            # Record rearrangements
            for i, (old_idx, emoji, _) in enumerate(order):
                if i != old_idx:
                    rearrangements.append((str(old_idx), str(i), f"Reordered by emotional impact"))
            
            return new_order, rearrangements
        
        elif context.profile == OptimizationProfile.CONCISE:
            # For concise profile, order by frequency (most frequent first)
            order = []
            for i, emoji in enumerate(result):
                frequency = self._get_emoji_frequency(emoji)
                order.append((i, emoji, frequency))
            
            # Sort by frequency (descending)
            order.sort(key=lambda x: x[2], reverse=True)
            
            # Build new order
            new_order = [emoji for _, emoji, _ in order]
            
            # Record rearrangements
            for i, (old_idx, emoji, _) in enumerate(order):
                if i != old_idx:
                    rearrangements.append((str(old_idx), str(i), f"Reordered by frequency"))
            
            return new_order, rearrangements
        
        # Default ordering (e.g., for UNIVERSAL, BUSINESS, etc.) - keep original order
        return result, rearrangements
    
    def _apply_grouping(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> List[str]:
        """
        Apply grouping strategy to the emoji sequence.
        
        Args:
            emojis: List of emojis to group
            context: Optimization context
            
        Returns:
            List of grouped emoji sequences
        """
        if not emojis:
            return []
            
        if context.grouping_strategy == GroupingStrategy.NONE:
            # No grouping, return as single group
            return ["".join(emojis)]
        
        # Number of emojis
        n_emojis = len(emojis)
        
        # For very short sequences, don't group
        if n_emojis <= 3:
            return ["".join(emojis)]
        
        if context.grouping_strategy == GroupingStrategy.SEMANTIC:
            # Group by semantic meaning
            groups = []
            current_group = [emojis[0]]
            
            for i in range(1, n_emojis):
                emoji = emojis[i]
                prev_emoji = emojis[i-1]
                
                # Check if current and previous emoji are semantically related
                related = self._are_semantically_related(emoji, prev_emoji)
                
                if related:
                    # Add to current group
                    current_group.append(emoji)
                else:
                    # Finish current group and start a new one
                    if current_group:
                        groups.append("".join(current_group))
                    current_group = [emoji]
            
            # Add the last group
            if current_group:
                groups.append("".join(current_group))
            
            return groups
            
        elif context.grouping_strategy == GroupingStrategy.VISUAL:
            # Group by visual similarity (using emoji categories)
            groups = []
            current_group = [emojis[0]]
            current_category = self._get_emoji_category(emojis[0])
            
            for i in range(1, n_emojis):
                emoji = emojis[i]
                category = self._get_emoji_category(emoji)
                
                if category == current_category:
                    # Add to current group
                    current_group.append(emoji)
                else:
                    # Finish current group and start a new one
                    if current_group:
                        groups.append("".join(current_group))
                    current_group = [emoji]
                    current_category = category
            
            # Add the last group
            if current_group:
                groups.append("".join(current_group))
            
            return groups
            
        elif context.grouping_strategy == GroupingStrategy.SYNTACTIC:
            # Group by grammatical function (e.g., subject, verb, object)
            # This is a simplified approach; a more sophisticated implementation
            # would use the EmojiGrammarSystem component
            
            # For this simple implementation, we'll use a basic heuristic:
            # - People emojis often represent subjects
            # - Action/object emojis often represent verbs/actions
            # - Other emojis often represent objects or modifiers
            
            # Group sizes (approximately balanced)
            group_size = max(2, n_emojis // 3)
            
            groups = []
            for i in range(0, n_emojis, group_size):
                group = emojis[i:min(i+group_size, n_emojis)]
                groups.append("".join(group))
            
            return groups
        
        # Fallback: return as single group
        return ["".join(emojis)]
    
    def _calculate_scores(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> Dict[OptimizationWeight, float]:
        """
        Calculate optimization scores for different factors.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Dictionary mapping optimization weights to scores
        """
        scores = {}
        
        # No emojis means perfect scores (nothing to optimize)
        if not emojis:
            for weight in OptimizationWeight:
                scores[weight] = 1.0
            return scores
        
        # Calculate expressiveness score
        expr_score = self._calculate_expressiveness_score(emojis, context)
        scores[OptimizationWeight.EXPRESSIVENESS] = expr_score
        
        # Calculate conciseness score
        conc_score = self._calculate_conciseness_score(emojis, context)
        scores[OptimizationWeight.CONCISENESS] = conc_score
        
        # Calculate clarity score
        clar_score = self._calculate_clarity_score(emojis, context)
        scores[OptimizationWeight.CLARITY] = clar_score
        
        # Calculate universality score
        univ_score = self._calculate_universality_score(emojis, context)
        scores[OptimizationWeight.UNIVERSALITY] = univ_score
        
        # Calculate emotionality score
        emot_score = self._calculate_emotionality_score(emojis, context)
        scores[OptimizationWeight.EMOTIONALITY] = emot_score
        
        # Calculate precision score
        prec_score = self._calculate_precision_score(emojis, context)
        scores[OptimizationWeight.PRECISION] = prec_score
        
        # Calculate creativity score
        crea_score = self._calculate_creativity_score(emojis, context)
        scores[OptimizationWeight.CREATIVITY] = crea_score
        
        # Calculate consistency score
        cons_score = self._calculate_consistency_score(emojis, context)
        scores[OptimizationWeight.CONSISTENCY] = cons_score
        
        return scores
    
    def _calculate_weighted_score(
        self,
        scores: Dict[OptimizationWeight, float],
        weights: Dict[OptimizationWeight, float]
    ) -> float:
        """
        Calculate weighted average of optimization scores.
        
        Args:
            scores: Dictionary mapping optimization weights to scores
            weights: Dictionary mapping optimization weights to importance weights
            
        Returns:
            Weighted average score
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for weight_type, weight_value in weights.items():
            if weight_type in scores:
                score = scores[weight_type]
                weighted_sum += score * weight_value
                total_weight += weight_value
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight
    
    def _calculate_expressiveness_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate expressiveness score (how well the sequence expresses emotions).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Expressiveness score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Count emojis with high expressiveness
        expressive_count = 0
        total_expressiveness = 0.0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata:
                # Combine intensity and sentiment strength
                intensity = metadata.intensity_score
                sentiment_strength = 0.5
                if metadata.sentiment in [SentimentValue.VERY_POSITIVE, SentimentValue.VERY_NEGATIVE]:
                    sentiment_strength = 1.0
                elif metadata.sentiment in [SentimentValue.POSITIVE, SentimentValue.NEGATIVE]:
                    sentiment_strength = 0.7
                
                expressiveness = (intensity + sentiment_strength) / 2
                total_expressiveness += expressiveness
                
                if expressiveness > 0.6:
                    expressive_count += 1
        
        # Calculate ratio of expressive emojis
        expressive_ratio = expressive_count / len(emojis) if emojis else 0
        
        # Calculate average expressiveness
        avg_expressiveness = total_expressiveness / len(emojis) if emojis else 0
        
        # Combine metrics (weighted towards average expressiveness)
        expressiveness_score = 0.4 * expressive_ratio + 0.6 * avg_expressiveness
        
        return expressiveness_score
    
    def _calculate_conciseness_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate conciseness score (brevity relative to complexity).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Conciseness score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Count unique concepts represented
        unique_concepts = set()
        for emoji in emojis:
            concepts = self.knowledge_base.find_concept_for_emoji(emoji)
            unique_concepts.update(concepts)
        
        # If there are no known concepts, estimate based on emoji count
        if not unique_concepts:
            # Assume a good ratio is about 1.5 emojis per concept
            return min(1.0, 1.5 / len(emojis)) if emojis else 1.0
        
        # Calculate the ratio of concepts to emojis
        concept_emoji_ratio = len(unique_concepts) / len(emojis)
        
        # Ideal ratio depends on profile, but generally around 0.7-1.0
        ideal_ratio = 0.8
        
        # A ratio close to the ideal scores highest
        conciseness_score = max(0.0, 1.0 - abs(concept_emoji_ratio - ideal_ratio))
        
        return conciseness_score
    
    def _calculate_clarity_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate clarity score (lack of ambiguity).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Clarity score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Calculate average ambiguity
        total_ambiguity = 0.0
        for emoji in emojis:
            ambiguity = self._get_emoji_ambiguity(emoji)
            total_ambiguity += ambiguity
        
        avg_ambiguity = total_ambiguity / len(emojis) if emojis else 0
        
        # Clarity is inverse of ambiguity
        clarity_score = 1.0 - avg_ambiguity
        
        return clarity_score
    
    def _calculate_universality_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate universality score (how universally recognized).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Universality score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Count universal and common emojis
        universal_count = 0
        common_count = 0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata:
                if metadata.familiarity_level == FamiliarityLevel.UNIVERSAL:
                    universal_count += 1
                elif metadata.familiarity_level == FamiliarityLevel.COMMON:
                    common_count += 1
        
        # Calculate weighted ratio
        weighted_count = universal_count + 0.7 * common_count
        universality_score = weighted_count / len(emojis) if emojis else 0
        
        return universality_score
    
    def _calculate_emotionality_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate emotionality score (how emotionally rich).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Emotionality score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Count emotional emojis
        emotional_count = 0
        emotional_intensity = 0.0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata:
                # Check if it's an emotional emoji
                is_emotional = (
                    metadata.category == EmojiCategory.FACE_EMOTION or
                    metadata.sentiment != SentimentValue.NEUTRAL
                )
                
                if is_emotional:
                    emotional_count += 1
                    emotional_intensity += metadata.intensity_score
        
        # Calculate ratio of emotional emojis
        emotional_ratio = emotional_count / len(emojis) if emojis else 0
        
        # Calculate average emotional intensity
        avg_intensity = emotional_intensity / emotional_count if emotional_count else 0
        
        # Combine metrics (weighted more towards ratio for emotionality)
        emotionality_score = 0.7 * emotional_ratio + 0.3 * avg_intensity
        
        return emotionality_score
    
    def _calculate_precision_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate precision score (how precisely it captures meaning).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Precision score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Check if emojis are appropriate for the domain
        domain_score = 0.0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata and metadata.domain_meanings and context.domain in metadata.domain_meanings:
                # Domain-specific meaning exists
                domain_score += 1.0
            else:
                # No domain-specific meaning, use a lower score
                domain_score += 0.5
        
        avg_domain_score = domain_score / len(emojis) if emojis else 0
        
        # Combine with clarity for precision
        clarity_score = self._calculate_clarity_score(emojis, context)
        precision_score = 0.6 * avg_domain_score + 0.4 * clarity_score
        
        return precision_score
    
    def _calculate_creativity_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate creativity score (novelty and unexpectedness).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Creativity score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        # Count unusual and specialized emojis
        unusual_count = 0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata:
                if metadata.familiarity_level in [
                    FamiliarityLevel.SPECIALIZED,
                    FamiliarityLevel.RARE,
                    FamiliarityLevel.NOVEL
                ]:
                    unusual_count += 1
        
        # Calculate ratio of unusual emojis
        unusual_ratio = unusual_count / len(emojis) if emojis else 0
        
        # Check for unique combinations not in known patterns
        combination = "".join(emojis)
        is_novel_combination = True
        
        for pattern in self.knowledge_base.patterns.values():
            if "".join(pattern.emoji_sequence) == combination:
                is_novel_combination = False
                break
        
        # Combine metrics
        creativity_score = 0.7 * unusual_ratio + 0.3 * (1.0 if is_novel_combination else 0.0)
        
        return creativity_score
    
    def _calculate_consistency_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate consistency score (with prior sequences).
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Consistency score (0.0-1.0)
        """
        if not emojis or not context.prior_sequences:
            return 1.0
            
        # Extract emojis from prior sequences
        prior_emojis = []
        for seq in context.prior_sequences:
            prior_emojis.extend(self.emoji_pattern.findall(seq))
        
        if not prior_emojis:
            return 1.0
            
        # Count emojis that appear in prior sequences
        consistent_count = 0
        for emoji in emojis:
            if emoji in prior_emojis:
                consistent_count += 1
        
        # Calculate consistency ratio
        consistency_score = consistent_count / len(emojis) if emojis else 0
        
        return consistency_score
    
    def _calculate_cultural_specificity(
        self, 
        emojis: List[str],
        cultural_context: CulturalContext
    ) -> float:
        """
        Calculate cultural specificity score.
        
        Args:
            emojis: List of emojis
            cultural_context: Target cultural context
            
        Returns:
            Cultural specificity score (0.0-1.0)
        """
        if not emojis:
            return 0.0
            
        # Count culturally specific emojis
        cultural_count = 0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata and metadata.cultural_meanings and cultural_context in metadata.cultural_meanings:
                cultural_count += 1
        
        # Calculate ratio
        cultural_ratio = cultural_count / len(emojis) if emojis else 0
        
        return cultural_ratio
    
    def _calculate_domain_specificity(
        self, 
        emojis: List[str],
        domain: EmojiDomain
    ) -> float:
        """
        Calculate domain specificity score.
        
        Args:
            emojis: List of emojis
            domain: Target domain
            
        Returns:
            Domain specificity score (0.0-1.0)
        """
        if not emojis:
            return 0.0
            
        # Count domain-specific emojis
        domain_count = 0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata and metadata.domain_meanings and domain in metadata.domain_meanings:
                domain_count += 1
        
        # Calculate ratio
        domain_ratio = domain_count / len(emojis) if emojis else 0
        
        return domain_ratio
    
    def _get_emoji_familiarity_score(self, emoji: str) -> float:
        """
        Get a numeric familiarity score for an emoji.
        
        Args:
            emoji: The emoji to score
            
        Returns:
            Familiarity score (0.0-1.0)
        """
        # Check cache first
        if emoji in self._familiarity_cache:
            return self._familiarity_cache[emoji]
            
        # Get from knowledge base
        metadata = self.knowledge_base.get_emoji(emoji)
        if metadata:
            # Convert enum to score
            if metadata.familiarity_level == FamiliarityLevel.UNIVERSAL:
                score = 1.0
            elif metadata.familiarity_level == FamiliarityLevel.COMMON:
                score = 0.8
            elif metadata.familiarity_level == FamiliarityLevel.FAMILIAR:
                score = 0.6
            elif metadata.familiarity_level == FamiliarityLevel.SPECIALIZED:
                score = 0.4
            elif metadata.familiarity_level == FamiliarityLevel.RARE:
                score = 0.2
            else:  # NOVEL
                score = 0.1
        else:
            # Default for unknown emoji
            score = 0.5
        
        # Cache the result
        self._familiarity_cache[emoji] = score
        
        return score
    
    def _get_emoji_ambiguity(self, emoji: str) -> float:
        """
        Get ambiguity score for an emoji.
        
        Args:
            emoji: The emoji to check
            
        Returns:
            Ambiguity score (0.0-1.0)
        """
        # Check cache first
        if emoji in self._ambiguity_cache:
            return self._ambiguity_cache[emoji]
            
        # Get from knowledge base
        metadata = self.knowledge_base.get_emoji(emoji)
        if metadata:
            score = metadata.ambiguity_score
        else:
            # Default for unknown emoji - assume somewhat ambiguous
            score = 0.5
        
        # Cache the result
        self._ambiguity_cache[emoji] = score
        
        return score
    
    def _get_emoji_frequency(self, emoji: str) -> float:
        """
        Get usage frequency score for an emoji.
        
        Args:
            emoji: The emoji to check
            
        Returns:
            Frequency score (0.0-1.0)
        """
        # Check cache first
        if emoji in self._frequency_cache:
            return self._frequency_cache[emoji]
            
        # Get from knowledge base
        metadata = self.knowledge_base.get_emoji(emoji)
        if metadata:
            score = metadata.frequency_score
        else:
            # Default for unknown emoji
            score = 0.3
        
        # Cache the result
        self._frequency_cache[emoji] = score
        
        return score
    
    def _get_emoji_category(self, emoji: str) -> EmojiCategory:
        """
        Get category for an emoji.
        
        Args:
            emoji: The emoji to check
            
        Returns:
            EmojiCategory
        """
        # Get from knowledge base
        metadata = self.knowledge_base.get_emoji(emoji)
        if metadata:
            return metadata.category
        else:
            # Default for unknown emoji
            return EmojiCategory.OBJECT
    
    def _are_semantically_related(self, emoji1: str, emoji2: str) -> bool:
        """
        Check if two emojis are semantically related.
        
        Args:
            emoji1: First emoji
            emoji2: Second emoji
            
        Returns:
            True if related, False otherwise
        """
        # Get concepts for both emojis
        concepts1 = self.knowledge_base.find_concept_for_emoji(emoji1)
        concepts2 = self.knowledge_base.find_concept_for_emoji(emoji2)
        
        # Check for common concepts
        for concept in concepts1:
            if concept in concepts2:
                return True
        
        # Check if either emoji is in the other's common prefixes/suffixes
        metadata1 = self.knowledge_base.get_emoji(emoji1)
        metadata2 = self.knowledge_base.get_emoji(emoji2)
        
        if metadata1:
            if emoji2 in metadata1.common_prefixes or emoji2 in metadata1.common_suffixes:
                return True
                
        if metadata2:
            if emoji1 in metadata2.common_prefixes or emoji1 in metadata2.common_suffixes:
                return True
                
        # Check for category relationship
        cat1 = self._get_emoji_category(emoji1)
        cat2 = self._get_emoji_category(emoji2)
        
        if cat1 == cat2:
            return True
            
        return False
    
    def _calculate_familiarity_score(
        self,
        emojis: List[str],
        context: OptimizationContext
    ) -> float:
        """
        Calculate overall familiarity score for a sequence.
        
        Args:
            emojis: List of emojis
            context: Optimization context
            
        Returns:
            Familiarity score (0.0-1.0)
        """
        if not emojis:
            return 1.0
            
        total_familiarity = 0.0
        
        for emoji in emojis:
            total_familiarity += self._get_emoji_familiarity_score(emoji)
        
        avg_familiarity = total_familiarity / len(emojis) if emojis else 0
        
        return avg_familiarity
