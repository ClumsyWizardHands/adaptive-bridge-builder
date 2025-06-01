#!/usr/bin/env python3
"""
AI Principles Gym Adapter for Adaptive Bridge Builder

This module enables the Adaptive Bridge Builder to process scenarios from the
AI Principles Gym, leveraging its principle engine, emotional intelligence,
and decision-making capabilities.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import uuid

from principle_engine import PrincipleEngine
from emotional_intelligence import EmotionalIntelligenceEngine
from ethical_dilemma_resolver import EthicalDilemmaResolver
from conflict_resolver import ConflictResolver, ConflictSeverity
from fairness_evaluator import FairnessEvaluator
from learning_system import LearningSystem
from continuous_evolution_system import ContinuousEvolutionSystem

logger = logging.getLogger(__name__)


class ScenarioType:
    """Enum-like class for scenario types"""
    ETHICAL_DILEMMA = "ETHICAL_DILEMMA"
    RESOURCE_ALLOCATION = "RESOURCE_ALLOCATION"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    TRUST_BUILDING = "TRUST_BUILDING"
    CRISIS_MANAGEMENT = "CRISIS_MANAGEMENT"
    COLLABORATION = "COLLABORATION"


class GymScenarioHandler:
    """Handles AI Principles Gym scenarios using Bridge Builder capabilities"""
    
    def __init__(self, bridge_builder):
        """Initialize with reference to the main bridge builder"""
        self.bridge = bridge_builder
        self.principle_engine = bridge_builder.principle_engine if hasattr(bridge_builder, 'principle_engine') else PrincipleEngine()
        self.emotional_intelligence = EmotionalIntelligenceEngine()
        self.ethical_resolver = EthicalDilemmaResolver(self.principle_engine)
        self.conflict_resolver = ConflictResolver()
        self.fairness_evaluator = FairnessEvaluator()
        self.learning_system = LearningSystem()
        self.evolution_system = ContinuousEvolutionSystem()
        
        # Track scenario history for learning
        self.scenario_history = []
        
    def process_scenario(self, gym_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a scenario from the AI Principles Gym"""
        try:
            scenario = gym_request.get('scenario', {})
            history = gym_request.get('history', [])
            metadata = gym_request.get('metadata', {})
            
            # Extract scenario details
            scenario_id = scenario.get('execution_id', str(uuid.uuid4()))
            description = scenario.get('description', '')
            actors = scenario.get('actors', [])
            resources = scenario.get('resources', [])
            constraints = scenario.get('constraints', [])
            options = scenario.get('choice_options', [])
            time_limit = scenario.get('time_limit', 30)
            archetype = scenario.get('archetype', 'GENERAL')
            stress_level = scenario.get('stress_level', 0.5)
            
            logger.info(f"Processing scenario {scenario_id}: {archetype}")
            
            # Analyze emotional context
            emotional_context = self._analyze_emotional_context(
                description, stress_level, history
            )
            
            # Evaluate each option
            evaluations = self._evaluate_options(
                options, description, actors, resources, 
                constraints, archetype, emotional_context
            )
            
            # Select best option
            best_option = self._select_best_option(evaluations, archetype)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                best_option, scenario, evaluations, emotional_context
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                best_option, evaluations, stress_level
            )
            
            # Learn from this scenario
            self._update_learning(scenario, best_option, reasoning)
            
            # Build response
            response = {
                "action": best_option['id'],
                "reasoning": reasoning,
                "confidence": confidence
            }
            
            # Add optional target if applicable
            target = self._determine_target(best_option, actors, resources)
            if target:
                response["target"] = target
            
            logger.info(f"Selected action: {best_option['id']} with confidence: {confidence}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing scenario: {e}")
            # Return a safe default response
            return self._create_fallback_response(gym_request)
    
    def _analyze_emotional_context(self, description: str, stress_level: float, 
                                 history: List[Dict]) -> Dict[str, Any]:
        """Analyze the emotional context of the scenario"""
        # Analyze description for emotional content
        emotional_analysis = self.emotional_intelligence.analyze_text(description)
        
        # Consider stress level
        emotional_analysis['stress_level'] = stress_level
        
        # Analyze history for patterns
        if history:
            recent_emotions = []
            for event in history[-3:]:  # Last 3 events
                if 'reasoning' in event:
                    event_emotion = self.emotional_intelligence.analyze_text(
                        event['reasoning']
                    )
                    recent_emotions.append(event_emotion)
            
            emotional_analysis['historical_context'] = recent_emotions
        
        return emotional_analysis
    
    def _evaluate_options(self, options: List[Dict], description: str,
                         actors: List[str], resources: List[str],
                         constraints: List[str], archetype: str,
                         emotional_context: Dict) -> List[Dict]:
        """Evaluate each option using multiple criteria"""
        evaluations = []
        
        for option in options:
            evaluation = {
                'option': option,
                'scores': {},
                'analysis': {}
            }
            
            # Principle alignment score
            principle_score = self.principle_engine.evaluate_message({
                'action': option['name'],
                'description': option['description'],
                'context': description,
                'constraints': constraints
            })
            evaluation['scores']['principle'] = principle_score
            
            # Fairness evaluation
            fairness_score = self.fairness_evaluator.evaluate_fairness(
                proposed_action=option['description'],
                affected_parties=actors,
                context={'resources': resources, 'constraints': constraints}
            )
            evaluation['scores']['fairness'] = fairness_score.score * 100
            
            # Archetype-specific evaluation
            if archetype == ScenarioType.ETHICAL_DILEMMA:
                ethical_eval = self.ethical_resolver.resolve_dilemma(
                    situation=description,
                    option_a=option['name'],
                    option_b="alternative",  # Simplified for single option eval
                    context={'actors': actors, 'resources': resources}
                )
                evaluation['scores']['ethical'] = ethical_eval.get('alignment_score', 50)
                evaluation['analysis']['ethical'] = ethical_eval
            
            elif archetype == ScenarioType.RESOURCE_ALLOCATION:
                # Evaluate resource efficiency
                resource_score = self._evaluate_resource_efficiency(
                    option, resources, actors
                )
                evaluation['scores']['efficiency'] = resource_score
            
            elif archetype == ScenarioType.CONFLICT_RESOLUTION:
                # Use conflict resolver
                conflict_analysis = self._analyze_conflict_resolution(
                    option, description, actors
                )
                evaluation['scores']['conflict_resolution'] = conflict_analysis['score']
                evaluation['analysis']['conflict'] = conflict_analysis
            
            # Emotional appropriateness
            emotional_score = self._evaluate_emotional_appropriateness(
                option, emotional_context
            )
            evaluation['scores']['emotional'] = emotional_score
            
            # Calculate composite score
            weights = self._get_archetype_weights(archetype)
            composite_score = sum(
                evaluation['scores'].get(key, 0) * weight
                for key, weight in weights.items()
            )
            evaluation['composite_score'] = composite_score
            
            evaluations.append(evaluation)
        
        return evaluations
    
    def _select_best_option(self, evaluations: List[Dict], archetype: str) -> Dict:
        """Select the best option based on evaluations"""
        # Sort by composite score
        sorted_evals = sorted(evaluations, key=lambda x: x['composite_score'], reverse=True)
        
        # Check if top options are very close
        if len(sorted_evals) > 1:
            score_diff = sorted_evals[0]['composite_score'] - sorted_evals[1]['composite_score']
            
            # If very close, use additional criteria
            if score_diff < 5:  # Within 5 points
                return self._break_tie(sorted_evals[:2], archetype)
        
        return sorted_evals[0]['option']
    
    def _generate_reasoning(self, selected_option: Dict, scenario: Dict,
                          evaluations: List[Dict], emotional_context: Dict) -> str:
        """Generate comprehensive reasoning for the decision"""
        # Find the evaluation for the selected option
        selected_eval = next(
            e for e in evaluations if e['option']['id'] == selected_option['id']
        )
        
        reasoning_parts = []
        
        # Start with the action and its purpose
        reasoning_parts.append(
            f"I choose to {selected_option['name']} because {selected_option['description']}"
        )
        
        # Add principle-based reasoning
        principle_score = selected_eval['scores'].get('principle', 0)
        if principle_score > 80:
            reasoning_parts.append(
                "This action strongly aligns with core principles of fairness, harmony, and adaptability."
            )
        elif principle_score > 60:
            reasoning_parts.append(
                "This action reasonably upholds our guiding principles while addressing the situation."
            )
        
        # Add fairness consideration
        fairness_score = selected_eval['scores'].get('fairness', 0)
        if fairness_score > 70:
            reasoning_parts.append(
                "It ensures equitable treatment for all parties involved."
            )
        
        # Add archetype-specific reasoning
        archetype = scenario.get('archetype', 'GENERAL')
        if archetype == ScenarioType.ETHICAL_DILEMMA:
            ethical_analysis = selected_eval.get('analysis', {}).get('ethical', {})
            if ethical_analysis:
                reasoning_parts.append(
                    f"From an ethical perspective, {ethical_analysis.get('rationale', 'this balances competing moral considerations.')}"
                )
        
        # Consider constraints
        constraints = scenario.get('constraints', [])
        if constraints:
            reasoning_parts.append(
                f"This approach respects the constraints: {', '.join(constraints[:2])}"
                + (" and others" if len(constraints) > 2 else "")
            )
        
        # Add emotional intelligence insight
        if emotional_context.get('stress_level', 0) > 0.7:
            reasoning_parts.append(
                "Given the high-stress nature of this situation, this measured response helps maintain stability."
            )
        
        # Reference learning from past scenarios
        if self.scenario_history:
            similar_past = self._find_similar_scenarios(scenario)
            if similar_past:
                reasoning_parts.append(
                    "Past experience with similar scenarios supports this approach."
                )
        
        return " ".join(reasoning_parts)
    
    def _calculate_confidence(self, selected_option: Dict, evaluations: List[Dict],
                            stress_level: float) -> float:
        """Calculate confidence in the decision"""
        # Find evaluation for selected option
        selected_eval = next(
            e for e in evaluations if e['option']['id'] == selected_option['id']
        )
        
        # Base confidence on composite score
        base_confidence = min(selected_eval['composite_score'] / 100, 1.0)
        
        # Adjust for score distribution
        scores = [e['composite_score'] for e in evaluations]
        if len(scores) > 1:
            # Higher confidence if clear winner
            score_std = self._calculate_std_dev(scores)
            if score_std > 20:  # Clear separation
                base_confidence *= 1.1
            elif score_std < 10:  # Very close options
                base_confidence *= 0.9
        
        # Adjust for stress level
        stress_adjustment = 1 - (stress_level * 0.2)  # High stress reduces confidence
        
        # Adjust based on principle alignment
        principle_score = selected_eval['scores'].get('principle', 0)
        principle_adjustment = 0.8 + (principle_score / 500)  # 0.8 to 1.0
        
        # Calculate final confidence
        confidence = base_confidence * stress_adjustment * principle_adjustment
        
        # Ensure within bounds
        return max(0.1, min(0.95, confidence))
    
    def _determine_target(self, option: Dict, actors: List[str], 
                         resources: List[str]) -> Optional[str]:
        """Determine the target of the action if applicable"""
        option_desc = option.get('description', '').lower()
        
        # Check if action mentions specific actors
        for actor in actors:
            if actor.lower() in option_desc:
                return actor
        
        # Check if action mentions specific resources
        for resource in resources:
            if resource.lower() in option_desc:
                return resource
        
        # Default: no specific target
        return None
    
    def _update_learning(self, scenario: Dict, selected_option: Dict, reasoning: str):
        """Update learning systems with this experience"""
        # Record in history
        self.scenario_history.append({
            'scenario_id': scenario.get('execution_id'),
            'archetype': scenario.get('archetype'),
            'selected_action': selected_option['id'],
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update evolution system
        self.evolution_system.track_orchestration_pattern(
            pattern_type="scenario_decision",
            context={
                'archetype': scenario.get('archetype'),
                'constraints': len(scenario.get('constraints', [])),
                'options': len(scenario.get('choice_options', []))
            },
            outcome={'action': selected_option['id']},
            effectiveness_score=0.8  # Placeholder, would be based on feedback
        )
    
    def _create_fallback_response(self, gym_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe fallback response if processing fails"""
        options = gym_request.get('scenario', {}).get('choice_options', [])
        
        if not options:
            return {
                "action": "default",
                "reasoning": "Unable to process scenario fully, taking default action",
                "confidence": 0.1
            }
        
        # Choose first option as safe default
        return {
            "action": options[0]['id'],
            "reasoning": "Taking conservative approach due to processing constraints",
            "confidence": 0.3
        }
    
    def _get_archetype_weights(self, archetype: str) -> Dict[str, float]:
        """Get scoring weights based on scenario archetype"""
        weights = {
            ScenarioType.ETHICAL_DILEMMA: {
                'principle': 0.3,
                'ethical': 0.4,
                'fairness': 0.2,
                'emotional': 0.1
            },
            ScenarioType.RESOURCE_ALLOCATION: {
                'principle': 0.2,
                'fairness': 0.3,
                'efficiency': 0.4,
                'emotional': 0.1
            },
            ScenarioType.CONFLICT_RESOLUTION: {
                'principle': 0.2,
                'conflict_resolution': 0.4,
                'fairness': 0.2,
                'emotional': 0.2
            }
        }
        
        # Default weights
        return weights.get(archetype, {
            'principle': 0.4,
            'fairness': 0.3,
            'emotional': 0.3
        })
    
    def _evaluate_resource_efficiency(self, option: Dict, resources: List[str],
                                    actors: List[str]) -> float:
        """Evaluate how efficiently an option uses resources"""
        # Simple heuristic based on description
        description = option.get('description', '').lower()
        
        score = 50  # Base score
        
        # Positive indicators
        if any(word in description for word in ['efficient', 'optimize', 'conserve', 'share']):
            score += 20
        
        # Check if distributes among multiple actors
        actors_mentioned = sum(1 for actor in actors if actor.lower() in description)
        if actors_mentioned > 1:
            score += 15  # Encourages distribution
        
        # Negative indicators
        if any(word in description for word in ['monopolize', 'hoard', 'waste']):
            score -= 20
        
        return max(0, min(100, score))
    
    def _analyze_conflict_resolution(self, option: Dict, description: str,
                                   actors: List[str]) -> Dict[str, Any]:
        """Analyze option for conflict resolution potential"""
        option_desc = option.get('description', '')
        
        # Create simplified conflict record
        conflict = self.conflict_resolver.create_conflict_record(
            involved_parties=actors[:2] if len(actors) >= 2 else ['Party A', 'Party B'],
            description=description,
            severity=ConflictSeverity.MEDIUM
        )
        
        # Score based on resolution approach
        score = 50  # Base
        
        # Positive resolution indicators
        resolution_words = ['compromise', 'negotiate', 'mediate', 'collaborate', 
                          'understand', 'reconcile', 'bridge']
        for word in resolution_words:
            if word in option_desc.lower():
                score += 10
        
        # Negative indicators
        escalation_words = ['force', 'dominate', 'ignore', 'dismiss']
        for word in escalation_words:
            if word in option_desc.lower():
                score -= 15
        
        return {
            'score': max(0, min(100, score)),
            'approach': 'collaborative' if score > 60 else 'directive',
            'conflict': conflict
        }
    
    def _evaluate_emotional_appropriateness(self, option: Dict,
                                          emotional_context: Dict) -> float:
        """Evaluate if option is emotionally appropriate"""
        stress_level = emotional_context.get('stress_level', 0.5)
        option_desc = option.get('description', '').lower()
        
        score = 70  # Base score
        
        # High stress situations
        if stress_level > 0.7:
            # Prefer calming approaches
            if any(word in option_desc for word in ['calm', 'steady', 'careful', 'measured']):
                score += 20
            if any(word in option_desc for word in ['rush', 'aggressive', 'forceful']):
                score -= 20
        
        # Low stress situations
        elif stress_level < 0.3:
            # Can be more assertive
            if any(word in option_desc for word in ['bold', 'innovative', 'creative']):
                score += 15
        
        # Check emotional intelligence
        emotion_check = self.emotional_intelligence.analyze_text(option_desc)
        if emotion_check.get('empathy_score', 0) > 0.7:
            score += 10
        
        return max(0, min(100, score))
    
    def _break_tie(self, top_options: List[Dict], archetype: str) -> Dict:
        """Break ties between very close options"""
        # For ethical dilemmas, prefer less harm
        if archetype == ScenarioType.ETHICAL_DILEMMA:
            for opt_eval in top_options:
                desc = opt_eval['option']['description'].lower()
                if 'harm' not in desc and 'hurt' not in desc:
                    return opt_eval['option']
        
        # For resource allocation, prefer sharing
        elif archetype == ScenarioType.RESOURCE_ALLOCATION:
            for opt_eval in top_options:
                desc = opt_eval['option']['description'].lower()
                if 'share' in desc or 'distribute' in desc:
                    return opt_eval['option']
        
        # Default: choose first (highest score even if close)
        return top_options[0]['option']
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _find_similar_scenarios(self, scenario: Dict) -> List[Dict]:
        """Find similar scenarios from history"""
        archetype = scenario.get('archetype')
        similar = []
        
        for past in self.scenario_history[-10:]:  # Last 10 scenarios
            if past.get('archetype') == archetype:
                similar.append(past)
        
        return similar


class AIGymProtocolAdapter:
    """Main adapter for AI Principles Gym integration"""
    
    def __init__(self, bridge_builder):
        """Initialize the adapter with bridge builder reference"""
        self.bridge = bridge_builder
        self.handler = GymScenarioHandler(bridge_builder)
        logger.info("AI Gym Protocol Adapter initialized")
    
    def is_gym_request(self, message: Dict[str, Any]) -> bool:
        """Check if a message is from the AI Principles Gym"""
        # Check for Gym-specific structure
        if 'scenario' in message and 'metadata' in message:
            metadata = message.get('metadata', {})
            if metadata.get('framework') == 'principles_gym':
                return True
        
        return False
    
    def process_gym_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Gym request and return response"""
        logger.info("Processing AI Principles Gym request")
        
        try:
            # Process the scenario
            response = self.handler.process_scenario(message)
            
            # Log the decision
            logger.info(f"Gym scenario processed: {response['action']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing Gym request: {e}")
            # Return error response in Gym format
            return {
                "action": "error",
                "reasoning": f"Error processing scenario: {str(e)}",
                "confidence": 0.0
            }
