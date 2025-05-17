import unittest
from emoji_sequence_optimizer import (
    EmojiSequenceOptimizer,
    OptimizationContext,
    OptimizationProfile,
    OptimizationWeight,
    GroupingStrategy,
    OptimizationResult
)

from emoji_knowledge_base import (
    EmojiKnowledgeBase,
    EmojiDomain,
    CulturalContext,
    FamiliarityLevel
)

class TestEmojiSequenceOptimizer(unittest.TestCase):
    """Test cases for the EmojiSequenceOptimizer."""
    
    def setUp(self):
        """Setup for each test case."""
        # Create a knowledge base with default data
        self.knowledge_base = EmojiKnowledgeBase(load_default=True)
        
        # Create an optimizer instance
        self.optimizer = EmojiSequenceOptimizer(knowledge_base=self.knowledge_base)
        
        # Test data - emoji sequences for different contexts
        self.technical_sequence = "🚀💻🐛🔨✅"  # Deploy, code, debug, fix, success
        self.business_sequence = "📊💼📈🤝💰"  # Stats, business, growth, deal, money
        self.social_sequence = "😀👋🎉👍❤️"  # Happy, hello, celebrate, like, love
        self.complex_sequence = "🚀📈💻🐛🔥👀✅📊🤔😊🎉💯👍"  # Mixed contexts
    
    def test_basic_optimization(self):
        """Test basic sequence optimization with default context."""
        # Test with default context
        result = self.optimizer.optimize_sequence(self.technical_sequence)
        
        # Basic validations
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.optimized_sequence)
        self.assertGreaterEqual(result.optimization_score, 0.0)
        self.assertLessEqual(result.optimization_score, 1.0)
        self.assertEqual(result.original_sequence, self.technical_sequence)
        
        # Default context should be PRECISE and GENERAL
        self.assertEqual(result.profile_used, OptimizationProfile.PRECISE)
        self.assertEqual(result.context.domain, EmojiDomain.GENERAL)
    
    def test_optimization_profiles(self):
        """Test different optimization profiles."""
        # Test PRECISE profile (focused on clarity)
        precise_context = OptimizationContext(profile=OptimizationProfile.PRECISE)
        precise_result = self.optimizer.optimize_sequence(self.complex_sequence, precise_context)
        
        # Test CONCISE profile (fewer emojis)
        concise_context = OptimizationContext(profile=OptimizationProfile.CONCISE)
        concise_result = self.optimizer.optimize_sequence(self.complex_sequence, concise_context)
        
        # Test EXPRESSIVE profile (more emotional)
        expressive_context = OptimizationContext(profile=OptimizationProfile.EXPRESSIVE)
        expressive_result = self.optimizer.optimize_sequence(self.social_sequence, expressive_context)
        
        # Verify differences between profiles
        # CONCISE should typically result in fewer emojis
        concise_count = len(concise_result.optimized_sequence.replace(" ", ""))
        original_count = len(self.complex_sequence)
        self.assertLessEqual(concise_count, original_count)
        
        # EXPRESSIVE should have higher expressiveness score than PRECISE
        self.assertGreater(
            expressive_result.scores[OptimizationWeight.EXPRESSIVENESS],
            precise_result.scores[OptimizationWeight.EXPRESSIVENESS]
        )
        
        # PRECISE should have higher clarity score than EXPRESSIVE
        self.assertGreater(
            precise_result.scores[OptimizationWeight.CLARITY],
            expressive_result.scores[OptimizationWeight.CLARITY]
        )
    
    def test_domain_specific_optimization(self):
        """Test domain-specific optimization."""
        # Test technical domain
        tech_context = OptimizationContext(
            domain=EmojiDomain.TECHNICAL,
            profile=OptimizationProfile.PRECISE
        )
        tech_result = self.optimizer.optimize_sequence(self.complex_sequence, tech_context)
        
        # Test business domain
        business_context = OptimizationContext(
            domain=EmojiDomain.BUSINESS,
            profile=OptimizationProfile.BUSINESS
        )
        business_result = self.optimizer.optimize_sequence(self.complex_sequence, business_context)
        
        # Test social domain
        social_context = OptimizationContext(
            domain=EmojiDomain.SOCIAL,
            profile=OptimizationProfile.SOCIAL
        )
        social_result = self.optimizer.optimize_sequence(self.complex_sequence, social_context)
        
        # Verify domain influences optimization
        self.assertEqual(tech_result.context.domain, EmojiDomain.TECHNICAL)
        self.assertEqual(business_result.context.domain, EmojiDomain.BUSINESS)
        self.assertEqual(social_result.context.domain, EmojiDomain.SOCIAL)
    
    def test_cultural_context_adaptation(self):
        """Test cultural context adaptation."""
        # Test global context
        global_context = OptimizationContext(
            cultural_context=CulturalContext.GLOBAL,
            profile=OptimizationProfile.UNIVERSAL
        )
        global_result = self.optimizer.optimize_sequence(self.social_sequence, global_context)
        
        # Test western context
        western_context = OptimizationContext(
            cultural_context=CulturalContext.WESTERN,
            profile=OptimizationProfile.UNIVERSAL
        )
        western_result = self.optimizer.optimize_sequence(self.social_sequence, western_context)
        
        # Test eastern Asian context
        eastern_context = OptimizationContext(
            cultural_context=CulturalContext.EASTERN_ASIAN,
            profile=OptimizationProfile.UNIVERSAL
        )
        eastern_result = self.optimizer.optimize_sequence(self.social_sequence, eastern_context)
        
        # Verify cultural context influences optimization
        self.assertEqual(global_result.context.cultural_context, CulturalContext.GLOBAL)
        self.assertEqual(western_result.context.cultural_context, CulturalContext.WESTERN)
        self.assertEqual(eastern_result.context.cultural_context, CulturalContext.EASTERN_ASIAN)
    
    def test_grouping_strategies(self):
        """Test different grouping strategies."""
        # Test semantic grouping
        semantic_context = OptimizationContext(
            grouping_strategy=GroupingStrategy.SEMANTIC,
            space_between_groups=True
        )
        semantic_result = self.optimizer.optimize_sequence(self.complex_sequence, semantic_context)
        
        # Test visual grouping
        visual_context = OptimizationContext(
            grouping_strategy=GroupingStrategy.VISUAL,
            space_between_groups=True
        )
        visual_result = self.optimizer.optimize_sequence(self.complex_sequence, visual_context)
        
        # Test syntactic grouping
        syntactic_context = OptimizationContext(
            grouping_strategy=GroupingStrategy.SYNTACTIC,
            space_between_groups=True
        )
        syntactic_result = self.optimizer.optimize_sequence(self.complex_sequence, syntactic_context)
        
        # Test no grouping
        no_group_context = OptimizationContext(
            grouping_strategy=GroupingStrategy.NONE
        )
        no_group_result = self.optimizer.optimize_sequence(self.complex_sequence, no_group_context)
        
        # Verify grouping strategy affects result
        # Grouped results should have spaces (unless no grouping)
        self.assertIn(" ", semantic_result.optimized_sequence)
        self.assertIn(" ", visual_result.optimized_sequence)
        self.assertIn(" ", syntactic_result.optimized_sequence)
        self.assertNotIn(" ", no_group_result.optimized_sequence)
        
        # Verify groups are created
        self.assertGreater(len(semantic_result.groups), 1)
        self.assertGreater(len(visual_result.groups), 1)
        self.assertGreater(len(syntactic_result.groups), 1)
        self.assertEqual(len(no_group_result.groups), 1)
    
    def test_custom_optimization_profile(self):
        """Test creating and using custom optimization profiles."""
        # Create custom weights
        custom_weights = {
            OptimizationWeight.PRECISION: 0.9,
            OptimizationWeight.CLARITY: 0.8,
            OptimizationWeight.UNIVERSALITY: 0.7,
            OptimizationWeight.EXPRESSIVENESS: 0.7,
            OptimizationWeight.EMOTIONALITY: 0.5,
            OptimizationWeight.CONCISENESS: 0.6,
            OptimizationWeight.CREATIVITY: 0.4,
            OptimizationWeight.CONSISTENCY: 0.7
        }
        
        # Create custom optimization context
        custom_context = self.optimizer.create_custom_profile(custom_weights)
        custom_context.domain = EmojiDomain.TECHNICAL
        
        # Optimize with custom profile
        custom_result = self.optimizer.optimize_sequence(self.complex_sequence, custom_context)
        
        # Verify custom weights are applied
        for weight, value in custom_weights.items():
            self.assertEqual(custom_context.weights[weight], value)
    
    def test_sequence_analysis(self):
        """Test analyzing emoji sequences without optimizing."""
        # Analyze a simple sequence
        analysis = self.optimizer.analyze_sequence(self.technical_sequence)
        
        # Verify analysis results
        self.assertEqual(len(analysis["emojis"]), 5)
        self.assertIn("familiarity", analysis)
        self.assertIn("ambiguity", analysis)
        self.assertIn("universality", analysis)
        self.assertIn("cultural_specificity", analysis)
        self.assertIn("domain_specificity", analysis)
        
        # Analyze with specific context
        tech_analysis = self.optimizer.analyze_sequence(
            self.technical_sequence,
            OptimizationContext(domain=EmojiDomain.TECHNICAL)
        )
        
        # Domain-specific analysis should include domain specificity
        self.assertGreaterEqual(tech_analysis["domain_specificity"], 0.0)
        self.assertIsNotNone(tech_analysis["most_ambiguous"])
        self.assertIsNotNone(tech_analysis["least_familiar"])
    
    def test_optimization_with_constraints(self):
        """Test optimization with various constraints."""
        # Test length constraints
        length_context = OptimizationContext(
            profile=OptimizationProfile.PRECISE,
            max_sequence_length=3
        )
        length_result = self.optimizer.optimize_sequence(self.complex_sequence, length_context)
        
        # Verify length constraint is respected
        optimized_emojis = [c for c in length_result.optimized_sequence if c not in " "]
        self.assertLessEqual(len(optimized_emojis), 3)
        
        # Test required emojis
        required_context = OptimizationContext(
            profile=OptimizationProfile.CONCISE,
            required_emojis={"✅", "🚀"}
        )
        required_result = self.optimizer.optimize_sequence(self.complex_sequence, required_context)
        
        # Verify required emojis are included
        self.assertIn("✅", required_result.optimized_sequence)
        self.assertIn("🚀", required_result.optimized_sequence)
        
        # Test forbidden emojis
        forbidden_context = OptimizationContext(
            profile=OptimizationProfile.PRECISE,
            forbidden_emojis={"🐛", "👀"}
        )
        forbidden_result = self.optimizer.optimize_sequence(self.complex_sequence, forbidden_context)
        
        # Verify forbidden emojis are excluded
        self.assertNotIn("🐛", forbidden_result.optimized_sequence)
        self.assertNotIn("👀", forbidden_result.optimized_sequence)
        
        # Test minimum familiarity constraint
        familiarity_context = OptimizationContext(
            profile=OptimizationProfile.UNIVERSAL,
            min_familiarity=FamiliarityLevel.UNIVERSAL
        )
        familiarity_result = self.optimizer.optimize_sequence(self.complex_sequence, familiarity_context)
        
        # Verify substitutions and removals are recorded
        self.assertIsInstance(familiarity_result.substitutions, list)
        self.assertIsInstance(familiarity_result.removals, list)
        self.assertIsInstance(familiarity_result.additions, list)
        self.assertIsInstance(familiarity_result.rearrangements, list)
    
    def test_batch_optimization(self):
        """Test optimizing multiple sequences in batch."""
        # Create test sequences
        sequences = [
            self.technical_sequence,
            self.business_sequence,
            self.social_sequence
        ]
        
        # Create context
        context = OptimizationContext(profile=OptimizationProfile.UNIVERSAL)
        
        # Optimize batch
        results = self.optimizer.optimize_sequence_batch(sequences, context)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], OptimizationResult)
        self.assertIsInstance(results[1], OptimizationResult)
        self.assertIsInstance(results[2], OptimizationResult)
        
        # Verify each result matches its original
        self.assertEqual(results[0].original_sequence, self.technical_sequence)
        self.assertEqual(results[1].original_sequence, self.business_sequence)
        self.assertEqual(results[2].original_sequence, self.social_sequence)
    
    def test_edge_cases(self):
        """Test edge cases like empty sequences and invalid inputs."""
        # Test empty sequence
        empty_result = self.optimizer.optimize_sequence("")
        self.assertEqual(empty_result.optimized_sequence, "")
        self.assertEqual(empty_result.optimization_score, 1.0)
        
        # Test single emoji
        single_result = self.optimizer.optimize_sequence("🚀")
        self.assertEqual(single_result.original_sequence, "🚀")
        
        # Test non-emoji content (should still work but not make changes)
        text_result = self.optimizer.optimize_sequence("Hello, world!")
        self.assertEqual(text_result.original_sequence, "Hello, world!")
        self.assertEqual(text_result.optimized_sequence, "Hello, world!")
        
        # Test with custom but empty weights (should use defaults)
        empty_weights = {}
        empty_weights_context = self.optimizer.create_custom_profile(empty_weights)
        self.assertIsNotNone(empty_weights_context.weights)
        self.assertGreater(len(empty_weights_context.weights), 0)
    
    def test_weight_calculations(self):
        """Test calculations of various weighted scores."""
        # Use a specific profile to verify score calculation
        context = OptimizationContext(profile=OptimizationProfile.PRECISE)
        result = self.optimizer.optimize_sequence(self.technical_sequence, context)
        
        # Verify all score types are calculated
        self.assertIn(OptimizationWeight.EXPRESSIVENESS, result.scores)
        self.assertIn(OptimizationWeight.CONCISENESS, result.scores)
        self.assertIn(OptimizationWeight.CLARITY, result.scores)
        self.assertIn(OptimizationWeight.UNIVERSALITY, result.scores)
        self.assertIn(OptimizationWeight.EMOTIONALITY, result.scores)
        self.assertIn(OptimizationWeight.PRECISION, result.scores)
        self.assertIn(OptimizationWeight.CREATIVITY, result.scores)
        self.assertIn(OptimizationWeight.CONSISTENCY, result.scores)
        
        # All scores should be between 0 and 1
        for weight, score in result.scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
