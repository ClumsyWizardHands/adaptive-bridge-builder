import emoji
"""
Script to fix critical issues in the codebase.
"""

import os
import re
import json
from typing import List, Tuple

def fix_empire_framework_imports():
    """Fix import paths in empire framework modules (use relative imports)."""
    empire_dirs = [
        'empire_framework/validation',
        'empire_framework/registry', 
        'empire_framework/storage',
        'empire_framework/a2a'
    ]
    
    fixes_made = []
    
    for dir_path in empire_dirs:
        if not os.path.exists(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                filepath = os.path.join(dir_path, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix absolute imports to relative imports
                # from empire_framework.validation import -> from . import
                content = re.sub(
                    r'from empire_framework\.(\w+) import',
                    r'from . import',
                    content
                )
                
                # from empire_framework.validation.module import -> from .module import
                content = re.sub(
                    r'from empire_framework\.\w+\.(\w+) import',
                    r'from .\1 import',
                    content
                )
                
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes_made.append(filepath)
    
    return fixes_made

def add_rate_limit_error():
    """Add RateLimitError class to llm_adapter_interface.py."""
    filepath = 'llm_adapter_interface.py'
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if RateLimitError already exists
    if 'class RateLimitError' in content:
        print("RateLimitError already exists")
        return False
    
    # Find where to insert the error class (after imports, before other classes)
    import_end = content.rfind('\n\n', 0, content.find('class'))
    if import_end == -1:
        import_end = content.rfind('\n', 0, content.find('class'))
    
    # Insert the RateLimitError class
    rate_limit_error = '''

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message="Rate limit exceeded", retry_after=None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)
'''
    
    content = content[:import_end] + rate_limit_error + content[import_end:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def create_agent_card_json():
    """Create the missing agent_card.json file."""
    filepath = 'agent_card.json'
    
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return False
    
    agent_card = {
        "agent_id": "alex-familiar-001",
        "name": "Alex Familiar",
        "description": "A sophisticated orchestration agent implementing the EMPIRE framework",
        "version": "1.0.0",
        "capabilities": [
            "project_orchestration",
            "task_decomposition",
            "resource_management",
            "multi_agent_coordination",
            "principle_based_reasoning",
            "emotional_intelligence",
            "multilingual_communication",
            "conflict_resolution"
        ],
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ja", "zh"],
        "communication_protocols": ["a2a", "adk", "http", "websocket"],
        "principle_framework": "EMPIRE",
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "author": "Alex Familiar Development Team",
            "license": "MIT"
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(agent_card, f, indent=4)
    
    return True

def add_missing_methods():
    """Add missing methods to various classes."""
    
    # Fix 1: Add analyze_message to CommunicationStyleAnalyzer
    filepath1 = 'communication_style_analyzer.py'
    if os.path.exists(filepath1):
        with open(filepath1, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def analyze_message' not in content:
            # Find the class definition
            class_match = re.search(r'class CommunicationStyleAnalyzer.*?:\n', content)
            if class_match:
                # Find the end of __init__ method
                init_end = content.find('\n\n', content.find('def __init__', class_match.end()))
                if init_end == -1:
                    init_end = content.find('\n    def ', content.find('def __init__', class_match.end()))
                
                method_code = '''
    
    def analyze_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a message to determine communication style characteristics.
        
        Args:
            message: The message to analyze
            context: Optional context for the analysis
            
        Returns:
            Dictionary containing style analysis results
        """
        analysis = {
            "formality_level": self._analyze_formality(message),
            "tone": self._analyze_tone(message),
            "clarity": self._analyze_clarity(message),
            "assertiveness": self._analyze_assertiveness(message),
            "empathy_level": self._analyze_empathy(message),
            "technical_level": self._analyze_technical_level(message),
            "context_awareness": self._analyze_context_awareness(message, context)
        }
        
        return analysis
    
    def _analyze_formality(self, message: str) -> float:
        """Analyze formality level (0.0 to 1.0)."""
        formal_indicators = ['please', 'kindly', 'would you', 'could you', 'regards']
        informal_indicators = ['hey', 'yeah', 'gonna', 'wanna', 'btw']
        
        formal_count = sum(1 for word in formal_indicators if word in message.lower())
        informal_count = sum(1 for word in informal_indicators if word in message.lower())
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _analyze_tone(self, message: str) -> str:
        """Analyze the tone of the message."""
        # Simple tone analysis
        if any(word in message.lower() for word in ['urgent', 'asap', 'immediately']):
            return 'urgent'
        elif any(word in message.lower() for word in ['please', 'thank', 'appreciate']):
            return 'polite'
        elif any(word in message.lower() for word in ['concern', 'worry', 'problem']):
            return 'concerned'
        else:
            return 'neutral'
    
    def _analyze_clarity(self, message: str) -> float:
        """Analyze message clarity (0.0 to 1.0)."""
        # Simple clarity metric based on sentence structure
        sentences = message.split('.')
        avg_words = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_words <= 20:
            return 1.0
        elif avg_words < 10 or avg_words > 30:
            return 0.5
        else:
            return 0.75
    
    def _analyze_assertiveness(self, message: str) -> float:
        """Analyze assertiveness level (0.0 to 1.0)."""
        assertive_words = ['must', 'will', 'need', 'require', 'expect']
        passive_words = ['might', 'could', 'perhaps', 'maybe', 'possibly']
        
        assertive_count = sum(1 for word in assertive_words if word in message.lower())
        passive_count = sum(1 for word in passive_words if word in message.lower())
        
        if assertive_count + passive_count == 0:
            return 0.5
        
        return assertive_count / (assertive_count + passive_count)
    
    def _analyze_empathy(self, message: str) -> float:
        """Analyze empathy level (0.0 to 1.0)."""
        empathy_words = ['understand', 'feel', 'appreciate', 'concern', 'support']
        return min(sum(1 for word in empathy_words if word in message.lower()) / 3.0, 1.0)
    
    def _analyze_technical_level(self, message: str) -> float:
        """Analyze technical complexity (0.0 to 1.0)."""
        technical_terms = ['api', 'framework', 'algorithm', 'protocol', 'implementation']
        return min(sum(1 for term in technical_terms if term in message.lower()) / 3.0, 1.0)
    
    def _analyze_context_awareness(self, message: str, context: Optional[Dict[str, Any]]) -> float:
        """Analyze context awareness (0.0 to 1.0)."""
        if not context:
            return 0.5
        
        # Check if message references context elements
        context_refs = 0
        for key, value in context.items():
            if isinstance(value, str) and value.lower() in message.lower():
                context_refs += 1
        
        return min(context_refs / max(len(context), 1), 1.0)
'''
                
                content = content[:init_end] + method_code + content[init_end:]
                
                with open(filepath1, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Added analyze_message method to {filepath1}")
    
    # Fix 2: Add _analyze_communication_bottlenecks to OrchestrationAnalytics
    filepath2 = 'orchestration_analytics.py'
    if os.path.exists(filepath2):
        with open(filepath2, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def _analyze_communication_bottlenecks' not in content:
            # Find a good place to insert (before the last method or at the end of class)
            last_def = content.rfind('\n    def ')
            if last_def != -1:
                insert_point = content.rfind('\n\n', 0, last_def) + 1
            else:
                insert_point = len(content) - 1
            
            method_code = '''
    def _analyze_communication_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze communication bottlenecks in the orchestration system.
        
        Returns:
            Dictionary containing bottleneck analysis
        """
        bottlenecks = {
            "identified_bottlenecks": [],
            "communication_delays": {},
            "agent_response_times": {},
            "message_queue_sizes": {},
            "recommendations": []
        }
        
        # Analyze response times
        for agent_id, comms in self.communication_metrics.items():
            if comms:
                response_times = []
                for comm in comms:
                    if 'response_time' in comm:
                        response_times.append(comm['response_time'])
                
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    bottlenecks["agent_response_times"][agent_id] = avg_response_time
                    
                    # Flag as bottleneck if avg response time > 5 seconds
                    if avg_response_time > 5.0:
                        bottlenecks["identified_bottlenecks"].append({
                            "type": "slow_response",
                            "agent_id": agent_id,
                            "avg_response_time": avg_response_time
                        })
        
        # Analyze message patterns
        total_messages = sum(len(comms) for comms in self.communication_metrics.values())
        if total_messages > 100:
            # Check for communication imbalances
            for agent_id, comms in self.communication_metrics.items():
                message_ratio = len(comms) / total_messages
                if message_ratio > 0.5:  # One agent handling > 50% of messages
                    bottlenecks["identified_bottlenecks"].append({
                        "type": "communication_imbalance",
                        "agent_id": agent_id,
                        "message_ratio": message_ratio
                    })
        
        # Generate recommendations
        if bottlenecks["identified_bottlenecks"]:
            for bottleneck in bottlenecks["identified_bottlenecks"]:
                if bottleneck["type"] == "slow_response":
                    bottlenecks["recommendations"].append(
                        f"Consider optimizing agent {bottleneck['agent_id']} or distributing its workload"
                    )
                elif bottleneck["type"] == "communication_imbalance":
                    bottlenecks["recommendations"].append(
                        f"Agent {bottleneck['agent_id']} is handling {bottleneck['message_ratio']:.1%} of messages. Consider load balancing."
                    )
        
        return bottlenecks

'''
            
            content = content[:insert_point] + method_code + content[insert_point:]
            
            with open(filepath2, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Added _analyze_communication_bottlenecks method to {filepath2}")
    
    # Fix 3: Add register_resource to ProjectOrchestrator
    filepath3 = 'project_orchestrator.py'
    if os.path.exists(filepath3):
        with open(filepath3, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def register_resource' not in content:
            # Find the end of the class (look for the last method)
            last_method = content.rfind('\n    def ')
            if last_method != -1:
                # Find the end of that method
                next_class_or_end = content.find('\nclass ', last_method)
                if next_class_or_end == -1:
                    next_class_or_end = len(content)
                
                # Find a good insertion point (before the next class or at the end)
                insert_point = content.rfind('\n\n', last_method, next_class_or_end)
                if insert_point == -1:
                    insert_point = next_class_or_end
                
                method_code = '''
    
    def register_resource(
        self,
        resource_type: ResourceType,
        name: str,
        capacity: float = 1.0,
        cost_per_unit: float = 0.0,
        tags: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Resource:
        """
        Register a new resource in the global resource pool.
        
        Args:
            resource_type: Type of the resource
            name: Name of the resource
            capacity: Total capacity of the resource
            cost_per_unit: Cost per unit of resource usage
            tags: Optional tags for categorizing the resource
            constraints: Optional constraints on resource usage
            metadata: Optional additional metadata
            
        Returns:
            The registered Resource object
        """
        resource_id = f"resource-{str(uuid.uuid4())}"
        
        with self.resource_lock:
            resource = Resource(
                resource_id=resource_id,
                resource_type=resource_type,
                name=name,
                capacity=capacity,
                allocated=0.0,
                cost_per_unit=cost_per_unit,
                tags=tags or [],
                constraints=constraints or {},
                metadata=metadata or {}
            )
            
            self.global_resources[resource_id] = resource
            
            # Update conscience metrics
            self._update_conscience_metrics()
            
            logger.info(f"Registered resource '{name}' of type {resource_type.value} with ID {resource_id}")
            
            return resource

'''
                
                content = content[:insert_point] + method_code + content[insert_point:]
                
                with open(filepath3, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Added register_resource method to {filepath3}")
    
    # Fix 4: Add _calculate_ambiguity_score to EmojiSequenceOptimizer
    filepath4 = 'emoji_sequence_optimizer.py'
    if os.path.exists(filepath4):
        with open(filepath4, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def _calculate_ambiguity_score' not in content:
            # Find a good place to insert (look for other private methods)
            private_method = content.rfind('\n    def _')
            if private_method != -1:
                # Find the end of that method
                next_method = content.find('\n    def ', private_method + 1)
                if next_method == -1:
                    next_method = len(content)
                
                insert_point = content.rfind('\n', private_method, next_method)
            else:
                # Insert before the last public method
                insert_point = content.rfind('\n    def ')
            
            method_code = '''
    
    def _calculate_ambiguity_score(self, emoji: str, context: Optional[str] = None) -> float:
        """
        Calculate ambiguity score for an emoji in context.
        
        Args:
            emoji: The emoji to analyze
            context: Optional context string
            
        Returns:
            Ambiguity score from 0.0 (unambiguous) to 1.0 (highly ambiguous)
        """
        # Base ambiguity scores for common emojis
        ambiguity_map = {
            "😊": 0.2,  # Generally positive, low ambiguity
            "😐": 0.8,  # Neutral face, high ambiguity
            "🤔": 0.7,  # Thinking face, moderately ambiguous
            "😏": 0.9,  # Smirking face, very ambiguous
            "👍": 0.1,  # Thumbs up, low ambiguity
            "🙃": 0.8,  # Upside down face, high ambiguity
            "😶": 0.9,  # No mouth, very ambiguous
            "🤷": 0.6,  # Shrug, moderate ambiguity
        }
        
        # Get base score or default
        base_score = ambiguity_map.get(emoji, 0.5)
        
        # Adjust based on context if provided
        if context:
            context_lower = context.lower()
            
            # Clear positive context reduces ambiguity
            if any(word in context_lower for word in ['happy', 'great', 'excellent', 'love']):
                base_score *= 0.7
            
            # Clear negative context reduces ambiguity
            elif any(word in context_lower for word in ['sad', 'angry', 'hate', 'terrible']):
                base_score *= 0.7
            
            # Question context increases ambiguity
            elif '?' in context:
                base_score *= 1.2
            
            # Sarcasm indicators increase ambiguity
            elif any(word in context_lower for word in ['sure', 'right', 'totally', 'obviously']):
                base_score *= 1.3
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, base_score))

'''
            
            content = content[:insert_point] + method_code + content[insert_point:]
            
            # Also need to import Optional if not already imported
            if 'from typing import' in content and 'Optional' not in content:
                typing_import = content.find('from typing import')
                import_end = content.find('\n', typing_import)
                imports = content[typing_import:import_end]
                if not imports.endswith('Optional'):
                    content = content[:import_end] + ', Optional' + content[import_end:]
            
            with open(filepath4, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Added _calculate_ambiguity_score method to {filepath4}")

def main():
    """Run all fixes."""
    print("Starting critical fixes...\n")
    
    # Fix 1: Empire framework imports
    print("1. Fixing Empire framework imports...")
    fixed_files = fix_empire_framework_imports()
    if fixed_files:
        print(f"   Fixed imports in: {', '.join(fixed_files)}")
    else:
        print("   No import fixes needed")
    
    # Fix 2: Add RateLimitError
    print("\n2. Adding RateLimitError class...")
    if add_rate_limit_error():
        print("   Added RateLimitError to llm_adapter_interface.py")
    else:
        print("   RateLimitError already exists or file not found")
    
    # Fix 3: Create agent_card.json
    print("\n3. Creating agent_card.json...")
    if create_agent_card_json():
        print("   Created agent_card.json")
    else:
        print("   agent_card.json already exists")
    
    # Fix 4: Add missing methods
    print("\n4. Adding missing methods...")
    add_missing_methods()
    
    print("\nAll critical fixes completed!")

if __name__ == "__main__":
    main()