import emoji
#!/usr/bin/env python3
"""
Fix final remaining issues found in the test results.
"""

import os
import re
from pathlib import Path


def fix_universal_agent_connector_indentation():
    """Fix indentation issue in universal_agent_connector.py"""
    filepath = Path(__file__).parent / "universal_agent_connector.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the problematic section around line 735
        lines = content.split('\n')
        fixed_lines = []
        in_try_block = False
        
        for i, line in enumerate(lines):
            if i == 737 and line.strip().startswith('__all__'):
                # This line needs proper indentation inside a try block
                fixed_lines.append('    ' + line.strip())
            else:
                fixed_lines.append(line)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Fixed: Indentation in universal_agent_connector.py")
        return True
    except Exception as e:
        print(f"❌ Error fixing universal_agent_connector.py: {e}")
        return False


def add_request_error_to_llm_adapter():
    """Add RequestError to llm_adapter_interface.py"""
    filepath = Path(__file__).parent / "llm_adapter_interface.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if RequestError is already defined
        if 'class RequestError' not in content:
            # Find where to add it (after AuthenticationError)
            auth_error_pos = content.find('class AuthenticationError')
            if auth_error_pos > 0:
                # Find the end of AuthenticationError class
                next_class_pos = content.find('\n\nclass', auth_error_pos + 1)
                if next_class_pos > 0:
                    insert_pos = next_class_pos
                else:
                    # Find the end of the class
                    pass_pos = content.find('pass', auth_error_pos)
                    if pass_pos > 0:
                        insert_pos = content.find('\n', pass_pos) + 1
                    else:
                        insert_pos = len(content)
                
                request_error_class = '''

class RequestError(Exception):
    """Raised when a request to the LLM provider fails."""
    pass
'''
                content = content[:insert_pos] + request_error_class + content[insert_pos:]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ Fixed: Added RequestError to llm_adapter_interface")
                return True
    except Exception as e:
        print(f"❌ Error fixing llm_adapter_interface: {e}")
    return False


def add_principle_to_principle_engine():
    """Add Principle class to principle_engine.py"""
    filepath = Path(__file__).parent / "principle_engine.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if Principle class exists
        if 'class Principle' not in content:
            # Add after imports
            import_section = content.split('\n\n')[0]
            
            # Make sure dataclass is imported
            if '@dataclass' not in content and 'from dataclasses import' not in content:
                content = content.replace('from typing import', 'from dataclasses import dataclass\nfrom typing import')
            
            # Find where to insert the Principle class
            class_pos = content.find('class ')
            if class_pos > 0:
                # Insert before the first class
                principle_class = '''
@dataclass
class Principle:
    """Represents a principle that guides agent behavior."""
    name: str
    description: str
    weight: float = 1.0
    category: str = "general"
    
    def __post_init__(self):
        """Validate principle data."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Principle weight must be between 0 and 1, got {self.weight}")


'''
                content = content[:class_pos] + principle_class + content[class_pos:]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ Fixed: Added Principle class to principle_engine")
                return True
    except Exception as e:
        print(f"❌ Error adding Principle to principle_engine: {e}")
    return False


def fix_empire_framework_imports():
    """Fix empire framework import issues"""
    filepath = Path(__file__).parent / "empire_framework" / "registry" / "__init__.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the import path
        content = content.replace(
            'from src.empire_framework.registry.component_registry import',
            'from empire_framework.registry.component_registry import'
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed: Empire framework import paths")
        return True
    except Exception as e:
        print(f"❌ Error fixing empire framework imports: {e}")
    return False


def add_missing_emoji_kb_methods():
    """Add missing methods to EmojiKnowledgeBase"""
    filepath = Path(__file__).parent / "emoji_knowledge_base.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add find_concept_for_emoji method
        if 'def find_concept_for_emoji' not in content:
            method_to_add = '''
    def find_concept_for_emoji(self, emoji: str) -> List[str]:
        """Find concepts that use this emoji.
        
        Args:
            emoji: The emoji to find concepts for
            
        Returns:
            List of concept names that include this emoji
        """
        concepts = []
        
        # Search in primary emojis
        for concept, mapping in self.concepts.items():
            for domain, primary_emoji in mapping.primary_emoji.items():
                if primary_emoji == emoji:
                    concepts.append(concept)
                    break
            
            # Search in alternative emojis
            for domain, alt_emojis in mapping.alternative_emojis.items():
                if emoji in alt_emojis:
                    concepts.append(concept)
                    break
            
            # Search in emoji combinations
            for domain, combinations in mapping.emoji_combinations.items():
                for combo in combinations:
                    if emoji in combo:
                        concepts.append(concept)
                        break
        
        return list(set(concepts))  # Remove duplicates
'''
            # Find where to insert (after get_emoji method)
            get_emoji_pos = content.find('def get_emoji(self')
            if get_emoji_pos > 0:
                # Find the end of get_emoji method
                next_def_pos = content.find('\n    def ', get_emoji_pos + 1)
                if next_def_pos > 0:
                    content = content[:next_def_pos] + '\n' + method_to_add + content[next_def_pos:]
                else:
                    # Add at the end of the class
                    content += method_to_add
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed: Added find_concept_for_emoji to EmojiKnowledgeBase")
            return True
    except Exception as e:
        print(f"❌ Error adding methods to EmojiKnowledgeBase: {e}")
    return False


def fix_emoji_sequence_optimizer():
    """Add missing _calculate_ambiguity_score method"""
    filepath = Path(__file__).parent / "emoji_sequence_optimizer.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add _calculate_ambiguity_score method
        if 'def _calculate_ambiguity_score' not in content:
            method_to_add = '''
    def _calculate_ambiguity_score(self, emojis: List[str], context: OptimizationContext) -> float:
        """Calculate ambiguity score for emoji sequence.
        
        Args:
            emojis: List of emoji characters
            context: Optimization context
            
        Returns:
            Ambiguity score (0.0-1.0, higher = more ambiguous)
        """
        if not emojis:
            return 0.0
        
        total_ambiguity = 0.0
        
        for emoji in emojis:
            metadata = self.knowledge_base.get_emoji(emoji)
            if metadata:
                # Use the ambiguity score from metadata
                total_ambiguity += metadata.ambiguity_score
            else:
                # Unknown emoji is considered ambiguous
                total_ambiguity += 0.5
        
        # Average ambiguity
        avg_ambiguity = total_ambiguity / len(emojis)
        
        # Adjust based on context
        if context.domain == EmojiDomain.TECHNICAL:
            # Technical context prefers less ambiguity
            avg_ambiguity *= 1.2
        elif context.domain == EmojiDomain.CREATIVE:
            # Creative context is more tolerant of ambiguity
            avg_ambiguity *= 0.8
        
        return min(1.0, avg_ambiguity)
'''
            # Find where to insert (near other _calculate methods)
            calc_familiarity_pos = content.find('def _calculate_familiarity_score')
            if calc_familiarity_pos > 0:
                # Find the end of this method
                next_def_pos = content.find('\n    def ', calc_familiarity_pos + 1)
                if next_def_pos > 0:
                    content = content[:next_def_pos] + '\n' + method_to_add + content[next_def_pos:]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed: Added _calculate_ambiguity_score to EmojiSequenceOptimizer")
            return True
    except Exception as e:
        print(f"❌ Error fixing EmojiSequenceOptimizer: {e}")
    return False


def fix_unicode_in_test_files():
    """Fix remaining Unicode issues"""
    filepath = Path(__file__).parent / "test_ai_framework_detector.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace warning symbol
        content = content.replace('\u26a0', '[WARNING]')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed: Unicode issue in test_ai_framework_detector.py")
        return True
    except Exception as e:
        print(f"❌ Error fixing Unicode issues: {e}")
    return False


def fix_communication_style_analyzer_regex():
    """Fix regex issue in communication_style_analyzer.py"""
    filepath = Path(__file__).parent / "communication_style_analyzer.py"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the problematic regex call
        # The issue is passing 4 arguments to re.search which only takes 2-3
        old_pattern = r"re\.search\(r'\^\\s\*\(Section\|Part\|Step\|Phase\|Category\)\[\\s:\]', content, re\.MULTILINE, re\.IGNORECASE\)"
        new_pattern = r"re.search(r'^\s*(Section|Part|Step|Phase|Category)[\s:]', content, re.MULTILINE | re.IGNORECASE)"
        
        content = re.sub(old_pattern, new_pattern, content)
        
        # Also look for the specific line and fix it
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "re.search(r'^\\s*(Section|Part|Step|Phase|Category)[\\s:]', content, re.MULTILINE, re.IGNORECASE)" in line:
                lines[i] = line.replace(", re.MULTILINE, re.IGNORECASE)", ", re.MULTILINE | re.IGNORECASE)")
        
        content = '\n'.join(lines)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed: Regex issue in communication_style_analyzer.py")
        return True
    except Exception as e:
        print(f"❌ Error fixing communication_style_analyzer regex: {e}")
    return False


def main():
    """Main function to fix final issues."""
    print("🔧 Fixing final issues found in tests...")
    
    # Fix each issue
    fix_universal_agent_connector_indentation()
    add_request_error_to_llm_adapter()
    add_principle_to_principle_engine()
    fix_empire_framework_imports()
    add_missing_emoji_kb_methods()
    fix_emoji_sequence_optimizer()
    fix_unicode_in_test_files()
    fix_communication_style_analyzer_regex()
    
    print("\n✨ Finished fixing final issues")


if __name__ == "__main__":
    main()