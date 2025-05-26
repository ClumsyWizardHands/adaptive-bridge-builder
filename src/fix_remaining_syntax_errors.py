"""
Fix remaining syntax errors with more targeted approaches
"""

import os
import re
import json
from pathlib import Path

class TargetedSyntaxFixer:
    def __init__(self):
        self.fixes_applied = []
        
    def fix_file(self, file_path: str, line_number: int, error_type: str) -> bool:
        """Apply targeted fixes based on file and error"""
        
        # Map of specific files to their fix functions
        specific_fixes = {
            'src/agent_card.py': self.fix_agent_card,
            'src/run_comprehensive_tests.py': self.fix_run_comprehensive_tests,
            'src/principle_engine_llm.py': self.fix_principle_engine_llm,
            'src/empire_framework/registry/component_registry.py': self.fix_component_registry,
        }
        
        # Use specific fix if available
        if file_path in specific_fixes:
            return specific_fixes[file_path](line_number, error_type)
        
        # Otherwise try generic fix
        return self.generic_fix(file_path, line_number, error_type)
        
    def fix_agent_card(self, line_number: int, error_type: str) -> bool:
        """Fix agent_card.py specific issues"""
        file_path = 'src/agent_card.py'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Line 389: closing parenthesis ']' does not match opening parenthesis '{'
            if line_number == 389:
                # Find the opening bracket and fix the closing
                for i in range(line_number - 1, max(0, line_number - 20), -1):
                    if '{' in lines[i] and '}' not in lines[i]:
                        # Found unclosed dict, fix line 389
                        if lines[388].strip().endswith(']'):
                            lines[388] = lines[388].rstrip()[:-1] + '}\n'
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            return True
                            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
        return False
        
    def fix_run_comprehensive_tests(self, line_number: int, error_type: str) -> bool:
        """Fix run_comprehensive_tests.py"""
        file_path = 'src/run_comprehensive_tests.py'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Line 116: missing comma
            if line_number == 116 and "forgot a comma" in error_type:
                # Check if line ends without comma but should have one
                if lines[115].strip() and not lines[115].rstrip().endswith(','):
                    lines[115] = lines[115].rstrip() + ',\n'
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    return True
                    
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
        return False
        
    def fix_principle_engine_llm(self, line_number: int, error_type: str) -> bool:
        """Fix principle_engine_llm.py"""
        file_path = 'src/principle_engine_llm.py'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Line 350: closing parenthesis ']' does not match opening parenthesis '('
            if line_number == 350 and "does not match" in error_type:
                # Replace ] with )
                lines[349] = lines[349].replace(']', ')')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True
                    
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
        return False
        
    def fix_component_registry(self, line_number: int, error_type: str) -> bool:
        """Fix component_registry.py"""
        file_path = 'src/empire_framework/registry/component_registry.py'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Line 315: closing parenthesis '}' does not match opening parenthesis '('
            if line_number == 315 and "does not match" in error_type:
                # Replace } with )
                lines[314] = lines[314].replace('}', ')')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True
                    
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
        return False
        
    def generic_fix(self, file_path: str, line_number: int, error_type: str) -> bool:
        """Generic fixes for common patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            idx = line_number - 1
            if idx < 0 or idx >= len(lines):
                return False
                
            fixed = False
            
            # Fix missing commas
            if "forgot a comma" in error_type:
                # Check previous line
                if idx > 0:
                    prev = lines[idx-1].rstrip()
                    # Add comma if line ends with quote or bracket
                    if (prev.endswith('"') or prev.endswith("'") or 
                        prev.endswith('}') or prev.endswith(']')) and not prev.endswith(','):
                        lines[idx-1] = prev + ',\n'
                        fixed = True
                        
            # Fix mismatched brackets
            elif "does not match" in error_type:
                match = re.search(r"closing parenthesis '(.)'.*opening parenthesis '(.)'", error_type)
                if match:
                    closing, opening = match.groups()
                    pairs = {'(': ')', '[': ']', '{': '}'}
                    if opening in pairs:
                        correct = pairs[opening]
                        lines[idx] = lines[idx].replace(closing, correct, 1)
                        fixed = True
                        
            if fixed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True
                    
        except Exception as e:
            print(f"Error in generic fix for {file_path}: {e}")
            
        return False
        
    def fix_import_errors(self):
        """Fix remaining import errors"""
        import_fixes = [
            # Fix empire_framework imports
            ('src/empire_framework/a2a/component_task_handler.py', 
             [('from message_structures import', 'from .message_structures import'),
              ('from registry.component_registry import', 'from ..registry.component_registry import'),
              ('from validation.schema_validator import', 'from ..validation.schema_validator import')]),
            
            ('src/empire_framework/a2a/streaming_adapter.py',
             [('from a2a_adapter import', 'from .a2a_adapter import')]),
             
            ('src/empire_framework/adk/example_usage.py',
             [('from empire_adk_adapter import', 'from .empire_adk_adapter import')]),
             
            ('src/empire_framework/registry/test_component_registry.py',
             [('from component_registry import', 'from .component_registry import'),
              ('from validation.', 'from ..validation.')]),
              
            ('src/empire_framework/registry/__init__.py',
             [('from component_registry import', 'from .component_registry import')]),
             
            ('src/empire_framework/storage/test_versioned_component_store.py',
             [('from versioned_component_store import', 'from .versioned_component_store import')]),
             
            ('src/empire_framework/storage/__init__.py',
             [('from versioned_component_store import', 'from .versioned_component_store import')]),
             
            ('src/empire_framework/validation/test_schema_validator.py',
             [('from schema_validator import', 'from .schema_validator import'),
              ('from validator_example import', 'from .validator_example import')]),
              
            ('src/empire_framework/validation/validator_example.py',
             [('from schema_validator import', 'from .schema_validator import')]),
             
            ('src/empire_framework/validation/__init__.py',
             [('from schema_validator import', 'from .schema_validator import')]),
             
            # Fix api/integration_assistant imports
            ('src/api/integration_assistant/app.py',
             [('from models import', 'from .models import'),
              ('from websocket_manager import', 'from .websocket_manager import'),
              ('from code_generator import', 'from .code_generator import')]),
              
            ('src/api/integration_assistant/code_generator.py',
             [('from models import', 'from .models import')]),
             
            ('src/api/integration_assistant/websocket_manager.py',
             [('from models import', 'from .models import')]),
             
            ('src/api/integration_assistant/__init__.py',
             [('from app import', 'from .app import'),
              ('from models import', 'from .models import'),
              ('from websocket_manager import', 'from .websocket_manager import')]),
        ]
        
        fixed_count = 0
        for file_path, replacements in import_fixes:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original = content
                for old, new in replacements:
                    content = content.replace(old, new)
                    
                if content != original:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_count += 1
                    print(f"Fixed imports in {file_path}")
                    
            except Exception as e:
                print(f"Error fixing imports in {file_path}: {e}")
                
        return fixed_count
        
def main():
    # Load error report
    try:
        with open('error_scan_report.json', 'r') as f:
            report = json.load(f)
    except:
        print("Error: Could not load error_scan_report.json")
        return
        
    fixer = TargetedSyntaxFixer()
    
    print("🔧 Applying targeted syntax fixes...")
    
    # Fix syntax errors
    syntax_fixed = 0
    for error in report['syntax_errors']:
        if fixer.fix_file(error['file'], error['line'], error['error']):
            syntax_fixed += 1
            print(f"✓ Fixed {error['file']}:{error['line']}")
        else:
            print(f"✗ Failed to fix {error['file']}:{error['line']}")
            
    print(f"\nFixed {syntax_fixed}/{len(report['syntax_errors'])} syntax errors")
    
    # Fix import errors
    print("\n🔧 Fixing import errors...")
    import_fixed = fixer.fix_import_errors()
    print(f"Fixed {import_fixed} files with import errors")
    
    print("\n✅ Targeted fixes complete!")

if __name__ == "__main__":
    main()
