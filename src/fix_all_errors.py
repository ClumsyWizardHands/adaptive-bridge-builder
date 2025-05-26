"""
Comprehensive Error Fixer
Fixes all types of errors found in the codebase
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set

class ComprehensiveErrorFixer:
    def __init__(self):
        self.fixed_files = []
        self.failed_fixes = []
        
    def fix_syntax_errors(self, file_path: str, line_number: int, error_type: str) -> bool:
        """Fix syntax errors based on error type"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Adjust for 0-based indexing
            idx = line_number - 1
            if idx < 0 or idx >= len(lines):
                return False
                
            original_line = lines[idx]
            fixed = False
            
            # Fix missing commas
            if "Perhaps you forgot a comma?" in error_type:
                # Common patterns where commas are missing
                # In dictionaries/lists after strings or numbers
                if re.search(r'["\']}\s*["\'{]', lines[idx]):
                    lines[idx] = re.sub(r'(["\'])\s*(["\'{])', r'\1, \2', lines[idx])
                    fixed = True
                elif re.search(r'[}\]]\s*["\'{]', lines[idx]):
                    lines[idx] = re.sub(r'([}\]])\s*(["\'{])', r'\1, \2', lines[idx])
                    fixed = True
                elif re.search(r'\d\s+["\']', lines[idx]):
                    lines[idx] = re.sub(r'(\d)\s+(["\'])', r'\1, \2', lines[idx])
                    fixed = True
                # Look for missing comma in multi-line structures
                elif idx > 0:
                    prev_line = lines[idx-1].rstrip()
                    if (prev_line.endswith('"') or prev_line.endswith("'") or 
                        prev_line.endswith('}') or prev_line.endswith(']')) and \
                       not prev_line.endswith(','):
                        lines[idx-1] = prev_line + ',\n'
                        fixed = True
                        
            # Fix mismatched parentheses
            elif "does not match opening parenthesis" in error_type:
                # Extract the mismatched characters
                match = re.search(r"closing parenthesis '(.)'.*opening parenthesis '(.)'", error_type)
                if match:
                    closing, opening = match.groups()
                    # Define matching pairs
                    pairs = {'(': ')', '[': ']', '{': '}'}
                    if opening in pairs:
                        correct_closing = pairs[opening]
                        lines[idx] = lines[idx].replace(closing, correct_closing, 1)
                        fixed = True
                        
            # Fix invalid assignment in conditionals
            elif "Maybe you meant '==' or ':=' instead of '='?" in error_type:
                # Look for assignment in if/while/elif conditions
                if re.search(r'(if|while|elif)\s+.*=(?!=)', lines[idx]):
                    lines[idx] = re.sub(r'(if|while|elif)(\s+.*)=(?!=)', r'\1\2==', lines[idx])
                    fixed = True
                    
            # Fix general syntax errors by checking context
            elif "invalid syntax" in error_type:
                # Check for unclosed strings
                if lines[idx].count('"') % 2 != 0 or lines[idx].count("'") % 2 != 0:
                    # Try to close unclosed strings
                    if lines[idx].rstrip().endswith('"'):
                        pass  # Already closed
                    elif '"' in lines[idx] and lines[idx].count('"') % 2 != 0:
                        lines[idx] = lines[idx].rstrip() + '"\n'
                        fixed = True
                    elif "'" in lines[idx] and lines[idx].count("'") % 2 != 0:
                        lines[idx] = lines[idx].rstrip() + "'\n"
                        fixed = True
                        
            if fixed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True
                
        except Exception as e:
            print(f"Error fixing {file_path}:{line_number} - {e}")
            
        return False
        
    def fix_import_errors(self, file_path: str, module_name: str) -> bool:
        """Fix import errors, especially relative imports in subpackages"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            fixed = False
            lines = content.split('\n')
            
            # Determine the package structure
            file_parts = Path(file_path).parts
            
            # Fix relative imports in empire_framework subpackages
            if 'empire_framework' in file_parts:
                # Find the index of empire_framework
                empire_idx = file_parts.index('empire_framework')
                current_package = '.'.join(file_parts[empire_idx:-1])
                
                for i, line in enumerate(lines):
                    # Fix various import patterns
                    if f'from {module_name} import' in line or f'import {module_name}' in line:
                        if module_name in ['component_registry', 'schema_validator', 'versioned_component_store',
                                         'message_structures', 'a2a_adapter', 'empire_adk_adapter']:
                            # These are sibling modules
                            if 'from component_registry import' in line:
                                lines[i] = line.replace('from component_registry import', 
                                                      'from empire_framework.registry.component_registry import')
                                fixed = True
                            elif 'from schema_validator import' in line:
                                lines[i] = line.replace('from schema_validator import', 
                                                      'from empire_framework.validation.schema_validator import')
                                fixed = True
                            elif 'from versioned_component_store import' in line:
                                lines[i] = line.replace('from versioned_component_store import', 
                                                      'from empire_framework.storage.versioned_component_store import')
                                fixed = True
                            elif 'from message_structures import' in line:
                                lines[i] = line.replace('from message_structures import', 
                                                      'from empire_framework.a2a.message_structures import')
                                fixed = True
                            elif 'from a2a_adapter import' in line:
                                lines[i] = line.replace('from a2a_adapter import', 
                                                      'from empire_framework.a2a.a2a_adapter import')
                                fixed = True
                            elif 'from empire_adk_adapter import' in line:
                                lines[i] = line.replace('from empire_adk_adapter import', 
                                                      'from empire_framework.adk.empire_adk_adapter import')
                                fixed = True
                                
                    # Fix imports with dots (relative imports)
                    elif 'from validation.schema_validator import' in line:
                        lines[i] = line.replace('from validation.schema_validator import', 
                                              'from empire_framework.validation.schema_validator import')
                        fixed = True
                    elif 'from validation.validator_example import' in line:
                        lines[i] = line.replace('from validation.validator_example import', 
                                              'from empire_framework.validation.validator_example import')
                        fixed = True
                    elif 'from registry.component_registry import' in line:
                        lines[i] = line.replace('from registry.component_registry import', 
                                              'from empire_framework.registry.component_registry import')
                        fixed = True
                        
            # Fix imports in api/integration_assistant
            elif 'integration_assistant' in file_parts:
                for i, line in enumerate(lines):
                    if 'from models import' in line:
                        lines[i] = line.replace('from models import', 
                                              'from .models import')
                        fixed = True
                    elif 'from websocket_manager import' in line:
                        lines[i] = line.replace('from websocket_manager import', 
                                              'from .websocket_manager import')
                        fixed = True
                    elif 'from code_generator import' in line:
                        lines[i] = line.replace('from code_generator import', 
                                              'from .code_generator import')
                        fixed = True
                    elif 'from app import' in line:
                        lines[i] = line.replace('from app import', 
                                              'from .app import')
                        fixed = True
                        
            if fixed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                return True
                    
        except Exception as e:
            print(f"Error fixing imports in {file_path} - {e}")
            
        return False
        
    def add_missing_dependencies(self):
        """Add missing external dependencies to requirements.txt"""
        missing_deps = {
            'astor': 'astor>=0.8.1',
            'llama_cpp': 'llama-cpp-python>=0.2.0',
        }
        
        try:
            with open('requirements.txt', 'r') as f:
                existing_deps = f.read()
                
            deps_to_add = []
            for module, package in missing_deps.items():
                if module not in existing_deps and package not in existing_deps:
                    deps_to_add.append(package)
                    
            if deps_to_add:
                with open('requirements.txt', 'a') as f:
                    f.write('\n')
                    for dep in deps_to_add:
                        f.write(f'{dep}\n')
                print(f"Added {len(deps_to_add)} missing dependencies to requirements.txt")
                
        except Exception as e:
            print(f"Error updating requirements.txt - {e}")
            
    def fix_specific_syntax_patterns(self):
        """Fix specific known syntax patterns"""
        specific_fixes = [
            # Fix async_best_practices_example.py line 161
            ('src/async_best_practices_example.py', 161, 
             lambda line: line.replace('task = asyncio.create_task(', 'untracked_task = asyncio.create_task(')),
             
            # Fix run_comprehensive_tests.py line 116  
            ('src/run_comprehensive_tests.py', 116,
             lambda line: line if line.strip().endswith(',') else line.rstrip() + ',\n' if line.strip() else line),
             
            # Fix principle_engine_llm.py line 350
            ('src/principle_engine_llm.py', 350,
             lambda line: line.replace(']', ')') if "closing parenthesis ']' does not match" in line else line),
        ]
        
        for file_path, line_num, fix_func in specific_fixes:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                if 0 <= line_num - 1 < len(lines):
                    lines[line_num - 1] = fix_func(lines[line_num - 1])
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                        
                    print(f"Applied specific fix to {file_path}:{line_num}")
                    
            except Exception as e:
                print(f"Error applying specific fix to {file_path}:{line_num} - {e}")
                
    def run_fixes(self, error_report_path: str = 'error_scan_report.json'):
        """Run all fixes based on the error report"""
        try:
            with open(error_report_path, 'r') as f:
                report = json.load(f)
                
            print("🔧 Starting comprehensive error fixes...")
            
            # First fix specific known patterns
            self.fix_specific_syntax_patterns()
            
            # Fix syntax errors (highest priority)
            print("\n📝 Fixing syntax errors...")
            syntax_fixed = 0
            for error in report['syntax_errors']:
                if self.fix_syntax_errors(error['file'], error['line'], error['error']):
                    syntax_fixed += 1
                    self.fixed_files.append(error['file'])
                else:
                    self.failed_fixes.append(f"Syntax: {error['file']}:{error['line']}")
                    
            print(f"  Fixed {syntax_fixed}/{len(report['syntax_errors'])} syntax errors")
            
            # Fix import errors
            print("\n📦 Fixing import errors...")
            import_fixed = 0
            for error in report['import_errors']:
                if self.fix_import_errors(error['file'], error['module']):
                    import_fixed += 1
                    if error['file'] not in self.fixed_files:
                        self.fixed_files.append(error['file'])
                else:
                    self.failed_fixes.append(f"Import: {error['file']} - {error['module']}")
                    
            print(f"  Fixed {import_fixed}/{len(report['import_errors'])} import errors")
            
            # Add missing dependencies
            print("\n📚 Adding missing dependencies...")
            self.add_missing_dependencies()
            
            # Summary
            print("\n✅ Fix Summary:")
            print(f"  Total files modified: {len(set(self.fixed_files))}")
            print(f"  Syntax errors fixed: {syntax_fixed}")
            print(f"  Import errors fixed: {import_fixed}")
            
            if self.failed_fixes:
                print(f"\n⚠️ Failed to fix {len(self.failed_fixes)} errors:")
                for failed in self.failed_fixes[:10]:
                    print(f"  - {failed}")
                    
        except Exception as e:
            print(f"Error running fixes: {e}")
            
def main():
    fixer = ComprehensiveErrorFixer()
    fixer.run_fixes()
    
    print("\n🔍 Re-scanning to verify fixes...")
    # Run the error scanner again to verify
    os.system('python src/comprehensive_error_scanner.py')

if __name__ == "__main__":
    main()
