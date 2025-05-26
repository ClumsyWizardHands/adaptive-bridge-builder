#!/usr/bin/env python3
"""
Script to find and fix async functions without proper error handling.
Ensures all async functions have try-catch blocks.
"""

import os
import re
import ast
from typing import List, Tuple, Dict, Any

class AsyncErrorHandlingChecker(ast.NodeVisitor):
    """AST visitor to check async functions for error handling."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.current_function = None
        self.in_try_block = False
        self.has_try_catch = {}
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        self.current_function = node.name
        self.in_try_block = False
        self.has_try_catch[node.name] = False
        
        # Check if function body has try-except
        for stmt in node.body:
            if isinstance(stmt, ast.Try):
                self.has_try_catch[node.name] = True
                break
                
        # Visit children
        old_func = self.current_function
        old_try = self.in_try_block
        self.generic_visit(node)
        self.current_function = old_func
        self.in_try_block = old_try
        
        # Report if no try-catch found
        if not self.has_try_catch[node.name]:
            self.issues.append({
                'function': node.name,
                'line': node.lineno,
                'has_error_handling': False
            })
    
    def visit_Try(self, node: ast.Try):
        """Visit try blocks."""
        old_try = self.in_try_block
        self.in_try_block = True
        self.generic_visit(node)
        self.in_try_block = old_try


def find_async_functions_without_error_handling(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all async functions without proper error handling.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary mapping file paths to list of issues
    """
    issues_by_file = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip test directories and __pycache__
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse AST
                    tree = ast.parse(content, filename=filepath)
                    
                    # Check for async functions
                    checker = AsyncErrorHandlingChecker(filepath)
                    checker.visit(tree)
                    
                    if checker.issues:
                        issues_by_file[filepath] = checker.issues
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return issues_by_file


def add_error_handling_to_async_function(content: str, function_name: str, line_number: int) -> str:
    """
    Add try-catch block to an async function.
    
    Args:
        content: File content
        function_name: Name of the async function
        line_number: Line number where function starts
        
    Returns:
        Modified content
    """
    lines = content.split('\n')
    
    # Find the function definition
    func_start = -1
    for i in range(line_number - 1, len(lines)):
        if f"async def {function_name}" in lines[i]:
            func_start = i
            break
    
    if func_start == -1:
        return content
    
    # Find the indentation level
    indent_match = re.match(r'^(\s*)', lines[func_start])
    base_indent = indent_match.group(1) if indent_match else ''
    func_indent = base_indent + '    '
    
    # Find the function body start
    body_start = func_start + 1
    while body_start < len(lines) and (not lines[body_start].strip() or 
                                       lines[body_start].strip().startswith('"""') or
                                       lines[body_start].strip().startswith("'''")):
        body_start += 1
    
    # Check if already has try-catch at the top level
    if body_start < len(lines) and lines[body_start].strip().startswith('try:'):
        return content
    
    # Find the end of the function
    func_end = body_start
    for i in range(body_start, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(func_indent) and not line.startswith(base_indent + ' '):
            func_end = i
            break
    else:
        func_end = len(lines)
    
    # Extract function body
    body_lines = lines[body_start:func_end]
    
    # Add try-catch wrapper
    new_lines = lines[:body_start]
    
    # Add try block
    new_lines.append(func_indent + 'try:')
    
    # Indent existing body
    for line in body_lines:
        if line.strip():
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    
    # Add except blocks
    new_lines.extend([
        func_indent + 'except asyncio.CancelledError:',
        func_indent + '    # Allow cancellation to propagate',
        func_indent + '    raise',
        func_indent + 'except Exception as e:',
        func_indent + '    # Log the error',
        func_indent + '    if hasattr(self, "logger"):',
        func_indent + '        self.logger.error(f"Error in {}: {e}", exc_info=True)'.format(function_name),
        func_indent + '    else:',
        func_indent + '        import logging',
        func_indent + '        logging.error(f"Error in {}: {e}", exc_info=True)'.format(function_name),
        func_indent + '    # Re-raise or handle based on function context',
        func_indent + '    raise'
    ])
    
    # Add remaining lines
    new_lines.extend(lines[func_end:])
    
    return '\n'.join(new_lines)


def fix_file_async_error_handling(filepath: str, issues: List[Dict[str, Any]]) -> bool:
    """
    Fix async error handling in a file.
    
    Args:
        filepath: Path to the file
        issues: List of issues found
        
    Returns:
        True if file was modified
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Sort issues by line number in reverse order to avoid line number shifts
        sorted_issues = sorted(issues, key=lambda x: x['line'], reverse=True)
        
        # Fix each function
        for issue in sorted_issues:
            if not issue['has_error_handling']:
                print(f"  Adding error handling to {issue['function']} at line {issue['line']}")
                content = add_error_handling_to_async_function(
                    content, 
                    issue['function'], 
                    issue['line']
                )
        
        # Write back if modified
        if content != original_content:
            # Ensure asyncio is imported if not already
            if 'import asyncio' not in content:
                lines = content.split('\n')
                # Find a good place to add the import
                import_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_index = i + 1
                lines.insert(import_index, 'import asyncio')
                content = '\n'.join(lines)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Main function to find and fix async error handling issues."""
    print("Scanning for async functions without proper error handling...")
    
    # Scan the src directory
    issues_by_file = find_async_functions_without_error_handling('src')
    
    if not issues_by_file:
        print("✅ All async functions have proper error handling!")
        return
    
    # Report findings
    total_issues = sum(len(issues) for issues in issues_by_file.values())
    print(f"\nFound {total_issues} async functions without proper error handling in {len(issues_by_file)} files:")
    
    for filepath, issues in sorted(issues_by_file.items()):
        print(f"\n{filepath}:")
        for issue in issues:
            print(f"  - {issue['function']} (line {issue['line']})")
    
    # Ask for confirmation
    response = input("\nWould you like to add error handling to these functions? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\nAdding error handling...")
        
        fixed_files = 0
        for filepath, issues in issues_by_file.items():
            print(f"\nProcessing {filepath}...")
            if fix_file_async_error_handling(filepath, issues):
                fixed_files += 1
                print(f"✅ Fixed {filepath}")
            else:
                print(f"❌ Failed to fix {filepath}")
        
        print(f"\n✅ Fixed {fixed_files} files!")
        print("\nNote: The added error handling includes:")
        print("  - try-catch blocks around function bodies")
        print("  - Proper handling of asyncio.CancelledError")
        print("  - Logging of unexpected errors")
        print("  - Re-raising exceptions to maintain error propagation")
        print("\nYou may want to customize the error handling based on specific function requirements.")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()
