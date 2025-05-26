#!/usr/bin/env python3
"""
Add missing type annotations to all Python files in the project.
This script analyzes functions and methods to add proper type hints.
"""

import ast
import os
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from pathlib import Path

class TypeAnnotationAnalyzer(ast.NodeTransformer):
    """Analyze and add type annotations to Python AST nodes."""
    
    def __init__(self, file_content: str):
        self.file_content = file_content
        self.lines = file_content.split('\n')
        self.changes: List[Tuple[int, str, str]] = []
        self.imports_needed: Set[str] = set()
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit function definitions and add missing type annotations."""
        # Check if return type annotation is missing
        if node.returns is None:
            return_type = self._infer_return_type(node)
            if return_type:
                # Record the change needed
                line_num = node.lineno - 1
                col_offset = node.col_offset
                
                # Find the end of the function signature
                func_line = self.lines[line_num]
                match = re.search(r'def\s+' + node.name + r'\s*\([^)]*\)\s*:', func_line)
                if match:
                    insert_pos = match.end() - 1  # Before the colon
                    old_line = func_line
                    new_line = func_line[:insert_pos] + f' -> {return_type}' + func_line[insert_pos:]
                    self.changes = [*self.changes, (line_num, old_line, new_line)]
                    
                    # Add necessary imports
                    self._add_imports_for_type(return_type)
        
        # Check function arguments
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != 'self' and arg.arg != 'cls':
                # Try to infer type from usage or defaults
                arg_type = self._infer_arg_type(node, arg.arg)
                if arg_type:
                    # This is more complex as we need to modify the argument inline
                    pass
        
        self.generic_visit(node)
        return node
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Visit async function definitions."""
        # Similar to FunctionDef but for async functions
        if node.returns is None:
            return_type = self._infer_return_type(node, is_async=True)
            if return_type:
                line_num = node.lineno - 1
                func_line = self.lines[line_num]
                match = re.search(r'async\s+def\s+' + node.name + r'\s*\([^)]*\)\s*:', func_line)
                if match:
                    insert_pos = match.end() - 1
                    old_line = func_line
                    new_line = func_line[:insert_pos] + f' -> {return_type}' + func_line[insert_pos:]
                    self.changes = [*self.changes, (line_num, old_line, new_line)]
                    self._add_imports_for_type(return_type)
        
        self.generic_visit(node)
        return node
    
    def _infer_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> Optional[str]:
        """Infer the return type of a function based on its implementation."""
        # Special cases for common patterns
        if node.name == '__init__':
            return 'None'
        elif node.name == '__str__' or node.name == '__repr__':
            return 'str'
        elif node.name.startswith('__') and node.name.endswith('__'):
            # Other magic methods
            return self._get_magic_method_return_type(node.name)
        
        # Analyze return statements
        return_types = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return_types.add('None')
                else:
                    inferred = self._infer_expr_type(stmt.value)
                    if inferred:
                        return_types.add(inferred)
        
        if not return_types:
            # No explicit return, functions return None by default
            return 'None'
        elif len(return_types) == 1:
            return_type = return_types.pop()
            if is_async and not return_type.startswith('Coroutine'):
                # Wrap in appropriate async type
                return f'Coroutine[Any, Any, {return_type}]'
            return return_type
        else:
            # Multiple return types
            types_str = ', '.join(sorted(return_types))
            if 'None' in return_types:
                # Optional pattern
                other_types = return_types - {'None'}
                if len(other_types) == 1:
                    return f'Optional[{other_types.pop()}]'
            return f'Union[{types_str}]'
    
    def _infer_expr_type(self, expr: ast.expr) -> Optional[str]:
        """Infer the type of an expression."""
        if isinstance(expr, ast.Constant):
            # Literal values
            value = expr.value
            if isinstance(value, str):
                return 'str'
            elif isinstance(value, int):
                return 'int'
            elif isinstance(value, float):
                return 'float'
            elif isinstance(value, bool):
                return 'bool'
            elif value is None:
                return 'None'
        elif isinstance(expr, ast.Dict):
            return 'Dict[str, Any]'
        elif isinstance(expr, ast.List):
            return 'List[Any]'
        elif isinstance(expr, ast.Set):
            return 'Set[Any]'
        elif isinstance(expr, ast.Tuple):
            return 'Tuple[Any, ...]'
        elif isinstance(expr, ast.Name):
            # Variable reference - harder to infer
            if expr.id in ('True', 'False'):
                return 'bool'
            elif expr.id == 'None':
                return 'None'
        elif isinstance(expr, ast.Call):
            # Function call
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id
                # Common constructors
                if func_name == 'dict':
                    return 'Dict[Any, Any]'
                elif func_name == 'list':
                    return 'List[Any]'
                elif func_name == 'set':
                    return 'Set[Any]'
                elif func_name == 'tuple':
                    return 'Tuple[Any, ...]'
                elif func_name == 'str':
                    return 'str'
                elif func_name == 'int':
                    return 'int'
                elif func_name == 'float':
                    return 'float'
                elif func_name == 'bool':
                    return 'bool'
        
        return None
    
    def _get_magic_method_return_type(self, method_name: str) -> str:
        """Get return type for magic methods."""
        magic_returns = {
            '__len__': 'int',
            '__bool__': 'bool',
            '__bytes__': 'bytes',
            '__hash__': 'int',
            '__int__': 'int',
            '__float__': 'float',
            '__complex__': 'complex',
            '__index__': 'int',
            '__round__': 'int',
            '__trunc__': 'int',
            '__floor__': 'int',
            '__ceil__': 'int',
            '__contains__': 'bool',
            '__eq__': 'bool',
            '__ne__': 'bool',
            '__lt__': 'bool',
            '__le__': 'bool',
            '__gt__': 'bool',
            '__ge__': 'bool',
        }
        return magic_returns.get(method_name, 'Any')
    
    def _infer_arg_type(self, node: ast.FunctionDef, arg_name: str) -> Optional[str]:
        """Infer argument type based on usage in function."""
        # This is a simplified version - real inference would be more complex
        return None
    
    def _add_imports_for_type(self, type_str: str) -> None:
        """Add necessary imports for a type annotation."""
        if 'Dict' in type_str:
            self.imports_needed.add('Dict')
        if 'List' in type_str:
            self.imports_needed.add('List')
        if 'Set' in type_str:
            self.imports_needed.add('Set')
        if 'Tuple' in type_str:
            self.imports_needed.add('Tuple')
        if 'Optional' in type_str:
            self.imports_needed.add('Optional')
        if 'Union' in type_str:
            self.imports_needed.add('Union')
        if 'Any' in type_str:
            self.imports_needed.add('Any')
        if 'Coroutine' in type_str:
            self.imports_needed.add('Coroutine')


def process_file(file_path: Path) -> bool:
    """Process a single Python file to add type annotations."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Analyze for type annotations
        analyzer = TypeAnnotationAnalyzer(content)
        analyzer.visit(tree)
        
        if not analyzer.changes:
            return False
        
        # Apply changes
        lines = content.split('\n')
        
        # Apply changes in reverse order to maintain line numbers
        for line_num, old_line, new_line in reversed(analyzer.changes):
            if line_num < len(lines) and lines[line_num] == old_line:
                lines[line_num] = new_line
        
        # Add imports if needed
        if analyzer.imports_needed:
            # Find where to insert imports
            import_line = -1
            for i, line in enumerate(lines):
                if line.startswith('from typing import'):
                    # Update existing typing import
                    existing_imports = re.findall(r'\b\w+\b', line[len('from typing import'):])
                    all_imports = set(existing_imports) | analyzer.imports_needed
                    lines[i] = f"from typing import {', '.join(sorted(all_imports))}"
                    analyzer.imports_needed.clear()
                    break
                elif line.startswith('import') or line.startswith('from'):
                    import_line = i
            
            if analyzer.imports_needed:
                # Add new typing import
                import_stmt = f"from typing import {', '.join(sorted(analyzer.imports_needed))}"
                if import_line >= 0:
                    lines.insert(import_line + 1, import_stmt)
                else:
                    # Add after module docstring
                    insert_pos = 0
                    if lines[0].startswith('#!'):
                        insert_pos = 1
                    if insert_pos < len(lines) and lines[insert_pos].startswith('"""'):
                        # Find end of docstring
                        for i in range(insert_pos + 1, len(lines)):
                            if '"""' in lines[i]:
                                insert_pos = i + 1
                                break
                    lines.insert(insert_pos, '')
                    lines.insert(insert_pos + 1, import_stmt)
        
        # Write back
        new_content = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all Python files."""
    # Get all Python files in src directory
    src_dir = Path('src')
    python_files = list(src_dir.rglob('*.py'))
    
    # Exclude this script and other utility scripts
    exclude_files = {
        'add_type_annotations.py',
        'fix_universal_agent_connector.py',
        'fix_test_issues.py',
        'fix_datetime_utc_issues.py',
        'fix_remaining_issues.py',
        'fix_final_issues.py',
        'fix_universal_agent_connector_indent.py',
        'fix_critical_issues.py',
        'update_deprecated_datetime.py',
        'fix_import_paths.py',
        'setup_claude_key.py',
        'run_comprehensive_tests.py',
        'run_real_test.py',
        'run_multi_model_agent.py',
        'simple_test.py',
        'debug_test.py',
        'test_endpoints.py',
        'download_mistral_model.py'
    }
    
    python_files = [f for f in python_files if f.name not in exclude_files]
    
    print(f"Found {len(python_files)} Python files to process")
    
    modified_count = 0
    for file_path in python_files:
        print(f"Processing {file_path}...")
        if process_file(file_path):
            modified_count += 1
            print(f"  ✓ Modified {file_path}")
        else:
            print(f"  - No changes needed for {file_path}")
    
    print(f"\nModified {modified_count} files")


if __name__ == "__main__":
    main()
