#!/usr/bin/env python3
"""
Script to detect and fix direct state mutations in Python code.
Ensures immutability principles are followed.
"""

import os
import ast
import re
from typing import List, Tuple, Dict, Set
from pathlib import Path


class StateMutationDetector(ast.NodeVisitor):
    """AST visitor to detect state mutations."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.mutations = []
        self.in_init = False
        self.in_class = False
        self.current_class = None
        self.current_function = None
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.in_class = True
        self.generic_visit(node)
        self.current_class = old_class
        self.in_class = self.current_class is not None
        
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        old_in_init = self.in_init
        self.current_function = node.name
        self.in_init = (node.name == '__init__' and self.in_class)
        self.generic_visit(node)
        self.current_function = old_function
        self.in_init = old_in_init
        
    def visit_Call(self, node):
        """Detect method calls that mutate state."""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # List of mutating methods
            mutating_methods = {
                'append', 'extend', 'insert', 'remove', 'pop', 'clear',
                'reverse', 'sort', 'update', 'setdefault', 'popitem'
            }
            
            if method_name in mutating_methods:
                # Check if it's called on self attribute
                if isinstance(node.func.value, ast.Attribute) and \
                   isinstance(node.func.value.value, ast.Name) and \
                   node.func.value.value.id == 'self':
                    
                    # Allow mutations in __init__ for initial setup
                    if not self.in_init:
                        self.mutations.append({
                            'type': 'method_call',
                            'line': node.lineno,
                            'method': method_name,
                            'attribute': node.func.value.attr,
                            'context': f"{self.current_class}.{self.current_function}"
                        })
        
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Detect direct assignments to attributes."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                # Check for self.attribute[key] = value
                if isinstance(target.value, ast.Attribute) and \
                   isinstance(target.value.value, ast.Name) and \
                   target.value.value.id == 'self':
                    
                    # Allow in __init__
                    if not self.in_init:
                        self.mutations.append({
                            'type': 'subscript_assign',
                            'line': node.lineno,
                            'attribute': target.value.attr,
                            'context': f"{self.current_class}.{self.current_function}"
                        })
            elif isinstance(target, ast.Attribute):
                # Check for self.attribute = value (reassignment)
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    # Only flag if it's a collection being directly modified
                    # (not initial assignment in __init__)
                    if not self.in_init and self._is_collection_reassignment(node):
                        self.mutations.append({
                            'type': 'attribute_reassign',
                            'line': node.lineno,
                            'attribute': target.attr,
                            'context': f"{self.current_class}.{self.current_function}"
                        })
        
        self.generic_visit(node)
        
    def visit_AugAssign(self, node):
        """Detect augmented assignments like +=, -=, etc."""
        if isinstance(node.target, ast.Attribute) and \
           isinstance(node.target.value, ast.Name) and \
           node.target.value.id == 'self':
            
            if not self.in_init:
                self.mutations.append({
                    'type': 'aug_assign',
                    'line': node.lineno,
                    'attribute': node.target.attr,
                    'op': type(node.op).__name__,
                    'context': f"{self.current_class}.{self.current_function}"
                })
        
        self.generic_visit(node)
        
    def visit_Delete(self, node):
        """Detect del statements."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Attribute) and \
                   isinstance(target.value.value, ast.Name) and \
                   target.value.value.id == 'self':
                    
                    self.mutations.append({
                        'type': 'delete',
                        'line': node.lineno,
                        'attribute': target.value.attr,
                        'context': f"{self.current_class}.{self.current_function}"
                    })
        
        self.generic_visit(node)
    
    def _is_collection_reassignment(self, node):
        """Check if assignment is modifying a collection."""
        # Simple heuristic: if the value is a method call on the same attribute
        if isinstance(node.value, ast.Call) and \
           isinstance(node.value.func, ast.Attribute):
            return True
        return False


class StateMutationFixer:
    """Fix state mutations to follow immutability principles."""
    
    def __init__(self):
        self.fixes = []
        
    def fix_file(self, filepath: str, mutations: List[Dict]) -> str:
        """Fix mutations in a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Sort mutations by line number in reverse order to avoid offset issues
        mutations = sorted(mutations, key=lambda x: x['line'], reverse=True)
        
        for mutation in mutations:
            line_idx = mutation['line'] - 1
            if line_idx >= len(lines):
                continue
                
            original_line = lines[line_idx]
            fixed_line = self._fix_mutation(original_line, mutation)
            
            if fixed_line != original_line:
                lines[line_idx] = fixed_line
                self.fixes.append({
                    'file': filepath,
                    'line': mutation['line'],
                    'type': mutation['type'],
                    'original': original_line.strip(),
                    'fixed': fixed_line.strip()
                })
        
        return ''.join(lines)
    
    def _fix_mutation(self, line: str, mutation: Dict) -> str:
        """Fix a specific mutation."""
        indent = len(line) - len(line.lstrip())
        indent_str = line[:indent]
        
        if mutation['type'] == 'method_call':
            return self._fix_method_call(line, mutation, indent_str)
        elif mutation['type'] == 'subscript_assign':
            return self._fix_subscript_assign(line, mutation, indent_str)
        elif mutation['type'] == 'aug_assign':
            return self._fix_aug_assign(line, mutation, indent_str)
        elif mutation['type'] == 'delete':
            return self._fix_delete(line, mutation, indent_str)
        
        return line
    
    def _fix_method_call(self, line: str, mutation: Dict, indent: str) -> str:
        """Fix mutating method calls."""
        attr = mutation['attribute']
        method = mutation['method']
        
        if method == 'append':
            # self.items.append(x) -> self.items = [*self.items, x]
            match = re.search(rf'self\.{attr}\.append\((.*?)\)', line)
            if match:
                value = match.group(1)
                return f"{indent}self.{attr} = [*self.{attr}, {value}]\n"
                
        elif method == 'extend':
            # self.items.extend(x) -> self.items = [*self.items, *x]
            match = re.search(rf'self\.{attr}\.extend\((.*?)\)', line)
            if match:
                value = match.group(1)
                return f"{indent}self.{attr} = [*self.{attr}, *{value}]\n"
                
        elif method == 'update':
            # self.data.update(x) -> self.data = {{**self.data, **x}}
            match = re.search(rf'self\.{attr}\.update\((.*?)\)', line)
            if match:
                value = match.group(1)
                return f"{indent}self.{attr} = {{**self.{attr}, **{value}}}\n"
                
        elif method == 'clear':
            # self.items.clear() -> self.items = [] or {}
            if '.clear()' in line:
                # Try to infer type from attribute name
                if any(keyword in attr.lower() for keyword in ['list', 'items', 'queue', 'stack']):
                    return f"{indent}self.{attr} = []\n"
                else:
                    return f"{indent}self.{attr} = {{}}\n"
                    
        elif method == 'pop':
            # self.items.pop() -> self.items = self.items[:-1]
            if f'self.{attr}.pop()' in line:
                # Check if the result is being used
                if '=' in line.split(f'self.{attr}.pop()')[0]:
                    # Result is being assigned, need to handle differently
                    var_match = re.match(r'\s*(\w+)\s*=', line)
                    if var_match:
                        var_name = var_match.group(1)
                        return f"{indent}{var_name} = self.{attr}[-1] if self.{attr} else None\n{indent}self.{attr} = self.{attr}[:-1]\n"
                else:
                    return f"{indent}self.{attr} = self.{attr}[:-1]\n"
                    
        elif method == 'remove':
            # self.items.remove(x) -> self.items = [i for i in self.items if i != x]
            match = re.search(rf'self\.{attr}\.remove\((.*?)\)', line)
            if match:
                value = match.group(1)
                return f"{indent}self.{attr} = [i for i in self.{attr} if i != {value}]\n"
        
        return line
    
    def _fix_subscript_assign(self, line: str, mutation: Dict, indent: str) -> str:
        """Fix subscript assignments."""
        attr = mutation['attribute']
        
        # self.data[key] = value -> self.data = {**self.data, key: value}
        match = re.search(rf'self\.{attr}\[(.*?)\]\s*=\s*(.*?)$', line.strip())
        if match:
            key = match.group(1)
            value = match.group(2)
            return f"{indent}self.{attr} = {{**self.{attr}, {key}: {value}}}\n"
        
        return line
    
    def _fix_aug_assign(self, line: str, mutation: Dict, indent: str) -> str:
        """Fix augmented assignments."""
        attr = mutation['attribute']
        op = mutation['op']
        
        # Map operator types to symbols
        op_map = {
            'Add': '+',
            'Sub': '-',
            'Mult': '*',
            'Div': '/',
            'Mod': '%',
            'Pow': '**',
            'LShift': '<<',
            'RShift': '>>',
            'BitOr': '|',
            'BitXor': '^',
            'BitAnd': '&',
            'FloorDiv': '//'
        }
        
        op_symbol = op_map.get(op, '+')
        
        # self.count += 1 -> self.count = self.count + 1
        match = re.search(rf'self\.{attr}\s*{re.escape(op_symbol)}=\s*(.*?)$', line.strip())
        if match:
            value = match.group(1)
            return f"{indent}self.{attr} = self.{attr} {op_symbol} {value}\n"
        
        return line
    
    def _fix_delete(self, line: str, mutation: Dict, indent: str) -> str:
        """Fix delete statements."""
        attr = mutation['attribute']
        
        # del self.data[key] -> self.data = {k: v for k, v in self.data.items() if k != key}
        match = re.search(rf'del\s+self\.{attr}\[(.*?)\]', line)
        if match:
            key = match.group(1)
            return f"{indent}self.{attr} = {{k: v for k, v in self.{attr}.items() if k != {key}}}\n"
        
        return line


def analyze_file(filepath: str) -> List[Dict]:
    """Analyze a single file for state mutations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        detector = StateMutationDetector(filepath)
        detector.visit(tree)
        
        return detector.mutations
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return []


def main():
    """Main function to detect and fix state mutations."""
    # Use current directory if we're already in src, otherwise use src
    current_dir = Path.cwd()
    if current_dir.name == 'src':
        src_dir = current_dir
    else:
        src_dir = Path('src')
    
    # Collect all Python files
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip test files and fix scripts
        if 'test' in root or 'fix_' in os.path.basename(root):
            continue
            
        for file in files:
            if file.endswith('.py') and not file.startswith(('test_', 'fix_')):
                python_files.append(os.path.join(root, file))
    
    print(f"Analyzing {len(python_files)} Python files for state mutations...")
    
    # Analyze all files
    all_mutations = {}
    total_mutations = 0
    
    for filepath in python_files:
        mutations = analyze_file(filepath)
        if mutations:
            all_mutations[filepath] = mutations
            total_mutations += len(mutations)
            print(f"\n{filepath}: {len(mutations)} mutations found")
            for mut in mutations[:5]:  # Show first 5
                print(f"  Line {mut['line']}: {mut['type']} on self.{mut['attribute']}")
            if len(mutations) > 5:
                print(f"  ... and {len(mutations) - 5} more")
    
    print(f"\n\nTotal mutations found: {total_mutations}")
    
    if total_mutations == 0:
        print("No state mutations found!")
        return
    
    # Ask user if they want to fix the mutations
    response = input("\nDo you want to automatically fix these mutations? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without making changes.")
        return
    
    # Fix mutations
    fixer = StateMutationFixer()
    fixed_files = 0
    
    for filepath, mutations in all_mutations.items():
        print(f"\nFixing {filepath}...")
        
        try:
            fixed_content = fixer.fix_file(filepath, mutations)
            
            # Write the fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            fixed_files += 1
            print(f"  Fixed {len(mutations)} mutations")
            
        except Exception as e:
            print(f"  Error fixing file: {e}")
    
    # Summary
    print(f"\n\nSummary:")
    print(f"  Files analyzed: {len(python_files)}")
    print(f"  Files with mutations: {len(all_mutations)}")
    print(f"  Files fixed: {fixed_files}")
    print(f"  Total fixes applied: {len(fixer.fixes)}")
    
    # Show some example fixes
    if fixer.fixes:
        print("\nExample fixes applied:")
        for fix in fixer.fixes[:10]:
            print(f"\n{fix['file']}:{fix['line']}")
            print(f"  Type: {fix['type']}")
            print(f"  Before: {fix['original']}")
            print(f"  After:  {fix['fixed']}")
    
    # Create a detailed report
    report_path = 'state_mutations_report.txt'
    with open(report_path, 'w') as f:
        f.write("State Mutations Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total files analyzed: {len(python_files)}\n")
        f.write(f"Files with mutations: {len(all_mutations)}\n")
        f.write(f"Total mutations found: {total_mutations}\n")
        f.write(f"Files fixed: {fixed_files}\n")
        f.write(f"Total fixes applied: {len(fixer.fixes)}\n\n")
        
        f.write("Mutations by type:\n")
        type_counts = {}
        for mutations in all_mutations.values():
            for mut in mutations:
                type_counts[mut['type']] = type_counts.get(mut['type'], 0) + 1
        
        for mut_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {mut_type}: {count}\n")
        
        f.write("\n\nDetailed fixes:\n")
        f.write("-" * 50 + "\n")
        
        for fix in fixer.fixes:
            f.write(f"\n{fix['file']}:{fix['line']} ({fix['type']})\n")
            f.write(f"  Original: {fix['original']}\n")
            f.write(f"  Fixed:    {fix['fixed']}\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Create a best practices guide
    guide_path = 'immutability_guide.md'
    with open(guide_path, 'w') as f:
        f.write("# Immutability Best Practices Guide\n\n")
        f.write("## Why Immutability?\n\n")
        f.write("Immutability helps prevent bugs by ensuring that data structures ")
        f.write("aren't accidentally modified. This makes code more predictable ")
        f.write("and easier to reason about.\n\n")
        
        f.write("## Common Patterns\n\n")
        f.write("### Lists\n")
        f.write("```python\n")
        f.write("# Instead of:\n")
        f.write("self.items.append(item)\n\n")
        f.write("# Use:\n")
        f.write("self.items = [*self.items, item]\n")
        f.write("```\n\n")
        
        f.write("### Dictionaries\n")
        f.write("```python\n")
        f.write("# Instead of:\n")
        f.write("self.data[key] = value\n\n")
        f.write("# Use:\n")
        f.write("self.data = {**self.data, key: value}\n")
        f.write("```\n\n")
        
        f.write("### Removing items\n")
        f.write("```python\n")
        f.write("# Instead of:\n")
        f.write("self.items.remove(item)\n\n")
        f.write("# Use:\n")
        f.write("self.items = [i for i in self.items if i != item]\n")
        f.write("```\n\n")
        
        f.write("## When Mutation is Acceptable\n\n")
        f.write("1. In `__init__` methods for initial setup\n")
        f.write("2. In performance-critical sections (document why)\n")
        f.write("3. When using thread-safe collections with proper locking\n\n")
        
        f.write("## Alternative Approaches\n\n")
        f.write("1. Use immutable data structures (tuple, frozenset)\n")
        f.write("2. Return new instances from methods\n")
        f.write("3. Use dataclasses with frozen=True\n")
        f.write("4. Consider using libraries like `pyrsistent` for persistent data structures\n")
    
    print(f"Immutability guide saved to: {guide_path}")


if __name__ == '__main__':
    main()
