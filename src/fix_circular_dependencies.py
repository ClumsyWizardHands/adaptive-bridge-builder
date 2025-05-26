#!/usr/bin/env python3
"""
Fix circular dependencies in the project.
"""

import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, deque


class CircularDependencyFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.import_graph = defaultdict(set)
        self.circular_deps = []
        self.fixes_applied = []
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Relative import
                            parts = file_path.relative_to(self.src_root).parts
                            if node.level <= len(parts):
                                base_module = '.'.join(parts[:-node.level])
                                if base_module and node.module:
                                    imports.add(f"{base_module}.{node.module}")
                                elif node.module:
                                    imports.add(node.module)
                        else:
                            # Absolute import
                            imports.add(node.module)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return imports
    
    def build_import_graph(self):
        """Build a graph of module dependencies."""
        print("Building import dependency graph...")
        
        # Get all Python files
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            if "__pycache__" in str(file_path):
                continue
            
            # Get module name from file path
            module_name = self.get_module_name(file_path)
            
            # Extract imports
            imports = self.extract_imports(file_path)
            
            # Add to graph
            for imp in imports:
                # Only track internal imports
                if self.is_internal_import(imp):
                    self.import_graph[module_name].add(imp)
    
    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.src_root)
        if rel_path.name == "__init__.py":
            parts = rel_path.parts[:-1]
        else:
            parts = rel_path.parts[:-1] + (rel_path.stem,)
        return '.'.join(parts)
    
    def is_internal_import(self, module: str) -> bool:
        """Check if an import is internal to the project."""
        # Check if module exists in src directory
        parts = module.split('.')
        potential_paths = [
            self.src_root / '/'.join(parts) / "__init__.py",
            self.src_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py" if parts else None,
            self.src_root / f"{parts[0]}.py" if len(parts) == 1 else None
        ]
        
        return any(p and p.exists() for p in potential_paths if p)
    
    def find_circular_dependencies(self):
        """Find all circular dependencies using DFS."""
        print("Detecting circular dependencies...")
        
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(module: str) -> bool:
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            for neighbor in self.import_graph.get(module, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found circular dependency
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    self.circular_deps.append(cycle)
                    return True
            
            path.pop()
            rec_stack.remove(module)
            return False
        
        for module in self.import_graph:
            if module not in visited:
                dfs(module)
        
        # Remove duplicate cycles
        unique_cycles = []
        seen = set()
        for cycle in self.circular_deps:
            # Normalize cycle (start from lexicographically smallest)
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized))
        
        self.circular_deps = unique_cycles
    
    def fix_circular_dependency(self, cycle: List[str]):
        """Fix a circular dependency by refactoring imports."""
        print(f"\nFixing circular dependency: {' -> '.join(cycle)}")
        
        # Strategy 1: Move imports inside functions (lazy import)
        for i, module in enumerate(cycle[:-1]):
            next_module = cycle[i + 1]
            
            # Find the file
            file_path = self.find_module_file(module)
            if not file_path:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                # Find import of next_module
                import_lines = []
                for j, line in enumerate(lines):
                    if self.is_import_line(line, next_module):
                        import_lines.append(j)
                
                if not import_lines:
                    continue
                
                # Check if import is used at module level
                tree = ast.parse(content)
                module_level_usage = False
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue  # Skip function/class bodies for now
                    if isinstance(node, ast.Name) and node.id in self.get_imported_names(next_module):
                        module_level_usage = True
                        break
                
                if not module_level_usage:
                    # Move import to function level
                    self.move_imports_to_functions(file_path, next_module)
                    self.fixes_applied.append(
                        f"Moved import of {next_module} to function level in {module}"
                    )
                else:
                    # Try to refactor the dependency
                    print(f"  - Module-level usage detected in {module}, considering refactoring")
                    
            except Exception as e:
                print(f"  - Error fixing {module}: {e}")
    
    def find_module_file(self, module: str) -> Path:
        """Find the file path for a module."""
        parts = module.split('.')
        potential_paths = [
            self.src_root / '/'.join(parts) / "__init__.py",
            self.src_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py" if parts else None,
            self.src_root / f"{parts[0]}.py" if len(parts) == 1 else None
        ]
        
        for path in potential_paths:
            if path and path.exists():
                return path
        return None
    
    def is_import_line(self, line: str, module: str) -> bool:
        """Check if a line imports the given module."""
        line = line.strip()
        return (
            f"import {module}" in line or
            f"from {module}" in line or
            f"from .{module}" in line or
            f"from ..{module}" in line
        )
    
    def get_imported_names(self, module: str) -> Set[str]:
        """Get names that would be imported from a module."""
        # This is a simplified version - in reality would need to parse the import
        parts = module.split('.')
        return {parts[-1], module.replace('.', '_')}
    
    def move_imports_to_functions(self, file_path: Path, module_to_move: str):
        """Move imports inside functions to break circular dependencies."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.splitlines()
        
        # Find functions that use the imported module
        functions_using_import = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id in self.get_imported_names(module_to_move):
                        functions_using_import.append(node)
                        break
        
        if not functions_using_import:
            return
        
        # Remove top-level import
        new_lines = []
        import_line = None
        for i, line in enumerate(lines):
            if self.is_import_line(line, module_to_move) and import_line is None:
                import_line = line.strip()
                continue  # Skip this line
            new_lines.append(line)
        
        if not import_line:
            return
        
        # Add import to each function that uses it
        lines = new_lines
        offset = 0
        for func in functions_using_import:
            # Find the first line of the function body
            func_line = func.lineno - 1 + offset
            # Find the first non-decorator line
            while func_line < len(lines) and lines[func_line].strip().startswith('@'):
                func_line += 1
            # Find the colon
            while func_line < len(lines) and ':' not in lines[func_line]:
                func_line += 1
            # Insert import after function definition
            if func_line < len(lines):
                indent = self.get_indent(lines[func_line + 1]) if func_line + 1 < len(lines) else "    "
                lines.insert(func_line + 1, f"{indent}{import_line}")
                offset += 1
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def get_indent(self, line: str) -> str:
        """Get the indentation of a line."""
        return line[:len(line) - len(line.lstrip())]
    
    def fix_all_circular_dependencies(self):
        """Find and fix all circular dependencies."""
        # Build import graph
        self.build_import_graph()
        
        # Find circular dependencies
        self.find_circular_dependencies()
        
        if not self.circular_deps:
            print("No circular dependencies found!")
            return
        
        print(f"\nFound {len(self.circular_deps)} circular dependencies:")
        for cycle in self.circular_deps:
            print(f"  - {' -> '.join(cycle)}")
        
        # Fix each circular dependency
        for cycle in self.circular_deps:
            self.fix_circular_dependency(cycle)
        
        print(f"\nApplied {len(self.fixes_applied)} fixes:")
        for fix in self.fixes_applied:
            print(f"  - {fix}")
        
        print("\nCircular dependency fixes completed!")
        print("\nNote: Some circular dependencies may require manual refactoring.")
        print("Consider:")
        print("  1. Moving shared code to a separate module")
        print("  2. Using dependency injection")
        print("  3. Refactoring to reduce coupling between modules")


def main():
    """Run the circular dependency fixer."""
    fixer = CircularDependencyFixer()
    fixer.fix_all_circular_dependencies()


if __name__ == "__main__":
    main()
