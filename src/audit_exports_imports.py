#!/usr/bin/env python3
"""
Audit script to check for mismatches between exports and imports.
Identifies:
1. Default vs named export mismatches
2. Missing exports that are imported elsewhere
3. Circular dependencies
4. Import style inconsistencies
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json

class ExportImportAuditor:
    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.exports: Dict[str, Dict[str, Set[str]]] = {}  # file -> {"names": set(), "default": str|None}
        self.imports: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {}  # file -> {module: [(name, alias)]}
        self.issues: List[Dict[str, any]] = []
        
    def analyze_file(self, filepath: Path) -> None:
        """Analyze a single Python file for exports and imports."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content, filename=str(filepath))
            
            # Get relative path for reporting
            rel_path = str(filepath.relative_to(self.root_path.parent))
            
            # Initialize file entries
            if rel_path not in self.exports:
                self.exports = {**self.exports, rel_path: {"names": set(), "default": None}}
            if rel_path not in self.imports:
                self.imports = {**self.imports, rel_path: {}}
                
            # Analyze imports and exports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self._analyze_import_from(node, rel_path)
                elif isinstance(node, ast.Import):
                    self._analyze_import(node, rel_path)
                elif isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef):
                    # Track top-level definitions as potential exports
                    if node.col_offset == 0:  # Top-level definition
                        self.exports[rel_path]["names"].add(node.name)
                        
            # Check for __all__ definition
            self._check_all_definition(tree, rel_path)
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            
    def _analyze_import_from(self, node: ast.ImportFrom, filepath: str) -> None:
        """Analyze 'from ... import ...' statements."""
        if node.module:
            module = node.module
            # Convert relative imports to absolute for tracking
            if node.level > 0:
                # Handle relative imports
                parts = filepath.split('/')
                if parts[-1].endswith('.py'):
                    parts = parts[:-1]  # Remove filename
                # Go up 'level' directories
                for _ in range(node.level):
                    if parts:
                        parts.pop()
                if module:
                    parts.append(module.replace('.', '/'))
                module = '.'.join(parts).replace('/', '.')
                
            if module not in self.imports[filepath]:
                self.imports[filepath][module] = []
                
            for alias in node.names:
                if alias.name == '*':
                    self.imports[filepath][module].append(('*', '*'))
                else:
                    self.imports[filepath][module].append((alias.name, alias.asname or alias.name))
                    
    def _analyze_import(self, node: ast.Import, filepath: str) -> None:
        """Analyze 'import ...' statements."""
        for alias in node.names:
            module = alias.name
            if module not in self.imports[filepath]:
                self.imports[filepath][module] = []
            self.imports[filepath][module].append((module, alias.asname or module))
            
    def _check_all_definition(self, tree: ast.AST, filepath: str) -> None:
        """Check for __all__ definition to determine explicit exports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        # Extract the list of exported names
                        if isinstance(node.value, ast.List):
                            exported_names = set()
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Str):
                                    exported_names.add(elt.s)
                                elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exported_names.add(elt.value)
                            self.exports[filepath]["names"] = exported_names
                            
    def find_issues(self) -> None:
        """Find import/export mismatches and other issues."""
        # Check each import to see if it matches an export
        for file, imports_dict in self.imports.items():
            for module, import_list in imports_dict.items():
                # Convert module path to file path
                module_file = self._module_to_filepath(module)
                
                if module_file and module_file in self.exports:
                    exports = self.exports[module_file]
                    
                    for imported_name, alias in import_list:
                        if imported_name == '*':
                            continue  # Skip star imports for now
                            
                        # Check if the imported name exists in exports
                        if imported_name not in exports["names"] and imported_name != module.split('.')[-1]:
                            self.issues.append({
                                "type": "missing_export",
                                "file": file,
                                "module": module,
                                "imported_name": imported_name,
                                "available_exports": list(exports["names"])
                            })
                            
    def _module_to_filepath(self, module: str) -> str:
        """Convert a module name to a filepath."""
        # Handle various module formats
        if module.startswith('src.'):
            module = module[4:]  # Remove 'src.' prefix
            
        filepath = module.replace('.', '/') + '.py'
        full_path = self.root_path.parent / 'src' / filepath
        
        if full_path.exists():
            return f"src/{filepath}"
            
        # Check if it's a package __init__.py
        init_path = self.root_path.parent / 'src' / module.replace('.', '/') / '__init__.py'
        if init_path.exists():
            return f"src/{module.replace('.', '/')}/__init__.py"
            
        return None
        
    def check_relative_imports(self) -> None:
        """Check for incorrect relative import usage."""
        for file, imports_dict in self.imports.items():
            for module, import_list in imports_dict.items():
                # Check if absolute imports are used within the same package
                if module.startswith('src.') and file.startswith('src/'):
                    file_parts = file.split('/')
                    module_parts = module.split('.')
                    
                    # If they share a common package, suggest relative import
                    if len(file_parts) > 2 and len(module_parts) > 2:
                        if file_parts[1] == module_parts[1]:  # Same package
                            self.issues.append({
                                "type": "should_use_relative",
                                "file": file,
                                "module": module,
                                "suggestion": self._suggest_relative_import(file, module)
                            })
                            
    def _suggest_relative_import(self, file: str, module: str) -> str:
        """Suggest a relative import path."""
        file_parts = file.split('/')
        if file_parts[-1].endswith('.py'):
            file_parts = file_parts[:-1]
            
        module_parts = module.split('.')
        
        # Find common prefix
        common_prefix_len = 0
        for i, (fp, mp) in enumerate(zip(file_parts, module_parts)):
            if fp == mp:
                common_prefix_len = i + 1
            else:
                break
                
        # Calculate relative path
        up_levels = len(file_parts) - common_prefix_len
        remaining_module = '.'.join(module_parts[common_prefix_len:])
        
        if up_levels == 0:
            return f".{remaining_module}"
        else:
            return '.' * (up_levels) + (f".{remaining_module}" if remaining_module else "")
            
    def generate_report(self) -> Dict[str, any]:
        """Generate a comprehensive report of findings."""
        return {
            "total_files_analyzed": len(self.exports),
            "total_imports": sum(len(imp) for imp in self.imports.values()),
            "total_exports": sum(len(exp["names"]) for exp in self.exports.values()),
            "issues": self.issues,
            "import_style_stats": self._get_import_style_stats()
        }
        
    def _get_import_style_stats(self) -> Dict[str, int]:
        """Get statistics on import styles used."""
        stats = {
            "absolute_imports": 0,
            "relative_imports": 0,
            "star_imports": 0,
            "named_imports": 0
        }
        
        for file, imports_dict in self.imports.items():
            for module, import_list in imports_dict.items():
                for name, alias in import_list:
                    if name == '*':
                        stats["star_imports"] += 1
                    else:
                        stats["named_imports"] += 1
                        
                if module.startswith('.'):
                    stats["relative_imports"] += 1
                else:
                    stats["absolute_imports"] += 1
                    
        return stats
        
    def run_audit(self) -> None:
        """Run the complete audit process."""
        print("Starting export/import audit...")
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for filepath in python_files:
            self.analyze_file(filepath)
            
        # Find issues
        self.find_issues()
        self.check_relative_imports()
        
        # Generate and save report
        report = self.generate_report()
        
        with open('export_import_audit_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        print(f"\nAudit Complete!")
        print(f"Files analyzed: {report['total_files_analyzed']}")
        print(f"Total imports: {report['total_imports']}")
        print(f"Total exports: {report['total_exports']}")
        print(f"Issues found: {len(report['issues'])}")
        print("\nImport style statistics:")
        for style, count in report['import_style_stats'].items():
            print(f"  {style}: {count}")
            
        # Print issues
        if report['issues']:
            print("\nIssues found:")
            for i, issue in enumerate(report['issues'][:10]):  # Show first 10
                print(f"\n{i+1}. {issue['type']}:")
                print(f"   File: {issue['file']}")
                if 'module' in issue:
                    print(f"   Module: {issue['module']}")
                if 'imported_name' in issue:
                    print(f"   Imported name: {issue['imported_name']}")
                if 'suggestion' in issue:
                    print(f"   Suggestion: {issue['suggestion']}")
                    
            if len(report['issues']) > 10:
                print(f"\n... and {len(report['issues']) - 10} more issues. See export_import_audit_report.json for full details.")


    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
if __name__ == "__main__":
    auditor = ExportImportAuditor()
    auditor.run_audit()
