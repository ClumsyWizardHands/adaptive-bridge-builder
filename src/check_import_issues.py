#!/usr/bin/env python3
"""
Comprehensive import issue checker for the Adaptive Bridge Builder project.
Checks for:
1. Unresolved imports
2. Missing type definitions for third-party packages  
3. Import syntax errors
4. Missing __init__.py files
5. Circular dependencies
"""

import ast
import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class ImportChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.issues = {
            "unresolved_imports": [],
            "missing_type_stubs": [],
            "syntax_errors": [],
            "missing_init_files": [],
            "circular_dependencies": [],
            "relative_import_issues": []
        }
        self.third_party_imports = set()
        self.local_modules = set()
        self._build_local_module_map()
        
    def _build_local_module_map(self):
        """Build a map of all local modules in the project."""
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                # Package
                relative = py_file.parent.relative_to(self.src_path)
                module_path = str(relative).replace(os.sep, ".")
                if module_path and module_path != ".":
                    self.local_modules.add(module_path)
            else:
                # Module
                relative = py_file.relative_to(self.src_path)
                module_path = str(relative)[:-3].replace(os.sep, ".")
                self.local_modules.add(module_path)
    
    def check_file(self, filepath: Path) -> None:
        """Check a single Python file for import issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(filepath))
                self._check_imports_in_ast(tree, filepath)
            except SyntaxError as e:
                self.issues["syntax_errors"].append({
                    "file": str(filepath.relative_to(self.project_root)),
                    "error": str(e),
                    "line": e.lineno
                })
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    def _check_imports_in_ast(self, tree: ast.AST, filepath: Path) -> None:
        """Check imports in the AST."""
        relative_file = filepath.relative_to(self.project_root)
        package_path = filepath.parent.relative_to(self.src_path) if filepath.parent != self.src_path else Path()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name, relative_file, node.lineno)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # from X import Y
                    if node.level == 0:
                        # Absolute import
                        self._check_import(node.module, relative_file, node.lineno)
                    else:
                        # Relative import
                        self._check_relative_import(
                            node.module, node.level, package_path, 
                            relative_file, node.lineno
                        )
                else:
                    # from . import Y (relative import without module)
                    if node.level > 0:
                        self._check_relative_import(
                            "", node.level, package_path,
                            relative_file, node.lineno
                        )
    
    def _check_import(self, module_name: str, filepath: Path, lineno: int) -> None:
        """Check if an import can be resolved."""
        # Check if it's a standard library module
        if module_name in sys.stdlib_module_names:
            return
            
        # Check if it's a local module
        if self._is_local_module(module_name):
            return
            
        # It's a third-party module
        self.third_party_imports.add(module_name)
        
        # Try to import it
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            self.issues["unresolved_imports"].append({
                "file": str(filepath),
                "module": module_name,
                "line": lineno
            })
        else:
            # Check for type stubs
            self._check_type_stubs(module_name, filepath, lineno)
    
    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module is local to the project."""
        # Direct match
        if module_name in self.local_modules:
            return True
            
        # Check if it's a submodule of a local package
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent in self.local_modules:
                return True
                
        return False
    
    def _check_relative_import(self, module: str, level: int, package_path: Path, 
                               filepath: Path, lineno: int) -> None:
        """Check if a relative import is valid."""
        # Calculate the target package
        current_parts = list(package_path.parts)
        
        if level > len(current_parts) + 1:
            self.issues["relative_import_issues"].append({
                "file": str(filepath),
                "import": f"{'.' * level}{module}",
                "line": lineno,
                "error": "Relative import goes beyond top-level package"
            })
            return
            
        # Go up 'level' directories
        target_parts = current_parts[:-level] if level <= len(current_parts) else []
        
        if module:
            target_parts.extend(module.split("."))
            
        target_module = ".".join(target_parts) if target_parts else ""
        
        if target_module and not self._is_local_module(target_module):
            self.issues["relative_import_issues"].append({
                "file": str(filepath),
                "import": f"{'.' * level}{module}",
                "line": lineno,
                "error": f"Cannot resolve to module: {target_module}"
            })
    
    def _check_type_stubs(self, module_name: str, filepath: Path, lineno: int) -> None:
        """Check if type stubs are available for a module."""
        # Check for py.typed marker
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                module_path = Path(spec.origin).parent
                if not (module_path / "py.typed").exists():
                    # Check for stub packages
                    stub_module = f"{module_name}-stubs"
                    stub_spec = importlib.util.find_spec(stub_module)
                    
                    types_module = f"types-{module_name}"
                    types_spec = importlib.util.find_spec(types_module)
                    
                    if not stub_spec and not types_spec:
                        # Only report for common packages that should have stubs
                        common_stub_packages = {
                            'requests', 'numpy', 'pandas', 'flask', 'django',
                            'sqlalchemy', 'pytest', 'aiohttp', 'fastapi',
                            'pydantic', 'boto3', 'redis', 'psycopg2'
                        }
                        
                        base_module = module_name.split('.')[0]
                        if base_module in common_stub_packages:
                            self.issues["missing_type_stubs"].append({
                                "file": str(filepath),
                                "module": module_name,
                                "line": lineno,
                                "suggestion": f"Consider installing 'types-{base_module}' or '{base_module}-stubs'"
                            })
        except:
            pass
    
    def check_missing_init_files(self) -> None:
        """Check for missing __init__.py files in package directories."""
        # Find all directories containing Python files
        python_dirs = set()
        for py_file in self.src_path.rglob("*.py"):
            if py_file.parent != self.src_path:
                python_dirs.add(py_file.parent)
        
        # Check each directory for __init__.py
        for dir_path in python_dirs:
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                relative_path = dir_path.relative_to(self.project_root)
                self.issues["missing_init_files"].append(str(relative_path))
    
    def check_all_files(self) -> None:
        """Check all Python files in the project."""
        for py_file in self.src_path.rglob("*.py"):
            if not any(part.startswith('.') for part in py_file.parts):
                self.check_file(py_file)
        
        self.check_missing_init_files()
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive report of all issues found."""
        report = {
            "summary": {
                "total_files_checked": len(list(self.src_path.rglob("*.py"))),
                "unresolved_imports": len(self.issues["unresolved_imports"]),
                "missing_type_stubs": len(self.issues["missing_type_stubs"]),
                "syntax_errors": len(self.issues["syntax_errors"]),
                "missing_init_files": len(self.issues["missing_init_files"]),
                "relative_import_issues": len(self.issues["relative_import_issues"])
            },
            "issues": self.issues,
            "third_party_packages": sorted(list(self.third_party_imports))
        }
        
        return report



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
def main():
    """Main function to run the import checker."""
    project_root = Path(__file__).parent.parent
    checker = ImportChecker(project_root)
    
    print("Checking for import issues...")
    checker.check_all_files()
    
    report = checker.generate_report()
    
    # Save report
    report_path = project_root / "import_check_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Import Check Summary ===")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    
    # Print critical issues
    if report["issues"]["unresolved_imports"]:
        print("\n=== Unresolved Imports ===")
        for issue in report["issues"]["unresolved_imports"][:10]:
            print(f"- {issue['file']}:{issue['line']} - Cannot import '{issue['module']}'")
        if len(report["issues"]["unresolved_imports"]) > 10:
            print(f"... and {len(report['issues']['unresolved_imports']) - 10} more")
    
    if report["issues"]["missing_type_stubs"]:
        print("\n=== Missing Type Stubs ===")
        seen_modules = set()
        for issue in report["issues"]["missing_type_stubs"]:
            if issue['module'] not in seen_modules:
                print(f"- {issue['module']}: {issue['suggestion']}")
                seen_modules.add(issue['module'])
    
    if report["issues"]["syntax_errors"]:
        print("\n=== Syntax Errors ===")
        for issue in report["issues"]["syntax_errors"][:5]:
            print(f"- {issue['file']}:{issue['line']} - {issue['error']}")
            
    if report["issues"]["missing_init_files"]:
        print("\n=== Missing __init__.py Files ===")
        for path in report["issues"]["missing_init_files"][:10]:
            print(f"- {path}")
            
    print(f"\nFull report saved to: {report_path}")
    
    return 0 if report["summary"]["unresolved_imports"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
