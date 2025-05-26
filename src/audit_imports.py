#!/usr/bin/env python3
"""
Comprehensive import statement auditor for the Alex Familiar project.
Identifies and fixes circular dependencies, missing imports, and incorrect paths.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json


class ImportAuditor:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.module_paths: Dict[str, Path] = {}
        self.issues: List[Dict[str, any]] = []
        self.circular_dependencies: List[List[str]] = []
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, _, files in os.walk(self.src_root):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            # Get relative path from src directory
            rel_path = file_path.relative_to(self.src_root)
            # Convert to module name
            parts = rel_path.parts[:-1] + (rel_path.stem,)
            return '.'.join(parts)
        except ValueError:
            # File is not under src directory
            return str(file_path.stem)
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # This is a relative import
                            module_name = self.get_module_name(file_path)
                            parent_parts = module_name.split('.')[:-node.level]
                            if node.module:
                                full_module = '.'.join(parent_parts + [node.module])
                            else:
                                full_module = '.'.join(parent_parts)
                            imports.add(full_module)
                        else:
                            imports.add(node.module)
                    
                    # Also track what's being imported
                    for alias in node.names:
                        if alias.name != '*':
                            if node.module:
                                imports.add(f"{node.module}.{alias.name}")
                            
        except SyntaxError as e:
            self.issues.append({
                'type': 'syntax_error',
                'file': str(file_path),
                'error': str(e),
                'line': e.lineno
            })
        except Exception as e:
            self.issues.append({
                'type': 'parse_error',
                'file': str(file_path),
                'error': str(e)
            })
        
        return imports
    
    def check_import_exists(self, import_name: str, importing_file: Path) -> bool:
        """Check if an import can be resolved."""
        # Standard library and third-party packages
        standard_libs = {
            'os', 'sys', 'json', 'ast', 'datetime', 'time', 'random', 'math',
            'collections', 'itertools', 'functools', 'typing', 'pathlib', 're',
            'asyncio', 'threading', 'multiprocessing', 'queue', 'copy', 'logging',
            'unittest', 'pytest', 'setuptools', 'pip', 'enum', 'abc', 'dataclasses',
            'warnings', 'contextlib', 'traceback', 'inspect', 'importlib', 'types',
            'weakref', 'gc', 'atexit', 'signal', 'sqlite3', 'csv', 'io', 'string',
            'textwrap', 'difflib', 'hashlib', 'hmac', 'secrets', 'uuid', 'base64',
            'binascii', 'zlib', 'gzip', 'tarfile', 'zipfile', 'configparser', 'argparse',
            'getopt', 'tempfile', 'shutil', 'glob', 'fnmatch', 'pickle', 'shelve',
            'dbm', 'sqlite3', 'http', 'urllib', 'email', 'smtplib', 'ftplib',
            'telnetlib', 'socketserver', 'xmlrpc', 'ipaddress', 'socket', 'ssl',
            'select', 'selectors', 'subprocess', 'sched', 'statistics', 'decimal',
            'fractions', 'numbers', 'cmath', 'array', 'bisect', 'heapq', 'operator',
            'pdb', 'profile', 'timeit', 'trace', 'doctest', 'unittest.mock'
        }
        
        third_party = {
            'numpy', 'pandas', 'requests', 'flask', 'django', 'sqlalchemy',
            'pytest', 'setuptools', 'wheel', 'six', 'click', 'jinja2',
            'werkzeug', 'itsdangerous', 'markupsafe', 'certifi', 'chardet',
            'idna', 'urllib3', 'pytz', 'dateutil', 'yaml', 'toml', 'rich',
            'pydantic', 'fastapi', 'uvicorn', 'starlette', 'httpx', 'aiohttp',
            'beautifulsoup4', 'lxml', 'pillow', 'matplotlib', 'seaborn',
            'scikit-learn', 'scipy', 'sympy', 'networkx', 'pygame', 'tkinter',
            'pyqt5', 'pyside2', 'kivy', 'streamlit', 'gradio', 'transformers',
            'torch', 'tensorflow', 'keras', 'anthropic', 'openai', 'langchain',
            'chromadb', 'faiss', 'sentence_transformers', 'huggingface_hub',
            'tokenizers', 'datasets', 'accelerate', 'optimum', 'diffusers',
            'websockets', 'aiofiles', 'cryptography', 'pyjwt', 'passlib',
            'python-multipart', 'python-jose', 'emails', 'jinja2', 'aioredis',
            'motor', 'beanie', 'tortoise', 'peewee', 'mongoengine', 'redis',
            'celery', 'dramatiq', 'rq', 'apscheduler', 'schedule', 'arrow',
            'pendulum', 'humanize', 'tabulate', 'colorama', 'termcolor',
            'questionary', 'typer', 'fire', 'docopt', 'configargparse',
            'python-dotenv', 'environs', 'dynaconf', 'hydra', 'omegaconf',
            'bs4', 'newspaper3k', 'scrapy', 'selenium', 'playwright'
        }
        
        # Check if it's a standard library or known third-party module
        base_module = import_name.split('.')[0]
        if base_module in standard_libs or base_module in third_party:
            return True
        
        # Check if it's a project module
        if import_name.startswith('src.'):
            # Convert to relative path
            module_path = import_name.replace('.', '/')
            possible_paths = [
                self.project_root / f"{module_path}.py",
                self.project_root / module_path / "__init__.py"
            ]
            return any(p.exists() for p in possible_paths)
        
        # Check relative imports within src
        if not import_name.startswith('src.'):
            # Could be a relative import
            module_path = import_name.replace('.', '/')
            possible_paths = [
                self.src_root / f"{module_path}.py",
                self.src_root / module_path / "__init__.py"
            ]
            return any(p.exists() for p in possible_paths)
        
        return False
    
    def detect_circular_dependencies(self) -> None:
        """Detect circular import dependencies."""
        def find_cycles(module: str, path: List[str], visited: Set[str]) -> None:
            if module in path:
                # Found a cycle
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                # Normalize cycle to start with smallest module name
                min_idx = cycle.index(min(cycle))
                normalized_cycle = cycle[min_idx:] + cycle[:min_idx]
                self.circular_dependencies = [*self.circular_dependencies, normalized_cycle]
                return
            
            if module in visited:
                return
            
            visited.add(module)
            path.append(module)
            
            for imported in self.imports.get(module, set()):
                if imported in self.imports:  # Only check project modules
                    find_cycles(imported, path.copy(), visited.copy())
            
            path.pop()
        
        # Check each module for cycles
        for module in self.imports:
            find_cycles(module, [], set())
        
        # Remove duplicate cycles
        unique_cycles = []
        seen = set()
        for cycle in self.circular_dependencies:
            cycle_tuple = tuple(cycle)
            if cycle_tuple not in seen:
                seen.add(cycle_tuple)
                unique_cycles.append(cycle)
        
        self.circular_dependencies = unique_cycles
    
    def check_import_style(self, file_path: Path) -> List[Dict]:
        """Check for import style issues (absolute vs relative imports within package)."""
        issues = []
        module_name = self.get_module_name(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('src.'):
                        # This is an absolute import within the project
                        # Check if it should be a relative import
                        if 'empire_framework' in str(file_path) and 'empire_framework' in node.module:
                            issues.append({
                                'type': 'absolute_import_in_package',
                                'file': str(file_path),
                                'line': node.lineno,
                                'import': f"from {node.module} import ...",
                                'suggestion': 'Use relative import within package'
                            })
                    elif node.level == 0 and node.module:
                        # Check if this should be a relative import
                        importing_package = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
                        if importing_package and node.module.startswith(importing_package):
                            issues.append({
                                'type': 'should_be_relative',
                                'file': str(file_path),
                                'line': node.lineno,
                                'import': f"from {node.module} import ...",
                                'suggestion': f"Use relative import: from .{node.module[len(importing_package)+1:]} import ..."
                            })
        except:
            pass
        
        return issues
    
    def analyze(self) -> Dict[str, any]:
        """Run the complete import analysis."""
        print("Starting import audit...")
        
        # Find all Python files
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # Extract imports from each file
        for file_path in python_files:
            module_name = self.get_module_name(file_path)
            self.module_paths = {**self.module_paths, module_name: file_path}
            imports = self.extract_imports(file_path)
            
            # Filter to only project imports for dependency tracking
            project_imports = {imp for imp in imports if not self.is_external_import(imp)}
            self.imports = {**self.imports, module_name: project_imports}
            
            # Check for missing imports
            for imp in imports:
                if not self.check_import_exists(imp, file_path):
                    self.issues.append({
                        'type': 'missing_import',
                        'file': str(file_path),
                        'import': imp,
                        'module': module_name
                    })
            
            # Check import style
            style_issues = self.check_import_style(file_path)
            self.issues = [*self.issues, *style_issues]
        
        # Detect circular dependencies
        self.detect_circular_dependencies()
        
        # Create summary
        summary = {
            'total_files': len(python_files),
            'total_issues': len(self.issues),
            'circular_dependencies': len(self.circular_dependencies),
            'issues_by_type': defaultdict(int),
            'files_with_issues': set()
        }
        
        for issue in self.issues:
            summary['issues_by_type'][issue['type']] += 1
            summary['files_with_issues'].add(issue['file'])
        
        summary['files_with_issues'] = len(summary['files_with_issues'])
        summary['issues_by_type'] = dict(summary['issues_by_type'])
        
        return {
            'summary': summary,
            'issues': self.issues,
            'circular_dependencies': self.circular_dependencies
        }
    
    def is_external_import(self, import_name: str) -> bool:
        """Check if an import is external (not part of the project)."""
        base = import_name.split('.')[0]
        return not (base == 'src' or base in self.module_paths)
    
    def generate_report(self, output_file: str = "import_audit_report.json") -> None:
        """Generate a detailed report of the import audit."""
        results = self.analyze()
        
        # Save JSON report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("IMPORT AUDIT SUMMARY")
        print("="*60)
        print(f"Total Python files analyzed: {results['summary']['total_files']}")
        print(f"Total issues found: {results['summary']['total_issues']}")
        print(f"Files with issues: {results['summary']['files_with_issues']}")
        print(f"Circular dependencies: {results['summary']['circular_dependencies']}")
        
        print("\nIssues by type:")
        for issue_type, count in results['summary']['issues_by_type'].items():
            print(f"  {issue_type}: {count}")
        
        # Print circular dependencies
        if results['circular_dependencies']:
            print("\n" + "="*60)
            print("CIRCULAR DEPENDENCIES DETECTED:")
            print("="*60)
            for i, cycle in enumerate(results['circular_dependencies'], 1):
                print(f"{i}. {' -> '.join(cycle)}")
        
        # Print critical issues
        critical_issues = [i for i in results['issues'] if i['type'] in ['missing_import', 'syntax_error']]
        if critical_issues:
            print("\n" + "="*60)
            print("CRITICAL ISSUES (Missing imports and syntax errors):")
            print("="*60)
            for issue in critical_issues[:10]:  # Show first 10
                print(f"\nFile: {issue['file']}")
                print(f"Type: {issue['type']}")
                if 'import' in issue:
                    print(f"Import: {issue['import']}")
                if 'error' in issue:
                    print(f"Error: {issue['error']}")
                if 'line' in issue:
                    print(f"Line: {issue['line']}")
        
        print(f"\nFull report saved to: {output_file}")



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
    """Run the import audit."""
    auditor = ImportAuditor()
    auditor.generate_report()


if __name__ == "__main__":
    main()
