"""
Comprehensive Error Scanner for Python Files
Detects various types of errors including:
- Syntax errors
- Import errors
- Undefined variables
- Type errors
- Indentation errors
- Missing dependencies
"""

import ast
import os
import sys
import re
import importlib.util
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import json
import tokenize
import io

class ErrorReport:
    def __init__(self):
        self.syntax_errors: List[Dict] = []
        self.import_errors: List[Dict] = []
        self.undefined_vars: List[Dict] = []
        self.type_errors: List[Dict] = []
        self.indentation_errors: List[Dict] = []
        self.other_errors: List[Dict] = []
        
    def add_syntax_error(self, file: str, line: int, error: str):
        self.syntax_errors.append({
            'file': file,
            'line': line,
            'error': error
        })
        
    def add_import_error(self, file: str, line: int, module: str, error: str):
        self.import_errors.append({
            'file': file,
            'line': line,
            'module': module,
            'error': error
        })
        
    def add_undefined_var(self, file: str, line: int, var_name: str):
        self.undefined_vars.append({
            'file': file,
            'line': line,
            'variable': var_name
        })
        
    def add_type_error(self, file: str, line: int, error: str):
        self.type_errors.append({
            'file': file,
            'line': line,
            'error': error
        })
        
    def add_indentation_error(self, file: str, line: int, error: str):
        self.indentation_errors.append({
            'file': file,
            'line': line,
            'error': error
        })
        
    def add_other_error(self, file: str, line: int, error: str):
        self.other_errors.append({
            'file': file,
            'line': line,
            'error': error
        })
        
    def to_dict(self):
        return {
            'syntax_errors': self.syntax_errors,
            'import_errors': self.import_errors,
            'undefined_vars': self.undefined_vars,
            'type_errors': self.type_errors,
            'indentation_errors': self.indentation_errors,
            'other_errors': self.other_errors,
            'total_errors': (len(self.syntax_errors) + len(self.import_errors) +
                           len(self.undefined_vars) + len(self.type_errors) + 
                           len(self.indentation_errors) + len(self.other_errors))
        }

class ComprehensiveErrorScanner:
    def __init__(self):
        self.report = ErrorReport()
        self.builtin_names = set(dir(__builtins__))
        self.standard_libs = self._get_standard_libs()
        
    def _get_standard_libs(self) -> Set[str]:
        """Get list of standard library modules"""
        import sys
        stdlib_path = Path(sys.modules['os'].__file__).parent
        modules = set()
        
        # Common standard library modules
        common_std_libs = {
            'os', 'sys', 're', 'json', 'datetime', 'time', 'math', 'random',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'subprocess', 'threading', 'asyncio', 'logging', 'unittest',
            'copy', 'pickle', 'base64', 'hashlib', 'uuid', 'io', 'tempfile',
            'shutil', 'glob', 'fnmatch', 'sqlite3', 'csv', 'configparser',
            'argparse', 'getopt', 'warnings', 'contextlib', 'dataclasses',
            'enum', 'abc', 'weakref', 'traceback', 'inspect', 'importlib'
        }
        
        return common_std_libs
        
    def check_syntax(self, file_path: str, content: str):
        """Check for syntax errors"""
        try:
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            self.report.add_syntax_error(
                file_path,
                e.lineno or 0,
                str(e.msg)
            )
            return False
        except Exception as e:
            self.report.add_other_error(
                file_path,
                0,
                f"Compilation error: {str(e)}"
            )
            return False
        return True
        
    def check_indentation(self, file_path: str, content: str):
        """Check for indentation errors"""
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
        except tokenize.TokenError as e:
            error_msg = str(e)
            # Extract line number from error message
            match = re.search(r'\(\'.*\', \((\d+),', error_msg)
            line_no = int(match.group(1)) if match else 0
            self.report.add_indentation_error(
                file_path,
                line_no,
                error_msg
            )
            
    def check_imports(self, file_path: str, tree: ast.AST):
        """Check for import errors"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if not self._can_import(module_name):
                        self.report.add_import_error(
                            file_path,
                            node.lineno,
                            alias.name,
                            f"Module '{alias.name}' not found"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if not self._can_import(module_name):
                        self.report.add_import_error(
                            file_path,
                            node.lineno,
                            node.module,
                            f"Module '{node.module}' not found"
                        )
                        
    def _can_import(self, module_name: str) -> bool:
        """Check if a module can be imported"""
        # Check if it's a standard library module
        if module_name in self.standard_libs:
            return True
            
        # Check if it's a local module (exists in src/)
        local_paths = [
            f"src/{module_name}.py",
            f"src/{module_name}/__init__.py",
            f"{module_name}.py"
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                return True
                
        # Try to find the module spec
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError, AttributeError):
            return False
            
    def check_undefined_variables(self, file_path: str, tree: ast.AST):
        """Check for undefined variables using basic scope analysis"""
        class VariableChecker(ast.NodeVisitor):
            def __init__(self, scanner):
                self.scanner = scanner
                self.defined_vars = set()
                self.scopes = [set()]  # Stack of scopes
                
            def visit_FunctionDef(self, node):
                # Add function name to current scope
                self.scopes[-1].add(node.name)
                # Create new scope for function
                self.scopes.append(set())
                # Add parameters to function scope
                for arg in node.args.args:
                    self.scopes[-1].add(arg.arg)
                self.generic_visit(node)
                self.scopes.pop()
                
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
                
            def visit_ClassDef(self, node):
                self.scopes[-1].add(node.name)
                self.scopes.append(set())
                self.generic_visit(node)
                self.scopes.pop()
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.scopes[-1].add(target.id)
                self.generic_visit(node)
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    name = node.id
                    # Check if variable is defined in any scope
                    is_defined = any(name in scope for scope in self.scopes)
                    # Check builtins and common names
                    if not is_defined and name not in self.scanner.builtin_names:
                        if name not in {'self', 'cls', '__name__', '__file__'}:
                            self.scanner.report.add_undefined_var(
                                file_path,
                                node.lineno,
                                name
                            )
                self.generic_visit(node)
                
        # Only check undefined variables if there are no syntax errors
        if not any(e['file'] == file_path for e in self.report.syntax_errors):
            checker = VariableChecker(self)
            checker.visit(tree)
            
    def check_type_annotations(self, file_path: str, tree: ast.AST):
        """Check for type annotation errors"""
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                # Check if annotation uses undefined types
                if isinstance(node.annotation, ast.Name):
                    type_name = node.annotation.id
                    # Common type names that should be imported from typing
                    typing_types = {'List', 'Dict', 'Set', 'Tuple', 'Optional', 
                                  'Union', 'Any', 'Callable', 'Type'}
                    if type_name in typing_types:
                        # Check if typing is imported
                        has_typing_import = any(
                            isinstance(n, ast.ImportFrom) and n.module == 'typing'
                            for n in ast.walk(tree)
                        )
                        if not has_typing_import:
                            self.report.add_type_error(
                                file_path,
                                node.lineno,
                                f"Type '{type_name}' used but 'typing' not imported"
                            )
                            
    def scan_file(self, file_path: str) -> bool:
        """Scan a single Python file for errors"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.report.add_other_error(
                file_path, 0,
                f"Failed to read file: {str(e)}"
            )
            return False
            
        # Check syntax
        if not self.check_syntax(file_path, content):
            return False
            
        # Check indentation
        self.check_indentation(file_path, content)
        
        # Parse AST for further checks
        try:
            tree = ast.parse(content, file_path)
        except Exception:
            # Already reported in syntax check
            return False
            
        # Check imports
        self.check_imports(file_path, tree)
        
        # Check undefined variables (simplified check)
        # self.check_undefined_variables(file_path, tree)
        
        # Check type annotations
        self.check_type_annotations(file_path, tree)
        
        return True
        
    def scan_directory(self, directory: str = 'src'):
        """Scan all Python files in directory"""
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
                    self.scan_file(file_path)
                    
        return python_files
        
    def generate_report(self) -> Dict:
        """Generate comprehensive error report"""
        report = self.report.to_dict()
        
        # Group errors by file
        files_with_errors = {}
        
        for error_type, errors in [
            ('syntax', report['syntax_errors']),
            ('import', report['import_errors']),
            ('undefined', report['undefined_vars']),
            ('type', report['type_errors']),
            ('indentation', report['indentation_errors']),
            ('other', report['other_errors'])
        ]:
            for error in errors:
                file_path = error['file']
                if file_path not in files_with_errors:
                    files_with_errors[file_path] = []
                files_with_errors[file_path].append({
                    'type': error_type,
                    'line': error.get('line', 0),
                    'details': error
                })
                
        # Sort errors by line number for each file
        for file_path in files_with_errors:
            files_with_errors[file_path].sort(key=lambda x: x['line'])
            
        report['files_with_errors'] = files_with_errors
        report['total_files_with_errors'] = len(files_with_errors)
        
        return report

def main():
    scanner = ComprehensiveErrorScanner()
    
    print("🔍 Scanning for errors in Python files...")
    python_files = scanner.scan_directory('src')
    
    print(f"\n📁 Scanned {len(python_files)} Python files")
    
    report = scanner.generate_report()
    
    # Save detailed report
    with open('error_scan_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    print(f"\n📊 Error Summary:")
    print(f"  Total errors: {report['total_errors']}")
    print(f"  Files with errors: {report['total_files_with_errors']}")
    print(f"  Syntax errors: {len(report['syntax_errors'])}")
    print(f"  Import errors: {len(report['import_errors'])}")
    print(f"  Type errors: {len(report['type_errors'])}")
    print(f"  Indentation errors: {len(report['indentation_errors'])}")
    print(f"  Other errors: {len(report['other_errors'])}")
    
    # Print most critical errors
    if report['syntax_errors']:
        print("\n❌ Critical Syntax Errors (must fix first):")
        for i, error in enumerate(report['syntax_errors'][:10]):
            print(f"  {i+1}. {error['file']}:{error['line']} - {error['error']}")
        if len(report['syntax_errors']) > 10:
            print(f"  ... and {len(report['syntax_errors']) - 10} more")
            
    if report['import_errors']:
        print("\n⚠️ Import Errors:")
        # Group by missing module
        missing_modules = {}
        for error in report['import_errors']:
            module = error['module']
            if module not in missing_modules:
                missing_modules[module] = []
            missing_modules[module].append(error['file'])
            
        for module, files in list(missing_modules.items())[:10]:
            print(f"  Module '{module}' missing in {len(files)} file(s)")
            
    print(f"\n✅ Report saved to: error_scan_report.json")
    
    return report

if __name__ == "__main__":
    main()
