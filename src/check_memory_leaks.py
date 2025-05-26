#!/usr/bin/env python3
"""
Memory Leak Detection Tool

This tool scans Python code for potential memory leaks related to:
1. Event listeners/callbacks not being unregistered
2. Subscriptions (observer pattern) not being removed
3. Timers (threading.Timer, asyncio tasks) not being cancelled
4. Thread/Process resources not being properly cleaned up
5. WebSocket/network connections not being closed
6. File handles not being closed
7. Database connections not being properly managed
"""

import ast
import os
import sys
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path


class MemoryLeakDetector(ast.NodeVisitor):
    """AST visitor to detect potential memory leaks"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        
        # Track various patterns
        self.event_listeners = []  # addEventListener, subscribe, on_*, etc.
        self.timers = []  # Timer, setTimeout, setInterval equivalents
        self.async_tasks = []  # create_task, ensure_future
        self.threads = []  # Thread, Process
        self.connections = []  # WebSocket, database connections
        self.file_handles = []  # open() calls
        self.callbacks = []  # Functions passed as callbacks
        
        # Track cleanup methods
        self.has_cleanup = False
        self.has_close = False
        self.has_destructor = False
        self.cleanup_calls = []
        
        # Current context
        self.current_class = None
        self.current_function = None
        self.in_init = False
        self.in_cleanup = False
        self.in_context_manager = False
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        
        # Check for cleanup methods
        method_names = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
        if any(name in ['cleanup', 'close', 'shutdown', 'dispose', 'teardown'] for name in method_names):
            self.has_cleanup = True
        if '__del__' in method_names:
            self.has_destructor = True
        if '__exit__' in method_names or '__aexit__' in method_names:
            self.in_context_manager = True
            
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        
        # Check if this is __init__ or a cleanup method
        if node.name == '__init__':
            self.in_init = True
        elif node.name in ['cleanup', 'close', 'shutdown', 'dispose', 'teardown', '__del__', '__exit__', '__aexit__']:
            self.in_cleanup = True
            
        self.generic_visit(node)
        
        self.current_function = old_function
        self.in_init = False
        self.in_cleanup = False
        
    def visit_AsyncFunctionDef(self, node):
        # Treat async functions the same way
        self.visit_FunctionDef(node)
        
    def visit_Call(self, node):
        func_name = self._get_call_name(node)
        
        # Check for event listener registration
        if func_name and any(pattern in func_name for pattern in [
            'addEventListener', 'subscribe', 'on', 'register', 'attach',
            'bind', 'connect', 'add_listener', 'add_handler', 'add_callback'
        ]):
            if not self.in_cleanup:
                self.event_listeners.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'func': func_name,
                    'in_init': self.in_init,
                    'class': self.current_class,
                    'function': self.current_function
                })
                
        # Check for timer creation
        if func_name in ['Timer', 'threading.Timer', 'asyncio.create_task', 
                        'asyncio.ensure_future', 'asyncio.get_event_loop().call_later',
                        'asyncio.get_event_loop().call_at', 'schedule']:
            if not self.in_cleanup:
                self.timers.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'func': func_name,
                    'in_init': self.in_init,
                    'class': self.current_class,
                    'function': self.current_function
                })
                
        # Check for thread/process creation
        if func_name in ['Thread', 'threading.Thread', 'Process', 'multiprocessing.Process']:
            self.threads.append({
                'line': node.lineno,
                'col': node.col_offset,
                'func': func_name,
                'class': self.current_class,
                'function': self.current_function
            })
            
        # Check for WebSocket/connection creation
        if func_name and any(pattern in func_name for pattern in [
            'WebSocket', 'connect', 'create_connection', 'websockets.connect',
            'aiohttp.ClientSession', 'requests.Session', 'psycopg2.connect',
            'pymongo.MongoClient', 'redis.Redis'
        ]):
            self.connections.append({
                'line': node.lineno,
                'col': node.col_offset,
                'func': func_name,
                'class': self.current_class,
                'function': self.current_function
            })
            
        # Check for file operations
        if func_name == 'open':
            # Check if it's in a with statement
            parent = getattr(node, 'parent', None)
            if not isinstance(parent, ast.withitem):
                self.file_handles.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'class': self.current_class,
                    'function': self.current_function
                })
                
        # Check for cleanup calls
        if func_name and any(pattern in func_name for pattern in [
            'removeEventListener', 'unsubscribe', 'off', 'unregister', 'detach',
            'unbind', 'disconnect', 'remove_listener', 'remove_handler', 'cancel',
            'close', 'shutdown', 'stop', 'join', 'terminate', 'kill'
        ]):
            self.cleanup_calls.append({
                'line': node.lineno,
                'func': func_name,
                'in_cleanup': self.in_cleanup
            })
            
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        # Check for callback assignments
        if isinstance(node.value, ast.Name) or isinstance(node.value, ast.Attribute):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    attr_name = target.attr
                    if any(pattern in attr_name for pattern in ['callback', 'handler', 'listener']):
                        self.callbacks.append({
                            'line': node.lineno,
                            'attr': attr_name,
                            'class': self.current_class,
                            'function': self.current_function
                        })
                        
        self.generic_visit(node)
        
    def _get_call_name(self, node):
        """Extract the function name from a Call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None
        
    def analyze(self, tree):
        """Analyze the AST and generate issues"""
        # Add parent references for context checking
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
                
        self.visit(tree)
        
        # Generate issues based on findings
        issues = []
        
        # Check for event listeners without cleanup
        if self.event_listeners and not self.has_cleanup and not self.in_context_manager:
            for listener in self.event_listeners:
                issues.append({
                    'type': 'event_listener_leak',
                    'line': listener['line'],
                    'message': f"Event listener '{listener['func']}' registered but no cleanup method found",
                    'severity': 'high' if listener['in_init'] else 'medium',
                    'class': listener['class'],
                    'function': listener['function']
                })
                
        # Check for timers without cancellation
        for timer in self.timers:
            has_cancel = any(
                'cancel' in call['func'] or 'stop' in call['func'] 
                for call in self.cleanup_calls
            )
            if not has_cancel and not self.in_context_manager:
                issues.append({
                    'type': 'timer_leak',
                    'line': timer['line'],
                    'message': f"Timer '{timer['func']}' created but not cancelled in cleanup",
                    'severity': 'high',
                    'class': timer['class'],
                    'function': timer['function']
                })
                
        # Check for threads without join
        for thread in self.threads:
            has_join = any('join' in call['func'] for call in self.cleanup_calls)
            if not has_join:
                issues.append({
                    'type': 'thread_leak',
                    'line': thread['line'],
                    'message': f"Thread/Process created but not joined/terminated",
                    'severity': 'high',
                    'class': thread['class'],
                    'function': thread['function']
                })
                
        # Check for connections without close
        for conn in self.connections:
            has_close = any('close' in call['func'] or 'disconnect' in call['func'] 
                          for call in self.cleanup_calls)
            if not has_close and not self.in_context_manager:
                issues.append({
                    'type': 'connection_leak',
                    'line': conn['line'],
                    'message': f"Connection '{conn['func']}' opened but not closed",
                    'severity': 'high',
                    'class': conn['class'],
                    'function': conn['function']
                })
                
        # Check for file handles without close
        for handle in self.file_handles:
            issues.append({
                'type': 'file_handle_leak',
                'line': handle['line'],
                'message': "File opened without using 'with' statement or explicit close",
                'severity': 'medium',
                'class': handle['class'],
                'function': handle['function']
            })
            
        # Check for callbacks that might create circular references
        if self.callbacks and self.current_class:
            for callback in self.callbacks:
                if callback['class']:
                    issues.append({
                        'type': 'circular_reference_risk',
                        'line': callback['line'],
                        'message': f"Callback '{callback['attr']}' might create circular reference",
                        'severity': 'low',
                        'class': callback['class'],
                        'function': callback['function']
                    })
                    
        return issues


def check_file(filepath: Path) -> List[Dict]:
    """Check a single file for memory leaks"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(filepath))
        detector = MemoryLeakDetector(str(filepath))
        return detector.analyze(tree)
        
    except SyntaxError as e:
        return [{
            'type': 'syntax_error',
            'line': e.lineno or 0,
            'message': f"Syntax error: {e.msg}",
            'severity': 'error'
        }]
    except Exception as e:
        return [{
            'type': 'error',
            'line': 0,
            'message': f"Error analyzing file: {str(e)}",
            'severity': 'error'
        }]


def main():
    """Main function to scan the project"""
    src_dir = Path('.')
    if len(sys.argv) > 1:
        src_dir = Path(sys.argv[1])
        
    print("Memory Leak Detection Tool")
    print("=" * 80)
    print(f"Scanning directory: {src_dir.absolute()}")
    print()
    
    # Collect all Python files
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(Path(root) / file)
                
    print(f"Found {len(python_files)} Python files to analyze")
    print()
    
    # Analyze files
    all_issues = defaultdict(list)
    total_issues = 0
    
    for filepath in sorted(python_files):
        issues = check_file(filepath)
        if issues:
            all_issues[str(filepath)] = issues
            total_issues += len(issues)
            
    # Report results
    if not all_issues:
        print("✓ No memory leak issues detected!")
        return 0
        
    print(f"Found {total_issues} potential memory leak issues in {len(all_issues)} files:")
    print()
    
    # Group by issue type
    issue_types = defaultdict(int)
    high_severity = 0
    
    for filepath, issues in all_issues.items():
        for issue in issues:
            issue_types[issue['type']] += 1
            if issue['severity'] == 'high':
                high_severity += 1
                
    print("Issue Summary:")
    for issue_type, count in sorted(issue_types.items()):
        print(f"  - {issue_type}: {count}")
    print(f"\nHigh severity issues: {high_severity}")
    print()
    
    # Detailed report
    print("Detailed Report:")
    print("-" * 80)
    
    for filepath, issues in sorted(all_issues.items()):
        print(f"\n{filepath}:")
        for issue in sorted(issues, key=lambda x: x['line']):
            severity_marker = "!" if issue['severity'] == 'high' else "•"
            location = f"line {issue['line']}"
            if issue.get('class'):
                location += f", class {issue['class']}"
            if issue.get('function'):
                location += f", function {issue['function']}"
            print(f"  {severity_marker} {location}: {issue['message']}")
            
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("1. Add cleanup methods to classes that register listeners or create resources")
    print("2. Use context managers (with statements) for file operations")
    print("3. Cancel timers and join threads in cleanup methods")
    print("4. Implement __exit__ or __aexit__ for proper resource cleanup")
    print("5. Use weak references for callbacks to avoid circular references")
    
    return 1 if high_severity > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
