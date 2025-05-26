"""
Check for missing await keywords and potential race conditions in async code.
"""

import ast
import os
from typing import List, Dict, Set, Tuple
from pathlib import Path


class AsyncIssueChecker(ast.NodeVisitor):
    """AST visitor to check for async/await issues."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.async_functions = set()
        self.current_function = None
        self.in_async_function = False
        self.created_tasks = {}  # Track created tasks that might not be awaited
        self.current_scope_tasks = set()
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track async function definitions."""
        self.async_functions.add(node.name)
        old_function = self.current_function
        old_in_async = self.in_async_function
        old_tasks = self.current_scope_tasks
        
        self.current_function = node.name
        self.in_async_function = True
        self.current_scope_tasks = set()
        
        self.generic_visit(node)
        
        # Check if any tasks were created but not awaited in this scope
        if self.current_scope_tasks:
            self.issues.append({
                'type': 'unjoined_tasks',
                'line': node.lineno,
                'function': node.name,
                'tasks': list(self.current_scope_tasks),
                'message': f"Tasks created with create_task() but not explicitly awaited/joined: {self.current_scope_tasks}"
            })
        
        self.current_function = old_function
        self.in_async_function = old_in_async
        self.current_scope_tasks = old_tasks
        
    def visit_Call(self, node: ast.Call):
        """Check for async function calls and task creation."""
        # Check for asyncio.create_task without assignment or await
        if self._is_create_task_call(node):
            # Check if the result is assigned
            parent = self._get_parent_node(node)
            if not isinstance(parent, (ast.Assign, ast.AnnAssign)):
                self.issues.append({
                    'type': 'unassigned_task',
                    'line': node.lineno,
                    'function': self.current_function,
                    'message': "asyncio.create_task() called without assigning result - task reference lost"
                })
            else:
                # Track the task for later checking
                if isinstance(parent, ast.Assign):
                    for target in parent.targets:
                        if isinstance(target, ast.Name):
                            self.current_scope_tasks.add(target.id)
                            
        # Check for asyncio.gather without await
        if self._is_gather_call(node) and self.in_async_function:
            parent = self._get_parent_node(node)
            if not self._is_awaited(parent):
                self.issues.append({
                    'type': 'missing_await',
                    'line': node.lineno,
                    'function': self.current_function,
                    'message': "asyncio.gather() called without await"
                })
                
        # Check for potential async function calls without await
        if self.in_async_function:
            func_name = self._get_function_name(node)
            if func_name and self._looks_like_async_function(func_name):
                parent = self._get_parent_node(node)
                if not self._is_awaited(parent) and not self._is_create_task_call(node):
                    self.issues.append({
                        'type': 'potential_missing_await',
                        'line': node.lineno,
                        'function': self.current_function,
                        'called_function': func_name,
                        'message': f"Possible async function '{func_name}' called without await"
                    })
                    
        self.generic_visit(node)
        
    def visit_Await(self, node: ast.Await):
        """Track awaited expressions."""
        # Check if we're awaiting a task variable
        if isinstance(node.value, ast.Name) and node.value.id in self.current_scope_tasks:
            self.current_scope_tasks.discard(node.value.id)
        self.generic_visit(node)
        
    def _is_create_task_call(self, node: ast.Call) -> bool:
        """Check if this is an asyncio.create_task call."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'create_task':
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'asyncio':
                    return True
        return False
        
    def _is_gather_call(self, node: ast.Call) -> bool:
        """Check if this is an asyncio.gather call."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'gather':
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'asyncio':
                    return True
        return False
        
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract the function name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
        
    def _looks_like_async_function(self, name: str) -> bool:
        """Heuristic to detect async function names."""
        # Common async function patterns
        async_patterns = [
            'send_request', 'complete', 'chat_complete', 'execute_operation',
            'evaluate_action', 'evaluate_principle', 'process_request',
            'send_message', 'receive_message', 'authenticate', 'connect',
            'disconnect', 'process', 'handle', 'fetch', 'get', 'post',
            'update', 'delete', 'create', 'save', 'load'
        ]
        
        name_lower = name.lower()
        
        # Check for explicit async naming
        if 'async' in name_lower or name_lower.endswith('_async'):
            return True
            
        # Check for common async operations
        for pattern in async_patterns:
            if pattern in name_lower:
                return True
                
        # Check if it's in our known async functions
        return name in self.async_functions
        
    def _is_awaited(self, node) -> bool:
        """Check if a node is awaited."""
        # This is simplified - in reality we'd need to walk up the AST
        return hasattr(node, 'value') and isinstance(node, ast.Expr) and isinstance(node.value, ast.Await)
        
    def _get_parent_node(self, node) -> ast.AST:
        """Get the parent node (simplified version)."""
        # In a real implementation, we'd track parent nodes during traversal
        return None


class RaceConditionChecker:
    """Check for potential race conditions in concurrent code."""
    
    def __init__(self):
        self.issues = []
        
    def check_file(self, filepath: str) -> List[Dict]:
        """Check a file for race conditions."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for common race condition patterns
            issues = []
            
            # Pattern 1: Multiple asyncio.gather without proper synchronization
            if 'asyncio.gather' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'asyncio.gather' in line and 'await' not in line:
                        issues.append({
                            'type': 'race_condition',
                            'line': i + 1,
                            'pattern': 'gather_without_await',
                            'message': 'asyncio.gather() without await can cause race conditions'
                        })
                        
            # Pattern 2: Shared state modification in concurrent tasks
            if 'create_task' in content or 'gather' in content:
                # Look for assignments to instance variables in async contexts
                for i, line in enumerate(lines):
                    if 'self.' in line and '=' in line and ('async def' in content[:content.find(line)]):
                        # Crude check for assignment in async context
                        if any(pattern in content for pattern in ['create_task', 'gather', 'ensure_future']):
                            issues.append({
                                'type': 'potential_race_condition',
                                'line': i + 1,
                                'pattern': 'shared_state_modification',
                                'message': 'Modifying shared state in concurrent async code - potential race condition'
                            })
                            
            return issues
            
        except Exception as e:
            return [{'type': 'error', 'message': str(e)}]


def check_directory(directory: str) -> Dict[str, List[Dict]]:
    """Check all Python files in a directory for async issues."""
    issues_by_file = {}
    race_checker = RaceConditionChecker()
    
    for root, dirs, files in os.walk(directory):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, directory)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Parse AST
                    tree = ast.parse(content)
                    
                    # Check for async/await issues
                    checker = AsyncIssueChecker(relative_path)
                    checker.visit(tree)
                    
                    # Check for race conditions
                    race_issues = race_checker.check_file(filepath)
                    
                    all_issues = checker.issues + race_issues
                    
                    if all_issues:
                        issues_by_file[relative_path] = all_issues
                        
                except Exception as e:
                    issues_by_file[relative_path] = [{
                        'type': 'parse_error',
                        'message': str(e)
                    }]
                    
    return issues_by_file


def main():
    """Main function to check for async issues."""
    src_dir = 'src'
    
    print("Checking for missing await keywords and race conditions...")
    print("=" * 80)
    
    issues = check_directory(src_dir)
    
    if not issues:
        print("✅ No async/await issues or race conditions found!")
        return
        
    # Report issues
    total_issues = 0
    
    for filepath, file_issues in sorted(issues.items()):
        print(f"\n📄 {filepath}")
        print("-" * 40)
        
        for issue in file_issues:
            total_issues += 1
            issue_type = issue.get('type', 'unknown')
            line = issue.get('line', '?')
            message = issue.get('message', 'Unknown issue')
            
            icon = "⚠️" if 'potential' in issue_type else "❌"
            print(f"{icon} Line {line}: {message}")
            
            if 'function' in issue:
                print(f"   Function: {issue['function']}")
            if 'called_function' in issue:
                print(f"   Called: {issue['called_function']}")
                
    print(f"\n\n📊 Summary: Found {total_issues} issues in {len(issues)} files")
    
    # Group by issue type
    issue_types = {}
    for file_issues in issues.values():
        for issue in file_issues:
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
    print("\nIssue breakdown:")
    for issue_type, count in sorted(issue_types.items()):
        print(f"  - {issue_type}: {count}")


if __name__ == "__main__":
    main()
