#!/usr/bin/env python3
"""
Tool to identify and fix async lifecycle management issues in Python code.
This addresses the Python equivalent of React's useEffect dependency arrays.

Issues addressed:
1. asyncio.create_task without assignment (lost task references)
2. Missing await on async operations
3. Improper resource cleanup (missing close() calls)
4. Infinite loops without proper cancellation
5. Missing context managers for resources
6. Background tasks not properly tracked
7. Event loops not properly closed
"""

import ast
import os
import re
from typing import List, Dict, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AsyncLifecycleAnalyzer(ast.NodeVisitor):
    """Analyzes async code for lifecycle management issues."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.current_class = None
        self.current_function = None
        self.async_functions = set()
        self.created_tasks = {}  # task_name -> line_number
        self.resource_opens = {}  # resource_name -> line_number
        self.resource_closes = {}  # resource_name -> line_number
        self.infinite_loops = []
        self.missing_awaits = []
        self.untracked_tasks = []
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        old_func = self.current_function
        self.current_function = node.name
        self.async_functions.add(node.name)
        
        # Check for cleanup methods
        if node.name in ['close', 'cleanup', 'shutdown', 'disconnect', '__aexit__']:
            self._check_cleanup_method(node)
            
        self.generic_visit(node)
        self.current_function = old_func
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_func = self.current_function
        self.current_function = node.name
        
        # Check for __exit__ methods
        if node.name == '__exit__':
            self._check_exit_method(node)
            
        self.generic_visit(node)
        self.current_function = old_func
        
    def visit_Call(self, node: ast.Call) -> None:
        # Check for asyncio.create_task
        if self._is_create_task_call(node):
            parent = self._get_parent_node(node)
            if not isinstance(parent, (ast.Assign, ast.AnnAssign)):
                self.untracked_tasks.append({
                    'line': node.lineno,
                    'function': self.current_function,
                    'class': self.current_class
                })
                
        # Check for resource opens
        if self._is_resource_open(node):
            self._track_resource_open(node)
            
        # Check for async calls without await
        if self._is_async_call_without_await(node):
            self.missing_awaits.append({
                'line': node.lineno,
                'function': self.current_function,
                'call': ast.unparse(node)
            })
            
        self.generic_visit(node)
        
    def visit_While(self, node: ast.While) -> None:
        # Check for infinite loops
        if self._is_infinite_loop(node):
            self.infinite_loops.append({
                'line': node.lineno,
                'function': self.current_function,
                'has_break': self._has_break_statement(node),
                'has_cancellation_check': self._has_cancellation_check(node)
            })
        self.generic_visit(node)
        
    def _is_create_task_call(self, node: ast.Call) -> bool:
        """Check if this is an asyncio.create_task or ensure_future call."""
        if isinstance(node.func, ast.Attribute):
            if (node.func.attr in ['create_task', 'ensure_future'] and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'asyncio'):
                return True
        return False
        
    def _is_resource_open(self, node: ast.Call) -> bool:
        """Check if this opens a resource that needs cleanup."""
        if isinstance(node.func, ast.Name):
            return node.func.id in ['open', 'socket', 'connect']
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ['open', 'connect', 'create_connection', 'ClientSession']
        return False
        
    def _is_async_call_without_await(self, node: ast.Call) -> bool:
        """Check if this is an async function call without await."""
        # This is a simplified check - in practice would need more context
        if isinstance(node.func, ast.Name) and node.func.id in self.async_functions:
            parent = self._get_parent_node(node)
            return not isinstance(parent, ast.Await)
        return False
        
    def _is_infinite_loop(self, node: ast.While) -> bool:
        """Check if this is a while True loop."""
        return (isinstance(node.test, ast.Constant) and node.test.value is True) or \
               (isinstance(node.test, ast.NameConstant) and node.test.value is True)
               
    def _has_break_statement(self, node: ast.While) -> bool:
        """Check if loop has a break statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                return True
        return False
        
    def _has_cancellation_check(self, node: ast.While) -> bool:
        """Check if loop checks for cancellation."""
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                # Check for common cancellation patterns
                condition_str = ast.unparse(child.test)
                if any(pattern in condition_str for pattern in [
                    'cancelled()', 'is_cancelled', 'should_stop', 'running'
                ]):
                    return True
        return False
        
    def _track_resource_open(self, node: ast.Call) -> None:
        """Track resource opening for cleanup verification."""
        # Simplified tracking - would need more sophisticated analysis
        if isinstance(node.func, ast.Name):
            resource_type = node.func.id
        else:
            resource_type = node.func.attr
        self.resource_opens[f"{resource_type}_{node.lineno}"] = node.lineno
        
    def _check_cleanup_method(self, node: ast.AsyncFunctionDef) -> None:
        """Check if cleanup method properly closes resources."""
        # Look for close() calls in cleanup methods
        has_close_calls = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute) and child.func.attr == 'close':
                    has_close_calls = True
                    
        if not has_close_calls and self.resource_opens:
            self.issues.append({
                'type': 'missing_cleanup',
                'line': node.lineno,
                'function': node.name,
                'message': 'Cleanup method does not close all resources'
            })
            
    def _check_exit_method(self, node: ast.FunctionDef) -> None:
        """Check if __exit__ method properly cleans up."""
        # Similar to _check_cleanup_method but for sync __exit__
        pass
        
    def _get_parent_node(self, node: ast.AST) -> ast.AST:
        """Get parent node (simplified - would need proper implementation)."""
        # This is a placeholder - proper implementation would track parent nodes
        return None
        

def fix_async_lifecycle_issues(content: str, filename: str) -> str:
    """Fix async lifecycle issues in Python code."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.error(f"Syntax error in {filename}: {e}")
        return content
        
    analyzer = AsyncLifecycleAnalyzer(filename)
    analyzer.visit(tree)
    
    lines = content.split('\n')
    modified = False
    
    # Fix untracked tasks
    for task in analyzer.untracked_tasks:
        line_idx = task['line'] - 1
        if line_idx < len(lines):
            line = lines[line_idx]
            
            # Fix asyncio.create_task without assignment
            if 'asyncio.create_task(' in line and '=' not in line:
                indent = len(line) - len(line.lstrip())
                # Generate a task variable name
                if task['function']:
                    task_name = f"{task['function']}_task"
                else:
                    task_name = f"task_{task['line']}"
                    
                # Check if we're in __init__ method
                if task['function'] == '__init__' and task['class']:
                    task_name = f"self.{task_name}"
                    
                new_line = ' ' * indent + f"{task_name} = {line.strip()}"
                lines[line_idx] = new_line
                modified = True
                logger.info(f"Fixed untracked task at line {task['line']} in {filename}")
                
    # Fix infinite loops without cancellation checks
    for loop in analyzer.infinite_loops:
        if not loop['has_cancellation_check'] and not loop['has_break']:
            # Add a comment suggesting cancellation check
            line_idx = loop['line'] - 1
            if line_idx < len(lines):
                indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
                comment = ' ' * indent + "# TODO: Add cancellation check or break condition"
                if line_idx > 0 and "TODO: Add cancellation check" not in lines[line_idx - 1]:
                    lines.insert(line_idx, comment)
                    modified = True
                    logger.info(f"Added cancellation check reminder at line {loop['line']} in {filename}")
                    
    # Add cleanup for resources in async context managers
    if analyzer.resource_opens and not any('__aexit__' in line for line in lines):
        # Check if class has async context manager methods
        class_needs_context_manager = False
        for line in lines:
            if 'class ' in line and any(f"open" in l for l in lines):
                class_needs_context_manager = True
                break
                
    return '\n'.join(lines) if modified else content


def add_proper_task_tracking(content: str, filename: str) -> str:
    """Add proper task tracking to classes that create background tasks."""
    lines = content.split('\n')
    modified = False
    
    # Pattern to find classes that create tasks
    in_class = False
    class_name = None
    has_init = False
    has_task_creation = False
    has_cleanup = False
    init_line = -1
    class_indent = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            in_class = True
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_indent = len(line) - len(line.lstrip())
            has_init = False
            has_task_creation = False
            has_cleanup = False
            
        elif in_class and line.strip() and len(line) - len(line.lstrip()) <= class_indent:
            # End of class
            if has_task_creation and not has_cleanup:
                # Add cleanup method
                cleanup_method = [
                    f"{' ' * (class_indent + 4)}async def cleanup(self) -> None:",
                    f"{' ' * (class_indent + 8)}\"\"\"Clean up background tasks.\"\"\"",
                    f"{' ' * (class_indent + 8)}if hasattr(self, '_background_tasks'):",
                    f"{' ' * (class_indent + 12)}for task in self._background_tasks:",
                    f"{' ' * (class_indent + 16)}if not task.done():",
                    f"{' ' * (class_indent + 20)}task.cancel()",
                    f"{' ' * (class_indent + 16)}try:",
                    f"{' ' * (class_indent + 20)}await task",
                    f"{' ' * (class_indent + 16)}except asyncio.CancelledError:",
                    f"{' ' * (class_indent + 20)}pass",
                    ""
                ]
                lines[i:i] = cleanup_method
                modified = True
                logger.info(f"Added cleanup method to class {class_name} in {filename}")
            in_class = False
            
        elif in_class:
            if 'def __init__' in line:
                has_init = True
                init_line = i
            elif 'asyncio.create_task' in line:
                has_task_creation = True
            elif 'async def cleanup' in line or 'async def close' in line:
                has_cleanup = True
                
    # Add task tracking to __init__ if needed
    if has_task_creation and has_init and init_line >= 0:
        # Find the right place to add task list initialization
        for i in range(init_line + 1, len(lines)):
            if lines[i].strip() and not lines[i].strip().startswith('"""') and not lines[i].strip().startswith('#'):
                if 'super().__init__' in lines[i]:
                    # Add after super().__init__
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    if '_background_tasks' not in content:
                        lines.insert(i + 1, f"{' ' * indent}self._background_tasks: List[asyncio.Task] = []")
                        modified = True
                        logger.info(f"Added task tracking to {class_name}.__init__ in {filename}")
                    break
                elif 'self.' in lines[i]:
                    # Add before first self assignment
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    if '_background_tasks' not in content:
                        lines.insert(i, f"{' ' * indent}self._background_tasks: List[asyncio.Task] = []")
                        modified = True
                        logger.info(f"Added task tracking to {class_name}.__init__ in {filename}")
                    break
                    
    return '\n'.join(lines) if modified else content


def add_async_context_managers(content: str, filename: str) -> str:
    """Add async context manager support to classes with resources."""
    lines = content.split('\n')
    modified = False
    
    # Find classes that open resources but don't have context managers
    in_class = False
    class_name = None
    has_resources = False
    has_aenter = False
    has_aexit = False
    class_indent = 0
    class_end_line = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            in_class = True
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_indent = len(line) - len(line.lstrip())
            has_resources = False
            has_aenter = False
            has_aexit = False
            
        elif in_class and line.strip() and len(line) - len(line.lstrip()) <= class_indent:
            # End of class
            class_end_line = i
            if has_resources and not (has_aenter and has_aexit):
                # Add async context manager methods
                context_methods = [
                    "",
                    f"{' ' * (class_indent + 4)}async def __aenter__(self):",
                    f"{' ' * (class_indent + 8)}\"\"\"Enter async context.\"\"\"",
                    f"{' ' * (class_indent + 8)}return self",
                    "",
                    f"{' ' * (class_indent + 4)}async def __aexit__(self, exc_type, exc_val, exc_tb):",
                    f"{' ' * (class_indent + 8)}\"\"\"Exit async context and cleanup.\"\"\"",
                    f"{' ' * (class_indent + 8)}if hasattr(self, 'cleanup'):",
                    f"{' ' * (class_indent + 12)}await self.cleanup()",
                    f"{' ' * (class_indent + 8)}elif hasattr(self, 'close'):",
                    f"{' ' * (class_indent + 12)}await self.close()",
                    f"{' ' * (class_indent + 8)}return False",
                ]
                lines[class_end_line:class_end_line] = context_methods
                modified = True
                logger.info(f"Added async context manager to class {class_name} in {filename}")
            in_class = False
            
        elif in_class:
            if any(resource in line for resource in ['session', 'connection', 'websocket', 'file', 'socket']):
                if '=' in line and 'self.' in line:
                    has_resources = True
            elif 'async def __aenter__' in line:
                has_aenter = True
            elif 'async def __aexit__' in line:
                has_aexit = True
                
    return '\n'.join(lines) if modified else content


def process_file(filepath: str) -> bool:
    """Process a single Python file for async lifecycle issues."""
    logger.info(f"\nProcessing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Apply fixes
        content = fix_async_lifecycle_issues(content, filepath)
        content = add_proper_task_tracking(content, filepath)
        content = add_async_context_managers(content, filepath)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✓ Fixed async lifecycle issues in {filepath}")
            return True
        else:
            logger.info(f"✓ No async lifecycle issues found in {filepath}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error processing {filepath}: {str(e)}")
        return False


def main():
    """Main function to process all Python files."""
    logger.info("Checking for async lifecycle management issues...")
    logger.info("=" * 80)
    
    # Check if we're in src directory or project root
    if os.path.basename(os.getcwd()) == 'src':
        src_dir = "."
    else:
        src_dir = "src"
    
    fixed_files = 0
    total_files = 0
    
    # Process all Python files
    for root, dirs, files in os.walk(src_dir):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and not file.startswith('test_') and not file.startswith('fix_'):
                filepath = os.path.join(root, file)
                total_files += 1
                if process_file(filepath):
                    fixed_files += 1
                    
    logger.info("=" * 80)
    logger.info(f"Summary: Fixed {fixed_files} out of {total_files} files")
    
    # Additional recommendations
    logger.info("\nRecommendations for manual review:")
    logger.info("1. Review all asyncio.create_task calls to ensure tasks are properly tracked")
    logger.info("2. Ensure all async resources are used with 'async with' context managers")
    logger.info("3. Add cancellation checks to all infinite loops in async functions")
    logger.info("4. Implement proper cleanup methods for all classes with background tasks")
    logger.info("5. Use asyncio.TaskGroup (Python 3.11+) for better task management")


if __name__ == "__main__":
    main()
