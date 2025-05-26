"""
Async Best Practices Example

This module demonstrates best practices for writing async code in Python,
including proper task management, lock usage, and error handling.
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager


class AsyncBestPracticesDemo:
    """Demonstrates async best practices with proper patterns"""
    
    def __init__(self):
        # Shared state with proper lock
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        # Track tasks for proper cleanup
        self._background_tasks: List[asyncio.Task] = []
        
        # Event for graceful shutdown
        self._shutdown_event = asyncio.Event()
        
    async def safe_shared_state_update(self, key: str, value: Any) -> None:
        """Example of safe shared state modification with lock"""
        async with self._lock:
            # All shared state modifications happen within the lock
            old_value = self._data.get(key)
            self._data = {**self._data, key: value}
            print(f"Updated {key}: {old_value} -> {value}")
            
    async def concurrent_reads_single_write(self) -> None:
        """Demonstrates read-write lock pattern"""
        # For read-heavy workloads, consider using asyncio.Semaphore
        # or a custom ReadWriteLock implementation
        
        # Multiple reads can happen concurrently
        async def read_data(key: str) -> Any:
            # Reads don't need lock if data is immutable
            return self._data.get(key)
            
        # But writes need exclusive access
        async def write_data(key: str, value: Any) -> None:
            async with self._lock:
                self._data = {**self._data, key: value}
                
        # Example usage
        await write_data("counter", 0)
        
        # Concurrent reads
        tasks = [read_data("counter") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        print(f"Concurrent read results: {results}")
        
    async def proper_task_management(self) -> None:
        """Shows proper asyncio.create_task usage"""
        # Always assign tasks to variables or track them
        
        # Pattern 1: Track in a list for cleanup
        task1 = asyncio.create_task(self._background_worker("worker1"))
        self._background_tasks = [*self._background_tasks, task1]
        
        # Pattern 2: Use local variable with timeout
        task2 = asyncio.create_task(self._timed_worker("worker2", 2))
        
        # Pattern 3: Fire and forget with proper error handling
        task3 = asyncio.create_task(
            self._fire_and_forget_with_error_handling()
        )
        
        # Wait for specific task with timeout
        try:
            await asyncio.wait_for(task2, timeout=3)
        except asyncio.TimeoutError:
            print("Task2 completed or timed out")
        
        # Clean up all background tasks on shutdown
        # This is handled in the graceful_shutdown method
    
    async def _timed_worker(self, name: str, duration: float) -> None:
        """Worker that runs for a specific duration"""
        print(f"{name} starting (will run for {duration}s)...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < duration:
            if self._shutdown_event.is_set():
                break
            print(f"{name} working...")
            await asyncio.sleep(0.5)
            
        print(f"{name} completed after {duration}s")
        
    async def _background_worker(self, name: str) -> None:
        """Example background worker"""
        try:
            while not self._shutdown_event.is_set():
                print(f"{name} working...")
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print(f"{name} cancelled")
            raise
            
    async def _fire_and_forget_with_error_handling(self) -> None:
        """Example of fire-and-forget pattern with error handling"""
        try:
            await asyncio.sleep(0.5)
            # Simulate some work
            print("Fire-and-forget task completed")
        except Exception as e:
            # Always handle errors in fire-and-forget tasks
            print(f"Error in fire-and-forget task: {e}")
            
    async def proper_gather_usage(self) -> None:
        """Demonstrates proper asyncio.gather usage"""
        # Always await asyncio.gather
        
        async def fetch_data(url: str) -> str:
            await asyncio.sleep(0.1)  # Simulate network delay
            return f"Data from {url}"
            
        urls = ["api1.com", "api2.com", "api3.com"]
        
        # Pattern 1: Basic gather with await
        results = await asyncio.gather(
            *[fetch_data(url) for url in urls]
        )
        print(f"Gathered results: {results}")
        
        # Pattern 2: Gather with error handling
        results_with_errors = await asyncio.gather(
            *[fetch_data(url) for url in urls],
            return_exceptions=True
        )
        
        for i, result in enumerate(results_with_errors):
            if isinstance(result, Exception):
                print(f"Error fetching {urls[i]}: {result}")
            else:
                print(f"Success: {result}")
                
    async def task_group_example(self) -> None:
        """Demonstrates TaskGroup usage (Python 3.11+)"""
        if sys.version_info >= (3, 11):
            async with asyncio.TaskGroup() as tg:
                # Tasks are automatically awaited and errors are propagated
                task1 = tg.create_task(self._example_task("task1"))
                task2 = tg.create_task(self._example_task("task2"))
                task3 = tg.create_task(self._example_task("task3"))
                
            # All tasks are complete here
            print("All tasks in TaskGroup completed")
        else:
            print("TaskGroup requires Python 3.11+")
            # Fallback to traditional approach
            tasks = [
                asyncio.create_task(self._example_task("task1")),
                asyncio.create_task(self._example_task("task2")),
                asyncio.create_task(self._example_task("task3"))
            ]
            await asyncio.gather(*tasks)
            
    async def _example_task(self, name: str) -> str:
        """Example task for TaskGroup"""
        await asyncio.sleep(0.1)
        return f"{name} completed"
        
    async def timeout_handling(self) -> None:
        """Shows proper timeout handling"""
        async def slow_operation() -> str:
            await asyncio.sleep(5)
            return "completed"
            
        # Pattern 1: Using asyncio.timeout (Python 3.11+)
        if sys.version_info >= (3, 11):
            try:
                async with asyncio.timeout(2):
                    result = await slow_operation()
            except asyncio.TimeoutError:
                print("Operation timed out (asyncio.timeout)")
        else:
            # Pattern 2: Using wait_for
            try:
                result = await asyncio.wait_for(slow_operation(), timeout=2)
            except asyncio.TimeoutError:
                print("Operation timed out (wait_for)")
                
    async def graceful_shutdown(self) -> None:
        """Demonstrates graceful shutdown pattern"""
        print("Initiating graceful shutdown...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for all tasks to complete with timeout
        if self._background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("Some tasks didn't shutdown gracefully in time")
                
        print("Shutdown complete")
        
    @asynccontextmanager
    async def managed_resource(self):
        """Context manager for async resource management"""
        # Acquire resource
        print("Acquiring resource...")
        resource = {"connected": True}
        
        try:
            yield resource
        finally:
            # Always cleanup
            print("Releasing resource...")
            resource["connected"] = False
            
    async def exception_handling_patterns(self) -> None:
        """Shows exception handling in async code"""
        
        # Pattern 1: Try-except in async function
        async def risky_operation() -> None:
            await asyncio.sleep(0.1)
            raise ValueError("Something went wrong")
            
        try:
            await risky_operation()
        except ValueError as e:
            print(f"Caught expected error: {e}")
            
        # Pattern 2: Exception groups (Python 3.11+)
        if sys.version_info >= (3, 11):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(risky_operation())
                tg.create_task(self._example_task("safe_task"))
                
            # TaskGroup will raise ExceptionGroup if any task fails
            
    async def run_demo(self) -> None:
        """Run all demonstrations"""
        print("=== Async Best Practices Demo ===\n")
        
        print("1. Safe shared state updates:")
        await self.safe_shared_state_update("test_key", "test_value")
        print()
        
        print("2. Concurrent reads with single write:")
        await self.concurrent_reads_single_write()
        print()
        
        print("3. Proper task management:")
        await self.proper_task_management()
        # Give background worker a moment to show it's running
        await asyncio.sleep(1)
        print()
        
        print("4. Proper gather usage:")
        await self.proper_gather_usage()
        print()
        
        print("5. TaskGroup example:")
        await self.task_group_example()
        print()
        
        print("6. Timeout handling:")
        await self.timeout_handling()
        print()
        
        print("7. Resource management:")
        async with self.managed_resource() as resource:
            print(f"Using resource: {resource}")
        print()
        
        print("8. Graceful shutdown:")
        await self.graceful_shutdown()
        

class AsyncQueueExample:
    """Demonstrates async queue patterns for producer-consumer scenarios"""
    
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=10)
        self.results = []
        self._lock = asyncio.Lock()
        
    async def producer(self, producer_id: int, num_items: int) -> None:
        """Producer that adds items to queue"""
        for i in range(num_items):
            item = f"item_{producer_id}_{i}"
            await self.queue.put(item)
            print(f"Producer {producer_id} produced: {item}")
            await asyncio.sleep(0.1)
            
    async def consumer(self, consumer_id: int) -> None:
        """Consumer that processes items from queue"""
        while True:
            try:
                # Wait for item with timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                print(f"Consumer {consumer_id} processing: {item}")
                
                # Simulate processing
                await asyncio.sleep(0.2)
                
                # Thread-safe result storage
                async with self._lock:
                    self.results = [*self.results, f"processed_{item}"]
                    
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No more items, exit
                print(f"Consumer {consumer_id} exiting (timeout)")
                break
                
    async def run_queue_example(self) -> None:
        """Run producer-consumer example"""
        print("=== Async Queue Example ===\n")
        
        # Create producers and consumers
        producers = [
            asyncio.create_task(self.producer(i, 3))
            for i in range(2)
        ]
        
        consumers = [
            asyncio.create_task(self.consumer(i))
            for i in range(3)
        ]
        
        # Wait for producers to finish
        await asyncio.gather(*producers)
        
        # Wait for queue to be processed
        await self.queue.join()
        
        # Cancel consumers (they're waiting for timeout)
        for c in consumers:
            c.cancel()
            
        # Wait for consumers to finish
        await asyncio.gather(*consumers, return_exceptions=True)
        
        print(f"\nProcessed items: {self.results}")


    async def cleanup(self) -> None:
        """Clean up background tasks."""
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

async def main():
    """Main entry point"""
    # Run best practices demo
    demo = AsyncBestPracticesDemo()
    await demo.run_demo()
    
    print("\n" + "="*50 + "\n")
    
    # Run queue example
    queue_example = AsyncQueueExample()
    await queue_example.run_queue_example()
    

if __name__ == "__main__":
    # Proper way to run async main
    asyncio.run(main())
