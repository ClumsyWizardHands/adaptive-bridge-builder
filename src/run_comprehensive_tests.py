"""
Comprehensive Test Runner for Alex Familiar Project
This script runs all available tests and provides a detailed report.
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

class ComprehensiveTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def find_test_files(self) -> List[str]:
        """Find all test files in the current directory."""
        test_files = []
        for file in os.listdir('.'):
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(file)
        return sorted(test_files)
    
    def run_single_test(self, test_file: str) -> Tuple[bool, str, float]:
        """Run a single test file and return success status, output, and duration."""
        print(f"\n{'='*60}")
        print(f"Running {test_file}...")
        print(f"{'='*60}")
        
        start = time.time()
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per test
            )
            duration = time.time() - start
            
            # Check if test passed
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            # Print immediate result
            if success:
                print(f"✅ PASSED in {duration:.2f}s")
            else:
                print(f"❌ FAILED in {duration:.2f}s")
                print(f"Error output:\n{output}")
                
            return success, output, duration
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"⏱️ TIMEOUT after {duration:.2f}s")
            return False, "Test timed out", duration
        except Exception as e:
            duration = time.time() - start
            print(f"🔥 ERROR: {str(e)}")
            return False, str(e), duration
    
    def check_import_compatibility(self, test_file: str) -> Tuple[bool, str]:
        """Check if a test file can be imported without errors."""
        try:
            result = subprocess.run(
                [sys.executable, '-c', f'import {test_file[:-3]}'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True, "Import successful"
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)
    
    def run_all_tests(self):
        """Run all test files and collect results."""
        self.start_time = datetime.now()
        test_files = self.find_test_files()
        
        print(f"\n🧪 COMPREHENSIVE TEST SUITE")
        print(f"Found {len(test_files)} test files")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # First, check imports
        print("\n📦 Checking imports...")
        import_issues = []
        for test_file in test_files:
            can_import, error = self.check_import_compatibility(test_file)
            if not can_import:
                import_issues.append((test_file, error))
                print(f"  ⚠️ {test_file} - Import issue")
        
        if import_issues:
            print(f"\n⚠️ Found {len(import_issues)} import issues")
            for file, error in import_issues:
                print(f"\n{file}:")
                print(f"  {error.strip()}")
        
        # Run tests
        passed = 0
        failed = 0
        errors = 0
        total_duration = 0
        
        for test_file in test_files:
            success, output, duration = self.run_single_test(test_file)
            total_duration += duration
            
            self.test_results = {**self.test_results, test_file: {},
                'success': success,
                'output': output,
                'duration': duration
            }
            
            if success:
                passed += 1
            else:
                if "Error" in output or "error" in output:
                    errors += 1
                else:
                    failed += 1
        
        self.end_time = datetime.now()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {len(test_files)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🔥 Errors: {errors}")
        print(f"⏱️ Total duration: {total_duration:.2f}s")
        print(f"Completed at: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print failed tests details
        if failed > 0 or errors > 0:
            print(f"\n{'='*60}")
            print(f"❌ FAILED TESTS DETAILS")
            print(f"{'='*60}")
            for test_file, result in self.test_results.items():
                if not result['success']:
                    print(f"\n{test_file}:")
                    print(f"Duration: {result['duration']:.2f}s")
                    print(f"Output:\n{result['output'][:500]}...")  # First 500 chars
        
        # Save detailed report
        self.save_report()
        
        return passed == len(test_files)
    
    def save_report(self):
        """Save a detailed test report to file."""
        report = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration': (self.end_time - self.start_time).total_seconds(),
            'test_results': self.test_results,
            'summary': {
                'total': len(self.test_results),
                'passed': sum(1 for r in self.test_results.values() if r['success']),
                'failed': sum(1 for r in self.test_results.values() if not r['success'])
            }
        }
        
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


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
    """Main entry point."""
    runner = ComprehensiveTestRunner()
    all_passed = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
