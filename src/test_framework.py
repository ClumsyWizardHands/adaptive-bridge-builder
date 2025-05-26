#!/usr/bin/env python3
"""
Adaptive Bridge Builder Test Framework

This module provides the core framework for testing the Adaptive Bridge Builder,
including test case definitions, execution logic, and result reporting.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestFramework")


class TestResult(Enum):
    """Test result status enum."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


class TestSeverity(Enum):
    """Test severity level enum."""
    CRITICAL = "CRITICAL"   # Must pass for minimum functionality
    HIGH = "HIGH"           # Important for system reliability
    MEDIUM = "MEDIUM"       # Important for system quality
    LOW = "LOW"             # Nice to have, not critical


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    result: bool
    expected: Any
    actual: Any
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "result": "PASS" if self.result else "FAIL",
            "expected": str(self.expected),
            "actual": str(self.actual),
            "details": self.details
        }


@dataclass
class TestCaseResult:
    """Result of a test case execution."""
    test_id: str
    name: str
    status: TestResult
    duration_ms: float
    metrics: List[MetricResult] = field(default_factory=list)
    error: Optional[str] = None
    test_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "metrics": [m.to_dict() for m in self.metrics],
            "error": self.error,
            "test_data": self.test_data
        }
    
    @property
    def passed(self) -> bool:
        """Return True if the test case passed."""
        return self.status == TestResult.PASS


@dataclass
class TestMetric:
    """Definition of a test metric."""
    name: str
    description: str
    evaluator: Callable[[Dict[str, Any]], Tuple[bool, Any, Any, str]]
    weight: float = 1.0  # Relative importance of this metric (0.0-1.0)


@dataclass
class TestCase:
    """Definition of a test case."""
    id: str
    name: str
    description: str
    setup: Callable[[], Dict[str, Any]]
    execute: Callable[[Dict[str, Any]], Dict[str, Any]]
    metrics: List[TestMetric]
    teardown: Optional[Callable[[Dict[str, Any]], None]] = None
    severity: TestSeverity = TestSeverity.MEDIUM
    tags: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    
    def run(self) -> TestCaseResult:
        """Run the test case and return the result."""
        logger.info(f"Running test case: {self.name} [{self.id}]")
        start_time = time.time()
        test_data = {}
        
        try:
            # Setup
            logger.debug(f"Setting up test case: {self.name}")
            test_data = self.setup()
            
            # Execute
            logger.debug(f"Executing test case: {self.name}")
            result_data = self.execute(test_data)
            test_data.update(result_data)
            
            # Evaluate metrics
            metrics_results = []
            all_passed = True
            
            for metric in self.metrics:
                logger.debug(f"Evaluating metric: {metric.name}")
                try:
                    passed, expected, actual, details = metric.evaluator(test_data)
                    metric_result = MetricResult(
                        name=metric.name,
                        result=passed,
                        expected=expected,
                        actual=actual,
                        details=details
                    )
                    metrics_results.append(metric_result)
                    
                    if not passed:
                        all_passed = False
                        logger.warning(f"Metric failed: {metric.name} - {details}")
                except Exception as e:
                    logger.exception(f"Error evaluating metric {metric.name}: {str(e)}")
                    metric_result = MetricResult(
                        name=metric.name,
                        result=False,
                        expected="Successful evaluation",
                        actual=f"Error: {str(e)}",
                        details=traceback.format_exc()
                    )
                    metrics_results.append(metric_result)
                    all_passed = False
            
            status = TestResult.PASS if all_passed else TestResult.FAIL
            
        except Exception as e:
            logger.exception(f"Error in test case {self.name}: {str(e)}")
            status = TestResult.ERROR
            error = traceback.format_exc()
            metrics_results = []
        else:
            error = None
        
        finally:
            # Teardown
            if self.teardown:
                try:
                    logger.debug(f"Tearing down test case: {self.name}")
                    self.teardown(test_data)
                except Exception as e:
                    logger.exception(f"Error in teardown for {self.name}: {str(e)}")
                    if error is None:  # Don't overwrite original error
                        error = f"Teardown error: {str(e)}\n{traceback.format_exc()}"
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Test case completed: {self.name} [{self.id}] - {status.value} in {duration_ms:.2f}ms")
        
        return TestCaseResult(
            test_id=self.id,
            name=self.name,
            status=status,
            duration_ms=duration_ms,
            metrics=metrics_results,
            error=error,
            test_data=test_data
        )


@dataclass
class TestSuite:
    """Collection of related test cases."""
    name: str
    description: str
    test_cases: List[TestCase]
    tags: List[str] = field(default_factory=list)
    setup: Optional[Callable[[], Dict[str, Any]]] = None
    teardown: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def run(self, filter_tags: Optional[List[str]] = None) -> List[TestCaseResult]:
        """Run all test cases in the suite and return results."""
        logger.info(f"Running test suite: {self.name}")
        results = []
        suite_data = {}
        
        # Suite setup
        if self.setup:
            try:
                logger.debug(f"Setting up test suite: {self.name}")
                suite_data = self.setup()
            except Exception as e:
                logger.exception(f"Error in suite setup for {self.name}: {str(e)}")
                # Create failure results for all tests
                for test_case in self.test_cases:
                    results.append(TestCaseResult(
                        test_id=test_case.id,
                        name=test_case.name,
                        status=TestResult.ERROR,
                        duration_ms=0.0,
                        error=f"Suite setup failed: {str(e)}"
                    ))
                return results
        
        try:
            # Run test cases
            for test_case in self.test_cases:
                # Apply tag filtering
                if filter_tags and not any(tag in test_case.tags for tag in filter_tags):
                    logger.info(f"Skipping test case due to tag filter: {test_case.name}")
                    results.append(TestCaseResult(
                        test_id=test_case.id,
                        name=test_case.name,
                        status=TestResult.SKIP,
                        duration_ms=0.0,
                        error="Skipped due to tag filter"
                    ))
                    continue
                
                # Execute the test case
                result = test_case.run()
                results.append(result)
                
                # Optional: add suite-specific data to the test result
                if suite_data:
                    result.test_data["suite_data"] = suite_data
        
        finally:
            # Suite teardown
            if self.teardown:
                try:
                    logger.debug(f"Tearing down test suite: {self.name}")
                    self.teardown(suite_data)
                except Exception as e:
                    logger.exception(f"Error in suite teardown for {self.name}: {str(e)}")
        
        # Log suite summary
        passed = sum(1 for r in results if r.status == TestResult.PASS)
        failed = sum(1 for r in results if r.status == TestResult.FAIL)
        errors = sum(1 for r in results if r.status == TestResult.ERROR)
        skipped = sum(1 for r in results if r.status == TestResult.SKIP)
        
        logger.info(f"Suite {self.name} completed: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
        
        return results


class TestFramework:
    """Main test framework coordinator."""
    
    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.suites: List[TestSuite] = []
        self.output_dir = output_dir or os.path.join(os.getcwd(), "test_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_suite(self, suite: TestSuite) -> None:
        """Add a test suite to the framework."""
        self.suites.append(suite)
    
    def run_suite(self, suite_name: str, filter_tags: Optional[List[str]] = None) -> List[TestCaseResult]:
        """Run a specific test suite by name."""
        for suite in self.suites:
            if suite.name == suite_name:
                return suite.run(filter_tags)
        
        raise ValueError(f"Test suite not found: {suite_name}")
    
    def run_all(self, filter_tags: Optional[List[str]] = None) -> Dict[str, List[TestCaseResult]]:
        """Run all test suites and return results by suite name."""
        all_results = {}
        
        for suite in self.suites:
            logger.info(f"Running suite: {suite.name}")
            results = suite.run(filter_tags)
            all_results[suite.name] = results
        
        return all_results
    
    def generate_report(self, results: Dict[str, List[TestCaseResult]], filename: Optional[str] = None) -> str:
        """Generate and save a JSON test report."""
        timestamp = int(time.time())
        report_filename = filename or f"test_report_{timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Calculate overall statistics
        total_tests = sum(len(suite_results) for suite_results in results.values())
        passed_tests = sum(sum(1 for r in suite_results if r.status == TestResult.PASS) 
                          for suite_results in results.values())
        failed_tests = sum(sum(1 for r in suite_results if r.status == TestResult.FAIL) 
                          for suite_results in results.values())
        error_tests = sum(sum(1 for r in suite_results if r.status == TestResult.ERROR) 
                         for suite_results in results.values())
        skipped_tests = sum(sum(1 for r in suite_results if r.status == TestResult.SKIP) 
                           for suite_results in results.values())
        
        # Build the report
        report = {
            "timestamp": timestamp,
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "pass_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            "suites": {}
        }
        
        for suite_name, suite_results in results.items():
            suite_summary = {
                "total": len(suite_results),
                "passed": sum(1 for r in suite_results if r.status == TestResult.PASS),
                "failed": sum(1 for r in suite_results if r.status == TestResult.FAIL),
                "errors": sum(1 for r in suite_results if r.status == TestResult.ERROR),
                "skipped": sum(1 for r in suite_results if r.status == TestResult.SKIP)
            }
            suite_summary["pass_rate"] = (suite_summary["passed"] / suite_summary["total"]) * 100 if suite_summary["total"] > 0 else 0
            
            report["suites"][suite_name] = {
                "summary": suite_summary,
                "test_cases": [r.to_dict() for r in suite_results]
            }
        
        # Save the report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_path}")
        return report_path


def create_id() -> str:
    """Create a unique ID for a test case."""
    return str(uuid.uuid4())


# Helper functions for common metric evaluations

def expect_equal(name: str, expected_key: str, description: str = "") -> TestMetric:
    """Create a metric that checks if a value equals the expected value."""
    def evaluator(data: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
        if expected_key not in data:
            return False, "Key present", "Key missing", f"Expected key '{expected_key}' not found in test data"
        
        expected = data[expected_key]
        actual = data.get(f"actual_{expected_key}", data.get("result", {}).get(expected_key))
        
        if actual is None:
            return False, expected, None, f"Actual value for '{expected_key}' not found in test data"
        
        result = expected == actual
        detail = "" if result else f"Expected {expected} but got {actual}"
        
        return result, expected, actual, detail
    
    return TestMetric(
        name=name,
        description=description or f"Check if {expected_key} equals expected value",
        evaluator=evaluator
    )


def expect_contains(name: str, container_key: str, item_key: str, description: str = "") -> TestMetric:
    """Create a metric that checks if a container contains an item."""
    def evaluator(data: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
        if container_key not in data:
            return False, "Container present", "Container missing", f"Expected container key '{container_key}' not found in test data"
        
        if item_key not in data:
            return False, "Item key present", "Item key missing", f"Expected item key '{item_key}' not found in test data"
        
        container = data[container_key]
        item = data[item_key]
        
        if isinstance(container, (list, tuple, set)):
            result = item in container
            detail = "" if result else f"Item {item} not found in {container}"
            return result, f"{item} in container", f"{item} {'in' if result else 'not in'} container", detail
        
        if isinstance(container, dict):
            result = item in container
            detail = "" if result else f"Key {item} not found in dictionary {container}"
            return result, f"{item} in dict keys", f"{item} {'in' if result else 'not in'} dict keys", detail
        
        if isinstance(container, str):
            result = item in container
            detail = "" if result else f"Substring '{item}' not found in string '{container}'"
            return result, f"'{item}' in string", f"'{item}' {'in' if result else 'not in'} string", detail
        
        return False, "Container is iterable", f"Container is {type(container)}", f"Container type {type(container)} does not support containment check"
    
    return TestMetric(
        name=name,
        description=description or f"Check if {container_key} contains {item_key}",
        evaluator=evaluator
    )


def expect_greater_than(name: str, value_key: str, threshold_key: str, description: str = "") -> TestMetric:
    """Create a metric that checks if a value is greater than a threshold."""
    def evaluator(data: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
        if value_key not in data:
            return False, "Value present", "Value missing", f"Expected value key '{value_key}' not found in test data"
        
        if threshold_key not in data:
            return False, "Threshold present", "Threshold missing", f"Expected threshold key '{threshold_key}' not found in test data"
        
        value = data[value_key]
        threshold = data[threshold_key]
        
        try:
            result = value > threshold
            detail = "" if result else f"Value {value} is not greater than threshold {threshold}"
            return result, f"> {threshold}", value, detail
        except TypeError:
            return False, f"> {threshold}", value, f"Cannot compare {value} ({type(value)}) with {threshold} ({type(threshold)})"
    
    return TestMetric(
        name=name,
        description=description or f"Check if {value_key} is greater than {threshold_key}",
        evaluator=evaluator
    )


def expect_no_exceptions(name: str = "No exceptions", description: str = "Check that no exceptions occurred during test") -> TestMetric:
    """Create a metric that checks if no exceptions occurred during the test."""
    def evaluator(data: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
        exceptions = data.get("exceptions", [])
        if not exceptions:
            return True, "No exceptions", "No exceptions", ""
        
        return False, "No exceptions", f"{len(exceptions)} exceptions", "\n".join(str(e) for e in exceptions)
    
    return TestMetric(
        name=name,
        description=description,
        evaluator=evaluator
    )


def expect_response_time_below(threshold_ms: float, description: str = "") -> TestMetric:
    """Create a metric that checks if response time is below a threshold."""
    def evaluator(data: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
        response_time = data.get("response_time_ms")
        
        if response_time is None:
            return False, f"< {threshold_ms}ms", "No data", "Response time data not found in test data"
        
        result = response_time < threshold_ms
        detail = "" if result else f"Response time {response_time}ms exceeds threshold {threshold_ms}ms"
        
        return result, f"< {threshold_ms}ms", f"{response_time}ms", detail
    
    return TestMetric(
        name=f"Response time below {threshold_ms}ms",
        description=description or f"Check if response time is below {threshold_ms}ms",
        evaluator=evaluator
    )


def custom_metric(name: str, description: str, evaluator_func: Callable[[Dict[str, Any]], Tuple[bool, Any, Any, str]], weight: float = 1.0) -> TestMetric:
    """Create a custom metric with a provided evaluator function."""
    return TestMetric(
        name=name,
        description=description,
        evaluator=evaluator_func,
        weight=weight
    )
