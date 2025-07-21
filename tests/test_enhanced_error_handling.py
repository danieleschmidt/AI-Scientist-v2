#!/usr/bin/env python3
"""
Tests for enhanced error handling in the tree search system.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_scientist.treesearch.error_handling import (
    EnhancedErrorHandler,
    FailureType,
    NodeClassifier,
    AdaptiveTimeoutManager,
    NodeRecoveryStrategy,
    ResourceManager,
    ExecutionMetrics
)


class MockNode:
    """Mock node for testing."""
    
    def __init__(self, node_id="test_node", debug_depth=0, exc_type=None, exc_info=None):
        self.id = node_id
        self.debug_depth = debug_depth
        self.exc_type = exc_type
        self.exc_info = exc_info
        self.exec_time = 10.0
        self.code = "print('test')"
        self.term_out = ["test output"]
        self.stage_name = "test"


class TestEnhancedErrorHandler(unittest.TestCase):
    """Test the main error handler class."""
    
    def setUp(self):
        """Set up test environment."""
        self.handler = EnhancedErrorHandler()
    
    def test_initialization(self):
        """Test error handler initialization."""
        self.assertIsNotNone(self.handler.timeout_manager)
        self.assertIsNotNone(self.handler.resource_manager)
        self.assertIsNotNone(self.handler.metrics)
        self.assertIsNotNone(self.handler.recovery_strategy)
        self.assertIsNotNone(self.handler.classifier)
    
    def test_handle_node_success(self):
        """Test handling of successful node execution."""
        node = MockNode()
        
        initial_successful = self.handler.metrics.successful_nodes
        self.handler.handle_node_success(node)
        
        self.assertEqual(self.handler.metrics.successful_nodes, initial_successful + 1)
        self.assertEqual(self.handler.metrics.total_nodes, initial_successful + 1)
    
    def test_handle_node_failure(self):
        """Test handling of failed node execution."""
        node = MockNode(exc_type="RuntimeError", exc_info="Test error")
        
        initial_failed = self.handler.metrics.failed_nodes
        result = self.handler.handle_node_failure(node)
        
        self.assertEqual(self.handler.metrics.failed_nodes, initial_failed + 1)
        self.assertIn('failure_type', result)
        self.assertIn('recoverable', result)
        self.assertIn('recovery_result', result)
        self.assertIn('should_debug', result)
    
    def test_timeout_calculation(self):
        """Test adaptive timeout calculation."""
        node = MockNode()
        timeout = self.handler.get_timeout_for_node(node, "debug")
        
        self.assertIsInstance(timeout, int)
        self.assertGreater(timeout, 0)


class TestNodeClassifier(unittest.TestCase):
    """Test node failure classification."""
    
    def test_syntax_error_classification(self):
        """Test classification of syntax errors."""
        node = MockNode(exc_type="SyntaxError")
        failure_type = NodeClassifier.classify_failure(node)
        self.assertEqual(failure_type, FailureType.SYNTAX_ERROR)
    
    def test_timeout_classification(self):
        """Test classification of timeout errors."""
        node = MockNode(exc_type="TimeoutError")
        failure_type = NodeClassifier.classify_failure(node)
        self.assertEqual(failure_type, FailureType.TIMEOUT)
    
    def test_gpu_error_classification(self):
        """Test classification of GPU errors."""
        node = MockNode(exc_info="CUDA out of memory")
        failure_type = NodeClassifier.classify_failure(node)
        self.assertEqual(failure_type, FailureType.GPU_ERROR)
    
    def test_unknown_error_classification(self):
        """Test classification of unknown errors."""
        node = MockNode(exc_type="UnknownError")
        failure_type = NodeClassifier.classify_failure(node)
        self.assertEqual(failure_type, FailureType.UNKNOWN)


class TestAdaptiveTimeoutManager(unittest.TestCase):
    """Test adaptive timeout management."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = AdaptiveTimeoutManager(base_timeout=60)
    
    def test_base_timeout(self):
        """Test basic timeout calculation."""
        node = MockNode()
        timeout = self.manager.get_timeout(node)
        self.assertGreaterEqual(timeout, 60)
    
    def test_debug_timeout_increase(self):
        """Test timeout increase for debug nodes."""
        debug_node = MockNode(debug_depth=1)
        normal_node = MockNode(debug_depth=0)
        
        debug_timeout = self.manager.get_timeout(debug_node, "debug")
        normal_timeout = self.manager.get_timeout(normal_node)
        
        self.assertGreater(debug_timeout, normal_timeout)
    
    def test_complex_code_timeout_increase(self):
        """Test timeout increase for complex code."""
        simple_node = MockNode()
        simple_node.code = "print('hello')"
        
        complex_node = MockNode()
        complex_node.code = "x = 1\n" * 1000  # Long code
        
        simple_timeout = self.manager.get_timeout(simple_node)
        complex_timeout = self.manager.get_timeout(complex_node)
        
        self.assertGreater(complex_timeout, simple_timeout)
    
    def test_execution_history_tracking(self):
        """Test execution history tracking."""
        initial_history_len = len(self.manager.execution_history)
        
        self.manager.record_execution(30.0, timed_out=False)
        self.assertEqual(len(self.manager.execution_history), initial_history_len + 1)
        
        self.manager.record_execution(120.0, timed_out=True)
        self.assertEqual(len(self.manager.timeout_history), 1)


class TestNodeRecoveryStrategy(unittest.TestCase):
    """Test node recovery strategies."""
    
    def test_timeout_recovery(self):
        """Test recovery from timeout with partial results."""
        node = MockNode()
        node.term_out = [
            "Epoch 1: loss=0.5 accuracy=0.8",
            "Epoch 2: loss=0.3 accuracy=0.85",
            "Training interrupted..."
        ]
        
        result = NodeRecoveryStrategy.recover_from_timeout(node)
        self.assertIsNotNone(result)
        self.assertIn('partial_loss', result)
        self.assertIn('partial_accuracy', result)
    
    def test_gpu_failure_recovery(self):
        """Test recovery from GPU failure."""
        node = MockNode()
        node.code = """
import torch
device = torch.device('cuda')
model = model.cuda()
data = data.to(device)
"""
        
        cpu_code = NodeRecoveryStrategy.recover_from_gpu_failure(node)
        self.assertIsNotNone(cpu_code)
        self.assertIn('device = "cpu"', cpu_code)
        self.assertIn('.cpu()', cpu_code)
    
    def test_syntax_error_recovery(self):
        """Test recovery from syntax errors."""
        node = MockNode()
        node.code = """
plt.plot([1, 2, 3])
np.array([1, 2, 3])
"""
        
        fixed_code = NodeRecoveryStrategy.recover_from_syntax_error(node)
        self.assertIsNotNone(fixed_code)
        self.assertIn('import matplotlib.pyplot as plt', fixed_code)
        self.assertIn('import numpy as np', fixed_code)


class TestResourceManager(unittest.TestCase):
    """Test resource management."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = ResourceManager()
    
    def test_resource_registration(self):
        """Test resource registration."""
        cleanup_fn = Mock()
        
        self.manager.register_resource(
            "test_resource", "test_type", cleanup_fn, {"meta": "data"}
        )
        
        self.assertIn("test_resource", self.manager.active_resources)
        resource = self.manager.active_resources["test_resource"]
        self.assertEqual(resource["type"], "test_type")
        self.assertEqual(resource["cleanup"], cleanup_fn)
    
    def test_single_resource_cleanup(self):
        """Test cleanup of single resource."""
        cleanup_fn = Mock()
        
        self.manager.register_resource("test_resource", "test_type", cleanup_fn)
        success = self.manager.cleanup_resource("test_resource")
        
        self.assertTrue(success)
        cleanup_fn.assert_called_once()
        self.assertNotIn("test_resource", self.manager.active_resources)
    
    def test_all_resources_cleanup(self):
        """Test cleanup of all resources."""
        cleanup_fns = []
        for i in range(3):
            cleanup_fn = Mock()
            cleanup_fns.append(cleanup_fn)
            self.manager.register_resource(f"resource_{i}", "test_type", cleanup_fn)
        
        results = self.manager.cleanup_all()
        
        # All cleanup functions should be called
        for cleanup_fn in cleanup_fns:
            cleanup_fn.assert_called_once()
        
        # All resources should be cleaned up
        self.assertEqual(len(self.manager.active_resources), 0)
        self.assertEqual(len(results), 3)
    
    def test_resource_stats(self):
        """Test resource statistics."""
        cleanup_fn = Mock()
        
        self.manager.register_resource("res1", "type_a", cleanup_fn)
        self.manager.register_resource("res2", "type_a", cleanup_fn)
        self.manager.register_resource("res3", "type_b", cleanup_fn)
        
        stats = self.manager.get_resource_stats()
        
        self.assertEqual(stats["total_resources"], 3)
        self.assertEqual(stats["resource_types"]["type_a"], 2)
        self.assertEqual(stats["resource_types"]["type_b"], 1)
        self.assertGreater(stats["oldest_resource_age"], 0)


class TestExecutionMetrics(unittest.TestCase):
    """Test execution metrics tracking."""
    
    def setUp(self):
        """Set up test environment."""
        self.metrics = ExecutionMetrics()
    
    def test_success_recording(self):
        """Test recording successful executions."""
        initial_count = self.metrics.successful_nodes
        
        self.metrics.record_success(15.0)
        
        self.assertEqual(self.metrics.successful_nodes, initial_count + 1)
        self.assertEqual(self.metrics.total_nodes, initial_count + 1)
        self.assertEqual(self.metrics.avg_execution_time, 15.0)
    
    def test_failure_recording(self):
        """Test recording failed executions."""
        initial_count = self.metrics.failed_nodes
        
        self.metrics.record_failure(FailureType.TIMEOUT, 30.0)
        
        self.assertEqual(self.metrics.failed_nodes, initial_count + 1)
        self.assertEqual(self.metrics.total_nodes, initial_count + 1)
        self.assertEqual(self.metrics.timeout_nodes, 1)
        self.assertEqual(self.metrics.failure_by_type[FailureType.TIMEOUT], 1)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Record some successes and failures
        self.metrics.record_success(10.0)
        self.metrics.record_success(15.0)
        self.metrics.record_failure(FailureType.RUNTIME_ERROR, 5.0)
        
        # Success rate should be 2/3 = 66.67%
        expected_rate = (2 / 3) * 100
        self.assertAlmostEqual(self.metrics.success_rate, expected_rate, places=2)
    
    def test_average_execution_time_tracking(self):
        """Test average execution time calculation."""
        times = [10.0, 20.0, 30.0]
        
        for time_val in times:
            self.metrics.record_success(time_val)
        
        # Should be approximately the average of the times
        # (using exponential moving average, so not exactly 20.0)
        self.assertGreater(self.metrics.avg_execution_time, 15.0)
        self.assertLess(self.metrics.avg_execution_time, 25.0)


if __name__ == '__main__':
    # Create a simple test runner
    print("Running enhanced error handling tests...")
    
    test_classes = [
        TestEnhancedErrorHandler,
        TestNodeClassifier,
        TestAdaptiveTimeoutManager,
        TestNodeRecoveryStrategy,
        TestResourceManager,
        TestExecutionMetrics
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_passed = class_total - len(result.failures) - len(result.errors)
        
        total_tests += class_total
        passed_tests += class_passed
        
        print(f"{test_class.__name__}: {class_passed}/{class_total} tests passed")
        
        if result.failures:
            print(f"  Failures: {len(result.failures)}")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    print("Enhanced error handling tests completed!")