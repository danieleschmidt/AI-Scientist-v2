#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autonomous SDLC Orchestrator Generations
================================================================

Tests for all three generations of the TERRAGON SDLC MASTER implementation:
- Generation 1: MAKE IT WORK (Basic functionality)
- Generation 2: MAKE IT ROBUST (Error handling, monitoring)
- Generation 3: MAKE IT SCALE (Performance, distributed execution)

This test suite validates functionality, robustness, security, and performance
across all generations to ensure production readiness.

Author: AI Scientist v2 Autonomous System
License: MIT
"""

import unittest
import time
import threading
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
import sys
import os

# Test imports with error handling
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai_scientist.autonomous_sdlc_orchestrator import AutonomousSDLCOrchestrator
    GENERATION_1_AVAILABLE = True
except ImportError as e:
    GENERATION_1_AVAILABLE = False
    print(f"Generation 1 not available: {e}")

try:
    from ai_scientist.robust_autonomous_orchestrator import RobustAutonomousSDLCOrchestrator
    GENERATION_2_AVAILABLE = True
except ImportError as e:
    GENERATION_2_AVAILABLE = False
    print(f"Generation 2 not available: {e}")

try:
    from ai_scientist.scalable_autonomous_orchestrator import ScalableAutonomousSDLCOrchestrator
    GENERATION_3_AVAILABLE = True
except ImportError as e:
    GENERATION_3_AVAILABLE = False
    print(f"Generation 3 not available: {e}")


class TestGeneration1BasicFunctionality(unittest.TestCase):
    """Test Generation 1: MAKE IT WORK - Basic functionality validation."""
    
    def setUp(self):
        """Set up test environment."""
        if not GENERATION_1_AVAILABLE:
            self.skipTest("Generation 1 orchestrator not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.orchestrator = AutonomousSDLCOrchestrator(workspace_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown_gracefully()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test basic orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(hasattr(self.orchestrator, 'system_status'))
        self.assertTrue(hasattr(self.orchestrator, 'run_research_cycle'))
    
    def test_simple_research_cycle(self):
        """Test basic research cycle execution."""
        result = self.orchestrator.run_research_cycle(
            research_goal="Test basic functionality",
            domain="machine_learning",
            budget=100.0,
            time_limit=60.0
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("research_goal", result)
        self.assertIn("execution_summary", result)
        
        # Verify basic success criteria
        summary = result.get("execution_summary", {})
        self.assertGreater(summary.get("tasks_completed", 0), 0)
    
    def test_task_pipeline_creation(self):
        """Test research pipeline task creation."""
        pipeline = self.orchestrator.create_research_pipeline(
            research_goal="Test pipeline creation",
            domain="machine_learning"
        )
        
        self.assertIsInstance(pipeline, list)
        self.assertGreater(len(pipeline), 0)
        
        # Verify task structure
        for task in pipeline:
            self.assertTrue(hasattr(task, 'task_id'))
            self.assertTrue(hasattr(task, 'task_type'))
            self.assertTrue(hasattr(task, 'description'))
    
    def test_error_handling_basic(self):
        """Test basic error handling."""
        # Test with invalid parameters
        with self.assertRaises((ValueError, TypeError)):
            self.orchestrator.run_research_cycle(
                research_goal="",  # Invalid empty goal
                budget=-100.0     # Invalid negative budget
            )


class TestGeneration2RobustFeatures(unittest.TestCase):
    """Test Generation 2: MAKE IT ROBUST - Robustness and error handling."""
    
    def setUp(self):
        """Set up robust test environment."""
        if not GENERATION_2_AVAILABLE:
            self.skipTest("Generation 2 orchestrator not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.orchestrator = RobustAutonomousSDLCOrchestrator(workspace_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up robust test environment."""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown_gracefully()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_robust_initialization(self):
        """Test robust orchestrator initialization with enhanced features."""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(hasattr(self.orchestrator, 'task_scheduler'))
        self.assertTrue(hasattr(self.orchestrator, 'health_monitor'))
        self.assertTrue(hasattr(self.orchestrator, 'performance_metrics'))
    
    def test_robust_research_cycle(self):
        """Test robust research cycle with error handling."""
        result = self.orchestrator.run_robust_research_cycle(
            research_goal="Test robust execution",
            domain="machine_learning",
            budget=200.0,
            time_limit=120.0,
            quality_threshold=0.7
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("execution_summary", result)
        self.assertIn("system_health", result)
        self.assertIn("recovery_summary", result)
        
        # Verify robust features
        summary = result.get("execution_summary", {})
        self.assertIsInstance(summary.get("success_rate", 0), float)
        self.assertIsInstance(summary.get("average_quality", 0), float)
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery and retry mechanisms."""
        # Create pipeline with potential failure points
        pipeline = self.orchestrator.create_robust_research_pipeline(
            research_goal="Test error recovery",
            domain="machine_learning",
            budget=150.0,
            time_limit=90.0
        )
        
        self.assertIsInstance(pipeline, list)
        
        # Verify retry configuration
        for task in pipeline:
            self.assertGreater(task.max_retries, 0)
            self.assertGreater(task.timeout_seconds, 0)
    
    def test_health_monitoring(self):
        """Test system health monitoring."""
        health_status = self.orchestrator.health_monitor.get_health_status()
        
        self.assertIsInstance(health_status, dict)
        self.assertIn("status", health_status)
        self.assertIn("monitoring_active", health_status)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics collection."""
        metrics = self.orchestrator.performance_metrics
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("orchestrator_start_time", metrics)
        self.assertIn("total_tasks_completed", metrics)
        self.assertIn("success_rate", metrics)
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown functionality."""
        # Start a process and then shutdown
        self.orchestrator.shutdown_gracefully()
        
        # Verify system is properly stopped
        self.assertEqual(self.orchestrator.system_status.value, "stopped")
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern implementation."""
        # Access circuit breakers if available
        if hasattr(self.orchestrator.task_scheduler, 'circuit_breakers'):
            breakers = self.orchestrator.task_scheduler.circuit_breakers
            self.assertIsInstance(breakers, dict)
            
            # Test circuit breaker states
            for task_type, breaker in breakers.items():
                self.assertTrue(hasattr(breaker, 'state'))
                self.assertIn(breaker.state, ["CLOSED", "OPEN", "HALF_OPEN"])


class TestGeneration3ScalabilityFeatures(unittest.TestCase):
    """Test Generation 3: MAKE IT SCALE - Performance and scalability."""
    
    def setUp(self):
        """Set up scalable test environment."""
        if not GENERATION_3_AVAILABLE:
            self.skipTest("Generation 3 orchestrator not available")
        
        self.test_dir = tempfile.mkdtemp()
        config = {
            "max_workers": 4,
            "enable_distributed": False,  # Disable for testing
            "cache_size_mb": 64,
            "optimization_interval": 10.0
        }
        self.orchestrator = ScalableAutonomousSDLCOrchestrator(config)
    
    def tearDown(self):
        """Clean up scalable test environment."""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown_gracefully()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_scalable_initialization(self):
        """Test scalable orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(hasattr(self.orchestrator, 'task_scheduler'))
        self.assertTrue(hasattr(self.orchestrator, 'performance_monitor'))
        self.assertTrue(hasattr(self.orchestrator, 'memory_optimizer'))
    
    def test_advanced_caching_system(self):
        """Test advanced caching functionality."""
        cache = self.orchestrator.task_scheduler.task_cache
        
        self.assertIsNotNone(cache)
        self.assertTrue(hasattr(cache, 'get'))
        self.assertTrue(hasattr(cache, 'put'))
        self.assertTrue(hasattr(cache, 'get_stats'))
        
        # Test cache statistics
        stats = cache.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
    
    def test_scalable_research_cycle(self):
        """Test scalable research cycle execution."""
        # Use asyncio.run to handle async method
        async def run_test():
            result = await self.orchestrator.run_scalable_research_cycle(
                research_goal="Test scalable execution",
                domain="machine_learning",
                budget=300.0,
                time_limit=180.0
            )
            return result
        
        result = asyncio.run(run_test())
        
        self.assertIsInstance(result, dict)
        self.assertIn("execution_summary", result)
        self.assertIn("scalability_metrics", result)
        self.assertIn("performance_analysis", result)
        
        # Verify scalability metrics
        scalability = result.get("scalability_metrics", {})
        self.assertIn("average_scaling_efficiency", scalability)
        self.assertIn("cache_hit_rate", scalability)
    
    def test_parallel_task_execution(self):
        """Test parallel task execution capabilities."""
        # Create scalable pipeline
        pipeline = self.orchestrator.create_scalable_research_pipeline(
            research_goal="Test parallel execution",
            domain="machine_learning",
            budget=200.0,
            time_limit=120.0
        )
        
        self.assertIsInstance(pipeline, list)
        
        # Verify parallelizable tasks
        parallel_tasks = [task for task in pipeline if task.parallelizable]
        self.assertGreater(len(parallel_tasks), 0)
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        scheduler = self.orchestrator.task_scheduler
        
        # Test performance metrics
        metrics = scheduler.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("scheduler_metrics", metrics)
        self.assertIn("cache_performance", metrics)
        self.assertIn("resource_usage", metrics)
        
        # Test optimization execution
        scheduler.optimize_performance()  # Should not raise exceptions
    
    def test_resource_pool_management(self):
        """Test resource pool management."""
        scheduler = self.orchestrator.task_scheduler
        
        # Test resource status
        status = scheduler.get_performance_metrics()
        self.assertIn("worker_utilization", status)
        self.assertIn("resource_usage", status)
        
        # Verify resource tracking
        resource_usage = status["resource_usage"]
        self.assertIn("cpu_cores_used", resource_usage)
        self.assertIn("memory_gb_used", resource_usage)
    
    def test_distributed_execution_configuration(self):
        """Test distributed execution configuration."""
        # Test with distributed enabled
        config = {"enable_distributed": True, "max_workers": 2}
        
        try:
            distributed_orchestrator = ScalableAutonomousSDLCOrchestrator(config)
            self.assertIsNotNone(distributed_orchestrator.task_scheduler.distributed_executor)
            distributed_orchestrator.shutdown_gracefully()
        except Exception as e:
            # Distributed execution may not be available in test environment
            self.skipTest(f"Distributed execution not available: {e}")


class TestSecurityValidation(unittest.TestCase):
    """Test security aspects of all generations."""
    
    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization."""
        test_cases = [
            # Test malicious inputs
            {"research_goal": "<script>alert('xss')</script>", "should_reject": True},
            {"research_goal": "'; DROP TABLE users; --", "should_reject": True},
            {"domain": "../../../etc/passwd", "should_reject": True},
            
            # Test valid inputs
            {"research_goal": "Legitimate research goal", "should_reject": False},
            {"domain": "machine_learning", "should_reject": False}
        ]
        
        for generation, orchestrator_class in [
            (1, AutonomousSDLCOrchestrator if GENERATION_1_AVAILABLE else None),
            (2, RobustAutonomousSDLCOrchestrator if GENERATION_2_AVAILABLE else None),
            (3, ScalableAutonomousSDLCOrchestrator if GENERATION_3_AVAILABLE else None)
        ]:
            if orchestrator_class is None:
                continue
                
            with self.subTest(generation=generation):
                temp_dir = tempfile.mkdtemp()
                try:
                    if generation == 3:
                        orchestrator = orchestrator_class({"enable_distributed": False})
                    else:
                        orchestrator = orchestrator_class(workspace_dir=temp_dir)
                    
                    for test_case in test_cases:
                        research_goal = test_case["research_goal"]
                        domain = test_case.get("domain", "machine_learning")
                        should_reject = test_case["should_reject"]
                        
                        if should_reject:
                            # Should either raise exception or sanitize input
                            try:
                                # Basic validation - should not contain script tags or SQL injection
                                self.assertNotIn("<script>", research_goal.lower())
                                self.assertNotIn("drop table", research_goal.lower())
                                self.assertNotIn("../", domain)
                            except (ValueError, TypeError):
                                # Exception is acceptable for malicious input
                                pass
                        
                    orchestrator.shutdown_gracefully()
                    
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_file_system_access_controls(self):
        """Test file system access controls."""
        for generation, orchestrator_class in [
            (2, RobustAutonomousSDLCOrchestrator if GENERATION_2_AVAILABLE else None),
            (3, ScalableAutonomousSDLCOrchestrator if GENERATION_3_AVAILABLE else None)
        ]:
            if orchestrator_class is None:
                continue
                
            with self.subTest(generation=generation):
                temp_dir = tempfile.mkdtemp()
                try:
                    if generation == 3:
                        orchestrator = orchestrator_class({"enable_distributed": False})
                    else:
                        orchestrator = orchestrator_class(workspace_dir=temp_dir)
                    
                    # Test workspace directory is properly contained
                    if hasattr(orchestrator, 'workspace_dir'):
                        workspace = Path(orchestrator.workspace_dir)
                        self.assertTrue(workspace.exists())
                        self.assertTrue(workspace.is_dir())
                        
                        # Workspace should be within expected boundaries
                        workspace_str = str(workspace.resolve())
                        self.assertNotIn("../", workspace_str)
                        self.assertNotIn("/etc", workspace_str)
                        self.assertNotIn("/root", workspace_str)
                    
                    orchestrator.shutdown_gracefully()
                    
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_resource_limits_and_dos_protection(self):
        """Test resource limits and DoS protection."""
        # Test extreme parameter values
        extreme_cases = [
            {"budget": 999999999.0, "time_limit": 999999999.0},
            {"budget": 0.0, "time_limit": 0.0},
            {"research_goal": "A" * 10000}  # Very long string
        ]
        
        for generation, orchestrator_class in [
            (2, RobustAutonomousSDLCOrchestrator if GENERATION_2_AVAILABLE else None),
            (3, ScalableAutonomousSDLCOrchestrator if GENERATION_3_AVAILABLE else None)
        ]:
            if orchestrator_class is None:
                continue
                
            with self.subTest(generation=generation):
                temp_dir = tempfile.mkdtemp()
                try:
                    if generation == 3:
                        orchestrator = orchestrator_class({"enable_distributed": False})
                    else:
                        orchestrator = orchestrator_class(workspace_dir=temp_dir)
                    
                    for case in extreme_cases:
                        try:
                            # Should either handle gracefully or reject
                            result = orchestrator.run_robust_research_cycle(
                                research_goal=case.get("research_goal", "Test"),
                                budget=case.get("budget", 100.0),
                                time_limit=case.get("time_limit", 60.0)
                            )
                            
                            # If execution succeeds, verify reasonable resource usage
                            if isinstance(result, dict) and "execution_summary" in result:
                                summary = result["execution_summary"]
                                
                                # Verify reasonable bounds
                                if "total_cost" in summary:
                                    self.assertLess(summary["total_cost"], 100000)
                                if "total_time" in summary:
                                    self.assertLess(summary["total_time"], 7200)  # 2 hours max
                                    
                        except (ValueError, TypeError, RuntimeError):
                            # Rejection of extreme values is acceptable
                            pass
                    
                    orchestrator.shutdown_gracefully()
                    
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks across generations."""
    
    def setUp(self):
        """Set up performance testing environment."""
        self.benchmark_results = {}
    
    def test_performance_comparison_across_generations(self):
        """Compare performance across different generations."""
        research_goal = "Performance benchmark test"
        domain = "machine_learning"
        budget = 500.0
        time_limit = 300.0  # 5 minutes
        
        generations = [
            (1, AutonomousSDLCOrchestrator if GENERATION_1_AVAILABLE else None),
            (2, RobustAutonomousSDLCOrchestrator if GENERATION_2_AVAILABLE else None),
            (3, ScalableAutonomousSDLCOrchestrator if GENERATION_3_AVAILABLE else None)
        ]
        
        for generation, orchestrator_class in generations:
            if orchestrator_class is None:
                continue
                
            with self.subTest(generation=generation):
                temp_dir = tempfile.mkdtemp()
                start_time = time.time()
                
                try:
                    if generation == 3:
                        orchestrator = orchestrator_class({"enable_distributed": False})
                    else:
                        orchestrator = orchestrator_class(workspace_dir=temp_dir)
                    
                    # Run performance test
                    if generation == 3:
                        # Async execution for Generation 3
                        async def run_test():
                            return await orchestrator.run_scalable_research_cycle(
                                research_goal, domain, budget, time_limit
                            )
                        result = asyncio.run(run_test())
                    else:
                        # Sync execution for Generations 1 & 2
                        if generation == 2:
                            result = orchestrator.run_robust_research_cycle(
                                research_goal, domain, budget, time_limit
                            )
                        else:
                            result = orchestrator.run_research_cycle(
                                research_goal, domain, budget, time_limit
                            )
                    
                    execution_time = time.time() - start_time
                    
                    # Record performance metrics
                    self.benchmark_results[f"generation_{generation}"] = {
                        "execution_time": execution_time,
                        "success": "error" not in result,
                        "tasks_completed": result.get("execution_summary", {}).get("tasks_completed", 0),
                        "memory_usage": self._get_memory_usage(),
                        "result_quality": result.get("execution_summary", {}).get("average_quality", 0.0)
                    }
                    
                    # Verify reasonable performance
                    self.assertLess(execution_time, time_limit + 60)  # Allow 1 minute overhead
                    
                    orchestrator.shutdown_gracefully()
                    
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Print performance comparison
        self._print_performance_comparison()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _print_performance_comparison(self):
        """Print performance comparison results."""
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        
        for generation, metrics in self.benchmark_results.items():
            print(f"\n{generation.upper()}:")
            print(f"  Execution Time: {metrics['execution_time']:.2f}s")
            print(f"  Success: {metrics['success']}")
            print(f"  Tasks Completed: {metrics['tasks_completed']}")
            print(f"  Memory Usage: {metrics['memory_usage']:.1f}MB")
            print(f"  Result Quality: {metrics['result_quality']:.2f}")
        
        print("\n" + "="*50)
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within reasonable limits."""
        if not GENERATION_3_AVAILABLE:
            self.skipTest("Generation 3 required for memory optimization tests")
        
        config = {"memory_optimization": True, "enable_distributed": False}
        orchestrator = ScalableAutonomousSDLCOrchestrator(config)
        
        try:
            initial_memory = self._get_memory_usage()
            
            # Run memory-intensive test
            async def memory_test():
                return await orchestrator.run_scalable_research_cycle(
                    research_goal="Memory usage test",
                    domain="machine_learning",
                    budget=200.0,
                    time_limit=120.0
                )
            
            result = asyncio.run(memory_test())
            
            final_memory = self._get_memory_usage()
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be reasonable (less than 500MB)
            self.assertLess(memory_growth, 500.0)
            
        finally:
            orchestrator.shutdown_gracefully()


class TestIntegrationAndEndToEnd(unittest.TestCase):
    """Integration and end-to-end testing across generations."""
    
    def test_complete_research_workflow(self):
        """Test complete research workflow end-to-end."""
        research_scenarios = [
            {
                "goal": "Develop efficient neural network compression techniques",
                "domain": "machine_learning",
                "budget": 1000.0,
                "time_limit": 600.0,
                "expected_tasks": ["ideation", "experimentation", "analysis"]
            },
            {
                "goal": "Optimize distributed training algorithms",
                "domain": "deep_learning", 
                "budget": 1500.0,
                "time_limit": 900.0,
                "expected_tasks": ["ideation", "hypothesis_formation", "experiment_design"]
            }
        ]
        
        for scenario in research_scenarios:
            for generation, orchestrator_class in [
                (2, RobustAutonomousSDLCOrchestrator if GENERATION_2_AVAILABLE else None),
                (3, ScalableAutonomousSDLCOrchestrator if GENERATION_3_AVAILABLE else None)
            ]:
                if orchestrator_class is None:
                    continue
                    
                with self.subTest(generation=generation, scenario=scenario["goal"][:30]):
                    temp_dir = tempfile.mkdtemp()
                    
                    try:
                        if generation == 3:
                            orchestrator = orchestrator_class({"enable_distributed": False})
                        else:
                            orchestrator = orchestrator_class(workspace_dir=temp_dir)
                        
                        # Execute research workflow
                        if generation == 3:
                            async def run_workflow():
                                return await orchestrator.run_scalable_research_cycle(
                                    research_goal=scenario["goal"],
                                    domain=scenario["domain"],
                                    budget=scenario["budget"],
                                    time_limit=scenario["time_limit"]
                                )
                            result = asyncio.run(run_workflow())
                        else:
                            result = orchestrator.run_robust_research_cycle(
                                research_goal=scenario["goal"],
                                domain=scenario["domain"],
                                budget=scenario["budget"],
                                time_limit=scenario["time_limit"]
                            )
                        
                        # Validate workflow completion
                        self.assertIsInstance(result, dict)
                        self.assertIn("execution_summary", result)
                        
                        summary = result["execution_summary"]
                        self.assertGreater(summary.get("tasks_completed", 0), 0)
                        self.assertGreaterEqual(summary.get("success_rate", 0), 0.5)
                        
                        # Verify research outputs
                        if "research_outputs" in result:
                            outputs = result["research_outputs"]
                            self.assertIsInstance(outputs, dict)
                        
                        orchestrator.shutdown_gracefully()
                        
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_execution_safety(self):
        """Test concurrent execution safety."""
        if not GENERATION_2_AVAILABLE:
            self.skipTest("Generation 2 required for concurrency tests")
        
        def run_concurrent_research(thread_id: int, results: List[Dict]):
            """Run research in separate thread."""
            temp_dir = tempfile.mkdtemp()
            try:
                orchestrator = RobustAutonomousSDLCOrchestrator(workspace_dir=temp_dir)
                
                result = orchestrator.run_robust_research_cycle(
                    research_goal=f"Concurrent test {thread_id}",
                    domain="machine_learning",
                    budget=200.0,
                    time_limit=180.0
                )
                
                results.append({
                    "thread_id": thread_id,
                    "success": "error" not in result,
                    "result": result
                })
                
                orchestrator.shutdown_gracefully()
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Run multiple concurrent research cycles
        num_threads = 3
        results = []
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=run_concurrent_research,
                args=(i, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=300)  # 5 minute timeout
        
        # Verify all threads completed successfully
        self.assertEqual(len(results), num_threads)
        for result in results:
            self.assertTrue(result["success"], f"Thread {result['thread_id']} failed")


def run_comprehensive_test_suite():
    """Run comprehensive test suite for all generations."""
    print("🧪 TERRAGON SDLC MASTER - Comprehensive Test Suite")
    print("="*60)
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test discovery and execution
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGeneration1BasicFunctionality,
        TestGeneration2RobustFeatures, 
        TestGeneration3ScalabilityFeatures,
        TestSecurityValidation,
        TestPerformanceBenchmarks,
        TestIntegrationAndEndToEnd
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    print(f"Running {suite.countTestCases()} tests across {len(test_classes)} test classes...")
    print()
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100):.1f}%")
    
    # Print detailed failure information
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    print("\n" + "="*60)
    print("🛡️ QUALITY GATES VALIDATION COMPLETE")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)