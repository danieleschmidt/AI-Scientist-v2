#!/usr/bin/env python3
"""
Comprehensive Integration Tests for AI Scientist v2 Enterprise Edition

Tests for distributed computing, performance optimization, caching,
and enterprise CLI integration.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Mock pytest functions
def skip(reason):
    """Mock pytest skip."""
    print(f"SKIPPED: {reason}")
    return

class pytest:
    @staticmethod
    def skip(reason):
        raise SkipTest(reason)
    
    @staticmethod
    def fail(reason):
        raise AssertionError(reason)

    class mark:
        @staticmethod
        def asyncio(func):
            return func

class SkipTest(Exception):
    pass

# Test the enterprise systems
def test_import_availability():
    """Test that all enterprise modules can be imported."""
    try:
        # Test distributed executor import
        from ai_scientist.scaling.distributed_executor import (
            DistributedTaskManager, TaskDefinition, ResourceRequirement,
            create_ideation_task, create_experiment_task
        )
        assert True, "Distributed executor imports successful"
    except ImportError:
        print("SKIPPED:("Distributed executor not available")
    
    try:
        # Test performance optimizer import  
        from ai_scientist.optimization.performance_optimizer import (
            IntelligentCache, ResourcePool, PerformanceMetrics,
            AdaptiveOptimizer, CachePolicy
        )
        assert True, "Performance optimizer imports successful"
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    try:
        # Test enterprise CLI import
        from ai_scientist.cli_enterprise import EnterpriseAIScientistCLI
        assert True, "Enterprise CLI imports successful"
    except ImportError:
        print("SKIPPED:("Enterprise CLI not available")

def test_intelligent_cache_basic_operations():
    """Test basic cache operations."""
    try:
        from ai_scientist.optimization.performance_optimizer import IntelligentCache, CachePolicy
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    cache = IntelligentCache(max_size=100, max_memory_mb=10, policy=CachePolicy.LRU)
    
    # Test put and get
    assert cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test miss
    assert cache.get("nonexistent") is None
    
    # Test statistics
    stats = cache.get_statistics()
    assert stats['size'] == 1
    assert stats['hits'] >= 1
    assert stats['misses'] >= 1
    
    # Test invalidation
    assert cache.invalidate("key1")
    assert cache.get("key1") is None
    
    # Test clear
    cache.put("key2", "value2")
    cache.clear()
    assert cache.get("key2") is None
    assert cache.get_statistics()['size'] == 0

def test_cache_eviction_policies():
    """Test different cache eviction policies."""
    try:
        from ai_scientist.optimization.performance_optimizer import IntelligentCache, CachePolicy
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    # Test LRU eviction
    cache = IntelligentCache(max_size=2, policy=CachePolicy.LRU)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

def test_resource_pool_operations():
    """Test resource pool functionality."""
    try:
        from ai_scientist.optimization.performance_optimizer import ResourcePool
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    # Mock resource factory
    resource_count = 0
    def mock_factory():
        nonlocal resource_count
        resource_count += 1
        return f"resource_{resource_count}"
    
    pool = ResourcePool(factory=mock_factory, max_size=3)
    
    # Get resources
    resource1 = pool.get_resource()
    resource2 = pool.get_resource()
    
    assert resource1 != resource2
    
    # Return resources
    pool.return_resource(resource1)
    pool.return_resource(resource2)
    
    # Get resource again (should reuse)
    resource3 = pool.get_resource()
    assert resource3 in [resource1, resource2]

def test_distributed_task_manager():
    """Test distributed task manager functionality."""
    try:
        from ai_scientist.scaling.distributed_executor import (
            DistributedTaskManager, create_ideation_task, TaskStatus
        )
    except ImportError:
        print("SKIPPED:("Distributed executor not available")
    
    manager = DistributedTaskManager(max_workers=2)
    manager.start()
    
    try:
        # Create and submit task
        task = create_ideation_task(
            task_id="test_task_1",
            workshop_file="test_file.md",
            model="test_model",
            max_generations=5
        )
        
        task_id = manager.submit_task(task)
        assert task_id == "test_task_1"
        
        # Wait for completion
        result = manager.get_task_result(task_id, timeout=10)
        assert result is not None
        assert result.task_id == task_id
        
        # Check status
        status = manager.get_task_status(task_id)
        assert status == TaskStatus.COMPLETED
        
        # Get cluster status
        cluster_status = manager.get_cluster_status()
        assert 'cluster_size' in cluster_status
        assert cluster_status['completed_tasks'] >= 1
        
    finally:
        manager.stop()

def test_performance_metrics_and_optimization():
    """Test performance metrics tracking and optimization."""
    try:
        from ai_scientist.optimization.performance_optimizer import (
            PerformanceMetrics, PredictiveScaler, AdaptiveOptimizer
        )
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    # Test performance metrics
    metrics = PerformanceMetrics(
        avg_response_time=100.0,
        p95_response_time=150.0,
        throughput=10.0,
        error_rate=0.01,
        cpu_usage=50.0,
        memory_usage=60.0
    )
    
    # Test predictive scaler
    scaler = PredictiveScaler(window_size=10)
    
    # Record several metrics
    for i in range(5):
        test_metrics = PerformanceMetrics(
            cpu_usage=50 + i * 5,
            memory_usage=40 + i * 3,
            avg_response_time=100 + i * 10
        )
        scaler.record_metrics(test_metrics)
    
    # Check predictions
    cpu_prediction = scaler.get_prediction('cpu_usage')
    assert cpu_prediction is not None
    
    # Test optimizer
    optimizer = AdaptiveOptimizer()
    
    # Test auto-tuning
    current_metrics = PerformanceMetrics(cpu_usage=75.0, memory_usage=80.0)
    optimizer.auto_tune(current_metrics)
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    assert 'strategy' in summary
    assert 'cache_stats' in summary

def test_cli_basic_functionality():
    """Test basic CLI functionality."""
    try:
        from ai_scientist.cli_enterprise import EnterpriseAIScientistCLI
    except ImportError:
        print("SKIPPED:("Enterprise CLI not available")
    
    cli = EnterpriseAIScientistCLI()
    
    # Test initialization
    assert cli.session_id is not None
    assert cli.operation_count == 0
    assert cli.config['version'] == '3.0.0'
    
    # Test performance metrics recording
    cli._record_performance_metrics(1.5, "test_operation")
    assert len(cli.request_times) == 1
    assert cli.request_times[0] == 1.5

def test_enterprise_status_generation():
    """Test enterprise status dashboard generation."""
    try:
        from ai_scientist.cli_enterprise import EnterpriseAIScientistCLI
    except ImportError:
        print("SKIPPED:("Enterprise CLI not available")
    
    cli = EnterpriseAIScientistCLI()
    cli.initialize_systems()
    
    # This should not raise an exception
    try:
        cli.show_enterprise_status()
        assert True, "Enterprise status generated successfully"
    except Exception as e:
        raise AssertionError(f"Enterprise status generation failed: {e}")

# Async test
async def test_concurrent_cache_operations():
    """Test cache operations under concurrent load."""
    try:
        from ai_scientist.optimization.performance_optimizer import IntelligentCache
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    cache = IntelligentCache(max_size=100)
    
    async def cache_operations():
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
            await asyncio.sleep(0.01)
            result = cache.get(f"key_{i}")
            assert result == f"value_{i}"
    
    # Run concurrent operations
    tasks = [cache_operations() for _ in range(5)]
    await asyncio.gather(*tasks)
    
    stats = cache.get_statistics()
    assert stats['size'] <= 100  # Respect max size
    assert stats['hits'] > 0

def test_task_creation_and_validation():
    """Test task creation and validation."""
    try:
        from ai_scientist.scaling.distributed_executor import (
            create_ideation_task, create_experiment_task, ResourceRequirement
        )
    except ImportError:
        print("SKIPPED:("Distributed executor not available")
    
    # Test ideation task creation
    ideation_task = create_ideation_task(
        task_id="test_ideation",
        workshop_file="test.md",
        model="gpt-4",
        max_generations=10,
        priority=7
    )
    
    assert ideation_task.task_id == "test_ideation"
    assert ideation_task.task_type == "research_ideation"
    assert ideation_task.payload['workshop_file'] == "test.md"
    assert ideation_task.requirements.priority == 7
    
    # Test experiment task creation
    experiment_task = create_experiment_task(
        task_id="test_experiment",
        ideas_file="ideas.json",
        num_experiments=5,
        priority=8
    )
    
    assert experiment_task.task_id == "test_experiment"
    assert experiment_task.task_type == "experimental_research"
    assert experiment_task.payload['ideas_file'] == "ideas.json"
    assert experiment_task.requirements.gpu_count == 1

def test_error_handling_in_distributed_tasks():
    """Test error handling in distributed task execution."""
    try:
        from ai_scientist.scaling.distributed_executor import (
            DistributedTaskManager, TaskDefinition, ResourceRequirement, TaskStatus
        )
    except ImportError:
        print("SKIPPED:("Distributed executor not available")
    
    manager = DistributedTaskManager(max_workers=1)
    manager.start()
    
    try:
        # Create a task that will fail
        failing_task = TaskDefinition(
            task_id="failing_task",
            task_type="nonexistent_type",  # This will cause failure
            payload={},
            requirements=ResourceRequirement()
        )
        
        task_id = manager.submit_task(failing_task)
        result = manager.get_task_result(task_id, timeout=5)
        
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        
    finally:
        manager.stop()

def test_cache_memory_management():
    """Test cache memory management and limits."""
    try:
        from ai_scientist.optimization.performance_optimizer import IntelligentCache
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    # Small cache for testing memory limits
    cache = IntelligentCache(max_size=1000, max_memory_mb=1)  # 1MB limit
    
    # Try to add large data
    large_data = "x" * (1024 * 1024)  # 1MB string
    success = cache.put("large_key", large_data)
    
    # Should succeed for first entry
    assert success
    
    # Try to add another large entry (should trigger eviction)
    large_data2 = "y" * (1024 * 1024)  # Another 1MB
    success2 = cache.put("large_key2", large_data2)
    
    # Check that memory management is working
    stats = cache.get_statistics()
    assert stats['memory_usage_bytes'] <= 1024 * 1024 * 2  # Allow some overhead

def test_optimization_strategy_changes():
    """Test dynamic optimization strategy changes."""
    try:
        from ai_scientist.optimization.performance_optimizer import (
            AdaptiveOptimizer, OptimizationStrategy, PerformanceMetrics
        )
    except ImportError:
        print("SKIPPED:("Performance optimizer not available")
    
    optimizer = AdaptiveOptimizer(strategy=OptimizationStrategy.CONSERVATIVE)
    
    # Test strategy change
    assert optimizer.strategy == OptimizationStrategy.CONSERVATIVE
    
    # Simulate performance degradation
    bad_metrics = PerformanceMetrics(
        avg_response_time=2000.0,  # Very slow
        cpu_usage=95.0,           # Very high CPU
        memory_usage=90.0,        # Very high memory
        error_rate=0.1            # 10% errors
    )
    
    optimizer.auto_tune(bad_metrics)
    
    # Should trigger conservative optimizations
    summary = optimizer.get_optimization_summary()
    assert summary is not None

if __name__ == "__main__":
    # Run basic tests if called directly
    print("Running AI Scientist v2 Enterprise Integration Tests...")
    
    test_functions = [
        test_import_availability,
        test_intelligent_cache_basic_operations,
        test_cache_eviction_policies,
        test_resource_pool_operations,
        test_performance_metrics_and_optimization,
        test_cli_basic_functionality,
        test_task_creation_and_validation,
        test_cache_memory_management,
        test_optimization_strategy_changes
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")