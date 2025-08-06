"""
Tests for optimization and scaling components.

Comprehensive test suite for cache management, parallel execution,
resource pooling, and load balancing systems.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import Future

from quantum_task_planner.optimization.cache_manager import (
    QuantumCacheManager, CacheStrategy, CacheType, CacheEntry, CacheStatistics
)
from quantum_task_planner.optimization.parallel_executor import (
    ParallelQuantumExecutor, ExecutionMode, ExecutionTask, TaskStatus,
    ExecutionMetrics, WorkerPool
)
from quantum_task_planner.optimization.resource_pool import (
    ResourcePool, ResourceType, Resource, ResourceStatus, 
    AllocationRequest, AllocationResult, PoolStatistics
)
from quantum_task_planner.optimization.load_balancer import (
    QuantumLoadBalancer, LoadBalancingStrategy, LoadBalancerNode, 
    NodeStatus, RequestMetrics, LoadBalancerStats
)
from quantum_task_planner.core.planner import QuantumTaskPlanner, Task, TaskPriority


class TestQuantumCacheManager:
    """Test suite for QuantumCacheManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QuantumCacheManager(
            max_size_mb=10,
            max_entries=100,
            strategy=CacheStrategy.QUANTUM_PRIORITY
        )
    
    def test_initialization(self):
        """Test cache manager initialization."""
        cache = QuantumCacheManager(
            max_size_mb=50,
            max_entries=500,
            strategy=CacheStrategy.LRU,
            ttl_seconds=3600
        )
        
        assert cache.max_size_bytes == 50 * 1024 * 1024
        assert cache.max_entries == 500
        assert cache.strategy == CacheStrategy.LRU
        assert cache.ttl_seconds == 3600
        assert len(cache.cache) == 0
    
    def test_put_and_get_basic(self):
        """Test basic put and get operations."""
        key = "test_key"
        value = {"data": "test_data", "number": 42}
        
        # Put value
        success = self.cache.put(key, value, CacheType.COMPUTATION_ARTIFACT)
        assert success
        
        # Get value
        retrieved = self.cache.get(key)
        assert retrieved == value
        
        # Verify statistics
        assert self.cache.stats.hits > 0
        assert len(self.cache.cache) == 1
    
    def test_put_with_quantum_priority(self):
        """Test put operation with quantum priority."""
        key = "priority_test"
        value = "test_value"
        quantum_priority = 0.9
        
        success = self.cache.put(
            key, value, 
            cache_type=CacheType.OPTIMIZATION_RESULT,
            quantum_priority=quantum_priority
        )
        
        assert success
        assert key in self.cache.cache
        assert self.cache.cache[key].quantum_priority == quantum_priority
        assert self.cache.quantum_weights[key] == quantum_priority
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        result = self.cache.get("nonexistent_key")
        
        assert result is None
        assert self.cache.stats.misses > 0
    
    def test_exists(self):
        """Test exists functionality."""
        key = "exists_test"
        value = "test_value"
        
        assert not self.cache.exists(key)
        
        self.cache.put(key, value)
        assert self.cache.exists(key)
    
    def test_invalidate(self):
        """Test cache invalidation."""
        key = "invalidate_test"
        value = "test_value"
        
        self.cache.put(key, value)
        assert self.cache.exists(key)
        
        success = self.cache.invalidate(key)
        assert success
        assert not self.cache.exists(key)
        
        # Try to invalidate non-existent key
        success = self.cache.invalidate("nonexistent")
        assert not success
    
    def test_invalidate_type(self):
        """Test type-based invalidation."""
        # Put entries of different types
        self.cache.put("opt1", "value1", CacheType.OPTIMIZATION_RESULT)
        self.cache.put("opt2", "value2", CacheType.OPTIMIZATION_RESULT)
        self.cache.put("quantum1", "value3", CacheType.QUANTUM_STATE)
        
        assert len(self.cache.cache) == 3
        
        # Invalidate optimization results
        removed_count = self.cache.invalidate_type(CacheType.OPTIMIZATION_RESULT)
        
        assert removed_count == 2
        assert len(self.cache.cache) == 1
        assert self.cache.exists("quantum1")
        assert not self.cache.exists("opt1")
        assert not self.cache.exists("opt2")
    
    def test_clear(self):
        """Test cache clearing."""
        # Add some entries
        for i in range(5):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        assert len(self.cache.cache) > 0
        
        self.cache.clear()
        
        assert len(self.cache.cache) == 0
        assert self.cache.stats.entry_count == 0
        assert self.cache.stats.total_size_bytes == 0
    
    def test_capacity_eviction(self):
        """Test eviction when capacity is exceeded."""
        # Set small cache for testing
        small_cache = QuantumCacheManager(
            max_size_mb=1,  # 1MB limit
            max_entries=3,
            strategy=CacheStrategy.LRU
        )
        
        # Fill cache to capacity
        for i in range(5):  # More than max_entries
            large_value = "x" * 1000  # 1KB value
            small_cache.put(f"key_{i}", large_value)
        
        # Should have evicted some entries
        assert len(small_cache.cache) <= 3
        assert small_cache.stats.evictions > 0
    
    def test_lru_eviction_strategy(self):
        """Test LRU eviction strategy."""
        lru_cache = QuantumCacheManager(
            max_entries=2,
            strategy=CacheStrategy.LRU
        )
        
        # Add entries
        lru_cache.put("first", "value1")
        lru_cache.put("second", "value2")
        lru_cache.put("third", "value3")  # Should evict "first"
        
        assert not lru_cache.exists("first")
        assert lru_cache.exists("second")
        assert lru_cache.exists("third")
    
    def test_quantum_priority_calculation(self):
        """Test quantum priority calculation."""
        # Test different cache types
        priorities = {}
        
        for cache_type in CacheType:
            priority = self.cache._calculate_quantum_priority(
                "test_key", "test_value", cache_type
            )
            priorities[cache_type] = priority
            assert 0.0 <= priority <= 1.0
        
        # Optimization results should have higher priority than metrics
        assert priorities[CacheType.OPTIMIZATION_RESULT] > priorities[CacheType.METRICS]
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        ttl_cache = QuantumCacheManager(ttl_seconds=0.1)  # 100ms TTL
        
        ttl_cache.put("expire_test", "value")
        assert ttl_cache.exists("expire_test")
        
        # Wait for expiration
        time.sleep(0.15)
        
        assert not ttl_cache.exists("expire_test")
        result = ttl_cache.get("expire_test")
        assert result is None
    
    def test_periodic_cleanup(self):
        """Test periodic cleanup functionality."""
        # Add expired entry
        ttl_cache = QuantumCacheManager(ttl_seconds=0.1)
        ttl_cache.put("cleanup_test", "value")
        
        time.sleep(0.15)  # Wait for expiration
        
        removed_count = ttl_cache.periodic_cleanup()
        assert removed_count == 1
        assert not ttl_cache.exists("cleanup_test")
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        # Perform various operations
        self.cache.put("stat_test1", "value1")
        self.cache.put("stat_test2", "value2")
        self.cache.get("stat_test1")  # Hit
        self.cache.get("stat_test1")  # Hit
        self.cache.get("nonexistent")  # Miss
        
        stats = self.cache.get_statistics()
        
        assert stats["hits"] >= 2
        assert stats["misses"] >= 1
        assert stats["hit_rate"] > 0
        assert stats["entry_count"] >= 2
        assert "quantum_coherence" in stats
    
    def test_precompute_common_queries(self):
        """Test precomputation of common queries."""
        patterns = ["pattern1", "pattern2", "pattern3"]
        
        precomputed_count = self.cache.precompute_common_queries(patterns)
        
        assert precomputed_count == len(patterns)
        
        # Check that precomputed entries exist
        for pattern in patterns:
            import hashlib
            cache_key = f"precomputed_{hashlib.md5(pattern.encode()).hexdigest()}"
            assert self.cache.exists(cache_key)


class TestParallelQuantumExecutor:
    """Test suite for ParallelQuantumExecutor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ParallelQuantumExecutor(
            min_workers=2,
            max_workers=4,
            execution_mode=ExecutionMode.THREAD_BASED,
            auto_scaling=False  # Disable for predictable testing
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'executor') and self.executor.is_running:
            self.executor.stop()
    
    def test_initialization(self):
        """Test executor initialization."""
        executor = ParallelQuantumExecutor(
            min_workers=3,
            max_workers=8,
            execution_mode=ExecutionMode.PROCESS_BASED,
            auto_scaling=True
        )
        
        assert executor.min_workers == 3
        assert executor.max_workers == 8
        assert executor.execution_mode == ExecutionMode.PROCESS_BASED
        assert executor.auto_scaling == True
        assert not executor.is_running
    
    def test_start_stop(self):
        """Test executor start and stop."""
        assert not self.executor.is_running
        
        self.executor.start()
        assert self.executor.is_running
        
        self.executor.stop()
        assert not self.executor.is_running
    
    def test_submit_task(self):
        """Test task submission."""
        self.executor.start()
        
        def test_function(x, y=1):
            return x + y
        
        task_id = self.executor.submit_task(
            task_id="test_task",
            function=test_function,
            args=(5,),
            kwargs={"y": 3},
            priority=0.8
        )
        
        assert task_id == "test_task"
        assert self.executor.metrics.total_tasks == 1
        assert self.executor.metrics.queue_size > 0
        
        # Wait for completion and get result
        result = self.executor.get_task_result("test_task", timeout=5.0)
        assert result == 8  # 5 + 3
    
    def test_submit_planning_task(self):
        """Test quantum planning task submission."""
        self.executor.start()
        
        planner = QuantumTaskPlanner(max_iterations=10)
        tasks = [
            Task("plan_task1", "Task 1", TaskPriority.HIGH, 1.0, [], {"cpu": 1.0}),
            Task("plan_task2", "Task 2", TaskPriority.MEDIUM, 2.0, [], {"memory": 512.0})
        ]
        
        task_id = self.executor.submit_planning_task(planner, tasks, "planning_test")
        
        assert task_id == "planning_test"
        
        # Wait for completion
        result = self.executor.get_task_result(task_id, timeout=10.0)
        
        assert isinstance(result, dict)
        assert "schedule" in result
        assert "energy" in result
    
    def test_task_status_tracking(self):
        """Test task status tracking."""
        self.executor.start()
        
        def slow_function():
            time.sleep(0.1)
            return "completed"
        
        task_id = self.executor.submit_task("slow_task", slow_function)
        
        # Initially pending or running
        initial_status = self.executor.get_task_status(task_id)
        assert initial_status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        
        # Wait for completion
        result = self.executor.get_task_result(task_id, timeout=5.0)
        assert result == "completed"
        
        final_status = self.executor.get_task_status(task_id)
        assert final_status == TaskStatus.COMPLETED
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        self.executor.start()
        
        def long_function():
            time.sleep(10)
            return "should_not_complete"
        
        task_id = self.executor.submit_task("cancel_task", long_function)
        
        # Cancel the task
        success = self.executor.cancel_task(task_id)
        assert success
        
        status = self.executor.get_task_status(task_id)
        assert status == TaskStatus.CANCELLED
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        self.executor.start()
        
        def simple_function(value):
            return value * 2
        
        # Submit multiple tasks
        for i in range(3):
            self.executor.submit_task(f"metrics_task_{i}", simple_function, args=(i,))
        
        # Wait for completion
        for i in range(3):
            self.executor.get_task_result(f"metrics_task_{i}", timeout=5.0)
        
        metrics = self.executor.get_metrics()
        
        assert "execution_metrics" in metrics
        assert "load_metrics" in metrics
        assert "system_info" in metrics
        
        exec_metrics = metrics["execution_metrics"]
        assert exec_metrics["total_tasks"] == 3
        assert exec_metrics["completed_tasks"] == 3
        assert exec_metrics["success_rate"] == 1.0
    
    def test_worker_pool_scaling(self):
        """Test worker pool scaling."""
        worker_pool = WorkerPool(min_workers=2, max_workers=6)
        worker_pool.start()
        
        try:
            # Test scaling up
            success = worker_pool.scale_workers(4)
            assert success
            assert worker_pool.current_workers == 4
            
            # Test scaling down
            success = worker_pool.scale_workers(3)
            assert success
            assert worker_pool.current_workers == 3
            
            # Test invalid scaling
            success = worker_pool.scale_workers(10)  # Above max
            assert success
            assert worker_pool.current_workers == 6  # Clamped to max
            
        finally:
            worker_pool.stop()
    
    def test_execution_task_lifecycle(self):
        """Test ExecutionTask lifecycle."""
        def test_func(x):
            return x * x
        
        task = ExecutionTask(
            id="lifecycle_test",
            function=test_func,
            args=(5,),
            kwargs={},
            priority=1.0,
            timeout=10.0
        )
        
        assert task.status == TaskStatus.PENDING
        assert task.execution_time is None
        
        # Simulate execution
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        time.sleep(0.01)
        task.completed_at = time.time()
        task.status = TaskStatus.COMPLETED
        task.result = 25
        
        assert task.execution_time is not None
        assert task.execution_time > 0
        assert task.result == 25


class TestResourcePool:
    """Test suite for ResourcePool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pool = ResourcePool(
            pool_name="test_pool",
            enable_prediction=False,
            enable_auto_scaling=False
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'pool') and self.pool.is_running:
            self.pool.stop()
    
    def test_initialization(self):
        """Test resource pool initialization."""
        pool = ResourcePool("custom_pool", enable_prediction=True, enable_auto_scaling=True)
        
        assert pool.pool_name == "custom_pool"
        assert pool.enable_prediction == True
        assert pool.enable_auto_scaling == True
        assert not pool.is_running
        assert len(pool.resources) == 0
    
    def test_add_remove_resource(self):
        """Test adding and removing resources."""
        # Add resource
        success = self.pool.add_resource(
            resource_id="cpu_1",
            resource_type=ResourceType.COMPUTE,
            capacity=8.0,
            metadata={"cores": 8}
        )
        
        assert success
        assert "cpu_1" in self.pool.resources
        assert len(self.pool.resource_types[ResourceType.COMPUTE]) == 1
        assert self.pool.statistics.total_resources == 1
        
        # Try to add duplicate
        success = self.pool.add_resource("cpu_1", ResourceType.COMPUTE, 4.0)
        assert not success  # Should fail
        
        # Remove resource
        success = self.pool.remove_resource("cpu_1")
        assert success
        assert "cpu_1" not in self.pool.resources
        assert len(self.pool.resource_types[ResourceType.COMPUTE]) == 0
        
        # Try to remove non-existent
        success = self.pool.remove_resource("nonexistent")
        assert not success
    
    def test_resource_allocation(self):
        """Test resource allocation."""
        # Add resources
        self.pool.add_resource("cpu_1", ResourceType.COMPUTE, 8.0)
        self.pool.add_resource("mem_1", ResourceType.MEMORY, 16.0)
        
        # Create allocation request
        request = AllocationRequest(
            id="alloc_1",
            resource_type=ResourceType.COMPUTE,
            amount=4.0,
            priority=0.8,
            requester="test_user"
        )
        
        # Allocate resources
        result = self.pool.allocate_resources(request)
        
        assert result.success
        assert result.allocated_amount == 4.0
        assert len(result.allocated_resources) == 1
        assert "cpu_1" in result.allocated_resources
        
        # Check resource state
        cpu_resource = self.pool.resources["cpu_1"]
        assert cpu_resource.allocated == 4.0
        assert cpu_resource.available_capacity == 4.0
    
    def test_resource_deallocation(self):
        """Test resource deallocation."""
        # Add resource and allocate
        self.pool.add_resource("cpu_1", ResourceType.COMPUTE, 8.0)
        
        request = AllocationRequest("alloc_1", ResourceType.COMPUTE, 6.0)
        result = self.pool.allocate_resources(request)
        assert result.success
        
        # Check allocation
        cpu_resource = self.pool.resources["cpu_1"]
        assert cpu_resource.allocated == 6.0
        
        # Deallocate
        success = self.pool.deallocate_resources("alloc_1")
        assert success
        
        # Check deallocation
        assert cpu_resource.allocated == 0.0
        assert cpu_resource.available_capacity == 8.0
    
    def test_insufficient_resources(self):
        """Test allocation with insufficient resources."""
        # Add small resource
        self.pool.add_resource("small_cpu", ResourceType.COMPUTE, 2.0)
        
        # Request more than available
        request = AllocationRequest("big_alloc", ResourceType.COMPUTE, 5.0)
        result = self.pool.allocate_resources(request)
        
        assert not result.success
        assert "No suitable resources available" in result.error_message
    
    def test_quantum_allocation_algorithm(self):
        """Test quantum-inspired allocation algorithm."""
        # Add multiple resources with different quantum weights
        self.pool.add_resource("cpu_high", ResourceType.COMPUTE, 8.0)
        self.pool.add_resource("cpu_low", ResourceType.COMPUTE, 4.0)
        
        # Artificially set different quantum weights
        self.pool.quantum_weights["cpu_high"] = 0.9
        self.pool.quantum_weights["cpu_low"] = 0.3
        
        # Make multiple allocation requests
        allocations = []
        for i in range(5):
            request = AllocationRequest(f"alloc_{i}", ResourceType.COMPUTE, 2.0)
            result = self.pool.allocate_resources(request)
            if result.success:
                allocations.append(result)
        
        # Should prefer high-weight resource
        high_allocations = sum(1 for alloc in allocations if "cpu_high" in alloc.allocated_resources)
        assert high_allocations >= 1  # Should get some allocations on high-weight resource
    
    def test_resource_health_checking(self):
        """Test resource health checking."""
        self.pool.start()
        
        try:
            # Add resource
            self.pool.add_resource("test_cpu", ResourceType.COMPUTE, 4.0)
            
            # Simulate stuck allocation
            cpu_resource = self.pool.resources["test_cpu"]
            cpu_resource.allocated = 4.0
            cpu_resource.status = ResourceStatus.ALLOCATED
            cpu_resource.last_used = time.time() - 7200  # 2 hours ago
            
            # Trigger health check
            self.pool._check_resource_health()
            
            # Resource should be marked as failed
            assert cpu_resource.status == ResourceStatus.FAILED
            
        finally:
            self.pool.stop()
    
    def test_pool_statistics(self):
        """Test pool statistics calculation."""
        # Add resources and perform allocations
        self.pool.add_resource("cpu_1", ResourceType.COMPUTE, 8.0)
        self.pool.add_resource("mem_1", ResourceType.MEMORY, 16.0)
        
        # Perform some allocations
        request1 = AllocationRequest("alloc_1", ResourceType.COMPUTE, 4.0)
        request2 = AllocationRequest("alloc_2", ResourceType.MEMORY, 8.0)
        
        result1 = self.pool.allocate_resources(request1)
        result2 = self.pool.allocate_resources(request2)
        
        assert result1.success
        assert result2.success
        
        # Get status
        status = self.pool.get_pool_status()
        
        assert status["pool_name"] == "test_pool"
        assert status["statistics"]["total_resources"] == 2
        assert status["statistics"]["allocated_resources"] == 2
        assert status["statistics"]["utilization_ratio"] > 0
        assert "type_breakdown" in status
        assert ResourceType.COMPUTE.value in status["type_breakdown"]
        assert ResourceType.MEMORY.value in status["type_breakdown"]


class TestQuantumLoadBalancer:
    """Test suite for QuantumLoadBalancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.balancer = QuantumLoadBalancer(
            strategy=LoadBalancingStrategy.QUANTUM_SUPERPOSITION,
            health_check_interval=1.0,  # Fast for testing
            enable_circuit_breaker=True
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'balancer') and self.balancer.is_running:
            self.balancer.stop()
    
    def test_initialization(self):
        """Test load balancer initialization."""
        balancer = QuantumLoadBalancer(
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            health_check_interval=30.0,
            enable_circuit_breaker=False
        )
        
        assert balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert balancer.health_check_interval == 30.0
        assert balancer.enable_circuit_breaker == False
        assert not balancer.is_running
        assert len(balancer.nodes) == 0
    
    def test_add_remove_node(self):
        """Test adding and removing nodes."""
        # Add node
        success = self.balancer.add_node(
            node_id="node_1",
            endpoint="http://localhost:8001",
            weight=1.0,
            max_connections=100,
            metadata={"region": "us-east"}
        )
        
        assert success
        assert "node_1" in self.balancer.nodes
        assert len(self.balancer.node_order) == 1
        assert self.balancer.stats.node_count == 1
        
        # Try to add duplicate
        success = self.balancer.add_node("node_1", "http://localhost:8002")
        assert not success  # Should fail
        
        # Remove node
        success = self.balancer.remove_node("node_1")
        assert success
        assert "node_1" not in self.balancer.nodes
        assert len(self.balancer.node_order) == 0
        
        # Try to remove non-existent
        success = self.balancer.remove_node("nonexistent")
        assert not success
    
    def test_route_request_no_nodes(self):
        """Test request routing with no available nodes."""
        result = self.balancer.route_request("req_1", {"data": "test"})
        
        assert result is None
    
    def test_round_robin_routing(self):
        """Test round-robin routing strategy."""
        self.balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Add nodes
        self.balancer.add_node("node_1", "http://node1:8000")
        self.balancer.add_node("node_2", "http://node2:8000")
        self.balancer.add_node("node_3", "http://node3:8000")
        
        # Route requests
        routes = []
        for i in range(6):
            node_id = self.balancer.route_request(f"req_{i}")
            routes.append(node_id)
        
        # Should cycle through nodes
        assert routes[0] != routes[1] or routes[1] != routes[2]  # Should distribute
        assert len(set(routes)) >= 2  # Should use multiple nodes
    
    def test_weighted_round_robin_routing(self):
        """Test weighted round-robin routing."""
        self.balancer.strategy = LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
        
        # Add nodes with different weights
        self.balancer.add_node("heavy", "http://heavy:8000", weight=3.0)
        self.balancer.add_node("light", "http://light:8000", weight=1.0)
        
        # Route many requests
        routes = []
        for i in range(20):
            node_id = self.balancer.route_request(f"req_{i}")
            routes.append(node_id)
        
        # Heavy node should get more requests
        heavy_count = routes.count("heavy")
        light_count = routes.count("light")
        assert heavy_count > light_count
    
    def test_least_connections_routing(self):
        """Test least connections routing."""
        self.balancer.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        
        # Add nodes
        self.balancer.add_node("node_1", "http://node1:8000")
        self.balancer.add_node("node_2", "http://node2:8000")
        
        # Artificially set different connection counts
        self.balancer.nodes["node_1"].current_connections = 5
        self.balancer.nodes["node_2"].current_connections = 2
        
        # Route request - should go to node with fewer connections
        node_id = self.balancer.route_request("req_1")
        assert node_id == "node_2"
    
    def test_quantum_superposition_routing(self):
        """Test quantum superposition routing."""
        self.balancer.strategy = LoadBalancingStrategy.QUANTUM_SUPERPOSITION
        
        # Add nodes with different quantum scores
        self.balancer.add_node("good", "http://good:8000")
        self.balancer.add_node("bad", "http://bad:8000")
        
        # Set different quantum scores
        self.balancer.nodes["good"].quantum_score = 0.9
        self.balancer.nodes["bad"].quantum_score = 0.1
        
        # Route many requests
        routes = []
        for i in range(20):
            node_id = self.balancer.route_request(f"req_{i}")
            routes.append(node_id)
        
        # Good node should get more requests (probabilistically)
        good_count = routes.count("good")
        assert good_count > 5  # Should get majority of requests
    
    def test_request_completion(self):
        """Test request completion tracking."""
        # Add node and route request
        self.balancer.add_node("node_1", "http://node1:8000")
        node_id = self.balancer.route_request("req_1")
        
        assert node_id == "node_1"
        assert self.balancer.nodes["node_1"].current_connections == 1
        
        # Complete request successfully
        self.balancer.complete_request("req_1", success=True, response_time=0.5)
        
        # Check updates
        node = self.balancer.nodes["node_1"]
        assert node.current_connections == 0
        assert node.successful_requests == 1
        assert node.last_response_time == 0.5
        assert self.balancer.stats.successful_requests == 1
    
    def test_request_completion_failure(self):
        """Test failed request completion."""
        # Add node and route request
        self.balancer.add_node("node_1", "http://node1:8000")
        self.balancer.route_request("req_1")
        
        # Complete request with failure
        self.balancer.complete_request("req_1", success=False, error="Connection timeout")
        
        # Check updates
        node = self.balancer.nodes["node_1"]
        assert node.failed_requests == 1
        assert node.success_rate < 1.0
        assert self.balancer.stats.failed_requests == 1
    
    def test_node_health_monitoring(self):
        """Test node health monitoring."""
        self.balancer.start()
        
        try:
            # Add node
            self.balancer.add_node("node_1", "http://node1:8000")
            
            # Simulate poor performance
            node = self.balancer.nodes["node_1"]
            node.successful_requests = 10
            node.failed_requests = 40  # 20% success rate
            node.avg_response_time = 6.0  # Slow
            
            # Trigger health check
            self.balancer._perform_health_checks()
            
            # Node should be marked as failed or degraded
            assert node.status in [NodeStatus.FAILED, NodeStatus.DEGRADED]
            
        finally:
            self.balancer.stop()
    
    def test_quantum_state_updates(self):
        """Test quantum state vector updates."""
        # Add nodes
        self.balancer.add_node("node_1", "http://node1:8000")
        self.balancer.add_node("node_2", "http://node2:8000")
        
        # Should update quantum state
        assert len(self.balancer.quantum_state_vector) == 2
        assert abs(np.sum(self.balancer.quantum_state_vector ** 2) - 1.0) < 1e-10  # Normalized
        
        # Remove node
        self.balancer.remove_node("node_2")
        
        # Quantum state should update
        assert len(self.balancer.quantum_state_vector) == 1
    
    def test_coherence_calculation(self):
        """Test quantum coherence calculation."""
        # Single node - perfect coherence
        self.balancer.add_node("single", "http://single:8000")
        coherence = self.balancer._calculate_coherence()
        assert coherence == 1.0
        
        # Multiple nodes - measure coherence
        self.balancer.add_node("node_2", "http://node2:8000")
        self.balancer.add_node("node_3", "http://node3:8000")
        
        coherence = self.balancer._calculate_coherence()
        assert 0.0 <= coherence <= 1.0
    
    def test_load_balancer_status(self):
        """Test load balancer status reporting."""
        # Add nodes and simulate some activity
        self.balancer.add_node("node_1", "http://node1:8000")
        self.balancer.add_node("node_2", "http://node2:8000")
        
        # Route and complete some requests
        for i in range(5):
            node_id = self.balancer.route_request(f"req_{i}")
            self.balancer.complete_request(f"req_{i}", success=True, response_time=0.1)
        
        status = self.balancer.get_load_balancer_status()
        
        assert status["strategy"] == LoadBalancingStrategy.QUANTUM_SUPERPOSITION.value
        assert status["statistics"]["total_requests"] == 5
        assert status["statistics"]["success_rate"] == 1.0
        assert status["statistics"]["node_count"] == 2
        assert "node_details" in status
        assert "quantum_metrics" in status
        assert "coherence_factor" in status["quantum_metrics"]


class TestIntegration:
    """Integration tests for optimization components."""
    
    def test_cache_with_parallel_execution(self):
        """Test cache integration with parallel execution."""
        cache = QuantumCacheManager(max_size_mb=5, strategy=CacheStrategy.QUANTUM_PRIORITY)
        executor = ParallelQuantumExecutor(min_workers=2, max_workers=4, auto_scaling=False)
        
        executor.start()
        
        try:
            def cached_computation(x):
                # Check cache first
                cache_key = f"computation_{x}"
                result = cache.get(cache_key)
                if result is not None:
                    return result
                
                # Compute and cache
                computed_result = x ** 2 + x
                cache.put(cache_key, computed_result, CacheType.COMPUTATION_ARTIFACT)
                return computed_result
            
            # Submit multiple tasks
            task_ids = []
            for i in range(5):
                task_id = executor.submit_task(f"cached_task_{i}", cached_computation, args=(i,))
                task_ids.append(task_id)
            
            # Get results
            results = []
            for task_id in task_ids:
                result = executor.get_task_result(task_id, timeout=5.0)
                results.append(result)
            
            # Verify results
            expected = [i**2 + i for i in range(5)]
            assert results == expected
            
            # Verify caching worked
            assert len(cache.cache) == 5
            for i in range(5):
                cached_value = cache.get(f"computation_{i}")
                assert cached_value == i**2 + i
                
        finally:
            executor.stop()
    
    def test_resource_pool_with_load_balancer(self):
        """Test resource pool integration with load balancer."""
        pool = ResourcePool("integration_pool")
        balancer = QuantumLoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # Add resources to pool
        pool.add_resource("cpu_1", ResourceType.COMPUTE, 8.0)
        pool.add_resource("cpu_2", ResourceType.COMPUTE, 4.0)
        
        # Add nodes to balancer
        balancer.add_node("worker_1", "http://worker1:8000", max_connections=50)
        balancer.add_node("worker_2", "http://worker2:8000", max_connections=30)
        
        # Simulate coordinated resource allocation and load balancing
        allocations = []
        routes = []
        
        for i in range(5):
            # Allocate resources
            request = AllocationRequest(f"req_{i}", ResourceType.COMPUTE, 2.0)
            alloc_result = pool.allocate_resources(request)
            
            if alloc_result.success:
                allocations.append(alloc_result)
                
                # Route request to worker
                node_id = balancer.route_request(f"req_{i}")
                if node_id:
                    routes.append(node_id)
                    # Simulate successful completion
                    balancer.complete_request(f"req_{i}", success=True, response_time=0.2)
        
        # Verify coordinated behavior
        assert len(allocations) > 0
        assert len(routes) == len(allocations)
        
        # Check resource utilization
        pool_status = pool.get_pool_status()
        assert pool_status["statistics"]["utilization_ratio"] > 0
        
        # Check load balancer metrics
        balancer_status = balancer.get_load_balancer_status()
        assert balancer_status["statistics"]["total_requests"] == len(routes)
        assert balancer_status["statistics"]["success_rate"] == 1.0
        
        # Clean up
        for alloc in allocations:
            pool.deallocate_resources(alloc.request_id)
    
    def test_full_optimization_stack(self):
        """Test full optimization stack integration."""
        # Initialize all components
        cache = QuantumCacheManager(max_size_mb=10)
        executor = ParallelQuantumExecutor(min_workers=2, max_workers=4, auto_scaling=False)
        pool = ResourcePool("full_stack_pool")
        balancer = QuantumLoadBalancer()
        
        # Set up resources
        pool.add_resource("cpu", ResourceType.COMPUTE, 16.0)
        pool.add_resource("memory", ResourceType.MEMORY, 32.0)
        
        # Set up load balancer nodes
        balancer.add_node("node_1", "http://node1:8000")
        balancer.add_node("node_2", "http://node2:8000")
        
        executor.start()
        
        try:
            def optimized_task(task_id, computation_size):
                # 1. Check cache
                cache_key = f"task_result_{task_id}_{computation_size}"
                cached_result = cache.get(cache_key)
                if cached_result:
                    return {"result": cached_result, "from_cache": True}
                
                # 2. Allocate resources
                cpu_request = AllocationRequest(f"cpu_{task_id}", ResourceType.COMPUTE, computation_size)
                cpu_alloc = pool.allocate_resources(cpu_request)
                
                if not cpu_alloc.success:
                    return {"error": "Resource allocation failed"}
                
                # 3. Route to worker
                worker_node = balancer.route_request(f"work_{task_id}")
                if not worker_node:
                    pool.deallocate_resources(f"cpu_{task_id}")
                    return {"error": "No available workers"}
                
                try:
                    # 4. Simulate computation
                    result = computation_size ** 2
                    time.sleep(0.01)  # Simulate work
                    
                    # 5. Cache result
                    cache.put(cache_key, result, CacheType.COMPUTATION_ARTIFACT)
                    
                    # 6. Complete request
                    balancer.complete_request(f"work_{task_id}", success=True, response_time=0.01)
                    
                    return {"result": result, "from_cache": False, "worker": worker_node}
                    
                finally:
                    # 7. Deallocate resources
                    pool.deallocate_resources(f"cpu_{task_id}")
            
            # Submit tasks to the optimization stack
            task_futures = []
            for i in range(3):
                task_id = executor.submit_task(
                    f"stack_task_{i}", 
                    optimized_task, 
                    args=(i, 2.0)
                )
                task_futures.append(task_id)
            
            # Collect results
            results = []
            for task_id in task_futures:
                result = executor.get_task_result(task_id, timeout=10.0)
                results.append(result)
            
            # Verify full stack worked
            assert len(results) == 3
            assert all("result" in r for r in results)
            
            # Verify caching worked (submit same tasks again)
            cached_futures = []
            for i in range(3):
                task_id = executor.submit_task(
                    f"cached_stack_task_{i}", 
                    optimized_task, 
                    args=(i, 2.0)
                )
                cached_futures.append(task_id)
            
            cached_results = []
            for task_id in cached_futures:
                result = executor.get_task_result(task_id, timeout=10.0)
                cached_results.append(result)
            
            # At least some should come from cache
            from_cache_count = sum(1 for r in cached_results if r.get("from_cache", False))
            assert from_cache_count > 0
            
        finally:
            executor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])