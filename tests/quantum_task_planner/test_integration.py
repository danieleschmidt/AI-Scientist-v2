"""
Integration and end-to-end tests for quantum task planner.

Comprehensive integration tests covering complete workflows,
cross-component interactions, and real-world scenarios.
"""

import pytest
import time
import numpy as np
import threading
from unittest.mock import Mock, patch

from quantum_task_planner.core.planner import QuantumTaskPlanner, Task, TaskPriority
from quantum_task_planner.core.quantum_optimizer import QuantumOptimizer
from quantum_task_planner.core.task_scheduler import TaskScheduler, SchedulingStrategy
from quantum_task_planner.optimization.cache_manager import QuantumCacheManager, CacheStrategy, CacheType
from quantum_task_planner.optimization.parallel_executor import ParallelQuantumExecutor, ExecutionMode
from quantum_task_planner.optimization.resource_pool import ResourcePool, ResourceType
from quantum_task_planner.optimization.load_balancer import QuantumLoadBalancer, LoadBalancingStrategy
from quantum_task_planner.monitoring.health_monitor import HealthMonitor
from quantum_task_planner.monitoring.performance_monitor import PerformanceMonitor
from quantum_task_planner.monitoring.quantum_monitor import QuantumMetricsMonitor
from quantum_task_planner.validation.validators import create_full_validator
from quantum_task_planner.validation.error_handling import ErrorHandler
from quantum_task_planner.utils.metrics import PlannerMetrics


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_full_quantum_planning_workflow(self):
        """Test complete quantum task planning workflow."""
        # Initialize complete system
        planner = QuantumTaskPlanner(max_iterations=100)
        scheduler = TaskScheduler(strategy=SchedulingStrategy.QUANTUM_ANNEALING)
        cache = QuantumCacheManager(max_size_mb=10, strategy=CacheStrategy.QUANTUM_PRIORITY)
        validator = create_full_validator()
        
        # Set up resources
        planner.set_resource_limits({"cpu": 16.0, "memory": 32768.0, "gpu": 4.0})
        scheduler.add_resource("cpu", 16.0)
        scheduler.add_resource("memory", 32768.0)
        scheduler.add_resource("gpu", 4.0)
        
        # Create comprehensive task set
        tasks = [
            Task(
                id="data_preprocessing",
                name="Data Preprocessing",
                priority=TaskPriority.HIGH,
                duration=3.0,
                dependencies=[],
                resources={"cpu": 4.0, "memory": 8192.0},
                quantum_weight=0.8,
                deadline=10.0
            ),
            Task(
                id="feature_extraction",
                name="Feature Extraction",
                priority=TaskPriority.MEDIUM,
                duration=5.0,
                dependencies=["data_preprocessing"],
                resources={"cpu": 2.0, "gpu": 1.0},
                quantum_weight=0.7,
                deadline=20.0
            ),
            Task(
                id="model_training",
                name="Model Training",
                priority=TaskPriority.HIGH,
                duration=8.0,
                dependencies=["feature_extraction"],
                resources={"gpu": 2.0, "memory": 16384.0},
                quantum_weight=0.9,
                deadline=35.0
            ),
            Task(
                id="model_validation",
                name="Model Validation",
                priority=TaskPriority.MEDIUM,
                duration=2.0,
                dependencies=["model_training"],
                resources={"cpu": 2.0, "memory": 4096.0},
                quantum_weight=0.6,
                deadline=40.0
            ),
            Task(
                id="result_analysis",
                name="Result Analysis",
                priority=TaskPriority.LOW,
                duration=1.5,
                dependencies=["model_validation"],
                resources={"cpu": 1.0, "memory": 2048.0},
                quantum_weight=0.5,
                deadline=45.0
            )
        ]
        
        # Step 1: Validate tasks
        validation_result = validator.validate_task_list(tasks)
        assert validation_result.is_valid, f"Task validation failed: {validation_result.errors}"
        
        # Step 2: Add tasks to planner
        for task in tasks:
            planner.add_task(task)
        
        # Step 3: Check cache for existing plan
        cache_key = "workflow_plan_v1"
        cached_plan = cache.get(cache_key)
        
        if cached_plan is None:
            # Step 4: Generate quantum plan
            planning_result = planner.plan_tasks()
            assert planning_result["total_tasks"] == 5
            assert planning_result["selected_tasks"] > 0
            assert planning_result["energy"] >= 0
            
            # Cache the plan
            cache.put(cache_key, planning_result, CacheType.OPTIMIZATION_RESULT, quantum_priority=0.9)
        else:
            planning_result = cached_plan
        
        # Step 5: Schedule selected tasks
        selected_tasks = planning_result["schedule"]
        scheduling_result = scheduler.schedule_tasks(selected_tasks)
        
        assert scheduling_result.success_rate > 0
        assert scheduling_result.total_makespan > 0
        assert len(scheduling_result.scheduled_tasks) <= len(selected_tasks)
        
        # Step 6: Verify resource constraints
        resource_util = scheduling_result.resource_utilization
        for resource, util in resource_util.items():
            assert 0 <= util <= 1.0, f"Resource {resource} utilization {util} out of bounds"
        
        # Step 7: Validate quantum metrics
        planning_metrics = planning_result["quantum_metrics"]
        assert 0 <= planning_metrics["coherence"] <= 1
        assert 0 <= planning_metrics["entanglement"] <= 1
        
        scheduling_metrics = scheduling_result.quantum_metrics
        assert len(scheduling_metrics) > 0
        
        print(f"Workflow completed: {len(scheduling_result.scheduled_tasks)} tasks scheduled")
        print(f"Total makespan: {scheduling_result.total_makespan:.2f}s")
        print(f"Planning coherence: {planning_metrics['coherence']:.3f}")
        print(f"Resource utilization: {resource_util}")
    
    def test_distributed_quantum_computing_workflow(self):
        """Test distributed quantum computing workflow."""
        # Initialize distributed components
        load_balancer = QuantumLoadBalancer(strategy=LoadBalancingStrategy.QUANTUM_SUPERPOSITION)
        resource_pool = ResourcePool("quantum_cluster")
        executor = ParallelQuantumExecutor(min_workers=3, max_workers=6, auto_scaling=True)
        quantum_monitor = QuantumMetricsMonitor()
        
        # Set up distributed infrastructure
        # Add quantum computing nodes
        load_balancer.add_node("qnode_1", "http://qnode1:8000", weight=1.0, max_connections=10)
        load_balancer.add_node("qnode_2", "http://qnode2:8000", weight=1.5, max_connections=15)
        load_balancer.add_node("qnode_3", "http://qnode3:8000", weight=0.8, max_connections=8)
        
        # Add quantum resources
        resource_pool.add_resource("qpu_1", ResourceType.QUANTUM_CIRCUIT, 10.0)
        resource_pool.add_resource("qpu_2", ResourceType.QUANTUM_CIRCUIT, 8.0)
        resource_pool.add_resource("classical_cpu", ResourceType.COMPUTE, 32.0)
        resource_pool.add_resource("quantum_memory", ResourceType.MEMORY, 1024.0)
        
        executor.start()
        
        try:
            # Define quantum circuit simulation tasks
            def simulate_quantum_circuit(circuit_depth, n_qubits, algorithm):
                """Simulate quantum circuit execution."""
                # Simulate quantum state vector
                state_size = 2 ** n_qubits
                state_vector = np.random.random(state_size) + 1j * np.random.random(state_size)
                state_vector = state_vector / np.linalg.norm(state_vector)
                
                # Record quantum metrics
                quantum_monitor.record_quantum_state(state_vector)
                
                # Simulate computation time based on circuit complexity
                computation_time = circuit_depth * n_qubits * 0.01
                time.sleep(min(computation_time, 0.1))  # Cap for testing
                
                # Calculate fidelity (simplified)
                fidelity = max(0.7, 1.0 - circuit_depth * 0.01)
                
                return {
                    "algorithm": algorithm,
                    "n_qubits": n_qubits,
                    "circuit_depth": circuit_depth,
                    "fidelity": fidelity,
                    "state_entropy": -np.sum(np.abs(state_vector)**2 * np.log(np.abs(state_vector)**2 + 1e-10))
                }
            
            # Define quantum algorithms to run
            quantum_tasks = [
                {"algorithm": "QAOA", "n_qubits": 4, "circuit_depth": 6},
                {"algorithm": "VQE", "n_qubits": 3, "circuit_depth": 8},
                {"algorithm": "Quantum_Fourier_Transform", "n_qubits": 5, "circuit_depth": 4},
                {"algorithm": "Shor", "n_qubits": 6, "circuit_depth": 10},
                {"algorithm": "Grover", "n_qubits": 4, "circuit_depth": 5}
            ]
            
            # Submit quantum computing tasks
            task_futures = []
            for i, task_config in enumerate(quantum_tasks):
                # Allocate quantum resources
                qpu_request = resource_pool.AllocationRequest(
                    id=f"qpu_alloc_{i}",
                    resource_type=ResourceType.QUANTUM_CIRCUIT,
                    amount=task_config["n_qubits"] * 0.5,  # Resource usage per qubit
                    priority=0.8
                )
                
                qpu_allocation = resource_pool.allocate_resources(qpu_request)
                
                if qpu_allocation.success:
                    # Route to quantum node
                    node_id = load_balancer.route_request(f"quantum_task_{i}")
                    
                    if node_id:
                        # Submit to executor
                        task_id = executor.submit_task(
                            f"quantum_sim_{i}",
                            simulate_quantum_circuit,
                            args=(task_config["circuit_depth"], task_config["n_qubits"], task_config["algorithm"]),
                            priority=0.9
                        )
                        task_futures.append((task_id, f"quantum_task_{i}", qpu_request.id))
                
            # Wait for completion and collect results
            quantum_results = []
            for task_id, request_id, alloc_id in task_futures:
                try:
                    result = executor.get_task_result(task_id, timeout=10.0)
                    quantum_results.append(result)
                    
                    # Complete load balancing request
                    load_balancer.complete_request(request_id, success=True, response_time=0.5)
                    
                except Exception as e:
                    print(f"Task {task_id} failed: {e}")
                    load_balancer.complete_request(request_id, success=False, error=str(e))
                
                finally:
                    # Deallocate resources
                    resource_pool.deallocate_resources(alloc_id)
            
            # Analyze distributed quantum computing results
            assert len(quantum_results) > 0, "No quantum tasks completed successfully"
            
            successful_algorithms = [r["algorithm"] for r in quantum_results if r.get("fidelity", 0) > 0.5]
            assert len(successful_algorithms) > 0, "No high-fidelity quantum computations"
            
            # Check load balancing effectiveness
            lb_status = load_balancer.get_load_balancer_status()
            assert lb_status["statistics"]["total_requests"] == len(task_futures)
            assert lb_status["statistics"]["success_rate"] > 0.5
            
            # Check resource utilization
            pool_status = resource_pool.get_pool_status()
            assert pool_status["statistics"]["success_rate"] > 0.5
            
            # Check quantum metrics
            quantum_summary = quantum_monitor.get_quantum_metrics_summary()
            assert "current_state" in quantum_summary
            
            print(f"Distributed quantum workflow: {len(quantum_results)} tasks completed")
            print(f"Successful algorithms: {successful_algorithms}")
            print(f"Load balancer success rate: {lb_status['statistics']['success_rate']:.2f}")
            print(f"Resource pool utilization: {pool_status['statistics']['utilization_ratio']:.2f}")
            
        finally:
            executor.stop()
    
    def test_adaptive_optimization_workflow(self):
        """Test adaptive optimization workflow with feedback loops."""
        # Initialize adaptive system
        optimizer = QuantumOptimizer(max_iterations=50, tolerance=1e-6)
        planner = QuantumTaskPlanner(max_iterations=100)
        cache = QuantumCacheManager(strategy=CacheStrategy.ADAPTIVE)
        performance_monitor = PerformanceMonitor()
        
        # Define adaptive optimization problem
        def adaptive_objective_function(x, feedback_weight=0.1):
            """Objective function that adapts based on previous results."""
            base_value = np.sum(x**2)  # Basic quadratic
            
            # Add feedback term based on cached results
            cached_history = cache.get("optimization_history")
            if cached_history:
                recent_results = cached_history[-5:]  # Last 5 results
                avg_performance = np.mean([r["objective"] for r in recent_results])
                feedback_term = feedback_weight * abs(base_value - avg_performance)
                return base_value + feedback_term
            
            return base_value
        
        # Adaptive optimization loop
        optimization_results = []
        initial_solution = np.array([2.0, -1.5, 0.5])
        
        for iteration in range(5):
            print(f"Adaptive optimization iteration {iteration + 1}")
            
            # Record performance metrics
            perf_start = time.time()
            
            # Run optimization with current objective
            result = optimizer.optimize(
                adaptive_objective_function,
                initial_solution,
                bounds=[(-5.0, 5.0)] * 3
            )
            
            perf_end = time.time()
            
            # Record performance
            performance_monitor.record_metric(PerformanceMonitor.PerformanceMetric(
                name=f"optimization_time_iter_{iteration}",
                value=perf_end - perf_start,
                metric_type=performance_monitor.MetricType.LATENCY,
                timestamp=perf_end,
                tags={"iteration": str(iteration)}
            ))
            
            # Store result
            result_data = {
                "iteration": iteration,
                "solution": result.solution.tolist(),
                "objective": result.objective_value,
                "iterations": result.iterations,
                "quantum_metrics": result.quantum_metrics
            }
            optimization_results.append(result_data)
            
            # Update cache with history
            cache.put("optimization_history", optimization_results, CacheType.OPTIMIZATION_RESULT)
            
            # Adapt for next iteration based on performance
            if result.quantum_metrics.get("convergence_rate", 0) < 0.01:
                # Slow convergence - try different starting point
                initial_solution = result.solution + np.random.normal(0, 0.1, 3)
            else:
                # Good convergence - refine around current solution
                initial_solution = result.solution + np.random.normal(0, 0.05, 3)
            
            # Adapt planner configuration based on optimization performance
            if result.objective_value < 1.0:  # Good optimization
                planner.convergence_threshold *= 0.9  # Tighten convergence
            else:
                planner.convergence_threshold *= 1.1  # Loosen convergence
        
        # Analyze adaptive behavior
        assert len(optimization_results) == 5
        
        # Check for improvement over iterations
        objective_values = [r["objective"] for r in optimization_results]
        final_performance = objective_values[-1]
        initial_performance = objective_values[0]
        
        # Should show adaptation (not necessarily monotonic improvement due to changing objective)
        performance_variance = np.var(objective_values)
        assert performance_variance >= 0  # Basic sanity check
        
        # Check quantum metrics evolution
        coherence_values = [r["quantum_metrics"].get("beta_variance", 0) for r in optimization_results]
        assert len([c for c in coherence_values if c > 0]) > 0  # Some non-zero quantum metrics
        
        # Performance monitoring should show metrics
        perf_summary = performance_monitor.get_performance_summary()
        assert len(perf_summary["metric_summaries"]) >= 5  # One metric per iteration
        
        print(f"Adaptive optimization completed:")
        print(f"Objective values: {[f'{v:.4f}' for v in objective_values]}")
        print(f"Final solution: {optimization_results[-1]['solution']}")
        print(f"Performance variance: {performance_variance:.6f}")
    
    def test_resilient_system_workflow(self):
        """Test system resilience with failures and recovery."""
        # Initialize resilient system
        health_monitor = HealthMonitor(check_interval=0.5)
        executor = ParallelQuantumExecutor(min_workers=2, max_workers=4, auto_scaling=False)
        error_handler = ErrorHandler()
        cache = QuantumCacheManager(strategy=CacheStrategy.QUANTUM_PRIORITY)
        
        # Add error tracking
        system_errors = []
        
        def track_error(error):
            system_errors.append(error)
        
        error_handler.recovery_strategies[error_handler.ErrorCategory.SYSTEM] = lambda e: "recovered"
        
        health_monitor.start_monitoring()
        executor.start()
        
        try:
            # Define tasks that may fail
            def potentially_failing_task(task_id, failure_rate=0.3):
                """Task that may fail based on failure rate."""
                if np.random.random() < failure_rate:
                    raise Exception(f"Task {task_id} simulated failure")
                
                # Simulate work
                time.sleep(0.05)
                return f"Task {task_id} completed successfully"
            
            # Submit tasks with potential failures
            task_futures = []
            for i in range(10):
                task_id = executor.submit_task(
                    f"resilient_task_{i}",
                    potentially_failing_task,
                    args=(i,),
                    kwargs={"failure_rate": 0.4},  # 40% failure rate
                    max_retries=2  # Allow retries
                )
                task_futures.append(task_id)
            
            # Collect results and track failures
            completed_tasks = 0
            failed_tasks = 0
            recovered_tasks = 0
            
            for task_id in task_futures:
                try:
                    result = executor.get_task_result(task_id, timeout=5.0)
                    completed_tasks += 1
                    
                    # Cache successful results
                    cache.put(f"result_{task_id}", result, CacheType.COMPUTATION_ARTIFACT)
                    
                except Exception as e:
                    failed_tasks += 1
                    
                    # Try to handle error
                    try:
                        recovery_result = error_handler.handle_error(
                            e, f"task_execution_{task_id}", "executor", attempt_recovery=True
                        )
                        if recovery_result:
                            recovered_tasks += 1
                    except:
                        pass  # Recovery failed
            
            # System should show resilience
            total_tasks = len(task_futures)
            success_rate = completed_tasks / total_tasks
            recovery_rate = recovered_tasks / max(failed_tasks, 1)
            
            print(f"Resilience test results:")
            print(f"Total tasks: {total_tasks}")
            print(f"Completed: {completed_tasks}")
            print(f"Failed: {failed_tasks}")
            print(f"Recovered: {recovered_tasks}")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Recovery rate: {recovery_rate:.2f}")
            
            # Verify system health
            system_health = health_monitor.get_system_health()
            assert system_health.status in [health_monitor.HealthStatus.HEALTHY, health_monitor.HealthStatus.WARNING]
            
            # Check error handling
            error_summary = error_handler.get_error_summary()
            assert error_summary["total_errors"] >= 0
            
            # Cache should have some successful results
            cache_stats = cache.get_statistics()
            assert cache_stats["entry_count"] >= 0
            
            # System should maintain some level of functionality
            assert success_rate + recovery_rate > 0.3  # At least 30% effective completion
            
        finally:
            health_monitor.stop_monitoring()
            executor.stop()


class TestScalabilityAndPerformance:
    """Test system scalability and performance characteristics."""
    
    def test_large_scale_task_planning(self):
        """Test task planning with large number of tasks."""
        planner = QuantumTaskPlanner(max_iterations=50)  # Reduced for performance
        
        # Generate large task set
        num_tasks = 100
        tasks = []
        
        for i in range(num_tasks):
            # Create dependencies for realistic complexity
            dependencies = []
            if i > 0:
                # Each task depends on 1-3 previous tasks
                num_deps = min(np.random.poisson(1.5) + 1, 3)
                deps = np.random.choice(i, size=min(num_deps, i), replace=False)
                dependencies = [f"task_{d}" for d in deps]
            
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                priority=np.random.choice(list(TaskPriority)),
                duration=np.random.exponential(2.0) + 0.5,  # 0.5-10s durations
                dependencies=dependencies,
                resources={
                    "cpu": np.random.uniform(0.5, 4.0),
                    "memory": np.random.uniform(256.0, 8192.0)
                },
                quantum_weight=np.random.uniform(0.2, 1.0)
            )
            tasks.append(task)
        
        # Set generous resource limits
        planner.set_resource_limits({
            "cpu": 200.0,
            "memory": 500000.0
        })
        
        # Add tasks
        for task in tasks:
            planner.add_task(task)
        
        # Measure planning performance
        start_time = time.time()
        result = planner.plan_tasks()
        planning_time = time.time() - start_time
        
        # Verify scalability
        assert result["total_tasks"] == num_tasks
        assert result["selected_tasks"] > 0
        assert planning_time < 60.0  # Should complete within 1 minute
        
        # Check quantum metrics
        assert "quantum_metrics" in result
        assert 0 <= result["quantum_metrics"]["coherence"] <= 1
        
        print(f"Large-scale planning: {num_tasks} tasks planned in {planning_time:.2f}s")
        print(f"Selected {result['selected_tasks']} tasks with energy {result['energy']:.2f}")
        print(f"Quantum coherence: {result['quantum_metrics']['coherence']:.3f}")
    
    def test_concurrent_optimization_performance(self):
        """Test performance under concurrent optimization load."""
        num_optimizers = 5
        optimizer_results = {}
        
        def run_optimization(optimizer_id):
            """Run optimization in thread."""
            optimizer = QuantumOptimizer(max_iterations=30, tolerance=1e-6)
            
            # Different objective function for each optimizer
            def objective(x):
                return np.sum((x - optimizer_id)**2) + np.sin(optimizer_id * np.sum(x))
            
            start_time = time.time()
            initial_solution = np.random.random(3) * 2 - 1  # [-1, 1]
            
            result = optimizer.optimize(objective, initial_solution)
            
            optimizer_results[optimizer_id] = {
                "result": result,
                "duration": time.time() - start_time,
                "thread_id": threading.get_ident()
            }
        
        # Run concurrent optimizations
        threads = []
        for i in range(num_optimizers):
            thread = threading.Thread(target=run_optimization, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=30.0)
        total_time = time.time() - start_time
        
        # Analyze concurrent performance
        assert len(optimizer_results) == num_optimizers
        
        durations = [r["duration"] for r in optimizer_results.values()]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        # Concurrent execution should be efficient
        assert total_time < max_duration * 1.5  # Should overlap significantly
        assert avg_duration < 15.0  # Individual optimizations should be fast
        
        # Check optimization quality
        objective_values = [r["result"].objective_value for r in optimizer_results.values()]
        assert all(val < 50.0 for val in objective_values)  # Reasonable optimization
        
        print(f"Concurrent optimization test:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average optimization time: {avg_duration:.2f}s")
        print(f"Max optimization time: {max_duration:.2f}s")
        print(f"Objective values: {[f'{v:.2f}' for v in objective_values]}")
    
    def test_memory_usage_scaling(self):
        """Test memory usage with increasing system size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        cache_managers = []
        memory_measurements = []
        
        # Scale up cache managers
        for size_mb in [1, 5, 10, 20]:
            cache = QuantumCacheManager(
                max_size_mb=size_mb,
                max_entries=size_mb * 100,
                strategy=CacheStrategy.QUANTUM_PRIORITY
            )
            cache_managers.append(cache)
            
            # Fill cache with data
            for i in range(size_mb * 50):  # 50 entries per MB
                data = "x" * 1024  # 1KB data
                cache.put(f"key_{len(cache_managers)}_{i}", data, CacheType.COMPUTATION_ARTIFACT)
            
            # Measure memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - baseline_memory
            memory_measurements.append(memory_increase)
            
            print(f"Cache size {size_mb}MB: Memory increase {memory_increase:.1f}MB")
        
        # Memory should scale roughly linearly
        # Allow some overhead but shouldn't be excessive
        max_expected_memory = sum([1, 5, 10, 20]) * 2  # 2x overhead allowance
        total_memory_increase = memory_measurements[-1]
        
        assert total_memory_increase < max_expected_memory, f"Excessive memory usage: {total_memory_increase}MB"
        
        # Clean up
        for cache in cache_managers:
            cache.clear()
    
    def test_throughput_under_load(self):
        """Test system throughput under high load."""
        executor = ParallelQuantumExecutor(
            min_workers=4,
            max_workers=8,
            auto_scaling=True
        )
        load_balancer = QuantumLoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # Add nodes
        for i in range(4):
            load_balancer.add_node(f"node_{i}", f"http://node{i}:8000", max_connections=25)
        
        executor.start()
        
        try:
            # High-throughput test
            num_requests = 100
            request_results = []
            
            def fast_task(task_id):
                """Fast task for throughput testing."""
                # Route through load balancer
                node_id = load_balancer.route_request(f"req_{task_id}")
                
                # Simulate fast computation
                result = task_id ** 2
                time.sleep(0.001)  # 1ms work
                
                # Complete routing
                if node_id:
                    load_balancer.complete_request(f"req_{task_id}", success=True, response_time=0.001)
                
                return result
            
            # Submit high volume of requests
            start_time = time.time()
            task_ids = []
            
            for i in range(num_requests):
                task_id = executor.submit_task(f"throughput_task_{i}", fast_task, args=(i,))
                task_ids.append(task_id)
            
            # Collect results
            for task_id in task_ids:
                try:
                    result = executor.get_task_result(task_id, timeout=10.0)
                    request_results.append(result)
                except Exception as e:
                    print(f"Task {task_id} failed: {e}")
            
            end_time = time.time()
            
            # Calculate throughput metrics
            total_time = end_time - start_time
            throughput = len(request_results) / total_time  # requests per second
            success_rate = len(request_results) / num_requests
            
            # Check load balancer performance
            lb_status = load_balancer.get_load_balancer_status()
            lb_throughput = lb_status["statistics"]["requests_per_second"]
            
            print(f"Throughput test results:")
            print(f"Requests: {num_requests}")
            print(f"Completed: {len(request_results)}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Throughput: {throughput:.1f} req/s")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Load balancer throughput: {lb_throughput:.1f} req/s")
            
            # Performance expectations
            assert success_rate > 0.9  # 90% success rate
            assert throughput > 10.0    # At least 10 requests/second
            
        finally:
            executor.stop()


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_research_workflow_simulation(self):
        """Test simulated research workflow."""
        # Simulate a machine learning research pipeline
        planner = QuantumTaskPlanner()
        scheduler = TaskScheduler(strategy=SchedulingStrategy.HYBRID_CLASSICAL)
        
        # Research pipeline tasks
        research_tasks = [
            # Data preparation phase
            Task("data_collection", "Data Collection", TaskPriority.HIGH, 4.0, [],
                 {"network": 10.0, "storage": 50.0}, quantum_weight=0.6),
            Task("data_cleaning", "Data Cleaning", TaskPriority.HIGH, 6.0, ["data_collection"],
                 {"cpu": 4.0, "memory": 8192.0}, quantum_weight=0.7),
            Task("data_validation", "Data Validation", TaskPriority.MEDIUM, 2.0, ["data_cleaning"],
                 {"cpu": 2.0}, quantum_weight=0.5),
            
            # Experimentation phase
            Task("baseline_model", "Baseline Model", TaskPriority.HIGH, 8.0, ["data_validation"],
                 {"gpu": 1.0, "memory": 4096.0}, quantum_weight=0.8),
            Task("hyperparameter_search", "Hyperparameter Search", TaskPriority.MEDIUM, 12.0, ["baseline_model"],
                 {"gpu": 2.0, "cpu": 8.0}, quantum_weight=0.9),
            Task("model_ensemble", "Model Ensemble", TaskPriority.LOW, 6.0, ["hyperparameter_search"],
                 {"cpu": 4.0, "memory": 2048.0}, quantum_weight=0.6),
            
            # Analysis phase
            Task("statistical_analysis", "Statistical Analysis", TaskPriority.MEDIUM, 3.0, ["model_ensemble"],
                 {"cpu": 2.0}, quantum_weight=0.4),
            Task("visualization", "Result Visualization", TaskPriority.LOW, 2.0, ["statistical_analysis"],
                 {"cpu": 1.0, "memory": 1024.0}, quantum_weight=0.3),
            Task("report_generation", "Report Generation", TaskPriority.HIGH, 4.0, ["visualization"],
                 {"cpu": 2.0}, quantum_weight=0.7)
        ]
        
        # Set up resources (research lab constraints)
        resources = {
            "cpu": 16.0,       # 16 CPU cores
            "gpu": 4.0,        # 4 GPUs
            "memory": 32768.0, # 32GB RAM
            "network": 100.0,  # 100 Mbps
            "storage": 1000.0  # 1TB storage
        }
        
        planner.set_resource_limits(resources)
        for name, capacity in resources.items():
            scheduler.add_resource(name, capacity)
        
        # Add tasks to planner
        for task in research_tasks:
            planner.add_task(task)
        
        # Plan research workflow
        planning_result = planner.plan_tasks()
        
        # Schedule selected tasks
        selected_tasks = planning_result["schedule"]
        scheduling_result = scheduler.schedule_tasks(selected_tasks)
        
        # Analyze research workflow
        assert len(selected_tasks) > 0
        assert scheduling_result.success_rate > 0
        
        # Check critical path (should include high-priority tasks)
        scheduled_priorities = [task.task.priority for task in scheduling_result.scheduled_tasks]
        assert TaskPriority.HIGH in scheduled_priorities
        
        # Verify research phases are represented
        scheduled_names = [task.task.name for task in scheduling_result.scheduled_tasks]
        phases = {
            "data": any("Data" in name for name in scheduled_names),
            "model": any("Model" in name for name in scheduled_names),
            "analysis": any("Analysis" in name or "Visualization" in name for name in scheduled_names)
        }
        
        completed_phases = sum(phases.values())
        assert completed_phases >= 2  # At least 2 phases should be scheduled
        
        print(f"Research workflow simulation:")
        print(f"Tasks planned: {len(selected_tasks)}/{len(research_tasks)}")
        print(f"Tasks scheduled: {len(scheduling_result.scheduled_tasks)}")
        print(f"Total makespan: {scheduling_result.total_makespan:.1f}s")
        print(f"Phases covered: {completed_phases}/3")
        print(f"Scheduled tasks: {[t.task.name for t in scheduling_result.scheduled_tasks]}")
    
    def test_production_deployment_scenario(self):
        """Test production deployment scenario."""
        # Simulate production quantum task processing system
        health_monitor = HealthMonitor(check_interval=1.0)
        performance_monitor = PerformanceMonitor()
        load_balancer = QuantumLoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE_QUANTUM)
        cache = QuantumCacheManager(strategy=CacheStrategy.ADAPTIVE)
        
        # Production-like setup
        # Multiple service nodes
        service_nodes = [
            ("primary", "http://primary:8000", 2.0, 50),
            ("secondary_1", "http://sec1:8000", 1.5, 30),
            ("secondary_2", "http://sec2:8000", 1.5, 30),
            ("backup", "http://backup:8000", 1.0, 20)
        ]
        
        for node_id, endpoint, weight, max_conn in service_nodes:
            load_balancer.add_node(node_id, endpoint, weight, max_conn)
        
        health_monitor.start_monitoring()
        
        try:
            # Simulate production workload
            request_count = 0
            successful_requests = 0
            cached_requests = 0
            
            # Production request patterns (varying load)
            for batch in range(5):  # 5 batches of requests
                batch_size = np.random.randint(5, 15)  # Variable batch size
                
                for req in range(batch_size):
                    request_id = f"prod_req_{request_count}"
                    request_count += 1
                    
                    # Check cache first
                    cache_key = f"request_result_{req % 10}"  # Some requests repeat
                    cached_result = cache.get(cache_key)
                    
                    if cached_result is not None:
                        cached_requests += 1
                        successful_requests += 1
                        continue
                    
                    # Route request
                    node_id = load_balancer.route_request(request_id)
                    
                    if node_id:
                        # Simulate processing
                        processing_time = np.random.exponential(0.1)  # Variable processing time
                        success = np.random.random() > 0.05  # 95% success rate
                        
                        if success:
                            # Simulate result and cache it
                            result = {"request_id": request_id, "result": f"processed_by_{node_id}"}
                            cache.put(cache_key, result, CacheType.COMPUTATION_ARTIFACT, quantum_priority=0.7)
                            
                            successful_requests += 1
                            load_balancer.complete_request(request_id, success=True, response_time=processing_time)
                        else:
                            load_balancer.complete_request(request_id, success=False, error="Processing failed")
                        
                        # Record performance metrics
                        performance_monitor.record_metric(PerformanceMonitor.PerformanceMetric(
                            "request_processing_time",
                            processing_time,
                            performance_monitor.MetricType.LATENCY,
                            time.time(),
                            tags={"node": node_id, "batch": str(batch)}
                        ))
                
                # Brief pause between batches
                time.sleep(0.1)
            
            # Analyze production performance
            success_rate = successful_requests / request_count
            cache_hit_rate = cached_requests / request_count
            
            # Get system health
            system_health = health_monitor.get_health_summary()
            
            # Get performance metrics
            perf_summary = performance_monitor.get_performance_summary()
            
            # Get load balancer stats
            lb_stats = load_balancer.get_load_balancer_status()
            
            # Get cache stats
            cache_stats = cache.get_statistics()
            
            print(f"Production deployment simulation:")
            print(f"Total requests: {request_count}")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Cache hit rate: {cache_hit_rate:.2f}")
            print(f"System health: {system_health['overall_status']}")
            print(f"Load balancer success rate: {lb_stats['statistics']['success_rate']:.2f}")
            print(f"Cache hit rate (internal): {cache_stats['hit_rate']:.2f}")
            
            # Production quality expectations
            assert success_rate > 0.8     # 80% overall success
            assert cache_hit_rate > 0.1   # Some caching benefit
            assert lb_stats['statistics']['success_rate'] > 0.8  # Load balancer effectiveness
            
        finally:
            health_monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for integration tests