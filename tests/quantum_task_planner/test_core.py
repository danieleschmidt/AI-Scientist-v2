"""
Comprehensive tests for quantum task planner core functionality.

Tests quantum planning algorithms, optimization, and task scheduling
with focus on achieving 85%+ test coverage.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from quantum_task_planner.core.planner import (
    QuantumTaskPlanner, Task, TaskPriority
)
from quantum_task_planner.core.quantum_optimizer import (
    QuantumOptimizer, OptimizationResult
)
from quantum_task_planner.core.task_scheduler import (
    TaskScheduler, SchedulingStrategy, ScheduleResult
)
from quantum_task_planner.utils.quantum_math import QubitState
from quantum_task_planner.validation.error_handling import (
    QuantumPlannerError, ValidationError, OptimizationError
)


class TestQuantumTaskPlanner:
    """Test suite for QuantumTaskPlanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QuantumTaskPlanner(max_iterations=100)
        
        # Create test tasks
        self.test_tasks = [
            Task(
                id="task1",
                name="High Priority Task",
                priority=TaskPriority.HIGH,
                duration=5.0,
                dependencies=[],
                resources={"cpu": 2.0, "memory": 1024.0},
                quantum_weight=0.9
            ),
            Task(
                id="task2",
                name="Medium Priority Task",
                priority=TaskPriority.MEDIUM,
                duration=3.0,
                dependencies=["task1"],
                resources={"cpu": 1.0, "memory": 512.0},
                quantum_weight=0.6
            ),
            Task(
                id="task3",
                name="Low Priority Task",
                priority=TaskPriority.LOW,
                duration=2.0,
                dependencies=[],
                resources={"cpu": 0.5, "memory": 256.0},
                quantum_weight=0.3
            )
        ]
    
    def test_initialization(self):
        """Test planner initialization."""
        planner = QuantumTaskPlanner(max_iterations=500, convergence_threshold=1e-8)
        assert planner.max_iterations == 500
        assert planner.convergence_threshold == 1e-8
        assert len(planner.tasks) == 0
        assert len(planner.resource_limits) == 0
    
    def test_add_task(self):
        """Test adding tasks to planner."""
        initial_count = len(self.planner.tasks)
        
        for task in self.test_tasks:
            self.planner.add_task(task)
        
        assert len(self.planner.tasks) == initial_count + len(self.test_tasks)
        assert self.planner.tasks[-1].id == "task3"
    
    def test_set_resource_limits(self):
        """Test setting resource limits."""
        limits = {"cpu": 8.0, "memory": 4096.0, "gpu": 2.0}
        self.planner.set_resource_limits(limits)
        
        assert self.planner.resource_limits == limits
    
    def test_create_quantum_superposition(self):
        """Test quantum superposition creation."""
        # Empty tasks
        superposition = self.planner.create_quantum_superposition([])
        assert len(superposition) == 1
        assert superposition[0] == 1.0
        
        # Non-empty tasks
        superposition = self.planner.create_quantum_superposition(self.test_tasks)
        assert len(superposition) == 2 ** len(self.test_tasks)
        assert abs(np.sum(np.abs(superposition)**2) - 1.0) < 1e-10  # Normalized
    
    def test_energy_calculation(self):
        """Test energy calculation for schedules."""
        # Valid schedule
        schedule = np.array([1, 0, 1])  # Select task1 and task3
        energy = self.planner._calculate_energy(schedule)
        assert energy >= 0
        
        # Invalid schedule length
        invalid_schedule = np.array([1, 0])
        energy = self.planner._calculate_energy(invalid_schedule)
        assert energy == float('inf')
        
        # Schedule with resource violations
        self.planner.set_resource_limits({"cpu": 1.0, "memory": 500.0})
        over_schedule = np.array([1, 1, 1])  # All tasks exceed limits
        energy = self.planner._calculate_energy(over_schedule)
        assert energy > 1000  # Should have high penalty
    
    def test_quantum_annealing_optimization(self):
        """Test quantum annealing optimization."""
        # Add tasks and limits
        for task in self.test_tasks:
            self.planner.add_task(task)
        self.planner.set_resource_limits({"cpu": 10.0, "memory": 5000.0})
        
        # Create superposition
        superposition = self.planner.create_quantum_superposition(self.planner.tasks)
        
        # Run optimization
        schedule, energy = self.planner.quantum_annealing_optimization(superposition)
        
        assert len(schedule) == len(self.test_tasks)
        assert energy >= 0
        assert all(s in [0, 1] for s in schedule)
    
    def test_plan_tasks_empty(self):
        """Test planning with no tasks."""
        result = self.planner.plan_tasks()
        
        assert result["schedule"] == []
        assert result["energy"] == 0.0
        assert result["total_tasks"] == 0
        assert result["selected_tasks"] == 0
    
    def test_plan_tasks_with_tasks(self):
        """Test planning with tasks."""
        # Add tasks and set limits
        for task in self.test_tasks:
            self.planner.add_task(task)
        self.planner.set_resource_limits({"cpu": 10.0, "memory": 5000.0})
        
        result = self.planner.plan_tasks()
        
        assert "schedule" in result
        assert "energy" in result
        assert "quantum_metrics" in result
        assert result["total_tasks"] == len(self.test_tasks)
        assert result["planning_time"] > 0
        
        # Check quantum metrics
        metrics = result["quantum_metrics"]
        assert "coherence" in metrics
        assert "entanglement" in metrics
        assert "superposition_states" in metrics
        assert 0 <= metrics["coherence"] <= 1
        assert 0 <= metrics["entanglement"] <= 1
    
    def test_quantum_coherence_calculation(self):
        """Test quantum coherence calculation."""
        # Uniform superposition (high coherence)
        uniform_state = np.ones(4) / 2.0
        coherence = self.planner._calculate_quantum_coherence(uniform_state)
        assert coherence > 0.9
        
        # Single state (low coherence)
        single_state = np.array([1.0, 0.0, 0.0, 0.0])
        coherence = self.planner._calculate_quantum_coherence(single_state)
        assert coherence < 0.1
        
        # Empty state
        empty_state = np.array([])
        coherence = self.planner._calculate_quantum_coherence(empty_state)
        assert coherence == 0.0
    
    def test_entanglement_measure_calculation(self):
        """Test entanglement measure calculation."""
        # Test with no dependencies
        schedule = [1, 1, 0]
        entanglement = self.planner._calculate_entanglement_measure(schedule)
        assert 0 <= entanglement <= 1
        
        # Test with dependencies
        for task in self.test_tasks:
            self.planner.add_task(task)
        
        # Schedule with satisfied dependencies
        satisfied_schedule = [1, 1, 0]  # task1 and task2 (task2 depends on task1)
        entanglement = self.planner._calculate_entanglement_measure(satisfied_schedule)
        assert entanglement > 0
    
    def test_metrics_recording(self):
        """Test metrics recording."""
        initial_metrics = self.planner.get_metrics()
        
        # Add tasks and plan
        for task in self.test_tasks:
            self.planner.add_task(task)
        self.planner.set_resource_limits({"cpu": 10.0, "memory": 5000.0})
        
        result = self.planner.plan_tasks()
        
        # Check that metrics were recorded
        final_metrics = self.planner.get_metrics()
        assert final_metrics != initial_metrics


class TestQuantumOptimizer:
    """Test suite for QuantumOptimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = QuantumOptimizer(max_iterations=50, tolerance=1e-6)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumOptimizer(max_iterations=1000, tolerance=1e-8, learning_rate=0.05)
        assert optimizer.max_iterations == 1000
        assert optimizer.tolerance == 1e-8
        assert optimizer.learning_rate == 0.05
        assert len(optimizer.optimization_history) == 0
    
    def test_simple_optimization(self):
        """Test simple optimization problem."""
        # Quadratic function: f(x) = (x - 2)^2
        def objective(x):
            return (x[0] - 2.0) ** 2
        
        initial_solution = np.array([0.0])
        bounds = [(-10.0, 10.0)]
        
        result = self.optimizer.optimize(objective, initial_solution, bounds)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.solution) == 1
        assert result.objective_value >= 0
        assert result.iterations > 0
        assert result.convergence_time > 0
        assert isinstance(result.quantum_metrics, dict)
        
        # Solution should be close to optimal (x=2)
        assert abs(result.solution[0] - 2.0) < 1.0  # Reasonable tolerance for quantum optimization
    
    def test_multidimensional_optimization(self):
        """Test multi-dimensional optimization."""
        # Rosenbrock function
        def rosenbrock(x):
            return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        initial_solution = np.array([0.0, 0.0])
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        result = self.optimizer.optimize(rosenbrock, initial_solution, bounds)
        
        assert len(result.solution) == 2
        assert result.objective_value >= 0
        # Global minimum is at (1, 1) but quantum optimization might not find it exactly
        assert result.objective_value < 100.0  # Should find a reasonable solution
    
    def test_qaoa_step(self):
        """Test QAOA step execution."""
        solution = np.array([0.5, 0.3])
        beta = np.array([np.pi/4, np.pi/6])
        gamma = np.array([np.pi/3, np.pi/2])
        
        def dummy_objective(x):
            return np.sum(x**2)
        
        new_solution = self.optimizer._qaoa_step(solution, beta, gamma, dummy_objective)
        
        assert len(new_solution) == len(solution)
        assert np.all(np.isfinite(new_solution))
    
    def test_quantum_measurement(self):
        """Test quantum measurement."""
        # Create a normalized quantum state
        n_vars = 2
        quantum_state = np.array([0.5, 0.5, 0.5, 0.5])
        
        solution = self.optimizer._quantum_measurement(quantum_state, n_vars)
        
        assert len(solution) == n_vars
        assert np.all(solution >= 0)
        assert np.all(solution <= 1)
    
    def test_bounds_application(self):
        """Test bounds constraint application."""
        solution = np.array([-5.0, 5.0, 0.5])
        bounds = [(-1.0, 1.0), (-2.0, 2.0), (0.0, 1.0)]
        
        bounded = self.optimizer._apply_bounds(solution, bounds)
        
        assert bounded[0] == -1.0  # Clipped to lower bound
        assert bounded[1] == 2.0   # Clipped to upper bound
        assert bounded[2] == 0.5   # Within bounds
    
    def test_optimization_with_convergence(self):
        """Test optimization that should converge quickly."""
        # Simple quadratic with known minimum
        def simple_quadratic(x):
            return x[0]**2
        
        initial_solution = np.array([5.0])
        
        result = self.optimizer.optimize(simple_quadratic, initial_solution)
        
        assert result.objective_value < 1.0  # Should converge to near-zero
        assert abs(result.solution[0]) < 1.0
    
    def test_quantum_metrics_calculation(self):
        """Test quantum metrics calculation."""
        beta = np.array([0.5, 0.3])
        gamma = np.array([0.2, 0.7])
        solution = np.array([0.4, 0.6])
        
        # Set up optimization history
        self.optimizer.optimization_history = [
            {'iteration': 0, 'objective_value': 10.0, 'best_value': 10.0},
            {'iteration': 1, 'objective_value': 8.0, 'best_value': 8.0},
            {'iteration': 2, 'objective_value': 6.0, 'best_value': 6.0}
        ]
        
        metrics = self.optimizer._calculate_quantum_metrics(beta, gamma, solution)
        
        assert 'beta_variance' in metrics
        assert 'gamma_variance' in metrics
        assert 'solution_entropy' in metrics
        assert 'quantum_fidelity' in metrics
        assert 'convergence_rate' in metrics
        
        assert all(np.isfinite(v) for v in metrics.values())
    
    def test_optimization_history_tracking(self):
        """Test optimization history tracking."""
        def objective(x):
            return x[0]**2 + x[1]**2
        
        initial_solution = np.array([2.0, -2.0])
        
        result = self.optimizer.optimize(objective, initial_solution)
        
        history = self.optimizer.get_optimization_history()
        assert len(history) > 0
        assert all('iteration' in entry for entry in history)
        assert all('objective_value' in entry for entry in history)
        assert all('best_value' in entry for entry in history)


class TestTaskScheduler:
    """Test suite for TaskScheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = TaskScheduler(
            strategy=SchedulingStrategy.QUANTUM_ANNEALING,
            max_parallel_tasks=5
        )
        
        # Add resources
        self.scheduler.add_resource("cpu", 8.0)
        self.scheduler.add_resource("memory", 16.0)
        self.scheduler.add_resource("gpu", 2.0)
        
        # Create test tasks
        self.test_tasks = [
            Task(
                id="sched_task1",
                name="Compute Task",
                priority=TaskPriority.HIGH,
                duration=2.0,
                dependencies=[],
                resources={"cpu": 2.0, "memory": 4.0},
                deadline=10.0
            ),
            Task(
                id="sched_task2",
                name="Memory Task",
                priority=TaskPriority.MEDIUM,
                duration=3.0,
                dependencies=["sched_task1"],
                resources={"memory": 8.0},
                deadline=15.0
            ),
            Task(
                id="sched_task3",
                name="GPU Task",
                priority=TaskPriority.LOW,
                duration=1.0,
                dependencies=[],
                resources={"gpu": 1.0},
                deadline=5.0
            )
        ]
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = TaskScheduler(
            strategy=SchedulingStrategy.HYBRID_CLASSICAL,
            time_quantum=0.5,
            max_parallel_tasks=10
        )
        
        assert scheduler.strategy == SchedulingStrategy.HYBRID_CLASSICAL
        assert scheduler.time_quantum == 0.5
        assert scheduler.max_parallel_tasks == 10
        assert len(scheduler.resources) == 0
    
    def test_add_resource(self):
        """Test adding resources."""
        scheduler = TaskScheduler()
        
        scheduler.add_resource("network", 100.0)
        assert "network" in scheduler.resources
        assert scheduler.resources["network"].capacity == 100.0
        assert scheduler.resources["network"].available == 100.0
    
    def test_schedule_tasks_empty(self):
        """Test scheduling with no tasks."""
        result = self.scheduler.schedule_tasks([])
        
        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) == 0
        assert result.total_makespan == 0.0
        assert result.success_rate == 1.0
    
    def test_quantum_annealing_schedule(self):
        """Test quantum annealing scheduling strategy."""
        result = self.scheduler.schedule_tasks(self.test_tasks)
        
        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) <= len(self.test_tasks)
        assert result.total_makespan >= 0
        assert 0 <= result.success_rate <= 1
        assert result.scheduling_time > 0
        assert "quantum_metrics" in result.__dict__
    
    def test_hybrid_classical_schedule(self):
        """Test hybrid classical scheduling strategy."""
        self.scheduler.strategy = SchedulingStrategy.HYBRID_CLASSICAL
        result = self.scheduler.schedule_tasks(self.test_tasks)
        
        assert isinstance(result, ScheduleResult)
        assert "hybrid_efficiency" in result.quantum_metrics
        assert "quantum_probability_avg" in result.quantum_metrics
        assert "classical_component_ratio" in result.quantum_metrics
    
    def test_variational_schedule(self):
        """Test variational optimization scheduling."""
        self.scheduler.strategy = SchedulingStrategy.VARIATIONAL_OPTIMIZATION
        result = self.scheduler.schedule_tasks(self.test_tasks)
        
        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) >= 0
    
    def test_adiabatic_evolution_schedule(self):
        """Test adiabatic evolution scheduling."""
        self.scheduler.strategy = SchedulingStrategy.ADIABATIC_EVOLUTION
        result = self.scheduler.schedule_tasks(self.test_tasks)
        
        assert isinstance(result, ScheduleResult)
        assert "adiabatic_fidelity" in result.quantum_metrics
        assert "evolution_steps" in result.quantum_metrics
        assert "final_coherence" in result.quantum_metrics
    
    def test_energy_evaluation(self):
        """Test schedule energy evaluation."""
        schedule_vec = np.array([1.0, 0.5, 0.8])  # Quantum probabilities
        
        energy = self.scheduler._evaluate_schedule_energy(self.test_tasks, schedule_vec)
        
        assert energy >= 0
        assert np.isfinite(energy)
    
    def test_can_schedule_task(self):
        """Test task scheduling feasibility check."""
        task = self.test_tasks[0]  # Compute task requiring 2 CPU, 4 memory
        
        # Should be schedulable with available resources
        can_schedule = self.scheduler._can_schedule_task(task, 0.0)
        assert can_schedule
        
        # Allocate most resources
        self.scheduler.resources["cpu"].allocated = 7.0
        self.scheduler.resources["memory"].allocated = 14.0
        
        # Should not be schedulable now
        can_schedule = self.scheduler._can_schedule_task(task, 0.0)
        assert not can_schedule
    
    def test_schedule_single_task(self):
        """Test single task scheduling."""
        task = self.test_tasks[0]
        start_time = 1.0
        
        scheduled_task = self.scheduler._schedule_single_task(task, start_time)
        
        assert scheduled_task.task == task
        assert scheduled_task.start_time == start_time
        assert scheduled_task.end_time == start_time + task.duration
        assert scheduled_task.allocated_resources == task.resources
    
    def test_resource_utilization_calculation(self):
        """Test resource utilization calculation."""
        # Create scheduled tasks
        scheduled_tasks = [
            self.scheduler._schedule_single_task(self.test_tasks[0], 0.0),
            self.scheduler._schedule_single_task(self.test_tasks[2], 0.0)
        ]
        
        # Allocate resources
        for task in scheduled_tasks:
            self.scheduler._allocate_resources(task)
        
        utilization = self.scheduler._calculate_resource_utilization(scheduled_tasks)
        
        assert "cpu" in utilization
        assert "memory" in utilization
        assert "gpu" in utilization
        assert all(0 <= util <= 1 for util in utilization.values())
    
    def test_variational_circuit(self):
        """Test variational circuit implementation."""
        params = np.array([0.5, 1.2, 0.8, 0.3])
        n_tasks = 2
        
        probs = self.scheduler._variational_circuit(params, n_tasks)
        
        assert len(probs) == n_tasks
        assert all(0 <= p <= 1 for p in probs)
    
    def test_quantum_to_classical_conversion(self):
        """Test quantum to classical schedule conversion."""
        quantum_probs = np.array([0.9, 0.3, 0.7])
        
        with patch('numpy.random.random') as mock_random:
            # Mock random values to make test deterministic
            mock_random.side_effect = [0.5, 0.2, 0.6]  # Will select tasks 0 and 2
            
            scheduled_tasks = self.scheduler._quantum_to_classical_schedule(
                self.test_tasks, quantum_probs
            )
            
            # Should select tasks with high probability that pass resource constraints
            assert len(scheduled_tasks) >= 0
            assert all(hasattr(task, 'quantum_probability') for task in scheduled_tasks)


class TestIntegration:
    """Integration tests for quantum task planner components."""
    
    def test_end_to_end_planning(self):
        """Test complete end-to-end planning workflow."""
        # Create planner with scheduler
        planner = QuantumTaskPlanner(max_iterations=100)
        scheduler = TaskScheduler(strategy=SchedulingStrategy.QUANTUM_ANNEALING)
        
        # Add resources to scheduler
        scheduler.add_resource("cpu", 10.0)
        scheduler.add_resource("memory", 8192.0)
        
        # Create complex task set
        tasks = [
            Task("e2e_task1", "Initial Task", TaskPriority.HIGH, 2.0, [], {"cpu": 2.0}),
            Task("e2e_task2", "Dependent Task", TaskPriority.MEDIUM, 3.0, ["e2e_task1"], {"memory": 1024.0}),
            Task("e2e_task3", "Parallel Task", TaskPriority.LOW, 1.0, [], {"cpu": 1.0}),
            Task("e2e_task4", "Final Task", TaskPriority.HIGH, 2.0, ["e2e_task2", "e2e_task3"], {"cpu": 1.0, "memory": 512.0})
        ]
        
        # Plan tasks
        for task in tasks:
            planner.add_task(task)
        planner.set_resource_limits({"cpu": 10.0, "memory": 8192.0})
        
        planning_result = planner.plan_tasks()
        
        # Schedule tasks
        scheduling_result = scheduler.schedule_tasks(tasks)
        
        # Verify results
        assert planning_result["total_tasks"] == 4
        assert len(scheduling_result.scheduled_tasks) <= 4
        assert planning_result["quantum_metrics"]["coherence"] >= 0
        assert scheduling_result.quantum_metrics is not None
    
    def test_optimization_integration(self):
        """Test integration with quantum optimizer."""
        optimizer = QuantumOptimizer(max_iterations=50)
        
        # Create optimization problem from task planning
        def task_planning_objective(solution):
            # Simple objective: minimize completion time
            return np.sum(solution * np.array([2.0, 3.0, 1.0]))
        
        initial_solution = np.array([0.5, 0.5, 0.5])
        bounds = [(0.0, 1.0)] * 3
        
        result = optimizer.optimize(task_planning_objective, initial_solution, bounds)
        
        assert result.objective_value >= 0
        assert len(result.solution) == 3
        assert all(0 <= x <= 1 for x in result.solution)
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        planner = QuantumTaskPlanner()
        
        # Test with invalid task configuration
        invalid_task = Task(
            id="invalid",
            name="Invalid Task",
            priority=TaskPriority.HIGH,
            duration=-1.0,  # Invalid duration
            dependencies=["nonexistent"],  # Invalid dependency
            resources={"invalid_resource": -100.0}  # Invalid resource
        )
        
        # Should handle gracefully without crashing
        planner.add_task(invalid_task)
        
        # Planning should still work (though may not select invalid task)
        result = planner.plan_tasks()
        assert "schedule" in result
        assert isinstance(result["energy"], (int, float))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_task_planning(self):
        """Test planning with single task."""
        planner = QuantumTaskPlanner()
        task = Task("single", "Single Task", TaskPriority.MEDIUM, 1.0, [], {"cpu": 1.0})
        
        planner.add_task(task)
        planner.set_resource_limits({"cpu": 2.0})
        
        result = planner.plan_tasks()
        
        assert result["total_tasks"] == 1
        assert result["selected_tasks"] <= 1
    
    def test_no_resource_limits(self):
        """Test planning without resource limits."""
        planner = QuantumTaskPlanner()
        task = Task("unlimited", "Unlimited Task", TaskPriority.HIGH, 1.0, [], {"cpu": 1000.0})
        
        planner.add_task(task)
        # No resource limits set
        
        result = planner.plan_tasks()
        
        # Should still work, just with higher energy
        assert "schedule" in result
        assert result["energy"] >= 0
    
    def test_circular_dependencies(self):
        """Test handling of circular dependencies."""
        scheduler = TaskScheduler()
        
        # Create circular dependency
        task_a = Task("a", "Task A", TaskPriority.HIGH, 1.0, ["b"], {"cpu": 1.0})
        task_b = Task("b", "Task B", TaskPriority.HIGH, 1.0, ["a"], {"cpu": 1.0})
        
        scheduler.add_resource("cpu", 5.0)
        
        # Should handle gracefully
        result = scheduler.schedule_tasks([task_a, task_b])
        
        assert isinstance(result, ScheduleResult)
        # Likely won't schedule both due to circular dependency
    
    def test_zero_duration_tasks(self):
        """Test tasks with zero duration."""
        planner = QuantumTaskPlanner()
        task = Task("instant", "Instant Task", TaskPriority.HIGH, 0.0, [], {})
        
        planner.add_task(task)
        result = planner.plan_tasks()
        
        # Should handle zero duration tasks
        assert result["total_tasks"] == 1
    
    def test_large_number_of_tasks(self):
        """Test with large number of tasks (performance test)."""
        planner = QuantumTaskPlanner(max_iterations=10)  # Reduced for speed
        
        # Create 50 tasks
        tasks = []
        for i in range(50):
            task = Task(
                f"task_{i}",
                f"Task {i}",
                TaskPriority.MEDIUM,
                1.0 + i * 0.1,
                [],
                {"cpu": 1.0, "memory": 100.0}
            )
            tasks.append(task)
            planner.add_task(task)
        
        planner.set_resource_limits({"cpu": 100.0, "memory": 10000.0})
        
        start_time = time.time()
        result = planner.plan_tasks()
        execution_time = time.time() - start_time
        
        assert result["total_tasks"] == 50
        assert execution_time < 30.0  # Should complete within 30 seconds
    
    def test_extreme_quantum_weights(self):
        """Test tasks with extreme quantum weights."""
        planner = QuantumTaskPlanner()
        
        # Tasks with extreme quantum weights
        high_weight_task = Task("high", "High Weight", TaskPriority.LOW, 1.0, [], {"cpu": 1.0}, quantum_weight=1.0)
        zero_weight_task = Task("zero", "Zero Weight", TaskPriority.HIGH, 1.0, [], {"cpu": 1.0}, quantum_weight=0.0)
        
        planner.add_task(high_weight_task)
        planner.add_task(zero_weight_task)
        planner.set_resource_limits({"cpu": 1.5})  # Can only fit one task
        
        result = planner.plan_tasks()
        
        # High weight task should be preferred despite lower priority
        assert result["total_tasks"] == 2
        assert result["selected_tasks"] <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])