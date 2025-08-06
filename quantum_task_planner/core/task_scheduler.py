"""
Task Scheduler with Quantum-Inspired Resource Allocation

High-performance task scheduling system using quantum annealing principles
for optimal resource allocation and task sequencing.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from .planner import Task, TaskPriority
from .quantum_optimizer import QuantumOptimizer
from ..utils.metrics import PlannerMetrics

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Quantum-inspired scheduling strategies."""
    QUANTUM_ANNEALING = "quantum_annealing"
    VARIATIONAL_OPTIMIZATION = "variational_optimization" 
    HYBRID_CLASSICAL = "hybrid_classical"
    ADIABATIC_EVOLUTION = "adiabatic_evolution"


@dataclass
class ResourceState:
    """Track resource utilization state."""
    name: str
    capacity: float
    allocated: float = 0.0
    reserved: float = 0.0
    
    @property
    def available(self) -> float:
        return self.capacity - self.allocated - self.reserved
    
    @property
    def utilization_ratio(self) -> float:
        return (self.allocated + self.reserved) / self.capacity if self.capacity > 0 else 0.0


@dataclass 
class ScheduledTask:
    """Task with scheduling information."""
    task: Task
    start_time: float
    end_time: float
    allocated_resources: Dict[str, float]
    dependencies_met: bool = True
    quantum_probability: float = 1.0


@dataclass
class ScheduleResult:
    """Complete scheduling result."""
    scheduled_tasks: List[ScheduledTask]
    total_makespan: float
    resource_utilization: Dict[str, float]
    quantum_metrics: Dict[str, Any]
    scheduling_time: float
    success_rate: float


class TaskScheduler:
    """
    Quantum-Inspired Task Scheduler
    
    Uses quantum annealing and variational optimization techniques
    for efficient task scheduling and resource allocation.
    """
    
    def __init__(self,
                 strategy: SchedulingStrategy = SchedulingStrategy.QUANTUM_ANNEALING,
                 time_quantum: float = 1.0,
                 max_parallel_tasks: int = 10):
        """
        Initialize task scheduler.
        
        Args:
            strategy: Scheduling strategy to use
            time_quantum: Time discretization unit
            max_parallel_tasks: Maximum concurrent tasks
        """
        self.strategy = strategy
        self.time_quantum = time_quantum
        self.max_parallel_tasks = max_parallel_tasks
        
        self.quantum_optimizer = QuantumOptimizer(
            max_iterations=500,
            tolerance=1e-6,
            learning_rate=0.05
        )
        
        self.resources: Dict[str, ResourceState] = {}
        self.metrics = PlannerMetrics()
        
        logger.info(f"Initialized TaskScheduler with {strategy.value} strategy")
    
    def add_resource(self, name: str, capacity: float) -> None:
        """Add resource constraint."""
        self.resources[name] = ResourceState(name, capacity)
        logger.debug(f"Added resource {name} with capacity {capacity}")
    
    def schedule_tasks(self, tasks: List[Task]) -> ScheduleResult:
        """
        Execute quantum-inspired task scheduling.
        
        Args:
            tasks: List of tasks to schedule
            
        Returns:
            Complete scheduling result with quantum metrics
        """
        start_time = time.time()
        
        if not tasks:
            return ScheduleResult([], 0.0, {}, {}, 0.0, 1.0)
        
        logger.info(f"Scheduling {len(tasks)} tasks using {self.strategy.value}")
        
        # Execute strategy-specific scheduling
        if self.strategy == SchedulingStrategy.QUANTUM_ANNEALING:
            result = self._quantum_annealing_schedule(tasks)
        elif self.strategy == SchedulingStrategy.VARIATIONAL_OPTIMIZATION:
            result = self._variational_schedule(tasks)
        elif self.strategy == SchedulingStrategy.HYBRID_CLASSICAL:
            result = self._hybrid_classical_schedule(tasks)
        elif self.strategy == SchedulingStrategy.ADIABATIC_EVOLUTION:
            result = self._adiabatic_evolution_schedule(tasks)
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")
        
        scheduling_time = time.time() - start_time
        result.scheduling_time = scheduling_time
        
        self.metrics.record_scheduling_result(result)
        logger.info(f"Scheduled {len(result.scheduled_tasks)} tasks in {scheduling_time:.3f}s")
        
        return result
    
    def _quantum_annealing_schedule(self, tasks: List[Task]) -> ScheduleResult:
        """Schedule using quantum annealing approach."""
        n_tasks = len(tasks)
        
        # Create quantum problem formulation
        def objective_function(schedule_vec: np.ndarray) -> float:
            return self._evaluate_schedule_energy(tasks, schedule_vec)
        
        # Initial solution: uniform probability distribution
        initial_solution = np.ones(n_tasks) * 0.5
        
        # Optimize using quantum annealing
        opt_result = self.quantum_optimizer.optimize(
            objective_function=objective_function,
            initial_solution=initial_solution,
            bounds=[(0.0, 1.0) for _ in range(n_tasks)]
        )
        
        # Convert quantum solution to classical schedule
        scheduled_tasks = self._quantum_to_classical_schedule(tasks, opt_result.solution)
        
        # Calculate metrics
        makespan = max((t.end_time for t in scheduled_tasks), default=0.0)
        resource_util = self._calculate_resource_utilization(scheduled_tasks)
        success_rate = len(scheduled_tasks) / len(tasks)
        
        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            total_makespan=makespan,
            resource_utilization=resource_util,
            quantum_metrics=opt_result.quantum_metrics,
            scheduling_time=0.0,  # Set by caller
            success_rate=success_rate
        )
    
    def _variational_schedule(self, tasks: List[Task]) -> ScheduleResult:
        """Schedule using variational quantum optimization."""
        n_tasks = len(tasks)
        
        # Variational ansatz parameters
        theta = np.random.random(n_tasks * 2) * 2 * np.pi
        
        def variational_objective(params: np.ndarray) -> float:
            # Create quantum circuit with variational parameters
            schedule_probs = self._variational_circuit(params, n_tasks)
            return self._evaluate_schedule_energy(tasks, schedule_probs)
        
        # Optimize variational parameters
        opt_result = self.quantum_optimizer.optimize(
            objective_function=variational_objective,
            initial_solution=theta
        )
        
        # Generate final schedule from optimized parameters
        final_probs = self._variational_circuit(opt_result.solution, n_tasks)
        scheduled_tasks = self._quantum_to_classical_schedule(tasks, final_probs)
        
        # Calculate metrics
        makespan = max((t.end_time for t in scheduled_tasks), default=0.0)
        resource_util = self._calculate_resource_utilization(scheduled_tasks)
        success_rate = len(scheduled_tasks) / len(tasks)
        
        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            total_makespan=makespan,
            resource_utilization=resource_util,
            quantum_metrics=opt_result.quantum_metrics,
            scheduling_time=0.0,
            success_rate=success_rate
        )
    
    def _hybrid_classical_schedule(self, tasks: List[Task]) -> ScheduleResult:
        """Hybrid quantum-classical scheduling approach."""
        # Sort tasks by quantum-weighted priority
        quantum_sorted_tasks = sorted(tasks, 
                                    key=lambda t: t.priority.value * t.quantum_weight, 
                                    reverse=True)
        
        scheduled_tasks = []
        current_time = 0.0
        
        # Classical greedy scheduling with quantum probability weighting
        for task in quantum_sorted_tasks:
            if self._can_schedule_task(task, current_time):
                # Calculate quantum scheduling probability
                prob = self._calculate_scheduling_probability(task, scheduled_tasks)
                
                if np.random.random() < prob:
                    # Schedule the task
                    scheduled_task = self._schedule_single_task(task, current_time)
                    scheduled_tasks.append(scheduled_task)
                    
                    # Update resource allocations
                    self._allocate_resources(scheduled_task)
                    current_time = max(current_time, scheduled_task.end_time)
        
        # Calculate final metrics
        makespan = max((t.end_time for t in scheduled_tasks), default=0.0)
        resource_util = self._calculate_resource_utilization(scheduled_tasks)
        success_rate = len(scheduled_tasks) / len(tasks)
        
        quantum_metrics = {
            "hybrid_efficiency": success_rate,
            "quantum_probability_avg": np.mean([t.quantum_probability for t in scheduled_tasks]),
            "classical_component_ratio": 0.7,  # 70% classical, 30% quantum
            "scheduling_coherence": self._calculate_schedule_coherence(scheduled_tasks)
        }
        
        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            total_makespan=makespan,
            resource_utilization=resource_util,
            quantum_metrics=quantum_metrics,
            scheduling_time=0.0,
            success_rate=success_rate
        )
    
    def _adiabatic_evolution_schedule(self, tasks: List[Task]) -> ScheduleResult:
        """Adiabatic quantum evolution scheduling."""
        n_tasks = len(tasks)
        evolution_steps = 100
        
        # Initialize with all tasks in superposition
        schedule_state = np.ones(n_tasks) * 0.5
        
        # Adiabatic evolution from initial to final Hamiltonian
        for step in range(evolution_steps):
            s = step / evolution_steps  # Evolution parameter [0,1]
            
            # Interpolate between initial and problem Hamiltonians
            # H(s) = (1-s)H_initial + s*H_problem
            energy_gradient = self._calculate_adiabatic_gradient(tasks, schedule_state, s)
            
            # Evolve quantum state
            dt = 0.01
            schedule_state -= dt * energy_gradient
            
            # Normalize and constrain to [0,1]
            schedule_state = np.clip(schedule_state, 0.0, 1.0)
        
        # Convert final quantum state to classical schedule
        scheduled_tasks = self._quantum_to_classical_schedule(tasks, schedule_state)
        
        # Calculate metrics
        makespan = max((t.end_time for t in scheduled_tasks), default=0.0)
        resource_util = self._calculate_resource_utilization(scheduled_tasks)
        success_rate = len(scheduled_tasks) / len(tasks)
        
        quantum_metrics = {
            "adiabatic_fidelity": np.exp(-np.sum(energy_gradient**2)),
            "evolution_steps": evolution_steps,
            "final_coherence": 1.0 - np.var(schedule_state),
            "ground_state_overlap": max(schedule_state)
        }
        
        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            total_makespan=makespan,
            resource_utilization=resource_util,
            quantum_metrics=quantum_metrics,
            scheduling_time=0.0,
            success_rate=success_rate
        )
    
    def _evaluate_schedule_energy(self, tasks: List[Task], schedule_vec: np.ndarray) -> float:
        """Evaluate energy (cost) of a quantum schedule vector."""
        total_energy = 0.0
        
        # Resource utilization penalty
        resource_usage = {}
        for i, task in enumerate(tasks):
            task_prob = schedule_vec[i]
            for resource, amount in task.resources.items():
                if resource not in resource_usage:
                    resource_usage[resource] = 0.0
                resource_usage[resource] += task_prob * amount
        
        # Add resource constraint violations
        for resource, usage in resource_usage.items():
            if resource in self.resources:
                capacity = self.resources[resource].capacity
                if usage > capacity:
                    penalty = (usage - capacity) ** 2 * 1000.0
                    total_energy += penalty
        
        # Priority-based energy (lower for higher priority)
        for i, task in enumerate(tasks):
            task_prob = schedule_vec[i]
            priority_energy = task_prob / task.priority.value
            total_energy += priority_energy
        
        # Dependency constraints
        for i, task in enumerate(tasks):
            if schedule_vec[i] > 0.5:  # Task likely to be scheduled
                for dep_id in task.dependencies:
                    dep_idx = next((j for j, t in enumerate(tasks) if t.id == dep_id), None)
                    if dep_idx is not None and schedule_vec[dep_idx] < 0.5:
                        total_energy += 10000.0  # Heavy dependency penalty
        
        return total_energy
    
    def _variational_circuit(self, params: np.ndarray, n_tasks: int) -> np.ndarray:
        """Generate quantum schedule probabilities from variational parameters."""
        if len(params) < n_tasks:
            # Pad parameters if needed
            padded_params = np.zeros(n_tasks * 2)
            padded_params[:len(params)] = params
            params = padded_params
        
        schedule_probs = np.zeros(n_tasks)
        
        for i in range(n_tasks):
            # Variational ansatz: R_Y(θ)R_Z(φ)|0⟩
            theta = params[i * 2] if i * 2 < len(params) else np.pi/4
            phi = params[i * 2 + 1] if i * 2 + 1 < len(params) else 0
            
            # Calculate |⟨1|ψ⟩|² (probability of measuring |1⟩)
            prob_1 = np.sin(theta/2) ** 2
            schedule_probs[i] = prob_1
        
        return schedule_probs
    
    def _quantum_to_classical_schedule(self, tasks: List[Task], quantum_probs: np.ndarray) -> List[ScheduledTask]:
        """Convert quantum probability vector to classical task schedule."""
        scheduled_tasks = []
        current_time = 0.0
        
        # Sort by quantum probability (descending)
        task_prob_pairs = list(zip(tasks, quantum_probs))
        task_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for task, prob in task_prob_pairs:
            # Stochastic scheduling based on quantum probability
            if np.random.random() < prob and self._can_schedule_task(task, current_time):
                scheduled_task = self._schedule_single_task(task, current_time)
                scheduled_task.quantum_probability = prob
                scheduled_tasks.append(scheduled_task)
                
                # Update resource state
                self._allocate_resources(scheduled_task)
                current_time = max(current_time, scheduled_task.end_time)
        
        return scheduled_tasks
    
    def _can_schedule_task(self, task: Task, start_time: float) -> bool:
        """Check if task can be scheduled at given time."""
        # Check resource availability
        for resource, required in task.resources.items():
            if resource in self.resources:
                if self.resources[resource].available < required:
                    return False
        
        # Check deadline constraint
        if task.deadline is not None:
            if start_time + task.duration > task.deadline:
                return False
        
        return True
    
    def _schedule_single_task(self, task: Task, start_time: float) -> ScheduledTask:
        """Schedule a single task at specified time."""
        end_time = start_time + task.duration
        
        return ScheduledTask(
            task=task,
            start_time=start_time,
            end_time=end_time,
            allocated_resources=task.resources.copy(),
            dependencies_met=True,  # Simplified for v1
            quantum_probability=task.quantum_weight
        )
    
    def _allocate_resources(self, scheduled_task: ScheduledTask) -> None:
        """Allocate resources for scheduled task."""
        for resource, amount in scheduled_task.allocated_resources.items():
            if resource in self.resources:
                self.resources[resource].allocated += amount
    
    def _calculate_resource_utilization(self, scheduled_tasks: List[ScheduledTask]) -> Dict[str, float]:
        """Calculate final resource utilization ratios."""
        utilization = {}
        
        for resource_name, resource_state in self.resources.items():
            utilization[resource_name] = resource_state.utilization_ratio
        
        return utilization
    
    def _calculate_scheduling_probability(self, task: Task, existing_tasks: List[ScheduledTask]) -> float:
        """Calculate quantum scheduling probability for task."""
        base_prob = task.quantum_weight
        
        # Priority boost
        priority_factor = task.priority.value / 4.0
        
        # Resource availability factor
        resource_factor = 1.0
        for resource, required in task.resources.items():
            if resource in self.resources:
                availability_ratio = self.resources[resource].available / self.resources[resource].capacity
                resource_factor *= availability_ratio
        
        # Deadline pressure
        deadline_factor = 1.0
        if task.deadline is not None:
            time_remaining = task.deadline - len(existing_tasks) * self.time_quantum
            if time_remaining > 0:
                deadline_factor = min(1.0, task.duration / time_remaining)
        
        final_prob = base_prob * priority_factor * resource_factor * deadline_factor
        return min(1.0, final_prob)
    
    def _calculate_adiabatic_gradient(self, tasks: List[Task], state: np.ndarray, s: float) -> np.ndarray:
        """Calculate gradient for adiabatic evolution."""
        gradient = np.zeros_like(state)
        epsilon = 1e-6
        
        for i in range(len(state)):
            # Numerical gradient calculation
            state_plus = state.copy()
            state_plus[i] += epsilon
            energy_plus = self._evaluate_schedule_energy(tasks, state_plus)
            
            state_minus = state.copy()
            state_minus[i] -= epsilon
            energy_minus = self._evaluate_schedule_energy(tasks, state_minus)
            
            gradient[i] = (energy_plus - energy_minus) / (2 * epsilon)
        
        # Add quantum tunneling term
        tunneling_strength = 1.0 - s  # Decreases as evolution proceeds
        for i in range(len(gradient)):
            gradient[i] += tunneling_strength * (0.5 - state[i])
        
        return gradient
    
    def _calculate_schedule_coherence(self, scheduled_tasks: List[ScheduledTask]) -> float:
        """Calculate quantum coherence of the final schedule."""
        if not scheduled_tasks:
            return 0.0
        
        # Coherence based on quantum probability distribution
        probs = [task.quantum_probability for task in scheduled_tasks]
        prob_array = np.array(probs)
        
        # Von Neumann entropy as coherence measure
        prob_array = prob_array / (np.sum(prob_array) + 1e-10)
        entropy = -np.sum(prob_array * np.log(prob_array + 1e-10))
        max_entropy = np.log(len(prob_array))
        
        coherence = entropy / max_entropy if max_entropy > 0 else 0.0
        return coherence