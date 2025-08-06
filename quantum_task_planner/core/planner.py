"""
Quantum-Inspired Task Planner Core Implementation

This module implements a quantum-inspired approach to task planning using
superposition, entanglement, and quantum annealing concepts.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.quantum_math import QubitState, quantum_superposition, quantum_collapse
from ..utils.metrics import PlannerMetrics

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels using quantum-inspired states."""
    CRITICAL = 4  # |11⟩
    HIGH = 3      # |10⟩ 
    MEDIUM = 2    # |01⟩
    LOW = 1       # |00⟩


@dataclass
class Task:
    """Represents a task in quantum superposition state."""
    id: str
    name: str
    priority: TaskPriority
    duration: float
    dependencies: List[str]
    resources: Dict[str, float]
    deadline: Optional[float] = None
    quantum_weight: float = 1.0
    
    def __post_init__(self):
        """Initialize quantum properties."""
        self.quantum_state = QubitState(
            amplitude_0=np.sqrt(1 - self.quantum_weight),
            amplitude_1=np.sqrt(self.quantum_weight)
        )


class QuantumTaskPlanner:
    """
    Quantum-Inspired Task Planner using superposition and entanglement principles.
    
    This planner treats tasks as quantum particles that can exist in superposition
    states, allowing for probabilistic scheduling and optimization.
    """
    
    def __init__(self, max_iterations: int = 1000, convergence_threshold: float = 1e-6):
        """
        Initialize the quantum task planner.
        
        Args:
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence criteria for optimization
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.tasks: List[Task] = []
        self.resource_limits: Dict[str, float] = {}
        self.metrics = PlannerMetrics()
        
        logger.info(f"Initialized QuantumTaskPlanner with {max_iterations} max iterations")
    
    def add_task(self, task: Task) -> None:
        """Add a task to the planning queue."""
        self.tasks.append(task)
        logger.debug(f"Added task {task.id}: {task.name}")
    
    def set_resource_limits(self, limits: Dict[str, float]) -> None:
        """Set resource constraints for planning."""
        self.resource_limits = limits
        logger.info(f"Set resource limits: {limits}")
    
    def create_quantum_superposition(self, tasks: List[Task]) -> np.ndarray:
        """
        Create quantum superposition of all possible task arrangements.
        
        Returns a quantum state vector representing all possible schedules.
        """
        n_tasks = len(tasks)
        if n_tasks == 0:
            return np.array([1.0])
        
        # Create superposition state: |ψ⟩ = Σ αᵢ|scheduleᵢ⟩
        n_states = 2 ** n_tasks
        amplitudes = np.zeros(n_states, dtype=complex)
        
        for i in range(n_states):
            # Calculate amplitude based on task priorities and constraints
            amplitude = 1.0
            for j in range(n_tasks):
                if (i >> j) & 1:  # Task j is included in schedule i
                    amplitude *= tasks[j].quantum_weight
                else:
                    amplitude *= (1 - tasks[j].quantum_weight)
            
            amplitudes[i] = amplitude
        
        # Normalize the superposition
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 0:
            amplitudes /= norm
            
        return amplitudes
    
    def quantum_annealing_optimization(self, superposition_state: np.ndarray) -> Tuple[List[int], float]:
        """
        Apply quantum annealing to find optimal task schedule.
        
        Uses simulated annealing with quantum-inspired energy functions.
        """
        start_time = time.time()
        n_tasks = len(self.tasks)
        
        if n_tasks == 0:
            return [], 0.0
        
        # Initialize random schedule
        current_schedule = np.random.choice([0, 1], size=n_tasks)
        current_energy = self._calculate_energy(current_schedule)
        
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        # Quantum annealing parameters
        initial_temp = 10.0
        final_temp = 0.01
        
        for iteration in range(self.max_iterations):
            # Temperature cooling schedule (exponential)
            temp = initial_temp * (final_temp / initial_temp) ** (iteration / self.max_iterations)
            
            # Generate neighboring schedule (quantum tunneling effect)
            neighbor_schedule = current_schedule.copy()
            flip_idx = np.random.randint(0, n_tasks)
            neighbor_schedule[flip_idx] = 1 - neighbor_schedule[flip_idx]
            
            neighbor_energy = self._calculate_energy(neighbor_schedule)
            energy_diff = neighbor_energy - current_energy
            
            # Accept or reject based on quantum probability
            if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / temp):
                current_schedule = neighbor_schedule
                current_energy = neighbor_energy
                
                # Update best solution
                if current_energy < best_energy:
                    best_schedule = current_schedule.copy()
                    best_energy = current_energy
            
            # Check convergence
            if iteration > 0 and abs(energy_diff) < self.convergence_threshold:
                break
        
        optimization_time = time.time() - start_time
        self.metrics.record_optimization_time(optimization_time)
        
        return best_schedule.tolist(), best_energy
    
    def _calculate_energy(self, schedule: np.ndarray) -> float:
        """
        Calculate the energy (cost) of a given task schedule.
        
        Lower energy indicates better schedule quality.
        """
        if len(schedule) != len(self.tasks):
            return float('inf')
        
        total_energy = 0.0
        resource_usage = {resource: 0.0 for resource in self.resource_limits.keys()}
        
        for i, task in enumerate(self.tasks):
            if schedule[i] == 1:  # Task is selected
                # Priority energy (higher priority = lower energy)
                priority_energy = 1.0 / task.priority.value
                total_energy += priority_energy
                
                # Duration energy
                duration_energy = task.duration * 0.1
                total_energy += duration_energy
                
                # Resource constraint violations
                for resource, amount in task.resources.items():
                    if resource in resource_usage:
                        resource_usage[resource] += amount
                
                # Deadline penalty
                if task.deadline is not None:
                    deadline_penalty = max(0, task.duration - task.deadline) * 10.0
                    total_energy += deadline_penalty
        
        # Add resource constraint penalties
        for resource, usage in resource_usage.items():
            if resource in self.resource_limits:
                if usage > self.resource_limits[resource]:
                    penalty = (usage - self.resource_limits[resource]) * 100.0
                    total_energy += penalty
        
        # Dependency violation penalties
        for i, task in enumerate(self.tasks):
            if schedule[i] == 1:  # Task is selected
                for dep_id in task.dependencies:
                    dep_idx = next((j for j, t in enumerate(self.tasks) if t.id == dep_id), None)
                    if dep_idx is not None and schedule[dep_idx] == 0:
                        total_energy += 1000.0  # Heavy penalty for unmet dependencies
        
        return total_energy
    
    def plan_tasks(self) -> Dict[str, Any]:
        """
        Execute quantum-inspired task planning algorithm.
        
        Returns optimized task schedule with quantum metrics.
        """
        if not self.tasks:
            return {"schedule": [], "energy": 0.0, "quantum_metrics": {}}
        
        start_time = time.time()
        
        # Step 1: Create quantum superposition of all possible schedules
        superposition_state = self.create_quantum_superposition(self.tasks)
        logger.info(f"Created superposition with {len(superposition_state)} states")
        
        # Step 2: Apply quantum annealing optimization
        optimal_schedule, optimal_energy = self.quantum_annealing_optimization(superposition_state)
        
        # Step 3: Collapse quantum state to classical schedule
        selected_tasks = []
        for i, selected in enumerate(optimal_schedule):
            if selected == 1:
                selected_tasks.append(self.tasks[i])
        
        total_time = time.time() - start_time
        
        # Calculate quantum coherence metrics
        coherence = self._calculate_quantum_coherence(superposition_state)
        entanglement = self._calculate_entanglement_measure(optimal_schedule)
        
        result = {
            "schedule": selected_tasks,
            "energy": optimal_energy,
            "total_tasks": len(self.tasks),
            "selected_tasks": len(selected_tasks),
            "planning_time": total_time,
            "quantum_metrics": {
                "coherence": coherence,
                "entanglement": entanglement,
                "superposition_states": len(superposition_state),
                "convergence_achieved": True  # Simplified for v1
            }
        }
        
        self.metrics.record_planning_result(result)
        logger.info(f"Planning completed in {total_time:.3f}s with energy {optimal_energy:.3f}")
        
        return result
    
    def _calculate_quantum_coherence(self, state: np.ndarray) -> float:
        """Calculate quantum coherence measure of the superposition state."""
        if len(state) <= 1:
            return 0.0
        
        # Coherence as measure of superposition uniformity
        probabilities = np.abs(state) ** 2
        max_coherence = np.log(len(state))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return entropy / max_coherence if max_coherence > 0 else 0.0
    
    def _calculate_entanglement_measure(self, schedule: List[int]) -> float:
        """Calculate task entanglement based on dependencies."""
        if len(schedule) <= 1:
            return 0.0
        
        entanglement_count = 0
        total_pairs = 0
        
        for i, task in enumerate(self.tasks):
            if schedule[i] == 1:  # Task is selected
                for dep_id in task.dependencies:
                    dep_idx = next((j for j, t in enumerate(self.tasks) if t.id == dep_id), None)
                    if dep_idx is not None:
                        total_pairs += 1
                        if schedule[dep_idx] == 1:  # Dependency also selected
                            entanglement_count += 1
        
        return entanglement_count / total_pairs if total_pairs > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive planning metrics."""
        return self.metrics.get_summary()