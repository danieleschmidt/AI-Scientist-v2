"""
Quantum Optimization Engine

Advanced quantum-inspired optimization algorithms for task scheduling,
resource allocation, and performance optimization.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from quantum optimization process."""
    solution: np.ndarray
    objective_value: float
    iterations: int
    convergence_time: float
    quantum_metrics: Dict[str, float]


class QuantumGate:
    """Quantum gate operations for optimization algorithms."""
    
    @staticmethod
    def hadamard(state: np.ndarray) -> np.ndarray:
        """Apply Hadamard gate to create superposition."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return H @ state
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Y-rotation gate for quantum state manipulation."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]])


class QuantumOptimizer:
    """
    Quantum-inspired optimization engine using variational quantum algorithms.
    
    Implements quantum approximate optimization algorithm (QAOA) concepts
    for classical optimization problems.
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8,
                 learning_rate: float = 0.1):
        """
        Initialize quantum optimizer.
        
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            learning_rate: Learning rate for parameter updates
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.optimization_history = []
        
        logger.info("Initialized QuantumOptimizer with QAOA-inspired algorithm")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float],
                initial_solution: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """
        Execute quantum-inspired optimization.
        
        Args:
            objective_function: Function to optimize
            initial_solution: Starting point for optimization
            bounds: Variable bounds [(min, max), ...]
            
        Returns:
            OptimizationResult with solution and metrics
        """
        start_time = time.time()
        
        # Initialize quantum parameters
        n_vars = len(initial_solution)
        beta = np.random.random(n_vars) * np.pi  # Mixing angles
        gamma = np.random.random(n_vars) * np.pi  # Phase angles
        
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        best_value = objective_function(best_solution)
        
        self.optimization_history = []
        
        for iteration in range(self.max_iterations):
            # Quantum variational optimization step
            quantum_solution = self._qaoa_step(current_solution, beta, gamma, objective_function)
            
            # Apply bounds constraints if provided
            if bounds:
                quantum_solution = self._apply_bounds(quantum_solution, bounds)
            
            current_value = objective_function(quantum_solution)
            
            # Update best solution
            if current_value < best_value:
                best_solution = quantum_solution.copy()
                best_value = current_value
            
            # Update quantum parameters using gradient descent
            beta, gamma = self._update_parameters(beta, gamma, quantum_solution, objective_function)
            
            # Record progress
            self.optimization_history.append({
                'iteration': iteration,
                'objective_value': current_value,
                'best_value': best_value
            })
            
            # Check convergence
            if iteration > 0:
                improvement = abs(self.optimization_history[-2]['best_value'] - best_value)
                if improvement < self.tolerance:
                    logger.info(f"Converged at iteration {iteration} with improvement {improvement}")
                    break
            
            current_solution = quantum_solution
        
        optimization_time = time.time() - start_time
        
        # Calculate quantum metrics
        quantum_metrics = self._calculate_quantum_metrics(beta, gamma, best_solution)
        
        result = OptimizationResult(
            solution=best_solution,
            objective_value=best_value,
            iterations=iteration + 1,
            convergence_time=optimization_time,
            quantum_metrics=quantum_metrics
        )
        
        logger.info(f"Optimization completed in {optimization_time:.3f}s with value {best_value:.6f}")
        return result
    
    def _qaoa_step(self, 
                   solution: np.ndarray, 
                   beta: np.ndarray, 
                   gamma: np.ndarray,
                   objective_function: Callable) -> np.ndarray:
        """
        Execute one QAOA (Quantum Approximate Optimization Algorithm) step.
        
        This applies the quantum circuit consisting of:
        1. Problem Hamiltonian (phase separation)
        2. Mixer Hamiltonian (quantum tunneling)
        """
        n_vars = len(solution)
        
        # Initialize quantum state in uniform superposition
        quantum_state = np.ones(2**n_vars) / np.sqrt(2**n_vars)
        
        # Apply phase separation operator exp(-iγH_C)
        for i in range(n_vars):
            phase_factor = gamma[i] * solution[i]
            quantum_state = self._apply_phase_gate(quantum_state, i, phase_factor)
        
        # Apply mixer operator exp(-iβH_B) 
        for i in range(n_vars):
            mixing_angle = beta[i]
            quantum_state = self._apply_rotation_gate(quantum_state, i, mixing_angle)
        
        # Measure expectation value and collapse to classical solution
        new_solution = self._quantum_measurement(quantum_state, n_vars)
        
        return new_solution
    
    def _apply_phase_gate(self, state: np.ndarray, qubit_idx: int, phase: float) -> np.ndarray:
        """Apply phase gate to specific qubit in quantum state."""
        n_qubits = int(np.log2(len(state)))
        new_state = state.copy()
        
        for i in range(len(state)):
            if (i >> qubit_idx) & 1:  # Qubit is in |1⟩ state
                new_state[i] *= np.exp(-1j * phase)
        
        return new_state
    
    def _apply_rotation_gate(self, state: np.ndarray, qubit_idx: int, angle: float) -> np.ndarray:
        """Apply rotation gate for quantum tunneling effect."""
        n_qubits = int(np.log2(len(state)))
        new_state = np.zeros_like(state)
        
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        for i in range(len(state)):
            if (i >> qubit_idx) & 1:  # Qubit is in |1⟩
                # Apply rotation: |1⟩ → cos(θ/2)|1⟩ - sin(θ/2)|0⟩
                j = i ^ (1 << qubit_idx)  # Flip qubit
                new_state[i] += cos_half * state[i]
                new_state[j] -= sin_half * state[i]
            else:  # Qubit is in |0⟩
                # Apply rotation: |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                j = i ^ (1 << qubit_idx)  # Flip qubit
                new_state[i] += cos_half * state[i]
                new_state[j] += sin_half * state[i]
        
        return new_state
    
    def _quantum_measurement(self, quantum_state: np.ndarray, n_vars: int) -> np.ndarray:
        """Measure quantum state to obtain classical solution."""
        probabilities = np.abs(quantum_state) ** 2
        
        # Sample from probability distribution
        measured_state_idx = np.random.choice(len(probabilities), p=probabilities)
        
        # Convert binary state index to solution vector
        solution = np.zeros(n_vars)
        for i in range(n_vars):
            if (measured_state_idx >> i) & 1:
                solution[i] = 1.0
            else:
                solution[i] = 0.0
        
        # Add continuous variation based on quantum amplitudes
        for i in range(n_vars):
            # Calculate expected value for each variable
            expectation = 0.0
            for j in range(len(quantum_state)):
                if (j >> i) & 1:
                    expectation += np.abs(quantum_state[j]) ** 2
            
            # Smooth transition from discrete to continuous
            solution[i] = 0.7 * solution[i] + 0.3 * expectation
        
        return solution
    
    def _update_parameters(self, 
                          beta: np.ndarray, 
                          gamma: np.ndarray, 
                          solution: np.ndarray,
                          objective_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Update quantum parameters using parameter-shift rule."""
        # Approximate gradient calculation for parameter updates
        epsilon = 1e-6
        
        # Update beta parameters
        new_beta = beta.copy()
        for i in range(len(beta)):
            # Forward difference
            beta_plus = beta.copy()
            beta_plus[i] += epsilon
            sol_plus = self._qaoa_step(solution, beta_plus, gamma, objective_function)
            val_plus = objective_function(sol_plus)
            
            # Backward difference
            beta_minus = beta.copy()
            beta_minus[i] -= epsilon
            sol_minus = self._qaoa_step(solution, beta_minus, gamma, objective_function)
            val_minus = objective_function(sol_minus)
            
            # Gradient approximation
            gradient = (val_plus - val_minus) / (2 * epsilon)
            new_beta[i] -= self.learning_rate * gradient
        
        # Update gamma parameters
        new_gamma = gamma.copy()
        for i in range(len(gamma)):
            # Forward difference
            gamma_plus = gamma.copy()
            gamma_plus[i] += epsilon
            sol_plus = self._qaoa_step(solution, beta, gamma_plus, objective_function)
            val_plus = objective_function(sol_plus)
            
            # Backward difference
            gamma_minus = gamma.copy()
            gamma_minus[i] -= epsilon
            sol_minus = self._qaoa_step(solution, beta, gamma_minus, objective_function)
            val_minus = objective_function(sol_minus)
            
            # Gradient approximation
            gradient = (val_plus - val_minus) / (2 * epsilon)
            new_gamma[i] -= self.learning_rate * gradient
        
        # Keep parameters in valid range [0, 2π]
        new_beta = np.mod(new_beta, 2 * np.pi)
        new_gamma = np.mod(new_gamma, 2 * np.pi)
        
        return new_beta, new_gamma
    
    def _apply_bounds(self, solution: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Apply variable bounds constraints."""
        bounded_solution = solution.copy()
        
        for i, (lower, upper) in enumerate(bounds):
            if i < len(solution):
                bounded_solution[i] = np.clip(solution[i], lower, upper)
        
        return bounded_solution
    
    def _calculate_quantum_metrics(self, 
                                  beta: np.ndarray, 
                                  gamma: np.ndarray, 
                                  solution: np.ndarray) -> Dict[str, float]:
        """Calculate quantum-specific optimization metrics."""
        metrics = {}
        
        # Parameter variance (measure of quantum coherence)
        metrics['beta_variance'] = np.var(beta)
        metrics['gamma_variance'] = np.var(gamma)
        
        # Solution entropy
        prob_dist = np.abs(solution) / (np.sum(np.abs(solution)) + 1e-10)
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        metrics['solution_entropy'] = entropy
        
        # Quantum fidelity approximation
        fidelity = np.exp(-np.sum(beta**2 + gamma**2) / len(solution))
        metrics['quantum_fidelity'] = fidelity
        
        # Convergence rate
        if len(self.optimization_history) > 1:
            initial_value = self.optimization_history[0]['objective_value']
            final_value = self.optimization_history[-1]['objective_value']
            improvement_rate = abs(final_value - initial_value) / len(self.optimization_history)
            metrics['convergence_rate'] = improvement_rate
        
        return metrics
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get detailed optimization history."""
        return self.optimization_history.copy()