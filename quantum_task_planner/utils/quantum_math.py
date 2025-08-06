"""
Quantum Mathematics Utilities

Mathematical functions and classes for quantum-inspired computations
including qubit states, superposition, and quantum operations.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import cmath


@dataclass
class QubitState:
    """Represents a quantum qubit state |ψ⟩ = α|0⟩ + β|1⟩."""
    amplitude_0: complex = 1.0  # α coefficient
    amplitude_1: complex = 0.0  # β coefficient
    
    def __post_init__(self):
        """Normalize the quantum state."""
        self.normalize()
    
    def normalize(self) -> None:
        """Normalize the quantum state to unit probability."""
        norm = np.sqrt(abs(self.amplitude_0)**2 + abs(self.amplitude_1)**2)
        if norm > 0:
            self.amplitude_0 /= norm
            self.amplitude_1 /= norm
    
    @property
    def probability_0(self) -> float:
        """Probability of measuring |0⟩."""
        return abs(self.amplitude_0)**2
    
    @property 
    def probability_1(self) -> float:
        """Probability of measuring |1⟩."""
        return abs(self.amplitude_1)**2
    
    def measure(self) -> int:
        """Perform quantum measurement, collapsing to classical state."""
        if np.random.random() < self.probability_0:
            return 0
        else:
            return 1
    
    def apply_gate(self, gate_matrix: np.ndarray) -> 'QubitState':
        """Apply quantum gate operation."""
        state_vector = np.array([self.amplitude_0, self.amplitude_1])
        new_state = gate_matrix @ state_vector
        
        return QubitState(
            amplitude_0=new_state[0],
            amplitude_1=new_state[1]
        )
    
    def phase_shift(self, phase: float) -> 'QubitState':
        """Apply phase shift to |1⟩ state."""
        return QubitState(
            amplitude_0=self.amplitude_0,
            amplitude_1=self.amplitude_1 * cmath.exp(1j * phase)
        )
    
    def hadamard(self) -> 'QubitState':
        """Apply Hadamard gate to create superposition."""
        hadamard_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return self.apply_gate(hadamard_gate)
    
    def pauli_x(self) -> 'QubitState':
        """Apply Pauli-X (NOT) gate."""
        pauli_x_gate = np.array([[0, 1], [1, 0]])
        return self.apply_gate(pauli_x_gate)
    
    def pauli_y(self) -> 'QubitState':
        """Apply Pauli-Y gate."""
        pauli_y_gate = np.array([[0, -1j], [1j, 0]])
        return self.apply_gate(pauli_y_gate)
    
    def pauli_z(self) -> 'QubitState':
        """Apply Pauli-Z gate."""
        pauli_z_gate = np.array([[1, 0], [0, -1]])
        return self.apply_gate(pauli_z_gate)
    
    def __str__(self) -> str:
        """String representation of quantum state."""
        return f"({self.amplitude_0:.3f})|0⟩ + ({self.amplitude_1:.3f})|1⟩"


def quantum_superposition(states: List[QubitState], weights: Optional[List[float]] = None) -> QubitState:
    """
    Create quantum superposition of multiple qubit states.
    
    Args:
        states: List of QubitState objects
        weights: Optional weights for superposition (normalized automatically)
        
    Returns:
        QubitState representing the superposition
    """
    if not states:
        return QubitState()
    
    if weights is None:
        weights = [1.0] * len(states)
    
    if len(weights) != len(states):
        raise ValueError("Number of weights must match number of states")
    
    # Normalize weights
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]
    else:
        weights = [1.0 / len(states)] * len(states)
    
    # Create superposition
    total_amp_0 = sum(w * s.amplitude_0 for w, s in zip(weights, states))
    total_amp_1 = sum(w * s.amplitude_1 for w, s in zip(weights, states))
    
    return QubitState(amplitude_0=total_amp_0, amplitude_1=total_amp_1)


def quantum_collapse(state: QubitState, measurement_basis: str = "computational") -> Tuple[int, QubitState]:
    """
    Perform quantum measurement and collapse the state.
    
    Args:
        state: QubitState to measure
        measurement_basis: Measurement basis ("computational", "hadamard")
        
    Returns:
        Tuple of (measurement_result, collapsed_state)
    """
    if measurement_basis == "computational":
        # Standard computational basis measurement
        measurement = state.measure()
        if measurement == 0:
            collapsed_state = QubitState(amplitude_0=1.0, amplitude_1=0.0)
        else:
            collapsed_state = QubitState(amplitude_0=0.0, amplitude_1=1.0)
        
        return measurement, collapsed_state
    
    elif measurement_basis == "hadamard":
        # Hadamard basis measurement (X-basis)
        # First rotate to computational basis
        rotated_state = state.hadamard()
        measurement = rotated_state.measure()
        
        if measurement == 0:
            # |+⟩ = (|0⟩ + |1⟩)/√2
            collapsed_state = QubitState(
                amplitude_0=1/np.sqrt(2), 
                amplitude_1=1/np.sqrt(2)
            )
        else:
            # |-⟩ = (|0⟩ - |1⟩)/√2
            collapsed_state = QubitState(
                amplitude_0=1/np.sqrt(2), 
                amplitude_1=-1/np.sqrt(2)
            )
        
        return measurement, collapsed_state
    
    else:
        raise ValueError(f"Unknown measurement basis: {measurement_basis}")


def quantum_entanglement(state1: QubitState, state2: QubitState) -> np.ndarray:
    """
    Create entangled two-qubit state from individual qubit states.
    
    Args:
        state1: First qubit state
        state2: Second qubit state
        
    Returns:
        4D complex array representing entangled state |ψ⟩ = Σ cᵢⱼ|ij⟩
    """
    # Tensor product of individual states
    entangled_state = np.kron(
        [state1.amplitude_0, state1.amplitude_1],
        [state2.amplitude_0, state2.amplitude_1]
    )
    
    return entangled_state


def bell_state(bell_type: str = "phi_plus") -> np.ndarray:
    """
    Generate Bell state (maximally entangled two-qubit state).
    
    Args:
        bell_type: Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
        
    Returns:
        4D complex array representing Bell state
    """
    if bell_type == "phi_plus":
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        return np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    elif bell_type == "phi_minus":
        # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
        return np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])
    elif bell_type == "psi_plus":
        # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
        return np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
    elif bell_type == "psi_minus":
        # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        return np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
    else:
        raise ValueError(f"Unknown Bell state type: {bell_type}")


def quantum_fidelity(state1: Union[QubitState, np.ndarray], state2: Union[QubitState, np.ndarray]) -> float:
    """
    Calculate quantum fidelity between two quantum states.
    
    Fidelity F = |⟨ψ₁|ψ₂⟩|²
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity value between 0 and 1
    """
    # Convert QubitState to vector if needed
    if isinstance(state1, QubitState):
        vec1 = np.array([state1.amplitude_0, state1.amplitude_1])
    else:
        vec1 = np.array(state1)
    
    if isinstance(state2, QubitState):
        vec2 = np.array([state2.amplitude_0, state2.amplitude_1])
    else:
        vec2 = np.array(state2)
    
    # Calculate inner product
    inner_product = np.conj(vec1) @ vec2
    fidelity = abs(inner_product) ** 2
    
    return float(fidelity)


def quantum_distance(state1: Union[QubitState, np.ndarray], state2: Union[QubitState, np.ndarray]) -> float:
    """
    Calculate trace distance between two quantum states.
    
    Distance = 1 - Fidelity
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Distance value between 0 and 1
    """
    fidelity = quantum_fidelity(state1, state2)
    return 1.0 - fidelity


def bloch_coordinates(state: QubitState) -> Tuple[float, float, float]:
    """
    Calculate Bloch sphere coordinates for qubit state.
    
    Args:
        state: QubitState to convert
        
    Returns:
        Tuple of (x, y, z) coordinates on Bloch sphere
    """
    α = state.amplitude_0
    β = state.amplitude_1
    
    # Bloch vector components
    x = 2 * np.real(np.conj(α) * β)
    y = 2 * np.imag(np.conj(α) * β)
    z = abs(α)**2 - abs(β)**2
    
    return float(x), float(y), float(z)


def quantum_phase(state: QubitState) -> float:
    """
    Extract global phase from quantum state.
    
    Args:
        state: QubitState to analyze
        
    Returns:
        Global phase in radians
    """
    if abs(state.amplitude_0) > 1e-10:
        phase = np.angle(state.amplitude_0)
    elif abs(state.amplitude_1) > 1e-10:
        phase = np.angle(state.amplitude_1)
    else:
        phase = 0.0
    
    return float(phase)


def random_qubit_state() -> QubitState:
    """Generate random normalized qubit state."""
    # Random angles on Bloch sphere
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    
    # Convert to amplitudes
    amplitude_0 = np.cos(theta/2)
    amplitude_1 = np.sin(theta/2) * cmath.exp(1j * phi)
    
    return QubitState(amplitude_0=amplitude_0, amplitude_1=amplitude_1)