"""
Quantum-Specific Monitoring for Quantum Task Planner

Specialized monitoring for quantum states, coherence, entanglement,
and quantum algorithm performance metrics.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

from ..utils.quantum_math import QubitState, quantum_fidelity
from .performance_monitor import PerformanceMonitor, PerformanceMetric, MetricType

logger = logging.getLogger(__name__)


class QuantumMetricType(Enum):
    """Types of quantum-specific metrics."""
    COHERENCE = "coherence"
    ENTANGLEMENT = "entanglement" 
    FIDELITY = "fidelity"
    DECOHERENCE_RATE = "decoherence_rate"
    GATE_ERROR_RATE = "gate_error_rate"
    QUANTUM_VOLUME = "quantum_volume"
    SUPERPOSITION_STRENGTH = "superposition_strength"
    MEASUREMENT_ACCURACY = "measurement_accuracy"


@dataclass
class QuantumState:
    """Quantum state snapshot for monitoring."""
    timestamp: float
    state_vector: np.ndarray
    coherence: float
    entanglement_measure: float
    phase_info: Dict[str, complex]
    measurement_probabilities: Dict[str, float]


@dataclass
class QuantumCircuitMetrics:
    """Quantum circuit execution metrics."""
    circuit_depth: int
    gate_count: int
    two_qubit_gate_count: int
    execution_time: float
    fidelity: float
    success_probability: float
    error_rate: float


@dataclass
class DecoherenceEvent:
    """Quantum decoherence event record."""
    timestamp: float
    initial_coherence: float
    final_coherence: float
    decoherence_rate: float
    cause: str
    affected_qubits: List[int]


class QuantumMetricsMonitor:
    """
    Specialized monitoring for quantum-specific metrics and phenomena.
    
    Tracks quantum coherence, entanglement, decoherence events, and
    provides quantum algorithm performance analysis.
    """
    
    def __init__(self,
                 coherence_threshold: float = 0.3,
                 entanglement_threshold: float = 0.5,
                 max_history: int = 1000):
        """
        Initialize quantum metrics monitor.
        
        Args:
            coherence_threshold: Alert threshold for low coherence
            entanglement_threshold: Minimum expected entanglement
            max_history: Maximum quantum state history to keep
        """
        self.coherence_threshold = coherence_threshold
        self.entanglement_threshold = entanglement_threshold
        self.max_history = max_history
        
        # Quantum state tracking
        self.quantum_states: deque = deque(maxlen=max_history)
        self.circuit_metrics: deque = deque(maxlen=max_history)
        self.decoherence_events: deque = deque(maxlen=max_history)
        
        # Real-time monitoring
        self.current_quantum_state: Optional[QuantumState] = None
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance integration
        self.performance_monitor = PerformanceMonitor()
        
        # Quantum algorithm tracking
        self.algorithm_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'executions': 0,
            'avg_fidelity': 0.0,
            'avg_coherence': 0.0,
            'success_rate': 0.0,
            'avg_execution_time': 0.0
        })
        
        # Quantum error tracking
        self.quantum_errors: Dict[str, int] = defaultdict(int)
        self.error_patterns: List[Dict[str, Any]] = []
        
        # Alerting callbacks
        self.coherence_alerts: List[Callable[[float, float], None]] = []
        self.decoherence_alerts: List[Callable[[DecoherenceEvent], None]] = []
        
        logger.info(f"Initialized QuantumMetricsMonitor with coherence threshold {coherence_threshold}")
    
    def add_coherence_alert(self, callback: Callable[[float, float], None]) -> None:
        """Add callback for coherence alerts."""
        self.coherence_alerts.append(callback)
    
    def add_decoherence_alert(self, callback: Callable[[DecoherenceEvent], None]) -> None:
        """Add callback for decoherence events."""
        self.decoherence_alerts.append(callback)
    
    def start_monitoring(self) -> None:
        """Start continuous quantum monitoring."""
        if self.is_monitoring:
            logger.warning("Quantum monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started continuous quantum monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop quantum monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped quantum monitoring")
    
    def _monitor_loop(self) -> None:
        """Main quantum monitoring loop."""
        while self.is_monitoring:
            try:
                # Check current quantum state
                if self.current_quantum_state:
                    self._analyze_quantum_state(self.current_quantum_state)
                
                # Detect decoherence events
                self._detect_decoherence_events()
                
                # Update algorithm performance metrics
                self._update_algorithm_metrics()
                
                # Sleep until next check
                time.sleep(1.0)  # Check every second for quantum phenomena
                
            except Exception as e:
                logger.error(f"Quantum monitoring error: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def record_quantum_state(self, 
                           state_vector: np.ndarray,
                           metadata: Dict[str, Any] = None) -> None:
        """
        Record a quantum state snapshot.
        
        Args:
            state_vector: Quantum state vector
            metadata: Additional state metadata
        """
        current_time = time.time()
        
        # Calculate quantum metrics
        coherence = self._calculate_coherence(state_vector)
        entanglement = self._calculate_entanglement_measure(state_vector)
        phase_info = self._extract_phase_info(state_vector)
        measurement_probs = self._calculate_measurement_probabilities(state_vector)
        
        # Create quantum state snapshot
        quantum_state = QuantumState(
            timestamp=current_time,
            state_vector=state_vector.copy(),
            coherence=coherence,
            entanglement_measure=entanglement,
            phase_info=phase_info,
            measurement_probabilities=measurement_probs
        )
        
        # Store state
        self.quantum_states.append(quantum_state)
        self.current_quantum_state = quantum_state
        
        # Record metrics in performance monitor
        self.performance_monitor.record_metric(PerformanceMetric(
            name="quantum_coherence",
            value=coherence,
            metric_type=MetricType.QUANTUM_METRIC,
            timestamp=current_time,
            unit="ratio",
            tags={"type": "coherence"}
        ))
        
        self.performance_monitor.record_metric(PerformanceMetric(
            name="quantum_entanglement",
            value=entanglement,
            metric_type=MetricType.QUANTUM_METRIC,
            timestamp=current_time,
            unit="ratio",
            tags={"type": "entanglement"}
        ))
        
        # Check for alerts
        self._check_coherence_alerts(coherence)
        
        logger.debug(f"Recorded quantum state: coherence={coherence:.3f}, entanglement={entanglement:.3f}")
    
    def record_quantum_circuit_execution(self,
                                       algorithm_name: str,
                                       circuit_metrics: QuantumCircuitMetrics) -> None:
        """
        Record quantum circuit execution metrics.
        
        Args:
            algorithm_name: Name of quantum algorithm
            circuit_metrics: Circuit execution metrics
        """
        # Store circuit metrics
        self.circuit_metrics.append(circuit_metrics)
        
        # Update algorithm performance
        algo_stats = self.algorithm_performance[algorithm_name]
        algo_stats['executions'] += 1
        
        # Update running averages
        n = algo_stats['executions']
        algo_stats['avg_fidelity'] = ((n-1) * algo_stats['avg_fidelity'] + circuit_metrics.fidelity) / n
        algo_stats['avg_execution_time'] = ((n-1) * algo_stats['avg_execution_time'] + circuit_metrics.execution_time) / n
        
        if circuit_metrics.success_probability > 0.5:  # Successful execution
            success_count = algo_stats.get('success_count', 0) + 1
            algo_stats['success_count'] = success_count
            algo_stats['success_rate'] = success_count / n
        
        # Record performance metrics
        current_time = time.time()
        
        self.performance_monitor.record_metric(PerformanceMetric(
            name=f"quantum_fidelity_{algorithm_name}",
            value=circuit_metrics.fidelity,
            metric_type=MetricType.QUANTUM_METRIC,
            timestamp=current_time,
            unit="ratio",
            tags={"algorithm": algorithm_name, "type": "fidelity"}
        ))
        
        self.performance_monitor.record_metric(PerformanceMetric(
            name=f"quantum_execution_time_{algorithm_name}",
            value=circuit_metrics.execution_time,
            metric_type=MetricType.LATENCY,
            timestamp=current_time,
            unit="seconds",
            tags={"algorithm": algorithm_name, "type": "execution_time"}
        ))
        
        logger.debug(f"Recorded {algorithm_name} execution: fidelity={circuit_metrics.fidelity:.3f}")
    
    def _calculate_coherence(self, state_vector: np.ndarray) -> float:
        """Calculate quantum coherence measure."""
        if len(state_vector) == 0:
            return 0.0
        
        # Von Neumann entropy as coherence measure
        probabilities = np.abs(state_vector) ** 2
        probabilities = probabilities[probabilities > 1e-15]  # Remove near-zero probabilities
        
        if len(probabilities) <= 1:
            return 0.0
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(probabilities))
        
        # Normalized coherence (0 = no coherence, 1 = maximum coherence)
        coherence = entropy / max_entropy if max_entropy > 0 else 0.0
        return coherence
    
    def _calculate_entanglement_measure(self, state_vector: np.ndarray) -> float:
        """Calculate entanglement measure for quantum state."""
        n_qubits = int(np.log2(len(state_vector)))
        
        if n_qubits <= 1:
            return 0.0  # No entanglement for single qubit
        
        # For 2-qubit system, calculate concurrence as entanglement measure
        if n_qubits == 2:
            return self._calculate_concurrence(state_vector)
        
        # For multi-qubit systems, use tangle as approximation
        return self._calculate_tangle(state_vector)
    
    def _calculate_concurrence(self, state_vector: np.ndarray) -> float:
        """Calculate concurrence for 2-qubit state."""
        if len(state_vector) != 4:
            return 0.0
        
        # Reshape state vector to 2x2 density matrix
        psi = state_vector.reshape((2, 2))
        
        # Calculate density matrix
        rho = np.outer(state_vector, np.conj(state_vector))
        
        # Pauli-Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # Calculate spin-flipped state
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        rho_tilde = sigma_y_tensor @ np.conj(rho) @ sigma_y_tensor
        
        # Calculate eigenvalues of rho * rho_tilde
        eigenvals = np.linalg.eigvals(rho @ rho_tilde)
        eigenvals = np.sqrt(np.maximum(eigenvals.real, 0))  # Take real part and ensure non-negative
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        
        # Concurrence
        concurrence = max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])
        return concurrence
    
    def _calculate_tangle(self, state_vector: np.ndarray) -> float:
        """Calculate tangle (generalized entanglement measure)."""
        # Simplified tangle calculation - in practice, this is very complex
        # This is an approximation based on state vector entropy
        probabilities = np.abs(state_vector) ** 2
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        n_qubits = int(np.log2(len(state_vector)))
        
        # Normalize by maximum possible entropy
        max_entropy = n_qubits
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Tangle approximation (0 = separable, 1 = maximally entangled)
        return min(1.0, normalized_entropy)
    
    def _extract_phase_info(self, state_vector: np.ndarray) -> Dict[str, complex]:
        """Extract phase information from quantum state."""
        phase_info = {}
        
        for i, amplitude in enumerate(state_vector):
            if abs(amplitude) > 1e-10:  # Only store significant amplitudes
                phase_info[f"state_{i}"] = amplitude
        
        # Calculate relative phases
        if len(phase_info) > 1:
            reference_amp = list(phase_info.values())[0]
            relative_phases = {}
            for state, amp in phase_info.items():
                relative_phase = amp / reference_amp if reference_amp != 0 else 0
                relative_phases[f"{state}_relative"] = relative_phase
            phase_info.update(relative_phases)
        
        return phase_info
    
    def _calculate_measurement_probabilities(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Calculate measurement probabilities for computational basis."""
        probabilities = {}
        
        for i, amplitude in enumerate(state_vector):
            prob = abs(amplitude) ** 2
            if prob > 1e-15:  # Only store significant probabilities
                basis_state = format(i, f'0{int(np.log2(len(state_vector)))}b')
                probabilities[f"|{basis_state}⟩"] = prob
        
        return probabilities
    
    def _analyze_quantum_state(self, quantum_state: QuantumState) -> None:
        """Analyze quantum state for interesting phenomena."""
        # Check for decoherence
        if len(self.quantum_states) > 1:
            prev_state = list(self.quantum_states)[-2]
            coherence_change = quantum_state.coherence - prev_state.coherence
            
            if coherence_change < -0.1:  # Significant coherence loss
                decoherence_event = DecoherenceEvent(
                    timestamp=quantum_state.timestamp,
                    initial_coherence=prev_state.coherence,
                    final_coherence=quantum_state.coherence,
                    decoherence_rate=coherence_change / (quantum_state.timestamp - prev_state.timestamp),
                    cause="unknown",  # Would need more sophisticated analysis
                    affected_qubits=[]  # Would need qubit-level analysis
                )
                
                self.decoherence_events.append(decoherence_event)
                self._send_decoherence_alert(decoherence_event)
        
        # Analyze superposition strength
        superposition_strength = self._calculate_superposition_strength(quantum_state.state_vector)
        self.performance_monitor.record_metric(PerformanceMetric(
            name="superposition_strength",
            value=superposition_strength,
            metric_type=MetricType.QUANTUM_METRIC,
            timestamp=quantum_state.timestamp,
            unit="ratio",
            tags={"type": "superposition"}
        ))
        
        # Check for quantum volume metrics
        quantum_volume = self._estimate_quantum_volume(quantum_state.state_vector)
        self.performance_monitor.record_metric(PerformanceMetric(
            name="quantum_volume",
            value=quantum_volume,
            metric_type=MetricType.QUANTUM_METRIC,
            timestamp=quantum_state.timestamp,
            unit="qubits",
            tags={"type": "volume"}
        ))
    
    def _calculate_superposition_strength(self, state_vector: np.ndarray) -> float:
        """Calculate strength of quantum superposition."""
        probabilities = np.abs(state_vector) ** 2
        
        # Superposition strength based on how evenly distributed the probabilities are
        # Maximum for uniform distribution, minimum for single-state
        n_states = len(probabilities)
        uniform_prob = 1.0 / n_states
        
        # Calculate distance from uniform distribution
        distance_from_uniform = np.sum(np.abs(probabilities - uniform_prob))
        max_distance = 2.0 * (1.0 - uniform_prob)  # Maximum possible distance
        
        # Superposition strength (1 = maximum superposition, 0 = no superposition)
        strength = 1.0 - (distance_from_uniform / max_distance)
        return max(0.0, min(1.0, strength))
    
    def _estimate_quantum_volume(self, state_vector: np.ndarray) -> float:
        """Estimate quantum volume of the system."""
        # Simplified quantum volume estimation
        n_qubits = int(np.log2(len(state_vector)))
        
        # Factor in coherence and entanglement
        coherence = self._calculate_coherence(state_vector)
        entanglement = self._calculate_entanglement_measure(state_vector)
        
        # Effective quantum volume considering decoherence
        effective_qubits = n_qubits * coherence * (1.0 + entanglement)
        return effective_qubits
    
    def _detect_decoherence_events(self) -> None:
        """Detect and analyze decoherence patterns."""
        if len(self.quantum_states) < 3:
            return
        
        # Analyze coherence trend over recent states
        recent_states = list(self.quantum_states)[-5:]  # Last 5 states
        coherence_values = [state.coherence for state in recent_states]
        
        # Check for consistent coherence decline
        if len(coherence_values) >= 3:
            declining_trend = all(
                coherence_values[i] < coherence_values[i-1] 
                for i in range(1, len(coherence_values))
            )
            
            if declining_trend and coherence_values[-1] < self.coherence_threshold:
                # Significant decoherence event
                rate = (coherence_values[0] - coherence_values[-1]) / (
                    recent_states[-1].timestamp - recent_states[0].timestamp
                )
                
                event = DecoherenceEvent(
                    timestamp=time.time(),
                    initial_coherence=coherence_values[0],
                    final_coherence=coherence_values[-1],
                    decoherence_rate=rate,
                    cause="trend_analysis",
                    affected_qubits=[]
                )
                
                self.decoherence_events.append(event)
                logger.warning(f"Decoherence trend detected: rate={rate:.4f}/s")
    
    def _update_algorithm_metrics(self) -> None:
        """Update quantum algorithm performance metrics."""
        for algorithm_name, stats in self.algorithm_performance.items():
            if stats['executions'] > 0:
                # Record aggregate metrics
                current_time = time.time()
                
                self.performance_monitor.record_metric(PerformanceMetric(
                    name=f"algorithm_avg_fidelity_{algorithm_name}",
                    value=stats['avg_fidelity'],
                    metric_type=MetricType.QUANTUM_METRIC,
                    timestamp=current_time,
                    unit="ratio",
                    tags={"algorithm": algorithm_name, "metric": "avg_fidelity"}
                ))
                
                self.performance_monitor.record_metric(PerformanceMetric(
                    name=f"algorithm_success_rate_{algorithm_name}",
                    value=stats['success_rate'],
                    metric_type=MetricType.SUCCESS_RATE,
                    timestamp=current_time,
                    unit="ratio",
                    tags={"algorithm": algorithm_name, "metric": "success_rate"}
                ))
    
    def _check_coherence_alerts(self, coherence: float) -> None:
        """Check if coherence is below threshold and send alerts."""
        if coherence < self.coherence_threshold:
            for callback in self.coherence_alerts:
                try:
                    callback(coherence, self.coherence_threshold)
                except Exception as e:
                    logger.error(f"Coherence alert callback failed: {e}")
    
    def _send_decoherence_alert(self, event: DecoherenceEvent) -> None:
        """Send decoherence event alert."""
        logger.warning(f"Decoherence event: {event.initial_coherence:.3f} → {event.final_coherence:.3f}")
        
        for callback in self.decoherence_alerts:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Decoherence alert callback failed: {e}")
    
    def get_quantum_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive quantum metrics summary."""
        current_time = time.time()
        
        summary = {
            'timestamp': current_time,
            'current_state': {},
            'algorithm_performance': dict(self.algorithm_performance),
            'decoherence_events': len(self.decoherence_events),
            'quantum_errors': dict(self.quantum_errors),
            'monitoring_active': self.is_monitoring
        }
        
        # Current quantum state info
        if self.current_quantum_state:
            state = self.current_quantum_state
            summary['current_state'] = {
                'coherence': state.coherence,
                'entanglement': state.entanglement_measure,
                'n_qubits': int(np.log2(len(state.state_vector))),
                'significant_amplitudes': len([a for a in state.state_vector if abs(a) > 0.1]),
                'measurement_entropy': -sum(p * np.log2(p + 1e-15) for p in state.measurement_probabilities.values())
            }
        
        # Recent decoherence events
        if self.decoherence_events:
            recent_events = [e for e in self.decoherence_events if current_time - e.timestamp < 3600]
            summary['recent_decoherence_events'] = len(recent_events)
            
            if recent_events:
                avg_decoherence_rate = np.mean([e.decoherence_rate for e in recent_events])
                summary['avg_decoherence_rate_1h'] = avg_decoherence_rate
        
        # Circuit execution statistics
        if self.circuit_metrics:
            recent_circuits = [c for c in self.circuit_metrics if current_time - 3600 < current_time]  # Last hour
            
            if recent_circuits:
                summary['circuit_stats_1h'] = {
                    'executions': len(recent_circuits),
                    'avg_fidelity': np.mean([c.fidelity for c in recent_circuits]),
                    'avg_execution_time': np.mean([c.execution_time for c in recent_circuits]),
                    'avg_gate_count': np.mean([c.gate_count for c in recent_circuits]),
                    'avg_error_rate': np.mean([c.error_rate for c in recent_circuits])
                }
        
        return summary
    
    def export_quantum_data(self, filepath: str) -> None:
        """Export quantum monitoring data to file."""
        import json
        
        data = {
            'quantum_metrics_summary': self.get_quantum_metrics_summary(),
            'quantum_states': [
                {
                    'timestamp': state.timestamp,
                    'coherence': state.coherence,
                    'entanglement': state.entanglement_measure,
                    'measurement_probabilities': state.measurement_probabilities
                }
                for state in list(self.quantum_states)[-100:]  # Last 100 states
            ],
            'decoherence_events': [
                {
                    'timestamp': event.timestamp,
                    'initial_coherence': event.initial_coherence,
                    'final_coherence': event.final_coherence,
                    'decoherence_rate': event.decoherence_rate,
                    'cause': event.cause
                }
                for event in list(self.decoherence_events)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported quantum monitoring data to {filepath}")