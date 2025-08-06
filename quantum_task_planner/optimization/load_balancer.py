"""
Quantum-Inspired Load Balancer

Advanced load balancing system using quantum algorithms for optimal
distribution of quantum task planning operations across resources.
"""

import time
import random
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    ADAPTIVE_QUANTUM = "adaptive_quantum"
    PREDICTIVE = "predictive"


class NodeStatus(Enum):
    """Node status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class LoadBalancerNode:
    """Node in the load balancer."""
    id: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    created_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    quantum_score: float = 0.5  # Quantum-inspired performance score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate node success rate."""
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def connection_utilization(self) -> float:
        """Calculate connection utilization ratio."""
        return self.current_connections / max(self.max_connections, 1)
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for requests."""
        return (self.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED] and
                self.current_connections < self.max_connections)


@dataclass
class RequestMetrics:
    """Request routing metrics."""
    node_id: str
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    response_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get request duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class LoadBalancerStats:
    """Load balancer statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    node_count: int = 0
    healthy_nodes: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.failed_requests / max(self.total_requests, 1)


class QuantumLoadBalancer:
    """
    Quantum-inspired load balancer with advanced routing algorithms.
    
    Uses quantum superposition and entanglement principles for optimal
    request distribution across quantum task planning nodes.
    """
    
    def __init__(self,
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_QUANTUM,
                 health_check_interval: float = 30.0,
                 enable_circuit_breaker: bool = True):
        """
        Initialize quantum load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Interval between health checks (seconds)
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Node management
        self.nodes: Dict[str, LoadBalancerNode] = {}
        self.node_order: List[str] = []  # For round-robin
        self.current_node_index: int = 0
        
        # Request tracking
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.request_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = LoadBalancerStats()
        
        # Quantum-inspired state
        self.quantum_state_vector: np.ndarray = np.array([1.0])
        self.entanglement_matrix: np.ndarray = np.eye(1)
        self.coherence_factor: float = 1.0
        
        # Performance prediction
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prediction_weights: Dict[str, float] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background processing
        self.is_running = False
        self.health_check_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.node_failure_callbacks: List[Callable[[LoadBalancerNode], None]] = []
        self.request_callbacks: List[Callable[[RequestMetrics], None]] = []
        
        logger.info(f"Initialized QuantumLoadBalancer with {strategy.value} strategy")
    
    def start(self) -> None:
        """Start the load balancer."""
        if self.is_running:
            logger.warning("Load balancer already running")
            return
        
        self.is_running = True
        
        # Start health checking
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        # Start metrics processing
        self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self.metrics_thread.start()
        
        logger.info("Started quantum load balancer")
    
    def stop(self) -> None:
        """Stop the load balancer."""
        self.is_running = False
        
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5.0)
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)
        
        logger.info("Stopped quantum load balancer")
    
    def add_node(self,
                node_id: str,
                endpoint: str,
                weight: float = 1.0,
                max_connections: int = 100,
                metadata: Dict[str, Any] = None) -> bool:
        """
        Add node to load balancer.
        
        Args:
            node_id: Unique node identifier
            endpoint: Node endpoint/address
            weight: Node weight for weighted algorithms
            max_connections: Maximum concurrent connections
            metadata: Additional node metadata
            
        Returns:
            True if node was added successfully
        """
        with self.lock:
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already exists")
                return False
            
            node = LoadBalancerNode(
                id=node_id,
                endpoint=endpoint,
                weight=weight,
                max_connections=max_connections,
                metadata=metadata or {}
            )
            
            self.nodes[node_id] = node
            self.node_order.append(node_id)
            
            # Update quantum state
            self._update_quantum_state()
            
            # Update statistics
            self.stats.node_count = len(self.nodes)
            self.stats.healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
            
            logger.info(f"Added node {node_id} at {endpoint}")
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node from load balancer.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node was removed successfully
        """
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            # Remove from nodes and order
            self.nodes.pop(node_id)
            if node_id in self.node_order:
                self.node_order.remove(node_id)
            
            # Adjust current index if needed
            if self.current_node_index >= len(self.node_order):
                self.current_node_index = 0
            
            # Update quantum state
            self._update_quantum_state()
            
            # Update statistics
            self.stats.node_count = len(self.nodes)
            self.stats.healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
            
            logger.info(f"Removed node {node_id}")
            return True
    
    def route_request(self, request_id: str, request_data: Dict[str, Any] = None) -> Optional[str]:
        """
        Route request to optimal node.
        
        Args:
            request_id: Unique request identifier
            request_data: Request data for routing decisions
            
        Returns:
            Selected node ID or None if no nodes available
        """
        with self.lock:
            # Get available nodes
            available_nodes = [
                node for node in self.nodes.values() 
                if node.is_available
            ]
            
            if not available_nodes:
                logger.warning("No available nodes for request routing")
                return None
            
            # Select node based on strategy
            selected_node = self._select_node(available_nodes, request_data)
            
            if not selected_node:
                return None
            
            # Create request metrics
            request_metrics = RequestMetrics(
                node_id=selected_node.id,
                request_id=request_id,
                start_time=time.time()
            )
            
            # Update node and global statistics
            selected_node.current_connections += 1
            selected_node.total_requests += 1
            self.stats.total_requests += 1
            self.stats.active_connections += 1
            
            # Track active request
            self.active_requests[request_id] = request_metrics
            
            logger.debug(f"Routed request {request_id} to node {selected_node.id}")
            return selected_node.id
    
    def complete_request(self,
                        request_id: str,
                        success: bool = True,
                        response_time: Optional[float] = None,
                        error: Optional[str] = None) -> None:
        """
        Mark request as completed and update metrics.
        
        Args:
            request_id: Request identifier
            success: Whether request was successful
            response_time: Request response time
            error: Error message if failed
        """
        with self.lock:
            if request_id not in self.active_requests:
                logger.warning(f"No active request found: {request_id}")
                return
            
            request_metrics = self.active_requests.pop(request_id)
            request_metrics.end_time = time.time()
            request_metrics.success = success
            request_metrics.error = error
            
            # Calculate response time
            if response_time is not None:
                request_metrics.response_time = response_time
            elif request_metrics.duration:
                request_metrics.response_time = request_metrics.duration
            
            # Update node statistics
            node = self.nodes.get(request_metrics.node_id)
            if node:
                node.current_connections = max(0, node.current_connections - 1)
                
                if success:
                    node.successful_requests += 1
                    self.stats.successful_requests += 1
                else:
                    node.failed_requests += 1
                    self.stats.failed_requests += 1
                
                # Update response time
                if request_metrics.response_time:
                    node.last_response_time = request_metrics.response_time
                    self._update_avg_response_time(node, request_metrics.response_time)
                
                # Update quantum score
                self._update_node_quantum_score(node)
            
            # Update global statistics
            self.stats.active_connections = max(0, self.stats.active_connections - 1)
            
            if request_metrics.response_time:
                self.stats.total_response_time += request_metrics.response_time
                completed_requests = self.stats.successful_requests + self.stats.failed_requests
                self.stats.avg_response_time = self.stats.total_response_time / max(completed_requests, 1)
            
            # Store in history
            self.request_history.append(request_metrics)
            
            # Record load history for prediction
            if node:
                self.load_history[node.id].append(node.connection_utilization)
            
            # Trigger callbacks
            for callback in self.request_callbacks:
                try:
                    callback(request_metrics)
                except Exception as e:
                    logger.error(f"Request callback failed: {e}")
            
            logger.debug(f"Completed request {request_id}: success={success}")
    
    def _select_node(self, available_nodes: List[LoadBalancerNode], request_data: Dict[str, Any] = None) -> Optional[LoadBalancerNode]:
        """Select optimal node based on strategy."""
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.QUANTUM_SUPERPOSITION:
            return self._quantum_superposition_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE_QUANTUM:
            return self._adaptive_quantum_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
            return self._predictive_selection(available_nodes)
        else:
            # Default to round robin
            return self._round_robin_selection(available_nodes)
    
    def _round_robin_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Simple round-robin selection."""
        if not self.node_order:
            return available_nodes[0]
        
        # Find next available node in order
        attempts = 0
        while attempts < len(self.node_order):
            node_id = self.node_order[self.current_node_index]
            self.current_node_index = (self.current_node_index + 1) % len(self.node_order)
            
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node in available_nodes:
                    return node
            
            attempts += 1
        
        # Fallback to first available node
        return available_nodes[0]
    
    def _weighted_round_robin_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Weighted round-robin selection."""
        total_weight = sum(node.weight for node in available_nodes)
        if total_weight == 0:
            return available_nodes[0]
        
        # Generate random number for selection
        rand_val = random.random() * total_weight
        cumulative_weight = 0
        
        for node in available_nodes:
            cumulative_weight += node.weight
            if rand_val <= cumulative_weight:
                return node
        
        # Fallback
        return available_nodes[-1]
    
    def _least_connections_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Select node with least connections."""
        return min(available_nodes, key=lambda n: n.current_connections)
    
    def _least_response_time_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Select node with lowest average response time."""
        return min(available_nodes, key=lambda n: n.avg_response_time)
    
    def _quantum_superposition_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Select node using quantum superposition principle."""
        if not available_nodes:
            return None
        
        # Create quantum probability distribution
        quantum_probs = []
        
        for node in available_nodes:
            # Factors: inverse connection utilization, success rate, quantum score
            util_factor = 1.0 - node.connection_utilization
            success_factor = node.success_rate
            quantum_factor = node.quantum_score
            
            # Quantum probability with coherence
            prob = (0.4 * util_factor + 0.3 * success_factor + 0.3 * quantum_factor) * self.coherence_factor
            quantum_probs.append(max(0.1, prob))  # Minimum probability
        
        # Normalize probabilities
        total_prob = sum(quantum_probs)
        if total_prob > 0:
            quantum_probs = [p / total_prob for p in quantum_probs]
        else:
            quantum_probs = [1.0 / len(available_nodes)] * len(available_nodes)
        
        # Quantum measurement (stochastic selection)
        rand_val = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(quantum_probs):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return available_nodes[i]
        
        # Fallback
        return available_nodes[-1]
    
    def _adaptive_quantum_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Adaptive quantum selection that learns from performance."""
        if not available_nodes:
            return None
        
        # Adapt strategy based on current performance
        current_success_rate = self.stats.success_rate
        current_avg_response_time = self.stats.avg_response_time
        
        # If performance is good, use quantum superposition
        if current_success_rate > 0.95 and current_avg_response_time < 1.0:
            return self._quantum_superposition_selection(available_nodes)
        
        # If high error rate, use least connections
        elif current_success_rate < 0.9:
            return self._least_connections_selection(available_nodes)
        
        # If slow response, use least response time
        elif current_avg_response_time > 2.0:
            return self._least_response_time_selection(available_nodes)
        
        # Default to quantum superposition
        else:
            return self._quantum_superposition_selection(available_nodes)
    
    def _predictive_selection(self, available_nodes: List[LoadBalancerNode]) -> LoadBalancerNode:
        """Predictive selection based on historical patterns."""
        if not available_nodes:
            return None
        
        # Predict future load for each node
        node_predictions = {}
        
        for node in available_nodes:
            if node.id in self.load_history and len(self.load_history[node.id]) > 5:
                # Simple linear prediction
                recent_loads = list(self.load_history[node.id])[-10:]
                trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
                
                # Predict next load point
                predicted_load = recent_loads[-1] + trend
                predicted_load = max(0.0, min(1.0, predicted_load))  # Clamp to [0, 1]
                
                node_predictions[node.id] = predicted_load
            else:
                # No history, use current utilization
                node_predictions[node.id] = node.connection_utilization
        
        # Select node with lowest predicted load
        best_node = min(available_nodes, key=lambda n: node_predictions.get(n.id, 1.0))
        return best_node
    
    def _update_quantum_state(self) -> None:
        """Update quantum state vector based on current nodes."""
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            self.quantum_state_vector = np.array([1.0])
            self.entanglement_matrix = np.eye(1)
            return
        
        # Create superposition state for all nodes
        node_amplitudes = []
        for node in self.nodes.values():
            # Amplitude based on node health and performance
            amplitude = np.sqrt(node.quantum_score)
            if node.status != NodeStatus.HEALTHY:
                amplitude *= 0.5  # Reduce amplitude for unhealthy nodes
            
            node_amplitudes.append(amplitude)
        
        # Normalize
        node_amplitudes = np.array(node_amplitudes)
        norm = np.sqrt(np.sum(node_amplitudes ** 2))
        if norm > 0:
            self.quantum_state_vector = node_amplitudes / norm
        else:
            # Uniform superposition
            self.quantum_state_vector = np.ones(n_nodes) / np.sqrt(n_nodes)
        
        # Update entanglement matrix (correlations between nodes)
        self.entanglement_matrix = np.outer(self.quantum_state_vector, self.quantum_state_vector)
        
        # Update coherence factor
        self.coherence_factor = self._calculate_coherence()
    
    def _calculate_coherence(self) -> float:
        """Calculate quantum coherence of the system."""
        if len(self.quantum_state_vector) <= 1:
            return 1.0
        
        # Coherence as uniformity of probability distribution
        probabilities = self.quantum_state_vector ** 2
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-15))
        max_entropy = np.log(len(probabilities))
        
        coherence = entropy / max_entropy if max_entropy > 0 else 0.0
        return coherence
    
    def _update_node_quantum_score(self, node: LoadBalancerNode) -> None:
        """Update quantum performance score for node."""
        # Factors: success rate, response time, connection utilization
        success_factor = node.success_rate
        
        # Response time factor (lower is better)
        if node.avg_response_time > 0:
            response_factor = 1.0 / (1.0 + node.avg_response_time)
        else:
            response_factor = 1.0
        
        # Utilization factor (balanced utilization preferred)
        util = node.connection_utilization
        util_factor = 1.0 - abs(util - 0.5) * 2  # Peak at 50% utilization
        util_factor = max(0.1, util_factor)
        
        # Combine factors
        quantum_score = 0.4 * success_factor + 0.3 * response_factor + 0.3 * util_factor
        
        # Smooth update
        alpha = 0.1
        node.quantum_score = alpha * quantum_score + (1 - alpha) * node.quantum_score
        node.quantum_score = max(0.1, min(1.0, node.quantum_score))
    
    def _update_avg_response_time(self, node: LoadBalancerNode, response_time: float) -> None:
        """Update average response time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        if node.avg_response_time == 0.0:
            node.avg_response_time = response_time
        else:
            node.avg_response_time = alpha * response_time + (1 - alpha) * node.avg_response_time
    
    def _health_check_loop(self) -> None:
        """Background health checking loop."""
        while self.is_running:
            try:
                time.sleep(self.health_check_interval)
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all nodes."""
        with self.lock:
            current_time = time.time()
            
            for node in self.nodes.values():
                # Simple health check based on recent performance
                time_since_last_request = current_time - node.last_health_check
                
                # Update health status based on success rate and response time
                if node.success_rate < 0.5:
                    node.status = NodeStatus.FAILED
                elif node.success_rate < 0.8 or node.avg_response_time > 5.0:
                    node.status = NodeStatus.DEGRADED
                elif node.connection_utilization > 0.9:
                    node.status = NodeStatus.OVERLOADED
                else:
                    node.status = NodeStatus.HEALTHY
                
                # Check for failed nodes
                if node.status == NodeStatus.FAILED:
                    for callback in self.node_failure_callbacks:
                        try:
                            callback(node)
                        except Exception as e:
                            logger.error(f"Node failure callback failed: {e}")
                
                node.last_health_check = current_time
            
            # Update statistics
            self.stats.healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
            
            # Update quantum state based on health changes
            self._update_quantum_state()
    
    def _metrics_loop(self) -> None:
        """Background metrics processing loop."""
        while self.is_running:
            try:
                time.sleep(60.0)  # Update every minute
                self._update_metrics()
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
    
    def _update_metrics(self) -> None:
        """Update load balancer metrics."""
        with self.lock:
            # Calculate requests per second
            current_time = time.time()
            recent_requests = [
                req for req in self.request_history
                if current_time - req.start_time < 60.0  # Last minute
            ]
            
            self.stats.requests_per_second = len(recent_requests) / 60.0
            
            # Update prediction weights
            self._update_prediction_weights()
    
    def _update_prediction_weights(self) -> None:
        """Update prediction weights based on accuracy."""
        # Simplified prediction weight update
        for node_id, load_history in self.load_history.items():
            if len(load_history) > 10:
                # Calculate prediction accuracy for this node
                recent_loads = list(load_history)[-10:]
                
                # Simple accuracy based on variance
                variance = np.var(recent_loads)
                accuracy = 1.0 / (1.0 + variance)
                
                self.prediction_weights[node_id] = accuracy
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status."""
        with self.lock:
            # Node status breakdown
            node_status_counts = defaultdict(int)
            node_details = {}
            
            for node in self.nodes.values():
                node_status_counts[node.status.value] += 1
                node_details[node.id] = {
                    'endpoint': node.endpoint,
                    'status': node.status.value,
                    'weight': node.weight,
                    'current_connections': node.current_connections,
                    'max_connections': node.max_connections,
                    'utilization': node.connection_utilization,
                    'success_rate': node.success_rate,
                    'avg_response_time': node.avg_response_time,
                    'quantum_score': node.quantum_score,
                    'total_requests': node.total_requests
                }
            
            return {
                'strategy': self.strategy.value,
                'is_running': self.is_running,
                'statistics': {
                    'total_requests': self.stats.total_requests,
                    'success_rate': self.stats.success_rate,
                    'error_rate': self.stats.error_rate,
                    'avg_response_time': self.stats.avg_response_time,
                    'requests_per_second': self.stats.requests_per_second,
                    'active_connections': self.stats.active_connections,
                    'node_count': self.stats.node_count,
                    'healthy_nodes': self.stats.healthy_nodes
                },
                'node_status_counts': dict(node_status_counts),
                'node_details': node_details,
                'quantum_metrics': {
                    'coherence_factor': self.coherence_factor,
                    'quantum_state_entropy': -np.sum(
                        self.quantum_state_vector**2 * np.log(self.quantum_state_vector**2 + 1e-15)
                    ),
                    'entanglement_strength': np.trace(self.entanglement_matrix) / len(self.quantum_state_vector)
                },
                'configuration': {
                    'health_check_interval': self.health_check_interval,
                    'enable_circuit_breaker': self.enable_circuit_breaker
                }
            }
    
    def add_node_failure_callback(self, callback: Callable[[LoadBalancerNode], None]) -> None:
        """Add callback for node failure events."""
        self.node_failure_callbacks.append(callback)
    
    def add_request_callback(self, callback: Callable[[RequestMetrics], None]) -> None:
        """Add callback for request events."""
        self.request_callbacks.append(callback)