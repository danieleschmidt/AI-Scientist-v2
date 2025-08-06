"""
Resource Pool Management for Quantum Task Planner

Advanced resource pooling and management system for optimal
resource utilization and allocation in quantum task planning.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of pooled resources."""
    COMPUTE = "compute"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    QUANTUM_CIRCUIT = "quantum_circuit"
    OPTIMIZATION_ENGINE = "optimization_engine"
    CACHE = "cache"
    DATABASE_CONNECTION = "database_connection"


class ResourceStatus(Enum):
    """Resource allocation status."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class Resource:
    """Individual resource in the pool."""
    id: str
    resource_type: ResourceType
    capacity: float
    allocated: float = 0.0
    status: ResourceStatus = ResourceStatus.AVAILABLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> float:
        """Get available resource capacity."""
        return self.capacity - self.allocated
    
    @property
    def utilization_ratio(self) -> float:
        """Get resource utilization ratio."""
        return self.allocated / self.capacity if self.capacity > 0 else 0.0
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource capacity."""
        if self.available_capacity >= amount and self.status == ResourceStatus.AVAILABLE:
            self.allocated += amount
            self.last_used = time.time()
            self.use_count += 1
            if self.allocated >= self.capacity:
                self.status = ResourceStatus.ALLOCATED
            return True
        return False
    
    def deallocate(self, amount: float) -> None:
        """Deallocate resource capacity."""
        self.allocated = max(0, self.allocated - amount)
        if self.allocated == 0 and self.status == ResourceStatus.ALLOCATED:
            self.status = ResourceStatus.AVAILABLE


@dataclass
class AllocationRequest:
    """Resource allocation request."""
    id: str
    resource_type: ResourceType
    amount: float
    priority: float = 1.0
    timeout: Optional[float] = None
    requester: str = "unknown"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """Result of resource allocation."""
    request_id: str
    success: bool
    allocated_resources: List[str] = field(default_factory=list)
    allocated_amount: float = 0.0
    wait_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PoolStatistics:
    """Resource pool statistics."""
    total_resources: int = 0
    available_resources: int = 0
    allocated_resources: int = 0
    failed_resources: int = 0
    total_capacity: float = 0.0
    allocated_capacity: float = 0.0
    avg_utilization: float = 0.0
    allocation_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    avg_allocation_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate allocation success rate."""
        return self.successful_allocations / max(self.allocation_requests, 1)
    
    @property
    def utilization_ratio(self) -> float:
        """Calculate overall utilization ratio."""
        return self.allocated_capacity / max(self.total_capacity, 1)


class ResourcePool:
    """
    Advanced resource pool with quantum-inspired allocation algorithms.
    
    Manages pools of various resource types with intelligent allocation,
    load balancing, and predictive scaling.
    """
    
    def __init__(self,
                 pool_name: str = "default",
                 enable_prediction: bool = True,
                 enable_auto_scaling: bool = True):
        """
        Initialize resource pool.
        
        Args:
            pool_name: Name of the resource pool
            enable_prediction: Enable predictive resource allocation
            enable_auto_scaling: Enable automatic resource scaling
        """
        self.pool_name = pool_name
        self.enable_prediction = enable_prediction
        self.enable_auto_scaling = enable_auto_scaling
        
        # Resource storage
        self.resources: Dict[str, Resource] = {}
        self.resource_types: Dict[ResourceType, List[str]] = defaultdict(list)
        
        # Allocation management
        self.allocation_queue: deque = deque()
        self.active_allocations: Dict[str, List[str]] = {}  # request_id -> resource_ids
        
        # Statistics and monitoring
        self.statistics = PoolStatistics()
        self.allocation_history: deque = deque(maxlen=1000)
        self.usage_patterns: Dict[ResourceType, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background processing
        self.is_running = False
        self.management_thread: Optional[threading.Thread] = None
        
        # Quantum-inspired optimization
        self.quantum_weights: Dict[str, float] = {}
        self.allocation_matrix: np.ndarray = np.eye(len(ResourceType))
        
        # Callbacks for events
        self.allocation_callbacks: List[Callable[[AllocationResult], None]] = []
        self.resource_failure_callbacks: List[Callable[[Resource], None]] = []
        
        logger.info(f"Initialized ResourcePool '{pool_name}' with quantum-inspired allocation")
    
    def start(self) -> None:
        """Start resource pool management."""
        if self.is_running:
            logger.warning("Resource pool already running")
            return
        
        self.is_running = True
        self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.management_thread.start()
        
        logger.info(f"Started resource pool '{self.pool_name}'")
    
    def stop(self) -> None:
        """Stop resource pool management."""
        self.is_running = False
        if self.management_thread and self.management_thread.is_alive():
            self.management_thread.join(timeout=5.0)
        
        logger.info(f"Stopped resource pool '{self.pool_name}'")
    
    def add_resource(self,
                    resource_id: str,
                    resource_type: ResourceType,
                    capacity: float,
                    metadata: Dict[str, Any] = None) -> bool:
        """
        Add resource to the pool.
        
        Args:
            resource_id: Unique resource identifier
            resource_type: Type of resource
            capacity: Resource capacity
            metadata: Additional resource metadata
            
        Returns:
            True if resource was added successfully
        """
        with self.lock:
            if resource_id in self.resources:
                logger.warning(f"Resource {resource_id} already exists in pool")
                return False
            
            resource = Resource(
                id=resource_id,
                resource_type=resource_type,
                capacity=capacity,
                metadata=metadata or {}
            )
            
            self.resources[resource_id] = resource
            self.resource_types[resource_type].append(resource_id)
            
            # Update statistics
            self.statistics.total_resources += 1
            self.statistics.available_resources += 1
            self.statistics.total_capacity += capacity
            
            # Initialize quantum weight
            self.quantum_weights[resource_id] = self._calculate_quantum_weight(resource)
            
            logger.debug(f"Added resource {resource_id}: {resource_type.value}, capacity={capacity}")
            return True
    
    def remove_resource(self, resource_id: str) -> bool:
        """
        Remove resource from the pool.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if resource was removed successfully
        """
        with self.lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources.pop(resource_id)
            
            # Update resource type index
            if resource_id in self.resource_types[resource.resource_type]:
                self.resource_types[resource.resource_type].remove(resource_id)
            
            # Update statistics
            self.statistics.total_resources -= 1
            if resource.status == ResourceStatus.AVAILABLE:
                self.statistics.available_resources -= 1
            elif resource.status == ResourceStatus.ALLOCATED:
                self.statistics.allocated_resources -= 1
            elif resource.status == ResourceStatus.FAILED:
                self.statistics.failed_resources -= 1
            
            self.statistics.total_capacity -= resource.capacity
            self.statistics.allocated_capacity -= resource.allocated
            
            # Clean up quantum weight
            self.quantum_weights.pop(resource_id, None)
            
            logger.info(f"Removed resource {resource_id} from pool")
            return True
    
    def allocate_resources(self,
                          request: AllocationRequest) -> AllocationResult:
        """
        Allocate resources based on request.
        
        Args:
            request: Resource allocation request
            
        Returns:
            Allocation result
        """
        start_time = time.time()
        
        with self.lock:
            self.statistics.allocation_requests += 1
            
            # Find suitable resources
            suitable_resources = self._find_suitable_resources(request)
            
            if not suitable_resources:
                self.statistics.failed_allocations += 1
                return AllocationResult(
                    request_id=request.id,
                    success=False,
                    wait_time=time.time() - start_time,
                    error_message="No suitable resources available"
                )
            
            # Perform allocation using quantum-inspired algorithm
            allocated_resources = self._quantum_allocate(request, suitable_resources)
            
            if not allocated_resources:
                self.statistics.failed_allocations += 1
                return AllocationResult(
                    request_id=request.id,
                    success=False,
                    wait_time=time.time() - start_time,
                    error_message="Allocation failed"
                )
            
            # Track allocation
            self.active_allocations[request.id] = allocated_resources
            allocated_amount = sum(
                self.resources[rid].allocated for rid in allocated_resources
            )
            
            # Update statistics
            self.statistics.successful_allocations += 1
            self.statistics.allocated_capacity += allocated_amount
            
            wait_time = time.time() - start_time
            self._update_avg_allocation_time(wait_time)
            
            # Record allocation history
            result = AllocationResult(
                request_id=request.id,
                success=True,
                allocated_resources=allocated_resources,
                allocated_amount=allocated_amount,
                wait_time=wait_time
            )
            
            self.allocation_history.append(result)
            
            # Update usage patterns
            self.usage_patterns[request.resource_type].append(allocated_amount)
            if len(self.usage_patterns[request.resource_type]) > 100:
                self.usage_patterns[request.resource_type].pop(0)
            
            # Trigger callbacks
            for callback in self.allocation_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Allocation callback failed: {e}")
            
            logger.debug(f"Allocated {allocated_amount} {request.resource_type.value} to request {request.id}")
            return result
    
    def deallocate_resources(self, request_id: str) -> bool:
        """
        Deallocate resources for a request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            True if deallocation was successful
        """
        with self.lock:
            if request_id not in self.active_allocations:
                logger.warning(f"No active allocation found for request {request_id}")
                return False
            
            resource_ids = self.active_allocations.pop(request_id)
            total_deallocated = 0.0
            
            for resource_id in resource_ids:
                if resource_id in self.resources:
                    resource = self.resources[resource_id]
                    deallocated_amount = resource.allocated
                    resource.deallocate(resource.allocated)
                    total_deallocated += deallocated_amount
                    
                    # Update resource status counts
                    if resource.status == ResourceStatus.AVAILABLE:
                        self.statistics.available_resources += 1
                        self.statistics.allocated_resources -= 1
            
            # Update statistics
            self.statistics.allocated_capacity -= total_deallocated
            
            logger.debug(f"Deallocated {total_deallocated} resources for request {request_id}")
            return True
    
    def _find_suitable_resources(self, request: AllocationRequest) -> List[str]:
        """Find resources suitable for the allocation request."""
        resource_ids = self.resource_types[request.resource_type]
        suitable_resources = []
        
        for resource_id in resource_ids:
            resource = self.resources[resource_id]
            
            if (resource.status == ResourceStatus.AVAILABLE and 
                resource.available_capacity >= request.amount):
                suitable_resources.append(resource_id)
        
        # Sort by quantum weight (higher weight first)
        suitable_resources.sort(
            key=lambda rid: self.quantum_weights.get(rid, 0.5),
            reverse=True
        )
        
        return suitable_resources
    
    def _quantum_allocate(self,
                         request: AllocationRequest,
                         suitable_resources: List[str]) -> List[str]:
        """Allocate resources using quantum-inspired algorithm."""
        allocated_resources = []
        remaining_amount = request.amount
        
        # Use quantum superposition principle for resource selection
        for resource_id in suitable_resources:
            if remaining_amount <= 0:
                break
            
            resource = self.resources[resource_id]
            quantum_weight = self.quantum_weights[resource_id]
            
            # Quantum probability of allocation based on weight and availability
            allocation_probability = quantum_weight * min(
                resource.available_capacity / request.amount, 1.0
            )
            
            # Stochastic allocation decision
            if np.random.random() < allocation_probability:
                allocate_amount = min(remaining_amount, resource.available_capacity)
                
                if resource.allocate(allocate_amount):
                    allocated_resources.append(resource_id)
                    remaining_amount -= allocate_amount
                    
                    # Update resource status counts
                    if resource.status == ResourceStatus.ALLOCATED:
                        self.statistics.available_resources -= 1
                        self.statistics.allocated_resources += 1
        
        # If we couldn't allocate enough, try deterministic allocation
        if remaining_amount > 0:
            for resource_id in suitable_resources:
                if resource_id in allocated_resources:
                    continue
                
                resource = self.resources[resource_id]
                allocate_amount = min(remaining_amount, resource.available_capacity)
                
                if allocate_amount > 0 and resource.allocate(allocate_amount):
                    allocated_resources.append(resource_id)
                    remaining_amount -= allocate_amount
                    
                    # Update resource status counts
                    if resource.status == ResourceStatus.ALLOCATED:
                        self.statistics.available_resources -= 1
                        self.statistics.allocated_resources += 1
                    
                    if remaining_amount <= 0:
                        break
        
        return allocated_resources if remaining_amount <= 0 else []
    
    def _calculate_quantum_weight(self, resource: Resource) -> float:
        """Calculate quantum-inspired weight for resource."""
        base_weight = 0.5
        
        # Capacity factor (higher capacity gets higher weight)
        capacity_factor = min(resource.capacity / 100.0, 1.0)
        
        # Usage factor (less used resources get higher weight for load balancing)
        usage_factor = 1.0 - min(resource.use_count / 1000.0, 0.8)
        
        # Reliability factor (fewer failures get higher weight)
        reliability_factor = 1.0 - min(resource.failure_count / 10.0, 0.5)
        
        # Combine factors
        weight = base_weight * (0.3 * capacity_factor + 0.4 * usage_factor + 0.3 * reliability_factor)
        
        return max(0.1, min(1.0, weight))
    
    def _update_avg_allocation_time(self, allocation_time: float) -> None:
        """Update running average allocation time."""
        alpha = 0.1  # Smoothing factor
        if self.statistics.avg_allocation_time == 0.0:
            self.statistics.avg_allocation_time = allocation_time
        else:
            self.statistics.avg_allocation_time = (
                alpha * allocation_time + 
                (1 - alpha) * self.statistics.avg_allocation_time
            )
    
    def _management_loop(self) -> None:
        """Background management loop."""
        while self.is_running:
            try:
                time.sleep(30.0)  # Run every 30 seconds
                
                # Update statistics
                self._update_statistics()
                
                # Check for failed resources
                self._check_resource_health()
                
                # Update quantum weights
                self._update_quantum_weights()
                
                # Predictive scaling if enabled
                if self.enable_auto_scaling:
                    self._auto_scale_resources()
                
            except Exception as e:
                logger.error(f"Resource pool management error: {e}")
    
    def _update_statistics(self) -> None:
        """Update pool statistics."""
        with self.lock:
            # Count resources by status
            available_count = 0
            allocated_count = 0
            failed_count = 0
            total_capacity = 0.0
            allocated_capacity = 0.0
            utilizations = []
            
            for resource in self.resources.values():
                total_capacity += resource.capacity
                allocated_capacity += resource.allocated
                
                if resource.status == ResourceStatus.AVAILABLE:
                    available_count += 1
                elif resource.status == ResourceStatus.ALLOCATED:
                    allocated_count += 1
                elif resource.status == ResourceStatus.FAILED:
                    failed_count += 1
                
                utilizations.append(resource.utilization_ratio)
            
            # Update statistics
            self.statistics.available_resources = available_count
            self.statistics.allocated_resources = allocated_count
            self.statistics.failed_resources = failed_count
            self.statistics.total_capacity = total_capacity
            self.statistics.allocated_capacity = allocated_capacity
            self.statistics.avg_utilization = np.mean(utilizations) if utilizations else 0.0
    
    def _check_resource_health(self) -> None:
        """Check health of resources and mark failed ones."""
        with self.lock:
            for resource in self.resources.values():
                # Simple health check based on usage patterns
                current_time = time.time()
                
                # If resource hasn't been used in a long time and is allocated, it might be stuck
                if (resource.status == ResourceStatus.ALLOCATED and 
                    current_time - resource.last_used > 3600):  # 1 hour
                    
                    logger.warning(f"Resource {resource.id} appears to be stuck, marking as failed")
                    resource.status = ResourceStatus.FAILED
                    resource.failure_count += 1
                    
                    # Trigger failure callbacks
                    for callback in self.resource_failure_callbacks:
                        try:
                            callback(resource)
                        except Exception as e:
                            logger.error(f"Resource failure callback failed: {e}")
                
                # Auto-recovery for failed resources (simplified)
                if (resource.status == ResourceStatus.FAILED and
                    current_time - resource.last_used > 7200):  # 2 hours
                    
                    if resource.failure_count < 5:  # Don't recover resources that fail too often
                        logger.info(f"Attempting to recover resource {resource.id}")
                        resource.status = ResourceStatus.AVAILABLE
                        resource.allocated = 0.0
    
    def _update_quantum_weights(self) -> None:
        """Update quantum weights based on current resource performance."""
        with self.lock:
            for resource_id, resource in self.resources.items():
                new_weight = self._calculate_quantum_weight(resource)
                self.quantum_weights[resource_id] = new_weight
    
    def _auto_scale_resources(self) -> None:
        """Auto-scale resources based on usage patterns."""
        # Simplified auto-scaling - in practice would be more sophisticated
        with self.lock:
            for resource_type, usage_history in self.usage_patterns.items():
                if len(usage_history) >= 10:  # Need enough data
                    avg_usage = np.mean(usage_history[-10:])  # Recent average
                    
                    # Get current capacity for this resource type
                    type_resources = [
                        self.resources[rid] for rid in self.resource_types[resource_type]
                        if rid in self.resources
                    ]
                    
                    if not type_resources:
                        continue
                    
                    total_capacity = sum(r.capacity for r in type_resources)
                    utilization = avg_usage / total_capacity if total_capacity > 0 else 0
                    
                    # Scale up if high utilization
                    if utilization > 0.8 and len(type_resources) < 10:  # Max 10 resources per type
                        logger.info(f"High utilization detected for {resource_type.value}, considering scale up")
                        # In practice, would trigger resource creation
                    
                    # Scale down if low utilization
                    elif utilization < 0.2 and len(type_resources) > 1:  # Keep at least 1 resource
                        logger.info(f"Low utilization detected for {resource_type.value}, considering scale down")
                        # In practice, would trigger resource removal
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status."""
        with self.lock:
            self._update_statistics()
            
            # Resource breakdown by type
            type_breakdown = {}
            for resource_type in ResourceType:
                type_resources = [
                    self.resources[rid] for rid in self.resource_types[resource_type]
                    if rid in self.resources
                ]
                
                if type_resources:
                    type_breakdown[resource_type.value] = {
                        'count': len(type_resources),
                        'total_capacity': sum(r.capacity for r in type_resources),
                        'allocated_capacity': sum(r.allocated for r in type_resources),
                        'avg_utilization': np.mean([r.utilization_ratio for r in type_resources]),
                        'available_count': sum(1 for r in type_resources if r.status == ResourceStatus.AVAILABLE),
                        'allocated_count': sum(1 for r in type_resources if r.status == ResourceStatus.ALLOCATED),
                        'failed_count': sum(1 for r in type_resources if r.status == ResourceStatus.FAILED)
                    }
            
            # Recent allocation performance
            recent_allocations = list(self.allocation_history)[-50:]  # Last 50 allocations
            recent_success_rate = (
                sum(1 for a in recent_allocations if a.success) / len(recent_allocations)
                if recent_allocations else 0.0
            )
            
            return {
                'pool_name': self.pool_name,
                'is_running': self.is_running,
                'statistics': {
                    'total_resources': self.statistics.total_resources,
                    'available_resources': self.statistics.available_resources,
                    'allocated_resources': self.statistics.allocated_resources,
                    'failed_resources': self.statistics.failed_resources,
                    'utilization_ratio': self.statistics.utilization_ratio,
                    'avg_utilization': self.statistics.avg_utilization,
                    'success_rate': self.statistics.success_rate,
                    'avg_allocation_time': self.statistics.avg_allocation_time
                },
                'type_breakdown': type_breakdown,
                'recent_performance': {
                    'allocations_last_50': len(recent_allocations),
                    'recent_success_rate': recent_success_rate,
                    'active_allocations': len(self.active_allocations)
                },
                'configuration': {
                    'enable_prediction': self.enable_prediction,
                    'enable_auto_scaling': self.enable_auto_scaling
                }
            }
    
    def add_allocation_callback(self, callback: Callable[[AllocationResult], None]) -> None:
        """Add callback for allocation events."""
        self.allocation_callbacks.append(callback)
    
    def add_resource_failure_callback(self, callback: Callable[[Resource], None]) -> None:
        """Add callback for resource failure events."""
        self.resource_failure_callbacks.append(callback)