"""
Distributed Computing and Auto-Scaling Framework

High-performance distributed computing system with intelligent auto-scaling,
load balancing, and fault-tolerant distributed execution for autonomous scientific research.
"""

import time
import asyncio
import threading
import multiprocessing
import logging
import json
import uuid
import pickle
import hashlib
import socket
import struct
from typing import (
    Dict, List, Any, Optional, Union, Callable, Type, Tuple, 
    Generic, TypeVar, Protocol, Set, Coroutine
)
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import functools
import weakref
from collections import defaultdict, deque
import statistics
import psutil
import numpy as np
from datetime import datetime, timedelta
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class NodeType(Enum):
    """Types of distributed computing nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    MONITOR = "monitor"
    STORAGE = "storage"
    GATEWAY = "gateway"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class ScalingPolicy(Enum):
    """Auto-scaling policy types."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    LATENCY_BASED = "latency_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class NodeInfo:
    """Information about a distributed computing node."""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    capabilities: List[str] = field(default_factory=list)
    load_factor: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'host': self.host,
            'port': self.port,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'capabilities': self.capabilities,
            'load_factor': self.load_factor,
            'last_heartbeat': self.last_heartbeat,
            'is_active': self.is_active,
            'metadata': self.metadata
        }


@dataclass
class DistributedTask:
    """Distributed computing task definition."""
    task_id: str
    function_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    requirements: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'function_name': self.function_name,
            'priority': self.priority.value,
            'requirements': self.requirements,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'scheduled_at': self.scheduled_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'assigned_node': self.assigned_node,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    queue_size: int
    active_workers: int
    average_task_duration: float
    task_throughput: float
    error_rate: float
    latency_p95: float
    predicted_load: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LoadBalancer:
    """Intelligent load balancer for distributed tasks."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.node_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.task_history: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def select_node(self, task: DistributedTask, available_nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """Select optimal node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes based on task requirements
        compatible_nodes = self._filter_compatible_nodes(task, available_nodes)
        
        if not compatible_nodes:
            return None
        
        # Apply load balancing strategy
        if self.strategy == "round_robin":
            return self._round_robin_selection(compatible_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(compatible_nodes)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(compatible_nodes)
        elif self.strategy == "performance_based":
            return self._performance_based_selection(task, compatible_nodes)
        else:
            # Default to least loaded
            return self._least_loaded_selection(compatible_nodes)
    
    def _filter_compatible_nodes(self, task: DistributedTask, 
                                nodes: List[NodeInfo]) -> List[NodeInfo]:
        """Filter nodes that can handle the task requirements."""
        compatible = []
        
        for node in nodes:
            if not node.is_active:
                continue
            
            # Check CPU requirements
            required_cpus = task.requirements.get('cpu_cores', 1)
            if node.cpu_cores < required_cpus:
                continue
            
            # Check memory requirements
            required_memory = task.requirements.get('memory_gb', 0)
            if node.memory_gb < required_memory:
                continue
            
            # Check GPU requirements
            required_gpus = task.requirements.get('gpu_count', 0)
            if node.gpu_count < required_gpus:
                continue
            
            # Check capabilities
            required_capabilities = task.requirements.get('capabilities', [])
            if not all(cap in node.capabilities for cap in required_capabilities):
                continue
            
            # Check load factor
            if node.load_factor > 0.9:  # Node is too loaded
                continue
            
            compatible.append(node)
        
        return compatible
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node using round-robin strategy."""
        # Simple round-robin based on node ID hash
        current_time = int(time.time())
        index = current_time % len(nodes)
        return nodes[index]
    
    def _least_loaded_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with lowest load factor."""
        return min(nodes, key=lambda n: n.load_factor)
    
    def _weighted_round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node using weighted round-robin based on capacity."""
        # Calculate weights based on CPU cores and inverse load factor
        weights = []
        for node in nodes:
            weight = node.cpu_cores * (1.0 - node.load_factor)
            weights.append(max(0.1, weight))  # Minimum weight
        
        # Weighted random selection
        total_weight = sum(weights)
        random_value = np.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _performance_based_selection(self, task: DistributedTask, 
                                   nodes: List[NodeInfo]) -> NodeInfo:
        """Select node based on historical performance for similar tasks."""
        scores = []
        
        for node in nodes:
            # Base score from current load
            base_score = 1.0 - node.load_factor
            
            # Historical performance bonus
            task_type = task.function_name
            if node.node_id in self.task_history:
                history = self.task_history[node.node_id]
                if history:
                    # Faster execution = higher score
                    avg_duration = statistics.mean(history[-10:])  # Last 10 tasks
                    performance_bonus = max(0, 1.0 - (avg_duration / 60.0))  # Normalize
                    base_score += performance_bonus * 0.3
            
            # Resource capacity bonus
            capacity_bonus = (node.cpu_cores / 16.0) * 0.2  # Normalize to 16 cores
            base_score += capacity_bonus
            
            scores.append(base_score)
        
        # Select node with highest score
        best_index = scores.index(max(scores))
        return nodes[best_index]
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update metrics for a node."""
        with self.lock:
            self.node_metrics[node_id].update(metrics)
    
    def record_task_completion(self, node_id: str, duration: float):
        """Record task completion for performance tracking."""
        with self.lock:
            self.task_history[node_id].append(duration)
            # Keep only recent history
            if len(self.task_history[node_id]) > 100:
                self.task_history[node_id] = self.task_history[node_id][-100:]


class TaskScheduler:
    """Intelligent task scheduler with priority queues and resource allocation."""
    
    def __init__(self, max_concurrent_tasks: int = 100):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        self.lock = threading.RLock()
        
        # Scheduling metrics
        self.scheduling_metrics = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for scheduling."""
        with self.lock:
            self.task_queues[task.priority].append(task)
            self.scheduling_metrics['total_scheduled'] += 1
            
            logger.info(f"Task {task.task_id} submitted with priority {task.priority.name}")
            return task.task_id
    
    def get_next_task(self) -> Optional[DistributedTask]:
        """Get the next task to execute based on priority."""
        with self.lock:
            # Check if we've reached max concurrent tasks
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                return None
            
            # Get highest priority task
            for priority in reversed(list(TaskPriority)):  # Higher priority first
                if self.task_queues[priority]:
                    task = self.task_queues[priority].popleft()
                    task.status = TaskStatus.SCHEDULED
                    task.scheduled_at = time.time()
                    self.running_tasks[task.task_id] = task
                    return task
            
            return None
    
    def start_task(self, task_id: str, node_id: str):
        """Mark task as started on a specific node."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                task.assigned_node = node_id
                logger.info(f"Task {task_id} started on node {node_id}")
    
    def complete_task(self, task_id: str, result: Any):
        """Mark task as completed with result."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                self.completed_tasks[task_id] = task
                
                # Update metrics
                self.scheduling_metrics['total_completed'] += 1
                if task.scheduled_at and task.completed_at:
                    execution_time = task.completed_at - task.scheduled_at
                    self._update_average_metric('average_execution_time', execution_time)
                
                logger.info(f"Task {task_id} completed successfully")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed with error."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.error = error
                
                # Check if we should retry
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRY
                    task.assigned_node = None
                    task.started_at = None
                    
                    # Re-queue for retry
                    self.task_queues[task.priority].appendleft(task)
                    logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                else:
                    task.status = TaskStatus.FAILED
                    task.completed_at = time.time()
                    self.failed_tasks[task_id] = task
                    self.scheduling_metrics['total_failed'] += 1
                    logger.error(f"Task {task_id} failed permanently: {error}")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.lock:
            # Check if task is in queue
            for priority_queue in self.task_queues.values():
                for i, task in enumerate(priority_queue):
                    if task.task_id == task_id:
                        task.status = TaskStatus.CANCELLED
                        del priority_queue[i]
                        logger.info(f"Task {task_id} cancelled from queue")
                        return True
            
            # Check if task is running
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                self.failed_tasks[task_id] = task
                logger.info(f"Running task {task_id} cancelled")
                return True
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get current status of a task."""
        with self.lock:
            # Check running tasks
            if task_id in self.running_tasks:
                return self.running_tasks[task_id]
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            # Check queued tasks
            for priority_queue in self.task_queues.values():
                for task in priority_queue:
                    if task.task_id == task_id:
                        return task
            
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        with self.lock:
            queue_sizes = {
                priority.name: len(queue) 
                for priority, queue in self.task_queues.items()
            }
            
            return {
                'queue_sizes': queue_sizes,
                'total_queued': sum(queue_sizes.values()),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'metrics': self.scheduling_metrics.copy()
            }
    
    def _update_average_metric(self, metric_name: str, new_value: float):
        """Update average metric with exponential moving average."""
        alpha = 0.1  # Learning rate
        current_avg = self.scheduling_metrics[metric_name]
        self.scheduling_metrics[metric_name] = alpha * new_value + (1 - alpha) * current_avg


class AutoScaler:
    """Intelligent auto-scaling system for dynamic resource management."""
    
    def __init__(self, scaling_policies: List[ScalingPolicy], 
                 min_workers: int = 1, max_workers: int = 50):
        self.scaling_policies = scaling_policies
        self.min_workers = min_workers
        self.max_workers = max_workers
        
        # Scaling state
        self.current_workers = min_workers
        self.target_workers = min_workers
        self.last_scaling_decision = time.time()
        self.scaling_cooldown = 300.0  # 5 minutes
        
        # Metrics history for prediction
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_decisions: List[Dict[str, Any]] = []
        
        # Thresholds
        self.cpu_scale_up_threshold = 70.0
        self.cpu_scale_down_threshold = 30.0
        self.memory_scale_up_threshold = 80.0
        self.memory_scale_down_threshold = 40.0
        self.queue_scale_up_threshold = 10
        self.queue_scale_down_threshold = 2
        
        self.lock = threading.RLock()
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics for scaling decisions."""
        with self.lock:
            self.metrics_history.append(metrics)
    
    def make_scaling_decision(self) -> Optional[int]:
        """Make scaling decision based on current metrics and policies."""
        with self.lock:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_scaling_decision < self.scaling_cooldown:
                return None
            
            if not self.metrics_history:
                return None
            
            latest_metrics = self.metrics_history[-1]
            scaling_votes = []
            
            # Evaluate each scaling policy
            for policy in self.scaling_policies:
                vote = self._evaluate_policy(policy, latest_metrics)
                if vote is not None:
                    scaling_votes.append(vote)
            
            if not scaling_votes:
                return None
            
            # Aggregate scaling votes
            average_vote = statistics.mean(scaling_votes)
            
            # Determine new target worker count
            if average_vote > 0.5:  # Scale up
                new_target = min(self.max_workers, self.current_workers + 1)
            elif average_vote < -0.5:  # Scale down
                new_target = max(self.min_workers, self.current_workers - 1)
            else:
                new_target = self.current_workers
            
            # Record scaling decision
            if new_target != self.current_workers:
                decision = {
                    'timestamp': current_time,
                    'old_workers': self.current_workers,
                    'new_workers': new_target,
                    'metrics': latest_metrics.to_dict(),
                    'votes': scaling_votes,
                    'average_vote': average_vote
                }
                
                self.scaling_decisions.append(decision)
                self.last_scaling_decision = current_time
                self.target_workers = new_target
                
                logger.info(f"Scaling decision: {self.current_workers} -> {new_target} workers")
                return new_target
            
            return None
    
    def _evaluate_policy(self, policy: ScalingPolicy, metrics: ScalingMetrics) -> Optional[float]:
        """Evaluate a specific scaling policy and return vote (-1 to 1)."""
        if policy == ScalingPolicy.CPU_BASED:
            return self._evaluate_cpu_policy(metrics)
        elif policy == ScalingPolicy.MEMORY_BASED:
            return self._evaluate_memory_policy(metrics)
        elif policy == ScalingPolicy.QUEUE_BASED:
            return self._evaluate_queue_policy(metrics)
        elif policy == ScalingPolicy.LATENCY_BASED:
            return self._evaluate_latency_policy(metrics)
        elif policy == ScalingPolicy.PREDICTIVE:
            return self._evaluate_predictive_policy(metrics)
        elif policy == ScalingPolicy.HYBRID:
            return self._evaluate_hybrid_policy(metrics)
        
        return None
    
    def _evaluate_cpu_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate CPU-based scaling policy."""
        if metrics.cpu_usage > self.cpu_scale_up_threshold:
            return (metrics.cpu_usage - self.cpu_scale_up_threshold) / 30.0  # Normalize
        elif metrics.cpu_usage < self.cpu_scale_down_threshold:
            return -(self.cpu_scale_down_threshold - metrics.cpu_usage) / 30.0
        return 0.0
    
    def _evaluate_memory_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate memory-based scaling policy."""
        if metrics.memory_usage > self.memory_scale_up_threshold:
            return (metrics.memory_usage - self.memory_scale_up_threshold) / 20.0
        elif metrics.memory_usage < self.memory_scale_down_threshold:
            return -(self.memory_scale_down_threshold - metrics.memory_usage) / 40.0
        return 0.0
    
    def _evaluate_queue_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate queue-based scaling policy."""
        if metrics.queue_size > self.queue_scale_up_threshold:
            return min(1.0, (metrics.queue_size - self.queue_scale_up_threshold) / 20.0)
        elif metrics.queue_size < self.queue_scale_down_threshold:
            return max(-1.0, -(self.queue_scale_down_threshold - metrics.queue_size) / 5.0)
        return 0.0
    
    def _evaluate_latency_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate latency-based scaling policy."""
        target_latency = 5.0  # 5 seconds target P95 latency
        
        if metrics.latency_p95 > target_latency * 2:
            return 0.8
        elif metrics.latency_p95 > target_latency:
            return 0.4
        elif metrics.latency_p95 < target_latency * 0.5:
            return -0.4
        
        return 0.0
    
    def _evaluate_predictive_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate predictive scaling policy based on trends."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        # Analyze recent trend
        recent_metrics = list(self.metrics_history)[-10:]
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = self._calculate_trend(cpu_values)
        
        # Queue trend
        queue_values = [m.queue_size for m in recent_metrics]
        queue_trend = self._calculate_trend(queue_values)
        
        # Combine trends
        combined_trend = (cpu_trend + queue_trend) / 2.0
        
        # Scale proactively based on trend
        if combined_trend > 5.0:  # Strong upward trend
            return 0.6
        elif combined_trend > 2.0:  # Moderate upward trend
            return 0.3
        elif combined_trend < -5.0:  # Strong downward trend
            return -0.6
        elif combined_trend < -2.0:  # Moderate downward trend
            return -0.3
        
        return 0.0
    
    def _evaluate_hybrid_policy(self, metrics: ScalingMetrics) -> float:
        """Evaluate hybrid policy combining multiple factors."""
        cpu_vote = self._evaluate_cpu_policy(metrics)
        memory_vote = self._evaluate_memory_policy(metrics)
        queue_vote = self._evaluate_queue_policy(metrics)
        latency_vote = self._evaluate_latency_policy(metrics)
        predictive_vote = self._evaluate_predictive_policy(metrics)
        
        # Weighted combination
        weights = [0.3, 0.2, 0.3, 0.1, 0.1]  # CPU, Memory, Queue, Latency, Predictive
        votes = [cpu_vote, memory_vote, queue_vote, latency_vote, predictive_vote]
        
        return sum(w * v for w, v in zip(weights, votes))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope * n  # Scale by number of points
    
    def update_worker_count(self, new_count: int):
        """Update current worker count after scaling."""
        with self.lock:
            self.current_workers = new_count
            logger.info(f"Worker count updated to {new_count}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.lock:
            return {
                'current_workers': self.current_workers,
                'target_workers': self.target_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'last_scaling_decision': self.last_scaling_decision,
                'scaling_cooldown': self.scaling_cooldown,
                'recent_decisions': self.scaling_decisions[-10:],
                'policies': [policy.value for policy in self.scaling_policies]
            }


class DistributedWorkerNode:
    """Distributed worker node for executing tasks."""
    
    def __init__(self, node_info: NodeInfo, coordinator_host: str, coordinator_port: int):
        self.node_info = node_info
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        
        # ZeroMQ communication
        self.context = zmq.Context()
        self.coordinator_socket = self.context.socket(zmq.REQ)
        
        # Task execution
        self.executor = ThreadPoolExecutor(max_workers=node_info.cpu_cores)
        self.current_tasks: Dict[str, Future] = {}
        
        # State management
        self.running = False
        self.heartbeat_interval = 30.0
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.work_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.execution_metrics = {
            'tasks_executed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Function registry
        self.function_registry: Dict[str, Callable] = {}
    
    def register_function(self, name: str, func: Callable):
        """Register a function for distributed execution."""
        self.function_registry[name] = func
        logger.info(f"Registered function: {name}")
    
    def start(self):
        """Start the worker node."""
        if self.running:
            return
        
        self.running = True
        
        # Connect to coordinator
        coordinator_address = f"tcp://{self.coordinator_host}:{self.coordinator_port}"
        self.coordinator_socket.connect(coordinator_address)
        
        # Start threads
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.work_thread = threading.Thread(target=self._work_loop, daemon=True)
        
        self.heartbeat_thread.start()
        self.work_thread.start()
        
        logger.info(f"Worker node {self.node_info.node_id} started")
    
    def stop(self):
        """Stop the worker node."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel running tasks
        for task_id, future in self.current_tasks.items():
            future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close socket
        self.coordinator_socket.close()
        self.context.term()
        
        logger.info(f"Worker node {self.node_info.node_id} stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _send_heartbeat(self):
        """Send heartbeat to coordinator."""
        # Update load factor
        self.node_info.load_factor = len(self.current_tasks) / self.node_info.cpu_cores
        self.node_info.last_heartbeat = time.time()
        
        message = {
            'type': 'heartbeat',
            'node_info': self.node_info.to_dict(),
            'metrics': self.execution_metrics.copy()
        }
        
        self.coordinator_socket.send_json(message)
        response = self.coordinator_socket.recv_json()
        
        # Handle coordinator response
        if response.get('type') == 'shutdown':
            self.running = False
    
    def _work_loop(self):
        """Main work loop for task execution."""
        while self.running:
            try:
                # Request work from coordinator
                work_request = {
                    'type': 'work_request',
                    'node_id': self.node_info.node_id
                }
                
                self.coordinator_socket.send_json(work_request)
                response = self.coordinator_socket.recv_json()
                
                if response.get('type') == 'task':
                    task_data = response['task']
                    self._execute_task(task_data)
                elif response.get('type') == 'no_work':
                    time.sleep(1.0)  # Wait before requesting again
                
            except Exception as e:
                logger.error(f"Error in work loop: {e}")
                time.sleep(5.0)
    
    def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a distributed task."""
        task_id = task_data['task_id']
        function_name = task_data['function_name']
        
        if function_name not in self.function_registry:
            self._report_task_failure(task_id, f"Function {function_name} not registered")
            return
        
        func = self.function_registry[function_name]
        
        # Submit task to executor
        future = self.executor.submit(self._run_task, task_id, func, task_data)
        self.current_tasks[task_id] = future
        
        # Monitor completion
        def on_completion(future_result):
            try:
                result = future_result.result()
                self._report_task_completion(task_id, result)
            except Exception as e:
                self._report_task_failure(task_id, str(e))
            finally:
                if task_id in self.current_tasks:
                    del self.current_tasks[task_id]
        
        future.add_done_callback(on_completion)
    
    def _run_task(self, task_id: str, func: Callable, task_data: Dict[str, Any]) -> Any:
        """Run the actual task function."""
        start_time = time.time()
        
        try:
            # Extract arguments
            args = task_data.get('args', [])
            kwargs = task_data.get('kwargs', {})
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.execution_metrics['tasks_executed'] += 1
            self.execution_metrics['tasks_completed'] += 1
            self.execution_metrics['total_execution_time'] += execution_time
            self.execution_metrics['average_execution_time'] = (
                self.execution_metrics['total_execution_time'] / 
                self.execution_metrics['tasks_executed']
            )
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_metrics['tasks_executed'] += 1
            self.execution_metrics['tasks_failed'] += 1
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {e}")
            raise
    
    def _report_task_completion(self, task_id: str, result: Any):
        """Report task completion to coordinator."""
        message = {
            'type': 'task_completion',
            'task_id': task_id,
            'node_id': self.node_info.node_id,
            'result': self._serialize_result(result)
        }
        
        self.coordinator_socket.send_json(message)
        self.coordinator_socket.recv_json()  # Acknowledge
    
    def _report_task_failure(self, task_id: str, error: str):
        """Report task failure to coordinator."""
        message = {
            'type': 'task_failure',
            'task_id': task_id,
            'node_id': self.node_info.node_id,
            'error': error
        }
        
        self.coordinator_socket.send_json(message)
        self.coordinator_socket.recv_json()  # Acknowledge
    
    def _serialize_result(self, result: Any) -> Any:
        """Serialize task result for transmission."""
        try:
            # Try JSON serialization first
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return {
                '__pickled__': True,
                'data': pickle.dumps(result).hex()
            }


class DistributedCoordinator:
    """Distributed computing coordinator node."""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        
        # Components
        self.scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer(strategy="performance_based")
        self.auto_scaler = AutoScaler([ScalingPolicy.HYBRID])
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_last_seen: Dict[str, float] = {}
        self.node_heartbeat_timeout = 60.0
        
        # ZeroMQ communication
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # State management
        self.running = False
        self.server_thread: Optional[threading.Thread] = None
        self.maintenance_thread: Optional[threading.Thread] = None
        
        # Function registry (for validation)
        self.registered_functions: Set[str] = set()
        
        # Metrics collection
        self.coordinator_metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_nodes_connected': 0,
            'average_task_duration': 0.0
        }
    
    def start(self):
        """Start the coordinator."""
        if self.running:
            return
        
        self.running = True
        
        # Bind socket
        address = f"tcp://*:{self.port}"
        self.socket.bind(address)
        
        # Start threads
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        
        self.server_thread.start()
        self.maintenance_thread.start()
        
        logger.info(f"Coordinator started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the coordinator."""
        if not self.running:
            return
        
        self.running = False
        
        # Close socket
        self.socket.close()
        self.context.term()
        
        logger.info("Coordinator stopped")
    
    def submit_task(self, function_name: str, *args, 
                   priority: TaskPriority = TaskPriority.NORMAL,
                   requirements: Optional[Dict[str, Any]] = None,
                   timeout_seconds: float = 300.0,
                   **kwargs) -> str:
        """Submit a task for distributed execution."""
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            requirements=requirements or {},
            timeout_seconds=timeout_seconds
        )
        
        self.scheduler.submit_task(task)
        self.coordinator_metrics['total_tasks_submitted'] += 1
        
        logger.info(f"Task {task_id} submitted: {function_name}")
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of a completed task."""
        start_time = time.time()
        
        while True:
            task = self.scheduler.get_task_status(task_id)
            
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return self._deserialize_result(task.result)
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            elif task.status == TaskStatus.CANCELLED:
                raise RuntimeError(f"Task {task_id} was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            time.sleep(0.1)
    
    def register_function(self, name: str):
        """Register a function as available for distributed execution."""
        self.registered_functions.add(name)
        logger.info(f"Function {name} registered for distribution")
    
    def _server_loop(self):
        """Main server loop for handling worker communications."""
        while self.running:
            try:
                # Receive message from worker
                message = self.socket.recv_json(zmq.NOBLOCK)
                response = self._handle_worker_message(message)
                self.socket.send_json(response)
                
            except zmq.Again:
                time.sleep(0.01)  # No message available
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                # Send error response
                try:
                    self.socket.send_json({'type': 'error', 'message': str(e)})
                except:
                    pass
    
    def _handle_worker_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming worker message."""
        msg_type = message.get('type')
        
        if msg_type == 'heartbeat':
            return self._handle_heartbeat(message)
        elif msg_type == 'work_request':
            return self._handle_work_request(message)
        elif msg_type == 'task_completion':
            return self._handle_task_completion(message)
        elif msg_type == 'task_failure':
            return self._handle_task_failure(message)
        else:
            return {'type': 'error', 'message': f'Unknown message type: {msg_type}'}
    
    def _handle_heartbeat(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle worker heartbeat."""
        node_info_data = message['node_info']
        node_id = node_info_data['node_id']
        
        # Update node information
        node_info = NodeInfo(
            node_id=node_id,
            node_type=NodeType(node_info_data['node_type']),
            host=node_info_data['host'],
            port=node_info_data['port'],
            cpu_cores=node_info_data['cpu_cores'],
            memory_gb=node_info_data['memory_gb'],
            gpu_count=node_info_data.get('gpu_count', 0),
            capabilities=node_info_data.get('capabilities', []),
            load_factor=node_info_data.get('load_factor', 0.0),
            last_heartbeat=node_info_data['last_heartbeat'],
            is_active=True,
            metadata=node_info_data.get('metadata', {})
        )
        
        self.nodes[node_id] = node_info
        self.node_last_seen[node_id] = time.time()
        
        # Update load balancer metrics
        worker_metrics = message.get('metrics', {})
        self.load_balancer.update_node_metrics(node_id, worker_metrics)
        
        return {'type': 'heartbeat_ack'}
    
    def _handle_work_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle worker work request."""
        node_id = message['node_id']
        
        # Get next task from scheduler
        task = self.scheduler.get_next_task()
        
        if task is None:
            return {'type': 'no_work'}
        
        # Find suitable node for task
        available_nodes = [node for node in self.nodes.values() if node.is_active]
        selected_node = self.load_balancer.select_node(task, available_nodes)
        
        if selected_node is None or selected_node.node_id != node_id:
            # Put task back in queue
            self.scheduler.task_queues[task.priority].appendleft(task)
            return {'type': 'no_work'}
        
        # Assign task to node
        self.scheduler.start_task(task.task_id, node_id)
        
        # Serialize task data
        task_data = {
            'task_id': task.task_id,
            'function_name': task.function_name,
            'args': task.args,
            'kwargs': task.kwargs,
            'timeout_seconds': task.timeout_seconds
        }
        
        return {'type': 'task', 'task': task_data}
    
    def _handle_task_completion(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task completion from worker."""
        task_id = message['task_id']
        node_id = message['node_id']
        result = message['result']
        
        # Update scheduler
        self.scheduler.complete_task(task_id, result)
        self.coordinator_metrics['total_tasks_completed'] += 1
        
        # Update load balancer performance tracking
        task = self.scheduler.get_task_status(task_id)
        if task and task.started_at and task.completed_at:
            duration = task.completed_at - task.started_at
            self.load_balancer.record_task_completion(node_id, duration)
        
        return {'type': 'completion_ack'}
    
    def _handle_task_failure(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task failure from worker."""
        task_id = message['task_id']
        error = message['error']
        
        self.scheduler.fail_task(task_id, error)
        
        return {'type': 'failure_ack'}
    
    def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while self.running:
            try:
                self._cleanup_inactive_nodes()
                self._collect_scaling_metrics()
                self._make_scaling_decisions()
                time.sleep(30.0)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(30.0)
    
    def _cleanup_inactive_nodes(self):
        """Remove inactive nodes."""
        current_time = time.time()
        inactive_nodes = []
        
        for node_id, last_seen in self.node_last_seen.items():
            if current_time - last_seen > self.node_heartbeat_timeout:
                inactive_nodes.append(node_id)
        
        for node_id in inactive_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].is_active = False
                logger.warning(f"Node {node_id} marked as inactive")
    
    def _collect_scaling_metrics(self):
        """Collect metrics for auto-scaling decisions."""
        # Calculate current metrics
        active_nodes = [node for node in self.nodes.values() if node.is_active]
        
        if not active_nodes:
            return
        
        cpu_usage = statistics.mean(node.load_factor * 100 for node in active_nodes)
        memory_usage = 50.0  # Placeholder - would need actual memory metrics
        
        queue_stats = self.scheduler.get_queue_stats()
        queue_size = queue_stats['total_queued']
        active_workers = len(active_nodes)
        
        # Task performance metrics
        avg_task_duration = self.coordinator_metrics.get('average_task_duration', 0.0)
        task_throughput = self.coordinator_metrics['total_tasks_completed'] / max(1, time.time() - 0)  # Simplified
        error_rate = 0.0  # Would calculate from failed vs total tasks
        latency_p95 = avg_task_duration * 1.5  # Approximation
        
        metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_size=queue_size,
            active_workers=active_workers,
            average_task_duration=avg_task_duration,
            task_throughput=task_throughput,
            error_rate=error_rate,
            latency_p95=latency_p95
        )
        
        self.auto_scaler.add_metrics(metrics)
    
    def _make_scaling_decisions(self):
        """Make and implement scaling decisions."""
        new_worker_count = self.auto_scaler.make_scaling_decision()
        
        if new_worker_count is not None:
            current_workers = len([n for n in self.nodes.values() if n.is_active])
            
            if new_worker_count > current_workers:
                # Scale up - would trigger new worker node creation
                logger.info(f"Scaling up: need {new_worker_count - current_workers} more workers")
                # Implementation would depend on deployment platform (K8s, Docker, etc.)
            elif new_worker_count < current_workers:
                # Scale down - would gracefully shutdown excess workers
                logger.info(f"Scaling down: removing {current_workers - new_worker_count} workers")
                # Implementation would gracefully shutdown workers
            
            self.auto_scaler.update_worker_count(new_worker_count)
    
    def _deserialize_result(self, result: Any) -> Any:
        """Deserialize task result."""
        if isinstance(result, dict) and result.get('__pickled__'):
            return pickle.loads(bytes.fromhex(result['data']))
        return result
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_nodes = [node for node in self.nodes.values() if node.is_active]
        
        return {
            'coordinator_metrics': self.coordinator_metrics.copy(),
            'active_nodes': len(active_nodes),
            'total_nodes': len(self.nodes),
            'node_details': [node.to_dict() for node in active_nodes],
            'scheduler_stats': self.scheduler.get_queue_stats(),
            'auto_scaler_stats': self.auto_scaler.get_scaling_stats(),
            'registered_functions': list(self.registered_functions)
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example distributed computing setup
    def example_computation(x: float, y: float) -> float:
        """Example computation function."""
        time.sleep(0.1)  # Simulate work
        return x * y + np.random.random()
    
    def cpu_intensive_task(size: int) -> float:
        """CPU intensive task."""
        result = 0.0
        for i in range(size):
            result += np.sqrt(i)
        return result
    
    # Start coordinator
    coordinator = DistributedCoordinator(port=5555)
    coordinator.register_function("example_computation")
    coordinator.register_function("cpu_intensive_task")
    coordinator.start()
    
    try:
        print("Distributed computing framework demonstration...")
        
        # Create worker node
        worker_node_info = NodeInfo(
            node_id="worker_001",
            node_type=NodeType.WORKER,
            host="localhost",
            port=5556,
            cpu_cores=4,
            memory_gb=8.0,
            capabilities=["computation", "analysis"]
        )
        
        worker = DistributedWorkerNode(worker_node_info, "localhost", 5555)
        worker.register_function("example_computation", example_computation)
        worker.register_function("cpu_intensive_task", cpu_intensive_task)
        worker.start()
        
        # Wait for worker to connect
        time.sleep(2)
        
        # Submit tasks
        print("\nSubmitting distributed tasks...")
        task_ids = []
        
        for i in range(10):
            task_id = coordinator.submit_task(
                "example_computation",
                float(i), float(i + 1),
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        # Submit CPU intensive tasks
        for i in range(3):
            task_id = coordinator.submit_task(
                "cpu_intensive_task",
                100000 + i * 50000,
                priority=TaskPriority.HIGH
            )
            task_ids.append(task_id)
        
        # Wait for results
        print("\nWaiting for task results...")
        results = []
        
        for i, task_id in enumerate(task_ids):
            try:
                result = coordinator.get_task_result(task_id, timeout=30.0)
                results.append(result)
                print(f"Task {i+1}: {result:.4f}")
            except Exception as e:
                print(f"Task {i+1} failed: {e}")
        
        # Wait for metrics collection
        time.sleep(5)
        
        # Display cluster status
        print("\nCluster Status:")
        status = coordinator.get_cluster_status()
        print(f"Active nodes: {status['active_nodes']}")
        print(f"Completed tasks: {status['coordinator_metrics']['total_tasks_completed']}")
        print(f"Scheduler stats: {status['scheduler_stats']}")
        
        # Display auto-scaler status
        scaler_stats = status['auto_scaler_stats']
        print(f"\nAuto-scaler status:")
        print(f"Current workers: {scaler_stats['current_workers']}")
        print(f"Target workers: {scaler_stats['target_workers']}")
        
        if scaler_stats['recent_decisions']:
            print("Recent scaling decisions:")
            for decision in scaler_stats['recent_decisions'][-3:]:
                print(f"  {decision['old_workers']} -> {decision['new_workers']} workers")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        worker.stop()
        coordinator.stop()
        print("Distributed computing framework demonstration completed.")