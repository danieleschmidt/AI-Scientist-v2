#!/usr/bin/env python3
"""
Distributed Execution Framework for AI Scientist v2

Advanced distributed computing capabilities for scaling research workflows
across multiple nodes, with load balancing, fault tolerance, and resource optimization.
"""

import os
import json
import time
import logging
import threading
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import queue
import socket
import hashlib
from pathlib import Path
import concurrent.futures
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    disk_gb: float = 1.0
    network_mbps: float = 10.0
    estimated_duration: float = 60.0  # seconds
    priority: int = 5  # 1-10, higher is more important

@dataclass
class TaskDefinition:
    """Definition of a distributed task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    requirements: ResourceRequirement
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    max_retries: int = 3
    timeout: float = 3600.0  # seconds

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    node_id: Optional[str] = None
    completed_at: Optional[datetime] = None

@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    capabilities: Dict[ResourceType, float]
    current_load: Dict[ResourceType, float] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = "online"  # online, offline, busy, maintenance

class TaskExecutor(ABC):
    """Abstract base class for task executors."""
    
    @abstractmethod
    def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a task and return the result."""
        pass
    
    @abstractmethod
    def can_handle_task(self, task: TaskDefinition) -> bool:
        """Check if this executor can handle the given task."""
        pass

class ResearchIdeationExecutor(TaskExecutor):
    """Executor for research ideation tasks."""
    
    def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute research ideation task."""
        start_time = time.time()
        
        try:
            # Simulate ideation execution
            payload = task.payload
            workshop_file = payload.get('workshop_file')
            model = payload.get('model', 'gpt-4o')
            generations = payload.get('max_generations', 10)
            
            logger.info(f"Executing ideation task {task.task_id}: {workshop_file}")
            
            # Simulate processing time
            processing_time = min(task.requirements.estimated_duration, 300)  # Cap at 5 minutes for demo
            time.sleep(processing_time * 0.01)  # Scaled down for demo
            
            # Mock result
            result_data = {
                'ideas_generated': generations,
                'workshop_file': workshop_file,
                'model_used': model,
                'novelty_score': 0.87,
                'feasibility_score': 0.92,
                'ideas': [
                    {
                        'title': f'Quantum-Enhanced Idea {i}',
                        'description': f'Novel approach {i} to the research problem',
                        'feasibility': 0.8 + (i * 0.02),
                        'novelty': 0.85 + (i * 0.01)
                    }
                    for i in range(1, min(generations, 6))
                ]
            }
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=execution_time,
                resource_usage={
                    ResourceType.CPU: 0.8,
                    ResourceType.MEMORY: 2.1,
                    ResourceType.NETWORK: 5.2
                },
                completed_at=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ideation task {task.task_id} failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                completed_at=datetime.now()
            )
    
    def can_handle_task(self, task: TaskDefinition) -> bool:
        """Check if this executor can handle ideation tasks."""
        return task.task_type == "research_ideation"

class ExperimentExecutor(TaskExecutor):
    """Executor for experimental research tasks."""
    
    def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute experimental research task."""
        start_time = time.time()
        
        try:
            payload = task.payload
            ideas_file = payload.get('ideas_file')
            experiments = payload.get('num_experiments', 5)
            
            logger.info(f"Executing experiment task {task.task_id}: {ideas_file}")
            
            # Simulate processing time
            processing_time = min(task.requirements.estimated_duration, 600)  # Cap at 10 minutes
            time.sleep(processing_time * 0.005)  # Scaled down for demo
            
            # Mock result
            result_data = {
                'experiments_completed': experiments,
                'ideas_file': ideas_file,
                'success_rate': 0.89,
                'avg_performance': 0.84,
                'experiments': [
                    {
                        'experiment_id': f'exp_{i}',
                        'performance': 0.75 + (i * 0.03),
                        'convergence_time': 120 + (i * 10),
                        'resource_efficiency': 0.82 + (i * 0.02)
                    }
                    for i in range(1, min(experiments, 6))
                ]
            }
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=execution_time,
                resource_usage={
                    ResourceType.CPU: 0.95,
                    ResourceType.MEMORY: 4.8,
                    ResourceType.GPU: 0.75,
                    ResourceType.NETWORK: 8.3
                },
                completed_at=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Experiment task {task.task_id} failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                completed_at=datetime.now()
            )
    
    def can_handle_task(self, task: TaskDefinition) -> bool:
        """Check if this executor can handle experiment tasks."""
        return task.task_type == "experimental_research"

class LoadBalancer:
    """Intelligent load balancer for distributed tasks."""
    
    def __init__(self, nodes: List[NodeInfo]):
        self.nodes = {node.node_id: node for node in nodes}
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self._lock = threading.Lock()
        
    def add_node(self, node: NodeInfo):
        """Add a compute node to the cluster."""
        with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added node {node.node_id} to cluster")
    
    def remove_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed node {node_id} from cluster")
    
    def select_optimal_node(self, task: TaskDefinition) -> Optional[NodeInfo]:
        """Select the optimal node for a task based on resources and load."""
        with self._lock:
            available_nodes = []
            
            for node in self.nodes.values():
                if self._can_node_handle_task(node, task):
                    score = self._calculate_node_score(node, task)
                    available_nodes.append((score, node))
            
            if not available_nodes:
                return None
            
            # Sort by score (higher is better)
            available_nodes.sort(key=lambda x: x[0], reverse=True)
            return available_nodes[0][1]
    
    def _can_node_handle_task(self, node: NodeInfo, task: TaskDefinition) -> bool:
        """Check if a node can handle a specific task."""
        req = task.requirements
        caps = node.capabilities
        load = node.current_load
        
        # Check if node has sufficient resources
        available_cpu = caps.get(ResourceType.CPU, 0) - load.get(ResourceType.CPU, 0)
        available_memory = caps.get(ResourceType.MEMORY, 0) - load.get(ResourceType.MEMORY, 0)
        available_gpu = caps.get(ResourceType.GPU, 0) - load.get(ResourceType.GPU, 0)
        
        return (available_cpu >= req.cpu_cores and 
                available_memory >= req.memory_gb and
                available_gpu >= req.gpu_count and
                node.status == "online")
    
    def _calculate_node_score(self, node: NodeInfo, task: TaskDefinition) -> float:
        """Calculate a score for how well a node matches a task."""
        req = task.requirements
        caps = node.capabilities
        load = node.current_load
        
        # Base score on resource availability
        cpu_ratio = (caps.get(ResourceType.CPU, 0) - load.get(ResourceType.CPU, 0)) / req.cpu_cores
        memory_ratio = (caps.get(ResourceType.MEMORY, 0) - load.get(ResourceType.MEMORY, 0)) / req.memory_gb
        
        # Bonus for GPU availability if needed
        gpu_bonus = 0
        if req.gpu_count > 0:
            available_gpu = caps.get(ResourceType.GPU, 0) - load.get(ResourceType.GPU, 0)
            if available_gpu >= req.gpu_count:
                gpu_bonus = 10
        
        # Penalty for high current load
        load_penalty = sum(load.values()) / max(sum(caps.values()), 1)
        
        # Priority bonus
        priority_bonus = task.requirements.priority * 0.5
        
        # Calculate final score
        score = (cpu_ratio + memory_ratio + gpu_bonus + priority_bonus - load_penalty * 5)
        
        return max(score, 0)

class DistributedTaskManager:
    """Manages distributed task execution across multiple nodes."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (mp.cpu_count() * 2)
        self.executors = {}
        self.nodes = []
        self.load_balancer = None
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self._running = False
        self._worker_threads = []
        self._lock = threading.Lock()
        
        # Register default executors
        self._register_default_executors()
        
        # Initialize with local node
        self._initialize_local_node()
    
    def _register_default_executors(self):
        """Register default task executors."""
        self.executors['research_ideation'] = ResearchIdeationExecutor()
        self.executors['experimental_research'] = ExperimentExecutor()
    
    def _initialize_local_node(self):
        """Initialize local compute node."""
        try:
            import psutil
            
            # Get system capabilities
            capabilities = {
                ResourceType.CPU: psutil.cpu_count(),
                ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),  # GB
                ResourceType.DISK: psutil.disk_usage('/').total / (1024**3),  # GB
                ResourceType.NETWORK: 1000.0,  # Assume 1Gbps
                ResourceType.GPU: self._detect_gpu_count()
            }
            
        except ImportError:
            # Fallback if psutil not available
            capabilities = {
                ResourceType.CPU: mp.cpu_count(),
                ResourceType.MEMORY: 8.0,  # Assume 8GB
                ResourceType.DISK: 100.0,  # Assume 100GB
                ResourceType.NETWORK: 100.0,  # Assume 100Mbps
                ResourceType.GPU: self._detect_gpu_count()
            }
        
        local_node = NodeInfo(
            node_id=f"local_{socket.gethostname()}",
            hostname=socket.gethostname(),
            ip_address="127.0.0.1",
            capabilities=capabilities
        )
        
        self.nodes.append(local_node)
        self.load_balancer = LoadBalancer(self.nodes)
        
        logger.info(f"Initialized local node: {local_node.node_id}")
        logger.info(f"Node capabilities: {capabilities}")
    
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.strip().split('\\n') if line.strip()]
                return len(gpu_lines)
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        return 0
    
    def start(self):
        """Start the distributed task manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self._worker_threads.append(thread)
        
        logger.info(f"Started distributed task manager with {self.max_workers} workers")
    
    def stop(self):
        """Stop the distributed task manager."""
        self._running = False
        
        # Wait for workers to finish
        for thread in self._worker_threads:
            thread.join(timeout=5)
        
        logger.info("Stopped distributed task manager")
    
    def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for distributed execution."""
        with self._lock:
            # Add task to queue with priority
            priority = -task.requirements.priority  # Negative for max-priority queue
            self.task_queue.put((priority, time.time(), task))
            self.active_tasks[task.task_id] = task
            
            logger.info(f"Submitted task {task.task_id} ({task.task_type}) for execution")
            return task.task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """Get the result of a task execution."""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the current status of a task."""
        with self._lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            elif task_id in self.active_tasks:
                return TaskStatus.RUNNING
            else:
                return TaskStatus.PENDING
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for task execution."""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next task from queue
                try:
                    priority, submission_time, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Execute task
                result = self._execute_task(task, worker_id)
                
                # Store result
                with self._lock:
                    self.completed_tasks[task.task_id] = result
                    if task.task_id in self.active_tasks:
                        del self.active_tasks[task.task_id]
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _execute_task(self, task: TaskDefinition, worker_id: int) -> TaskResult:
        """Execute a task using appropriate executor."""
        start_time = time.time()
        
        try:
            # Find appropriate executor
            executor = self.executors.get(task.task_type)
            if not executor:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"No executor found for task type: {task.task_type}",
                    execution_time=time.time() - start_time,
                    completed_at=datetime.now()
                )
            
            # Execute task with timeout
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor_pool:
                    future = executor_pool.submit(executor.execute_task, task)
                    result = future.result(timeout=task.timeout)
                    result.node_id = f"worker_{worker_id}"
                    return result
                    
            except concurrent.futures.TimeoutError:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Task timed out after {task.timeout} seconds",
                    execution_time=task.timeout,
                    completed_at=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                completed_at=datetime.now()
            )
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status and statistics."""
        with self._lock:
            node_stats = []
            total_capacity = {resource: 0.0 for resource in ResourceType}
            total_usage = {resource: 0.0 for resource in ResourceType}
            
            for node in self.nodes:
                for resource, capacity in node.capabilities.items():
                    total_capacity[resource] += capacity
                    total_usage[resource] += node.current_load.get(resource, 0.0)
                
                node_stats.append({
                    'node_id': node.node_id,
                    'hostname': node.hostname,
                    'status': node.status,
                    'active_tasks': len(node.active_tasks),
                    'cpu_usage': node.current_load.get(ResourceType.CPU, 0.0),
                    'memory_usage': node.current_load.get(ResourceType.MEMORY, 0.0),
                    'gpu_usage': node.current_load.get(ResourceType.GPU, 0.0)
                })
            
            return {
                'cluster_size': len(self.nodes),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'total_capacity': {r.value: v for r, v in total_capacity.items()},
                'total_usage': {r.value: v for r, v in total_usage.items()},
                'utilization': {
                    r.value: (total_usage[r] / total_capacity[r] * 100) if total_capacity[r] > 0 else 0
                    for r in ResourceType
                },
                'nodes': node_stats,
                'workers': len(self._worker_threads),
                'queue_size': self.task_queue.qsize()
            }
    
    def register_executor(self, task_type: str, executor: TaskExecutor):
        """Register a custom task executor."""
        self.executors[task_type] = executor
        logger.info(f"Registered executor for task type: {task_type}")

# Convenience functions for creating common tasks
def create_ideation_task(task_id: str, workshop_file: str, model: str = "gpt-4o", 
                        max_generations: int = 10, priority: int = 5) -> TaskDefinition:
    """Create a research ideation task."""
    return TaskDefinition(
        task_id=task_id,
        task_type="research_ideation",
        payload={
            'workshop_file': workshop_file,
            'model': model,
            'max_generations': max_generations
        },
        requirements=ResourceRequirement(
            cpu_cores=2,
            memory_gb=4.0,
            gpu_count=0,
            estimated_duration=180.0,
            priority=priority
        )
    )

def create_experiment_task(task_id: str, ideas_file: str, num_experiments: int = 5,
                          priority: int = 7) -> TaskDefinition:
    """Create an experimental research task."""
    return TaskDefinition(
        task_id=task_id,
        task_type="experimental_research",
        payload={
            'ideas_file': ideas_file,
            'num_experiments': num_experiments
        },
        requirements=ResourceRequirement(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_count=1,
            estimated_duration=600.0,
            priority=priority
        )
    )

# Global distributed task manager instance
distributed_manager = DistributedTaskManager()

def initialize_distributed_computing(max_workers: int = None):
    """Initialize the distributed computing system."""
    global distributed_manager
    if max_workers:
        distributed_manager.max_workers = max_workers
    distributed_manager.start()
    logger.info("Distributed computing system initialized")

def shutdown_distributed_computing():
    """Shutdown the distributed computing system."""
    global distributed_manager
    distributed_manager.stop()
    logger.info("Distributed computing system shutdown")