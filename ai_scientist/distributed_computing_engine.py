#!/usr/bin/env python3
"""
Distributed Computing Engine - Generation 3: MAKE IT SCALE

High-performance distributed computing system with load balancing, auto-scaling, and intelligent task distribution.
Enables massive parallel processing across multiple nodes for AI research workflows.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import multiprocessing as mp
import threading
import queue
import os
import sys

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel


class NodeStatus(Enum):
    """Distributed computing node status."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Distributed task status."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class ComputeNode:
    """Represents a distributed computing node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus = NodeStatus.INITIALIZING
    
    # Resource capacity
    cpu_cores: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    storage_gb: float = 0.0
    
    # Current utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    storage_usage: float = 0.0
    
    # Performance metrics
    last_heartbeat: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_time: float = 0.0
    network_latency: float = 0.0
    
    # Node capabilities
    supported_frameworks: List[str] = field(default_factory=list)
    specialized_hardware: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy and available."""
        if self.status in [NodeStatus.FAILED, NodeStatus.MAINTENANCE]:
            return False
        
        if self.last_heartbeat:
            time_since_heartbeat = datetime.now() - self.last_heartbeat
            if time_since_heartbeat > timedelta(minutes=2):
                return False
        
        return True
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0+)."""
        cpu_load = self.cpu_usage / 100.0
        memory_load = self.memory_usage / 100.0
        
        # Weighted average of resource utilization
        return (cpu_load * 0.6 + memory_load * 0.4)
    
    @property
    def capacity_score(self) -> float:
        """Calculate node capacity score for task assignment."""
        # Higher score = better capacity
        base_score = (self.cpu_cores * 0.4 + 
                     self.memory_gb * 0.3 + 
                     self.gpu_count * 0.2 + 
                     self.storage_gb * 0.1)
        
        # Adjust for current load (lower load = higher score)
        load_adjustment = max(0.1, 1.0 - self.load_factor)
        
        # Adjust for reliability (more completed tasks = higher score)
        reliability = 1.0
        if self.tasks_completed > 0:
            reliability = self.tasks_completed / (self.tasks_completed + self.tasks_failed + 1)
        
        return base_score * load_adjustment * reliability


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    task_id: str
    name: str
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    # Resource requirements
    cpu_cores_required: int = 1
    memory_gb_required: float = 1.0
    gpu_required: bool = False
    estimated_duration: float = 60.0  # seconds
    
    # Execution tracking
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 5.0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready for execution (dependencies completed)."""
        return self.status == TaskStatus.QUEUED
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def should_retry(self) -> bool:
        """Check if task should be retried after failure."""
        return (self.status == TaskStatus.FAILED and 
                self.retry_count < self.max_retries)


@dataclass
class ClusterMetrics:
    """Cluster-wide performance metrics."""
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    total_gpu_count: int = 0
    # Utilization metrics
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_gpu_utilization: float = 0.0
    
    # Task metrics
    tasks_pending: int = 0
    tasks_running: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    # Performance metrics
    avg_task_completion_time: float = 0.0
    tasks_per_minute: float = 0.0
    cluster_efficiency: float = 0.0
    
    # Auto-scaling metrics
    scale_up_triggers: int = 0
    scale_down_triggers: int = 0
    last_scaling_event: Optional[datetime] = None


class LoadBalancer:
    """Intelligent load balancer for task distribution."""
    
    def __init__(self, strategy: str = "capacity_aware"):
        self.strategy = strategy
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_strategy,
            'least_loaded': self._least_loaded_strategy,
            'capacity_aware': self._capacity_aware_strategy,
            'task_affinity': self._task_affinity_strategy,
            'geographic': self._geographic_strategy
        }
        
        # Strategy state
        self.round_robin_index = 0
        self.task_assignments = {}  # task_id -> node_id
        self.node_affinities = {}   # framework -> preferred_nodes
    
    def select_node(
        self, 
        task: DistributedTask, 
        available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """Select the best node for task execution."""
        
        if not available_nodes:
            return None
        
        # Filter nodes that meet task requirements
        suitable_nodes = self._filter_suitable_nodes(task, available_nodes)
        
        if not suitable_nodes:
            self.logger.warning(f"No suitable nodes for task {task.task_id}")
            return None
        
        # Apply load balancing strategy
        strategy_func = self.strategies.get(self.strategy, self._capacity_aware_strategy)
        selected_node = strategy_func(task, suitable_nodes)
        
        self.logger.info(f"Selected node {selected_node.node_id} for task {task.task_id}")
        
        return selected_node
    
    def _filter_suitable_nodes(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Filter nodes that can handle the task requirements."""
        
        suitable_nodes = []
        
        for node in nodes:
            # Check if node is healthy and available
            if not node.is_healthy or node.status != NodeStatus.IDLE:
                continue
            
            # Check resource requirements
            if (node.cpu_cores >= task.cpu_cores_required and
                node.memory_gb >= task.memory_gb_required):
                
                # Check GPU requirement
                if task.gpu_required and node.gpu_count == 0:
                    continue
                
                # Check if node is not overloaded
                if node.load_factor < 0.9:  # 90% load threshold
                    suitable_nodes.append(node)
        
        return suitable_nodes
    
    def _round_robin_strategy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> ComputeNode:
        """Simple round-robin node selection."""
        
        selected_node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        
        return selected_node
    
    def _least_loaded_strategy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> ComputeNode:
        """Select node with lowest current load."""
        
        return min(nodes, key=lambda node: node.load_factor)
    
    def _capacity_aware_strategy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> ComputeNode:
        """Select node based on capacity score and task requirements."""
        
        # Score nodes based on multiple factors
        node_scores = []
        
        for node in nodes:
            score = node.capacity_score
            
            # Bonus for GPU availability if required
            if task.gpu_required and node.gpu_count > 0:
                score *= 1.5
            
            # Bonus for low network latency
            latency_factor = max(0.5, 1.0 - (node.network_latency / 1000.0))  # Assume ms
            score *= latency_factor
            
            # Priority adjustment
            if task.priority == TaskPriority.URGENT:
                score *= 2.0
            elif task.priority == TaskPriority.CRITICAL:
                score *= 1.5
            
            node_scores.append((node, score))
        
        # Select node with highest score
        best_node, _ = max(node_scores, key=lambda x: x[1])
        
        return best_node
    
    def _task_affinity_strategy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> ComputeNode:
        """Select node based on task-to-node affinity (framework support, etc.)."""
        
        # Try to find nodes with framework affinity first
        framework = task.kwargs.get('framework', 'general')
        
        preferred_nodes = []
        for node in nodes:
            if framework in node.supported_frameworks:
                preferred_nodes.append(node)
        
        # If no preferred nodes, fall back to capacity-aware strategy
        target_nodes = preferred_nodes if preferred_nodes else nodes
        
        return self._capacity_aware_strategy(task, target_nodes)
    
    def _geographic_strategy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> ComputeNode:
        """Select node based on geographic proximity."""
        
        # This would use geographic information in a real implementation
        # For demo purposes, use capacity-aware strategy
        return self._capacity_aware_strategy(task, nodes)


class DistributedComputingEngine:
    """
    Generation 3: MAKE IT SCALE
    High-performance distributed computing engine with auto-scaling.
    """
    
    def __init__(
        self,
        cluster_name: str = "ai_scientist_cluster",
        workspace_dir: str = "distributed_workspace"
    ):
        self.cluster_name = cluster_name
        self.console = Console()
        self.logger = self._setup_logging()
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Cluster state
        self.nodes: Dict[str, ComputeNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue = asyncio.Queue()
        
        # Load balancer
        self.load_balancer = LoadBalancer(strategy="capacity_aware")
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Auto-scaling configuration
        self.auto_scaling_config = {
            'enabled': True,
            'min_nodes': 1,
            'max_nodes': 10,
            'scale_up_threshold': 0.8,   # CPU utilization
            'scale_down_threshold': 0.3,
            'scale_up_cooldown': timedelta(minutes=5),
            'scale_down_cooldown': timedelta(minutes=10)
        }
        
        # Metrics and monitoring
        self.cluster_metrics = ClusterMetrics()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Control flags
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Initialize local node (master node)
        self._initialize_local_node()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup distributed computing logging."""
        
        logger = logging.getLogger(f"{__name__}.DistributedComputingEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - DISTRIBUTED - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_local_node(self):
        """Initialize the local master node."""
        
        local_node = ComputeNode(
            node_id=f"master_{uuid.uuid4().hex[:8]}",
            hostname=os.uname().nodename,
            ip_address="127.0.0.1",
            port=8080,
            cpu_cores=mp.cpu_count(),
            memory_gb=self._get_system_memory_gb(),
            gpu_count=self._get_gpu_count(),
            storage_gb=self._get_available_storage_gb(),
            status=NodeStatus.IDLE,
            supported_frameworks=['pytorch', 'tensorflow', 'scikit-learn', 'general']
        )
        
        self.nodes[local_node.node_id] = local_node
        self.logger.info(f"Initialized local master node: {local_node.node_id}")
    
    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            return 8.0  # Default assumption
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    
    def _get_available_storage_gb(self) -> float:
        """Get available storage in GB."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            return free / (1024 ** 3)
        except Exception:
            return 100.0  # Default assumption
    
    async def start_cluster(self):
        """Start the distributed computing cluster."""
        
        if self.is_running:
            self.logger.warning("Cluster already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting cluster: {self.cluster_name}")
        
        # Start scheduler and monitor tasks
        self.scheduler_task = asyncio.create_task(self._task_scheduler_loop())
        self.monitor_task = asyncio.create_task(self._cluster_monitor_loop())
        
        # Update cluster metrics
        self._update_cluster_metrics()
        
        self.console.print(f"[bold green]🚀 Distributed cluster '{self.cluster_name}' started[/bold green]")
        self.console.print(f"[cyan]• Nodes: {len(self.nodes)}[/cyan]")
        self.console.print(f"[cyan]• Total CPU Cores: {self.cluster_metrics.total_cpu_cores}[/cyan]")
        self.console.print(f"[cyan]• Total Memory: {self.cluster_metrics.total_memory_gb:.1f} GB[/cyan]")
    
    async def stop_cluster(self):
        """Stop the distributed computing cluster."""
        
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping cluster...")
        
        # Cancel background tasks
        if self.scheduler_task:
            self.scheduler_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Wait for tasks to complete
        active_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.RUNNING]
        if active_tasks:
            self.console.print(f"[yellow]Waiting for {len(active_tasks)} running tasks to complete...[/yellow]")
            
            # Give tasks some time to complete gracefully
            await asyncio.sleep(5)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.console.print("[bold yellow]⏹️ Distributed cluster stopped[/bold yellow]")
    
    def add_node(
        self,
        hostname: str,
        ip_address: str,
        port: int = 8080,
        cpu_cores: int = 4,
        memory_gb: float = 8.0,
        gpu_count: int = 0,
        supported_frameworks: Optional[List[str]] = None
    ) -> str:
        """Add a new compute node to the cluster."""
        
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        node = ComputeNode(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            status=NodeStatus.IDLE,
            supported_frameworks=supported_frameworks or ['general'],
            last_heartbeat=datetime.now()
        )
        
        self.nodes[node_id] = node
        self._update_cluster_metrics()
        
        self.logger.info(f"Added node {node_id} ({hostname}:{port})")
        self.console.print(f"[green]✅ Added compute node: {hostname} ({cpu_cores} cores, {memory_gb}GB RAM)[/green]")
        
        return node_id
    
    def remove_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        
        if node_id not in self.nodes:
            self.logger.warning(f"Attempted to remove non-existent node: {node_id}")
            return
        
        node = self.nodes[node_id]
        
        # Mark node as maintenance to prevent new task assignments
        node.status = NodeStatus.MAINTENANCE
        
        # Wait for running tasks to complete or reassign them
        running_tasks = [task for task in self.tasks.values() 
                        if task.assigned_node == node_id and task.status == TaskStatus.RUNNING]
        
        if running_tasks:
            self.logger.info(f"Reassigning {len(running_tasks)} tasks from node {node_id}")
            for task in running_tasks:
                task.assigned_node = None
                task.status = TaskStatus.QUEUED
        
        # Remove node
        del self.nodes[node_id]
        self._update_cluster_metrics()
        
        self.logger.info(f"Removed node {node_id}")
        self.console.print(f"[yellow]➖ Removed compute node: {node.hostname}[/yellow]")
    
    def submit_task(
        self,
        function: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        cpu_cores: int = 1,
        memory_gb: float = 1.0,
        gpu_required: bool = False,
        estimated_duration: float = 60.0,
        max_retries: int = 3,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Submit a task for distributed execution."""
        
        task_id = f"task_{uuid.uuid4().hex}"
        
        task = DistributedTask(
            task_id=task_id,
            name=getattr(function, '__name__', 'anonymous_function'),
            function_name=function.__name__,
            args=args,
            kwargs=kwargs,
            priority=priority,
            cpu_cores_required=cpu_cores,
            memory_gb_required=memory_gb,
            gpu_required=gpu_required,
            estimated_duration=estimated_duration,
            max_retries=max_retries,
            dependencies=dependencies or []
        )
        
        # Store function for execution (in real implementation, this would be serialized)
        task._function = function  # Store reference for execution
        
        self.tasks[task_id] = task
        
        # Check dependencies and queue task
        if self._are_dependencies_satisfied(task):
            task.status = TaskStatus.QUEUED
            asyncio.create_task(self.task_queue.put(task))
        else:
            task.status = TaskStatus.PENDING
        
        self.logger.info(f"Submitted task {task_id}: {task.name}")
        
        return task_id
    
    def _are_dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if all task dependencies are satisfied."""
        
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.tasks:
                return False
            
            dep_task = self.tasks[dep_task_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _task_scheduler_loop(self):
        """Main task scheduler loop."""
        
        while self.is_running:
            try:
                # Get next task from queue (with timeout)
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Find available node for task
                available_nodes = [
                    node for node in self.nodes.values() 
                    if node.is_healthy and node.status in [NodeStatus.IDLE, NodeStatus.BUSY]
                ]
                
                if not available_nodes:
                    # No nodes available, put task back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                # Use load balancer to select best node
                selected_node = self.load_balancer.select_node(task, available_nodes)
                
                if selected_node:
                    # Assign task to node
                    task.assigned_node = selected_node.node_id
                    task.status = TaskStatus.ASSIGNED
                    
                    # Execute task asynchronously
                    asyncio.create_task(self._execute_task(task, selected_node))
                else:
                    # No suitable node found, put task back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: DistributedTask, node: ComputeNode):
        """Execute a task on a specific node."""
        
        try:
            # Update task and node status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            node.status = NodeStatus.BUSY
            
            self.logger.info(f"Executing task {task.task_id} on node {node.node_id}")
            
            # Execute function based on task requirements
            if hasattr(task, '_function'):
                function = task._function
                
                # Choose execution method based on requirements
                if task.cpu_cores_required > 1 or task.estimated_duration > 300:  # 5 minutes
                    # Use process pool for CPU-intensive or long-running tasks
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.process_pool,
                        function,
                        *task.args,
                        **task.kwargs
                    )
                else:
                    # Use thread pool for I/O-bound tasks
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.thread_pool,
                        function,
                        *task.args,
                        **task.kwargs
                    )
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                # Update node metrics
                node.tasks_completed += 1
                if task.execution_time:
                    node.average_task_time = (
                        (node.average_task_time * (node.tasks_completed - 1) + task.execution_time) /
                        node.tasks_completed
                    )
                
                self.logger.info(f"Task {task.task_id} completed successfully in {task.execution_time:.2f}s")
                
                # Check for dependent tasks
                self._check_dependent_tasks(task.task_id)
                
            else:
                raise ValueError("Task function not found")
        
        except Exception as e:
            # Task failed
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            # Update node metrics
            node.tasks_failed += 1
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Check if task should be retried
            if task.should_retry:
                task.retry_count += 1
                task.status = TaskStatus.RETRY
                task.assigned_node = None
                
                # Re-queue task with delay
                asyncio.create_task(self._requeue_task_with_delay(task))
                
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
        
        finally:
            # Update node status
            node.status = NodeStatus.IDLE
            self._update_cluster_metrics()
    
    async def _requeue_task_with_delay(self, task: DistributedTask):
        """Re-queue a failed task after delay."""
        
        await asyncio.sleep(task.retry_delay)
        task.status = TaskStatus.QUEUED
        await self.task_queue.put(task)
    
    def _check_dependent_tasks(self, completed_task_id: str):
        """Check and queue tasks that depend on the completed task."""
        
        dependent_tasks = [
            task for task in self.tasks.values()
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies)
        ]
        
        for task in dependent_tasks:
            if self._are_dependencies_satisfied(task):
                task.status = TaskStatus.QUEUED
                asyncio.create_task(self.task_queue.put(task))
    
    async def _cluster_monitor_loop(self):
        """Monitor cluster health and performance."""
        
        while self.is_running:
            try:
                # Update node heartbeats and status
                current_time = datetime.now()
                
                for node in self.nodes.values():
                    # Simulate heartbeat updates (in real implementation, nodes would send heartbeats)
                    node.last_heartbeat = current_time
                    
                    # Update resource utilization (simulated)
                    if node.status == NodeStatus.BUSY:
                        node.cpu_usage = min(95.0, node.cpu_usage + 10.0)
                        node.memory_usage = min(90.0, node.memory_usage + 5.0)
                    else:
                        node.cpu_usage = max(5.0, node.cpu_usage - 5.0)
                        node.memory_usage = max(10.0, node.memory_usage - 3.0)
                
                # Update cluster metrics
                self._update_cluster_metrics()
                
                # Check auto-scaling conditions
                if self.auto_scaling_config['enabled']:
                    await self._check_auto_scaling()
                
                # Record performance history
                self._record_performance_snapshot()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
            
            except Exception as e:
                self.logger.error(f"Error in cluster monitor: {e}")
                await asyncio.sleep(30)
    
    def _update_cluster_metrics(self):
        """Update cluster-wide metrics."""
        
        healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]
        
        self.cluster_metrics.total_nodes = len(self.nodes)
        self.cluster_metrics.healthy_nodes = len(healthy_nodes)
        
        if healthy_nodes:
            self.cluster_metrics.total_cpu_cores = sum(node.cpu_cores for node in healthy_nodes)
            self.cluster_metrics.total_memory_gb = sum(node.memory_gb for node in healthy_nodes)
            self.cluster_metrics.total_gpu_count = sum(node.gpu_count for node in healthy_nodes)
            
            self.cluster_metrics.avg_cpu_utilization = sum(node.cpu_usage for node in healthy_nodes) / len(healthy_nodes)
            self.cluster_metrics.avg_memory_utilization = sum(node.memory_usage for node in healthy_nodes) / len(healthy_nodes)
            self.cluster_metrics.avg_gpu_utilization = sum(node.gpu_usage for node in healthy_nodes) / len(healthy_nodes)
        
        # Task counts
        self.cluster_metrics.tasks_pending = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        self.cluster_metrics.tasks_running = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        self.cluster_metrics.tasks_completed = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        self.cluster_metrics.tasks_failed = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        # Performance metrics
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED and t.execution_time]
        if completed_tasks:
            self.cluster_metrics.avg_task_completion_time = sum(t.execution_time for t in completed_tasks) / len(completed_tasks)
        
        # Calculate cluster efficiency
        if self.cluster_metrics.total_cpu_cores > 0:
            utilized_cores = (self.cluster_metrics.avg_cpu_utilization / 100.0) * self.cluster_metrics.total_cpu_cores
            self.cluster_metrics.cluster_efficiency = utilized_cores / self.cluster_metrics.total_cpu_cores
    
    async def _check_auto_scaling(self):
        """Check auto-scaling conditions and trigger scaling events."""
        
        config = self.auto_scaling_config
        metrics = self.cluster_metrics
        
        current_time = datetime.now()
        
        # Check cooldown period
        if (metrics.last_scaling_event and 
            current_time - metrics.last_scaling_event < config['scale_up_cooldown']):
            return
        
        # Scale up conditions
        should_scale_up = (
            metrics.avg_cpu_utilization > config['scale_up_threshold'] * 100 and
            metrics.healthy_nodes < config['max_nodes'] and
            metrics.tasks_pending > 0
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics.avg_cpu_utilization < config['scale_down_threshold'] * 100 and
            metrics.healthy_nodes > config['min_nodes'] and
            metrics.tasks_pending == 0 and
            metrics.tasks_running == 0
        )
        
        if should_scale_up:
            await self._scale_up()
            metrics.scale_up_triggers += 1
            metrics.last_scaling_event = current_time
            
        elif should_scale_down:
            await self._scale_down()
            metrics.scale_down_triggers += 1
            metrics.last_scaling_event = current_time
    
    async def _scale_up(self):
        """Scale up the cluster by adding nodes."""
        
        # In a real implementation, this would provision new nodes
        # For demo purposes, simulate adding a node
        
        new_node_id = self.add_node(
            hostname=f"auto-node-{len(self.nodes)}",
            ip_address=f"192.168.1.{100 + len(self.nodes)}",
            cpu_cores=4,
            memory_gb=8.0,
            gpu_count=0
        )
        
        self.logger.info(f"Auto-scaled up: Added node {new_node_id}")
        self.console.print("[bold green]📈 Auto-scaled up: Added new compute node[/bold green]")
    
    async def _scale_down(self):
        """Scale down the cluster by removing nodes."""
        
        # Find nodes suitable for removal (not master, idle, no running tasks)
        removable_nodes = [
            node for node in self.nodes.values()
            if (not node.node_id.startswith('master') and
                node.status == NodeStatus.IDLE and
                not any(task.assigned_node == node.node_id and task.status == TaskStatus.RUNNING 
                       for task in self.tasks.values()))
        ]
        
        if removable_nodes:
            # Remove the node with lowest utilization
            node_to_remove = min(removable_nodes, key=lambda n: n.load_factor)
            self.remove_node(node_to_remove.node_id)
            
            self.logger.info(f"Auto-scaled down: Removed node {node_to_remove.node_id}")
            self.console.print("[bold yellow]📉 Auto-scaled down: Removed idle compute node[/bold yellow]")
    
    def _record_performance_snapshot(self):
        """Record current performance metrics for history."""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'cluster_metrics': asdict(self.cluster_metrics),
            'node_count': len(self.nodes),
            'task_queue_size': self.task_queue.qsize()
        }
        
        self.performance_history.append(snapshot)
        
        # Keep only recent history (last 24 hours worth of 30-second intervals)
        max_history = 24 * 60 * 2  # 2880 entries
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def get_task_result(self, task_id: str) -> Tuple[TaskStatus, Any]:
        """Get the result of a specific task."""
        
        if task_id not in self.tasks:
            return TaskStatus.FAILED, f"Task {task_id} not found"
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.COMPLETED:
            return task.status, task.result
        elif task.status == TaskStatus.FAILED:
            return task.status, task.error
        else:
            return task.status, None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Tuple[TaskStatus, Any]:
        """Wait for a task to complete and return its result."""
        
        start_time = time.time()
        
        while True:
            status, result = self.get_task_result(task_id)
            
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return status, result
            
            if timeout and time.time() - start_time > timeout:
                return TaskStatus.FAILED, f"Task {task_id} timed out after {timeout} seconds"
            
            await asyncio.sleep(1)
    
    def create_cluster_dashboard(self) -> Table:
        """Create cluster monitoring dashboard."""
        
        table = Table(title=f"🖥️ Distributed Cluster: {self.cluster_name}")
        
        table.add_column("Node ID", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("CPU", style="green")
        table.add_column("Memory", style="blue")
        table.add_column("Tasks", style="yellow")
        table.add_column("Load", style="magenta")
        
        for node in self.nodes.values():
            # Status formatting
            status_colors = {
                NodeStatus.IDLE: "green",
                NodeStatus.BUSY: "yellow",
                NodeStatus.OVERLOADED: "red",
                NodeStatus.FAILED: "red",
                NodeStatus.MAINTENANCE: "dim"
            }
            
            status_color = status_colors.get(node.status, "white")
            status_text = f"[{status_color}]{node.status.value.upper()}[/{status_color}]"
            
            # Resource usage
            cpu_text = f"{node.cpu_usage:.1f}%"
            memory_text = f"{node.memory_usage:.1f}%"
            
            # Task counts
            node_tasks = [t for t in self.tasks.values() if t.assigned_node == node.node_id]
            tasks_text = f"{node.tasks_completed}/{node.tasks_failed}"
            
            # Load factor
            load_text = f"{node.load_factor:.2f}"
            
            table.add_row(
                node.node_id[:12],
                status_text,
                cpu_text,
                memory_text,
                tasks_text,
                load_text
            )
        
        return table
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get comprehensive cluster summary."""
        
        task_summary = {}
        for status in TaskStatus:
            count = len([t for t in self.tasks.values() if t.status == status])
            if count > 0:
                task_summary[status.value] = count
        
        return {
            'cluster_name': self.cluster_name,
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'cluster_metrics': asdict(self.cluster_metrics),
            'task_summary': task_summary,
            'auto_scaling_config': {
                k: v for k, v in self.auto_scaling_config.items()
                if not isinstance(v, timedelta)
            },
            'performance_history_size': len(self.performance_history)
        }


# Example functions for distributed execution
def cpu_intensive_task(duration: int = 5) -> Dict[str, Any]:
    """CPU-intensive task for testing distributed execution."""
    import time
    import random
    
    start_time = time.time()
    
    # Simulate CPU-intensive computation
    result = 0
    for i in range(duration * 100000):
        result += random.random() ** 0.5
    
    end_time = time.time()
    
    return {
        'task_type': 'cpu_intensive',
        'duration': duration,
        'execution_time': end_time - start_time,
        'result': result,
        'iterations': duration * 100000
    }


def io_bound_task(delay: float = 2.0) -> Dict[str, Any]:
    """I/O-bound task for testing distributed execution."""
    import time
    import asyncio
    
    start_time = time.time()
    
    # Simulate I/O operation
    time.sleep(delay)
    
    end_time = time.time()
    
    return {
        'task_type': 'io_bound',
        'delay': delay,
        'execution_time': end_time - start_time,
        'status': 'completed'
    }


def memory_intensive_task(size_mb: int = 100) -> Dict[str, Any]:
    """Memory-intensive task for testing distributed execution."""
    import time
    
    start_time = time.time()
    
    # Allocate memory
    data = [0] * (size_mb * 1024 * 128)  # Approximately size_mb MB
    
    # Process data
    processed = sum(data)
    
    end_time = time.time()
    
    return {
        'task_type': 'memory_intensive',
        'size_mb': size_mb,
        'execution_time': end_time - start_time,
        'processed_sum': processed
    }


# Demo and testing functions
async def demo_distributed_computing():
    """Demonstrate distributed computing capabilities."""
    
    console = Console()
    console.print("[bold blue]🖥️ Distributed Computing Engine - Generation 3 Demo[/bold blue]")
    
    # Initialize distributed computing engine
    engine = DistributedComputingEngine(cluster_name="demo_cluster")
    
    # Add some additional nodes to simulate a cluster
    console.print("\n[yellow]🔧 Setting up distributed cluster...[/yellow]")
    
    engine.add_node("worker-01", "192.168.1.101", cpu_cores=8, memory_gb=16.0, gpu_count=1)
    engine.add_node("worker-02", "192.168.1.102", cpu_cores=4, memory_gb=8.0, gpu_count=0)
    engine.add_node("worker-03", "192.168.1.103", cpu_cores=6, memory_gb=12.0, gpu_count=2)
    
    # Start the cluster
    await engine.start_cluster()
    
    # Submit various types of tasks
    console.print("\n[cyan]📤 Submitting distributed tasks...[/cyan]")
    
    task_ids = []
    
    # CPU-intensive tasks
    for i in range(5):
        task_id = engine.submit_task(
            cpu_intensive_task,
            duration=3 + i,
            priority=TaskPriority.NORMAL,
            cpu_cores=2,
            memory_gb=1.0,
            estimated_duration=30.0
        )
        task_ids.append(task_id)
        console.print(f"  • Submitted CPU task {i+1}: {task_id[:12]}...")
    
    # I/O-bound tasks
    for i in range(3):
        task_id = engine.submit_task(
            io_bound_task,
            delay=1.0 + i * 0.5,
            priority=TaskPriority.HIGH,
            cpu_cores=1,
            memory_gb=0.5,
            estimated_duration=5.0
        )
        task_ids.append(task_id)
        console.print(f"  • Submitted I/O task {i+1}: {task_id[:12]}...")
    
    # Memory-intensive tasks
    for i in range(2):
        task_id = engine.submit_task(
            memory_intensive_task,
            size_mb=50 + i * 25,
            priority=TaskPriority.LOW,
            cpu_cores=1,
            memory_gb=2.0,
            estimated_duration=15.0
        )
        task_ids.append(task_id)
        console.print(f"  • Submitted memory task {i+1}: {task_id[:12]}...")
    
    # Create task dependency chain
    dependency_task_1 = engine.submit_task(
        cpu_intensive_task,
        duration=2,
        priority=TaskPriority.HIGH
    )
    
    dependency_task_2 = engine.submit_task(
        io_bound_task,
        delay=1.0,
        priority=TaskPriority.HIGH,
        dependencies=[dependency_task_1]
    )
    
    console.print(f"  • Submitted dependency chain: {dependency_task_1[:12]}... → {dependency_task_2[:12]}...")
    
    # Monitor cluster with live dashboard
    console.print(f"\n[bold yellow]📊 Monitoring cluster execution...[/bold yellow]")
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            monitoring_start = time.time()
            
            while time.time() - monitoring_start < 60:  # Monitor for 60 seconds
                # Update dashboard
                cluster_dashboard = engine.create_cluster_dashboard()
                
                # Create metrics panel
                metrics = engine.cluster_metrics
                metrics_text = f"""
[bold cyan]Cluster Metrics:[/bold cyan]
• Nodes: {metrics.healthy_nodes}/{metrics.total_nodes}
• CPU Utilization: {metrics.avg_cpu_utilization:.1f}%
• Memory Utilization: {metrics.avg_memory_utilization:.1f}%
• Tasks Pending: {metrics.tasks_pending}
• Tasks Running: {metrics.tasks_running}
• Tasks Completed: {metrics.tasks_completed}
• Tasks Failed: {metrics.tasks_failed}
• Cluster Efficiency: {metrics.cluster_efficiency:.1%}
• Scale Events: ↑{metrics.scale_up_triggers} ↓{metrics.scale_down_triggers}
"""
                
                metrics_panel = Panel(metrics_text.strip(), title="📈 Performance Metrics")
                
                # Combine dashboard and metrics
                layout = Panel(f"{cluster_dashboard}\n\n{metrics_panel}", title="🖥️ Distributed Computing Dashboard")
                
                live.update(layout)
                
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping cluster monitoring...[/yellow]")
    
    # Wait for remaining tasks to complete
    console.print(f"\n[cyan]⏳ Waiting for tasks to complete...[/cyan]")
    
    completed_count = 0
    for task_id in task_ids[:5]:  # Wait for first 5 tasks
        status, result = await engine.wait_for_task(task_id, timeout=30)
        if status == TaskStatus.COMPLETED:
            completed_count += 1
            console.print(f"  • Task {task_id[:12]}... completed: {type(result).__name__}")
        else:
            console.print(f"  • Task {task_id[:12]}... {status.value}: {result}")
    
    # Show final cluster summary
    console.print(f"\n[bold green]📋 Final Cluster Summary:[/bold green]")
    summary = engine.get_cluster_summary()
    
    console.print(f"• Cluster: {summary['cluster_name']}")
    console.print(f"• Total Tasks: {sum(summary['task_summary'].values())}")
    console.print(f"• Completed Tasks: {summary['task_summary'].get('completed', 0)}")
    console.print(f"• Failed Tasks: {summary['task_summary'].get('failed', 0)}")
    console.print(f"• Average Task Time: {summary['cluster_metrics']['avg_task_completion_time']:.2f}s")
    console.print(f"• Cluster Efficiency: {summary['cluster_metrics']['cluster_efficiency']:.1%}")
    
    # Stop the cluster
    await engine.stop_cluster()
    
    return engine


async def main():
    """Main entry point for distributed computing demo."""
    
    try:
        distributed_engine = await demo_distributed_computing()
        
        console = Console()
        console.print(f"\n[bold green]✅ Distributed computing demo completed successfully![/bold green]")
        
        # Show performance history
        if distributed_engine.performance_history:
            console.print(f"\n[bold cyan]📊 Performance History:[/bold cyan]")
            console.print(f"• History Entries: {len(distributed_engine.performance_history)}")
            
            last_snapshot = distributed_engine.performance_history[-1]
            console.print(f"• Final Cluster Efficiency: {last_snapshot['cluster_metrics']['cluster_efficiency']:.1%}")
            console.print(f"• Peak Node Count: {max(s['node_count'] for s in distributed_engine.performance_history)}")
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())