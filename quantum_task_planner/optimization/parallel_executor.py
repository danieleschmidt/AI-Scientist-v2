"""
Parallel Quantum Execution Engine

High-performance parallel execution system for quantum-inspired
task planning operations with auto-scaling and load balancing.
"""

import asyncio
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import queue
import logging
import numpy as np

from ..core.planner import Task, QuantumTaskPlanner
from ..core.quantum_optimizer import QuantumOptimizer
from ..monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Parallel execution modes."""
    THREAD_BASED = "thread_based"
    PROCESS_BASED = "process_based"
    ASYNC_BASED = "async_based"
    HYBRID = "hybrid"
    AUTO = "auto"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionTask:
    """Task for parallel execution."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: float = 1.0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ExecutionMetrics:
    """Parallel execution performance metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    peak_concurrency: int = 0
    current_concurrency: int = 0
    queue_size: int = 0
    throughput_per_second: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        return self.completed_tasks / max(self.total_tasks, 1)
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate (completed + failed)."""
        return (self.completed_tasks + self.failed_tasks) / max(self.total_tasks, 1)


class WorkerPool:
    """Adaptive worker pool with quantum-inspired scaling."""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = None,
                 execution_mode: ExecutionMode = ExecutionMode.AUTO):
        """
        Initialize worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (auto-detected if None)
            execution_mode: Execution mode for workers
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
        self.execution_mode = execution_mode
        
        self.current_workers = min_workers
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.active_futures: Dict[str, Future] = {}
        
        # Scaling metrics
        self.load_history = []
        self.scaling_cooldown = 30.0  # seconds
        self.last_scale_time = 0.0
        
        # Performance tracking
        self.task_completion_times = []
        self.quantum_efficiency = 1.0  # Quantum-inspired efficiency score
        
        logger.info(f"Initialized WorkerPool: {min_workers}-{max_workers} workers, {execution_mode.value}")
    
    def start(self) -> None:
        """Start the worker pool."""
        if self.executor is not None:
            logger.warning("Worker pool already started")
            return
        
        self._create_executor()
        logger.info(f"Started worker pool with {self.current_workers} workers")
    
    def stop(self) -> None:
        """Stop the worker pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            
        self.active_futures.clear()
        logger.info("Stopped worker pool")
    
    def _create_executor(self) -> None:
        """Create appropriate executor based on execution mode."""
        if self.execution_mode == ExecutionMode.THREAD_BASED:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        elif self.execution_mode == ExecutionMode.PROCESS_BASED:
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        elif self.execution_mode == ExecutionMode.AUTO:
            # Choose based on system characteristics
            if mp.cpu_count() >= 4:
                self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        else:
            # Default to thread-based
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
    
    def submit_task(self, task: ExecutionTask) -> Future:
        """Submit task for execution."""
        if not self.executor:
            raise RuntimeError("Worker pool not started")
        
        future = self.executor.submit(self._execute_task_with_monitoring, task)
        self.active_futures[task.id] = future
        
        return future
    
    def _execute_task_with_monitoring(self, task: ExecutionTask) -> Any:
        """Execute task with monitoring and error handling."""
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        
        try:
            # Execute the actual task
            result = task.function(*task.args, **task.kwargs)
            
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED
            task.result = result
            
            # Record completion time for scaling decisions
            if task.execution_time:
                self.task_completion_times.append(task.execution_time)
                if len(self.task_completion_times) > 100:
                    self.task_completion_times.pop(0)
            
            return result
            
        except Exception as e:
            task.completed_at = time.time()
            task.status = TaskStatus.FAILED
            task.error = e
            
            logger.error(f"Task {task.id} failed: {e}")
            raise
        
        finally:
            # Clean up future reference
            self.active_futures.pop(task.id, None)
    
    def scale_workers(self, target_workers: int) -> bool:
        """Scale worker pool to target size."""
        current_time = time.time()
        
        # Check scaling cooldown
        if current_time - self.last_scale_time < self.scaling_cooldown:
            return False
        
        # Validate target size
        target_workers = max(self.min_workers, min(target_workers, self.max_workers))
        
        if target_workers == self.current_workers:
            return False
        
        # Stop current executor
        if self.executor:
            self.executor.shutdown(wait=False)
        
        # Create new executor with target size
        self.current_workers = target_workers
        self._create_executor()
        
        self.last_scale_time = current_time
        logger.info(f"Scaled worker pool to {target_workers} workers")
        
        return True
    
    def get_load_metrics(self) -> Dict[str, float]:
        """Get current load metrics."""
        active_count = len(self.active_futures)
        utilization = active_count / self.current_workers
        
        # Calculate average completion time
        avg_completion_time = (
            np.mean(self.task_completion_times) 
            if self.task_completion_times 
            else 0.0
        )
        
        return {
            'active_tasks': active_count,
            'worker_count': self.current_workers,
            'utilization': utilization,
            'avg_completion_time': avg_completion_time,
            'quantum_efficiency': self.quantum_efficiency
        }
    
    def update_quantum_efficiency(self, success_rate: float, avg_latency: float) -> None:
        """Update quantum-inspired efficiency score."""
        # Quantum efficiency based on success rate and latency
        latency_factor = 1.0 / (1.0 + avg_latency / 10.0)  # Normalize around 10s
        self.quantum_efficiency = 0.7 * success_rate + 0.3 * latency_factor
        self.quantum_efficiency = max(0.1, min(1.0, self.quantum_efficiency))


class ParallelQuantumExecutor:
    """
    High-performance parallel execution engine for quantum operations.
    
    Provides auto-scaling, load balancing, and quantum-inspired optimization
    for parallel quantum task planning operations.
    """
    
    def __init__(self,
                 min_workers: int = 2,
                 max_workers: Optional[int] = None,
                 execution_mode: ExecutionMode = ExecutionMode.AUTO,
                 auto_scaling: bool = True):
        """
        Initialize parallel quantum executor.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            execution_mode: Execution mode
            auto_scaling: Enable automatic worker scaling
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
        self.execution_mode = execution_mode
        self.auto_scaling = auto_scaling
        
        # Worker pool management
        self.worker_pool = WorkerPool(min_workers, self.max_workers, execution_mode)
        
        # Task queue and management
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.task_registry: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionTask] = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics = ExecutionMetrics()
        
        # Execution control
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.scaling_thread: Optional[threading.Thread] = None
        
        # Quantum optimization
        self.quantum_optimizer = QuantumOptimizer(max_iterations=100)
        self.optimization_enabled = True
        
        logger.info(f"Initialized ParallelQuantumExecutor: {min_workers}-{self.max_workers} workers")
    
    def start(self) -> None:
        """Start the parallel executor."""
        if self.is_running:
            logger.warning("Parallel executor already running")
            return
        
        self.is_running = True
        
        # Start worker pool
        self.worker_pool.start()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start auto-scaling thread if enabled
        if self.auto_scaling:
            self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self.scaling_thread.start()
        
        logger.info("Started parallel quantum executor")
    
    def stop(self) -> None:
        """Stop the parallel executor."""
        self.is_running = False
        
        # Wait for threads to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5.0)
        
        # Stop worker pool
        self.worker_pool.stop()
        
        logger.info("Stopped parallel quantum executor")
    
    def submit_task(self, 
                   task_id: str,
                   function: Callable,
                   args: tuple = (),
                   kwargs: dict = None,
                   priority: float = 1.0,
                   timeout: Optional[float] = None,
                   max_retries: int = 3) -> str:
        """
        Submit task for parallel execution.
        
        Args:
            task_id: Unique task identifier
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (higher = more important)
            timeout: Execution timeout
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
        """
        task = ExecutionTask(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Add to registry and queue
        self.task_registry[task_id] = task
        
        # Priority queue uses negative priority for max-heap behavior
        self.task_queue.put((-priority, time.time(), task))
        
        # Update metrics
        self.metrics.total_tasks += 1
        self.metrics.queue_size = self.task_queue.qsize()
        
        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id
    
    def submit_planning_task(self, 
                           planner: QuantumTaskPlanner,
                           tasks: List[Task],
                           task_id: Optional[str] = None) -> str:
        """
        Submit quantum task planning operation.
        
        Args:
            planner: QuantumTaskPlanner instance
            tasks: List of tasks to plan
            task_id: Optional task identifier
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"planning_{int(time.time() * 1000)}"
        
        # Set tasks in planner
        planner.tasks = tasks
        
        return self.submit_task(
            task_id=task_id,
            function=planner.plan_tasks,
            priority=0.8,  # High priority for planning
            timeout=120.0   # 2 minute timeout
        )
    
    def submit_optimization_batch(self,
                                objective_functions: List[Callable],
                                initial_solutions: List[np.ndarray],
                                batch_id: Optional[str] = None) -> str:
        """
        Submit batch optimization tasks.
        
        Args:
            objective_functions: List of objective functions
            initial_solutions: List of initial solutions
            batch_id: Optional batch identifier
            
        Returns:
            Batch task ID
        """
        if batch_id is None:
            batch_id = f"optimization_batch_{int(time.time() * 1000)}"
        
        def batch_optimize():
            """Execute batch optimization."""
            results = []
            
            for i, (obj_func, initial_sol) in enumerate(zip(objective_functions, initial_solutions)):
                try:
                    optimizer = QuantumOptimizer(max_iterations=500)
                    result = optimizer.optimize(obj_func, initial_sol)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Optimization {i} failed: {e}")
                    results.append(None)
            
            return results
        
        return self.submit_task(
            task_id=batch_id,
            function=batch_optimize,
            priority=0.7,  # Medium-high priority
            timeout=300.0  # 5 minute timeout
        )
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of specific task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        elif task_id in self.task_registry:
            return self.task_registry[task_id].status
        else:
            return None
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result, waiting if necessary.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise task.error or Exception(f"Task {task_id} failed")
                elif task.status == TaskStatus.CANCELLED:
                    raise Exception(f"Task {task_id} was cancelled")
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Wait a bit before checking again
            time.sleep(0.1)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Check if task is in registry
        if task_id not in self.task_registry:
            return False
        
        task = self.task_registry[task_id]
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            # Move to completed tasks
            self.completed_tasks[task_id] = self.task_registry.pop(task_id)
            self.metrics.cancelled_tasks += 1
            
            logger.info(f"Cancelled pending task {task_id}")
            return True
        
        elif task.status == TaskStatus.RUNNING:
            # For running tasks, we can only mark as cancelled
            # The actual cancellation depends on the executor implementation
            task.status = TaskStatus.CANCELLED
            logger.warning(f"Requested cancellation of running task {task_id}")
            return True
        
        return False
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop for task execution."""
        while self.is_running:
            try:
                # Get next task from queue (with timeout)
                try:
                    _, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if task was cancelled
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Submit task to worker pool
                try:
                    future = self.worker_pool.submit_task(task)
                    self.metrics.current_concurrency += 1
                    self.metrics.peak_concurrency = max(
                        self.metrics.peak_concurrency, 
                        self.metrics.current_concurrency
                    )
                    
                    # Monitor task completion in background
                    threading.Thread(
                        target=self._monitor_task_completion,
                        args=(task, future),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    logger.error(f"Failed to submit task {task.id}: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = e
                    task.completed_at = time.time()
                    self._handle_task_completion(task)
                
                # Update queue size
                self.metrics.queue_size = self.task_queue.qsize()
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1.0)
    
    def _monitor_task_completion(self, task: ExecutionTask, future: Future) -> None:
        """Monitor task completion and handle results."""
        try:
            # Wait for task completion
            result = future.result(timeout=task.timeout)
            
            # Task completed successfully
            if task.status != TaskStatus.CANCELLED:
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.metrics.completed_tasks += 1
            
        except Exception as e:
            # Task failed
            if task.retry_count < task.max_retries and task.status != TaskStatus.CANCELLED:
                # Retry the task
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                
                # Re-submit to queue with slight delay
                threading.Timer(
                    delay=1.0 * task.retry_count,  # Exponential backoff
                    function=lambda: self.task_queue.put((-task.priority, time.time(), task))
                ).start()
                
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                return
            else:
                # Max retries reached or task cancelled
                task.status = TaskStatus.FAILED
                task.error = e
                self.metrics.failed_tasks += 1
                
                logger.error(f"Task {task.id} failed permanently: {e}")
        
        finally:
            self.metrics.current_concurrency -= 1
            self._handle_task_completion(task)
    
    def _handle_task_completion(self, task: ExecutionTask) -> None:
        """Handle task completion and update metrics."""
        # Move task to completed registry
        self.completed_tasks[task.id] = self.task_registry.pop(task.id, task)
        
        # Update execution time metrics
        if task.execution_time:
            total_time = self.metrics.total_execution_time + task.execution_time
            completed_count = self.metrics.completed_tasks + self.metrics.failed_tasks
            
            self.metrics.avg_execution_time = total_time / max(completed_count, 1)
            self.metrics.total_execution_time = total_time
        
        # Calculate throughput
        if self.metrics.total_execution_time > 0:
            self.metrics.throughput_per_second = (
                (self.metrics.completed_tasks + self.metrics.failed_tasks) / 
                self.metrics.total_execution_time
            )
        
        # Update worker pool quantum efficiency
        if self.metrics.total_tasks > 0:
            self.worker_pool.update_quantum_efficiency(
                self.metrics.success_rate,
                self.metrics.avg_execution_time
            )
        
        # Record performance metrics
        self.performance_monitor.record_metric(
            self.performance_monitor.PerformanceMetric(
                name="parallel_task_completion",
                value=1.0,
                metric_type=self.performance_monitor.MetricType.THROUGHPUT,
                timestamp=time.time(),
                unit="task",
                tags={
                    "status": task.status.value,
                    "execution_time": str(task.execution_time or 0)
                }
            )
        )
    
    def _scaling_loop(self) -> None:
        """Auto-scaling loop for dynamic worker management."""
        while self.is_running:
            try:
                time.sleep(10.0)  # Check every 10 seconds
                
                # Get current load metrics
                load_metrics = self.worker_pool.get_load_metrics()
                
                # Scaling decision based on quantum-inspired algorithm
                target_workers = self._calculate_optimal_workers(load_metrics)
                
                # Apply scaling if needed
                if target_workers != self.worker_pool.current_workers:
                    self.worker_pool.scale_workers(target_workers)
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
    
    def _calculate_optimal_workers(self, load_metrics: Dict[str, float]) -> int:
        """Calculate optimal number of workers using quantum-inspired algorithm."""
        current_workers = load_metrics['worker_count']
        utilization = load_metrics['utilization']
        quantum_efficiency = load_metrics['quantum_efficiency']
        queue_size = self.metrics.queue_size
        
        # Quantum-inspired scaling function
        # Considers utilization, efficiency, and queue size
        
        # Base scaling factor
        if utilization > 0.8:
            # High utilization - scale up
            scale_factor = 1.2 + (utilization - 0.8) * 2.0
        elif utilization < 0.3:
            # Low utilization - scale down
            scale_factor = 0.8 - (0.3 - utilization) * 1.0
        else:
            # Moderate utilization - maintain
            scale_factor = 1.0
        
        # Quantum efficiency adjustment
        efficiency_factor = 0.5 + quantum_efficiency * 0.5
        scale_factor *= efficiency_factor
        
        # Queue size pressure
        if queue_size > current_workers * 2:
            scale_factor *= 1.3  # Scale up for large queues
        elif queue_size == 0:
            scale_factor *= 0.9  # Scale down for empty queues
        
        # Calculate target workers
        target_workers = int(current_workers * scale_factor)
        
        # Apply constraints
        target_workers = max(self.min_workers, min(target_workers, self.max_workers))
        
        return target_workers
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics."""
        load_metrics = self.worker_pool.get_load_metrics()
        
        return {
            'execution_metrics': {
                'total_tasks': self.metrics.total_tasks,
                'completed_tasks': self.metrics.completed_tasks,
                'failed_tasks': self.metrics.failed_tasks,
                'cancelled_tasks': self.metrics.cancelled_tasks,
                'success_rate': self.metrics.success_rate,
                'completion_rate': self.metrics.completion_rate,
                'avg_execution_time': self.metrics.avg_execution_time,
                'throughput_per_second': self.metrics.throughput_per_second,
                'queue_size': self.metrics.queue_size
            },
            'load_metrics': load_metrics,
            'system_info': {
                'is_running': self.is_running,
                'execution_mode': self.execution_mode.value,
                'auto_scaling': self.auto_scaling,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers
            }
        }