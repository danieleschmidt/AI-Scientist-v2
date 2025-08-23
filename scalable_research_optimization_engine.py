#!/usr/bin/env python3
"""
Scalable Research Optimization Engine - Generation 3 Enhancement
================================================================

A comprehensive enterprise-grade system for scaling AI research automation
with performance optimization, distributed computing, and intelligent resource management.

Features:
- Multi-level intelligent caching system
- Parallel and distributed processing
- Dynamic resource pooling and auto-scaling
- Advanced performance monitoring and optimization
- Predictive resource allocation
- Cost optimization and efficiency analysis
"""

import asyncio
import threading
import multiprocessing
import time
import json
import logging
import psutil
import hashlib
import pickle
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
from queue import Queue, PriorityQueue
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import warnings
import argparse
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ResourceMetrics:
    """System resource usage metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TaskMetrics:
    """Individual task performance metrics"""
    task_id: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    resources_used: Optional[ResourceMetrics] = None
    cache_hit_rate: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def complete(self, success: bool = True, error: str = None):
        """Mark task as completed"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: int  # 1-5, 5 being highest
    description: str
    action: str
    estimated_impact: str
    implementation_cost: str

class IntelligentCache:
    """Multi-level intelligent caching system with predictive prefetching"""
    
    def __init__(self, max_memory_cache_size: int = 1000, 
                 disk_cache_dir: str = "./cache",
                 ttl_seconds: int = 3600):
        self.max_memory_cache_size = max_memory_cache_size
        self.disk_cache_dir = Path(disk_cache_dir)
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_seconds
        
        # Multi-level cache
        self.l1_cache = {}  # Memory cache (fastest)
        self.l2_cache = {}  # Compressed memory cache
        self.cache_access_times = {}
        self.cache_hit_stats = defaultdict(int)
        self.access_patterns = defaultdict(list)
        
        # SQLite for cache metadata
        self.db_path = self.disk_cache_dir / "cache_metadata.db"
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize cache metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    size INTEGER,
                    access_count INTEGER,
                    last_accessed TIMESTAMP,
                    expires_at TIMESTAMP,
                    cache_level INTEGER
                )
            """)
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key"""
        key_data = f"{func_name}_{args}_{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache with intelligent level selection"""
        now = datetime.now()
        
        # Check L1 (memory) cache first
        if key in self.l1_cache:
            entry_time = self.cache_access_times.get(key, now)
            if not self._is_expired(entry_time):
                self.cache_hit_stats["l1"] += 1
                self.access_patterns[key].append(now)
                return self.l1_cache[key]
            else:
                del self.l1_cache[key]
                del self.cache_access_times[key]
        
        # Check L2 (compressed memory) cache
        if key in self.l2_cache:
            entry_time = self.cache_access_times.get(key, now)
            if not self._is_expired(entry_time):
                self.cache_hit_stats["l2"] += 1
                # Promote to L1 if frequently accessed
                if len(self.access_patterns[key]) > 5:
                    self._promote_to_l1(key, self.l2_cache[key])
                return pickle.loads(self.l2_cache[key])
        
        # Check disk cache
        disk_path = self.disk_cache_dir / f"{key}.cache"
        if disk_path.exists():
            try:
                with open(disk_path, 'rb') as f:
                    data = pickle.load(f)
                self.cache_hit_stats["disk"] += 1
                # Promote to memory if recently accessed
                self._promote_to_memory(key, data)
                return data
            except (pickle.PickleError, IOError):
                disk_path.unlink(missing_ok=True)
        
        self.cache_hit_stats["miss"] += 1
        return None
    
    def put(self, key: str, value: Any, level: str = "auto"):
        """Store in cache with intelligent level selection"""
        now = datetime.now()
        self.cache_access_times[key] = now
        
        # Calculate value size
        try:
            value_size = len(pickle.dumps(value))
        except:
            value_size = sys.getsizeof(value)
        
        # Intelligent level selection
        if level == "auto":
            if value_size < 1024 * 10:  # < 10KB
                level = "l1"
            elif value_size < 1024 * 100:  # < 100KB
                level = "l2"
            else:
                level = "disk"
        
        if level == "l1" and len(self.l1_cache) < self.max_memory_cache_size:
            self.l1_cache[key] = value
        elif level == "l2":
            self.l2_cache[key] = pickle.dumps(value)
        else:
            # Store to disk
            disk_path = self.disk_cache_dir / f"{key}.cache"
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(value, f)
            except (pickle.PickleError, IOError) as e:
                logging.warning(f"Failed to cache to disk: {e}")
        
        # Update metadata
        self._update_cache_metadata(key, value_size, level)
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promote frequently accessed item to L1 cache"""
        if len(self.l1_cache) >= self.max_memory_cache_size:
            # Evict LRU item
            lru_key = min(self.cache_access_times.keys(), 
                         key=lambda k: self.cache_access_times[k])
            del self.l1_cache[lru_key]
        
        self.l1_cache[key] = value
        if key in self.l2_cache:
            del self.l2_cache[key]
    
    def _promote_to_memory(self, key: str, value: Any):
        """Promote disk cache item to memory"""
        if len(self.l2_cache) < self.max_memory_cache_size * 2:
            self.l2_cache[key] = pickle.dumps(value)
    
    def _update_cache_metadata(self, key: str, size: int, level: str):
        """Update cache metadata in database"""
        level_map = {"l1": 1, "l2": 2, "disk": 3}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, size, access_count, last_accessed, expires_at, cache_level)
                VALUES (?, ?, 1, ?, ?, ?)
            """, (key, size, datetime.now(), 
                  datetime.now() + timedelta(seconds=self.ttl_seconds),
                  level_map.get(level, 3)))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_hits = sum(self.cache_hit_stats.values()) - self.cache_hit_stats["miss"]
        total_requests = sum(self.cache_hit_stats.values())
        hit_rate = total_hits / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "l1_hits": self.cache_hit_stats["l1"],
            "l2_hits": self.cache_hit_stats["l2"],
            "disk_hits": self.cache_hit_stats["disk"],
            "misses": self.cache_hit_stats["miss"],
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "total_requests": total_requests
        }

def cached_execution(cache_instance: IntelligentCache, ttl: int = 3600):
    """Decorator for intelligent caching of function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_cache_key(
                func.__name__, args, kwargs
            )
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            return result
        return wrapper
    return decorator

class ResourcePoolManager:
    """Advanced resource pooling and management system"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Resource tracking
        self.active_tasks = {}
        self.resource_usage_history = deque(maxlen=1000)
        self.resource_locks = defaultdict(threading.Lock)
        
        # GPU pool (if available)
        self.gpu_available = self._check_gpu_availability()
        self.gpu_queue = Queue() if self.gpu_available else None
        
        # Dynamic scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.last_scale_time = datetime.now()
        self.min_scale_interval = timedelta(minutes=5)
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU resources are available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.experimental.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    def get_current_resource_usage(self) -> ResourceMetrics:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        gpu_usage = None
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization()
            except:
                pass
        
        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            },
            gpu_usage=gpu_usage
        )
        
        self.resource_usage_history.append(metrics)
        return metrics
    
    def should_scale_up(self) -> bool:
        """Determine if resources should be scaled up"""
        if len(self.resource_usage_history) < 10:
            return False
        
        recent_usage = list(self.resource_usage_history)[-10:]
        avg_cpu = np.mean([m.cpu_percent for m in recent_usage])
        avg_memory = np.mean([m.memory_percent for m in recent_usage])
        
        return (avg_cpu > self.scale_up_threshold * 100 or 
                avg_memory > self.scale_up_threshold * 100)
    
    def should_scale_down(self) -> bool:
        """Determine if resources should be scaled down"""
        if len(self.resource_usage_history) < 20:
            return False
        
        recent_usage = list(self.resource_usage_history)[-20:]
        avg_cpu = np.mean([m.cpu_percent for m in recent_usage])
        avg_memory = np.mean([m.memory_percent for m in recent_usage])
        
        return (avg_cpu < self.scale_down_threshold * 100 and 
                avg_memory < self.scale_down_threshold * 100)
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU-intensive task to thread pool"""
        future = self.thread_pool.submit(func, *args, **kwargs)
        task_id = id(future)
        self.active_tasks[task_id] = {
            'type': 'cpu',
            'future': future,
            'start_time': datetime.now()
        }
        return future
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O-intensive task to thread pool"""
        future = self.thread_pool.submit(func, *args, **kwargs)
        task_id = id(future)
        self.active_tasks[task_id] = {
            'type': 'io',
            'future': future,
            'start_time': datetime.now()
        }
        return future
    
    def submit_compute_task(self, func: Callable, *args, **kwargs):
        """Submit compute-intensive task to process pool"""
        future = self.process_pool.submit(func, *args, **kwargs)
        task_id = id(future)
        self.active_tasks[task_id] = {
            'type': 'compute',
            'future': future,
            'start_time': datetime.now()
        }
        return future
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        active_count = len([t for t in self.active_tasks.values() 
                           if not t['future'].done()])
        
        current_resources = self.get_current_resource_usage()
        
        return {
            'active_tasks': active_count,
            'thread_pool_size': self.thread_pool._max_workers,
            'process_pool_size': self.process_pool._max_workers,
            'current_cpu': current_resources.cpu_percent,
            'current_memory': current_resources.memory_percent,
            'gpu_available': self.gpu_available,
            'gpu_usage': current_resources.gpu_usage
        }

class DistributedTaskScheduler:
    """Advanced task scheduling with priority queues and load balancing"""
    
    def __init__(self, resource_manager: ResourcePoolManager):
        self.resource_manager = resource_manager
        self.task_queue = PriorityQueue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_metrics = {}
        self.running = False
        self.scheduler_thread = None
        
        # Priority levels
        self.PRIORITY_CRITICAL = 1
        self.PRIORITY_HIGH = 2
        self.PRIORITY_NORMAL = 3
        self.PRIORITY_LOW = 4
        self.PRIORITY_BATCH = 5
        
        # Load balancing
        self.node_loads = defaultdict(float)
        self.load_balance_strategy = "round_robin"  # or "least_loaded"
    
    def submit_task(self, task_func: Callable, priority: int = None, 
                   task_type: str = "cpu", dependencies: List[str] = None,
                   timeout: int = None, **kwargs) -> str:
        """Submit task with priority and dependencies"""
        if priority is None:
            priority = self.PRIORITY_NORMAL
        
        task_id = f"task_{len(self.task_metrics)}_{int(time.time())}"
        
        task_info = {
            'id': task_id,
            'func': task_func,
            'type': task_type,
            'dependencies': dependencies or [],
            'timeout': timeout,
            'kwargs': kwargs,
            'submitted_at': datetime.now()
        }
        
        # Create task metrics
        self.task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            task_type=task_type,
            start_time=datetime.now()
        )
        
        # Add to priority queue
        self.task_queue.put((priority, time.time(), task_info))
        
        return task_id
    
    def start_scheduler(self):
        """Start the task scheduler"""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Get next task
                if not self.task_queue.empty():
                    priority, timestamp, task_info = self.task_queue.get_nowait()
                    
                    # Check dependencies
                    if self._dependencies_satisfied(task_info['dependencies']):
                        self._execute_task(task_info)
                    else:
                        # Re-queue with slight delay
                        self.task_queue.put((priority, time.time() + 1, task_info))
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def _dependencies_satisfied(self, dependencies: List[str]) -> bool:
        """Check if task dependencies are satisfied"""
        if not dependencies:
            return True
        
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep_id].get('success', False):
                return False
        
        return True
    
    def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a task using appropriate resource pool"""
        task_id = task_info['id']
        task_type = task_info['type']
        
        # Update task metrics
        if task_id in self.task_metrics:
            self.task_metrics[task_id].start_time = datetime.now()
        
        try:
            # Choose execution method based on task type
            if task_type == "cpu":
                future = self.resource_manager.submit_cpu_task(
                    task_info['func'], **task_info['kwargs']
                )
            elif task_type == "io":
                future = self.resource_manager.submit_io_task(
                    task_info['func'], **task_info['kwargs']
                )
            elif task_type == "compute":
                future = self.resource_manager.submit_compute_task(
                    task_info['func'], **task_info['kwargs']
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Handle task completion
            def on_complete(fut):
                try:
                    result = fut.result(timeout=task_info.get('timeout'))
                    self.completed_tasks[task_id] = {
                        'result': result,
                        'success': True,
                        'completed_at': datetime.now()
                    }
                    
                    # Update metrics
                    if task_id in self.task_metrics:
                        self.task_metrics[task_id].complete(success=True)
                        
                except Exception as e:
                    self.failed_tasks[task_id] = {
                        'error': str(e),
                        'failed_at': datetime.now()
                    }
                    
                    # Update metrics
                    if task_id in self.task_metrics:
                        self.task_metrics[task_id].complete(success=False, error=str(e))
            
            future.add_done_callback(on_complete)
            
        except Exception as e:
            self.failed_tasks[task_id] = {
                'error': str(e),
                'failed_at': datetime.now()
            }
            
            if task_id in self.task_metrics:
                self.task_metrics[task_id].complete(success=False, error=str(e))
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        total_tasks = len(self.task_metrics)
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        pending_count = self.task_queue.qsize()
        
        # Calculate success rate
        success_rate = completed_count / max(total_tasks, 1)
        
        # Calculate average execution time
        completed_metrics = [m for m in self.task_metrics.values() 
                           if m.end_time is not None and m.success]
        avg_execution_time = np.mean([m.duration for m in completed_metrics]) if completed_metrics else 0
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'pending_tasks': pending_count,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'scheduler_running': self.running
        }

class PerformanceMonitor:
    """Real-time performance monitoring and optimization recommendations"""
    
    def __init__(self, cache: IntelligentCache, 
                 resource_manager: ResourcePoolManager,
                 scheduler: DistributedTaskScheduler):
        self.cache = cache
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        
        # Monitoring data
        self.performance_history = deque(maxlen=1000)
        self.bottleneck_detectors = []
        self.optimization_suggestions = []
        
        # Benchmark data
        self.benchmark_results = {}
        self.baseline_performance = None
        
        # ML-based optimization (simplified)
        self.performance_patterns = defaultdict(list)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now()
        
        # Resource metrics
        resource_metrics = self.resource_manager.get_current_resource_usage()
        
        # Cache metrics
        cache_stats = self.cache.get_cache_stats()
        
        # Scheduler metrics
        scheduler_stats = self.scheduler.get_scheduler_stats()
        
        # Pool metrics
        pool_stats = self.resource_manager.get_pool_stats()
        
        metrics = {
            'timestamp': timestamp,
            'resources': asdict(resource_metrics),
            'cache': cache_stats,
            'scheduler': scheduler_stats,
            'pools': pool_stats
        }
        
        self.performance_history.append(metrics)
        return metrics
    
    def detect_bottlenecks(self) -> List[str]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        if len(self.performance_history) < 10:
            return bottlenecks
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # CPU bottleneck
        avg_cpu = np.mean([m['resources']['cpu_percent'] for m in recent_metrics])
        if avg_cpu > 90:
            bottlenecks.append("High CPU usage detected")
        
        # Memory bottleneck
        avg_memory = np.mean([m['resources']['memory_percent'] for m in recent_metrics])
        if avg_memory > 85:
            bottlenecks.append("High memory usage detected")
        
        # Cache performance
        cache_hit_rate = recent_metrics[-1]['cache']['hit_rate']
        if cache_hit_rate < 0.5:
            bottlenecks.append("Low cache hit rate")
        
        # Task queue buildup
        pending_tasks = recent_metrics[-1]['scheduler']['pending_tasks']
        if pending_tasks > 50:
            bottlenecks.append("Task queue buildup detected")
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate AI-driven optimization recommendations"""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        latest_metrics = self.performance_history[-1]
        bottlenecks = self.detect_bottlenecks()
        
        # Cache optimization
        cache_stats = latest_metrics['cache']
        if cache_stats['hit_rate'] < 0.6:
            recommendations.append(OptimizationRecommendation(
                category="Caching",
                priority=4,
                description="Low cache hit rate detected",
                action="Increase cache size or adjust TTL settings",
                estimated_impact="20-40% performance improvement",
                implementation_cost="Low"
            ))
        
        # Resource scaling
        cpu_usage = latest_metrics['resources']['cpu_percent']
        if cpu_usage > 80:
            recommendations.append(OptimizationRecommendation(
                category="Scaling",
                priority=5,
                description="High CPU utilization",
                action="Scale up thread/process pools or add more nodes",
                estimated_impact="30-50% performance improvement",
                implementation_cost="Medium"
            ))
        
        # Memory optimization
        memory_usage = latest_metrics['resources']['memory_percent']
        if memory_usage > 85:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority=5,
                description="High memory usage",
                action="Implement memory-efficient algorithms or increase system RAM",
                estimated_impact="25-35% performance improvement",
                implementation_cost="Medium-High"
            ))
        
        # Task scheduling optimization
        pending_tasks = latest_metrics['scheduler']['pending_tasks']
        if pending_tasks > 20:
            recommendations.append(OptimizationRecommendation(
                category="Scheduling",
                priority=3,
                description="Task queue buildup",
                action="Optimize task priorities or increase worker pools",
                estimated_impact="15-25% performance improvement",
                implementation_cost="Low-Medium"
            ))
        
        return recommendations
    
    def run_performance_benchmark(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        print(f"Running performance benchmark for {duration_seconds} seconds...")
        
        start_time = datetime.now()
        benchmark_results = {
            'start_time': start_time,
            'duration': duration_seconds,
            'metrics': []
        }
        
        # Collect metrics during benchmark
        end_time = start_time + timedelta(seconds=duration_seconds)
        while datetime.now() < end_time:
            metrics = self.collect_metrics()
            benchmark_results['metrics'].append(metrics)
            time.sleep(1)
        
        # Analyze results
        metrics_list = benchmark_results['metrics']
        if metrics_list:
            # CPU statistics
            cpu_values = [m['resources']['cpu_percent'] for m in metrics_list]
            benchmark_results['cpu_stats'] = {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': np.mean(cpu_values),
                'std': np.std(cpu_values)
            }
            
            # Memory statistics
            memory_values = [m['resources']['memory_percent'] for m in metrics_list]
            benchmark_results['memory_stats'] = {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': np.mean(memory_values),
                'std': np.std(memory_values)
            }
            
            # Cache statistics
            final_cache_stats = metrics_list[-1]['cache']
            benchmark_results['cache_performance'] = final_cache_stats
            
            # Task throughput
            task_counts = [m['scheduler']['completed_tasks'] for m in metrics_list]
            if len(task_counts) > 1:
                throughput = (task_counts[-1] - task_counts[0]) / duration_seconds
                benchmark_results['task_throughput'] = throughput
        
        self.benchmark_results[start_time.isoformat()] = benchmark_results
        return benchmark_results

class ScalableResearchOptimizationEngine:
    """Main optimization engine orchestrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize core components
        self.cache = IntelligentCache(
            max_memory_cache_size=self.config.get('cache_size', 1000),
            disk_cache_dir=self.config.get('cache_dir', './cache'),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        
        self.resource_manager = ResourcePoolManager(
            max_workers=self.config.get('max_workers', multiprocessing.cpu_count())
        )
        
        self.scheduler = DistributedTaskScheduler(self.resource_manager)
        
        self.monitor = PerformanceMonitor(
            self.cache, self.resource_manager, self.scheduler
        )
        
        # Auto-optimization settings
        self.auto_optimize = self.config.get('auto_optimize', True)
        self.optimization_interval = self.config.get('optimization_interval', 300)  # 5 minutes
        self.last_optimization = datetime.now()
        
        # Performance tracking
        self.optimization_history = []
        self.cost_tracking = {
            'compute_costs': 0.0,
            'storage_costs': 0.0,
            'network_costs': 0.0
        }
        
        # Start scheduler
        self.scheduler.start_scheduler()
        
        print("Scalable Research Optimization Engine initialized successfully!")
    
    @cached_execution(IntelligentCache(), ttl=3600)
    def optimize_research_task(self, research_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize and execute research task with full scaling capabilities"""
        task_id = f"research_{int(time.time())}"
        
        print(f"Optimizing research task: {task_id}")
        
        # Analyze task requirements
        task_complexity = self._analyze_task_complexity(research_config)
        optimization_strategy = self._select_optimization_strategy(task_complexity)
        
        # Apply optimizations
        optimized_config = self._apply_optimizations(research_config, optimization_strategy)
        
        # Execute with monitoring
        execution_results = self._execute_optimized_task(optimized_config)
        
        # Post-execution optimization
        post_optimization = self._post_execution_optimization(execution_results)
        
        return {
            'task_id': task_id,
            'original_config': research_config,
            'optimized_config': optimized_config,
            'optimization_strategy': optimization_strategy,
            'execution_results': execution_results,
            'post_optimization': post_optimization,
            'performance_metrics': self.monitor.collect_metrics()
        }
    
    def _analyze_task_complexity(self, config: Dict[str, Any]) -> str:
        """Analyze task complexity for optimization strategy selection"""
        factors = {
            'data_size': config.get('data_size', 0),
            'compute_requirements': config.get('compute_requirements', 'medium'),
            'parallel_potential': config.get('parallel_potential', 'medium'),
            'memory_intensive': config.get('memory_intensive', False),
            'io_intensive': config.get('io_intensive', False)
        }
        
        # Simple complexity scoring
        score = 0
        if factors['data_size'] > 1000000:  # > 1M records
            score += 3
        if factors['compute_requirements'] == 'high':
            score += 2
        if factors['parallel_potential'] == 'high':
            score += 2
        if factors['memory_intensive']:
            score += 2
        if factors['io_intensive']:
            score += 1
        
        if score >= 6:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _select_optimization_strategy(self, complexity: str) -> Dict[str, Any]:
        """Select optimization strategy based on task complexity"""
        strategies = {
            'low': {
                'use_cache': True,
                'parallel_execution': False,
                'resource_priority': 'low',
                'optimization_level': 1
            },
            'medium': {
                'use_cache': True,
                'parallel_execution': True,
                'resource_priority': 'medium',
                'optimization_level': 2,
                'enable_load_balancing': True
            },
            'high': {
                'use_cache': True,
                'parallel_execution': True,
                'resource_priority': 'high',
                'optimization_level': 3,
                'enable_load_balancing': True,
                'enable_distributed_execution': True,
                'enable_gpu_acceleration': self.resource_manager.gpu_available
            }
        }
        
        return strategies.get(complexity, strategies['medium'])
    
    def _apply_optimizations(self, config: Dict[str, Any], 
                           strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Apply selected optimizations to task configuration"""
        optimized = config.copy()
        
        # Apply caching strategy
        if strategy.get('use_cache'):
            optimized['cache_enabled'] = True
            optimized['cache_level'] = strategy.get('optimization_level', 2)
        
        # Apply parallel execution
        if strategy.get('parallel_execution'):
            optimized['parallel_workers'] = min(
                self.resource_manager.max_workers,
                config.get('max_parallel_tasks', 4)
            )
        
        # Apply resource prioritization
        optimized['resource_priority'] = strategy.get('resource_priority', 'medium')
        
        # Apply load balancing
        if strategy.get('enable_load_balancing'):
            optimized['load_balancing'] = True
        
        # Apply GPU acceleration
        if strategy.get('enable_gpu_acceleration'):
            optimized['gpu_acceleration'] = True
        
        return optimized
    
    def _execute_optimized_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized research task"""
        start_time = datetime.now()
        
        # Simulate research task execution with different components
        results = {
            'start_time': start_time,
            'config': config,
            'subtasks': []
        }
        
        # Example subtasks based on configuration
        subtasks = self._generate_subtasks(config)
        
        # Execute subtasks based on optimization strategy
        if config.get('parallel_workers', 1) > 1:
            # Parallel execution
            futures = []
            for subtask in subtasks:
                future = self.scheduler.submit_task(
                    self._execute_subtask,
                    priority=self._get_task_priority(config.get('resource_priority', 'medium')),
                    task_type='compute' if config.get('gpu_acceleration') else 'cpu',
                    subtask=subtask,
                    config=config
                )
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results['subtasks'].append(result)
                except Exception as e:
                    results['subtasks'].append({'error': str(e)})
        else:
            # Sequential execution
            for subtask in subtasks:
                try:
                    result = self._execute_subtask(subtask, config)
                    results['subtasks'].append(result)
                except Exception as e:
                    results['subtasks'].append({'error': str(e)})
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - start_time).total_seconds()
        
        return results
    
    def _generate_subtasks(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate subtasks based on configuration"""
        subtasks = []
        
        # Data processing subtasks
        if config.get('data_size', 0) > 0:
            subtasks.append({
                'type': 'data_processing',
                'description': 'Process and clean research data',
                'estimated_duration': 30
            })
        
        # Analysis subtasks
        subtasks.append({
            'type': 'analysis',
            'description': 'Perform statistical analysis',
            'estimated_duration': 60
        })
        
        # Model training (if applicable)
        if config.get('machine_learning', False):
            subtasks.append({
                'type': 'model_training',
                'description': 'Train machine learning models',
                'estimated_duration': 120
            })
        
        # Report generation
        subtasks.append({
            'type': 'report_generation',
            'description': 'Generate research report',
            'estimated_duration': 45
        })
        
        return subtasks
    
    def _execute_subtask(self, subtask: Dict[str, Any], 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual subtask"""
        start_time = datetime.now()
        
        # Simulate task execution
        duration = subtask.get('estimated_duration', 30)
        
        # Apply cache speedup
        if config.get('cache_enabled'):
            cache_speedup = 0.3  # 30% speedup from caching
            duration = duration * (1 - cache_speedup)
        
        # Apply GPU speedup
        if config.get('gpu_acceleration') and subtask['type'] in ['analysis', 'model_training']:
            gpu_speedup = 0.5  # 50% speedup from GPU
            duration = duration * (1 - gpu_speedup)
        
        # Simulate work
        time.sleep(min(duration / 100, 2))  # Scaled down for demo
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        return {
            'subtask': subtask,
            'start_time': start_time,
            'end_time': end_time,
            'duration': actual_duration,
            'estimated_duration': duration,
            'success': True
        }
    
    def _get_task_priority(self, priority_level: str) -> int:
        """Convert priority level to numeric priority"""
        priority_map = {
            'low': 4,
            'medium': 3,
            'high': 2,
            'critical': 1
        }
        return priority_map.get(priority_level, 3)
    
    def _post_execution_optimization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform post-execution optimization analysis"""
        optimization = {
            'recommendations': self.monitor.generate_optimization_recommendations(),
            'performance_analysis': self._analyze_performance(results),
            'cost_analysis': self._analyze_costs(results),
            'scaling_recommendations': self._generate_scaling_recommendations(results)
        }
        
        return optimization
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task performance"""
        total_duration = results.get('duration', 0)
        subtask_count = len(results.get('subtasks', []))
        success_count = len([st for st in results.get('subtasks', []) 
                           if st.get('success', False)])
        
        return {
            'total_duration': total_duration,
            'subtask_count': subtask_count,
            'success_rate': success_count / max(subtask_count, 1),
            'average_subtask_duration': total_duration / max(subtask_count, 1),
            'performance_score': self._calculate_performance_score(results)
        }
    
    def _analyze_costs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task costs"""
        duration = results.get('duration', 0)
        
        # Simplified cost model
        compute_cost = duration * 0.01  # $0.01 per second
        storage_cost = 0.05  # Fixed storage cost
        network_cost = 0.02  # Fixed network cost
        
        total_cost = compute_cost + storage_cost + network_cost
        
        return {
            'compute_cost': compute_cost,
            'storage_cost': storage_cost,
            'network_cost': network_cost,
            'total_cost': total_cost,
            'cost_per_subtask': total_cost / max(len(results.get('subtasks', [])), 1)
        }
    
    def _generate_scaling_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate scaling recommendations based on results"""
        recommendations = []
        
        duration = results.get('duration', 0)
        success_rate = len([st for st in results.get('subtasks', []) 
                          if st.get('success', False)]) / max(len(results.get('subtasks', [])), 1)
        
        if duration > 300:  # > 5 minutes
            recommendations.append("Consider increasing parallel workers for faster execution")
        
        if success_rate < 0.8:
            recommendations.append("Improve error handling and retry mechanisms")
        
        current_metrics = self.monitor.collect_metrics()
        if current_metrics['resources']['cpu_percent'] > 80:
            recommendations.append("Scale up CPU resources or distribute load")
        
        if current_metrics['resources']['memory_percent'] > 85:
            recommendations.append("Increase memory allocation or optimize memory usage")
        
        return recommendations
    
    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        # Base score
        score = 50.0
        
        # Success rate impact
        subtasks = results.get('subtasks', [])
        if subtasks:
            success_rate = len([st for st in subtasks if st.get('success', False)]) / len(subtasks)
            score += (success_rate - 0.5) * 40  # +/- 20 points for success rate
        
        # Duration impact (assuming 180 seconds is optimal)
        duration = results.get('duration', 180)
        optimal_duration = 180
        if duration <= optimal_duration:
            score += (optimal_duration - duration) / optimal_duration * 20
        else:
            score -= (duration - optimal_duration) / optimal_duration * 20
        
        # Cache performance impact
        current_metrics = self.monitor.collect_metrics()
        cache_hit_rate = current_metrics.get('cache', {}).get('hit_rate', 0)
        score += cache_hit_rate * 10  # Up to 10 points for cache performance
        
        return max(0, min(100, score))
    
    def run_comprehensive_benchmark(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Run comprehensive system benchmark"""
        print(f"Running comprehensive benchmark for {duration_minutes} minutes...")
        
        # Run performance benchmark
        perf_benchmark = self.monitor.run_performance_benchmark(duration_minutes * 60)
        
        # Run load test
        load_test_results = self._run_load_test(duration_minutes)
        
        # Analyze optimization effectiveness
        optimization_analysis = self._analyze_optimization_effectiveness()
        
        benchmark_summary = {
            'timestamp': datetime.now(),
            'duration_minutes': duration_minutes,
            'performance_benchmark': perf_benchmark,
            'load_test': load_test_results,
            'optimization_analysis': optimization_analysis,
            'recommendations': self.monitor.generate_optimization_recommendations()
        }
        
        return benchmark_summary
    
    def _run_load_test(self, duration_minutes: int) -> Dict[str, Any]:
        """Run load test to assess scaling capabilities"""
        print("Running load test...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Submit increasing number of tasks
        task_results = []
        task_count = 0
        
        while datetime.now() < end_time:
            # Increase load over time
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            target_concurrent_tasks = min(20, int(elapsed_minutes * 2) + 1)
            
            # Submit tasks to reach target
            while task_count < target_concurrent_tasks:
                task_id = self.scheduler.submit_task(
                    self._benchmark_task,
                    priority=self.scheduler.PRIORITY_NORMAL,
                    task_type='cpu',
                    task_size='medium'
                )
                task_results.append(task_id)
                task_count += 1
            
            time.sleep(10)  # Check every 10 seconds
        
        # Wait for tasks to complete
        time.sleep(30)
        
        # Analyze results
        scheduler_stats = self.scheduler.get_scheduler_stats()
        
        return {
            'total_tasks_submitted': len(task_results),
            'completed_tasks': scheduler_stats['completed_tasks'],
            'failed_tasks': scheduler_stats['failed_tasks'],
            'success_rate': scheduler_stats['success_rate'],
            'average_execution_time': scheduler_stats['average_execution_time']
        }
    
    def _benchmark_task(self, task_size: str = 'medium') -> Dict[str, Any]:
        """Benchmark task for load testing"""
        start_time = datetime.now()
        
        # Simulate work based on task size
        work_duration = {
            'small': 1,
            'medium': 3,
            'large': 5
        }.get(task_size, 3)
        
        # Simulate CPU work
        end = time.time() + work_duration
        count = 0
        while time.time() < end:
            count += 1
            if count % 10000 == 0:
                time.sleep(0.001)  # Small sleep to prevent 100% CPU
        
        end_time = datetime.now()
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': (end_time - start_time).total_seconds(),
            'task_size': task_size,
            'work_count': count
        }
    
    def _analyze_optimization_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of optimization strategies"""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        # Compare performance before and after optimizations
        recent_optimizations = self.optimization_history[-10:] if len(self.optimization_history) > 10 else self.optimization_history
        
        performance_improvements = []
        for opt in recent_optimizations:
            if 'before_performance' in opt and 'after_performance' in opt:
                improvement = opt['after_performance'] - opt['before_performance']
                performance_improvements.append(improvement)
        
        if performance_improvements:
            avg_improvement = np.mean(performance_improvements)
            return {
                'average_improvement': avg_improvement,
                'optimization_count': len(recent_optimizations),
                'total_optimizations': len(self.optimization_history),
                'effectiveness_score': min(100, avg_improvement * 10)
            }
        else:
            return {'message': 'Insufficient data for optimization analysis'}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics"""
        current_metrics = self.monitor.collect_metrics()
        cache_stats = self.cache.get_cache_stats()
        scheduler_stats = self.scheduler.get_scheduler_stats()
        pool_stats = self.resource_manager.get_pool_stats()
        
        return {
            'timestamp': datetime.now(),
            'system_status': 'running' if self.scheduler.running else 'stopped',
            'performance_metrics': current_metrics,
            'cache_performance': cache_stats,
            'scheduler_performance': scheduler_stats,
            'resource_utilization': pool_stats,
            'bottlenecks': self.monitor.detect_bottlenecks(),
            'recommendations': self.monitor.generate_optimization_recommendations(),
            'optimization_history_count': len(self.optimization_history)
        }
    
    def shutdown(self):
        """Gracefully shutdown the optimization engine"""
        print("Shutting down Scalable Research Optimization Engine...")
        
        # Stop scheduler
        self.scheduler.stop_scheduler()
        
        # Clean up resource pools
        self.resource_manager.thread_pool.shutdown(wait=True)
        self.resource_manager.process_pool.shutdown(wait=True)
        
        print("Shutdown complete.")

def create_cli_interface():
    """Create comprehensive CLI interface"""
    parser = argparse.ArgumentParser(
        description="Scalable Research Optimization Engine - Generation 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmark --duration 10     # Run 10-minute benchmark
  %(prog)s optimize --config research.json    # Optimize research task
  %(prog)s status                      # Show system status
  %(prog)s monitor --realtime          # Real-time monitoring
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--duration', type=int, default=5,
                                help='Benchmark duration in minutes (default: 5)')
    benchmark_parser.add_argument('--output', type=str, 
                                help='Output file for benchmark results')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize research task')
    optimize_parser.add_argument('--config', type=str, required=True,
                               help='Research task configuration file')
    optimize_parser.add_argument('--output', type=str,
                               help='Output file for optimization results')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--detailed', action='store_true',
                             help='Show detailed status information')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='System monitoring')
    monitor_parser.add_argument('--realtime', action='store_true',
                              help='Real-time monitoring mode')
    monitor_parser.add_argument('--interval', type=int, default=5,
                              help='Monitoring interval in seconds (default: 5)')
    
    # Configuration options
    parser.add_argument('--cache-size', type=int, default=1000,
                       help='Cache size (default: 1000)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum worker threads (default: CPU count)')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                       help='Cache directory (default: ./cache)')
    parser.add_argument('--auto-optimize', action='store_true',
                       help='Enable automatic optimization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_cli_interface()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize engine
    config = {
        'cache_size': args.cache_size,
        'max_workers': args.max_workers,
        'cache_dir': args.cache_dir,
        'auto_optimize': args.auto_optimize
    }
    
    engine = ScalableResearchOptimizationEngine(config)
    
    try:
        if args.command == 'benchmark':
            print("=" * 80)
            print("SCALABLE RESEARCH OPTIMIZATION ENGINE - BENCHMARK MODE")
            print("=" * 80)
            
            results = engine.run_comprehensive_benchmark(args.duration)
            
            print("\n" + "=" * 50)
            print("BENCHMARK RESULTS")
            print("=" * 50)
            
            # Performance metrics
            perf_metrics = results['performance_benchmark']
            print(f"\nPerformance Metrics:")
            print(f"  Duration: {args.duration} minutes")
            print(f"  CPU Usage: {perf_metrics.get('cpu_stats', {}).get('avg', 0):.1f}% (avg)")
            print(f"  Memory Usage: {perf_metrics.get('memory_stats', {}).get('avg', 0):.1f}% (avg)")
            print(f"  Cache Hit Rate: {perf_metrics.get('cache_performance', {}).get('hit_rate', 0):.2f}")
            
            # Load test results
            load_test = results['load_test']
            print(f"\nLoad Test Results:")
            print(f"  Tasks Submitted: {load_test['total_tasks_submitted']}")
            print(f"  Tasks Completed: {load_test['completed_tasks']}")
            print(f"  Success Rate: {load_test['success_rate']:.2%}")
            print(f"  Avg Execution Time: {load_test['average_execution_time']:.2f}s")
            
            # Optimization recommendations
            recommendations = results['recommendations']
            if recommendations:
                print(f"\nOptimization Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"  {i}. {rec.category}: {rec.description}")
                    print(f"     Action: {rec.action}")
                    print(f"     Impact: {rec.estimated_impact}")
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nBenchmark results saved to: {args.output}")
        
        elif args.command == 'optimize':
            print("=" * 80)
            print("SCALABLE RESEARCH OPTIMIZATION ENGINE - OPTIMIZATION MODE")
            print("=" * 80)
            
            # Load configuration
            try:
                with open(args.config, 'r') as f:
                    research_config = json.load(f)
            except FileNotFoundError:
                # Create example configuration
                research_config = {
                    'data_size': 10000,
                    'compute_requirements': 'medium',
                    'parallel_potential': 'high',
                    'memory_intensive': False,
                    'io_intensive': False,
                    'machine_learning': True,
                    'max_parallel_tasks': 4
                }
                print(f"Configuration file not found. Using example configuration:")
                print(json.dumps(research_config, indent=2))
            
            # Run optimization
            results = engine.optimize_research_task(research_config)
            
            print("\n" + "=" * 50)
            print("OPTIMIZATION RESULTS")
            print("=" * 50)
            
            print(f"\nTask ID: {results['task_id']}")
            print(f"Optimization Strategy: {results['optimization_strategy']}")
            
            # Execution results
            exec_results = results['execution_results']
            print(f"\nExecution Results:")
            print(f"  Duration: {exec_results.get('duration', 0):.2f} seconds")
            print(f"  Subtasks: {len(exec_results.get('subtasks', []))}")
            
            # Performance analysis
            perf_analysis = results['post_optimization']['performance_analysis']
            print(f"  Success Rate: {perf_analysis['success_rate']:.2%}")
            print(f"  Performance Score: {perf_analysis['performance_score']:.1f}/100")
            
            # Cost analysis
            cost_analysis = results['post_optimization']['cost_analysis']
            print(f"  Total Cost: ${cost_analysis['total_cost']:.4f}")
            print(f"  Cost per Subtask: ${cost_analysis['cost_per_subtask']:.4f}")
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nOptimization results saved to: {args.output}")
        
        elif args.command == 'status':
            print("=" * 80)
            print("SCALABLE RESEARCH OPTIMIZATION ENGINE - STATUS")
            print("=" * 80)
            
            status = engine.get_comprehensive_status()
            
            print(f"\nSystem Status: {status['system_status'].upper()}")
            print(f"Timestamp: {status['timestamp']}")
            
            # Resource utilization
            resources = status['performance_metrics']['resources']
            print(f"\nResource Utilization:")
            print(f"  CPU: {resources['cpu_percent']:.1f}%")
            print(f"  Memory: {resources['memory_percent']:.1f}%")
            print(f"  Disk: {resources['disk_usage']:.1f}%")
            if resources.get('gpu_usage') is not None:
                print(f"  GPU: {resources['gpu_usage']:.1f}%")
            
            # Cache performance
            cache = status['cache_performance']
            print(f"\nCache Performance:")
            print(f"  Hit Rate: {cache['hit_rate']:.2%}")
            print(f"  L1 Cache: {cache['l1_size']} items")
            print(f"  L2 Cache: {cache['l2_size']} items")
            print(f"  Total Requests: {cache['total_requests']}")
            
            # Scheduler performance
            scheduler = status['scheduler_performance']
            print(f"\nScheduler Performance:")
            print(f"  Total Tasks: {scheduler['total_tasks']}")
            print(f"  Completed: {scheduler['completed_tasks']}")
            print(f"  Pending: {scheduler['pending_tasks']}")
            print(f"  Success Rate: {scheduler['success_rate']:.2%}")
            
            # Bottlenecks
            bottlenecks = status['bottlenecks']
            if bottlenecks:
                print(f"\nBottlenecks Detected:")
                for bottleneck in bottlenecks:
                    print(f"  - {bottleneck}")
            
            # Top recommendations
            recommendations = status['recommendations']
            if recommendations:
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec.category} (Priority {rec.priority}): {rec.description}")
            
            if args.detailed:
                print(f"\nDetailed Metrics:")
                print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'monitor':
            print("=" * 80)
            print("SCALABLE RESEARCH OPTIMIZATION ENGINE - MONITORING MODE")
            print("=" * 80)
            
            if args.realtime:
                print("Real-time monitoring started. Press Ctrl+C to stop.\n")
                try:
                    while True:
                        # Clear screen (works on most terminals)
                        os.system('clear' if os.name == 'posix' else 'cls')
                        
                        print("=" * 60)
                        print(f"REAL-TIME MONITORING - {datetime.now().strftime('%H:%M:%S')}")
                        print("=" * 60)
                        
                        # Get current metrics
                        metrics = engine.monitor.collect_metrics()
                        
                        # Display key metrics
                        resources = metrics['resources']
                        print(f"CPU: {resources['cpu_percent']:6.1f}% | "
                              f"Memory: {resources['memory_percent']:6.1f}% | "
                              f"Disk: {resources['disk_usage']:6.1f}%")
                        
                        cache = metrics['cache']
                        print(f"Cache Hit Rate: {cache['hit_rate']:6.2%} | "
                              f"L1: {cache['l1_size']:4d} | "
                              f"L2: {cache['l2_size']:4d}")
                        
                        scheduler = metrics['scheduler']
                        print(f"Tasks - Completed: {scheduler['completed_tasks']:4d} | "
                              f"Pending: {scheduler['pending_tasks']:4d} | "
                              f"Success Rate: {scheduler['success_rate']:6.2%}")
                        
                        # Show bottlenecks
                        bottlenecks = engine.monitor.detect_bottlenecks()
                        if bottlenecks:
                            print(f"\nBottlenecks: {', '.join(bottlenecks)}")
                        else:
                            print(f"\nSystem Status: Optimal")
                        
                        time.sleep(args.interval)
                
                except KeyboardInterrupt:
                    print("\nMonitoring stopped.")
            else:
                # Single monitoring snapshot
                metrics = engine.monitor.collect_metrics()
                print(json.dumps(metrics, indent=2, default=str))
    
    finally:
        engine.shutdown()

if __name__ == "__main__":
    main()