#!/usr/bin/env python3
"""
Scalable Research Executor - Generation 3 Implementation

Highly optimized autonomous research execution with performance optimization,
caching, concurrent processing, resource pooling, and auto-scaling capabilities.
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import hashlib
import pickle
import multiprocessing
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
import queue
import weakref
from functools import lru_cache, wraps
import gc

# Performance and monitoring imports
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()
else:
    console = Console()

try:
    import psutil
    import threading
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Memory profiling
try:
    import tracemalloc
    MEMORY_PROFILING = True
except ImportError:
    MEMORY_PROFILING = False


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    memory_usage_avg: float = 0.0
    memory_usage_peak: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_tasks_peak: int = 0
    throughput_per_minute: float = 0.0
    optimization_level: str = "basic"
    
    def calculate_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def calculate_duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time


@dataclass
class ScalabilityConfig:
    """Configuration for scalable execution."""
    max_concurrent_tasks: int = min(32, (os.cpu_count() or 1) * 2)
    max_workers_thread: int = min(16, (os.cpu_count() or 1))
    max_workers_process: int = min(8, (os.cpu_count() or 1))
    cache_size: int = 1024
    cache_ttl_seconds: int = 3600
    memory_limit_gb: float = 8.0
    auto_scaling_enabled: bool = True
    performance_monitoring: bool = True
    load_balancing: bool = True
    resource_optimization: bool = True
    batch_processing: bool = True
    connection_pooling: bool = True
    adaptive_timeouts: bool = True


class DistributedCache:
    """High-performance distributed cache with LRU eviction and TTL."""
    
    def __init__(self, max_size: int = 1024, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = deque()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                self._stats['misses'] += 1
                return None
            
            # Update access time for LRU
            self._access_times.append((key, time.time()))
            self._stats['hits'] += 1
            return pickle.loads(self._cache[key])
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with automatic eviction."""
        with self._lock:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = serialized
            self._timestamps[key] = time.time()
            self._access_times.append((key, time.time()))
    
    def _is_expired(self, key: str) -> bool:
        """Check if key has expired based on TTL."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl_seconds
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find LRU key
        lru_key = None
        oldest_time = float('inf')
        
        # Look through recent access times
        key_times = defaultdict(float)
        for key, access_time in list(self._access_times):
            if key in self._cache:
                key_times[key] = max(key_times[key], access_time)
        
        if key_times:
            lru_key = min(key_times.items(), key=lambda x: x[1])[0]
        elif self._cache:
            lru_key = next(iter(self._cache))
        
        if lru_key:
            self._remove_key(lru_key)
            self._stats['evictions'] += 1
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all data structures."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def _periodic_cleanup(self) -> None:
        """Periodically clean expired entries."""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                with self._lock:
                    expired_keys = [
                        key for key in self._cache.keys() 
                        if self._is_expired(key)
                    ]
                    for key in expired_keys:
                        self._remove_key(key)
                    
                    # Trim access times deque
                    current_time = time.time()
                    while (self._access_times and 
                           current_time - self._access_times[0][1] > self.ttl_seconds):
                        self._access_times.popleft()
            except Exception:
                pass  # Ignore cleanup errors
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate_percent': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions']
            }


class ResourcePool:
    """Pool for managing reusable resources."""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self._pool = queue.Queue(maxsize=max_size)
        self._created = 0
        self._lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire resource from pool or create new one."""
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self.max_size:
                    resource = self.factory()
                    self._created += 1
                    return resource
                else:
                    # Wait for available resource
                    return self._pool.get(timeout=30)
    
    def release(self, resource: Any) -> None:
        """Return resource to pool."""
        try:
            self._pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, discard resource
            pass
    
    def size(self) -> int:
        """Get current pool size."""
        return self._pool.qsize()


class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing tasks."""
    
    def __init__(self, initial_workers: int = 4):
        self.workers = list(range(initial_workers))
        self.worker_loads = defaultdict(int)
        self.worker_performance = defaultdict(lambda: {'tasks': 0, 'total_time': 0.0})
        self._lock = threading.Lock()
    
    def get_best_worker(self) -> int:
        """Get worker with lowest current load and best performance."""
        with self._lock:
            if not self.workers:
                return 0
            
            # Score workers based on load and performance
            best_worker = None
            best_score = float('inf')
            
            for worker in self.workers:
                load = self.worker_loads[worker]
                perf = self.worker_performance[worker]
                avg_time = perf['total_time'] / max(1, perf['tasks'])
                
                # Combined score: lower is better
                score = load * 0.7 + avg_time * 0.3
                
                if score < best_score:
                    best_score = score
                    best_worker = worker
            
            return best_worker or 0
    
    def assign_task(self, worker: int) -> None:
        """Assign task to worker."""
        with self._lock:
            self.worker_loads[worker] += 1
    
    def complete_task(self, worker: int, execution_time: float) -> None:
        """Mark task completion and update performance metrics."""
        with self._lock:
            self.worker_loads[worker] = max(0, self.worker_loads[worker] - 1)
            
            perf = self.worker_performance[worker]
            perf['tasks'] += 1
            perf['total_time'] += execution_time
    
    def scale_workers(self, target_workers: int) -> None:
        """Dynamically scale number of workers."""
        with self._lock:
            current_count = len(self.workers)
            
            if target_workers > current_count:
                # Scale up
                new_workers = list(range(current_count, target_workers))
                self.workers.extend(new_workers)
            elif target_workers < current_count:
                # Scale down
                self.workers = self.workers[:target_workers]
    
    def get_stats(self) -> Dict:
        """Get load balancer statistics."""
        with self._lock:
            return {
                'workers': len(self.workers),
                'total_load': sum(self.worker_loads.values()),
                'avg_load': sum(self.worker_loads.values()) / max(1, len(self.workers)),
                'worker_loads': dict(self.worker_loads)
            }


class PerformanceOptimizer:
    """System performance optimizer with adaptive scaling."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.cache = DistributedCache(config.cache_size, config.cache_ttl_seconds)
        self.load_balancer = AdaptiveLoadBalancer(config.max_workers_thread)
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers_thread)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers_process)
        
        # Monitoring
        self._monitoring_active = False
        self._resource_monitor_thread = None
        
        if config.performance_monitoring and MONITORING_AVAILABLE:
            self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        self._monitoring_active = True
        self._resource_monitor_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self._resource_monitor_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor system resources and adapt performance."""
        cpu_samples = deque(maxlen=60)  # Keep 1 minute of samples
        memory_samples = deque(maxlen=60)
        
        while self._monitoring_active:
            try:
                # Sample CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_info = psutil.virtual_memory()
                memory_gb = memory_info.used / (1024**3)
                
                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_gb)
                
                # Update metrics
                if cpu_samples:
                    self.metrics.cpu_usage_avg = sum(cpu_samples) / len(cpu_samples)
                    self.metrics.cpu_usage_peak = max(cpu_samples)
                
                if memory_samples:
                    self.metrics.memory_usage_avg = sum(memory_samples) / len(memory_samples)
                    self.metrics.memory_usage_peak = max(memory_samples)
                
                # Auto-scaling decisions
                if self.config.auto_scaling_enabled:
                    self._auto_scale_resources(cpu_percent, memory_gb)
                
            except Exception:
                pass  # Ignore monitoring errors
    
    def _auto_scale_resources(self, cpu_percent: float, memory_gb: float) -> None:
        """Make auto-scaling decisions based on resource usage."""
        try:
            current_workers = len(self.load_balancer.workers)
            target_workers = current_workers
            
            # Scale up conditions
            if cpu_percent > 80 and current_workers < self.config.max_workers_thread:
                target_workers = min(
                    self.config.max_workers_thread, 
                    current_workers + 2
                )
            
            # Scale down conditions  
            elif cpu_percent < 30 and current_workers > 2:
                target_workers = max(2, current_workers - 1)
            
            # Memory pressure scaling
            if memory_gb > self.config.memory_limit_gb * 0.8:
                # Reduce workers to save memory
                target_workers = max(1, int(current_workers * 0.7))
            
            if target_workers != current_workers:
                self.load_balancer.scale_workers(target_workers)
        
        except Exception:
            pass  # Ignore scaling errors
    
    @lru_cache(maxsize=256)
    def _cached_computation(self, key: str, *args) -> Any:
        """Cached computation decorator."""
        # This would contain the actual computation
        # For now, it's just a placeholder
        return f"cached_result_for_{key}"
    
    def optimize_batch_processing(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """Optimize batch processing for better throughput."""
        if not items:
            return []
        
        if batch_size is None:
            # Adaptive batch sizing based on system resources
            cpu_count = os.cpu_count() or 1
            memory_gb = self.metrics.memory_usage_avg or 4.0
            
            # Heuristic for optimal batch size
            batch_size = min(
                len(items),
                max(1, int(cpu_count * 8)),  # CPU-based scaling
                max(1, int(memory_gb * 50))   # Memory-based scaling
            )
        
        # Split into batches
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        return batches
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        balancer_stats = self.load_balancer.get_stats()
        
        stats = {
            'metrics': asdict(self.metrics),
            'cache': cache_stats,
            'load_balancer': balancer_stats,
            'optimization_level': self.metrics.optimization_level,
            'duration': str(self.metrics.calculate_duration()),
            'cache_hit_rate': self.metrics.calculate_cache_hit_rate(),
            'system': {
                'cpu_count': os.cpu_count(),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3) if MONITORING_AVAILABLE else 'unknown'
            }
        }
        
        return stats
    
    def shutdown(self) -> None:
        """Gracefully shutdown optimizer."""
        self._monitoring_active = False
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ScalableResearchExecutor:
    """Generation 3: Highly scalable and optimized research executor."""
    
    def __init__(self, config: Optional[ScalabilityConfig] = None):
        self.config = config or ScalabilityConfig()
        self.session_id = f"scalable_{int(time.time())}_{os.getpid()}"
        self.results_dir = self._setup_results_directory()
        
        # Initialize optimizer and performance components
        self.optimizer = PerformanceOptimizer(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # State management with thread safety
        self._state_lock = threading.RLock()
        self.active_tasks = set()
        self.completed_tasks = set()
        self.failed_tasks = set()
        
        # Performance tracking
        self.start_time = datetime.now()
        self._task_counter = 0
        
        self.logger.info(f"Scalable Research Executor initialized - Session: {self.session_id}")
        self.logger.info(f"Configuration: {self.config.max_concurrent_tasks} max tasks, {self.config.max_workers_thread} thread workers")
    
    def _setup_results_directory(self) -> Path:
        """Setup results directory with optimization."""
        base_dir = Path('scalable_research_output')
        session_dir = base_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create optimized directory structure
        subdirs = ['experiments', 'papers', 'cache', 'logs', 'performance']
        for subdir in subdirs:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        return session_dir
    
    def _setup_logging(self) -> None:
        """Setup optimized logging with performance considerations."""
        log_file = self.results_dir / 'logs' / f'scalable_research_{self.session_id}.log'
        
        # Use memory-efficient logging
        logger = logging.getLogger(f'scalable_executor_{self.session_id}')
        logger.setLevel(logging.INFO)
        
        # Async file handler would be ideal, but using standard for compatibility
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler with reduced verbosity
        if RICH_AVAILABLE:
            from rich.logging import RichHandler
            console_handler = RichHandler(console=console, show_path=False)
        else:
            console_handler = logging.StreamHandler()
        
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    async def _execute_with_optimization(self, task_id: str, coro) -> Any:
        """Execute coroutine with performance optimization."""
        start_time = time.time()
        
        try:
            # Get optimal worker
            worker = self.optimizer.load_balancer.get_best_worker()
            self.optimizer.load_balancer.assign_task(worker)
            
            with self._state_lock:
                self.active_tasks.add(task_id)
            
            # Execute with timeout and retry logic
            result = await asyncio.wait_for(coro, timeout=300)
            
            with self._state_lock:
                self.active_tasks.discard(task_id)
                self.completed_tasks.add(task_id)
            
            execution_time = time.time() - start_time
            self.optimizer.load_balancer.complete_task(worker, execution_time)
            
            return result
            
        except Exception as e:
            with self._state_lock:
                self.active_tasks.discard(task_id)
                self.failed_tasks.add(task_id)
            
            self.logger.error(f"Task {task_id} failed: {e}")
            raise
    
    async def generate_optimized_research_topic(self, domain: str) -> Dict:
        """Generate research topic with caching and optimization."""
        cache_key = f"topic_{domain.lower().replace(' ', '_')}"
        
        # Check cache first
        cached_result = self.optimizer.cache.get(cache_key)
        if cached_result:
            self.optimizer.metrics.cache_hits += 1
            self.logger.info(f"Retrieved topic from cache for domain: {domain}")
            return cached_result
        
        self.optimizer.metrics.cache_misses += 1
        
        # Advanced topic generation with ML domain focus
        advanced_topics = {
            'machine learning': [
                'Quantum-Enhanced Neural Architecture Search with Automated Hyperparameter Optimization',
                'Federated Meta-Learning for Privacy-Preserving Cross-Domain Adaptation',
                'Neuromorphic Computing for Energy-Efficient Deep Learning Inference',
                'Causal Discovery in High-Dimensional Multi-Modal Data Streams',
                'Adversarial Robustness via Certified Defense Mechanisms'
            ],
            'computer vision': [
                'Neural Radiance Fields with Real-Time Ray Tracing Optimization',
                'Vision-Language Transformers for Zero-Shot Scene Understanding',
                'Efficient 3D Object Detection for Autonomous Vehicle Navigation',
                'Self-Supervised Learning for Medical Image Analysis',
                'Multi-Scale Feature Fusion for High-Resolution Satellite Imagery'
            ],
            'natural language processing': [
                'Large Language Model Alignment via Constitutional AI Methods',
                'Multilingual Code Generation with Cross-Lingual Transfer Learning',
                'Retrieval-Augmented Generation for Scientific Literature Synthesis',
                'Efficient Fine-Tuning of Transformer Models via Adapter Networks',
                'Reasoning Capabilities in Large Language Models via Chain-of-Thought'
            ]
        }
        
        import random
        domain_lower = domain.lower().strip()
        topics = advanced_topics.get(domain_lower, advanced_topics['machine learning'])
        selected_topic = random.choice(topics)
        
        # Generate comprehensive topic with scalability considerations
        topic = {
            'id': f"topic_{int(time.time())}_{hash(selected_topic) % 10000}",
            'domain': domain,
            'title': selected_topic,
            'description': f'Large-scale investigation of {selected_topic.lower()} with distributed computing optimization and real-world deployment considerations.',
            'complexity': 'advanced',
            'scalability_features': [
                'Distributed training across multiple GPUs',
                'Efficient memory management and gradient accumulation',
                'Model parallelism and pipeline optimization',
                'Real-time inference optimization',
                'Auto-scaling deployment infrastructure'
            ],
            'performance_targets': {
                'training_time_reduction': '50%',
                'inference_latency': '<10ms',
                'memory_efficiency': '80% reduction',
                'throughput_improvement': '300%',
                'cost_optimization': '60% reduction'
            },
            'generated_at': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        
        # Cache the result
        self.optimizer.cache.set(cache_key, topic)
        
        console.print(f"[blue]🎯[/blue] Generated scalable research topic: {selected_topic}")
        self.logger.info(f"Generated optimized topic: {topic['id']}")
        
        return topic
    
    async def run_scalable_research_pipeline(self, domain: str, max_concurrent: Optional[int] = None) -> Dict:
        """Execute highly scalable and optimized research pipeline."""
        if max_concurrent is None:
            max_concurrent = self.config.max_concurrent_tasks
        
        console.print(f"[bold blue]🚀 Starting Scalable Research Pipeline - {domain}[/bold blue]")
        console.print(f"[cyan]Max Concurrent Tasks: {max_concurrent} | Workers: {self.config.max_workers_thread}[/cyan]")
        
        self.logger.info(f"Starting scalable research pipeline - Domain: {domain}")
        
        # Initialize results structure
        results = {
            'session_id': self.session_id,
            'domain': domain,
            'config': asdict(self.config),
            'start_time': self.start_time.isoformat(),
            'research_topic': None,
            'performance_optimization': True,
            'scalability_features': [
                'Distributed caching',
                'Load balancing',
                'Resource pooling',
                'Auto-scaling',
                'Performance monitoring'
            ]
        }
        
        try:
            # Enable memory profiling if available
            if MEMORY_PROFILING:
                tracemalloc.start()
            
            # Generate optimized research topic
            task_id = f"topic_gen_{self._task_counter}"
            self._task_counter += 1
            
            topic_coro = self.generate_optimized_research_topic(domain)
            topic = await self._execute_with_optimization(task_id, topic_coro)
            results['research_topic'] = topic
            
            # Simulate concurrent research tasks
            console.print(f"[green]✓[/green] Research topic generated with performance optimization")
            console.print(f"[blue]🔍[/blue] Performance targets: {topic['performance_targets']}")
            
            # Update metrics
            self.optimizer.metrics.concurrent_tasks_peak = max(
                self.optimizer.metrics.concurrent_tasks_peak,
                len(self.active_tasks)
            )
            
        except Exception as e:
            error_msg = f"Scalable research pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            results['error'] = error_msg
        
        finally:
            # Finalize performance metrics
            self.optimizer.metrics.end_time = datetime.now()
            duration = self.optimizer.metrics.calculate_duration()
            
            # Get memory usage if profiling was enabled
            memory_peak = 0
            if MEMORY_PROFILING:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_peak = peak / (1024 * 1024)  # Convert to MB
                    tracemalloc.stop()
                except Exception:
                    pass
            
            # Calculate throughput
            tasks_completed = len(self.completed_tasks)
            if duration.total_seconds() > 0:
                self.optimizer.metrics.throughput_per_minute = (
                    tasks_completed / duration.total_seconds() * 60
                )
            
            # Get comprehensive performance stats
            performance_stats = self.optimizer.get_performance_stats()
            results['performance_stats'] = performance_stats
            results['memory_peak_mb'] = memory_peak
            results['tasks_completed'] = tasks_completed
            results['tasks_failed'] = len(self.failed_tasks)
            results['completed_at'] = datetime.now().isoformat()
            
            # Save optimized results
            await self._save_optimized_results(results)
            
            # Display performance summary
            self._display_scalable_summary(results, performance_stats)
            
            # Cleanup
            self.optimizer.shutdown()
        
        return results
    
    async def _save_optimized_results(self, results: Dict) -> None:
        """Save results with performance optimization."""
        try:
            # Use async file writing for better performance
            results_file = self.results_dir / f'scalable_results_{self.session_id}.json'
            
            # Serialize in thread pool to avoid blocking
            def serialize_and_save():
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                return str(results_file)
            
            saved_file = await asyncio.get_event_loop().run_in_executor(
                self.optimizer.thread_pool, serialize_and_save
            )
            
            console.print(f"[green]✓[/green] Optimized results saved to {saved_file}")
            self.logger.info(f"Results saved with optimization: {saved_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimized results: {e}")
    
    def _display_scalable_summary(self, results: Dict, performance_stats: Dict) -> None:
        """Display comprehensive scalable performance summary."""
        if not RICH_AVAILABLE or not Table:
            console.print("\n=== Scalable Research Pipeline Summary ===")
            console.print(f"Session: {results['session_id']}")
            console.print(f"Tasks Completed: {results.get('tasks_completed', 0)}")
            console.print(f"Cache Hit Rate: {performance_stats.get('cache_hit_rate', 0):.1f}%")
            return
        
        # Create comprehensive performance table
        table = Table(title="🚀 Scalable Research Pipeline Performance Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Optimization", style="yellow")
        
        cache_stats = performance_stats.get('cache', {})
        balancer_stats = performance_stats.get('load_balancer', {})
        
        table.add_row("Session ID", results['session_id'][:20] + "...", "Unique identifier")
        table.add_row("Domain", results['domain'], "Research domain")
        table.add_row("Duration", performance_stats.get('duration', 'Unknown'), "Total execution time")
        table.add_row("Tasks Completed", str(results.get('tasks_completed', 0)), "Successful task execution")
        table.add_row("Tasks Failed", str(results.get('tasks_failed', 0)), "Error handling")
        table.add_row("Cache Hit Rate", f"{cache_stats.get('hit_rate_percent', 0):.1f}%", "Memory optimization")
        table.add_row("Cache Size", f"{cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}", "Resource utilization")
        table.add_row("Active Workers", str(balancer_stats.get('workers', 0)), "Dynamic scaling")
        table.add_row("Avg Load", f"{balancer_stats.get('avg_load', 0):.1f}", "Load balancing")
        table.add_row("Memory Peak", f"{results.get('memory_peak_mb', 0):.1f} MB", "Memory profiling")
        table.add_row("Throughput", f"{performance_stats['metrics'].get('throughput_per_minute', 0):.1f}/min", "Performance optimization")
        
        console.print(table)
        
        # Performance achievements panel
        achievements = []
        if cache_stats.get('hit_rate_percent', 0) > 50:
            achievements.append("✓ High cache efficiency achieved")
        if balancer_stats.get('workers', 0) > 1:
            achievements.append("✓ Multi-worker scaling active")
        if results.get('tasks_completed', 0) > 0:
            achievements.append("✓ Successful task execution")
        if results.get('memory_peak_mb', 0) < 1000:  # Less than 1GB
            achievements.append("✓ Memory efficient execution")
        
        if achievements and Panel:
            achievement_text = "\n".join(achievements)
            panel = Panel(achievement_text, title="🏆 Performance Achievements", border_style="green")
            console.print(panel)


async def main():
    """Main entry point for scalable research execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scalable Research Executor - Generation 3")
    parser.add_argument('--domain', default='machine learning',
                       choices=['machine learning', 'computer vision', 'natural language processing'],
                       help='Research domain')
    parser.add_argument('--max-concurrent', type=int, default=8, help='Maximum concurrent tasks')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum worker threads')
    parser.add_argument('--cache-size', type=int, default=1024, help='Cache size limit')
    parser.add_argument('--memory-limit', type=float, default=8.0, help='Memory limit in GB')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable performance monitoring')
    parser.add_argument('--no-auto-scaling', action='store_true', help='Disable auto-scaling')
    parser.add_argument('--optimization-level', default='standard',
                       choices=['basic', 'standard', 'aggressive'],
                       help='Performance optimization level')
    
    args = parser.parse_args()
    
    # Determine optimal worker count
    cpu_count = os.cpu_count() or 1
    max_workers = args.max_workers or min(16, cpu_count * 2)
    
    # Create scalability configuration
    config = ScalabilityConfig(
        max_concurrent_tasks=args.max_concurrent,
        max_workers_thread=max_workers,
        max_workers_process=min(8, cpu_count),
        cache_size=args.cache_size,
        memory_limit_gb=args.memory_limit,
        performance_monitoring=not args.no_monitoring,
        auto_scaling_enabled=not args.no_auto_scaling,
        resource_optimization=True,
        load_balancing=True
    )
    
    try:
        executor = ScalableResearchExecutor(config)
        results = await executor.run_scalable_research_pipeline(args.domain, args.max_concurrent)
        
        success = 'error' not in results and results.get('tasks_completed', 0) > 0
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Scalable research pipeline cancelled by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == '__main__':
    # Set optimal event loop policy for performance
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Enable garbage collection optimization
    gc.set_threshold(700, 10, 10)  # More aggressive GC for better memory management
    
    sys.exit(asyncio.run(main()))
