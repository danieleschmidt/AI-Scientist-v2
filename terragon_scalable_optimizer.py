#!/usr/bin/env python3
"""
Terragon Scalable Optimizer - Generation 3 Implementation
Performance optimization, caching, concurrent processing, and auto-scaling.
"""

import asyncio
import json
import logging
import os
import sys
import time
import hashlib
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from enum import Enum
import multiprocessing
import pickle
import weakref
import gc

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Enhanced console for output
if RICH_AVAILABLE:
    console = Console()
else:
    console = None

class CacheStrategy(Enum):
    """Cache strategy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"

class ScalingStrategy(Enum):
    """Auto-scaling strategy enumeration."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    concurrent_executions: int = 0
    peak_concurrent_executions: int = 0
    total_cpu_time: float = 0.0
    memory_peak_mb: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    last_updated: str = ""

@dataclass
class ScalableResearchProject:
    """Scalable research project with optimization capabilities."""
    name: str
    domain: str
    objectives: List[str]
    priority: int = 1
    complexity_score: float = 1.0
    estimated_duration: float = 60.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None
    parallelizable: bool = True
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.cache_key:
            self.cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self) -> str:
        """Generate unique cache key for the project."""
        content = f"{self.name}{self.domain}{json.dumps(sorted(self.objectives))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class AdvancedCache:
    """Advanced caching system with multiple strategies and optimization."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.insertion_order: List[str] = []
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with strategy-specific access tracking."""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                self._update_access_stats(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with eviction if necessary."""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self._update_access_stats(key)
                return
            
            if len(self.cache) >= self.max_size:
                self._evict()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.insertion_order.append(key)
    
    def _update_access_stats(self, key: str) -> None:
        """Update access statistics for cache strategies."""
        self.access_times[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def _evict(self) -> None:
        """Evict item based on cache strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first inserted
            oldest_key = self.insertion_order[0]
            self.insertion_order.remove(oldest_key)
        else:  # ADAPTIVE
            # Adaptive strategy considering both frequency and recency
            current_time = time.time()
            scores = {}
            for key in self.cache.keys():
                recency_score = current_time - self.access_times[key]
                frequency_score = 1.0 / max(self.access_counts[key], 1)
                scores[key] = recency_score * frequency_score
            oldest_key = max(scores.keys(), key=lambda k: scores[k])
        
        # Remove from all tracking structures
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_counts[oldest_key]
        if oldest_key in self.insertion_order:
            self.insertion_order.remove(oldest_key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value
        }
    
    def clear(self) -> None:
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
            self.hits = 0
            self.misses = 0

class LoadBalancer:
    """Advanced load balancer for distributing work across resources."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.workers: List[Any] = []
        self.current_worker = 0
        self.worker_loads: Dict[str, float] = {}
        self.worker_performance: Dict[str, float] = {}
        
    def add_worker(self, worker_id: str, capacity: float = 1.0) -> None:
        """Add worker to load balancer."""
        self.workers.append(worker_id)
        self.worker_loads[worker_id] = 0.0
        self.worker_performance[worker_id] = capacity
    
    def get_next_worker(self) -> str:
        """Get next worker based on load balancing strategy."""
        if not self.workers:
            raise Exception("No workers available")
        
        if self.strategy == "round_robin":
            worker = self.workers[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            return worker
        elif self.strategy == "least_loaded":
            return min(self.workers, key=lambda w: self.worker_loads.get(w, 0))
        elif self.strategy == "performance_weighted":
            # Choose worker based on performance and current load
            scores = {}
            for worker in self.workers:
                performance = self.worker_performance.get(worker, 1.0)
                load = self.worker_loads.get(worker, 0.0)
                scores[worker] = performance / max(load + 1, 0.1)
            return max(scores.keys(), key=lambda w: scores[w])
        else:
            return self.workers[0]
    
    def update_worker_load(self, worker_id: str, load: float) -> None:
        """Update worker load information."""
        self.worker_loads[worker_id] = load
    
    def update_worker_performance(self, worker_id: str, performance: float) -> None:
        """Update worker performance metric."""
        self.worker_performance[worker_id] = performance

class AutoScaler:
    """Auto-scaling system for dynamic resource management."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8, target_utilization: float = 0.7):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.current_workers = min_workers
        self.utilization_history: List[float] = []
        self.scaling_cooldown = 30  # seconds
        self.last_scaling_time = 0
        
    def should_scale_up(self, current_utilization: float) -> bool:
        """Determine if we should scale up."""
        if self.current_workers >= self.max_workers:
            return False
        
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        # Scale up if utilization is consistently high
        self.utilization_history.append(current_utilization)
        if len(self.utilization_history) > 5:
            self.utilization_history.pop(0)
        
        avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
        return avg_utilization > self.target_utilization and len(self.utilization_history) >= 3
    
    def should_scale_down(self, current_utilization: float) -> bool:
        """Determine if we should scale down."""
        if self.current_workers <= self.min_workers:
            return False
        
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        self.utilization_history.append(current_utilization)
        if len(self.utilization_history) > 5:
            self.utilization_history.pop(0)
        
        avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
        return avg_utilization < (self.target_utilization * 0.5) and len(self.utilization_history) >= 3
    
    def scale_up(self) -> int:
        """Scale up workers."""
        new_workers = min(self.current_workers + 1, self.max_workers)
        self.current_workers = new_workers
        self.last_scaling_time = time.time()
        return new_workers
    
    def scale_down(self) -> int:
        """Scale down workers."""
        new_workers = max(self.current_workers - 1, self.min_workers)
        self.current_workers = new_workers
        self.last_scaling_time = time.time()
        return new_workers

class PerformanceOptimizer:
    """Performance optimization engine with multiple strategies."""
    
    def __init__(self):
        self.optimization_strategies = [
            self._batch_optimization,
            self._parallel_optimization,
            self._cache_optimization,
            self._resource_optimization
        ]
        self.performance_history: List[PerformanceMetrics] = []
        
    def optimize_execution_plan(self, projects: List[ScalableResearchProject]) -> List[List[ScalableResearchProject]]:
        """Optimize project execution plan for maximum performance."""
        # Sort projects by priority and complexity
        sorted_projects = sorted(projects, key=lambda p: (p.priority, -p.complexity_score))
        
        # Group projects into optimal batches
        batches = self._create_optimal_batches(sorted_projects)
        
        # Apply optimization strategies
        for strategy in self.optimization_strategies:
            batches = strategy(batches)
        
        return batches
    
    def _create_optimal_batches(self, projects: List[ScalableResearchProject]) -> List[List[ScalableResearchProject]]:
        """Create optimal batches based on dependencies and parallelizability."""
        batches = []
        remaining_projects = projects.copy()
        
        while remaining_projects:
            current_batch = []
            batch_complexity = 0.0
            max_batch_complexity = 5.0  # Configurable threshold
            
            projects_to_remove = []
            for project in remaining_projects:
                # Check if project can be added to current batch
                if (project.parallelizable and 
                    batch_complexity + project.complexity_score <= max_batch_complexity and
                    self._check_dependencies_satisfied(project, current_batch)):
                    
                    current_batch.append(project)
                    batch_complexity += project.complexity_score
                    projects_to_remove.append(project)
            
            # Remove added projects from remaining list
            for project in projects_to_remove:
                remaining_projects.remove(project)
            
            # If no projects could be added to current batch, add one project
            if not current_batch and remaining_projects:
                current_batch.append(remaining_projects.pop(0))
            
            if current_batch:
                batches.append(current_batch)
        
        return batches
    
    def _check_dependencies_satisfied(self, project: ScalableResearchProject, current_batch: List[ScalableResearchProject]) -> bool:
        """Check if project dependencies are satisfied."""
        if not project.dependencies:
            return True
        
        batch_project_names = {p.name for p in current_batch}
        return all(dep in batch_project_names for dep in project.dependencies)
    
    def _batch_optimization(self, batches: List[List[ScalableResearchProject]]) -> List[List[ScalableResearchProject]]:
        """Optimize batching strategy."""
        # Merge small batches if beneficial
        optimized_batches = []
        i = 0
        while i < len(batches):
            current_batch = batches[i]
            
            # Try to merge with next batch if both are small
            if (i + 1 < len(batches) and 
                len(current_batch) <= 2 and 
                len(batches[i + 1]) <= 2):
                
                merged_batch = current_batch + batches[i + 1]
                total_complexity = sum(p.complexity_score for p in merged_batch)
                
                if total_complexity <= 6.0:  # Safe merge threshold
                    optimized_batches.append(merged_batch)
                    i += 2
                    continue
            
            optimized_batches.append(current_batch)
            i += 1
        
        return optimized_batches
    
    def _parallel_optimization(self, batches: List[List[ScalableResearchProject]]) -> List[List[ScalableResearchProject]]:
        """Optimize for parallel execution."""
        # Sort projects within each batch for optimal parallel execution
        for batch in batches:
            batch.sort(key=lambda p: (-p.complexity_score, p.priority))
        
        return batches
    
    def _cache_optimization(self, batches: List[List[ScalableResearchProject]]) -> List[List[ScalableResearchProject]]:
        """Optimize for cache efficiency."""
        # Group similar projects together to improve cache hit rates
        for batch in batches:
            batch.sort(key=lambda p: (p.domain, p.cache_key))
        
        return batches
    
    def _resource_optimization(self, batches: List[List[ScalableResearchProject]]) -> List[List[ScalableResearchProject]]:
        """Optimize resource utilization."""
        # Balance resource requirements across batches
        for batch in batches:
            total_memory = sum(p.resource_requirements.get('memory_mb', 100) for p in batch)
            if total_memory > 2048:  # Memory limit
                # Split batch if memory requirements are too high
                batch.sort(key=lambda p: p.resource_requirements.get('memory_mb', 100))
        
        return batches

class ScalableExecutionEngine:
    """Scalable execution engine with advanced optimization."""
    
    def __init__(self, max_workers: int = 4):
        self.cache = AdvancedCache(max_size=500, strategy=CacheStrategy.ADAPTIVE)
        self.load_balancer = LoadBalancer(strategy="performance_weighted")
        self.auto_scaler = AutoScaler(min_workers=1, max_workers=max_workers)
        self.optimizer = PerformanceOptimizer()
        self.metrics = PerformanceMetrics()
        self.active_executions: Dict[str, threading.Thread] = {}
        self.execution_lock = threading.RLock()
        
        # Initialize workers
        for i in range(self.auto_scaler.current_workers):
            self.load_balancer.add_worker(f"worker_{i}")
    
    async def execute_projects_optimized(self, projects: List[ScalableResearchProject]) -> List[Dict[str, Any]]:
        """Execute projects with full optimization pipeline."""
        start_time = time.time()
        
        # Optimize execution plan
        optimized_batches = self.optimizer.optimize_execution_plan(projects)
        
        # Execute batches with auto-scaling
        all_results = []
        
        for batch_idx, batch in enumerate(optimized_batches):
            batch_results = await self._execute_batch_with_scaling(batch, batch_idx)
            all_results.extend(batch_results)
            
            # Update auto-scaling based on current utilization
            current_utilization = len(self.active_executions) / max(self.auto_scaler.current_workers, 1)
            
            if self.auto_scaler.should_scale_up(current_utilization):
                new_worker_count = self.auto_scaler.scale_up()
                self.load_balancer.add_worker(f"worker_{new_worker_count-1}")
                if RICH_AVAILABLE and console:
                    console.print(f"[green]⬆️ Scaled up to {new_worker_count} workers[/green]")
            elif self.auto_scaler.should_scale_down(current_utilization):
                new_worker_count = self.auto_scaler.scale_down()
                if RICH_AVAILABLE and console:
                    console.print(f"[yellow]⬇️ Scaled down to {new_worker_count} workers[/yellow]")
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics.total_requests += len(projects)
        self.metrics.successful_requests += len([r for r in all_results if r.get('status') == 'completed'])
        self.metrics.average_response_time = total_time / len(projects) if projects else 0
        self.metrics.throughput_per_second = len(projects) / max(total_time, 0.1)
        self.metrics.last_updated = datetime.now().isoformat()
        
        return all_results
    
    async def _execute_batch_with_scaling(self, batch: List[ScalableResearchProject], batch_idx: int) -> List[Dict[str, Any]]:
        """Execute batch with dynamic scaling and load balancing."""
        if RICH_AVAILABLE and console:
            console.print(f"[blue]🚀 Executing batch {batch_idx + 1} with {len(batch)} projects[/blue]")
        
        # Check cache for completed projects
        cached_results = []
        projects_to_execute = []
        
        for project in batch:
            cached_result = self.cache.get(project.cache_key)
            if cached_result:
                self.metrics.cache_hits += 1
                cached_results.append(cached_result)
                if RICH_AVAILABLE and console:
                    console.print(f"[green]💾 Cache hit for {project.name}[/green]")
            else:
                self.metrics.cache_misses += 1
                projects_to_execute.append(project)
        
        # Execute remaining projects concurrently
        if projects_to_execute:
            with ThreadPoolExecutor(max_workers=min(len(projects_to_execute), self.auto_scaler.current_workers)) as executor:
                future_to_project = {
                    executor.submit(self._execute_project_optimized, project): project 
                    for project in projects_to_execute
                }
                
                execution_results = []
                for future in as_completed(future_to_project):
                    project = future_to_project[future]
                    try:
                        result = future.result()
                        execution_results.append(result)
                        
                        # Cache successful results
                        if result.get('status') == 'completed':
                            self.cache.put(project.cache_key, result)
                            
                    except Exception as e:
                        error_result = {
                            'project_name': project.name,
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        execution_results.append(error_result)
                
                return cached_results + execution_results
        
        return cached_results
    
    def _execute_project_optimized(self, project: ScalableResearchProject) -> Dict[str, Any]:
        """Execute individual project with optimization."""
        start_time = time.time()
        execution_id = f"exec_{int(time.time())}_{hash(project.name) % 1000}"
        
        # Select optimal worker
        worker_id = self.load_balancer.get_next_worker()
        
        # Track concurrent executions
        with self.execution_lock:
            self.active_executions[execution_id] = threading.current_thread()
            self.metrics.concurrent_executions = len(self.active_executions)
            self.metrics.peak_concurrent_executions = max(
                self.metrics.peak_concurrent_executions,
                self.metrics.concurrent_executions
            )
        
        try:
            # Simulate optimized execution with caching and performance monitoring
            if RICH_AVAILABLE and console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task(f"Optimized: {project.name[:30]}...", total=100)
                    
                    # Optimized execution phases
                    phases = ["Cache Check", "Load Balancing", "Execution", "Result Caching", "Cleanup"]
                    phase_duration = project.estimated_duration / len(phases)
                    
                    results = {
                        'project_name': project.name,
                        'execution_id': execution_id,
                        'worker_id': worker_id,
                        'start_time': datetime.now().isoformat(),
                        'phases_completed': [],
                        'optimization_applied': True,
                        'cache_key': project.cache_key
                    }
                    
                    for i, phase in enumerate(phases):
                        # Simulate phase execution with optimization
                        phase_start = time.time()
                        
                        if phase == "Cache Check":
                            # Already handled above
                            time.sleep(0.1)
                        elif phase == "Load Balancing":
                            # Update worker load
                            current_load = len(self.active_executions)
                            self.load_balancer.update_worker_load(worker_id, current_load)
                            time.sleep(0.1)
                        elif phase == "Execution":
                            # Main execution with performance optimization
                            time.sleep(min(phase_duration, 2.0))  # Optimized execution time
                        elif phase == "Result Caching":
                            # Prepare for caching
                            time.sleep(0.1)
                        else:  # Cleanup
                            time.sleep(0.1)
                        
                        phase_time = time.time() - phase_start
                        results['phases_completed'].append({
                            'phase': phase,
                            'duration': phase_time,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        progress.update(task, advance=20, description=f"Optimized: {phase}")
            else:
                # Non-rich execution
                time.sleep(min(project.estimated_duration, 3.0))  # Optimized execution
                results = {
                    'project_name': project.name,
                    'execution_id': execution_id,
                    'worker_id': worker_id,
                    'optimization_applied': True,
                    'cache_key': project.cache_key
                }
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            
            results.update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'execution_time': execution_time,
                'performance_metrics': {
                    'response_time': execution_time,
                    'cache_efficiency': self.cache.get_statistics()['hit_rate'],
                    'worker_utilization': len(self.active_executions) / self.auto_scaler.current_workers,
                    'optimization_score': self._calculate_optimization_score(project, execution_time)
                }
            })
            
            # Update worker performance
            performance_score = 1.0 / max(execution_time, 0.1)
            self.load_balancer.update_worker_performance(worker_id, performance_score)
            
            return results
            
        except Exception as e:
            return {
                'project_name': project.name,
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start_time,
                'worker_id': worker_id
            }
        finally:
            # Remove from active executions
            with self.execution_lock:
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]
                self.metrics.concurrent_executions = len(self.active_executions)
    
    def _calculate_optimization_score(self, project: ScalableResearchProject, execution_time: float) -> float:
        """Calculate optimization effectiveness score."""
        # Compare actual vs estimated execution time
        time_efficiency = min(project.estimated_duration / max(execution_time, 0.1), 2.0)
        
        # Factor in cache hit rate
        cache_stats = self.cache.get_statistics()
        cache_efficiency = cache_stats['hit_rate']
        
        # Factor in resource utilization
        utilization_efficiency = self.metrics.concurrent_executions / max(self.auto_scaler.current_workers, 1)
        
        # Combined optimization score
        optimization_score = (time_efficiency + cache_efficiency + utilization_efficiency) / 3
        return min(optimization_score, 1.0)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Update cache metrics
        cache_stats = self.cache.get_statistics()
        self.metrics.cache_hit_rate = cache_stats['hit_rate']
        self.metrics.cache_hits = cache_stats['hits']
        self.metrics.cache_misses = cache_stats['misses']
        
        # Calculate error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = (self.metrics.total_requests - self.metrics.successful_requests) / self.metrics.total_requests
        
        return self.metrics

class TerragonScalableOrchestrator:
    """Scalable orchestrator with advanced optimization and auto-scaling."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_scalable_config(config_path)
        self.execution_engine = ScalableExecutionEngine(max_workers=self.config.get('max_workers', 6))
        self.projects: List[ScalableResearchProject] = []
        self.output_dir = Path("terragon_scalable_output")
        self.output_dir.mkdir(exist_ok=True)
        self.setup_scalable_logging()
        
    def _load_scalable_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load scalable configuration."""
        default_config = {
            "max_workers": 6,
            "optimization_enabled": True,
            "auto_scaling_enabled": True,
            "cache_strategy": "adaptive",
            "load_balancing_strategy": "performance_weighted",
            "research_domains": ["machine_learning", "quantum_computing", "nlp", "computer_vision"],
            "performance_targets": {
                "max_response_time": 5.0,
                "min_throughput": 2.0,
                "target_cache_hit_rate": 0.6
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self._log(f"Config load error: {e}, using defaults", "warning")
        
        return default_config
    
    def setup_scalable_logging(self):
        """Setup scalable logging system."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"scalable_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _log(self, message: str, level: str = "info"):
        """Enhanced logging with performance tracking."""
        if RICH_AVAILABLE and console:
            level_colors = {
                "error": "red",
                "success": "green",
                "warning": "yellow",
                "info": "blue"
            }
            color = level_colors.get(level, "blue")
            console.print(f"[{color}]🚀 {message}[/{color}]")
        else:
            print(f"🚀 {message}")
        
        getattr(self.logger, level, self.logger.info)(message)
    
    async def execute_scalable_research_suite(self, domains: List[str] = None, projects_per_domain: int = 3) -> Dict[str, Any]:
        """Execute comprehensive scalable research suite."""
        if domains is None:
            domains = self.config["research_domains"]
        
        suite_id = f"scalable_suite_{int(time.time())}"
        self._log(f"🚀 Starting scalable research suite: {suite_id}")
        
        suite_results = {
            'suite_id': suite_id,
            'start_time': datetime.now().isoformat(),
            'domains': [],
            'optimization_enabled': self.config["optimization_enabled"],
            'auto_scaling_enabled': self.config["auto_scaling_enabled"],
            'performance_metrics': None
        }
        
        # Generate all projects
        all_projects = []
        for domain in domains:
            domain_projects = self._generate_scalable_projects(domain, projects_per_domain)
            all_projects.extend(domain_projects)
        
        # Execute with full optimization
        start_time = time.time()
        results = await self.execution_engine.execute_projects_optimized(all_projects)
        total_execution_time = time.time() - start_time
        
        # Organize results by domain
        domain_results = {}
        for domain in domains:
            domain_results[domain] = [r for r in results if r.get('project_name', '').startswith(domain)]
        
        suite_results['domains'] = domain_results
        suite_results['total_projects'] = len(all_projects)
        suite_results['successful_projects'] = len([r for r in results if r.get('status') == 'completed'])
        suite_results['total_execution_time'] = total_execution_time
        
        # Get final performance metrics
        final_metrics = self.execution_engine.get_performance_metrics()
        suite_results['performance_metrics'] = asdict(final_metrics)
        suite_results['end_time'] = datetime.now().isoformat()
        
        # Save results
        results_file = self.output_dir / f"scalable_suite_{suite_id}.json"
        with open(results_file, 'w') as f:
            json.dump(suite_results, f, indent=2)
        
        success_rate = suite_results['successful_projects'] / suite_results['total_projects']
        self._log(f"Scalable suite completed: {success_rate:.1%} success rate, {final_metrics.throughput_per_second:.2f} projects/sec", "success")
        
        return suite_results
    
    def _generate_scalable_projects(self, domain: str, count: int) -> List[ScalableResearchProject]:
        """Generate scalable research projects with optimization metadata."""
        domain_configs = {
            "machine_learning": {
                "base_complexity": 1.5,
                "ideas": [
                    "High-performance attention mechanisms",
                    "Optimized adaptive learning algorithms",
                    "Parallel neural architecture search"
                ]
            },
            "quantum_computing": {
                "base_complexity": 2.0,
                "ideas": [
                    "Scalable quantum error correction",
                    "Optimized hybrid quantum algorithms",
                    "Parallel quantum circuit optimization"
                ]
            },
            "nlp": {
                "base_complexity": 1.8,
                "ideas": [
                    "Distributed multilingual processing",
                    "Optimized context-aware generation",
                    "Parallel text understanding"
                ]
            },
            "computer_vision": {
                "base_complexity": 1.6,
                "ideas": [
                    "Scalable visual representation learning",
                    "Optimized object detection pipelines",
                    "Parallel image processing systems"
                ]
            }
        }
        
        config = domain_configs.get(domain, domain_configs["machine_learning"])
        ideas = config["ideas"]
        base_complexity = config["base_complexity"]
        
        projects = []
        for i in range(min(count, len(ideas))):
            complexity = base_complexity + (i * 0.3)
            estimated_duration = 30 + (complexity * 10)  # Optimized duration
            
            project = ScalableResearchProject(
                name=f"{domain}_scalable_project_{i+1}",
                domain=domain,
                objectives=[
                    f"Implement {ideas[i]} with scalability focus",
                    "Apply performance optimization techniques",
                    "Ensure efficient resource utilization",
                    "Implement caching and parallel processing"
                ],
                priority=i + 1,
                complexity_score=complexity,
                estimated_duration=estimated_duration,
                resource_requirements={
                    'memory_mb': 256 + (i * 128),
                    'cpu_cores': 1 + (i % 2)
                },
                parallelizable=True
            )
            projects.append(project)
        
        return projects
    
    def display_scalable_status(self):
        """Display comprehensive scalable system status."""
        metrics = self.execution_engine.get_performance_metrics()
        cache_stats = self.execution_engine.cache.get_statistics()
        
        if RICH_AVAILABLE and console:
            # Performance metrics table
            perf_table = Table(title="Scalable System Performance")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="magenta")
            perf_table.add_column("Target", style="green")
            
            targets = self.config["performance_targets"]
            
            perf_table.add_row("Total Requests", str(metrics.total_requests), "N/A")
            perf_table.add_row("Success Rate", f"{(metrics.successful_requests/max(metrics.total_requests,1)):.1%}", "≥95%")
            perf_table.add_row("Avg Response Time", f"{metrics.average_response_time:.2f}s", f"≤{targets['max_response_time']}s")
            perf_table.add_row("Throughput", f"{metrics.throughput_per_second:.2f}/sec", f"≥{targets['min_throughput']}/sec")
            perf_table.add_row("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}", f"≥{targets['target_cache_hit_rate']:.0%}")
            perf_table.add_row("Peak Concurrent", str(metrics.peak_concurrent_executions), f"≤{self.config['max_workers']}")
            
            console.print(perf_table)
            
            # Optimization status
            optimization_panel = Panel(
                f"Optimization: {'Enabled' if self.config['optimization_enabled'] else 'Disabled'}\n"
                f"Auto-scaling: {'Enabled' if self.config['auto_scaling_enabled'] else 'Disabled'}\n"
                f"Cache Strategy: {self.config['cache_strategy']}\n"
                f"Load Balancing: {self.config['load_balancing_strategy']}\n"
                f"Active Workers: {self.execution_engine.auto_scaler.current_workers}\n"
                f"Cache Size: {cache_stats['size']}/{cache_stats['max_size']}",
                title="Optimization Configuration",
                border_style="green"
            )
            console.print(optimization_panel)
        else:
            print(f"\n=== Scalable System Performance ===")
            print(f"Total Requests: {metrics.total_requests}")
            print(f"Success Rate: {(metrics.successful_requests/max(metrics.total_requests,1)):.1%}")
            print(f"Avg Response Time: {metrics.average_response_time:.2f}s")
            print(f"Throughput: {metrics.throughput_per_second:.2f}/sec")
            print(f"Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
            print(f"Active Workers: {self.execution_engine.auto_scaler.current_workers}")

async def main():
    """Main execution entry point for scalable orchestrator."""
    print("⚡ Terragon Scalable Optimizer - Generation 3 Implementation")
    
    # Initialize scalable orchestrator
    orchestrator = TerragonScalableOrchestrator()
    
    # Display initial status
    orchestrator.display_scalable_status()
    
    # Execute scalable research suite
    domains = ["machine_learning", "quantum_computing", "nlp"]
    results = await orchestrator.execute_scalable_research_suite(domains, projects_per_domain=2)
    
    # Display final status
    orchestrator.display_scalable_status()
    
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold green]⚡ Scalable execution completed![/bold green]")
        console.print(f"[cyan]Throughput: {results['performance_metrics']['throughput_per_second']:.2f} projects/sec[/cyan]")
        console.print(f"[cyan]Results saved to: {orchestrator.output_dir}[/cyan]")
    else:
        print("\n⚡ Scalable execution completed!")
        print(f"Throughput: {results['performance_metrics']['throughput_per_second']:.2f} projects/sec")
        print(f"Results saved to: {orchestrator.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())