#!/usr/bin/env python3
"""
Performance Optimization Framework for AI Scientist v2

Advanced performance optimization with intelligent caching, resource pooling,
predictive scaling, and adaptive optimization strategies.
"""

import os
import json
import time
import logging
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import queue
import pickle
import weakref
from pathlib import Path
import concurrent.futures
from collections import defaultdict, OrderedDict
import statistics

logger = logging.getLogger(__name__)

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0

class IntelligentCache:
    """Advanced caching system with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 1024, 
                 default_ttl: float = 3600, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.policy = policy
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._access_frequency = defaultdict(int)  # For LFU
        self._current_memory = 0
        self._lock = threading.RLock()
        
        # Performance tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Adaptive policy learning
        self._policy_performance = {policy: [] for policy in CachePolicy}
        self._current_policy = policy
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._access_frequency[key] += 1
            
            # Update access order for LRU
            if key in self._access_order:
                self._access_order.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value into cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default size if serialization fails
            
            # Check if we need to make space
            while (len(self._cache) >= self.max_size or 
                   self._current_memory + size_bytes > self.max_memory_bytes):
                if not self._evict_one():
                    return False  # Cannot evict, cache full
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._cache[key] = entry
            self._access_order[key] = True
            self._access_frequency[key] = 1
            self._current_memory += size_bytes
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            self._current_memory = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._current_memory,
                'max_memory_bytes': self.max_memory_bytes,
                'hit_rate': hit_rate,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'policy': self._current_policy.value,
                'memory_efficiency': self._current_memory / self.max_memory_bytes,
                'avg_entry_size': self._current_memory / len(self._cache) if self._cache else 0
            }
    
    def _evict_one(self) -> bool:
        """Evict one entry based on current policy."""
        if not self._cache:
            return False
        
        if self._current_policy == CachePolicy.LRU:
            key = next(iter(self._access_order))
        elif self._current_policy == CachePolicy.LFU:
            key = min(self._access_frequency.keys(), key=lambda k: self._access_frequency[k])
        elif self._current_policy == CachePolicy.FIFO:
            key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        elif self._current_policy == CachePolicy.TTL:
            # Evict expired entries first
            expired_keys = [k for k, e in self._cache.items() if self._is_expired(e)]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = next(iter(self._access_order))  # Fallback to LRU
        else:  # ADAPTIVE
            key = self._adaptive_evict()
        
        self._remove_entry(key)
        self._evictions += 1
        return True
    
    def _adaptive_evict(self) -> str:
        """Adaptive eviction based on access patterns."""
        # Simple heuristic: combine LRU and LFU
        scores = {}
        current_time = datetime.now()
        
        for key, entry in self._cache.items():
            # Time factor (older = higher score for eviction)
            time_factor = (current_time - entry.last_accessed).total_seconds() / 3600
            
            # Frequency factor (less frequent = higher score for eviction)
            freq_factor = 1.0 / max(entry.access_count, 1)
            
            # Size factor (larger = higher score for eviction)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            
            scores[key] = time_factor + freq_factor + size_factor
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str):
        """Remove entry from all tracking structures."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
        
        self._access_order.pop(key, None)
        self._access_frequency.pop(key, None)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self._lock:
                    expired_keys = [k for k, e in self._cache.items() if self._is_expired(e)]
                    for key in expired_keys:
                        self._remove_entry(key)
                        self._evictions += 1
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

class ResourcePool:
    """Resource pooling for expensive objects."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 10, 
                 max_idle_time: float = 300):
        self.factory = factory
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        
        self._pool = queue.Queue(maxsize=max_size)
        self._active_resources = weakref.WeakSet()
        self._created_count = 0
        self._lock = threading.Lock()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def get_resource(self):
        """Get a resource from the pool."""
        try:
            # Try to get from pool first
            resource = self._pool.get_nowait()
            logger.debug("Resource retrieved from pool")
            return resource
        except queue.Empty:
            # Create new resource if under limit
            with self._lock:
                if self._created_count < self.max_size:
                    resource = self.factory()
                    self._created_count += 1
                    self._active_resources.add(resource)
                    logger.debug(f"Created new resource ({self._created_count}/{self.max_size})")
                    return resource
                else:
                    # Wait for resource to become available
                    logger.debug("Waiting for resource from pool")
                    return self._pool.get()
    
    def return_resource(self, resource):
        """Return a resource to the pool."""
        try:
            self._pool.put_nowait(resource)
            logger.debug("Resource returned to pool")
        except queue.Full:
            # Pool is full, let resource be garbage collected
            with self._lock:
                self._created_count = max(0, self._created_count - 1)
            logger.debug("Pool full, resource discarded")
    
    def _cleanup_loop(self):
        """Cleanup idle resources."""
        while True:
            try:
                time.sleep(self.max_idle_time)
                
                # Clean up idle resources (simplified)
                try:
                    while True:
                        resource = self._pool.get_nowait()
                        # Resource would be cleaned up by garbage collection
                        with self._lock:
                            self._created_count = max(0, self._created_count - 1)
                except queue.Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"Resource pool cleanup error: {e}")

class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, window_size: int = 100, prediction_horizon: int = 10):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        self._metrics_history: List[PerformanceMetrics] = []
        self._predictions: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Pattern recognition
        self._hourly_patterns = defaultdict(list)
        self._daily_patterns = defaultdict(list)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for prediction."""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # Maintain window size
            if len(self._metrics_history) > self.window_size:
                self._metrics_history = self._metrics_history[-self.window_size:]
            
            # Update patterns
            now = datetime.now()
            hour_key = now.hour
            day_key = now.weekday()
            
            self._hourly_patterns[hour_key].append(metrics.cpu_usage)
            self._daily_patterns[day_key].append(metrics.cpu_usage)
            
            # Generate predictions
            self._update_predictions()
    
    def get_prediction(self, metric_name: str) -> Optional[float]:
        """Get prediction for a specific metric."""
        with self._lock:
            return self._predictions.get(metric_name)
    
    def should_scale_up(self, current_metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale up."""
        cpu_prediction = self.get_prediction('cpu_usage')
        memory_prediction = self.get_prediction('memory_usage')
        
        # Scale up if predicted usage is high
        if cpu_prediction and cpu_prediction > 80:
            return True
        if memory_prediction and memory_prediction > 85:
            return True
        
        # Scale up if current response time is high
        if current_metrics.p95_response_time > 1000:  # > 1 second
            return True
        
        return False
    
    def should_scale_down(self, current_metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale down."""
        cpu_prediction = self.get_prediction('cpu_usage')
        
        # Scale down if predicted usage is consistently low
        if cpu_prediction and cpu_prediction < 30:
            # Check recent history for sustained low usage
            recent_cpu = [m.cpu_usage for m in self._metrics_history[-10:]]
            if recent_cpu and statistics.mean(recent_cpu) < 25:
                return True
        
        return False
    
    def _update_predictions(self):
        """Update predictions based on current data."""
        if len(self._metrics_history) < 10:
            return
        
        # Simple trend-based prediction
        recent_metrics = self._metrics_history[-10:]
        
        # CPU usage prediction
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = self._calculate_trend(cpu_values)
        self._predictions['cpu_usage'] = max(0, min(100, cpu_values[-1] + cpu_trend))
        
        # Memory usage prediction
        memory_values = [m.memory_usage for m in recent_metrics]
        memory_trend = self._calculate_trend(memory_values)
        self._predictions['memory_usage'] = max(0, min(100, memory_values[-1] + memory_trend))
        
        # Response time prediction
        response_values = [m.avg_response_time for m in recent_metrics]
        response_trend = self._calculate_trend(response_values)
        self._predictions['response_time'] = max(0, response_values[-1] + response_trend)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) from values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x_val ** 2 for x_val in x)
        
        # Slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope * self.prediction_horizon

class AdaptiveOptimizer:
    """Adaptive performance optimizer that learns from system behavior."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.cache = IntelligentCache()
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.scaler = PredictiveScaler()
        
        # Optimization history
        self._optimization_history: List[Dict[str, Any]] = []
        self._performance_baseline: Optional[PerformanceMetrics] = None
        
        # Configuration
        self._config = {
            'cache_size_multiplier': 1.0,
            'pool_size_multiplier': 1.0,
            'aggressive_caching': False,
            'predictive_scaling': True,
            'auto_tuning': True
        }
        
    def optimize_for_workload(self, workload_type: str, historical_metrics: List[PerformanceMetrics]):
        """Optimize system for specific workload patterns."""
        logger.info(f"Optimizing for workload: {workload_type}")
        
        # Analyze workload patterns
        avg_cpu = statistics.mean([m.cpu_usage for m in historical_metrics])
        avg_memory = statistics.mean([m.memory_usage for m in historical_metrics])
        avg_response_time = statistics.mean([m.avg_response_time for m in historical_metrics])
        
        # Adjust cache settings
        if workload_type == "cpu_intensive":
            self.cache.max_size = int(self.cache.max_size * 1.5)
            self._config['aggressive_caching'] = True
        elif workload_type == "memory_intensive":
            self.cache.max_memory_bytes = int(self.cache.max_memory_bytes * 0.8)
        elif workload_type == "io_intensive":
            self.cache.default_ttl = 7200  # Longer TTL for I/O heavy workloads
        
        # Adjust resource pools
        if avg_response_time > 500:  # High latency
            for pool_name, pool in self.resource_pools.items():
                pool.max_size = int(pool.max_size * 1.3)
        
        logger.info(f"Optimization applied - Cache size: {self.cache.max_size}, "
                   f"Aggressive caching: {self._config['aggressive_caching']}")
    
    def auto_tune(self, current_metrics: PerformanceMetrics):
        """Automatically tune system based on current performance."""
        if not self._config['auto_tuning']:
            return
        
        # Record metrics for prediction
        self.scaler.record_metrics(current_metrics)
        
        # Baseline establishment
        if self._performance_baseline is None:
            self._performance_baseline = current_metrics
            return
        
        # Performance comparison
        performance_change = self._calculate_performance_change(current_metrics)
        
        # Auto-tuning decisions
        if performance_change < -10:  # Performance degraded
            self._apply_conservative_optimizations()
        elif performance_change > 10:  # Performance improved
            self._apply_aggressive_optimizations()
        
        # Predictive scaling
        if self._config['predictive_scaling']:
            if self.scaler.should_scale_up(current_metrics):
                self._scale_up_resources()
            elif self.scaler.should_scale_down(current_metrics):
                self._scale_down_resources()
    
    def _calculate_performance_change(self, current: PerformanceMetrics) -> float:
        """Calculate performance change percentage."""
        if self._performance_baseline is None:
            return 0.0
        
        baseline = self._performance_baseline
        
        # Weighted performance score
        response_time_change = ((baseline.avg_response_time - current.avg_response_time) / 
                               baseline.avg_response_time * 100)
        throughput_change = ((current.throughput - baseline.throughput) / 
                            baseline.throughput * 100)
        error_rate_change = ((baseline.error_rate - current.error_rate) / 
                            max(baseline.error_rate, 0.01) * 100)
        
        # Weighted average
        performance_change = (response_time_change * 0.4 + 
                            throughput_change * 0.4 + 
                            error_rate_change * 0.2)
        
        return performance_change
    
    def _apply_conservative_optimizations(self):
        """Apply conservative optimizations when performance degrades."""
        logger.info("Applying conservative optimizations")
        
        # Reduce cache size to free memory
        self.cache.max_size = int(self.cache.max_size * 0.9)
        self.cache.policy = CachePolicy.LRU  # More predictable eviction
        
        # Reduce resource pool sizes
        for pool in self.resource_pools.values():
            pool.max_size = max(1, int(pool.max_size * 0.8))
    
    def _apply_aggressive_optimizations(self):
        """Apply aggressive optimizations when performance is good."""
        logger.info("Applying aggressive optimizations")
        
        # Increase cache size
        self.cache.max_size = int(self.cache.max_size * 1.1)
        self.cache.policy = CachePolicy.ADAPTIVE  # More intelligent eviction
        
        # Increase resource pool sizes
        for pool in self.resource_pools.values():
            pool.max_size = int(pool.max_size * 1.2)
    
    def _scale_up_resources(self):
        """Scale up system resources."""
        logger.info("Scaling up resources based on predictions")
        
        # Increase cache capacity
        self.cache.max_size = int(self.cache.max_size * 1.2)
        self.cache.max_memory_bytes = int(self.cache.max_memory_bytes * 1.1)
        
        # Increase resource pools
        for pool in self.resource_pools.values():
            pool.max_size = int(pool.max_size * 1.3)
    
    def _scale_down_resources(self):
        """Scale down system resources."""
        logger.info("Scaling down resources based on predictions")
        
        # Reduce cache capacity
        self.cache.max_size = max(100, int(self.cache.max_size * 0.8))
        
        # Reduce resource pools
        for pool in self.resource_pools.values():
            pool.max_size = max(1, int(pool.max_size * 0.7))
    
    def create_resource_pool(self, name: str, factory: Callable[[], Any], 
                           max_size: int = 10) -> ResourcePool:
        """Create a new resource pool."""
        pool = ResourcePool(factory, max_size)
        self.resource_pools[name] = pool
        return pool
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimizations."""
        return {
            'strategy': self.strategy.value,
            'cache_stats': self.cache.get_statistics(),
            'resource_pools': {
                name: {'max_size': pool.max_size, 'created_count': pool._created_count}
                for name, pool in self.resource_pools.items()
            },
            'config': self._config.copy(),
            'predictions': self.scaler._predictions.copy(),
            'optimization_count': len(self._optimization_history)
        }

# Global optimizer instance
performance_optimizer = AdaptiveOptimizer()

def initialize_performance_optimization(strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
    """Initialize the performance optimization system."""
    global performance_optimizer
    performance_optimizer.strategy = strategy
    logger.info(f"Performance optimization initialized with strategy: {strategy.value}")

def get_intelligent_cache() -> IntelligentCache:
    """Get the global intelligent cache instance."""
    return performance_optimizer.cache

def create_resource_pool(name: str, factory: Callable[[], Any], max_size: int = 10) -> ResourcePool:
    """Create a named resource pool."""
    return performance_optimizer.create_resource_pool(name, factory, max_size)

def auto_tune_performance(current_metrics: PerformanceMetrics):
    """Auto-tune system performance based on current metrics."""
    performance_optimizer.auto_tune(current_metrics)