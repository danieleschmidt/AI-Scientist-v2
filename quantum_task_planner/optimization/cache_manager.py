"""
Quantum-Inspired Cache Manager

High-performance caching system using quantum-inspired algorithms
for optimization results, state vectors, and computation artifacts.
"""

import time
import hashlib
import pickle
import threading
from typing import Any, Dict, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import OrderedDict, defaultdict
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    QUANTUM_PRIORITY = "quantum_priority"  # Quantum-inspired prioritization
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheType(Enum):
    """Types of cached data."""
    OPTIMIZATION_RESULT = "optimization_result"
    QUANTUM_STATE = "quantum_state" 
    TASK_SCHEDULE = "task_schedule"
    COMPUTATION_ARTIFACT = "computation_artifact"
    METRICS = "metrics"


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    quantum_priority: float = 0.5  # Quantum-inspired priority score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    quantum_coherence: float = 0.0  # Quantum-inspired cache coherence
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def avg_entry_size(self) -> float:
        """Calculate average entry size."""
        return self.total_size_bytes / self.entry_count if self.entry_count > 0 else 0.0


class QuantumCacheManager:
    """
    High-performance cache manager with quantum-inspired optimization.
    
    Uses quantum algorithms for cache replacement and priority management
    to optimize cache performance for quantum task planning operations.
    """
    
    def __init__(self,
                 max_size_mb: int = 1024,
                 max_entries: int = 10000,
                 strategy: CacheStrategy = CacheStrategy.QUANTUM_PRIORITY,
                 ttl_seconds: Optional[float] = None):
        """
        Initialize quantum cache manager.
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of cache entries
            strategy: Cache replacement strategy
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.type_indices: Dict[CacheType, set] = defaultdict(set)
        
        # Statistics and monitoring
        self.stats = CacheStatistics()
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Quantum-inspired optimization
        self.quantum_weights: Dict[str, float] = {}
        self.coherence_matrix: np.ndarray = np.eye(4)  # For quantum coherence tracking
        
        # Performance optimization
        self.precompute_enabled = True
        self.adaptive_ttl = {}  # Adaptive TTL per entry type
        
        # Background cleanup
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        logger.info(f"Initialized QuantumCacheManager: {max_size_mb}MB, {max_entries} entries, {strategy.value}")
    
    def put(self, 
            key: str, 
            value: Any, 
            cache_type: CacheType = CacheType.COMPUTATION_ARTIFACT,
            quantum_priority: Optional[float] = None,
            metadata: Dict[str, Any] = None) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cached data
            quantum_priority: Quantum-inspired priority score
            metadata: Additional metadata
            
        Returns:
            True if stored successfully
        """
        with self.lock:
            start_time = time.time()
            
            # Calculate value size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default size estimate
            
            # Check if we need to evict entries
            if not self._ensure_capacity(size_bytes):
                logger.warning(f"Failed to ensure capacity for cache entry {key}")
                return False
            
            # Calculate quantum priority if not provided
            if quantum_priority is None:
                quantum_priority = self._calculate_quantum_priority(key, value, cache_type)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                cache_type=cache_type,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                quantum_priority=quantum_priority,
                metadata=metadata or {}
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
                self.type_indices[old_entry.cache_type].discard(key)
            
            # Store new entry
            self.cache[key] = entry
            self.cache.move_to_end(key)  # Mark as most recently used
            
            # Update indices and statistics
            self.type_indices[cache_type].add(key)
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self.cache)
            
            # Update quantum weights
            self.quantum_weights[key] = quantum_priority
            
            # Record access pattern
            access_time = time.time() - start_time
            self.access_patterns[key].append(access_time)
            if len(self.access_patterns[key]) > 100:
                self.access_patterns[key].pop(0)  # Keep recent history
            
            # Update average access time
            self._update_avg_access_time(access_time)
            
            logger.debug(f"Cached {key}: {size_bytes} bytes, priority={quantum_priority:.3f}")
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            start_time = time.time()
            
            # Check if key exists and is not expired
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access statistics
            entry.update_access()
            self.cache.move_to_end(key)  # Mark as most recently used
            self.stats.hits += 1
            
            # Record access time
            access_time = time.time() - start_time
            self.access_patterns[key].append(access_time)
            self._update_avg_access_time(access_time)
            
            # Update quantum coherence
            self._update_quantum_coherence(key)
            
            logger.debug(f"Cache hit for {key}: {entry.access_count} accesses")
            return entry.value
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if self._is_expired(entry):
                self._remove_entry(key)
                return False
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """
        Remove specific entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed
        """
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                logger.debug(f"Invalidated cache entry: {key}")
                return True
            return False
    
    def invalidate_type(self, cache_type: CacheType) -> int:
        """
        Remove all entries of specific type.
        
        Args:
            cache_type: Type of entries to remove
            
        Returns:
            Number of entries removed
        """
        with self.lock:
            keys_to_remove = list(self.type_indices[cache_type])
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries of type {cache_type.value}")
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            entry_count = len(self.cache)
            self.cache.clear()
            self.type_indices.clear()
            self.quantum_weights.clear()
            self.access_patterns.clear()
            
            # Reset statistics
            self.stats = CacheStatistics()
            
            logger.info(f"Cleared cache: {entry_count} entries removed")
    
    def _ensure_capacity(self, required_size: int) -> bool:
        """Ensure cache has capacity for new entry."""
        # Check entry count limit
        while len(self.cache) >= self.max_entries:
            if not self._evict_entry():
                return False
        
        # Check size limit
        while self.stats.total_size_bytes + required_size > self.max_size_bytes:
            if not self._evict_entry():
                return False
        
        return True
    
    def _evict_entry(self) -> bool:
        """Evict one cache entry based on strategy."""
        if not self.cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = next(iter(self.cache))  # Oldest entry
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = min(self.cache.keys(), 
                              key=lambda k: self.cache[k].access_count)
        elif self.strategy == CacheStrategy.QUANTUM_PRIORITY:
            key_to_evict = self._quantum_eviction_selection()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction_selection()
        else:
            key_to_evict = next(iter(self.cache))  # Fallback to LRU
        
        self._remove_entry(key_to_evict)
        self.stats.evictions += 1
        
        logger.debug(f"Evicted cache entry: {key_to_evict}")
        return True
    
    def _quantum_eviction_selection(self) -> str:
        """Select entry for eviction using quantum-inspired algorithm."""
        if not self.cache:
            return ""
        
        # Calculate quantum scores for all entries
        quantum_scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Factors: recency, frequency, quantum priority, size
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count / 100.0  # Normalize
            priority_score = entry.quantum_priority
            size_penalty = entry.size_bytes / self.max_size_bytes
            
            # Quantum-inspired combination using superposition principles
            quantum_score = (
                0.3 * recency_score +
                0.3 * frequency_score +
                0.2 * priority_score -
                0.2 * size_penalty
            )
            
            # Add quantum coherence factor
            coherence_factor = self._get_coherence_factor(key)
            quantum_score *= (1.0 + coherence_factor)
            
            quantum_scores[key] = quantum_score
        
        # Select entry with lowest quantum score for eviction
        return min(quantum_scores.keys(), key=lambda k: quantum_scores[k])
    
    def _adaptive_eviction_selection(self) -> str:
        """Adaptive eviction based on access patterns."""
        if not self.cache:
            return ""
        
        # Analyze access patterns and adapt strategy
        recent_hits = sum(1 for pattern in self.access_patterns.values() 
                         if pattern and time.time() - pattern[-1] < 300)  # 5 minutes
        
        if recent_hits > len(self.cache) * 0.5:
            # High hit rate - use LFU
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].access_count)
        else:
            # Low hit rate - use quantum priority
            return self._quantum_eviction_selection()
    
    def _calculate_quantum_priority(self, key: str, value: Any, cache_type: CacheType) -> float:
        """Calculate quantum-inspired priority score."""
        priority = 0.5  # Base priority
        
        # Type-based priority
        type_priorities = {
            CacheType.OPTIMIZATION_RESULT: 0.9,
            CacheType.QUANTUM_STATE: 0.8,
            CacheType.TASK_SCHEDULE: 0.7,
            CacheType.COMPUTATION_ARTIFACT: 0.6,
            CacheType.METRICS: 0.3
        }
        priority *= type_priorities.get(cache_type, 0.5)
        
        # Size factor (smaller items get higher priority)
        try:
            size_bytes = len(pickle.dumps(value))
            size_factor = 1.0 - min(size_bytes / (1024 * 1024), 0.5)  # Max 50% penalty
            priority *= size_factor
        except Exception:
            pass  # Use base priority
        
        # Key-based heuristics
        if "optimization" in key.lower():
            priority *= 1.2
        elif "quantum" in key.lower():
            priority *= 1.1
        elif "temp" in key.lower() or "cache" in key.lower():
            priority *= 0.8
        
        # Ensure priority is in [0, 1] range
        return max(0.0, min(1.0, priority))
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        # Check adaptive TTL first
        adaptive_ttl = self.adaptive_ttl.get(entry.cache_type.value)
        if adaptive_ttl is not None:
            return time.time() - entry.created_at > adaptive_ttl
        
        # Use global TTL
        return time.time() - entry.created_at > self.ttl_seconds
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update statistics."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self.cache)
            
            # Update type index
            self.type_indices[entry.cache_type].discard(key)
            
            # Remove quantum weight and access pattern
            self.quantum_weights.pop(key, None)
            self.access_patterns.pop(key, None)
    
    def _update_avg_access_time(self, access_time: float) -> None:
        """Update running average access time."""
        alpha = 0.1  # Smoothing factor
        if self.stats.avg_access_time == 0.0:
            self.stats.avg_access_time = access_time
        else:
            self.stats.avg_access_time = (
                alpha * access_time + 
                (1 - alpha) * self.stats.avg_access_time
            )
    
    def _update_quantum_coherence(self, accessed_key: str) -> None:
        """Update quantum coherence measure based on access patterns."""
        # Simplified quantum coherence calculation
        # In practice, this would involve more sophisticated quantum mechanics
        
        if len(self.access_patterns) < 2:
            self.stats.quantum_coherence = 1.0
            return
        
        # Calculate coherence based on access pattern correlations
        recent_accesses = [
            len(pattern) for pattern in self.access_patterns.values()
            if pattern and time.time() - pattern[-1] < 3600  # Last hour
        ]
        
        if len(recent_accesses) < 2:
            self.stats.quantum_coherence = 1.0
            return
        
        # Coherence as inverse of variance in access patterns
        mean_accesses = np.mean(recent_accesses)
        if mean_accesses > 0:
            variance = np.var(recent_accesses) / mean_accesses
            coherence = 1.0 / (1.0 + variance)
            
            # Smooth update
            alpha = 0.1
            self.stats.quantum_coherence = (
                alpha * coherence + 
                (1 - alpha) * self.stats.quantum_coherence
            )
    
    def _get_coherence_factor(self, key: str) -> float:
        """Get quantum coherence factor for specific key."""
        # Check if key has high correlation with other frequently accessed keys
        if key not in self.access_patterns:
            return 0.0
        
        key_pattern = self.access_patterns[key]
        if not key_pattern:
            return 0.0
        
        # Calculate correlation with other access patterns
        correlations = []
        for other_key, other_pattern in self.access_patterns.items():
            if other_key != key and other_pattern:
                # Simple correlation based on access frequency similarity
                freq_diff = abs(len(key_pattern) - len(other_pattern))
                max_freq = max(len(key_pattern), len(other_pattern))
                if max_freq > 0:
                    correlation = 1.0 - freq_diff / max_freq
                    correlations.append(correlation)
        
        # Return average correlation as coherence factor
        return np.mean(correlations) if correlations else 0.0
    
    def periodic_cleanup(self) -> int:
        """Perform periodic cache cleanup and optimization."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return 0
        
        with self.lock:
            removed_count = 0
            
            # Remove expired entries
            keys_to_remove = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
                removed_count += 1
            
            # Optimize adaptive TTLs
            self._optimize_adaptive_ttls()
            
            # Update quantum coherence matrix
            self._update_coherence_matrix()
            
            self.last_cleanup = current_time
            
            if removed_count > 0:
                logger.info(f"Cache cleanup: removed {removed_count} expired entries")
            
            return removed_count
    
    def _optimize_adaptive_ttls(self) -> None:
        """Optimize TTL values based on access patterns."""
        for cache_type in CacheType:
            type_keys = self.type_indices[cache_type]
            if not type_keys:
                continue
            
            # Analyze access patterns for this type
            access_intervals = []
            for key in type_keys:
                if key in self.access_patterns and len(self.access_patterns[key]) > 1:
                    pattern = self.access_patterns[key]
                    intervals = [pattern[i] - pattern[i-1] for i in range(1, len(pattern))]
                    access_intervals.extend(intervals)
            
            if access_intervals:
                # Set adaptive TTL based on access patterns
                median_interval = np.median(access_intervals)
                adaptive_ttl = max(median_interval * 2, 300)  # At least 5 minutes
                adaptive_ttl = min(adaptive_ttl, 86400)  # At most 24 hours
                
                self.adaptive_ttl[cache_type.value] = adaptive_ttl
    
    def _update_coherence_matrix(self) -> None:
        """Update quantum coherence matrix for cache optimization."""
        # Simplified coherence matrix update
        # Would involve more complex quantum mechanics in practice
        
        n = len(CacheType)
        new_matrix = np.eye(n)
        
        # Calculate cross-correlations between cache types
        for i, type1 in enumerate(CacheType):
            for j, type2 in enumerate(CacheType):
                if i != j:
                    keys1 = self.type_indices[type1]
                    keys2 = self.type_indices[type2]
                    
                    # Simple correlation based on concurrent accesses
                    correlation = 0.0
                    if keys1 and keys2:
                        # Count concurrent accesses in last hour
                        recent_time = time.time() - 3600
                        
                        accesses1 = sum(
                            1 for key in keys1 
                            if key in self.access_patterns and self.access_patterns[key] 
                            and any(t > recent_time for t in self.access_patterns[key])
                        )
                        
                        accesses2 = sum(
                            1 for key in keys2 
                            if key in self.access_patterns and self.access_patterns[key] 
                            and any(t > recent_time for t in self.access_patterns[key])
                        )
                        
                        if accesses1 > 0 and accesses2 > 0:
                            correlation = min(accesses1, accesses2) / max(accesses1, accesses2)
                    
                    new_matrix[i, j] = correlation
        
        # Smooth update
        alpha = 0.1
        self.coherence_matrix = alpha * new_matrix + (1 - alpha) * self.coherence_matrix
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            # Perform cleanup to get accurate stats
            self.periodic_cleanup()
            
            stats_dict = {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'evictions': self.stats.evictions,
                'entry_count': self.stats.entry_count,
                'total_size_mb': self.stats.total_size_bytes / (1024 * 1024),
                'avg_entry_size_kb': self.stats.avg_entry_size / 1024,
                'avg_access_time_ms': self.stats.avg_access_time * 1000,
                'quantum_coherence': self.stats.quantum_coherence,
                'cache_utilization': self.stats.total_size_bytes / self.max_size_bytes,
                'strategy': self.strategy.value
            }
            
            # Add type-specific statistics
            type_stats = {}
            for cache_type in CacheType:
                type_keys = self.type_indices[cache_type]
                if type_keys:
                    type_entries = [self.cache[key] for key in type_keys if key in self.cache]
                    type_stats[cache_type.value] = {
                        'count': len(type_entries),
                        'total_size_mb': sum(e.size_bytes for e in type_entries) / (1024 * 1024),
                        'avg_access_count': np.mean([e.access_count for e in type_entries]) if type_entries else 0,
                        'adaptive_ttl': self.adaptive_ttl.get(cache_type.value)
                    }
            
            stats_dict['type_statistics'] = type_stats
            
            return stats_dict
    
    def precompute_common_queries(self, query_patterns: List[str]) -> int:
        """Precompute and cache results for common query patterns."""
        if not self.precompute_enabled:
            return 0
        
        precomputed_count = 0
        
        for pattern in query_patterns:
            try:
                # Generate cache key for pattern
                cache_key = f"precomputed_{hashlib.md5(pattern.encode()).hexdigest()}"
                
                if not self.exists(cache_key):
                    # This is a placeholder - actual implementation would
                    # compute the result based on the pattern
                    result = {"pattern": pattern, "precomputed": True}
                    
                    self.put(
                        key=cache_key,
                        value=result,
                        cache_type=CacheType.COMPUTATION_ARTIFACT,
                        quantum_priority=0.8  # High priority for precomputed results
                    )
                    
                    precomputed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to precompute pattern {pattern}: {e}")
        
        logger.info(f"Precomputed {precomputed_count} query patterns")
        return precomputed_count