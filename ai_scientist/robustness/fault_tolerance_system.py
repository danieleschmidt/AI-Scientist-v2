"""
Comprehensive Fault Tolerance and Circuit Breaker System

Advanced fault tolerance framework with intelligent circuit breakers, bulkhead patterns,
timeout management, and self-healing capabilities for autonomous scientific research systems.
"""

import time
import asyncio
import threading
import logging
import json
import uuid
import weakref
from typing import (
    Dict, List, Any, Optional, Union, Callable, Type, Tuple, 
    Generic, TypeVar, Protocol, Awaitable
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
import functools
import queue
from collections import defaultdict, deque
import statistics
import random
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(Enum):
    """Types of failures that can trigger circuit breakers."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    SLOW_RESPONSE = "slow_response"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"
    VALIDATION_FAILURE = "validation_failure"
    AUTHENTICATION_FAILURE = "authentication_failure"


class BulkheadStrategy(Enum):
    """Bulkhead isolation strategies."""
    THREAD_POOL = "thread_pool"
    SEMAPHORE = "semaphore"
    RATE_LIMITER = "rate_limiter"
    QUEUE_ISOLATION = "queue_isolation"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    name: str
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    slow_call_threshold: float = 5.0
    slow_call_rate_threshold: float = 0.5
    minimum_throughput: int = 10
    sliding_window_size: int = 100
    permitted_calls_in_half_open_state: int = 3
    max_wait_duration_in_half_open_state: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'failure_threshold': self.failure_threshold,
            'success_threshold': self.success_threshold,
            'timeout_seconds': self.timeout_seconds,
            'slow_call_threshold': self.slow_call_threshold,
            'slow_call_rate_threshold': self.slow_call_rate_threshold,
            'minimum_throughput': self.minimum_throughput,
            'sliding_window_size': self.sliding_window_size,
            'permitted_calls_in_half_open_state': self.permitted_calls_in_half_open_state,
            'max_wait_duration_in_half_open_state': self.max_wait_duration_in_half_open_state
        }


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    success: bool
    duration: float
    timestamp: float
    failure_type: Optional[FailureType] = None
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'duration': self.duration,
            'timestamp': self.timestamp,
            'failure_type': self.failure_type.value if self.failure_type else None,
            'error_message': self.error_message
        }


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker performance."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    state_transition_count: int = 0
    last_state_transition: float = 0.0
    failure_rate: float = 0.0
    slow_call_rate: float = 0.0
    mean_response_time: float = 0.0
    p95_response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'slow_calls': self.slow_calls,
            'current_state': self.current_state.value,
            'state_transition_count': self.state_transition_count,
            'last_state_transition': self.last_state_transition,
            'failure_rate': self.failure_rate,
            'slow_call_rate': self.slow_call_rate,
            'mean_response_time': self.mean_response_time,
            'p95_response_time': self.p95_response_time
        }


class SlidingWindow:
    """Sliding window for tracking call results."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.results: deque = deque(maxlen=window_size)
        self.lock = threading.RLock()
    
    def add_result(self, result: CallResult):
        """Add a call result to the window."""
        with self.lock:
            self.results.append(result)
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Calculate metrics from current window."""
        with self.lock:
            if not self.results:
                return CircuitBreakerMetrics()
            
            total_calls = len(self.results)
            successful_calls = sum(1 for r in self.results if r.success)
            failed_calls = total_calls - successful_calls
            slow_calls = sum(1 for r in self.results if r.duration > 5.0)  # 5s threshold
            
            failure_rate = failed_calls / total_calls if total_calls > 0 else 0.0
            slow_call_rate = slow_calls / total_calls if total_calls > 0 else 0.0
            
            # Response time statistics
            durations = [r.duration for r in self.results]
            mean_response_time = statistics.mean(durations) if durations else 0.0
            
            # Calculate P95
            if len(durations) >= 20:  # Need sufficient data for percentile
                sorted_durations = sorted(durations)
                p95_index = int(0.95 * len(sorted_durations))
                p95_response_time = sorted_durations[p95_index]
            else:
                p95_response_time = max(durations) if durations else 0.0
            
            return CircuitBreakerMetrics(
                total_calls=total_calls,
                successful_calls=successful_calls,
                failed_calls=failed_calls,
                slow_calls=slow_calls,
                failure_rate=failure_rate,
                slow_call_rate=slow_call_rate,
                mean_response_time=mean_response_time,
                p95_response_time=p95_response_time
            )
    
    def clear(self):
        """Clear the sliding window."""
        with self.lock:
            self.results.clear()


class CircuitBreaker:
    """Advanced circuit breaker with configurable thresholds and strategies."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.sliding_window = SlidingWindow(config.sliding_window_size)
        self.state_transition_count = 0
        self.last_state_transition = time.time()
        self.half_open_call_count = 0
        self.lock = threading.RLock()
        
        # Event callbacks
        self.state_change_callbacks: List[Callable] = []
        self.call_callbacks: List[Callable] = []
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self.lock:
            # Check if call is permitted
            if not self._is_call_permitted():
                raise CircuitBreakerOpenException(
                    f"Circuit breaker {self.config.name} is OPEN"
                )
            
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Record successful call
                duration = time.time() - start_time
                call_result = CallResult(
                    success=True,
                    duration=duration,
                    timestamp=start_time
                )
                
                self._record_call_result(call_result)
                self._on_success()
                
                return result
                
            except Exception as error:
                # Record failed call
                duration = time.time() - start_time
                failure_type = self._classify_failure(error, duration)
                
                call_result = CallResult(
                    success=False,
                    duration=duration,
                    timestamp=start_time,
                    failure_type=failure_type,
                    error_message=str(error)
                )
                
                self._record_call_result(call_result)
                self._on_error(call_result)
                
                raise
    
    def _is_call_permitted(self) -> bool:
        """Check if call is permitted based on current state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self.last_state_transition >= self.config.timeout_seconds:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_call_count < self.config.permitted_calls_in_half_open_state
        
        return False
    
    def _record_call_result(self, result: CallResult):
        """Record call result in sliding window."""
        self.sliding_window.add_result(result)
        
        # Notify callbacks
        for callback in self.call_callbacks:
            try:
                callback(self.config.name, result)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_call_count += 1
            
            # Check if we should close the circuit
            metrics = self.sliding_window.get_metrics()
            recent_results = list(self.sliding_window.results)[-self.config.success_threshold:]
            
            if (len(recent_results) >= self.config.success_threshold and
                all(r.success for r in recent_results)):
                self._transition_to_closed()
    
    def _on_error(self, call_result: CallResult):
        """Handle failed call."""
        if self.state == CircuitState.HALF_OPEN:
            # Immediately open on any failure in half-open state
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            metrics = self.sliding_window.get_metrics()
            
            if self._should_open_circuit(metrics):
                self._transition_to_open()
    
    def _should_open_circuit(self, metrics: CircuitBreakerMetrics) -> bool:
        """Determine if circuit should be opened based on metrics."""
        # Need minimum throughput to make decision
        if metrics.total_calls < self.config.minimum_throughput:
            return False
        
        # Check failure rate threshold
        if metrics.failure_rate >= (self.config.failure_threshold / 100.0):
            return True
        
        # Check slow call rate threshold
        if metrics.slow_call_rate >= self.config.slow_call_rate_threshold:
            return True
        
        return False
    
    def _classify_failure(self, error: Exception, duration: float) -> FailureType:
        """Classify the type of failure."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if isinstance(error, TimeoutError) or 'timeout' in error_message:
            return FailureType.TIMEOUT
        elif duration > self.config.slow_call_threshold:
            return FailureType.SLOW_RESPONSE
        elif 'memory' in error_message or 'resource' in error_message:
            return FailureType.RESOURCE_EXHAUSTION
        elif 'connection' in error_message or 'network' in error_message:
            return FailureType.EXTERNAL_SERVICE_FAILURE
        elif 'auth' in error_message or 'permission' in error_message:
            return FailureType.AUTHENTICATION_FAILURE
        elif 'validation' in error_message or 'invalid' in error_message:
            return FailureType.VALIDATION_FAILURE
        else:
            return FailureType.EXCEPTION
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_transition = time.time()
            self.state_transition_count += 1
            self.half_open_call_count = 0
            
            logger.warning(f"Circuit breaker {self.config.name} transitioned to OPEN")
            self._notify_state_change(CircuitState.OPEN)
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.last_state_transition = time.time()
            self.state_transition_count += 1
            self.half_open_call_count = 0
            
            logger.info(f"Circuit breaker {self.config.name} transitioned to HALF_OPEN")
            self._notify_state_change(CircuitState.HALF_OPEN)
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.last_state_transition = time.time()
            self.state_transition_count += 1
            self.half_open_call_count = 0
            
            logger.info(f"Circuit breaker {self.config.name} transitioned to CLOSED")
            self._notify_state_change(CircuitState.CLOSED)
    
    def _notify_state_change(self, new_state: CircuitState):
        """Notify callbacks of state change."""
        for callback in self.state_change_callbacks:
            try:
                callback(self.config.name, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def add_call_callback(self, callback: Callable):
        """Add callback for calls."""
        self.call_callbacks.append(callback)
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        with self.lock:
            metrics = self.sliding_window.get_metrics()
            metrics.current_state = self.state
            metrics.state_transition_count = self.state_transition_count
            metrics.last_state_transition = self.last_state_transition
            return metrics
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.sliding_window.clear()
            self.half_open_call_count = 0
            self.state_transition_count = 0
            self.last_state_transition = time.time()
            
            logger.info(f"Circuit breaker {self.config.name} reset")
    
    def force_open(self):
        """Force circuit breaker to open state."""
        with self.lock:
            self._transition_to_open()
    
    def force_closed(self):
        """Force circuit breaker to closed state."""
        with self.lock:
            self._transition_to_closed()


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BulkheadIsolation:
    """Bulkhead pattern implementation for resource isolation."""
    
    def __init__(self, name: str, strategy: BulkheadStrategy, **config):
        self.name = name
        self.strategy = strategy
        self.config = config
        
        # Initialize based on strategy
        if strategy == BulkheadStrategy.THREAD_POOL:
            max_workers = config.get('max_workers', 10)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif strategy == BulkheadStrategy.SEMAPHORE:
            max_permits = config.get('max_permits', 10)
            self.semaphore = threading.Semaphore(max_permits)
        elif strategy == BulkheadStrategy.RATE_LIMITER:
            max_calls_per_second = config.get('max_calls_per_second', 10)
            self.rate_limiter = RateLimiter(max_calls_per_second)
        elif strategy == BulkheadStrategy.QUEUE_ISOLATION:
            max_queue_size = config.get('max_queue_size', 100)
            self.work_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
            self.workers = []
            self._start_workers(config.get('num_workers', 5))
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'rejected_requests': 0,
            'active_requests': 0,
            'queue_size': 0
        }
        self.lock = threading.RLock()
    
    def execute(self, func: Callable[..., T], *args, timeout: Optional[float] = None, **kwargs) -> T:
        """Execute function with bulkhead isolation."""
        with self.lock:
            self.metrics['total_requests'] += 1
            self.metrics['active_requests'] += 1
        
        try:
            if self.strategy == BulkheadStrategy.THREAD_POOL:
                return self._execute_with_thread_pool(func, args, kwargs, timeout)
            elif self.strategy == BulkheadStrategy.SEMAPHORE:
                return self._execute_with_semaphore(func, args, kwargs, timeout)
            elif self.strategy == BulkheadStrategy.RATE_LIMITER:
                return self._execute_with_rate_limiter(func, args, kwargs, timeout)
            elif self.strategy == BulkheadStrategy.QUEUE_ISOLATION:
                return self._execute_with_queue(func, args, kwargs, timeout)
            else:
                # Direct execution as fallback
                return func(*args, **kwargs)
        
        except Exception as e:
            with self.lock:
                self.metrics['rejected_requests'] += 1
            raise
        finally:
            with self.lock:
                self.metrics['active_requests'] = max(0, self.metrics['active_requests'] - 1)
    
    def _execute_with_thread_pool(self, func: Callable, args: tuple, 
                                 kwargs: dict, timeout: Optional[float]) -> Any:
        """Execute with thread pool isolation."""
        future = self.executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            with self.lock:
                self.metrics['successful_requests'] += 1
            return result
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Execution timed out after {timeout} seconds")
    
    def _execute_with_semaphore(self, func: Callable, args: tuple, 
                               kwargs: dict, timeout: Optional[float]) -> Any:
        """Execute with semaphore isolation."""
        acquired = self.semaphore.acquire(timeout=timeout or 30.0)
        if not acquired:
            raise TimeoutError("Could not acquire semaphore permit")
        
        try:
            result = func(*args, **kwargs)
            with self.lock:
                self.metrics['successful_requests'] += 1
            return result
        finally:
            self.semaphore.release()
    
    def _execute_with_rate_limiter(self, func: Callable, args: tuple, 
                                  kwargs: dict, timeout: Optional[float]) -> Any:
        """Execute with rate limiting."""
        if not self.rate_limiter.acquire(timeout=timeout or 30.0):
            raise TimeoutError("Rate limit exceeded")
        
        result = func(*args, **kwargs)
        with self.lock:
            self.metrics['successful_requests'] += 1
        return result
    
    def _execute_with_queue(self, func: Callable, args: tuple, 
                           kwargs: dict, timeout: Optional[float]) -> Any:
        """Execute with queue isolation."""
        result_queue = queue.Queue()
        
        work_item = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'result_queue': result_queue,
            'timeout': timeout
        }
        
        try:
            self.work_queue.put(work_item, timeout=timeout or 30.0)
            with self.lock:
                self.metrics['queue_size'] = self.work_queue.qsize()
        except queue.Full:
            raise TimeoutError("Work queue is full")
        
        # Wait for result
        try:
            result = result_queue.get(timeout=timeout or 30.0)
            if isinstance(result, Exception):
                raise result
            with self.lock:
                self.metrics['successful_requests'] += 1
            return result
        except queue.Empty:
            raise TimeoutError("Execution timed out")
    
    def _start_workers(self, num_workers: int):
        """Start worker threads for queue isolation."""
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker loop for processing queue items."""
        while True:
            try:
                work_item = self.work_queue.get(timeout=1.0)
                if work_item is None:  # Shutdown signal
                    break
                
                try:
                    result = work_item['func'](*work_item['args'], **work_item['kwargs'])
                    work_item['result_queue'].put(result)
                except Exception as e:
                    work_item['result_queue'].put(e)
                finally:
                    self.work_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in bulkhead worker: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        with self.lock:
            metrics = self.metrics.copy()
            
            if self.strategy == BulkheadStrategy.QUEUE_ISOLATION:
                metrics['queue_size'] = self.work_queue.qsize()
            
            return metrics
    
    def shutdown(self):
        """Shutdown bulkhead resources."""
        if self.strategy == BulkheadStrategy.THREAD_POOL:
            self.executor.shutdown(wait=True)
        elif self.strategy == BulkheadStrategy.QUEUE_ISOLATION:
            # Signal workers to stop
            for _ in self.workers:
                self.work_queue.put(None)
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.tokens = max_calls_per_second
        self.last_refill = time.time()
        self.lock = threading.RLock()
    
    def acquire(self, timeout: float = 0.0) -> bool:
        """Acquire a token for rate limiting."""
        end_time = time.time() + timeout
        
        while time.time() <= end_time:
            with self.lock:
                self._refill_tokens()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            
            if timeout > 0:
                time.sleep(0.01)  # Small delay before retry
            else:
                break
        
        return False
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.max_calls_per_second
        self.tokens = min(self.max_calls_per_second, self.tokens + tokens_to_add)
        self.last_refill = now


class FaultToleranceManager:
    """Central manager for fault tolerance patterns."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.global_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_breaker_trips': 0,
            'bulkhead_rejections': 0
        }
        self.lock = threading.RLock()
        
        # Health monitoring
        self.health_check_interval = 30.0
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.monitoring_enabled = False
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
    
    def create_circuit_breaker(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config)
        
        # Add monitoring callbacks
        circuit_breaker.add_state_change_callback(self._on_circuit_breaker_state_change)
        circuit_breaker.add_call_callback(self._on_circuit_breaker_call)
        
        with self.lock:
            self.circuit_breakers[config.name] = circuit_breaker
        
        logger.info(f"Created circuit breaker: {config.name}")
        return circuit_breaker
    
    def create_bulkhead(self, name: str, strategy: BulkheadStrategy, **config) -> BulkheadIsolation:
        """Create and register a bulkhead."""
        bulkhead = BulkheadIsolation(name, strategy, **config)
        
        with self.lock:
            self.bulkheads[name] = bulkhead
        
        logger.info(f"Created bulkhead: {name} with strategy {strategy.value}")
        return bulkhead
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_bulkhead(self, name: str) -> Optional[BulkheadIsolation]:
        """Get bulkhead by name."""
        return self.bulkheads.get(name)
    
    def execute_with_fault_tolerance(self, name: str, func: Callable[..., T], 
                                   *args, **kwargs) -> T:
        """Execute function with both circuit breaker and bulkhead protection."""
        with self.lock:
            self.global_metrics['total_calls'] += 1
        
        circuit_breaker = self.circuit_breakers.get(name)
        bulkhead = self.bulkheads.get(name)
        
        try:
            if circuit_breaker and bulkhead:
                # Use both circuit breaker and bulkhead
                def protected_call():
                    return bulkhead.execute(func, *args, **kwargs)
                
                result = circuit_breaker.call(protected_call)
            elif circuit_breaker:
                # Use only circuit breaker
                result = circuit_breaker.call(func, *args, **kwargs)
            elif bulkhead:
                # Use only bulkhead
                result = bulkhead.execute(func, *args, **kwargs)
            else:
                # No protection
                result = func(*args, **kwargs)
            
            with self.lock:
                self.global_metrics['successful_calls'] += 1
            
            return result
            
        except Exception as e:
            with self.lock:
                self.global_metrics['failed_calls'] += 1
            
            # Emit event
            self._emit_event('function_failed', {
                'name': name,
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            raise
    
    def _on_circuit_breaker_state_change(self, name: str, new_state: CircuitState):
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            with self.lock:
                self.global_metrics['circuit_breaker_trips'] += 1
        
        self._emit_event('circuit_breaker_state_change', {
            'name': name,
            'new_state': new_state.value
        })
    
    def _on_circuit_breaker_call(self, name: str, call_result: CallResult):
        """Handle circuit breaker calls."""
        self._emit_event('circuit_breaker_call', {
            'name': name,
            'success': call_result.success,
            'duration': call_result.duration,
            'failure_type': call_result.failure_type.value if call_result.failure_type else None
        })
    
    def start_health_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_monitor_thread.start()
        
        logger.info("Started fault tolerance health monitoring")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_enabled = False
        
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped fault tolerance health monitoring")
    
    def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self.monitoring_enabled:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        health_report = self.get_health_report()
        
        # Check for unhealthy components
        unhealthy_circuit_breakers = []
        for name, cb_health in health_report['circuit_breakers'].items():
            if cb_health['health_score'] < 50:
                unhealthy_circuit_breakers.append(name)
        
        unhealthy_bulkheads = []
        for name, bh_health in health_report['bulkheads'].items():
            if bh_health['health_score'] < 50:
                unhealthy_bulkheads.append(name)
        
        # Emit health events
        if unhealthy_circuit_breakers:
            self._emit_event('unhealthy_circuit_breakers', {
                'names': unhealthy_circuit_breakers
            })
        
        if unhealthy_bulkheads:
            self._emit_event('unhealthy_bulkheads', {
                'names': unhealthy_bulkheads
            })
        
        # Overall system health
        overall_health = health_report['overall_health_score']
        if overall_health < 70:
            self._emit_event('system_health_degraded', {
                'health_score': overall_health,
                'recommendations': health_report['recommendations']
            })
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all fault tolerance components."""
        with self.lock:
            metrics = {
                'global_metrics': self.global_metrics.copy(),
                'circuit_breakers': {},
                'bulkheads': {}
            }
            
            for name, cb in self.circuit_breakers.items():
                metrics['circuit_breakers'][name] = cb.get_metrics().to_dict()
            
            for name, bh in self.bulkheads.items():
                metrics['bulkheads'][name] = bh.get_metrics()
            
            return metrics
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        metrics = self.get_comprehensive_metrics()
        
        # Calculate health scores
        circuit_breaker_health = {}
        for name, cb_metrics in metrics['circuit_breakers'].items():
            health_score = self._calculate_circuit_breaker_health(cb_metrics)
            circuit_breaker_health[name] = {
                'health_score': health_score,
                'current_state': cb_metrics['current_state'],
                'failure_rate': cb_metrics['failure_rate'],
                'mean_response_time': cb_metrics['mean_response_time']
            }
        
        bulkhead_health = {}
        for name, bh_metrics in metrics['bulkheads'].items():
            health_score = self._calculate_bulkhead_health(bh_metrics)
            bulkhead_health[name] = {
                'health_score': health_score,
                'active_requests': bh_metrics['active_requests'],
                'rejection_rate': (
                    bh_metrics['rejected_requests'] / 
                    max(1, bh_metrics['total_requests'])
                )
            }
        
        # Overall health score
        all_health_scores = []
        all_health_scores.extend(cb['health_score'] for cb in circuit_breaker_health.values())
        all_health_scores.extend(bh['health_score'] for bh in bulkhead_health.values())
        
        overall_health_score = statistics.mean(all_health_scores) if all_health_scores else 100
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            circuit_breaker_health, bulkhead_health, overall_health_score
        )
        
        return {
            'overall_health_score': overall_health_score,
            'circuit_breakers': circuit_breaker_health,
            'bulkheads': bulkhead_health,
            'recommendations': recommendations,
            'timestamp': time.time()
        }
    
    def _calculate_circuit_breaker_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate health score for circuit breaker."""
        # Base health score
        health_score = 100.0
        
        # Penalize for open state
        if metrics['current_state'] == 'open':
            health_score -= 50
        elif metrics['current_state'] == 'half_open':
            health_score -= 20
        
        # Penalize for high failure rate
        failure_rate = metrics['failure_rate']
        health_score -= failure_rate * 30  # Max 30 point penalty
        
        # Penalize for slow response times
        mean_response_time = metrics['mean_response_time']
        if mean_response_time > 5.0:
            health_score -= min(20, (mean_response_time - 5.0) * 4)
        
        # Penalize for frequent state transitions
        if metrics['state_transition_count'] > 10:
            health_score -= min(15, (metrics['state_transition_count'] - 10) * 1.5)
        
        return max(0, min(100, health_score))
    
    def _calculate_bulkhead_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate health score for bulkhead."""
        health_score = 100.0
        
        # Calculate rejection rate
        total_requests = metrics['total_requests']
        rejected_requests = metrics['rejected_requests']
        
        if total_requests > 0:
            rejection_rate = rejected_requests / total_requests
            health_score -= rejection_rate * 40  # Max 40 point penalty
        
        # Penalize for high active request count (potential overload)
        active_requests = metrics['active_requests']
        if active_requests > 50:  # Arbitrary threshold
            health_score -= min(30, (active_requests - 50) * 0.6)
        
        return max(0, min(100, health_score))
    
    def _generate_health_recommendations(self, cb_health: Dict, bh_health: Dict, 
                                       overall_health: float) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        # Overall system recommendations
        if overall_health < 50:
            recommendations.append("System health is critical - immediate attention required")
        elif overall_health < 70:
            recommendations.append("System health is degraded - review and optimize")
        
        # Circuit breaker recommendations
        for name, health in cb_health.items():
            if health['health_score'] < 50:
                if health['current_state'] == 'open':
                    recommendations.append(f"Circuit breaker {name} is open - investigate underlying service")
                if health['failure_rate'] > 0.5:
                    recommendations.append(f"High failure rate in {name} - review service reliability")
                if health['mean_response_time'] > 10:
                    recommendations.append(f"Slow responses in {name} - optimize service performance")
        
        # Bulkhead recommendations
        for name, health in bh_health.items():
            if health['health_score'] < 50:
                if health['rejection_rate'] > 0.3:
                    recommendations.append(f"High rejection rate in bulkhead {name} - increase capacity")
                if health['active_requests'] > 80:
                    recommendations.append(f"Bulkhead {name} is overloaded - scale resources")
        
        return recommendations
    
    def add_event_callback(self, callback: Callable):
        """Add event callback."""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to callbacks."""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': event_data
        }
        
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def shutdown(self):
        """Shutdown fault tolerance manager."""
        self.stop_health_monitoring()
        
        # Shutdown bulkheads
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()
        
        logger.info("Fault tolerance manager shutdown complete")


# Decorators for easy fault tolerance integration
def circuit_breaker(name: str, **config_kwargs):
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = CircuitBreakerConfig(name=name, **config_kwargs)
        cb = CircuitBreaker(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return cb.call(func, *args, **kwargs)
        
        wrapper._circuit_breaker = cb  # type: ignore
        return wrapper
    return decorator


def bulkhead(name: str, strategy: BulkheadStrategy, **config_kwargs):
    """Decorator for bulkhead isolation."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        bh = BulkheadIsolation(name, strategy, **config_kwargs)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return bh.execute(func, *args, **kwargs)
        
        wrapper._bulkhead = bh  # type: ignore
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create fault tolerance manager
    manager = FaultToleranceManager()
    
    # Event handler
    def event_handler(event):
        print(f"EVENT: {event['type']} - {event['data']}")
    
    manager.add_event_callback(event_handler)
    
    # Create circuit breaker
    cb_config = CircuitBreakerConfig(
        name="test_service",
        failure_threshold=3,
        timeout_seconds=10.0,
        minimum_throughput=5
    )
    circuit_breaker = manager.create_circuit_breaker(cb_config)
    
    # Create bulkhead
    bulkhead = manager.create_bulkhead(
        "test_bulkhead",
        BulkheadStrategy.THREAD_POOL,
        max_workers=5
    )
    
    # Test functions
    def reliable_service():
        """Service that works most of the time."""
        import random
        if random.random() < 0.2:  # 20% failure rate
            raise Exception("Service temporarily unavailable")
        time.sleep(0.1)  # Simulate work
        return "Success"
    
    def unreliable_service():
        """Service that fails frequently."""
        import random
        if random.random() < 0.8:  # 80% failure rate
            raise Exception("Service is down")
        time.sleep(0.1)
        return "Success"
    
    def slow_service():
        """Service that responds slowly."""
        import random
        delay = random.uniform(0.1, 2.0)
        time.sleep(delay)
        if delay > 1.5:
            raise Exception("Service timeout")
        return f"Slow response ({delay:.2f}s)"
    
    # Start health monitoring
    manager.start_health_monitoring()
    
    try:
        print("Testing fault tolerance framework...")
        
        # Test 1: Circuit breaker with reliable service
        print("\n1. Testing circuit breaker with reliable service:")
        for i in range(10):
            try:
                result = manager.execute_with_fault_tolerance("test_service", reliable_service)
                print(f"  Call {i+1}: {result}")
            except Exception as e:
                print(f"  Call {i+1}: Failed - {e}")
        
        # Test 2: Circuit breaker with unreliable service
        print("\n2. Testing circuit breaker with unreliable service:")
        for i in range(15):
            try:
                result = manager.execute_with_fault_tolerance("test_service", unreliable_service)
                print(f"  Call {i+1}: {result}")
            except Exception as e:
                print(f"  Call {i+1}: Failed - {e}")
            time.sleep(0.1)
        
        # Test 3: Bulkhead isolation
        print("\n3. Testing bulkhead isolation:")
        
        def parallel_calls():
            results = []
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(manager.execute_with_fault_tolerance, "test_bulkhead", slow_service)
                    for _ in range(20)
                ]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures, timeout=30)):
                    try:
                        result = future.result()
                        results.append(f"Call {i+1}: {result}")
                    except Exception as e:
                        results.append(f"Call {i+1}: Failed - {e}")
            
            return results
        
        parallel_results = parallel_calls()
        for result in parallel_results[:10]:  # Show first 10 results
            print(f"  {result}")
        
        # Wait for some monitoring cycles
        time.sleep(5)
        
        # Test 4: Health reporting
        print("\n4. Health report:")
        health_report = manager.get_health_report()
        print(f"  Overall health score: {health_report['overall_health_score']:.1f}")
        
        for name, cb_health in health_report['circuit_breakers'].items():
            print(f"  Circuit breaker {name}: {cb_health['health_score']:.1f} ({cb_health['current_state']})")
        
        for name, bh_health in health_report['bulkheads'].items():
            print(f"  Bulkhead {name}: {bh_health['health_score']:.1f}")
        
        if health_report['recommendations']:
            print("  Recommendations:")
            for rec in health_report['recommendations']:
                print(f"    - {rec}")
        
        # Test 5: Comprehensive metrics
        print("\n5. Comprehensive metrics:")
        metrics = manager.get_comprehensive_metrics()
        global_metrics = metrics['global_metrics']
        print(f"  Total calls: {global_metrics['total_calls']}")
        print(f"  Success rate: {global_metrics['successful_calls'] / max(1, global_metrics['total_calls']):.1%}")
        print(f"  Circuit breaker trips: {global_metrics['circuit_breaker_trips']}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.shutdown()
        print("Fault tolerance framework testing completed.")