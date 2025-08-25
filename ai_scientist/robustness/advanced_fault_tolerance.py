#!/usr/bin/env python3
"""
Advanced Fault Tolerance System - Generation 2 Enhancement
==========================================================

Comprehensive fault tolerance and error recovery system for robust AI research execution.
Includes circuit breakers, retry mechanisms, graceful degradation, and automatic recovery.

Key Features:
- Multi-level circuit breakers with intelligent state management
- Exponential backoff with jitter for retry mechanisms
- Graceful degradation with fallback strategies
- Automatic error classification and handling
- Resource leak detection and cleanup
- Distributed system failure handling

Author: AI Scientist v2 - Terragon Labs (Generation 2)
License: MIT
"""

import asyncio
import logging
import time
import threading
import weakref
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import psutil
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from contextlib import contextmanager, asynccontextmanager
import functools
import random
import subprocess
import os

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(Enum):
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    API_LIMIT = "api_limit"
    NETWORK_ERROR = "network_error"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    DISK_ERROR = "disk_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ABORT = "abort"
    RESTART = "restart"
    SCALE_DOWN = "scale_down"


@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    failure_type: FailureType
    timestamp: float
    error_message: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes to close from half-open
    timeout: float = 30.0  # Operation timeout
    monitoring_period: float = 300.0  # Window for failure counting


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0


class CircuitBreaker:
    """
    Circuit breaker implementation with intelligent state management.
    
    Prevents cascade failures by monitoring service health and temporarily
    blocking requests to failing services.
    """
    
    def __init__(self, 
                 name: str,
                 config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_history: List[FailureRecord] = []
        
        # Threading
        self._lock = threading.RLock()
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.async_call(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                raise ValueError("Use async_call for coroutine functions")
            
            result = self._execute_with_timeout(func, args, kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.timeout)
            except TimeoutError:
                # Cancel the future
                future.cancel()
                raise CircuitBreakerTimeoutError(f"Operation timed out after {self.config.timeout}s")
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Record failure
            failure_record = FailureRecord(
                failure_type=self._classify_error(error),
                timestamp=self.last_failure_time,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                context={'circuit_breaker': self.name}
            )
            self.failure_history.append(failure_record)
            
            # Keep failure history manageable
            if len(self.failure_history) > 100:
                self.failure_history = self.failure_history[-50:]
            
            # Transition to OPEN if threshold exceeded
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN after {self.failure_count} failures")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed during HALF_OPEN, back to OPEN")
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for better handling."""
        error_str = str(error).lower()
        
        if isinstance(error, TimeoutError) or 'timeout' in error_str:
            return FailureType.TIMEOUT
        elif isinstance(error, MemoryError) or 'memory' in error_str:
            return FailureType.MEMORY_ERROR
        elif isinstance(error, OSError) or 'disk' in error_str or 'space' in error_str:
            return FailureType.DISK_ERROR
        elif isinstance(error, PermissionError) or 'permission' in error_str:
            return FailureType.PERMISSION_ERROR
        elif 'network' in error_str or 'connection' in error_str:
            return FailureType.NETWORK_ERROR
        elif 'rate limit' in error_str or 'quota' in error_str:
            return FailureType.API_LIMIT
        elif 'import' in error_str or 'module' in error_str:
            return FailureType.DEPENDENCY_ERROR
        else:
            return FailureType.UNKNOWN
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'recent_failures': len([f for f in self.failure_history 
                                      if time.time() - f.timestamp < self.config.monitoring_period])
            }


class RetryMechanism:
    """
    Advanced retry mechanism with exponential backoff and jitter.
    
    Provides intelligent retry strategies for transient failures.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry capability to functions."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.async_execute(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain errors
                if not self._should_retry(e):
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        raise last_exception
    
    async def async_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain errors
                if not self._should_retry(e):
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        raise last_exception
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if error should be retried."""
        # Don't retry on certain permanent errors
        non_retryable = [
            ValueError,
            TypeError,
            AttributeError,
            PermissionError,
            FileNotFoundError
        ]
        
        if any(isinstance(error, exc_type) for exc_type in non_retryable):
            return False
        
        # Check error message for non-retryable patterns
        error_str = str(error).lower()
        non_retryable_patterns = [
            'invalid argument',
            'permission denied',
            'not found',
            'unauthorized',
            'forbidden'
        ]
        
        if any(pattern in error_str for pattern in non_retryable_patterns):
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        delay *= self.config.backoff_factor
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter
        
        return max(0.1, delay)  # Minimum delay


class ResourceLeakDetector:
    """
    Detects and reports resource leaks in the system.
    
    Monitors memory usage, file handles, threads, and other resources
    to identify potential leaks.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 60.0,
                 memory_threshold_mb: float = 1000.0,
                 handle_threshold: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.handle_threshold = handle_threshold
        
        # Tracking
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.initial_handles = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
        
        # Resource tracking
        self._tracked_resources: Dict[str, Any] = {}
        self._resource_history: List[Dict[str, Any]] = []
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info("ResourceLeakDetector initialized")
    
    def start_monitoring(self):
        """Start resource monitoring thread."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                # Collect resource metrics
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                handles = 0
                if hasattr(self.process, 'num_fds'):
                    handles = self.process.num_fds()
                elif hasattr(self.process, 'num_handles'):
                    handles = self.process.num_handles()
                
                threads = self.process.num_threads()
                
                # Check for leaks
                metrics = {
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'handles': handles,
                    'threads': threads,
                    'memory_growth': memory_mb - self.initial_memory,
                    'handle_growth': handles - self.initial_handles
                }
                
                self._resource_history.append(metrics)
                
                # Keep history manageable
                if len(self._resource_history) > 1000:
                    self._resource_history = self._resource_history[-500:]
                
                # Detect leaks
                self._detect_leaks(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _detect_leaks(self, metrics: Dict[str, Any]):
        """Detect potential resource leaks."""
        warnings = []
        
        # Memory leak detection
        if metrics['memory_growth'] > self.memory_threshold_mb:
            warnings.append(f"Memory leak detected: {metrics['memory_growth']:.1f}MB growth")
        
        # Handle leak detection
        if metrics['handle_growth'] > self.handle_threshold:
            warnings.append(f"Handle leak detected: {metrics['handle_growth']} handles growth")
        
        # Thread leak detection (simple heuristic)
        if metrics['threads'] > 50:  # Arbitrary threshold
            warnings.append(f"High thread count: {metrics['threads']} threads")
        
        # Trending analysis (if we have enough history)
        if len(self._resource_history) >= 10:
            recent_memory = [m['memory_mb'] for m in self._resource_history[-10:]]
            if len(recent_memory) >= 2:
                memory_trend = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
                if memory_trend > 10:  # Growing by 10MB per sample
                    warnings.append(f"Memory trending upward: {memory_trend:.1f}MB per sample")
        
        if warnings:
            logger.warning(f"Resource leak warnings: {', '.join(warnings)}")
    
    @contextmanager
    def track_resource(self, resource_name: str, resource_obj: Any):
        """Context manager to track resource usage."""
        self._tracked_resources[resource_name] = weakref.ref(resource_obj)
        try:
            yield resource_obj
        finally:
            # Resource should be cleaned up
            if resource_name in self._tracked_resources:
                ref = self._tracked_resources[resource_name]
                if ref() is not None:
                    logger.warning(f"Resource '{resource_name}' may not have been properly cleaned up")
                del self._tracked_resources[resource_name]
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report."""
        if not self._resource_history:
            return {'error': 'No monitoring data available'}
        
        latest = self._resource_history[-1]
        
        return {
            'current_metrics': latest,
            'initial_memory_mb': self.initial_memory,
            'initial_handles': self.initial_handles,
            'memory_growth_mb': latest['memory_growth'],
            'handle_growth': latest['handle_growth'],
            'tracked_resources': len(self._tracked_resources),
            'monitoring_duration_hours': (time.time() - self._resource_history[0]['timestamp']) / 3600,
            'history_samples': len(self._resource_history)
        }


class FaultTolerantExecutor:
    """
    Comprehensive fault-tolerant execution framework.
    
    Combines circuit breakers, retry mechanisms, resource monitoring,
    and graceful degradation for robust operation.
    """
    
    def __init__(self,
                 workspace_dir: str = "/tmp/fault_tolerant_execution",
                 enable_resource_monitoring: bool = True):
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Circuit breakers registry
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Retry configurations
        self.retry_configs: Dict[str, RetryConfig] = {}
        
        # Resource monitoring
        self.resource_detector = None
        if enable_resource_monitoring:
            self.resource_detector = ResourceLeakDetector()
            self.resource_detector.start_monitoring()
        
        # Graceful shutdown handling
        self._shutdown_handlers: List[Callable] = []
        self._register_signal_handlers()
        
        # Fallback strategies
        self._fallback_strategies: Dict[str, Callable] = {}
        
        logger.info("FaultTolerantExecutor initialized")
    
    def register_circuit_breaker(self, 
                               name: str, 
                               config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_retry_config(self, name: str, config: RetryConfig):
        """Register a retry configuration."""
        self.retry_configs[name] = config
    
    def register_fallback_strategy(self, name: str, strategy: Callable):
        """Register a fallback strategy."""
        self._fallback_strategies[name] = strategy
    
    def register_shutdown_handler(self, handler: Callable):
        """Register a graceful shutdown handler."""
        self._shutdown_handlers.append(handler)
    
    @contextmanager
    def robust_execution(self, 
                        operation_name: str,
                        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                        retry_config: Optional[RetryConfig] = None,
                        fallback_strategy: Optional[str] = None):
        """
        Context manager for robust execution with full fault tolerance.
        
        Combines circuit breaking, retries, and fallback strategies.
        """
        
        # Set up circuit breaker
        if operation_name not in self.circuit_breakers and circuit_breaker_config:
            self.register_circuit_breaker(operation_name, circuit_breaker_config)
        
        # Set up retry
        if operation_name not in self.retry_configs and retry_config:
            self.register_retry_config(operation_name, retry_config)
        
        try:
            yield self
        except Exception as e:
            logger.error(f"Robust execution failed for '{operation_name}': {e}")
            
            # Attempt fallback strategy
            if fallback_strategy and fallback_strategy in self._fallback_strategies:
                try:
                    logger.info(f"Attempting fallback strategy: {fallback_strategy}")
                    return self._fallback_strategies[fallback_strategy](e)
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy failed: {fallback_error}")
            
            raise
    
    def execute_with_circuit_breaker(self, 
                                   circuit_breaker_name: str,
                                   func: Callable,
                                   *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if circuit_breaker_name not in self.circuit_breakers:
            raise ValueError(f"Circuit breaker '{circuit_breaker_name}' not registered")
        
        circuit_breaker = self.circuit_breakers[circuit_breaker_name]
        return circuit_breaker.call(func, *args, **kwargs)
    
    def execute_with_retry(self, 
                          retry_config_name: str,
                          func: Callable,
                          *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        if retry_config_name not in self.retry_configs:
            raise ValueError(f"Retry config '{retry_config_name}' not registered")
        
        retry_config = self.retry_configs[retry_config_name]
        retry_mechanism = RetryMechanism(retry_config)
        return retry_mechanism.execute(func, *args, **kwargs)
    
    def execute_robust(self,
                      operation_name: str,
                      func: Callable,
                      *args, **kwargs) -> Any:
        """Execute function with full fault tolerance (circuit breaker + retry)."""
        
        # Create default configurations if not exist
        if operation_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.register_circuit_breaker(operation_name, config)
        
        if operation_name not in self.retry_configs:
            config = RetryConfig()
            self.register_retry_config(operation_name, config)
        
        # Combine circuit breaker and retry
        circuit_breaker = self.circuit_breakers[operation_name]
        retry_config = self.retry_configs[operation_name]
        retry_mechanism = RetryMechanism(retry_config)
        
        def protected_func(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        return retry_mechanism.execute(protected_func, *args, **kwargs)
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.graceful_shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def graceful_shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Initiating graceful shutdown")
        
        # Stop resource monitoring
        if self.resource_detector:
            self.resource_detector.stop_monitoring()
        
        # Execute shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")
        
        logger.info("Graceful shutdown completed")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        health = {
            'timestamp': time.time(),
            'circuit_breakers': {},
            'resource_usage': {},
            'tracked_resources': len(self._fallback_strategies)
        }
        
        # Circuit breaker states
        for name, cb in self.circuit_breakers.items():
            health['circuit_breakers'][name] = cb.get_state()
        
        # Resource usage
        if self.resource_detector:
            health['resource_usage'] = self.resource_detector.get_resource_report()
        
        return health


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(Exception):
    """Raised when operation times out in circuit breaker."""
    pass


# Example usage and testing functions
def test_fault_tolerance_system():
    """Test the fault tolerance system."""
    
    # Initialize fault-tolerant executor
    executor = FaultTolerantExecutor()
    
    # Register circuit breaker
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5.0,
        success_threshold=2
    )
    circuit_breaker = executor.register_circuit_breaker("test_service", cb_config)
    
    # Register retry config
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        exponential_base=2.0
    )
    executor.register_retry_config("test_retry", retry_config)
    
    # Test functions
    def flaky_function(fail_count: int = 0):
        """Function that fails intermittently."""
        if hasattr(flaky_function, 'call_count'):
            flaky_function.call_count += 1
        else:
            flaky_function.call_count = 1
        
        if flaky_function.call_count <= fail_count:
            raise Exception(f"Simulated failure {flaky_function.call_count}")
        
        return f"Success after {flaky_function.call_count} calls"
    
    # Test circuit breaker
    print("\\nTesting Circuit Breaker:")
    try:
        # This should succeed
        result = executor.execute_with_circuit_breaker("test_service", flaky_function, 0)
        print(f"Success: {result}")
        
        # Reset call count for retry test
        flaky_function.call_count = 0
        
    except Exception as e:
        print(f"Circuit breaker test failed: {e}")
    
    # Test retry mechanism
    print("\\nTesting Retry Mechanism:")
    try:
        result = executor.execute_with_retry("test_retry", flaky_function, 2)  # Fail twice, succeed on third
        print(f"Retry success: {result}")
    except Exception as e:
        print(f"Retry test failed: {e}")
    
    # Test robust execution (circuit breaker + retry)
    print("\\nTesting Robust Execution:")
    try:
        flaky_function.call_count = 0  # Reset
        result = executor.execute_robust("robust_test", flaky_function, 1)  # Fail once
        print(f"Robust execution success: {result}")
    except Exception as e:
        print(f"Robust execution failed: {e}")
    
    # Get system health
    health = executor.get_system_health()
    print(f"\\nSystem Health:")
    print(f"Circuit Breakers: {len(health['circuit_breakers'])}")
    print(f"Resource Monitoring: {'enabled' if health['resource_usage'] else 'disabled'}")
    
    # Cleanup
    executor.graceful_shutdown()
    
    return executor


if __name__ == "__main__":
    # Test the fault tolerance system
    test_fault_tolerance_system()