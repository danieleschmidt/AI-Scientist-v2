#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation

Advanced circuit breaker implementation for fault tolerance in AI Scientist v2.
Provides protection against cascading failures in LLM API calls and external services.
"""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union, List
from dataclasses import dataclass, field
import logging
import threading
from collections import deque, Counter

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states following Martin Fowler's pattern."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failure threshold exceeded, blocking calls
    HALF_OPEN = "half_open"    # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Number of failures before opening
    recovery_timeout: int = 60           # Seconds before attempting recovery
    success_threshold: int = 3           # Successes needed to close circuit
    timeout: int = 30                    # Request timeout in seconds
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    
    # Advanced configuration
    failure_rate_threshold: float = 0.5  # Failure rate threshold (0.5 = 50%)
    minimum_requests: int = 10           # Minimum requests before calculating failure rate
    sliding_window_size: int = 100       # Size of sliding window for failure tracking
    
    # Exponential backoff configuration
    exponential_backoff: bool = True     # Enable exponential backoff
    max_backoff_time: int = 300         # Maximum backoff time in seconds
    backoff_multiplier: float = 2.0     # Backoff multiplier


@dataclass
class CircuitBreakerStats:
    """Statistics tracking for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_consecutive_failures: int = 0
    current_consecutive_successes: int = 0
    
    # Sliding window for rate-based circuit breaking
    request_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def failure_rate(self) -> float:
        """Calculate current failure rate from sliding window."""
        if len(self.request_history) < self.minimum_requests:
            return 0.0
        
        failures = sum(1 for success in self.request_history if not success)
        return failures / len(self.request_history)
    
    @property
    def minimum_requests(self) -> int:
        """Minimum requests needed for failure rate calculation."""
        return 10


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with multiple failure detection strategies.
    
    Supports both count-based and rate-based circuit breaking with exponential backoff.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Configuration object (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.last_state_change = datetime.now()
        self._lock = threading.RLock()
        
        # Exponential backoff tracking
        self._backoff_count = 0
        self._next_attempt_time = None
        
        logger.info(f"Circuit breaker '{name}' initialized with {self.config}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset to half-open state."""
        if self.state != CircuitState.OPEN:
            return False
        
        # Calculate backoff time
        if self.config.exponential_backoff:
            backoff_time = min(
                self.config.recovery_timeout * (self.config.backoff_multiplier ** self._backoff_count),
                self.config.max_backoff_time
            )
        else:
            backoff_time = self.config.recovery_timeout
        
        time_since_open = (datetime.now() - self.last_state_change).total_seconds()
        return time_since_open >= backoff_time
    
    def _record_success(self):
        """Record a successful operation."""
        with self._lock:
            self.stats.successful_requests += 1
            self.stats.total_requests += 1
            self.stats.current_consecutive_successes += 1
            self.stats.current_consecutive_failures = 0
            self.stats.last_success_time = datetime.now()
            self.stats.request_history.append(True)
            
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.current_consecutive_successes >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
                    self._backoff_count = 0  # Reset backoff on successful recovery
    
    def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        with self._lock:
            self.stats.failed_requests += 1
            self.stats.total_requests += 1
            self.stats.current_consecutive_failures += 1
            self.stats.current_consecutive_successes = 0
            self.stats.last_failure_time = datetime.now()
            self.stats.request_history.append(False)
            
            if self.state == CircuitState.HALF_OPEN:
                # Single failure in half-open state reopens circuit
                self._change_state(CircuitState.OPEN)
                self._backoff_count += 1
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                should_open = False
                
                # Count-based failure detection
                if self.stats.current_consecutive_failures >= self.config.failure_threshold:
                    should_open = True
                    logger.warning(f"Circuit breaker '{self.name}' opening due to consecutive failures: {self.stats.current_consecutive_failures}")
                
                # Rate-based failure detection
                failure_rate = self.stats.failure_rate()
                if (len(self.stats.request_history) >= self.config.minimum_requests and
                    failure_rate >= self.config.failure_rate_threshold):
                    should_open = True
                    logger.warning(f"Circuit breaker '{self.name}' opening due to failure rate: {failure_rate:.2%}")
                
                if should_open:
                    self._change_state(CircuitState.OPEN)
                    self._backoff_count += 1
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        
        if new_state == CircuitState.OPEN:
            self.stats.circuit_open_count += 1
        
        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
    
    def _can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
        
        return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        if not self._can_execute():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
    
    async def async_call(self, coro_func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            coro_func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        if not self._can_execute():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Apply timeout if configured
            if self.config.timeout > 0:
                result = await asyncio.wait_for(
                    coro_func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await coro_func(*args, **kwargs)
            
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "success_rate": (self.stats.successful_requests / max(1, self.stats.total_requests)),
                "failure_rate": self.stats.failure_rate(),
                "current_consecutive_failures": self.stats.current_consecutive_failures,
                "current_consecutive_successes": self.stats.current_consecutive_successes,
                "circuit_open_count": self.stats.circuit_open_count,
                "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
                "last_state_change": self.last_state_change.isoformat(),
                "backoff_count": self._backoff_count
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._change_state(CircuitState.CLOSED)
            self.stats.current_consecutive_failures = 0
            self.stats.current_consecutive_successes = 0
            self._backoff_count = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)


# Global registry instance
_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for adding circuit breaker protection to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    """
    def decorator(func: Callable) -> Callable:
        breaker = _registry.get_or_create(name, config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    return _registry.get_or_create(name, config)


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return _registry.get_all_stats()


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    _registry.reset_all()


# Predefined configurations for common use cases
LLM_API_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    success_threshold=2,
    timeout=60,
    failure_rate_threshold=0.4,
    minimum_requests=5,
    exponential_backoff=True,
    max_backoff_time=300
)

SEMANTIC_SCHOLAR_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3,
    timeout=30,
    failure_rate_threshold=0.3,
    minimum_requests=10,
    exponential_backoff=True,
    max_backoff_time=600
)

FILE_SYSTEM_CONFIG = CircuitBreakerConfig(
    failure_threshold=10,
    recovery_timeout=15,
    success_threshold=5,
    timeout=10,
    failure_rate_threshold=0.6,
    minimum_requests=20,
    exponential_backoff=False
)


if __name__ == "__main__":
    # Example usage and testing
    import random
    
    @circuit_breaker("test_service", LLM_API_CONFIG)
    def unreliable_service():
        """Simulate an unreliable service."""
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Service unavailable")
        return "Success"
    
    # Test the circuit breaker
    for i in range(20):
        try:
            result = unreliable_service()
            print(f"Request {i+1}: {result}")
        except (Exception, CircuitBreakerError) as e:
            print(f"Request {i+1}: Failed - {e}")
        
        time.sleep(0.1)
    
    # Print final statistics
    stats = get_all_circuit_breaker_stats()
    print("\nFinal Statistics:")
    for name, stat in stats.items():
        print(f"{name}: {stat}")