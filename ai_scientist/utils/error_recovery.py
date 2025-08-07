#!/usr/bin/env python3
"""
Error Recovery System for AI Scientist v2

Advanced error recovery, fault tolerance, and resilience framework with
automatic healing and graceful degradation capabilities.
"""

import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import queue
import signal
from pathlib import Path
import subprocess
import psutil

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT = "abort"

class FailureType(Enum):
    """Types of failures that can occur."""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    strategy: RecoveryStrategy
    action: Callable[[], bool]
    description: str
    max_attempts: int = 3
    delay: float = 1.0
    timeout: Optional[float] = None

@dataclass
class FailureInfo:
    """Information about a failure."""
    failure_type: FailureType
    exception: Exception
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_str: str = ""
    recovery_attempts: int = 0
    recovered: bool = False

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator implementation."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker open for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self.failure_count = 0
            self.state = 'closed'
            if self.last_failure_time:
                logger.info("Circuit breaker reset after successful execution")
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class ErrorRecoveryManager:
    """Centralized error recovery management."""
    
    def __init__(self):
        self.recovery_strategies: Dict[FailureType, List[RecoveryAction]] = {}
        self.failure_history: List[FailureInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        
        # Initialize default recovery strategies
        self._setup_default_strategies()
    
    def register_recovery_strategy(self, failure_type: FailureType, 
                                 recovery_action: RecoveryAction):
        """Register a recovery strategy for a failure type."""
        if failure_type not in self.recovery_strategies:
            self.recovery_strategies[failure_type] = []
        
        self.recovery_strategies[failure_type].append(recovery_action)
        logger.info(f"Registered recovery strategy for {failure_type.value}: {recovery_action.description}")
    
    def get_circuit_breaker(self, name: str, failure_threshold: int = 5, 
                          recovery_timeout: float = 60.0) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold, recovery_timeout
            )
        return self.circuit_breakers[name]
    
    def handle_failure(self, exception: Exception, context: Dict[str, Any] = None) -> bool:
        """Handle a failure and attempt recovery."""
        failure_type = self._classify_failure(exception)
        failure_info = FailureInfo(
            failure_type=failure_type,
            exception=exception,
            timestamp=datetime.now(),
            context=context or {},
            traceback_str=traceback.format_exc()
        )
        
        with self._lock:
            self.failure_history.append(failure_info)
            # Keep only last 1000 failures
            if len(self.failure_history) > 1000:
                self.failure_history = self.failure_history[-1000:]
        
        logger.error(f"Failure detected: {failure_type.value} - {str(exception)}")
        
        # Attempt recovery
        recovered = self._attempt_recovery(failure_info)
        failure_info.recovered = recovered
        
        return recovered
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception."""
        exception_name = type(exception).__name__
        message = str(exception).lower()
        
        # Network-related errors
        if any(term in exception_name.lower() for term in ['connection', 'network', 'http', 'url']):
            return FailureType.NETWORK_ERROR
        if any(term in message for term in ['connection', 'network', 'timeout', 'refused']):
            return FailureType.NETWORK_ERROR
        
        # API-related errors
        if any(term in exception_name.lower() for term in ['api', 'rate', 'quota', 'auth']):
            return FailureType.API_ERROR
        if any(term in message for term in ['api', 'rate limit', 'quota', 'unauthorized', '429', '401', '403']):
            return FailureType.API_ERROR
        
        # Resource-related errors
        if any(term in exception_name.lower() for term in ['memory', 'disk', 'space', 'resource']):
            return FailureType.RESOURCE_ERROR
        if any(term in message for term in ['memory', 'disk', 'space', 'resource', 'cuda', 'gpu']):
            return FailureType.RESOURCE_ERROR
        
        # Timeout errors
        if 'timeout' in exception_name.lower() or 'timeout' in message:
            return FailureType.TIMEOUT_ERROR
        
        # Validation errors
        if any(term in exception_name.lower() for term in ['validation', 'value', 'type', 'key']):
            return FailureType.VALIDATION_ERROR
        
        # Processing errors
        if any(term in exception_name.lower() for term in ['processing', 'computation', 'calculation']):
            return FailureType.PROCESSING_ERROR
        
        # System errors
        if any(term in exception_name.lower() for term in ['system', 'os', 'permission', 'access']):
            return FailureType.SYSTEM_ERROR
        
        return FailureType.UNKNOWN_ERROR
    
    def _attempt_recovery(self, failure_info: FailureInfo) -> bool:
        """Attempt recovery using registered strategies."""
        strategies = self.recovery_strategies.get(failure_info.failure_type, [])
        
        if not strategies:
            logger.warning(f"No recovery strategies for {failure_info.failure_type.value}")
            return False
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery: {strategy.description}")
                
                for attempt in range(strategy.max_attempts):
                    try:
                        if strategy.timeout:
                            # Use threading for timeout
                            result_queue = queue.Queue()
                            
                            def run_action():
                                try:
                                    result = strategy.action()
                                    result_queue.put(('success', result))
                                except Exception as e:
                                    result_queue.put(('error', e))
                            
                            thread = threading.Thread(target=run_action)
                            thread.start()
                            thread.join(timeout=strategy.timeout)
                            
                            if thread.is_alive():
                                logger.warning(f"Recovery action timed out after {strategy.timeout}s")
                                continue
                            
                            if not result_queue.empty():
                                status, result = result_queue.get()
                                if status == 'success' and result:
                                    logger.info(f"Recovery successful: {strategy.description}")
                                    failure_info.recovery_attempts = attempt + 1
                                    return True
                        else:
                            result = strategy.action()
                            if result:
                                logger.info(f"Recovery successful: {strategy.description}")
                                failure_info.recovery_attempts = attempt + 1
                                return True
                        
                        if attempt < strategy.max_attempts - 1:
                            time.sleep(strategy.delay * (attempt + 1))  # Exponential backoff
                            
                    except Exception as e:
                        logger.warning(f"Recovery attempt {attempt + 1} failed: {str(e)}")
                        if attempt < strategy.max_attempts - 1:
                            time.sleep(strategy.delay * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Recovery strategy failed: {str(e)}")
                continue
        
        logger.error(f"All recovery strategies failed for {failure_info.failure_type.value}")
        return False
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        
        # Network error recovery
        self.register_recovery_strategy(
            FailureType.NETWORK_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=lambda: True,  # Simple retry
                description="Wait and retry network operation",
                max_attempts=3,
                delay=2.0
            )
        )
        
        # API error recovery
        self.register_recovery_strategy(
            FailureType.API_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=self._handle_api_error,
                description="Handle API rate limiting and retry",
                max_attempts=5,
                delay=5.0
            )
        )
        
        # Resource error recovery
        self.register_recovery_strategy(
            FailureType.RESOURCE_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART,
                action=self._cleanup_resources,
                description="Clean up resources and retry",
                max_attempts=2,
                delay=10.0
            )
        )
        
        # Timeout error recovery
        self.register_recovery_strategy(
            FailureType.TIMEOUT_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=lambda: True,
                description="Retry with extended timeout",
                max_attempts=2,
                delay=5.0
            )
        )
    
    def _handle_api_error(self) -> bool:
        """Handle API-related errors."""
        try:
            # Wait for rate limit reset (common pattern)
            time.sleep(60)
            return True
        except Exception:
            return False
    
    def _cleanup_resources(self) -> bool:
        """Clean up system resources."""
        try:
            # GPU cleanup
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            except ImportError:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clean up temporary files
            temp_dirs = ['/tmp', tempfile.gettempdir()]
            for temp_dir in temp_dirs:
                try:
                    temp_path = Path(temp_dir)
                    for file in temp_path.glob('ai_scientist_*'):
                        if file.is_file() and (time.time() - file.stat().st_mtime) > 3600:
                            file.unlink()
                except Exception as e:
                    logger.debug(f"Cleanup error in {temp_dir}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            return False
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics."""
        with self._lock:
            if not self.failure_history:
                return {'total_failures': 0}
            
            failure_counts = {}
            recovered_count = 0
            recent_failures = 0
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for failure in self.failure_history:
                failure_type = failure.failure_type.value
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
                
                if failure.recovered:
                    recovered_count += 1
                
                if failure.timestamp > cutoff_time:
                    recent_failures += 1
            
            recovery_rate = recovered_count / len(self.failure_history) if self.failure_history else 0
            
            return {
                'total_failures': len(self.failure_history),
                'failure_counts': failure_counts,
                'recovery_rate': recovery_rate,
                'recent_failures_24h': recent_failures,
                'circuit_breaker_states': {
                    name: cb.state for name, cb in self.circuit_breakers.items()
                }
            }

class GracefulShutdownHandler:
    """Handle graceful shutdown with cleanup."""
    
    def __init__(self):
        self.shutdown_handlers: List[Callable[[], None]] = []
        self.emergency_handlers: List[Callable[[], None]] = []
        self._shutdown_initiated = False
        self._setup_signal_handlers()
    
    def register_shutdown_handler(self, handler: Callable[[], None]):
        """Register a handler to be called during shutdown."""
        self.shutdown_handlers.append(handler)
    
    def register_emergency_handler(self, handler: Callable[[], None]):
        """Register an emergency handler for critical cleanup."""
        self.emergency_handlers.append(handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if self._shutdown_initiated:
                logger.warning("Force shutdown requested")
                self._emergency_shutdown()
                sys.exit(1)
            
            logger.info(f"Shutdown signal received: {signum}")
            self._graceful_shutdown()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown."""
        if self._shutdown_initiated:
            return
        
        self._shutdown_initiated = True
        logger.info("Initiating graceful shutdown...")
        
        # Execute shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                logger.debug(f"Executing shutdown handler: {handler.__name__}")
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")
        
        logger.info("Graceful shutdown completed")
    
    def _emergency_shutdown(self):
        """Perform emergency shutdown."""
        logger.critical("Initiating emergency shutdown...")
        
        # Execute emergency handlers
        for handler in self.emergency_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Emergency handler failed: {e}")
        
        # Force cleanup
        self._force_cleanup()
        
        logger.critical("Emergency shutdown completed")
    
    def _force_cleanup(self):
        """Force cleanup of resources."""
        try:
            # Kill all child processes
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait a bit then force kill
            time.sleep(2)
            
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except Exception as e:
            logger.error(f"Force cleanup failed: {e}")

# Global instances
error_recovery_manager = ErrorRecoveryManager()
shutdown_handler = GracefulShutdownHandler()

# Convenience functions
def with_recovery(failure_type: FailureType = None):
    """Decorator for automatic error recovery."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],
                    'kwargs': str(kwargs)[:100]
                }
                
                recovered = error_recovery_manager.handle_failure(e, context)
                if not recovered:
                    raise e
                
                # Retry once after recovery
                try:
                    return func(*args, **kwargs)
                except Exception as retry_e:
                    logger.error(f"Retry after recovery failed: {retry_e}")
                    raise retry_e
        
        return wrapper
    return decorator

def circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator for circuit breaker pattern."""
    def decorator(func):
        cb = error_recovery_manager.get_circuit_breaker(name, failure_threshold, recovery_timeout)
        return cb(func)
    return decorator