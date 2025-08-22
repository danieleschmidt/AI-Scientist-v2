"""
Advanced Error Handling and Recovery Mechanisms

Comprehensive error handling framework with intelligent recovery, adaptive retry strategies,
and fault-tolerant execution patterns for robust autonomous scientific research.
"""

import time
import asyncio
import threading
import traceback
import logging
import json
import uuid
import inspect
from typing import (
    Dict, List, Any, Optional, Union, Callable, Type, Tuple, 
    Generic, TypeVar, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
import functools
import weakref
from collections import defaultdict, deque
import signal
import sys
import gc

logger = logging.getLogger(__name__)

T = TypeVar('T')
ReturnType = TypeVar('ReturnType')


class ErrorSeverity(Enum):
    """Error severity levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART_COMPONENT = "restart_component"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    NETWORK = "network"
    COMPUTATION = "computation"
    MEMORY = "memory"
    STORAGE = "storage"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    CONCURRENCY = "concurrency"
    EXTERNAL_SERVICE = "external_service"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.UNKNOWN
    source_module: str = ""
    source_function: str = ""
    source_line: int = 0
    exception_type: str = ""
    exception_message: str = ""
    traceback_info: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE_RETRY
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'source_module': self.source_module,
            'source_function': self.source_function,
            'source_line': self.source_line,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'traceback_info': self.traceback_info,
            'context_data': self.context_data,
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts,
            'recovery_strategy': self.recovery_strategy.value,
            'tags': self.tags
        }


@dataclass
class RecoveryResult:
    """Result of recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    execution_time: float
    result: Any = None
    error_context: Optional[ErrorContext] = None
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'strategy_used': self.strategy_used.value,
            'execution_time': self.execution_time,
            'result': self.result,
            'error_context': self.error_context.to_dict() if self.error_context else None,
            'recovery_data': self.recovery_data
        }


@runtime_checkable
class ErrorHandler(Protocol):
    """Protocol for error handlers."""
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle the error."""
        ...
    
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle the error and return recovery result."""
        ...


@runtime_checkable
class RecoveryProvider(Protocol):
    """Protocol for recovery providers."""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this provider can recover from the error."""
        ...
    
    def recover(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Attempt recovery and return result."""
        ...


class BaseErrorHandler(ABC):
    """Base class for error handlers."""
    
    def __init__(self, priority: int = 0):
        self.priority = priority
        self.handled_errors: List[ErrorContext] = []
        self.success_rate: float = 0.0
        
    @abstractmethod
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle the error."""
        pass
    
    @abstractmethod
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle the error and return recovery result."""
        pass
    
    def update_success_rate(self, success: bool):
        """Update handler success rate."""
        # Simple moving average
        alpha = 0.1  # Learning rate
        new_rate = 1.0 if success else 0.0
        self.success_rate = alpha * new_rate + (1 - alpha) * self.success_rate


class NetworkErrorHandler(BaseErrorHandler):
    """Handler for network-related errors."""
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this is a network error."""
        network_exceptions = [
            'ConnectionError', 'TimeoutError', 'HTTPError',
            'URLError', 'DNSError', 'SSLError'
        ]
        
        return (
            context.category == ErrorCategory.NETWORK or
            any(exc in str(type(error)) for exc in network_exceptions) or
            'network' in str(error).lower() or
            'connection' in str(error).lower()
        )
    
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle network errors with exponential backoff."""
        start_time = time.time()
        
        try:
            # Update context
            context.category = ErrorCategory.NETWORK
            context.recovery_strategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
            
            # Implement exponential backoff
            base_delay = 1.0
            max_delay = 60.0
            delay = min(base_delay * (2 ** context.recovery_attempts), max_delay)
            
            logger.warning(f"Network error detected, waiting {delay:.1f}s before retry: {error}")
            time.sleep(delay)
            
            execution_time = time.time() - start_time
            
            # Return success to indicate retry should be attempted
            result = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                execution_time=execution_time,
                recovery_data={'delay_used': delay, 'attempt': context.recovery_attempts}
            )
            
            self.update_success_rate(True)
            return result
            
        except Exception as recovery_error:
            execution_time = time.time() - start_time
            logger.error(f"Error during network error recovery: {recovery_error}")
            
            result = RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                execution_time=execution_time,
                error_context=ErrorContext(
                    exception_type=type(recovery_error).__name__,
                    exception_message=str(recovery_error),
                    traceback_info=traceback.format_exc()
                )
            )
            
            self.update_success_rate(False)
            return result


class ComputationErrorHandler(BaseErrorHandler):
    """Handler for computation-related errors."""
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this is a computation error."""
        computation_exceptions = [
            'ValueError', 'TypeError', 'ArithmeticError', 'ZeroDivisionError',
            'OverflowError', 'FloatingPointError', 'RuntimeError'
        ]
        
        return (
            context.category == ErrorCategory.COMPUTATION or
            any(exc in str(type(error)) for exc in computation_exceptions)
        )
    
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle computation errors with input validation and fallback."""
        start_time = time.time()
        
        try:
            context.category = ErrorCategory.COMPUTATION
            context.recovery_strategy = RecoveryStrategy.FALLBACK
            
            # Attempt to provide safe fallback values
            fallback_result = self._get_fallback_result(error, context)
            
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                execution_time=execution_time,
                result=fallback_result,
                recovery_data={'fallback_provided': True}
            )
            
            self.update_success_rate(True)
            return result
            
        except Exception as recovery_error:
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                execution_time=execution_time,
                error_context=ErrorContext(
                    exception_type=type(recovery_error).__name__,
                    exception_message=str(recovery_error),
                    traceback_info=traceback.format_exc()
                )
            )
            
            self.update_success_rate(False)
            return result
    
    def _get_fallback_result(self, error: Exception, context: ErrorContext) -> Any:
        """Get appropriate fallback result based on error type."""
        if isinstance(error, ZeroDivisionError):
            return float('inf')  # Or 0, depending on context
        elif isinstance(error, ValueError):
            return None  # Or default value
        elif isinstance(error, TypeError):
            return {}  # Or appropriate default structure
        else:
            return None


class ResourceErrorHandler(BaseErrorHandler):
    """Handler for resource exhaustion errors."""
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this is a resource error."""
        resource_exceptions = [
            'MemoryError', 'OSError', 'IOError', 'ResourceWarning'
        ]
        
        return (
            context.category == ErrorCategory.RESOURCE_EXHAUSTION or
            any(exc in str(type(error)) for exc in resource_exceptions) or
            'memory' in str(error).lower() or
            'resource' in str(error).lower()
        )
    
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle resource errors with cleanup and optimization."""
        start_time = time.time()
        
        try:
            context.category = ErrorCategory.RESOURCE_EXHAUSTION
            context.recovery_strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
            
            # Perform resource cleanup
            cleanup_stats = self._perform_resource_cleanup()
            
            # Wait for resources to be available
            time.sleep(2.0)
            
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                execution_time=execution_time,
                recovery_data={
                    'cleanup_performed': True,
                    'cleanup_stats': cleanup_stats
                }
            )
            
            self.update_success_rate(True)
            return result
            
        except Exception as recovery_error:
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                execution_time=execution_time,
                error_context=ErrorContext(
                    exception_type=type(recovery_error).__name__,
                    exception_message=str(recovery_error),
                    traceback_info=traceback.format_exc()
                )
            )
            
            self.update_success_rate(False)
            return result
    
    def _perform_resource_cleanup(self) -> Dict[str, Any]:
        """Perform resource cleanup operations."""
        stats = {}
        
        # Force garbage collection
        collected = gc.collect()
        stats['garbage_collected'] = collected
        
        # Clear any global caches (implementation-specific)
        stats['caches_cleared'] = 0
        
        return stats


class CircuitBreakerHandler(BaseErrorHandler):
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, priority: int = 10):
        super().__init__(priority)
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Circuit breaker handles repeated failures of any type."""
        return context.recovery_attempts >= 2  # After multiple failures
    
    def handle(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle with circuit breaker pattern."""
        start_time = time.time()
        current_time = time.time()
        
        try:
            # Update failure tracking
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Determine circuit breaker state
            if self.failure_count >= self.failure_threshold:
                if self.state == "CLOSED":
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            # Handle based on state
            if self.state == "OPEN":
                if current_time - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    # Fail fast
                    execution_time = time.time() - start_time
                    return RecoveryResult(
                        success=False,
                        strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                        execution_time=execution_time,
                        recovery_data={
                            'circuit_state': self.state,
                            'failure_count': self.failure_count,
                            'time_until_retry': self.timeout - (current_time - self.last_failure_time)
                        }
                    )
            
            elif self.state == "HALF_OPEN":
                # Test if service is recovered
                logger.info("Circuit breaker testing service recovery")
                # Reset for potential success
                self.failure_count = 0
                self.state = "CLOSED"
            
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                execution_time=execution_time,
                recovery_data={
                    'circuit_state': self.state,
                    'failure_count': self.failure_count
                }
            )
            
            self.update_success_rate(True)
            return result
            
        except Exception as recovery_error:
            execution_time = time.time() - start_time
            
            result = RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                execution_time=execution_time,
                error_context=ErrorContext(
                    exception_type=type(recovery_error).__name__,
                    exception_message=str(recovery_error),
                    traceback_info=traceback.format_exc()
                )
            )
            
            self.update_success_rate(False)
            return result


class ErrorClassifier:
    """Intelligent error classification system."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        self.learning_data: Dict[str, ErrorCategory] = {}
    
    def classify_error(self, error: Exception, context_data: Dict[str, Any]) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error type and severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check learned classifications first
        if error_type in self.learning_data:
            category = self.learning_data[error_type]
        else:
            # Apply classification rules
            category = self._apply_classification_rules(error, error_message)
        
        # Determine severity
        severity = self._determine_severity(error, category, context_data)
        
        return category, severity
    
    def _build_classification_rules(self) -> Dict[str, ErrorCategory]:
        """Build error classification rules."""
        return {
            # Network errors
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.NETWORK,
            'HTTPError': ErrorCategory.NETWORK,
            'URLError': ErrorCategory.NETWORK,
            'SSLError': ErrorCategory.NETWORK,
            
            # Computation errors
            'ValueError': ErrorCategory.COMPUTATION,
            'TypeError': ErrorCategory.COMPUTATION,
            'ArithmeticError': ErrorCategory.COMPUTATION,
            'ZeroDivisionError': ErrorCategory.COMPUTATION,
            'OverflowError': ErrorCategory.COMPUTATION,
            
            # Memory errors
            'MemoryError': ErrorCategory.MEMORY,
            'ResourceWarning': ErrorCategory.RESOURCE_EXHAUSTION,
            
            # Storage errors
            'IOError': ErrorCategory.STORAGE,
            'OSError': ErrorCategory.STORAGE,
            'FileNotFoundError': ErrorCategory.STORAGE,
            'PermissionError': ErrorCategory.STORAGE,
            
            # Authentication/Authorization
            'AuthenticationError': ErrorCategory.AUTHENTICATION,
            'PermissionDeniedError': ErrorCategory.AUTHORIZATION,
            
            # Validation errors
            'ValidationError': ErrorCategory.VALIDATION,
            'AssertionError': ErrorCategory.VALIDATION,
            
            # Configuration errors
            'ConfigurationError': ErrorCategory.CONFIGURATION,
            'ImportError': ErrorCategory.CONFIGURATION,
            'ModuleNotFoundError': ErrorCategory.CONFIGURATION,
            
            # Concurrency errors
            'ThreadingError': ErrorCategory.CONCURRENCY,
            'DeadlockError': ErrorCategory.CONCURRENCY,
        }
    
    def _apply_classification_rules(self, error: Exception, error_message: str) -> ErrorCategory:
        """Apply classification rules to determine error category."""
        error_type = type(error).__name__
        
        # Direct type mapping
        if error_type in self.classification_rules:
            return self.classification_rules[error_type]
        
        # Message-based classification
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['memory', 'resource', 'allocation']):
            return ErrorCategory.MEMORY
        elif any(keyword in error_message for keyword in ['file', 'directory', 'path', 'permission']):
            return ErrorCategory.STORAGE
        elif any(keyword in error_message for keyword in ['auth', 'login', 'credential']):
            return ErrorCategory.AUTHENTICATION
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        elif any(keyword in error_message for keyword in ['thread', 'lock', 'deadlock']):
            return ErrorCategory.CONCURRENCY
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory, 
                          context_data: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['MemoryError', 'SystemExit', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity by category
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.DATA_CORRUPTION]:
            return ErrorSeverity.ERROR
        
        # Context-based severity
        if context_data.get('is_critical_operation', False):
            return ErrorSeverity.ERROR
        
        if context_data.get('retry_count', 0) > 3:
            return ErrorSeverity.ERROR
        
        # Default to warning
        return ErrorSeverity.WARNING
    
    def learn_classification(self, error_type: str, category: ErrorCategory):
        """Learn new error classification."""
        self.learning_data[error_type] = category


class RobustExecutor:
    """Robust execution framework with comprehensive error handling."""
    
    def __init__(self, max_retries: int = 3, default_timeout: float = 30.0):
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        
        # Error handling components
        self.error_classifier = ErrorClassifier()
        self.error_handlers: List[BaseErrorHandler] = []
        self.error_history: deque = deque(maxlen=1000)
        
        # Recovery tracking
        self.recovery_statistics = defaultdict(int)
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        # Monitoring
        self.execution_statistics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'recovered_executions': 0,
            'total_recovery_time': 0.0
        }
    
    def _initialize_default_handlers(self):
        """Initialize default error handlers."""
        self.error_handlers = [
            CircuitBreakerHandler(priority=10),
            NetworkErrorHandler(priority=5),
            ResourceErrorHandler(priority=7),
            ComputationErrorHandler(priority=3),
        ]
        
        # Sort by priority (higher priority first)
        self.error_handlers.sort(key=lambda h: h.priority, reverse=True)
    
    def execute_with_recovery(self, func: Callable[..., T], *args, 
                            timeout: Optional[float] = None, 
                            context_data: Optional[Dict[str, Any]] = None,
                            **kwargs) -> T:
        """Execute function with comprehensive error handling and recovery."""
        self.execution_statistics['total_executions'] += 1
        
        timeout = timeout or self.default_timeout
        context_data = context_data or {}
        
        # Create execution context
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting robust execution {execution_id}: {func.__name__}")
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute with timeout protection
                result = self._execute_with_timeout(func, args, kwargs, timeout)
                
                # Success
                execution_time = time.time() - start_time
                logger.info(f"Execution {execution_id} succeeded after {attempt} attempts in {execution_time:.2f}s")
                
                self.execution_statistics['successful_executions'] += 1
                if attempt > 0:
                    self.execution_statistics['recovered_executions'] += 1
                
                return result
                
            except Exception as error:
                # Create error context
                error_context = self._create_error_context(
                    error, func, attempt, context_data
                )
                
                # Classify error
                category, severity = self.error_classifier.classify_error(error, context_data)
                error_context.category = category
                error_context.severity = severity
                
                # Log error
                logger.error(f"Execution {execution_id} attempt {attempt} failed: {error}")
                
                # Record error in history
                self.error_history.append(error_context)
                
                # Check if we should attempt recovery
                if attempt >= self.max_retries:
                    logger.error(f"Execution {execution_id} failed permanently after {attempt} attempts")
                    self.execution_statistics['failed_executions'] += 1
                    raise error
                
                # Attempt recovery
                recovery_result = self._attempt_recovery(error, error_context)
                
                if recovery_result.success:
                    self.execution_statistics['recovered_executions'] += 1
                    self.execution_statistics['total_recovery_time'] += recovery_result.execution_time
                    logger.info(f"Recovery successful for {execution_id}, retrying...")
                    continue
                else:
                    # Recovery failed
                    if recovery_result.error_context:
                        self.error_history.append(recovery_result.error_context)
                    
                    logger.error(f"Recovery failed for {execution_id}: {recovery_result.recovery_data}")
                    
                    # Continue to next attempt unless it's the last one
                    if attempt >= self.max_retries:
                        self.execution_statistics['failed_executions'] += 1
                        raise error
        
        # This should never be reached, but just in case
        self.execution_statistics['failed_executions'] += 1
        raise RuntimeError(f"Execution {execution_id} failed unexpectedly")
    
    def _execute_with_timeout(self, func: Callable, args: tuple, 
                            kwargs: dict, timeout: float) -> Any:
        """Execute function with timeout protection."""
        # For simplicity, we'll use a basic timeout approach
        # In production, you might want to use more sophisticated timeout mechanisms
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")
        
        # Set up timeout (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def _create_error_context(self, error: Exception, func: Callable, 
                            attempt: int, context_data: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        # Get source information
        frame = inspect.currentframe()
        source_info = self._get_source_info(frame)
        
        return ErrorContext(
            severity=ErrorSeverity.ERROR,
            source_module=func.__module__ if hasattr(func, '__module__') else "",
            source_function=func.__name__ if hasattr(func, '__name__') else "",
            source_line=source_info.get('line', 0),
            exception_type=type(error).__name__,
            exception_message=str(error),
            traceback_info=traceback.format_exc(),
            context_data=context_data,
            recovery_attempts=attempt,
            tags=[f"attempt_{attempt}", f"function_{func.__name__}"]
        )
    
    def _get_source_info(self, frame) -> Dict[str, Any]:
        """Get source code information from frame."""
        try:
            if frame and frame.f_back:
                return {
                    'filename': frame.f_back.f_code.co_filename,
                    'line': frame.f_back.f_lineno,
                    'function': frame.f_back.f_code.co_name
                }
        except Exception:
            pass
        
        return {'filename': '', 'line': 0, 'function': ''}
    
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Attempt error recovery using available handlers."""
        recovery_start_time = time.time()
        
        # Find suitable handler
        for handler in self.error_handlers:
            if handler.can_handle(error, context):
                logger.info(f"Attempting recovery with {type(handler).__name__}")
                
                try:
                    recovery_result = handler.handle(error, context)
                    
                    # Update statistics
                    strategy_name = recovery_result.strategy_used.value
                    self.recovery_statistics[f"{strategy_name}_attempts"] += 1
                    
                    if recovery_result.success:
                        self.recovery_statistics[f"{strategy_name}_successes"] += 1
                    
                    return recovery_result
                    
                except Exception as recovery_error:
                    logger.error(f"Recovery handler {type(handler).__name__} failed: {recovery_error}")
                    continue
        
        # No suitable handler found
        recovery_time = time.time() - recovery_start_time
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            execution_time=recovery_time,
            recovery_data={'message': 'No suitable recovery handler found'}
        )
    
    def add_error_handler(self, handler: BaseErrorHandler):
        """Add custom error handler."""
        self.error_handlers.append(handler)
        self.error_handlers.sort(key=lambda h: h.priority, reverse=True)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        total_executions = self.execution_statistics['total_executions']
        
        if total_executions == 0:
            return self.execution_statistics.copy()
        
        stats = self.execution_statistics.copy()
        stats.update({
            'success_rate': stats['successful_executions'] / total_executions,
            'failure_rate': stats['failed_executions'] / total_executions,
            'recovery_rate': stats['recovered_executions'] / total_executions,
            'average_recovery_time': (
                stats['total_recovery_time'] / max(1, stats['recovered_executions'])
            ),
            'recovery_statistics': dict(self.recovery_statistics),
            'recent_errors': [
                error.to_dict() for error in list(self.error_history)[-10:]
            ]
        })
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status based on error patterns."""
        stats = self.get_error_statistics()
        
        # Determine health score (0-100)
        success_rate = stats.get('success_rate', 0)
        recovery_rate = stats.get('recovery_rate', 0)
        
        health_score = int((success_rate * 70 + recovery_rate * 30) * 100)
        
        # Health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'total_executions': stats['total_executions'],
            'success_rate': stats.get('success_rate', 0),
            'recovery_rate': stats.get('recovery_rate', 0),
            'recent_error_count': len([
                error for error in self.error_history 
                if time.time() - error.timestamp < 3600  # Last hour
            ]),
            'active_handlers': len(self.error_handlers),
            'recommendations': self._generate_health_recommendations(stats)
        }
    
    def _generate_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        success_rate = stats.get('success_rate', 0)
        failure_rate = stats.get('failure_rate', 0)
        recovery_rate = stats.get('recovery_rate', 0)
        
        if success_rate < 0.8:
            recommendations.append("Low success rate detected - review error patterns and root causes")
        
        if failure_rate > 0.2:
            recommendations.append("High failure rate - implement additional error prevention measures")
        
        if recovery_rate < 0.5:
            recommendations.append("Low recovery rate - improve error handling strategies")
        
        if stats.get('average_recovery_time', 0) > 10:
            recommendations.append("Slow recovery times - optimize recovery procedures")
        
        # Check for specific error patterns
        recent_errors = [error for error in self.error_history 
                        if time.time() - error.timestamp < 3600]
        
        if len(recent_errors) > 10:
            recommendations.append("High recent error frequency - investigate system stability")
        
        error_categories = defaultdict(int)
        for error in recent_errors:
            error_categories[error.category.value] += 1
        
        for category, count in error_categories.items():
            if count > 5:
                recommendations.append(f"Frequent {category} errors - focus on {category} resilience")
        
        return recommendations


# Decorator for robust execution
def robust_execution(max_retries: int = 3, timeout: float = 30.0, 
                    context_data: Optional[Dict[str, Any]] = None):
    """Decorator for robust function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Create a shared executor for this function
        executor = RobustExecutor(max_retries=max_retries, default_timeout=timeout)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return executor.execute_with_recovery(
                func, *args, timeout=timeout, context_data=context_data, **kwargs
            )
        
        # Attach executor for access to statistics
        wrapper._robust_executor = executor  # type: ignore
        
        return wrapper
    return decorator


# Context manager for robust execution
@contextmanager
def robust_context(max_retries: int = 3, timeout: float = 30.0):
    """Context manager for robust execution blocks."""
    executor = RobustExecutor(max_retries=max_retries, default_timeout=timeout)
    
    class RobustContextManager:
        def __init__(self, executor: RobustExecutor):
            self.executor = executor
        
        def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
            return self.executor.execute_with_recovery(func, *args, **kwargs)
    
    try:
        yield RobustContextManager(executor)
    except Exception as e:
        logger.error(f"Error in robust context: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create robust executor
    executor = RobustExecutor(max_retries=3)
    
    # Test functions
    def network_operation():
        """Simulated network operation that may fail."""
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Network connection failed")
        return "Network operation successful"
    
    def computation_operation(x: float, y: float):
        """Simulated computation that may fail."""
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise ValueError("Invalid computation parameters")
        return x / y
    
    def memory_intensive_operation():
        """Simulated memory-intensive operation."""
        import random
        if random.random() < 0.4:  # 40% failure rate
            raise MemoryError("Insufficient memory")
        return "Memory operation completed"
    
    # Test robust execution
    print("Testing robust execution framework...")
    
    # Test 1: Network operations
    print("\n1. Testing network operations:")
    for i in range(5):
        try:
            result = executor.execute_with_recovery(network_operation)
            print(f"  Attempt {i+1}: {result}")
        except Exception as e:
            print(f"  Attempt {i+1}: Failed - {e}")
    
    # Test 2: Computation operations
    print("\n2. Testing computation operations:")
    for i in range(3):
        try:
            result = executor.execute_with_recovery(computation_operation, 10.0, 2.0)
            print(f"  Computation {i+1}: {result}")
        except Exception as e:
            print(f"  Computation {i+1}: Failed - {e}")
    
    # Test 3: Memory operations
    print("\n3. Testing memory operations:")
    for i in range(3):
        try:
            result = executor.execute_with_recovery(memory_intensive_operation)
            print(f"  Memory operation {i+1}: {result}")
        except Exception as e:
            print(f"  Memory operation {i+1}: Failed - {e}")
    
    # Test decorator
    print("\n4. Testing robust execution decorator:")
    
    @robust_execution(max_retries=2, timeout=5.0)
    def decorated_function():
        import random
        if random.random() < 0.5:
            raise RuntimeError("Random failure")
        return "Decorated function success"
    
    for i in range(3):
        try:
            result = decorated_function()
            print(f"  Decorated call {i+1}: {result}")
        except Exception as e:
            print(f"  Decorated call {i+1}: Failed - {e}")
    
    # Test context manager
    print("\n5. Testing robust context manager:")
    
    def context_function():
        import random
        if random.random() < 0.6:
            raise IOError("File operation failed")
        return "Context operation success"
    
    with robust_context(max_retries=2) as ctx:
        for i in range(3):
            try:
                result = ctx.execute(context_function)
                print(f"  Context call {i+1}: {result}")
            except Exception as e:
                print(f"  Context call {i+1}: Failed - {e}")
    
    # Print statistics
    print("\n6. Error handling statistics:")
    stats = executor.get_error_statistics()
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"  Recovery rate: {stats.get('recovery_rate', 0):.1%}")
    print(f"  Average recovery time: {stats.get('average_recovery_time', 0):.2f}s")
    
    # Health status
    print("\n7. System health status:")
    health = executor.get_health_status()
    print(f"  Health score: {health['health_score']}/100")
    print(f"  Health status: {health['health_status']}")
    print(f"  Recommendations:")
    for rec in health['recommendations']:
        print(f"    - {rec}")
    
    print("\nRobust execution framework testing completed.")