"""
Comprehensive Error Handling for Quantum Task Planner

Custom exception classes and error handling utilities for robust
quantum-inspired task planning operations.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    OPTIMIZATION = "optimization" 
    QUANTUM = "quantum"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    SYSTEM = "system"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Error context information."""
    operation: str
    component: str
    parameters: Dict[str, Any]
    timestamp: float
    stack_trace: str
    additional_info: Dict[str, Any]


class QuantumPlannerError(Exception):
    """Base exception for quantum task planner errors."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize quantum planner error.
        
        Args:
            message: Error description
            category: Error category
            severity: Error severity level
            context: Additional error context
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        
        # Log error based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{category.value}]: {message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"ERROR [{category.value}]: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"WARNING [{category.value}]: {message}")
        else:
            logger.info(f"INFO [{category.value}]: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context.__dict__ if self.context else None,
            'cause': str(self.cause) if self.cause else None
        }


class ValidationError(QuantumPlannerError):
    """Errors related to input validation."""
    
    def __init__(self, 
                 message: str,
                 field_name: Optional[str] = None,
                 field_value: Optional[Any] = None,
                 validation_rule: Optional[str] = None,
                 **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field_name: Name of field that failed validation
            field_value: Value that failed validation
            validation_rule: Validation rule that was violated
        """
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class OptimizationError(QuantumPlannerError):
    """Errors during quantum optimization process."""
    
    def __init__(self, 
                 message: str,
                 optimization_type: Optional[str] = None,
                 iteration: Optional[int] = None,
                 objective_value: Optional[float] = None,
                 **kwargs):
        """
        Initialize optimization error.
        
        Args:
            message: Error description
            optimization_type: Type of optimization that failed
            iteration: Iteration number where error occurred
            objective_value: Last known objective function value
        """
        super().__init__(message, ErrorCategory.OPTIMIZATION, ErrorSeverity.HIGH, **kwargs)
        self.optimization_type = optimization_type
        self.iteration = iteration
        self.objective_value = objective_value


class QuantumStateError(QuantumPlannerError):
    """Errors related to quantum state operations."""
    
    def __init__(self, 
                 message: str,
                 state_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize quantum state error.
        
        Args:
            message: Error description
            state_info: Information about quantum state when error occurred
        """
        super().__init__(message, ErrorCategory.QUANTUM, ErrorSeverity.HIGH, **kwargs)
        self.state_info = state_info or {}


class ResourceConstraintError(QuantumPlannerError):
    """Errors related to resource constraints."""
    
    def __init__(self, 
                 message: str,
                 resource_type: Optional[str] = None,
                 required: Optional[float] = None,
                 available: Optional[float] = None,
                 **kwargs):
        """
        Initialize resource constraint error.
        
        Args:
            message: Error description
            resource_type: Type of constrained resource
            required: Required resource amount
            available: Available resource amount
        """
        super().__init__(message, ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM, **kwargs)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class DependencyError(QuantumPlannerError):
    """Errors related to task dependencies."""
    
    def __init__(self, 
                 message: str,
                 task_id: Optional[str] = None,
                 dependency_chain: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize dependency error.
        
        Args:
            message: Error description
            task_id: ID of task with dependency issue
            dependency_chain: Chain of dependencies involved
        """
        super().__init__(message, ErrorCategory.DEPENDENCY, ErrorSeverity.MEDIUM, **kwargs)
        self.task_id = task_id
        self.dependency_chain = dependency_chain or []


class ConfigurationError(QuantumPlannerError):
    """Errors related to system configuration."""
    
    def __init__(self, 
                 message: str,
                 config_key: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            config_key: Configuration key that caused error
            config_value: Configuration value that caused error
        """
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, **kwargs)
        self.config_key = config_key
        self.config_value = config_value


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides error tracking, recovery strategies, and monitoring.
    """
    
    def __init__(self, max_error_history: int = 1000):
        """
        Initialize error handler.
        
        Args:
            max_error_history: Maximum errors to keep in history
        """
        self.max_error_history = max_error_history
        self.error_history: List[QuantumPlannerError] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, callable] = {}
        
        # Set up default recovery strategies
        self._setup_default_recovery_strategies()
        
        logger.info("Initialized ErrorHandler with centralized error management")
    
    def handle_error(self, 
                    error: Exception,
                    operation: str = "unknown",
                    component: str = "unknown",
                    parameters: Dict[str, Any] = None,
                    attempt_recovery: bool = True) -> Optional[Any]:
        """
        Handle error with optional recovery.
        
        Args:
            error: Exception that occurred
            operation: Operation being performed
            component: Component where error occurred
            parameters: Operation parameters
            attempt_recovery: Whether to attempt error recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        import time
        
        # Create error context
        context = ErrorContext(
            operation=operation,
            component=component,
            parameters=parameters or {},
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            additional_info={}
        )
        
        # Wrap in QuantumPlannerError if needed
        if not isinstance(error, QuantumPlannerError):
            quantum_error = QuantumPlannerError(
                message=str(error),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                cause=error
            )
        else:
            quantum_error = error
            if not quantum_error.context:
                quantum_error.context = context
        
        # Record error
        self._record_error(quantum_error)
        
        # Attempt recovery if enabled
        if attempt_recovery and quantum_error.category in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[quantum_error.category](quantum_error)
                if recovery_result is not None:
                    logger.info(f"Successfully recovered from {quantum_error.category.value} error")
                    return recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {quantum_error.category.value} error: {recovery_error}")
        
        # Re-raise if no recovery or recovery failed
        raise quantum_error
    
    def _record_error(self, error: QuantumPlannerError) -> None:
        """Record error in history and update counts."""
        # Add to history
        self.error_history.append(error)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Update counts
        error_type = error.__class__.__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Check for error patterns
        self._analyze_error_patterns()
    
    def _setup_default_recovery_strategies(self) -> None:
        """Set up default error recovery strategies."""
        
        def validation_recovery(error: QuantumPlannerError) -> Optional[Any]:
            """Recovery strategy for validation errors."""
            if isinstance(error, ValidationError):
                logger.info(f"Attempting validation recovery for field: {error.field_name}")
                # Could implement default value assignment, field correction, etc.
                return None
            return None
        
        def optimization_recovery(error: QuantumPlannerError) -> Optional[Any]:
            """Recovery strategy for optimization errors."""
            if isinstance(error, OptimizationError):
                logger.info("Attempting optimization recovery with fallback algorithm")
                # Could implement fallback to simpler optimization method
                return None
            return None
        
        def resource_recovery(error: QuantumPlannerError) -> Optional[Any]:
            """Recovery strategy for resource constraint errors."""
            if isinstance(error, ResourceConstraintError):
                logger.info(f"Attempting resource recovery for {error.resource_type}")
                # Could implement resource reallocation, task prioritization, etc.
                return None
            return None
        
        self.recovery_strategies = {
            ErrorCategory.VALIDATION: validation_recovery,
            ErrorCategory.OPTIMIZATION: optimization_recovery,
            ErrorCategory.RESOURCE: resource_recovery
        }
    
    def _analyze_error_patterns(self) -> None:
        """Analyze error history for patterns and potential issues."""
        if len(self.error_history) < 10:
            return
        
        # Check for error frequency spikes
        recent_errors = self.error_history[-10:]
        recent_categories = [e.category for e in recent_errors]
        
        # Alert on repeated errors of same category
        for category in ErrorCategory:
            category_count = recent_categories.count(category)
            if category_count >= 5:  # 50% of recent errors
                logger.warning(f"High frequency of {category.value} errors detected: {category_count}/10")
        
        # Check for critical error patterns
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        if len(critical_errors) >= 2:
            logger.critical(f"Multiple critical errors detected: {len(critical_errors)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "severity_distribution": {}}
        
        # Category distribution
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Recent error rate
        recent_errors = [e for e in self.error_history[-100:]]
        
        return {
            "total_errors": len(self.error_history),
            "categories": category_counts,
            "severity_distribution": severity_counts,
            "recent_error_count": len(recent_errors),
            "error_types": dict(self.error_counts),
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        }
    
    def clear_error_history(self) -> None:
        """Clear error history (use with caution)."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Cleared error history")


@contextmanager
def error_context(operation: str, 
                 component: str = "unknown",
                 parameters: Dict[str, Any] = None,
                 error_handler: Optional[ErrorHandler] = None):
    """
    Context manager for automatic error handling.
    
    Usage:
        with error_context("optimization", "quantum_annealing", {"iterations": 100}):
            # Code that might raise errors
            result = some_operation()
    """
    try:
        yield
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, operation, component, parameters)
        else:
            # Create temporary error handler
            temp_handler = ErrorHandler()
            temp_handler.handle_error(e, operation, component, parameters)


def validate_and_handle(validator_func: callable, 
                       error_message: str = "Validation failed",
                       **error_kwargs) -> callable:
    """
    Decorator for validation with automatic error handling.
    
    Args:
        validator_func: Function that returns True if validation passes
        error_message: Error message if validation fails
        error_kwargs: Additional error parameters
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not validator_func(*args, **kwargs):
                raise ValidationError(
                    message=error_message,
                    **error_kwargs
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3,
                  backoff_factor: float = 1.0,
                  retry_on: tuple = (QuantumPlannerError,)) -> callable:
    """
    Decorator for automatic retry on specific errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.info(f"Retrying {func.__name__} after {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                except Exception as e:
                    # Don't retry on unexpected errors
                    raise
            
            # Should never reach here, but just in case
            raise last_exception
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()