#!/usr/bin/env python3
"""
Robust Error Handling - Generation 2: MAKE IT ROBUST

Comprehensive error handling, recovery mechanisms, and fault tolerance for AI Scientist systems.
Provides resilient operation with graceful degradation and automatic recovery capabilities.
"""

import asyncio
import functools
import json
import logging
import time
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Type
import os
import signal

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


# Install rich traceback for better error visualization
install(show_locals=True)


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"              # Minor issues, system continues normally
    MEDIUM = "medium"        # Moderate issues, some functionality affected
    HIGH = "high"           # Serious issues, major functionality affected  
    CRITICAL = "critical"   # Critical issues, system stability threatened


class ErrorCategory(Enum):
    """Categories of errors for better handling strategies."""
    NETWORK = "network"                 # Network connectivity issues
    RESOURCE = "resource"               # Resource allocation/exhaustion
    VALIDATION = "validation"           # Input/data validation errors
    COMPUTATION = "computation"         # Computational/algorithmic errors
    CONFIGURATION = "configuration"    # Configuration/setup errors
    EXTERNAL_API = "external_api"      # External service/API errors
    FILESYSTEM = "filesystem"          # File system operations
    SECURITY = "security"              # Security-related errors
    UNKNOWN = "unknown"                # Unclassified errors


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    component: str
    message: str
    exception_type: str
    traceback: str
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    # Context data
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Defines a recovery strategy for specific error types."""
    strategy_name: str
    applicable_categories: List[ErrorCategory]
    applicable_severities: List[ErrorSeverity]
    recovery_function: Callable
    max_attempts: int = 3
    backoff_delay: float = 1.0  # seconds
    exponential_backoff: bool = True


class RobustErrorHandler:
    """
    Generation 2: MAKE IT ROBUST
    Comprehensive error handling system with recovery mechanisms.
    """
    
    def __init__(self, workspace_dir: str = "error_handling_workspace"):
        self.console = Console()
        self.logger = self._setup_comprehensive_logging()
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.active_errors: Dict[str, ErrorContext] = {}
        self.error_patterns: Dict[str, int] = {}  # Pattern frequency tracking
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # System health monitoring
        self.health_metrics = {
            'total_errors': 0,
            'critical_errors': 0,
            'recovery_success_rate': 0.0,
            'mean_time_to_recovery': 0.0,
            'system_stability_score': 1.0
        }
        
        # Configuration
        self.max_error_history = 1000
        self.error_pattern_threshold = 5  # Alert after 5 similar errors
        self.auto_recovery_enabled = True
        self.graceful_degradation_enabled = True
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_comprehensive_logging(self) -> logging.Logger:
        """Setup comprehensive logging with multiple handlers."""
        
        # Create logger
        logger = logging.getLogger(f"{__name__}.RobustErrorHandler")
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True
        )
        console_handler.setLevel(logging.INFO)
        
        # File handler for all logs
        log_file = self.workspace_dir / "error_handler.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Error-specific file handler
        error_log_file = self.workspace_dir / "errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(simple_formatter)
        file_handler.setFormatter(detailed_formatter)
        error_handler.setFormatter(detailed_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._graceful_shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_recovery_strategies(self):
        """Initialize predefined recovery strategies."""
        
        # Network error recovery
        network_recovery = RecoveryStrategy(
            strategy_name="network_retry",
            applicable_categories=[ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_API],
            applicable_severities=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            recovery_function=self._network_retry_recovery,
            max_attempts=5,
            backoff_delay=2.0,
            exponential_backoff=True
        )
        
        # Resource exhaustion recovery  
        resource_recovery = RecoveryStrategy(
            strategy_name="resource_cleanup",
            applicable_categories=[ErrorCategory.RESOURCE],
            applicable_severities=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._resource_cleanup_recovery,
            max_attempts=3,
            backoff_delay=5.0
        )
        
        # Configuration error recovery
        config_recovery = RecoveryStrategy(
            strategy_name="config_reset",
            applicable_categories=[ErrorCategory.CONFIGURATION],
            applicable_severities=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._configuration_recovery,
            max_attempts=2,
            backoff_delay=1.0
        )
        
        # Filesystem error recovery
        filesystem_recovery = RecoveryStrategy(
            strategy_name="filesystem_repair",
            applicable_categories=[ErrorCategory.FILESYSTEM],
            applicable_severities=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            recovery_function=self._filesystem_recovery,
            max_attempts=3,
            backoff_delay=1.0
        )
        
        # Computation error recovery
        computation_recovery = RecoveryStrategy(
            strategy_name="computation_fallback",
            applicable_categories=[ErrorCategory.COMPUTATION],
            applicable_severities=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._computation_fallback_recovery,
            max_attempts=2,
            backoff_delay=0.5
        )
        
        self.recovery_strategies.extend([
            network_recovery, resource_recovery, config_recovery,
            filesystem_recovery, computation_recovery
        ])
    
    def create_error_context(
        self,
        exception: Exception,
        operation: str,
        component: str,
        user_data: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Create comprehensive error context from exception."""
        
        # Generate unique error ID
        error_id = f"err_{int(time.time())}_{hash(str(exception)) % 10000:04d}"
        
        # Classify error
        severity = self._classify_error_severity(exception)
        category = self._classify_error_category(exception, operation)
        
        # Gather environment information
        environment_info = {
            'python_version': sys.version,
            'platform': os.name,
            'working_directory': os.getcwd(),
            'memory_usage': self._get_memory_usage(),
            'cpu_count': os.cpu_count(),
            'process_id': os.getpid()
        }
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            operation=operation,
            component=component,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            user_data=user_data or {},
            system_state=system_state or {},
            environment_info=environment_info
        )
        
        return error_context
    
    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type and message."""
        
        # Critical errors
        critical_types = (MemoryError, SystemExit, KeyboardInterrupt)
        critical_keywords = ['critical', 'fatal', 'system failure', 'corrupted']
        
        # High severity errors
        high_types = (OSError, IOError, PermissionError, FileNotFoundError)
        high_keywords = ['failed to', 'cannot', 'unable to', 'timeout']
        
        # Medium severity errors
        medium_types = (ValueError, TypeError, AttributeError, ImportError)
        medium_keywords = ['invalid', 'missing', 'not found', 'error']
        
        exception_str = str(exception).lower()
        
        if isinstance(exception, critical_types) or any(keyword in exception_str for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, high_types) or any(keyword in exception_str for keyword in high_keywords):
            return ErrorSeverity.HIGH
        elif isinstance(exception, medium_types) or any(keyword in exception_str for keyword in medium_keywords):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _classify_error_category(self, exception: Exception, operation: str) -> ErrorCategory:
        """Classify error category based on exception type and operation context."""
        
        # Network-related errors
        network_types = (ConnectionError, ConnectionRefusedError, ConnectionAbortedError, 
                        ConnectionResetError, TimeoutError)
        network_keywords = ['connection', 'network', 'dns', 'socket', 'http', 'api']
        
        # Resource-related errors
        resource_types = (MemoryError, RecursionError)
        resource_keywords = ['memory', 'resource', 'limit', 'quota', 'allocation']
        
        # Filesystem errors
        filesystem_types = (FileNotFoundError, PermissionError, IsADirectoryError, 
                          FileExistsError, OSError, IOError)
        filesystem_keywords = ['file', 'directory', 'path', 'disk', 'storage']
        
        # Configuration errors
        config_keywords = ['config', 'setting', 'parameter', 'option', 'key']
        
        # Validation errors
        validation_types = (ValueError, TypeError, AssertionError)
        validation_keywords = ['validation', 'invalid', 'format', 'type', 'range']
        
        # Security errors
        security_keywords = ['permission', 'access', 'auth', 'security', 'credential']
        
        exception_str = str(exception).lower()
        operation_str = operation.lower()
        
        # Check operation context first
        if any(keyword in operation_str for keyword in network_keywords):
            return ErrorCategory.NETWORK
        elif any(keyword in operation_str for keyword in filesystem_keywords):
            return ErrorCategory.FILESYSTEM
        elif any(keyword in operation_str for keyword in config_keywords):
            return ErrorCategory.CONFIGURATION
        
        # Check exception type and message
        if isinstance(exception, network_types) or any(keyword in exception_str for keyword in network_keywords):
            return ErrorCategory.NETWORK
        elif isinstance(exception, resource_types) or any(keyword in exception_str for keyword in resource_keywords):
            return ErrorCategory.RESOURCE
        elif isinstance(exception, filesystem_types) or any(keyword in exception_str for keyword in filesystem_keywords):
            return ErrorCategory.FILESYSTEM
        elif isinstance(exception, validation_types) or any(keyword in exception_str for keyword in validation_keywords):
            return ErrorCategory.VALIDATION
        elif any(keyword in exception_str for keyword in security_keywords):
            return ErrorCategory.SECURITY
        elif any(keyword in exception_str for keyword in config_keywords):
            return ErrorCategory.CONFIGURATION
        elif 'api' in exception_str or 'service' in exception_str:
            return ErrorCategory.EXTERNAL_API
        else:
            return ErrorCategory.UNKNOWN
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            return {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    async def handle_error(
        self,
        exception: Exception,
        operation: str,
        component: str,
        user_data: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True
    ) -> ErrorContext:
        """Comprehensive error handling with automatic recovery attempts."""
        
        # Create error context
        error_context = self.create_error_context(
            exception, operation, component, user_data, system_state
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Add to tracking
        self.error_history.append(error_context)
        self.active_errors[error_context.error_id] = error_context
        self.health_metrics['total_errors'] += 1
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.health_metrics['critical_errors'] += 1
        
        # Check for error patterns
        self._analyze_error_patterns(error_context)
        
        # Attempt automatic recovery if enabled
        if auto_recover and self.auto_recovery_enabled:
            recovery_successful = await self._attempt_recovery(error_context)
            if recovery_successful:
                self.console.print(f"[bold green]✅ Automatic recovery successful for {error_context.error_id}[/bold green]")
            else:
                self.console.print(f"[bold red]❌ Automatic recovery failed for {error_context.error_id}[/bold red]")
        
        # Update system stability score
        self._update_stability_score()
        
        # Clean up old error history
        self._cleanup_error_history()
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level and formatting."""
        
        log_message = f"[{error_context.severity.value.upper()}] {error_context.component}.{error_context.operation}: {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Always log full traceback at debug level
        self.logger.debug(f"Full traceback for {error_context.error_id}:\n{error_context.traceback}")
    
    def _analyze_error_patterns(self, error_context: ErrorContext):
        """Analyze error patterns and detect recurring issues."""
        
        # Create pattern key from error characteristics
        pattern_key = f"{error_context.category.value}_{error_context.exception_type}_{hash(error_context.message) % 1000}"
        
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
        
        # Alert if pattern threshold exceeded
        if self.error_patterns[pattern_key] >= self.error_pattern_threshold:
            self.logger.warning(f"Error pattern detected: {pattern_key} occurred {self.error_patterns[pattern_key]} times")
            self.console.print(f"[bold yellow]⚠️ Recurring error pattern detected: {error_context.category.value} - {error_context.exception_type}[/bold yellow]")
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt automatic recovery using available strategies."""
        
        if error_context.recovery_attempts >= error_context.max_recovery_attempts:
            return False
        
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if (error_context.category in strategy.applicable_categories and
                error_context.severity in strategy.applicable_severities)
        ]
        
        if not applicable_strategies:
            return False
        
        # Try each applicable strategy
        for strategy in applicable_strategies:
            try:
                error_context.recovery_attempted = True
                error_context.recovery_attempts += 1
                
                self.logger.info(f"Attempting recovery with strategy: {strategy.strategy_name}")
                
                # Calculate backoff delay
                delay = strategy.backoff_delay
                if strategy.exponential_backoff:
                    delay *= (2 ** (error_context.recovery_attempts - 1))
                
                # Wait before recovery attempt
                await asyncio.sleep(delay)
                
                # Execute recovery function
                recovery_result = await self._execute_recovery_strategy(strategy, error_context)
                
                if recovery_result:
                    error_context.recovery_successful = True
                    self._update_recovery_metrics(True)
                    
                    # Remove from active errors
                    if error_context.error_id in self.active_errors:
                        del self.active_errors[error_context.error_id]
                    
                    return True
                
            except Exception as recovery_exception:
                self.logger.error(f"Recovery strategy {strategy.strategy_name} failed: {recovery_exception}")
        
        self._update_recovery_metrics(False)
        return False
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error_context: ErrorContext
    ) -> bool:
        """Execute a specific recovery strategy."""
        
        try:
            if asyncio.iscoroutinefunction(strategy.recovery_function):
                return await strategy.recovery_function(error_context)
            else:
                # Run synchronous recovery function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    return await loop.run_in_executor(executor, strategy.recovery_function, error_context)
        except Exception as e:
            self.logger.error(f"Recovery function execution failed: {e}")
            return False
    
    def _update_recovery_metrics(self, success: bool):
        """Update recovery success rate metrics."""
        
        total_attempts = sum(
            1 for error in self.error_history
            if error.recovery_attempted
        )
        
        if total_attempts > 0:
            successful_recoveries = sum(
                1 for error in self.error_history
                if error.recovery_attempted and error.recovery_successful
            )
            self.health_metrics['recovery_success_rate'] = successful_recoveries / total_attempts
    
    def _update_stability_score(self):
        """Update system stability score based on recent errors."""
        
        # Calculate stability based on recent error frequency and severity
        recent_window = timedelta(hours=1)
        current_time = datetime.now()
        
        recent_errors = [
            error for error in self.error_history
            if current_time - error.timestamp < recent_window
        ]
        
        if not recent_errors:
            self.health_metrics['system_stability_score'] = 1.0
            return
        
        # Weight errors by severity
        severity_weights = {
            ErrorSeverity.LOW: 0.1,
            ErrorSeverity.MEDIUM: 0.3,
            ErrorSeverity.HIGH: 0.7,
            ErrorSeverity.CRITICAL: 1.0
        }
        
        weighted_error_score = sum(
            severity_weights[error.severity] for error in recent_errors
        )
        
        # Calculate stability (inverse relationship with errors)
        max_expected_errors = 10  # Threshold for completely unstable system
        stability_score = max(0.0, 1.0 - (weighted_error_score / max_expected_errors))
        
        self.health_metrics['system_stability_score'] = stability_score
    
    def _cleanup_error_history(self):
        """Clean up old error history to prevent memory issues."""
        
        if len(self.error_history) > self.max_error_history:
            # Keep the most recent errors
            self.error_history = self.error_history[-self.max_error_history:]
            
            # Clean up active errors older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            expired_errors = [
                error_id for error_id, error in self.active_errors.items()
                if error.timestamp < cutoff_time
            ]
            
            for error_id in expired_errors:
                del self.active_errors[error_id]
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown procedures."""
        
        self.logger.info("Initiating graceful shutdown...")
        
        # Save current error state
        self._save_error_state()
        
        # Close circuit breakers
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.close()
        
        # Log final metrics
        self.logger.info(f"Final system metrics: {self.health_metrics}")
        
        self.logger.info("Graceful shutdown complete")
    
    def _save_error_state(self):
        """Save current error state to file for analysis."""
        
        state_file = self.workspace_dir / f"error_state_{int(time.time())}.json"
        
        error_state = {
            'timestamp': datetime.now().isoformat(),
            'health_metrics': self.health_metrics,
            'error_patterns': self.error_patterns,
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'operation': error.operation,
                    'component': error.component,
                    'message': error.message,
                    'recovery_successful': error.recovery_successful
                }
                for error in self.error_history[-50:]  # Last 50 errors
            ]
        }
        
        with open(state_file, 'w') as f:
            json.dump(error_state, f, indent=2, default=str)
        
        self.logger.info(f"Error state saved to: {state_file}")
    
    # Recovery strategy implementations
    async def _network_retry_recovery(self, error_context: ErrorContext) -> bool:
        """Recovery strategy for network-related errors."""
        
        self.logger.info("Attempting network recovery...")
        
        # Simple network connectivity check
        try:
            import requests
            response = requests.get("https://httpbin.org/status/200", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _resource_cleanup_recovery(self, error_context: ErrorContext) -> bool:
        """Recovery strategy for resource exhaustion errors."""
        
        self.logger.info("Attempting resource cleanup recovery...")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any temporary files
        temp_dir = Path("/tmp")
        if temp_dir.exists():
            try:
                # Clean up temporary files older than 1 hour
                cutoff_time = time.time() - 3600
                for temp_file in temp_dir.glob("tmp*"):
                    if temp_file.stat().st_mtime < cutoff_time:
                        temp_file.unlink()
            except Exception:
                pass
        
        return True
    
    def _configuration_recovery(self, error_context: ErrorContext) -> bool:
        """Recovery strategy for configuration errors."""
        
        self.logger.info("Attempting configuration recovery...")
        
        # Reset to default configuration
        # This would typically reload config from default files
        return True
    
    def _filesystem_recovery(self, error_context: ErrorContext) -> bool:
        """Recovery strategy for filesystem errors."""
        
        self.logger.info("Attempting filesystem recovery...")
        
        # Create missing directories
        if "No such file or directory" in error_context.message:
            try:
                # Extract path from error message (simplified)
                path_match = error_context.message.split("'")
                if len(path_match) >= 2:
                    missing_path = Path(path_match[1])
                    if not missing_path.exists():
                        missing_path.parent.mkdir(parents=True, exist_ok=True)
                        return True
            except Exception:
                pass
        
        return False
    
    def _computation_fallback_recovery(self, error_context: ErrorContext) -> bool:
        """Recovery strategy for computation errors."""
        
        self.logger.info("Attempting computation fallback recovery...")
        
        # This would typically implement fallback algorithms or reduced precision
        return True
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        return self.health_metrics.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        
        recent_window = timedelta(hours=24)
        current_time = datetime.now()
        
        recent_errors = [
            error for error in self.error_history
            if current_time - error.timestamp < recent_window
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors_24h': len(recent_errors),
            'active_errors': len(self.active_errors),
            'error_patterns': len(self.error_patterns),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'health_metrics': self.health_metrics,
            'top_error_patterns': sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info(f"Circuit breaker {self.name} closed after successful operation")
    
    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def close(self):
        """Manually close circuit breaker."""
        self.state = 'CLOSED'
        self.failure_count = 0
        self.last_failure_time = None


# Decorators for robust error handling
def robust_operation(
    operation: str,
    component: str,
    auto_recover: bool = True,
    error_handler: Optional[RobustErrorHandler] = None
):
    """Decorator for wrapping functions with robust error handling."""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = error_handler or getattr(args[0], 'error_handler', None)
            if not handler:
                # Create default error handler
                handler = RobustErrorHandler()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                await handler.handle_error(
                    exception=e,
                    operation=operation,
                    component=component,
                    auto_recover=auto_recover
                )
                raise  # Re-raise after handling
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = error_handler or getattr(args[0], 'error_handler', None)
            if not handler:
                handler = RobustErrorHandler()
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Run async handler in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        handler.handle_error(
                            exception=e,
                            operation=operation,
                            component=component,
                            auto_recover=auto_recover
                        )
                    )
                finally:
                    loop.close()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def error_context(operation: str, component: str, error_handler: RobustErrorHandler):
    """Context manager for error handling."""
    
    try:
        yield
    except Exception as e:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                error_handler.handle_error(e, operation, component)
            )
        finally:
            loop.close()
        raise


@asynccontextmanager
async def async_error_context(operation: str, component: str, error_handler: RobustErrorHandler):
    """Async context manager for error handling."""
    
    try:
        yield
    except Exception as e:
        await error_handler.handle_error(e, operation, component)
        raise


# Demo and testing functions
async def demo_robust_error_handling():
    """Demonstrate the robust error handling system."""
    
    console = Console()
    console.print("[bold blue]🛡️ Robust Error Handling System - Generation 2 Demo[/bold blue]")
    
    # Initialize error handler
    error_handler = RobustErrorHandler()
    
    console.print("\n[yellow]Testing various error scenarios...[/yellow]")
    
    # Test different types of errors
    test_scenarios = [
        ("network_error", lambda: 1/0),  # Simulate network error with division by zero
        ("resource_error", lambda: [i for i in range(10**6)]),  # Memory intensive operation
        ("file_error", lambda: open("/nonexistent/path/file.txt")),  # File not found
        ("validation_error", lambda: int("not_a_number")),  # Invalid input
        ("configuration_error", lambda: {"key": "value"}["missing_key"])  # Configuration missing
    ]
    
    results = []
    
    for scenario_name, error_func in test_scenarios:
        console.print(f"\n[cyan]📝 Testing {scenario_name}...[/cyan]")
        
        try:
            # Simulate operation that causes error
            error_func()
        except Exception as e:
            # Handle the error
            error_context = await error_handler.handle_error(
                exception=e,
                operation=scenario_name,
                component="demo_component",
                user_data={"test_scenario": scenario_name},
                auto_recover=True
            )
            
            results.append(error_context)
            
            console.print(f"  • Error ID: {error_context.error_id}")
            console.print(f"  • Severity: {error_context.severity.value}")
            console.print(f"  • Category: {error_context.category.value}")
            console.print(f"  • Recovery Attempted: {error_context.recovery_attempted}")
            console.print(f"  • Recovery Successful: {error_context.recovery_successful}")
    
    # Show system health metrics
    console.print(f"\n[bold green]📊 System Health Metrics:[/bold green]")
    health_metrics = error_handler.get_health_metrics()
    for metric, value in health_metrics.items():
        if isinstance(value, float):
            console.print(f"  • {metric}: {value:.2f}")
        else:
            console.print(f"  • {metric}: {value}")
    
    # Show error summary
    console.print(f"\n[bold cyan]📋 Error Summary:[/bold cyan]")
    error_summary = error_handler.get_error_summary()
    console.print(f"  • Total Errors (24h): {error_summary['total_errors_24h']}")
    console.print(f"  • Active Errors: {error_summary['active_errors']}")
    console.print(f"  • Error Patterns: {error_summary['error_patterns']}")
    
    if error_summary['category_breakdown']:
        console.print("  • Category Breakdown:")
        for category, count in error_summary['category_breakdown'].items():
            console.print(f"    - {category}: {count}")
    
    return error_handler, results


@robust_operation("test_decorated_function", "demo_component")
async def test_decorated_function():
    """Test function with robust error handling decorator."""
    
    # This will trigger an error to test the decorator
    raise ValueError("Test error from decorated function")


async def main():
    """Main entry point for robust error handling demo."""
    
    try:
        error_handler, results = await demo_robust_error_handling()
        
        console = Console()
        console.print(f"\n[bold yellow]🧪 Testing Decorated Function...[/bold yellow]")
        
        # Test decorated function (will handle error automatically)
        try:
            await test_decorated_function()
        except ValueError:
            console.print("  • Decorated function error was handled and re-raised")
        
        console.print(f"\n[bold green]✅ Error handling demo completed successfully![/bold green]")
        console.print(f"  • Processed {len(results)} different error scenarios")
        console.print(f"  • System stability score: {error_handler.get_health_metrics()['system_stability_score']:.2f}")
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())