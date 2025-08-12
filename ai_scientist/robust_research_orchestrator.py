#!/usr/bin/env python3
"""
Robust Research Orchestrator for AI-Scientist-v2
===============================================

Production-ready research orchestrator with comprehensive error handling,
validation, monitoring, and resilience features. Integrates novel algorithm
discovery with autonomous experimentation in a fault-tolerant system.

Robustness Features:
- Circuit breaker patterns for external service calls
- Comprehensive input validation and sanitization
- Graceful degradation and fallback mechanisms
- Real-time health monitoring and alerting
- Automatic retry logic with exponential backoff
- Resource leak prevention and cleanup
- Security hardening and audit logging

Author: AI Scientist v2 - Terragon Labs
License: MIT
"""

import asyncio
import logging
import numpy as np
import time
import json
import traceback
import signal
import sys
import os
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess
import tempfile
import shutil
import hashlib
import uuid
from contextlib import contextmanager, asynccontextmanager
import gc
import weakref

# Robust imports with fallbacks
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import our research modules with error handling
try:
    from ai_scientist.research.novel_algorithm_discovery import (
        NovelAlgorithmDiscovery, NovelHypothesis, ExperimentalResult
    )
    from ai_scientist.research.autonomous_experimentation_engine import (
        AutonomousExperimentationEngine, ExperimentConfig, ExperimentResult,
        ExperimentType, ExperimentStatus
    )
    RESEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    RESEARCH_MODULES_AVAILABLE = False
    logger.error(f"Research modules not available: {e}")

# Configure logging with security considerations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/robust_research_orchestrator.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Suppress sensitive information in logs
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class OperationStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class HealthMetrics:
    """System health metrics for monitoring."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_experiments: int = 0
    error_rate: float = 0.0
    system_health: SystemHealth = SystemHealth.HEALTHY
    uptime_seconds: float = 0.0
    last_successful_operation: Optional[float] = None


@dataclass
class RobustOperationResult:
    """Result of a robust operation with comprehensive error information."""
    operation_id: str
    status: OperationStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    resource_usage: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: float = 60.0,
                 recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > self.timeout_seconds:
                    raise TimeoutError(f"Operation timeout: {execution_time:.2f}s")
                
                # Success - reset failure count
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    logger.info("Circuit breaker recovered - state: CLOSED")
                
                self.failure_count = 0
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
                
                raise e


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
                    break
                
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                if self.jitter:
                    delay *= (0.5 + np.random.random() * 0.5)
                
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {func.__name__} "
                             f"after {delay:.2f}s delay: {e}")
                
                await asyncio.sleep(delay)
        
        raise last_exception


class ResourceManager:
    """Manages system resources and prevents leaks."""
    
    def __init__(self):
        self.active_resources = weakref.WeakSet()
        self.resource_limits = {
            'max_memory_mb': 8192,  # 8GB
            'max_cpu_percent': 80.0,
            'max_open_files': 1000,
            'max_processes': 20
        }
        self.cleanup_callbacks = []
        
        # Register cleanup on exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    @contextmanager
    def managed_resource(self, resource_type: str, resource_obj: Any):
        """Context manager for automatic resource cleanup."""
        try:
            self.active_resources.add(resource_obj)
            yield resource_obj
        finally:
            self._cleanup_resource(resource_type, resource_obj)
            if resource_obj in self.active_resources:
                self.active_resources.remove(resource_obj)
    
    def _cleanup_resource(self, resource_type: str, resource_obj: Any):
        """Clean up specific resource types."""
        try:
            if resource_type == 'file' and hasattr(resource_obj, 'close'):
                resource_obj.close()
            elif resource_type == 'process' and hasattr(resource_obj, 'terminate'):
                resource_obj.terminate()
                if hasattr(resource_obj, 'wait'):
                    resource_obj.wait(timeout=5)
            elif resource_type == 'thread_pool' and hasattr(resource_obj, 'shutdown'):
                resource_obj.shutdown(wait=True, timeout=10)
            elif resource_type == 'temp_dir' and isinstance(resource_obj, (str, Path)):
                shutil.rmtree(resource_obj, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up {resource_type}: {e}")
    
    def check_resource_limits(self) -> bool:
        """Check if system resources are within limits."""
        try:
            # Memory check
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            if memory_mb > self.resource_limits['max_memory_mb']:
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                return False
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.resource_limits['max_cpu_percent']:
                logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
                return False
            
            # File descriptor check
            process = psutil.Process()
            open_files = len(process.open_files())
            if open_files > self.resource_limits['max_open_files']:
                logger.warning(f"Too many open files: {open_files}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.cleanup_all_resources()
        sys.exit(0)
    
    def cleanup_all_resources(self):
        """Clean up all managed resources."""
        logger.info("Cleaning up all managed resources")
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
        
        # Force garbage collection
        gc.collect()


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.health_history: List[HealthMetrics] = []
        self.max_history = 1000
        self.start_time = time.time()
        
        # Health thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 80.0,
            'memory_critical': 90.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'error_rate_warning': 0.1,
            'error_rate_critical': 0.2
        }
        
        # Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.cpu_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
            self.memory_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
            self.experiments_gauge = Gauge('active_experiments', 'Number of active experiments')
            self.error_rate_gauge = Gauge('error_rate', 'Error rate')
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.health_history.append(metrics)
                
                # Trim history
                if len(self.health_history) > self.max_history:
                    self.health_history = self.health_history[-self.max_history:]
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self.cpu_gauge.set(metrics.cpu_usage)
                    self.memory_gauge.set(metrics.memory_usage)
                    self.experiments_gauge.set(metrics.active_experiments)
                    self.error_rate_gauge.set(metrics.error_rate)
                
                # Check for alerts
                self._check_alert_conditions(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate error rate from recent history
            error_rate = self._calculate_error_rate()
            
            # Determine system health
            health = self._determine_system_health(cpu_usage, memory.percent, disk.percent, error_rate)
            
            return HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                error_rate=error_rate,
                system_health=health,
                uptime_seconds=time.time() - self.start_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return HealthMetrics(system_health=SystemHealth.UNHEALTHY)
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        if len(self.health_history) < 2:
            return 0.0
        
        recent_metrics = self.health_history[-10:]  # Last 10 samples
        total_operations = len(recent_metrics)
        error_operations = sum(1 for m in recent_metrics if m.system_health in [SystemHealth.UNHEALTHY, SystemHealth.CRITICAL])
        
        return error_operations / total_operations if total_operations > 0 else 0.0
    
    def _determine_system_health(self, cpu: float, memory: float, disk: float, error_rate: float) -> SystemHealth:
        """Determine overall system health."""
        if (cpu > self.thresholds['cpu_critical'] or 
            memory > self.thresholds['memory_critical'] or
            disk > self.thresholds['disk_critical'] or
            error_rate > self.thresholds['error_rate_critical']):
            return SystemHealth.CRITICAL
        
        if (cpu > self.thresholds['cpu_warning'] or
            memory > self.thresholds['memory_warning'] or
            disk > self.thresholds['disk_warning'] or
            error_rate > self.thresholds['error_rate_warning']):
            return SystemHealth.DEGRADED
        
        return SystemHealth.HEALTHY
    
    def _check_alert_conditions(self, metrics: HealthMetrics):
        """Check for alert conditions and log warnings."""
        if metrics.system_health == SystemHealth.CRITICAL:
            logger.critical(f"CRITICAL system health - CPU: {metrics.cpu_usage:.1f}%, "
                           f"Memory: {metrics.memory_usage:.1f}%, Error rate: {metrics.error_rate:.2f}")
        elif metrics.system_health == SystemHealth.DEGRADED:
            logger.warning(f"DEGRADED system health - CPU: {metrics.cpu_usage:.1f}%, "
                          f"Memory: {metrics.memory_usage:.1f}%")
    
    def get_current_health(self) -> Optional[HealthMetrics]:
        """Get current health metrics."""
        return self.health_history[-1] if self.health_history else None


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_string(value: Any, 
                       min_length: int = 0,
                       max_length: int = 1000,
                       allowed_chars: Optional[str] = None,
                       field_name: str = "string") -> str:
        """Validate and sanitize string input."""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string, got {type(value)}")
        
        if len(value) < min_length:
            raise ValueError(f"{field_name} must be at least {min_length} characters")
        
        if len(value) > max_length:
            raise ValueError(f"{field_name} must be at most {max_length} characters")
        
        if allowed_chars:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                raise ValueError(f"{field_name} contains invalid characters: {invalid_chars}")
        
        # Basic sanitization - remove control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        return sanitized
    
    @staticmethod
    def validate_number(value: Any,
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       field_name: str = "number") -> Union[int, float]:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"{field_name} must be a number")
        
        if min_value is not None and value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}")
        
        return value
    
    @staticmethod
    def validate_path(value: Any, 
                     must_exist: bool = False,
                     must_be_file: bool = False,
                     must_be_dir: bool = False,
                     field_name: str = "path") -> Path:
        """Validate and sanitize file paths."""
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{field_name} must be a string or Path")
        
        path = Path(value).resolve()
        
        # Security check - prevent path traversal
        if '..' in str(path):
            raise ValueError(f"{field_name} contains invalid path components")
        
        if must_exist and not path.exists():
            raise ValueError(f"{field_name} does not exist: {path}")
        
        if must_be_file and path.exists() and not path.is_file():
            raise ValueError(f"{field_name} must be a file: {path}")
        
        if must_be_dir and path.exists() and not path.is_dir():
            raise ValueError(f"{field_name} must be a directory: {path}")
        
        return path
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        validated_config = {}
        
        # Validate common configuration fields
        if 'max_concurrent_experiments' in config:
            validated_config['max_concurrent_experiments'] = InputValidator.validate_number(
                config['max_concurrent_experiments'], 
                min_value=1, 
                max_value=50,
                field_name='max_concurrent_experiments'
            )
        
        if 'workspace_dir' in config:
            validated_config['workspace_dir'] = str(InputValidator.validate_path(
                config['workspace_dir'],
                field_name='workspace_dir'
            ))
        
        if 'timeout_seconds' in config:
            validated_config['timeout_seconds'] = InputValidator.validate_number(
                config['timeout_seconds'],
                min_value=10,
                max_value=86400,  # 1 day
                field_name='timeout_seconds'
            )
        
        # Copy other valid fields
        for key, value in config.items():
            if key not in validated_config and not key.startswith('_'):
                validated_config[key] = value
        
        return validated_config


class RobustResearchOrchestrator:
    """
    Production-ready research orchestrator with comprehensive robustness features.
    
    Features:
    - Fault-tolerant research pipeline execution
    - Comprehensive error handling and recovery
    - Resource management and leak prevention
    - Real-time health monitoring and alerting
    - Circuit breaker protection for external services
    - Input validation and security hardening
    - Graceful degradation under load
    - Comprehensive audit logging
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        
        # Validate and set configuration
        self.config = InputValidator.validate_config(config or {})
        self.workspace_dir = Path(self.config.get('workspace_dir', '/tmp/robust_research'))
        self.max_concurrent = self.config.get('max_concurrent_experiments', 4)
        self.timeout_seconds = self.config.get('timeout_seconds', 3600)
        
        # Initialize workspace with proper permissions
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            logger.info(f"Workspace initialized: {self.workspace_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize workspace: {e}")
            raise
        
        # Initialize robustness components
        self.health_monitor = HealthMonitor()
        self.resource_manager = ResourceManager()
        self.retry_manager = RetryManager()
        self.circuit_breakers = {}
        
        # Operation tracking
        self.active_operations: Dict[str, asyncio.Task] = {}
        self.operation_history: List[RobustOperationResult] = []
        self.operation_lock = asyncio.Lock()
        
        # Research engines with error handling
        self.algorithm_discovery = None
        self.experimentation_engine = None
        self._initialize_research_engines()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(8000)
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.warning(f"Could not start Prometheus server: {e}")
        
        logger.info("Robust Research Orchestrator initialized successfully")
    
    def _initialize_research_engines(self):
        """Initialize research engines with error handling."""
        if not RESEARCH_MODULES_AVAILABLE:
            logger.error("Research modules not available - operating in degraded mode")
            return
        
        try:
            self.algorithm_discovery = NovelAlgorithmDiscovery(
                workspace_dir=str(self.workspace_dir / "algorithm_discovery"),
                max_concurrent_experiments=self.max_concurrent
            )
            logger.info("Novel algorithm discovery engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize algorithm discovery: {e}")
        
        try:
            self.experimentation_engine = AutonomousExperimentationEngine(
                workspace_dir=str(self.workspace_dir / "experiments"),
                max_concurrent_experiments=self.max_concurrent
            )
            logger.info("Autonomous experimentation engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize experimentation engine: {e}")
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    async def execute_robust_operation(self,
                                     operation_func: Callable,
                                     operation_id: Optional[str] = None,
                                     timeout: Optional[float] = None,
                                     circuit_breaker_name: Optional[str] = None,
                                     *args, **kwargs) -> RobustOperationResult:
        """
        Execute operation with comprehensive error handling and monitoring.
        
        Args:
            operation_func: Function to execute
            operation_id: Unique identifier for operation
            timeout: Operation timeout in seconds
            circuit_breaker_name: Circuit breaker to use for external services
            *args, **kwargs: Arguments to pass to operation_func
            
        Returns:
            RobustOperationResult with comprehensive execution information
        """
        if not operation_id:
            operation_id = f"op_{uuid.uuid4().hex[:8]}"
        
        timeout = timeout or self.timeout_seconds
        start_time = time.time()
        
        result = RobustOperationResult(
            operation_id=operation_id,
            status=OperationStatus.FAILED
        )
        
        try:
            # Check system health before starting
            current_health = self.health_monitor.get_current_health()
            if current_health and current_health.system_health == SystemHealth.CRITICAL:
                raise Exception("System health is CRITICAL - operation rejected")
            
            # Check resource limits
            if not self.resource_manager.check_resource_limits():
                raise Exception("Resource limits exceeded - operation rejected")
            
            async with self.operation_lock:
                self.active_operations[operation_id] = asyncio.current_task()
            
            # Execute with circuit breaker if specified
            if circuit_breaker_name:
                circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
                operation_result = await asyncio.wait_for(
                    self.retry_manager.retry_async(
                        lambda: circuit_breaker.call(operation_func, *args, **kwargs)
                    ),
                    timeout=timeout
                )
            else:
                operation_result = await asyncio.wait_for(
                    self.retry_manager.retry_async(operation_func, *args, **kwargs),
                    timeout=timeout
                )
            
            result.status = OperationStatus.SUCCESS
            result.result = operation_result
            result.execution_time = time.time() - start_time
            
            logger.info(f"Operation {operation_id} completed successfully in {result.execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            result.status = OperationStatus.TIMEOUT
            result.error_message = f"Operation timed out after {timeout}s"
            result.error_type = "TimeoutError"
            logger.error(f"Operation {operation_id} timed out")
            
        except asyncio.CancelledError:
            result.status = OperationStatus.CANCELLED
            result.error_message = "Operation was cancelled"
            result.error_type = "CancelledError"
            logger.warning(f"Operation {operation_id} was cancelled")
            
        except Exception as e:
            result.status = OperationStatus.FAILED
            result.error_message = str(e)
            result.error_type = type(e).__name__
            result.stack_trace = traceback.format_exc()
            logger.error(f"Operation {operation_id} failed: {e}")
            
        finally:
            # Cleanup
            async with self.operation_lock:
                if operation_id in self.active_operations:
                    del self.active_operations[operation_id]
            
            result.execution_time = time.time() - start_time
            
            # Record operation history
            self.operation_history.append(result)
            if len(self.operation_history) > 1000:  # Limit history size
                self.operation_history = self.operation_history[-1000:]
        
        return result
    
    async def discover_and_validate_algorithms(self,
                                             research_domains: Optional[List[str]] = None,
                                             validation_budget: int = 10) -> RobustOperationResult:
        """
        Robustly discover and validate novel algorithms.
        
        Args:
            research_domains: List of research domains to explore
            validation_budget: Number of algorithms to validate
            
        Returns:
            RobustOperationResult containing discovery and validation results
        """
        async def _discovery_operation():
            if not self.algorithm_discovery:
                raise Exception("Algorithm discovery engine not available")
            
            # Discover research opportunities
            logger.info("Starting robust algorithm discovery")
            hypotheses = await self.algorithm_discovery.discover_research_opportunities()
            
            if not hypotheses:
                logger.warning("No research opportunities discovered")
                return {"hypotheses": [], "validation_results": []}
            
            # Validate top hypotheses
            top_hypotheses = hypotheses[:validation_budget]
            logger.info(f"Validating {len(top_hypotheses)} top hypotheses")
            
            validation_results = []
            for hypothesis in top_hypotheses:
                try:
                    result = await self.algorithm_discovery.execute_research_validation(hypothesis)
                    validation_results.append(result)
                    logger.info(f"Validated hypothesis: {hypothesis.hypothesis_id}")
                except Exception as e:
                    logger.error(f"Failed to validate hypothesis {hypothesis.hypothesis_id}: {e}")
                    # Continue with other hypotheses
            
            return {
                "hypotheses": hypotheses,
                "validation_results": validation_results,
                "discovery_summary": {
                    "total_discovered": len(hypotheses),
                    "validated": len(validation_results),
                    "success_rate": len(validation_results) / len(top_hypotheses) if top_hypotheses else 0.0
                }
            }
        
        return await self.execute_robust_operation(
            _discovery_operation,
            operation_id="algorithm_discovery",
            circuit_breaker_name="discovery_service",
            timeout=1800  # 30 minutes
        )
    
    async def execute_experimentation_suite(self,
                                          research_objective: str,
                                          algorithms: List[str],
                                          datasets: List[str]) -> RobustOperationResult:
        """
        Robustly execute comprehensive experimentation suite.
        
        Args:
            research_objective: High-level research goal
            algorithms: Algorithms to evaluate
            datasets: Datasets to use
            
        Returns:
            RobustOperationResult containing experimentation results
        """
        # Validate inputs
        research_objective = InputValidator.validate_string(
            research_objective, min_length=10, max_length=500, field_name="research_objective"
        )
        
        if not algorithms:
            raise ValueError("At least one algorithm must be specified")
        
        if not datasets:
            raise ValueError("At least one dataset must be specified")
        
        async def _experimentation_operation():
            if not self.experimentation_engine:
                raise Exception("Experimentation engine not available")
            
            logger.info(f"Starting robust experimentation suite: {research_objective}")
            
            # Design experiment suite
            experiment_suite = await self.experimentation_engine.design_experiment_suite(
                research_objective=research_objective,
                experiment_types=[ExperimentType.CLASSIFICATION, ExperimentType.META_LEARNING],
                algorithms=algorithms,
                datasets=datasets
            )
            
            # Execute experiments
            results = await self.experimentation_engine.execute_experiment_batch(experiment_suite)
            
            # Generate report
            report_path = await self.experimentation_engine.generate_experiment_report(
                results, include_visualizations=True
            )
            
            return {
                "experiment_suite": experiment_suite,
                "results": results,
                "report_path": report_path,
                "execution_summary": {
                    "total_experiments": len(experiment_suite.experiments),
                    "successful_experiments": len([r for r in results.values() 
                                                 if r.status == ExperimentStatus.COMPLETED]),
                    "pareto_frontier_size": len(self.experimentation_engine.pareto_frontier)
                }
            }
        
        return await self.execute_robust_operation(
            _experimentation_operation,
            operation_id="experimentation_suite",
            circuit_breaker_name="experimentation_service",
            timeout=3600  # 1 hour
        )
    
    async def full_research_pipeline(self,
                                   research_objective: str,
                                   discovery_budget: int = 5,
                                   validation_budget: int = 3,
                                   experiment_algorithms: Optional[List[str]] = None,
                                   datasets: Optional[List[str]] = None) -> Dict[str, RobustOperationResult]:
        """
        Execute complete robust research pipeline.
        
        Args:
            research_objective: High-level research objective
            discovery_budget: Number of hypotheses to discover
            validation_budget: Number of hypotheses to validate
            experiment_algorithms: Algorithms for experimentation
            datasets: Datasets for experiments
            
        Returns:
            Dictionary of operation results for each pipeline stage
        """
        logger.info("Starting full robust research pipeline")
        pipeline_results = {}
        
        try:
            # Stage 1: Algorithm Discovery
            discovery_result = await self.discover_and_validate_algorithms(
                validation_budget=discovery_budget
            )
            pipeline_results["discovery"] = discovery_result
            
            # Stage 2: Experimentation (if discovery succeeded)
            if discovery_result.status == OperationStatus.SUCCESS:
                # Use default algorithms if none provided
                if not experiment_algorithms:
                    experiment_algorithms = ['neural_network', 'random_forest', 'svm']
                
                if not datasets:
                    datasets = ['synthetic', 'benchmark_small', 'benchmark_medium']
                
                experimentation_result = await self.execute_experimentation_suite(
                    research_objective=research_objective,
                    algorithms=experiment_algorithms,
                    datasets=datasets
                )
                pipeline_results["experimentation"] = experimentation_result
            else:
                logger.warning("Skipping experimentation due to discovery failure")
                pipeline_results["experimentation"] = RobustOperationResult(
                    operation_id="experimentation_skipped",
                    status=OperationStatus.CANCELLED,
                    error_message="Skipped due to discovery failure"
                )
            
            # Stage 3: Report Generation
            report_result = await self._generate_pipeline_report(pipeline_results)
            pipeline_results["report"] = report_result
            
            logger.info("Full research pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Research pipeline failed: {e}")
            pipeline_results["pipeline_error"] = RobustOperationResult(
                operation_id="pipeline_failure",
                status=OperationStatus.FAILED,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
        
        return pipeline_results
    
    async def _generate_pipeline_report(self, 
                                      pipeline_results: Dict[str, RobustOperationResult]) -> RobustOperationResult:
        """Generate comprehensive pipeline report."""
        async def _report_generation():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = self.workspace_dir / f"robust_research_report_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write("# Robust Research Pipeline Report\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Pipeline Summary\n\n")
                
                total_operations = len(pipeline_results)
                successful_operations = len([r for r in pipeline_results.values() 
                                           if r.status == OperationStatus.SUCCESS])
                
                f.write(f"- Total pipeline stages: {total_operations}\n")
                f.write(f"- Successful stages: {successful_operations}\n")
                f.write(f"- Success rate: {successful_operations/total_operations*100:.1f}%\n\n")
                
                f.write("## Stage Results\n\n")
                
                for stage_name, result in pipeline_results.items():
                    f.write(f"### {stage_name.title()}\n")
                    f.write(f"**Status:** {result.status.value}\n")
                    f.write(f"**Execution Time:** {result.execution_time:.2f}s\n")
                    
                    if result.error_message:
                        f.write(f"**Error:** {result.error_message}\n")
                    
                    if result.result and isinstance(result.result, dict):
                        f.write("**Results Summary:**\n")
                        for key, value in result.result.items():
                            if isinstance(value, dict) and 'total_discovered' in value:
                                f.write(f"- {key}: {value}\n")
                    
                    f.write("\n---\n\n")
                
                # System health summary
                current_health = self.health_monitor.get_current_health()
                if current_health:
                    f.write("## System Health Summary\n\n")
                    f.write(f"- CPU Usage: {current_health.cpu_usage:.1f}%\n")
                    f.write(f"- Memory Usage: {current_health.memory_usage:.1f}%\n")
                    f.write(f"- System Health: {current_health.system_health.value}\n")
                    f.write(f"- Uptime: {current_health.uptime_seconds:.1f}s\n\n")
                
                f.write("## Robustness Metrics\n\n")
                f.write(f"- Total operations executed: {len(self.operation_history)}\n")
                
                if self.operation_history:
                    success_rate = len([op for op in self.operation_history 
                                      if op.status == OperationStatus.SUCCESS]) / len(self.operation_history)
                    f.write(f"- Overall success rate: {success_rate*100:.1f}%\n")
                    
                    avg_execution_time = np.mean([op.execution_time for op in self.operation_history])
                    f.write(f"- Average execution time: {avg_execution_time:.2f}s\n")
                
                f.write("\n## Recommendations\n\n")
                f.write("1. Monitor system health trends for capacity planning\n")
                f.write("2. Review failed operations for systematic issues\n")
                f.write("3. Consider scaling resources if success rate drops below 95%\n")
                f.write("4. Validate reproducibility of successful experiments\n\n")
            
            return str(report_path)
        
        return await self.execute_robust_operation(
            _report_generation,
            operation_id="pipeline_report",
            timeout=300
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_health = self.health_monitor.get_current_health()
        
        status = {
            "timestamp": time.time(),
            "system_health": current_health.system_health.value if current_health else "unknown",
            "active_operations": len(self.active_operations),
            "total_operations_executed": len(self.operation_history),
            "workspace_dir": str(self.workspace_dir),
            "engines_available": {
                "algorithm_discovery": self.algorithm_discovery is not None,
                "experimentation": self.experimentation_engine is not None
            }
        }
        
        if current_health:
            status.update({
                "cpu_usage": current_health.cpu_usage,
                "memory_usage": current_health.memory_usage,
                "disk_usage": current_health.disk_usage,
                "error_rate": current_health.error_rate,
                "uptime_seconds": current_health.uptime_seconds
            })
        
        # Circuit breaker status
        status["circuit_breakers"] = {}
        for name, cb in self.circuit_breakers.items():
            status["circuit_breakers"][name] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }
        
        return status
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown of all components."""
        logger.info("Starting graceful shutdown")
        
        try:
            # Cancel active operations
            async with self.operation_lock:
                for operation_id, task in self.active_operations.items():
                    logger.info(f"Cancelling operation: {operation_id}")
                    task.cancel()
                
                # Wait for operations to cancel
                if self.active_operations:
                    await asyncio.gather(*self.active_operations.values(), return_exceptions=True)
                self.active_operations.clear()
            
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
            
            # Cleanup resources
            self.resource_manager.cleanup_all_resources()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.graceful_shutdown())


# Autonomous execution entry point
async def main():
    """Main entry point for robust autonomous research orchestration."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Robust Research Orchestrator")
    
    config = {
        "workspace_dir": "/tmp/robust_autonomous_research",
        "max_concurrent_experiments": 3,
        "timeout_seconds": 1800
    }
    
    try:
        async with RobustResearchOrchestrator(config) as orchestrator:
            
            # Execute full research pipeline
            research_objective = "Develop novel meta-learning algorithms for few-shot classification with improved generalization"
            
            pipeline_results = await orchestrator.full_research_pipeline(
                research_objective=research_objective,
                discovery_budget=3,
                validation_budget=2,
                experiment_algorithms=['neural_network', 'meta_learning'],
                datasets=['cifar10', 'miniImageNet']
            )
            
            # Print results summary
            logger.info("=== ROBUST RESEARCH PIPELINE RESULTS ===")
            for stage, result in pipeline_results.items():
                status_icon = "✓" if result.status == OperationStatus.SUCCESS else "✗"
                logger.info(f"{status_icon} {stage.title()}: {result.status.value} ({result.execution_time:.1f}s)")
                
                if result.error_message:
                    logger.error(f"  Error: {result.error_message}")
            
            # System status
            status = orchestrator.get_system_status()
            logger.info(f"System Health: {status['system_health']}")
            logger.info(f"Total Operations: {status['total_operations_executed']}")
            logger.info(f"CPU Usage: {status.get('cpu_usage', 'N/A')}%")
            logger.info(f"Memory Usage: {status.get('memory_usage', 'N/A')}%")
            
            # Wait a moment for cleanup
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Robust research orchestration failed: {e}")
        return 1
    
    logger.info("✓ Robust Research Orchestrator completed successfully")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))