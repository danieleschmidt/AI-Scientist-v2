#!/usr/bin/env python3
"""
Advanced Monitoring System - Generation 2: MAKE IT ROBUST

Comprehensive monitoring, health checks, and observability for AI Scientist systems.
Provides real-time insights, alerting, and proactive system health management.
"""

import asyncio
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import statistics
import os
import sys

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.columns import Columns
from rich.align import Align


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics for categorization."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR = "error"
    BUSINESS = "business"
    AVAILABILITY = "availability"
    LATENCY = "latency"


@dataclass
class MetricDataPoint:
    """Single metric data point with timestamp."""
    timestamp: datetime
    value: Union[float, int, bool, str]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and history."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: List[MetricDataPoint] = field(default_factory=list)
    
    # Thresholds for alerting
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    healthy_range: Optional[tuple] = None
    
    # Configuration
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=7))
    collection_interval: float = 30.0  # seconds
    
    def add_data_point(self, value: Union[float, int, bool, str], tags: Optional[Dict[str, str]] = None):
        """Add a new data point to the metric."""
        data_point = MetricDataPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.data_points.append(data_point)
        
        # Clean up old data points
        cutoff_time = datetime.now() - self.retention_period
        self.data_points = [dp for dp in self.data_points if dp.timestamp > cutoff_time]
    
    def get_current_value(self) -> Optional[Union[float, int, bool, str]]:
        """Get the most recent metric value."""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_trend(self, window_size: int = 5) -> str:
        """Analyze trend over recent data points."""
        if len(self.data_points) < 2:
            return "stable"
            
        recent_points = self.data_points[-window_size:]
        if len(recent_points) < 2:
            return "stable"
        
        # Calculate trend for numeric values only
        numeric_points = [dp for dp in recent_points if isinstance(dp.value, (int, float))]
        if len(numeric_points) < 2:
            return "stable"
        
        values = [dp.value for dp in numeric_points]
        slope = (values[-1] - values[0]) / len(values)
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def get_health_status(self) -> HealthStatus:
        """Determine health status based on current value and thresholds."""
        current_value = self.get_current_value()
        
        if current_value is None:
            return HealthStatus.UNKNOWN
        
        if not isinstance(current_value, (int, float)):
            return HealthStatus.HEALTHY
        
        # Check critical threshold
        if self.critical_threshold is not None:
            if current_value >= self.critical_threshold:
                return HealthStatus.CRITICAL
        
        # Check warning threshold
        if self.warning_threshold is not None:
            if current_value >= self.warning_threshold:
                return HealthStatus.WARNING
        
        # Check healthy range
        if self.healthy_range is not None:
            min_val, max_val = self.healthy_range
            if not (min_val <= current_value <= max_val):
                return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY


@dataclass
class Alert:
    """Alert definition and tracking."""
    alert_id: str
    name: str
    description: str
    severity: HealthStatus
    metric_name: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledge_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    # Alert context
    trigger_value: Optional[Union[float, int, str]] = None
    threshold_value: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.resolved_at is None
    
    @property
    def duration(self) -> timedelta:
        """Get duration of alert (active time)."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.triggered_at


@dataclass
class HealthCheckResult:
    """Result of a health check execution."""
    check_name: str
    status: HealthStatus
    message: str
    execution_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if health check passed."""
        return self.status == HealthStatus.HEALTHY


class AdvancedMonitoringSystem:
    """
    Generation 2: MAKE IT ROBUST
    Comprehensive monitoring system with health checks and alerting.
    """
    
    def __init__(self, workspace_dir: str = "monitoring_workspace"):
        self.console = Console()
        self.logger = self._setup_logging()
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Core monitoring components
        self.metrics: Dict[str, Metric] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.health_checks: Dict[str, Callable] = {}
        self.health_check_results: Dict[str, List[HealthCheckResult]] = {}
        
        # System state tracking
        self.system_start_time = datetime.now()
        self.monitoring_active = False
        self.collection_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.alert_retention_days = 30
        self.health_check_interval = 60  # seconds
        self.metric_dashboard_refresh = 2  # seconds
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="monitoring")
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        # Initialize core health checks
        self._initialize_core_health_checks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for monitoring system."""
        logger = logging.getLogger(f"{__name__}.AdvancedMonitoringSystem")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics."""
        
        # Performance metrics
        self.register_metric(
            name="cpu_usage_percent",
            metric_type=MetricType.PERFORMANCE,
            description="CPU usage percentage",
            unit="%",
            warning_threshold=70.0,
            critical_threshold=90.0,
            collection_interval=10.0
        )
        
        self.register_metric(
            name="memory_usage_percent",
            metric_type=MetricType.RESOURCE,
            description="Memory usage percentage",
            unit="%",
            warning_threshold=80.0,
            critical_threshold=95.0,
            collection_interval=10.0
        )
        
        self.register_metric(
            name="disk_usage_percent",
            metric_type=MetricType.RESOURCE,
            description="Disk usage percentage",
            unit="%",
            warning_threshold=85.0,
            critical_threshold=95.0,
            collection_interval=60.0
        )
        
        # Error metrics
        self.register_metric(
            name="error_rate",
            metric_type=MetricType.ERROR,
            description="Errors per minute",
            unit="errors/min",
            warning_threshold=5.0,
            critical_threshold=20.0,
            collection_interval=60.0
        )
        
        # Business metrics
        self.register_metric(
            name="experiments_completed",
            metric_type=MetricType.BUSINESS,
            description="Number of completed experiments",
            unit="count",
            collection_interval=300.0
        )
        
        self.register_metric(
            name="hypotheses_generated",
            metric_type=MetricType.BUSINESS,
            description="Number of generated hypotheses",
            unit="count",
            collection_interval=300.0
        )
        
        # Availability metrics
        self.register_metric(
            name="system_uptime",
            metric_type=MetricType.AVAILABILITY,
            description="System uptime in hours",
            unit="hours",
            collection_interval=60.0
        )
        
        # Latency metrics
        self.register_metric(
            name="api_response_time",
            metric_type=MetricType.LATENCY,
            description="Average API response time",
            unit="ms",
            warning_threshold=1000.0,
            critical_threshold=5000.0,
            collection_interval=30.0
        )
    
    def _initialize_core_health_checks(self):
        """Initialize core health checks."""
        
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("network_connectivity", self._check_network_connectivity)
        self.register_health_check("critical_services", self._check_critical_services)
        self.register_health_check("configuration_validity", self._check_configuration_validity)
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        healthy_range: Optional[tuple] = None,
        collection_interval: float = 60.0,
        retention_days: int = 7
    ) -> Metric:
        """Register a new metric for monitoring."""
        
        metric = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            healthy_range=healthy_range,
            collection_interval=collection_interval,
            retention_period=timedelta(days=retention_days)
        )
        
        self.metrics[name] = metric
        self.logger.info(f"Registered metric: {name}")
        
        return metric
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int, bool, str],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a value for a metric."""
        
        if name not in self.metrics:
            self.logger.warning(f"Attempting to record unknown metric: {name}")
            return
        
        metric = self.metrics[name]
        metric.add_data_point(value, tags)
        
        # Check for alert conditions
        self._check_metric_alerts(metric)
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a health check function."""
        
        self.health_checks[name] = check_function
        self.health_check_results[name] = []
        self.logger.info(f"Registered health check: {name}")
    
    async def execute_health_check(self, name: str) -> HealthCheckResult:
        """Execute a specific health check."""
        
        if name not in self.health_checks:
            return HealthCheckResult(
                check_name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check {name} not found",
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            check_function = self.health_checks[name]
            
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                # Run synchronous health check in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, check_function)
            
            execution_time = time.time() - start_time
            
            # Handle different result formats
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, dict):
                return HealthCheckResult(
                    check_name=name,
                    status=HealthStatus(result.get('status', 'unknown')),
                    message=result.get('message', ''),
                    execution_time=execution_time,
                    timestamp=timestamp,
                    details=result.get('details', {})
                )
            elif isinstance(result, bool):
                return HealthCheckResult(
                    check_name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.CRITICAL,
                    message="Check passed" if result else "Check failed",
                    execution_time=execution_time,
                    timestamp=timestamp
                )
            else:
                return HealthCheckResult(
                    check_name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Unknown result type: {type(result)}",
                    execution_time=execution_time,
                    timestamp=timestamp
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Health check {name} failed: {e}")
            
            return HealthCheckResult(
                check_name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                execution_time=execution_time,
                timestamp=timestamp,
                details={'exception': str(e)}
            )
    
    async def execute_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Execute all registered health checks."""
        
        tasks = []
        for check_name in self.health_checks.keys():
            task = asyncio.create_task(self.execute_health_check(check_name))
            tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
                
                # Store result history
                if check_name in self.health_check_results:
                    self.health_check_results[check_name].append(result)
                    
                    # Limit history size
                    if len(self.health_check_results[check_name]) > 100:
                        self.health_check_results[check_name] = self.health_check_results[check_name][-100:]
                
            except Exception as e:
                self.logger.error(f"Failed to execute health check {check_name}: {e}")
                results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Execution failed: {str(e)}",
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
        
        return results
    
    def _check_metric_alerts(self, metric: Metric):
        """Check metric for alert conditions and trigger alerts if needed."""
        
        health_status = metric.get_health_status()
        current_value = metric.get_current_value()
        
        if health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            # Check if alert already exists
            alert_key = f"{metric.name}_{health_status.value}"
            
            if alert_key not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    alert_id=f"alert_{int(time.time())}_{hash(alert_key) % 10000:04d}",
                    name=f"{metric.name} {health_status.value}",
                    description=f"Metric {metric.name} is {health_status.value}",
                    severity=health_status,
                    metric_name=metric.name,
                    triggered_at=datetime.now(),
                    trigger_value=current_value,
                    threshold_value=metric.warning_threshold if health_status == HealthStatus.WARNING else metric.critical_threshold
                )
                
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(f"Alert triggered: {alert.name} - Value: {current_value}")
                self.console.print(f"[bold red]🚨 ALERT: {alert.name} - Value: {current_value}[/bold red]")
        
        elif health_status == HealthStatus.HEALTHY:
            # Check if we need to resolve any alerts
            alerts_to_resolve = [
                key for key in self.active_alerts.keys()
                if self.active_alerts[key].metric_name == metric.name
            ]
            
            for alert_key in alerts_to_resolve:
                alert = self.active_alerts[alert_key]
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_key]
                
                self.logger.info(f"Alert resolved: {alert.name}")
                self.console.print(f"[bold green]✅ RESOLVED: {alert.name}[/bold green]")
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.logger.info("Starting monitoring system...")
        
        # Start metric collection tasks
        for metric_name, metric in self.metrics.items():
            task = asyncio.create_task(
                self._metric_collection_loop(metric_name, metric.collection_interval)
            )
            self.collection_tasks.append(task)
        
        # Start health check loop
        health_check_task = asyncio.create_task(
            self._health_check_loop(self.health_check_interval)
        )
        self.collection_tasks.append(health_check_task)
        
        self.console.print("[bold green]🔍 Monitoring system started[/bold green]")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.logger.info("Stopping monitoring system...")
        
        # Cancel all collection tasks
        for task in self.collection_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        self.collection_tasks.clear()
        self.console.print("[bold yellow]⏹️ Monitoring system stopped[/bold yellow]")
    
    async def _metric_collection_loop(self, metric_name: str, interval: float):
        """Background loop for collecting metric data."""
        
        while self.monitoring_active:
            try:
                # Collect metric value based on metric type
                value = await self._collect_metric_value(metric_name)
                if value is not None:
                    self.record_metric(metric_name, value)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting metric {metric_name}: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metric_value(self, metric_name: str) -> Optional[Union[float, int, bool, str]]:
        """Collect current value for a specific metric."""
        
        try:
            if metric_name == "cpu_usage_percent":
                return await self._get_cpu_usage()
            elif metric_name == "memory_usage_percent":
                return await self._get_memory_usage()
            elif metric_name == "disk_usage_percent":
                return await self._get_disk_usage()
            elif metric_name == "system_uptime":
                return (datetime.now() - self.system_start_time).total_seconds() / 3600
            elif metric_name == "error_rate":
                return await self._get_error_rate()
            elif metric_name == "api_response_time":
                return await self._measure_api_response_time()
            elif metric_name in ["experiments_completed", "hypotheses_generated"]:
                return await self._get_business_metric(metric_name)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to collect {metric_name}: {e}")
            return None
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback without psutil
            return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    async def _get_disk_usage(self) -> float:
        """Get current disk usage percentage."""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except ImportError:
            return 0.0
    
    async def _get_error_rate(self) -> float:
        """Calculate current error rate."""
        # This would integrate with error tracking system
        # For demo purposes, return a simulated value
        return 0.5 + 2.0 * (time.time() % 60) / 60  # Varies from 0.5 to 2.5
    
    async def _measure_api_response_time(self) -> float:
        """Measure API response time."""
        start_time = time.time()
        try:
            # Simulate API call
            await asyncio.sleep(0.1)  # Simulate 100ms response
            return (time.time() - start_time) * 1000  # Return in milliseconds
        except Exception:
            return 5000.0  # Return high value on error
    
    async def _get_business_metric(self, metric_name: str) -> int:
        """Get business metric value."""
        # This would integrate with business logic
        # For demo purposes, return simulated values
        if metric_name == "experiments_completed":
            return int(time.time() % 100)  # Varies from 0 to 99
        elif metric_name == "hypotheses_generated":
            return int((time.time() % 200) + 50)  # Varies from 50 to 249
        return 0
    
    async def _health_check_loop(self, interval: float):
        """Background loop for executing health checks."""
        
        while self.monitoring_active:
            try:
                await self.execute_all_health_checks()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(interval)
    
    # Core health check implementations
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine overall status
            if cpu_percent > 90 or memory_percent > 95 or disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = "System resources critically low"
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 85:
                status = HealthStatus.WARNING
                message = "System resources elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return {
                'status': status.value,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                }
            }
            
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'psutil not available for resource monitoring'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        
        try:
            import shutil
            
            total, used, free = shutil.disk_usage('/')
            free_percent = (free / total) * 100
            
            if free_percent < 5:
                status = HealthStatus.CRITICAL
                message = f"Critical: Only {free_percent:.1f}% disk space available"
            elif free_percent < 15:
                status = HealthStatus.WARNING
                message = f"Warning: {free_percent:.1f}% disk space available"
            else:
                status = HealthStatus.HEALTHY
                message = f"Healthy: {free_percent:.1f}% disk space available"
            
            return {
                'status': status.value,
                'message': message,
                'details': {
                    'total_gb': total / (1024**3),
                    'used_gb': used / (1024**3),
                    'free_gb': free / (1024**3),
                    'free_percent': free_percent
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Disk space check failed: {str(e)}'
            }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity."""
        
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname("google.com")
            
            # Test HTTP connectivity
            import urllib.request
            response = urllib.request.urlopen("https://httpbin.org/status/200", timeout=10)
            
            if response.status == 200:
                return {
                    'status': HealthStatus.HEALTHY.value,
                    'message': 'Network connectivity normal'
                }
            else:
                return {
                    'status': HealthStatus.WARNING.value,
                    'message': f'HTTP test returned status {response.status}'
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Network connectivity failed: {str(e)}'
            }
    
    def _check_critical_services(self) -> Dict[str, Any]:
        """Check critical system services."""
        
        # This would check actual services in a real implementation
        # For demo purposes, simulate service checks
        services = ['database', 'cache', 'message_queue', 'api_gateway']
        service_status = {}
        
        overall_healthy = True
        
        for service in services:
            # Simulate service check
            if service == 'database':
                # Simulate database check
                healthy = True  # Would actually check database connection
            elif service == 'cache':
                # Simulate cache check
                healthy = True  # Would actually check cache connectivity
            else:
                healthy = True  # Default to healthy for demo
            
            service_status[service] = 'healthy' if healthy else 'unhealthy'
            if not healthy:
                overall_healthy = False
        
        status = HealthStatus.HEALTHY if overall_healthy else HealthStatus.CRITICAL
        message = 'All critical services healthy' if overall_healthy else 'Some critical services unhealthy'
        
        return {
            'status': status.value,
            'message': message,
            'details': {'services': service_status}
        }
    
    def _check_configuration_validity(self) -> Dict[str, Any]:
        """Check system configuration validity."""
        
        # Check for required environment variables
        required_vars = ['PYTHONPATH']  # Add actual required vars
        missing_vars = []
        
        for var in required_vars:
            if var not in os.environ:
                missing_vars.append(var)
        
        # Check configuration files
        config_files = [self.workspace_dir / 'config.json']  # Add actual config files
        missing_files = []
        
        for config_file in config_files:
            if not config_file.exists():
                missing_files.append(str(config_file))
        
        if missing_vars or missing_files:
            return {
                'status': HealthStatus.WARNING.value,
                'message': 'Configuration issues detected',
                'details': {
                    'missing_environment_variables': missing_vars,
                    'missing_configuration_files': missing_files
                }
            }
        else:
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Configuration valid'
            }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        
        # Collect current metric values
        current_metrics = {}
        for name, metric in self.metrics.items():
            current_value = metric.get_current_value()
            health_status = metric.get_health_status()
            trend = metric.get_trend()
            
            current_metrics[name] = {
                'value': current_value,
                'unit': metric.unit,
                'health_status': health_status.value,
                'trend': trend
            }
        
        # Get latest health check results
        latest_health_checks = {}
        for check_name, results in self.health_check_results.items():
            if results:
                latest_result = results[-1]
                latest_health_checks[check_name] = {
                    'status': latest_result.status.value,
                    'message': latest_result.message,
                    'execution_time': latest_result.execution_time,
                    'timestamp': latest_result.timestamp.isoformat()
                }
        
        # Calculate overall health
        all_statuses = []
        
        # Include metric health statuses
        for metric_data in current_metrics.values():
            all_statuses.append(HealthStatus(metric_data['health_status']))
        
        # Include health check statuses
        for health_data in latest_health_checks.values():
            all_statuses.append(HealthStatus(health_data['status']))
        
        # Determine overall health
        if HealthStatus.CRITICAL in all_statuses:
            overall_health = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in all_statuses:
            overall_health = HealthStatus.WARNING
        elif all_statuses and all(s == HealthStatus.HEALTHY for s in all_statuses):
            overall_health = HealthStatus.HEALTHY
        else:
            overall_health = HealthStatus.UNKNOWN
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': (datetime.now() - self.system_start_time).total_seconds(),
            'overall_health': overall_health.value,
            'monitoring_active': self.monitoring_active,
            'metrics': current_metrics,
            'health_checks': latest_health_checks,
            'active_alerts': len(self.active_alerts),
            'total_alerts_today': len([
                alert for alert in self.alert_history
                if alert.triggered_at.date() == datetime.now().date()
            ])
        }
    
    def create_monitoring_dashboard(self) -> Table:
        """Create a comprehensive monitoring dashboard."""
        
        # Main dashboard table
        dashboard = Table(title="🔍 Advanced Monitoring Dashboard", show_header=True)
        
        dashboard.add_column("Component", style="cyan", no_wrap=True, width=20)
        dashboard.add_column("Status", style="bold", width=12)
        dashboard.add_column("Value", style="white", width=15)
        dashboard.add_column("Trend", style="blue", width=12)
        dashboard.add_column("Last Updated", style="dim", width=20)
        
        # Add metrics
        for name, metric in self.metrics.items():
            current_value = metric.get_current_value()
            health_status = metric.get_health_status()
            trend = metric.get_trend()
            
            # Status color coding
            if health_status == HealthStatus.HEALTHY:
                status_text = "[green]✅ HEALTHY[/green]"
            elif health_status == HealthStatus.WARNING:
                status_text = "[yellow]⚠️ WARNING[/yellow]"
            elif health_status == HealthStatus.CRITICAL:
                status_text = "[red]🚨 CRITICAL[/red]"
            else:
                status_text = "[dim]❓ UNKNOWN[/dim]"
            
            # Trend indicators
            trend_indicators = {
                'increasing': '📈',
                'decreasing': '📉',
                'stable': '➡️'
            }
            trend_text = f"{trend_indicators.get(trend, '❓')} {trend}"
            
            # Format value
            if current_value is not None:
                if isinstance(current_value, float):
                    value_text = f"{current_value:.2f} {metric.unit}"
                else:
                    value_text = f"{current_value} {metric.unit}"
            else:
                value_text = "N/A"
            
            # Last update time
            if metric.data_points:
                last_update = metric.data_points[-1].timestamp.strftime("%H:%M:%S")
            else:
                last_update = "Never"
            
            dashboard.add_row(
                name.replace('_', ' ').title(),
                status_text,
                value_text,
                trend_text,
                last_update
            )
        
        return dashboard
    
    def create_alerts_panel(self) -> Panel:
        """Create alerts summary panel."""
        
        if not self.active_alerts:
            return Panel(
                "[green]✅ No active alerts[/green]",
                title="🚨 Active Alerts",
                border_style="green"
            )
        
        alert_text = []
        for alert in self.active_alerts.values():
            duration = alert.duration
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            
            severity_color = {
                HealthStatus.WARNING: "yellow",
                HealthStatus.CRITICAL: "red"
            }.get(alert.severity, "white")
            
            alert_text.append(
                f"[{severity_color}]• {alert.name}[/{severity_color}] "
                f"({hours}h {minutes}m ago)"
            )
        
        return Panel(
            "\n".join(alert_text),
            title=f"🚨 Active Alerts ({len(self.active_alerts)})",
            border_style="red" if any(a.severity == HealthStatus.CRITICAL for a in self.active_alerts.values()) else "yellow"
        )
    
    def create_health_checks_panel(self) -> Panel:
        """Create health checks summary panel."""
        
        health_text = []
        overall_healthy = True
        
        for check_name, results in self.health_check_results.items():
            if results:
                latest_result = results[-1]
                status = latest_result.status
                
                if status == HealthStatus.HEALTHY:
                    health_text.append(f"[green]✅ {check_name}[/green]: {latest_result.message}")
                elif status == HealthStatus.WARNING:
                    health_text.append(f"[yellow]⚠️ {check_name}[/yellow]: {latest_result.message}")
                    overall_healthy = False
                elif status == HealthStatus.CRITICAL:
                    health_text.append(f"[red]🚨 {check_name}[/red]: {latest_result.message}")
                    overall_healthy = False
                else:
                    health_text.append(f"[dim]❓ {check_name}[/dim]: {latest_result.message}")
                    overall_healthy = False
            else:
                health_text.append(f"[dim]❓ {check_name}[/dim]: Not executed")
                overall_healthy = False
        
        if not health_text:
            health_text = ["[dim]No health checks configured[/dim]"]
        
        border_style = "green" if overall_healthy else "red"
        
        return Panel(
            "\n".join(health_text),
            title="🏥 Health Checks",
            border_style=border_style
        )


# Demo and testing functions
async def demo_advanced_monitoring():
    """Demonstrate the advanced monitoring system."""
    
    console = Console()
    console.print("[bold blue]🔍 Advanced Monitoring System - Generation 2 Demo[/bold blue]")
    
    # Initialize monitoring system
    monitoring = AdvancedMonitoringSystem()
    
    # Add custom metrics for demo
    monitoring.register_metric(
        name="demo_processing_queue",
        metric_type=MetricType.BUSINESS,
        description="Number of items in processing queue",
        unit="items",
        warning_threshold=50.0,
        critical_threshold=100.0,
        collection_interval=5.0
    )
    
    # Start monitoring
    await monitoring.start_monitoring()
    
    console.print("\n[yellow]📊 Monitoring system running...[/yellow]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]")
    
    # Run live dashboard
    try:
        with Live(console=console, refresh_per_second=0.5) as live:
            for i in range(60):  # Run for 60 seconds
                # Simulate some data
                monitoring.record_metric("demo_processing_queue", 20 + i * 2)
                
                # Create dashboard layout
                dashboard = monitoring.create_monitoring_dashboard()
                alerts_panel = monitoring.create_alerts_panel()
                health_panel = monitoring.create_health_checks_panel()
                
                # Create overall layout
                layout = Columns([
                    Panel(dashboard, title="📊 Metrics Dashboard"),
                    Panel(
                        f"{alerts_panel}\n\n{health_panel}",
                        title="🚨 System Status"
                    )
                ])
                
                live.update(layout)
                
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitoring demo...[/yellow]")
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    
    # Show final system overview
    overview = monitoring.get_system_overview()
    console.print(f"\n[bold green]📋 Final System Overview:[/bold green]")
    console.print(f"• Overall Health: {overview['overall_health']}")
    console.print(f"• System Uptime: {overview['system_uptime']:.1f} seconds")
    console.print(f"• Active Alerts: {overview['active_alerts']}")
    console.print(f"• Total Alerts Today: {overview['total_alerts_today']}")
    
    return monitoring


async def main():
    """Main entry point for advanced monitoring demo."""
    
    try:
        monitoring_system = await demo_advanced_monitoring()
        
        console = Console()
        console.print(f"\n[bold green]✅ Advanced monitoring demo completed successfully![/bold green]")
        
        # Show metrics summary
        console.print(f"\n[bold cyan]📊 Metrics Summary:[/bold cyan]")
        for name, metric in monitoring_system.metrics.items():
            current_value = metric.get_current_value()
            data_points = len(metric.data_points)
            console.print(f"• {name}: {current_value} ({data_points} data points)")
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())