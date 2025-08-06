"""
System Health Monitoring for Quantum Task Planner

Comprehensive health monitoring, alerting, and diagnostic system
for quantum task planning operations.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], bool]
    description: str
    check_interval: float = 30.0  # seconds
    timeout: float = 5.0  # seconds
    retries: int = 2
    critical: bool = False  # Whether failure is critical


@dataclass
class HealthMetric:
    """Health metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    status: HealthStatus = HealthStatus.HEALTHY
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health snapshot."""
    status: HealthStatus
    timestamp: float
    metrics: Dict[str, HealthMetric]
    failing_checks: List[str]
    warnings: List[str]
    uptime: float
    last_restart: Optional[float] = None


class HealthMonitor:
    """
    Comprehensive system health monitoring.
    
    Monitors system resources, quantum planning operations,
    and provides alerting for issues.
    """
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 metric_history_size: int = 1000):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Default interval between health checks
            metric_history_size: Number of metric measurements to keep
        """
        self.check_interval = check_interval
        self.metric_history_size = metric_history_size
        
        # Health checks and metrics
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metric_history_size))
        self.current_metrics: Dict[str, HealthMetric] = {}
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = time.time()
        self.last_restart: Optional[float] = None
        
        # Alerting
        self.alert_callbacks: List[Callable[[SystemHealth], None]] = []
        self.alert_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # System resource monitoring
        self.process = psutil.Process()
        
        # Set up default health checks
        self._setup_default_health_checks()
        
        logger.info("Initialized HealthMonitor with comprehensive system monitoring")
    
    def _setup_default_health_checks(self) -> None:
        """Set up default system health checks."""
        
        # Memory usage check
        self.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            description="Monitor memory usage",
            check_interval=15.0,
            critical=True
        ))
        
        # CPU usage check
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            description="Monitor CPU usage",
            check_interval=15.0,
            critical=False
        ))
        
        # Disk space check
        self.add_health_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            description="Monitor available disk space",
            check_interval=60.0,
            critical=True
        ))
        
        # Thread count check
        self.add_health_check(HealthCheck(
            name="thread_count",
            check_function=self._check_thread_count,
            description="Monitor thread usage",
            check_interval=30.0,
            critical=False
        ))
        
        # File descriptor check
        self.add_health_check(HealthCheck(
            name="file_descriptors",
            check_function=self._check_file_descriptors,
            description="Monitor file descriptor usage",
            check_interval=30.0,
            critical=True
        ))
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a custom health check."""
        self.health_checks[health_check.name] = health_check
        logger.debug(f"Added health check: {health_check.name}")
    
    def add_alert_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
        logger.debug("Added health alert callback")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_running:
            logger.warning("Health monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started continuous health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_check_times = {}
        
        while self.is_running:
            current_time = time.time()
            
            # Run health checks based on their intervals
            for check_name, health_check in self.health_checks.items():
                last_check = last_check_times.get(check_name, 0)
                
                if current_time - last_check >= health_check.check_interval:
                    try:
                        self._run_health_check(health_check)
                        last_check_times[check_name] = current_time
                    except Exception as e:
                        logger.error(f"Health check {check_name} failed with exception: {e}")
            
            # Generate system health snapshot
            system_health = self.get_system_health()
            
            # Send alerts if needed
            if system_health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._send_alerts(system_health)
            
            # Sleep until next check
            time.sleep(min(5.0, self.check_interval))
    
    def _run_health_check(self, health_check: HealthCheck) -> bool:
        """Run a single health check with timeout and retries."""
        for attempt in range(health_check.retries + 1):
            try:
                # Simple timeout implementation
                start_time = time.time()
                result = health_check.check_function()
                execution_time = time.time() - start_time
                
                if execution_time > health_check.timeout:
                    logger.warning(f"Health check {health_check.name} exceeded timeout ({execution_time:.2f}s)")
                    if attempt == health_check.retries:
                        return False
                    continue
                
                if result:
                    return True
                else:
                    if attempt < health_check.retries:
                        logger.debug(f"Health check {health_check.name} failed, retrying ({attempt + 1}/{health_check.retries})")
                        time.sleep(1.0)  # Brief delay before retry
                    
            except Exception as e:
                logger.error(f"Health check {health_check.name} exception: {e}")
                if attempt == health_check.retries:
                    return False
        
        return False
    
    def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            # Process memory usage
            process_memory_mb = memory_info.rss / 1024 / 1024
            
            # System memory usage
            system_memory_percent = virtual_memory.percent
            
            # Record metrics
            self._record_metric(HealthMetric(
                name="process_memory_mb",
                value=process_memory_mb,
                unit="MB",
                timestamp=time.time(),
                threshold_warning=1000.0,  # 1GB warning
                threshold_critical=2000.0  # 2GB critical
            ))
            
            self._record_metric(HealthMetric(
                name="system_memory_percent",
                value=system_memory_percent,
                unit="%",
                timestamp=time.time(),
                threshold_warning=80.0,
                threshold_critical=90.0
            ))
            
            # Health check passes if under critical thresholds
            return process_memory_mb < 2000.0 and system_memory_percent < 90.0
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage."""
        try:
            # Process CPU usage (over 1 second interval)
            process_cpu = self.process.cpu_percent(interval=1.0)
            
            # System CPU usage
            system_cpu = psutil.cpu_percent(interval=None)
            
            # Record metrics
            self._record_metric(HealthMetric(
                name="process_cpu_percent",
                value=process_cpu,
                unit="%",
                timestamp=time.time(),
                threshold_warning=50.0,
                threshold_critical=80.0
            ))
            
            self._record_metric(HealthMetric(
                name="system_cpu_percent", 
                value=system_cpu,
                unit="%",
                timestamp=time.time(),
                threshold_warning=70.0,
                threshold_critical=90.0
            ))
            
            # Health check passes if under critical thresholds
            return process_cpu < 80.0 and system_cpu < 90.0
            
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / 1024 / 1024 / 1024
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Record metrics
            self._record_metric(HealthMetric(
                name="disk_free_gb",
                value=free_gb,
                unit="GB",
                timestamp=time.time(),
                threshold_warning=5.0,  # 5GB warning
                threshold_critical=1.0  # 1GB critical
            ))
            
            self._record_metric(HealthMetric(
                name="disk_used_percent",
                value=used_percent,
                unit="%",
                timestamp=time.time(),
                threshold_warning=85.0,
                threshold_critical=95.0
            ))
            
            # Health check passes if sufficient free space
            return free_gb > 1.0 and used_percent < 95.0
            
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False
    
    def _check_thread_count(self) -> bool:
        """Check thread count."""
        try:
            thread_count = self.process.num_threads()
            
            # Record metric
            self._record_metric(HealthMetric(
                name="thread_count",
                value=thread_count,
                unit="threads",
                timestamp=time.time(),
                threshold_warning=50,
                threshold_critical=100
            ))
            
            # Health check passes if under reasonable thread limit
            return thread_count < 100
            
        except Exception as e:
            logger.error(f"Thread check failed: {e}")
            return False
    
    def _check_file_descriptors(self) -> bool:
        """Check file descriptor usage."""
        try:
            num_fds = self.process.num_fds()
            
            # Record metric
            self._record_metric(HealthMetric(
                name="file_descriptors",
                value=num_fds,
                unit="fds",
                timestamp=time.time(),
                threshold_warning=500,
                threshold_critical=900
            ))
            
            # Health check passes if under FD limit
            return num_fds < 900
            
        except Exception as e:
            logger.error(f"File descriptor check failed: {e}")
            return False
    
    def _record_metric(self, metric: HealthMetric) -> None:
        """Record a health metric."""
        # Determine status based on thresholds
        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            metric.status = HealthStatus.CRITICAL
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            metric.status = HealthStatus.WARNING
        else:
            metric.status = HealthStatus.HEALTHY
        
        # Store current metric and add to history
        self.current_metrics[metric.name] = metric
        self.metrics_history[metric.name].append(metric)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health snapshot."""
        current_time = time.time()
        
        # Determine overall status
        failing_checks = []
        warnings = []
        overall_status = HealthStatus.HEALTHY
        
        # Check health check results (simplified - in practice you'd track results)
        critical_metrics = [m for m in self.current_metrics.values() if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in self.current_metrics.values() if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            overall_status = HealthStatus.CRITICAL
            failing_checks.extend([m.name for m in critical_metrics])
        elif warning_metrics:
            overall_status = HealthStatus.WARNING
            warnings.extend([m.name for m in warning_metrics])
        
        uptime = current_time - self.start_time
        
        return SystemHealth(
            status=overall_status,
            timestamp=current_time,
            metrics=self.current_metrics.copy(),
            failing_checks=failing_checks,
            warnings=warnings,
            uptime=uptime,
            last_restart=self.last_restart
        )
    
    def _send_alerts(self, system_health: SystemHealth) -> None:
        """Send health alerts to registered callbacks."""
        try:
            # Add to alert queue
            if not self.alert_queue.full():
                self.alert_queue.put_nowait(system_health)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(system_health)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
        
        except Exception as e:
            logger.error(f"Failed to send alerts: {e}")
    
    def get_metric_history(self, metric_name: str, duration_seconds: Optional[float] = None) -> List[HealthMetric]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of metric
            duration_seconds: Time window (None for all history)
            
        Returns:
            List of historical metric values
        """
        if metric_name not in self.metrics_history:
            return []
        
        metrics = list(self.metrics_history[metric_name])
        
        if duration_seconds is not None:
            cutoff_time = time.time() - duration_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        system_health = self.get_system_health()
        
        # Calculate metric summaries
        metric_summaries = {}
        for name, metric in system_health.metrics.items():
            history = self.get_metric_history(name, duration_seconds=3600)  # Last hour
            
            if history:
                values = [h.value for h in history]
                metric_summaries[name] = {
                    'current': metric.value,
                    'status': metric.status.value,
                    'avg_1h': sum(values) / len(values),
                    'min_1h': min(values),
                    'max_1h': max(values),
                    'unit': metric.unit
                }
        
        return {
            'overall_status': system_health.status.value,
            'uptime_hours': system_health.uptime / 3600,
            'failing_checks': system_health.failing_checks,
            'warnings': system_health.warnings,
            'metrics': metric_summaries,
            'monitoring_active': self.is_running,
            'last_check': system_health.timestamp
        }
    
    def reset_alerts(self) -> None:
        """Clear alert queue."""
        while not self.alert_queue.empty():
            try:
                self.alert_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Cleared health alerts")
    
    def restart_monitoring(self) -> None:
        """Restart health monitoring (useful for configuration changes)."""
        self.stop_monitoring()
        self.last_restart = time.time()
        time.sleep(1.0)  # Brief pause
        self.start_monitoring()
        logger.info("Restarted health monitoring")