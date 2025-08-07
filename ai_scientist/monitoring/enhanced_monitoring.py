#!/usr/bin/env python3
"""
Enhanced Monitoring System for AI Scientist v2

Comprehensive monitoring, alerting, and observability framework with
enterprise-grade reliability features.
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import queue
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """System alert with metadata."""
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class Metric:
    """System metric with metadata."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self._running = False
        self._thread = None
        self._metrics = {}
        self._callbacks = []
        
    def start(self):
        """Start resource monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Add callback for metric updates."""
        self._callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_average': load_avg,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'memory_total_gb': memory_total_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'disk_total_gb': disk_total_gb,
                'process_count': process_count,
            }
            
            metrics.update(gpu_metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self.get_current_metrics()
                self._metrics.update(metrics)
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.sampling_interval)
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if available."""
        try:
            # Try to get NVIDIA GPU info
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\\n')
                gpu_metrics = {}
                
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        memory_used = float(parts[0])
                        memory_total = float(parts[1])
                        utilization = float(parts[2])
                        
                        gpu_metrics[f'gpu_{i}_memory_used_mb'] = memory_used
                        gpu_metrics[f'gpu_{i}_memory_total_mb'] = memory_total
                        gpu_metrics[f'gpu_{i}_memory_percent'] = (memory_used / memory_total) * 100
                        gpu_metrics[f'gpu_{i}_utilization'] = utilization
                
                return gpu_metrics
        except Exception:
            pass
        
        return {}

class AlertManager:
    """Alert management with notification and escalation."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self._alerts: List[Alert] = []
        self._alert_handlers = {level: [] for level in AlertLevel}
        self._suppression_rules = {}
        self._lock = threading.Lock()
        
    def add_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """Add alert handler for specific level."""
        self._alert_handlers[level].append(handler)
    
    def create_alert(self, level: AlertLevel, message: str, component: str, 
                    metadata: Dict[str, Any] = None) -> Alert:
        """Create and process new alert."""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Check suppression rules
            suppression_key = f"{component}:{message}"
            if self._is_suppressed(suppression_key, alert):
                logger.debug(f"Alert suppressed: {suppression_key}")
                return alert
            
            # Add to alert history
            self._alerts.append(alert)
            
            # Maintain max alerts limit
            if len(self._alerts) > self.max_alerts:
                self._alerts = self._alerts[-self.max_alerts:]
            
            # Notify handlers
            for handler in self._alert_handlers[level]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
        
        logger.log(self._get_log_level(level), f"ALERT [{component}] {message}")
        return alert
    
    def resolve_alert(self, alert: Alert):
        """Resolve an existing alert."""
        with self._lock:
            alert.resolved = True
            alert.resolution_time = datetime.now()
        
        logger.info(f"Alert resolved: {alert.component} - {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        with self._lock:
            return [alert for alert in self._alerts if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[AlertLevel, int]:
        """Get summary of alerts by level."""
        active_alerts = self.get_active_alerts()
        summary = {level: 0 for level in AlertLevel}
        
        for alert in active_alerts:
            summary[alert.level] += 1
            
        return summary
    
    def add_suppression_rule(self, component: str, message_pattern: str, 
                           duration: timedelta):
        """Add alert suppression rule."""
        key = f"{component}:{message_pattern}"
        self._suppression_rules[key] = {
            'duration': duration,
            'last_alert': None
        }
    
    def _is_suppressed(self, suppression_key: str, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        if suppression_key not in self._suppression_rules:
            return False
        
        rule = self._suppression_rules[suppression_key]
        if rule['last_alert'] is None:
            rule['last_alert'] = alert.timestamp
            return False
        
        time_since_last = alert.timestamp - rule['last_alert']
        if time_since_last < rule['duration']:
            return True
        
        rule['last_alert'] = alert.timestamp
        return False
    
    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level."""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping[alert_level]

class PerformanceTracker:
    """Performance metrics tracking and analysis."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: Dict[str, List[Metric]] = {}
        self._timers: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None, description: str = ""):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            description=description
        )
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append(metric)
            
            # Maintain window size
            if len(self._metrics[name]) > self.window_size:
                self._metrics[name] = self._metrics[name][-self.window_size:]
    
    def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return {}
            
            values = [m.value for m in self._metrics[name]]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'last': values[-1],
                'p50': self._percentile(values, 50),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all metrics."""
        with self._lock:
            return {name: self.get_metric_stats(name) for name in self._metrics.keys()}
    
    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (p / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: PerformanceTracker, name: str, labels: Dict[str, str] = None):
        self.tracker = tracker
        self.name = name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.tracker.record_metric(
                self.name, 
                duration, 
                MetricType.TIMER, 
                self.labels, 
                f"Execution time for {self.name}"
            )

class HealthChecker:
    """Enhanced health checking with detailed diagnostics."""
    
    def __init__(self):
        self.checks = {}
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker()
        
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]], 
                      interval: float = 60, critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'interval': interval,
            'critical': critical,
            'last_run': None,
            'last_result': None
        }
    
    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check in self.checks.items():
            try:
                with self.performance_tracker.timer(f"health_check_{name}"):
                    result = check['func']()
                
                check['last_run'] = datetime.now()
                check['last_result'] = result
                
                results[name] = result
                
                # Check if this component is healthy
                is_healthy = result.get('healthy', False)
                if not is_healthy:
                    level = AlertLevel.CRITICAL if check['critical'] else AlertLevel.WARNING
                    self.alert_manager.create_alert(
                        level, 
                        f"Health check failed: {result.get('error', 'Unknown error')}", 
                        name
                    )
                    
                    if check['critical']:
                        overall_healthy = False
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {'healthy': False, 'error': str(e)}
                
                level = AlertLevel.CRITICAL if check['critical'] else AlertLevel.ERROR
                self.alert_manager.create_alert(level, f"Health check exception: {str(e)}", name)
                
                if check['critical']:
                    overall_healthy = False
        
        results['overall_health'] = overall_healthy
        results['alert_summary'] = self.alert_manager.get_alert_summary()
        results['performance_stats'] = self.performance_tracker.get_all_metrics()
        
        return results
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        if not self.checks:
            return 1.0
        
        healthy_count = 0
        total_count = 0
        
        for name, check in self.checks.items():
            if check['last_result']:
                total_count += 1
                if check['last_result'].get('healthy', False):
                    healthy_count += 1
        
        return healthy_count / total_count if total_count > 0 else 0.0

# Default health checks
def gpu_health_check() -> Dict[str, Any]:
    """Check GPU availability and health."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', 
                               '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        name = parts[0]
                        memory_used = int(parts[1].split()[0])
                        memory_total = int(parts[2].split()[0])
                        memory_percent = (memory_used / memory_total) * 100
                        
                        gpus.append({
                            'name': name,
                            'memory_used': memory_used,
                            'memory_total': memory_total,
                            'memory_percent': memory_percent
                        })
            
            return {
                'healthy': True,
                'details': f"{len(gpus)} GPU(s) available",
                'gpus': gpus
            }
        else:
            return {'healthy': False, 'error': 'nvidia-smi command failed'}
            
    except FileNotFoundError:
        return {'healthy': True, 'details': 'No NVIDIA GPUs detected', 'gpus': []}
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

def memory_health_check() -> Dict[str, Any]:
    """Check system memory usage."""
    try:
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_gb = memory.available / (1024**3)
        
        if memory_percent > 90:
            return {'healthy': False, 'error': f'High memory usage: {memory_percent:.1f}%'}
        elif memory_percent > 80:
            return {'healthy': True, 'warning': f'Moderate memory usage: {memory_percent:.1f}%', 
                   'details': f'{available_gb:.1f}GB available'}
        else:
            return {'healthy': True, 'details': f'{available_gb:.1f}GB available'}
            
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

def disk_health_check() -> Dict[str, Any]:
    """Check disk space."""
    try:
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        free_gb = disk.free / (1024**3)
        
        if disk_percent > 95:
            return {'healthy': False, 'error': f'Very low disk space: {disk_percent:.1f}% used'}
        elif disk_percent > 85:
            return {'healthy': True, 'warning': f'Low disk space: {disk_percent:.1f}% used',
                   'details': f'{free_gb:.1f}GB free'}
        else:
            return {'healthy': True, 'details': f'{free_gb:.1f}GB free'}
            
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

def api_health_check() -> Dict[str, Any]:
    """Check API key availability."""
    try:
        required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        available_keys = []
        missing_keys = []
        
        for key in required_keys:
            if os.getenv(key):
                available_keys.append(key)
            else:
                missing_keys.append(key)
        
        if missing_keys:
            return {'healthy': False, 'error': f'Missing API keys: {", ".join(missing_keys)}'}
        else:
            return {'healthy': True, 'details': f'All API keys available: {", ".join(available_keys)}'}
            
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

# Global instances
resource_monitor = ResourceMonitor()
alert_manager = AlertManager()
performance_tracker = PerformanceTracker()
health_checker = HealthChecker()

# Register default health checks
health_checker.register_check('gpu', gpu_health_check, interval=60, critical=False)
health_checker.register_check('memory', memory_health_check, interval=30, critical=True)
health_checker.register_check('disk', disk_health_check, interval=60, critical=True)
health_checker.register_check('api', api_health_check, interval=300, critical=False)

# Initialize monitoring
def initialize_monitoring():
    """Initialize the monitoring system."""
    resource_monitor.start()
    logger.info("Enhanced monitoring system initialized")

def shutdown_monitoring():
    """Shutdown the monitoring system."""
    resource_monitor.stop()
    logger.info("Enhanced monitoring system shutdown")