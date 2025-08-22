"""
Advanced Performance Monitoring and Metrics System

Comprehensive performance tracking with real-time analytics, predictive alerts,
and quantum-enhanced optimization insights.
"""

import time
import threading
import queue
import psutil
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta
import weakref
import gc
import sys
import traceback
import functools

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    THROUGHPUT = "throughput"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricSample:
    """Single metric sample with metadata."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass
class MetricAlert:
    """Performance alert definition."""
    name: str
    condition: str
    threshold: float
    level: AlertLevel
    message: str
    cooldown_seconds: float = 300.0
    last_triggered: float = 0.0
    
    def should_trigger(self, value: float, current_time: float) -> bool:
        """Check if alert should trigger."""
        # Check cooldown
        if current_time - self.last_triggered < self.cooldown_seconds:
            return False
        
        # Evaluate condition
        if self.condition == "greater_than":
            return value > self.threshold
        elif self.condition == "less_than":
            return value < self.threshold
        elif self.condition == "equal_to":
            return abs(value - self.threshold) < 0.001
        elif self.condition == "not_equal_to":
            return abs(value - self.threshold) >= 0.001
        
        return False


class PerformanceCollector:
    """High-performance metrics collector with minimal overhead."""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 max_samples: int = 10000):
        self.name = name
        self.metric_type = metric_type
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.lock = threading.RLock()
        
        # Statistics cache
        self._stats_cache: Optional[Dict[str, float]] = None
        self._cache_timestamp = 0.0
        self._cache_ttl = 1.0  # Cache TTL in seconds
        
        # Alert handlers
        self.alerts: List[MetricAlert] = []
        self.alert_callbacks: List[Callable] = []
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric sample with minimal overhead."""
        sample = MetricSample(
            timestamp=time.time(),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.samples.append(sample)
            self._invalidate_cache()
            self._check_alerts(sample)
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        if self.metric_type == MetricType.COUNTER:
            current_value = self.get_current_value()
            self.record(current_value + amount, labels)
        else:
            self.record(amount, labels)
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric value."""
        self.record(value, labels)
    
    def time_it(self, func: Optional[Callable] = None):
        """Decorator or context manager for timing operations."""
        if func is not None:
            # Used as decorator
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.record(execution_time, {'function': func.__name__})
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.record(execution_time, {
                        'function': func.__name__,
                        'error': str(e)
                    })
                    raise
            return wrapper
        else:
            # Used as context manager
            return TimingContext(self)
    
    def get_current_value(self) -> float:
        """Get the most recent value."""
        with self.lock:
            if not self.samples:
                return 0.0
            return self.samples[-1].value
    
    def get_statistics(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary of metrics."""
        current_time = time.time()
        
        # Check cache
        if (self._stats_cache and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._stats_cache.copy()
        
        with self.lock:
            if not self.samples:
                return {}
            
            # Filter samples by time window
            if window_seconds:
                cutoff_time = current_time - window_seconds
                values = [s.value for s in self.samples if s.timestamp >= cutoff_time]
            else:
                values = [s.value for s in self.samples]
            
            if not values:
                return {}
            
            # Calculate statistics
            stats = {
                'count': len(values),
                'sum': sum(values),
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'current': values[-1] if values else 0.0
            }
            
            # Additional statistics for sufficient data
            if len(values) >= 2:
                stats['stdev'] = statistics.stdev(values)
                stats['variance'] = statistics.variance(values)
            
            if len(values) >= 3:
                stats['median'] = statistics.median(values)
                
                # Percentiles
                sorted_values = sorted(values)
                n = len(sorted_values)
                stats['p25'] = sorted_values[int(0.25 * n)]
                stats['p75'] = sorted_values[int(0.75 * n)]
                stats['p90'] = sorted_values[int(0.90 * n)]
                stats['p95'] = sorted_values[int(0.95 * n)]
                stats['p99'] = sorted_values[int(0.99 * n)]
            
            # Calculate rate for counters
            if self.metric_type == MetricType.COUNTER and len(values) >= 2:
                time_diff = self.samples[-1].timestamp - self.samples[0].timestamp
                if time_diff > 0:
                    value_diff = values[-1] - values[0]
                    stats['rate'] = value_diff / time_diff
            
            # Cache results
            self._stats_cache = stats.copy()
            self._cache_timestamp = current_time
            
            return stats
    
    def get_samples(self, limit: Optional[int] = None, 
                   window_seconds: Optional[float] = None) -> List[MetricSample]:
        """Get metric samples with optional filtering."""
        with self.lock:
            samples = list(self.samples)
        
        # Filter by time window
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            samples = [s for s in samples if s.timestamp >= cutoff_time]
        
        # Apply limit
        if limit and len(samples) > limit:
            samples = samples[-limit:]
        
        return samples
    
    def add_alert(self, alert: MetricAlert) -> None:
        """Add performance alert."""
        self.alerts.append(alert)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, sample: MetricSample) -> None:
        """Check if any alerts should trigger."""
        for alert in self.alerts:
            if alert.should_trigger(sample.value, sample.timestamp):
                alert.last_triggered = sample.timestamp
                
                alert_data = {
                    'alert_name': alert.name,
                    'metric_name': self.name,
                    'level': alert.level.value,
                    'message': alert.message,
                    'value': sample.value,
                    'threshold': alert.threshold,
                    'timestamp': sample.timestamp
                }
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_data)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
    
    def _invalidate_cache(self) -> None:
        """Invalidate statistics cache."""
        self._stats_cache = None
    
    def export_data(self) -> Dict[str, Any]:
        """Export all metric data."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'statistics': self.get_statistics(),
            'sample_count': len(self.samples),
            'alerts': [
                {
                    'name': alert.name,
                    'condition': alert.condition,
                    'threshold': alert.threshold,
                    'level': alert.level.value,
                    'last_triggered': alert.last_triggered
                }
                for alert in self.alerts
            ]
        }


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: PerformanceCollector, 
                 labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.labels = labels or {}
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type:
            self.labels['error'] = str(exc_type.__name__)
        
        self.collector.record(execution_time, self.labels)


class SystemResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Metric collectors
        self.cpu_usage = PerformanceCollector("system.cpu.usage", MetricType.GAUGE)
        self.memory_usage = PerformanceCollector("system.memory.usage", MetricType.GAUGE)
        self.memory_percent = PerformanceCollector("system.memory.percent", MetricType.GAUGE)
        self.disk_usage = PerformanceCollector("system.disk.usage", MetricType.GAUGE)
        self.network_io = PerformanceCollector("system.network.io", MetricType.COUNTER)
        self.load_average = PerformanceCollector("system.load.average", MetricType.GAUGE)
        
        # Process-specific metrics
        self.process_memory = PerformanceCollector("process.memory.rss", MetricType.GAUGE)
        self.process_cpu = PerformanceCollector("process.cpu.percent", MetricType.GAUGE)
        self.process_threads = PerformanceCollector("process.threads", MetricType.GAUGE)
        self.process_fds = PerformanceCollector("process.file_descriptors", MetricType.GAUGE)
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default system alerts."""
        # CPU usage alerts
        self.cpu_usage.add_alert(MetricAlert(
            name="high_cpu_usage",
            condition="greater_than",
            threshold=80.0,
            level=AlertLevel.WARNING,
            message="High CPU usage detected"
        ))
        
        self.cpu_usage.add_alert(MetricAlert(
            name="critical_cpu_usage",
            condition="greater_than", 
            threshold=95.0,
            level=AlertLevel.CRITICAL,
            message="Critical CPU usage detected"
        ))
        
        # Memory usage alerts
        self.memory_percent.add_alert(MetricAlert(
            name="high_memory_usage",
            condition="greater_than",
            threshold=85.0,
            level=AlertLevel.WARNING,
            message="High memory usage detected"
        ))
        
        self.memory_percent.add_alert(MetricAlert(
            name="critical_memory_usage",
            condition="greater_than",
            threshold=95.0,
            level=AlertLevel.CRITICAL,
            message="Critical memory usage detected"
        ))
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self.running:
            logger.warning("System monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started system resource monitoring")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped system resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_percent.set(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                total_bytes = net_io.bytes_sent + net_io.bytes_recv
                self.network_io.set(total_bytes)
            
            # Load average (Unix-like systems)
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                self.load_average.set(load_avg)
            
            # Process-specific metrics
            process = psutil.Process()
            
            # Process memory
            process_memory_info = process.memory_info()
            self.process_memory.set(process_memory_info.rss)
            
            # Process CPU
            process_cpu_percent = process.cpu_percent()
            self.process_cpu.set(process_cpu_percent)
            
            # Process threads
            self.process_threads.set(process.num_threads())
            
            # File descriptors (Unix-like systems)
            try:
                if hasattr(process, "num_fds"):
                    self.process_fds.set(process.num_fds())
            except (psutil.AccessDenied, AttributeError):
                pass  # Skip if not available
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        return {
            'cpu': self.cpu_usage.get_statistics(window_seconds=60),
            'memory': {
                'usage_bytes': self.memory_usage.get_statistics(window_seconds=60),
                'usage_percent': self.memory_percent.get_statistics(window_seconds=60)
            },
            'disk': self.disk_usage.get_statistics(window_seconds=60),
            'network': self.network_io.get_statistics(window_seconds=60),
            'load': self.load_average.get_statistics(window_seconds=60),
            'process': {
                'memory': self.process_memory.get_statistics(window_seconds=60),
                'cpu': self.process_cpu.get_statistics(window_seconds=60),
                'threads': self.process_threads.get_statistics(window_seconds=60),
                'file_descriptors': self.process_fds.get_statistics(window_seconds=60)
            }
        }


class ApplicationPerformanceMonitor:
    """Monitor application-specific performance metrics."""
    
    def __init__(self):
        # Request/Response metrics
        self.request_duration = PerformanceCollector("app.request.duration", MetricType.TIMER)
        self.request_count = PerformanceCollector("app.request.count", MetricType.COUNTER)
        self.error_count = PerformanceCollector("app.error.count", MetricType.COUNTER)
        
        # Database metrics
        self.db_query_duration = PerformanceCollector("app.db.query.duration", MetricType.TIMER)
        self.db_connection_pool = PerformanceCollector("app.db.connections", MetricType.GAUGE)
        
        # Cache metrics
        self.cache_hits = PerformanceCollector("app.cache.hits", MetricType.COUNTER)
        self.cache_misses = PerformanceCollector("app.cache.misses", MetricType.COUNTER)
        
        # Custom business metrics
        self.custom_metrics: Dict[str, PerformanceCollector] = {}
        
        # Performance insights
        self.insights_history: List[Dict[str, Any]] = []
    
    def record_request(self, duration: float, status_code: int, 
                      endpoint: str, method: str = "GET"):
        """Record HTTP request metrics."""
        labels = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        self.request_duration.record(duration, labels)
        self.request_count.increment(labels=labels)
        
        if status_code >= 400:
            self.error_count.increment(labels=labels)
    
    def record_db_query(self, duration: float, query_type: str, table: str):
        """Record database query metrics."""
        labels = {
            'query_type': query_type,
            'table': table
        }
        
        self.db_query_duration.record(duration, labels)
    
    def record_cache_access(self, hit: bool, cache_type: str = "default"):
        """Record cache access metrics."""
        labels = {'cache_type': cache_type}
        
        if hit:
            self.cache_hits.increment(labels=labels)
        else:
            self.cache_misses.increment(labels=labels)
    
    def create_custom_metric(self, name: str, metric_type: MetricType) -> PerformanceCollector:
        """Create custom application metric."""
        collector = PerformanceCollector(f"app.custom.{name}", metric_type)
        self.custom_metrics[name] = collector
        return collector
    
    def get_custom_metric(self, name: str) -> Optional[PerformanceCollector]:
        """Get custom metric by name."""
        return self.custom_metrics.get(name)
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and generate insights."""
        insights = {}
        
        # Request performance analysis
        request_stats = self.request_duration.get_statistics(window_seconds=3600)  # Last hour
        if request_stats:
            insights['request_performance'] = {
                'average_response_time': request_stats.get('mean', 0),
                'p95_response_time': request_stats.get('p95', 0),
                'total_requests': self.request_count.get_statistics().get('sum', 0),
                'error_rate': self._calculate_error_rate()
            }
        
        # Database performance analysis
        db_stats = self.db_query_duration.get_statistics(window_seconds=3600)
        if db_stats:
            insights['database_performance'] = {
                'average_query_time': db_stats.get('mean', 0),
                'slowest_query_time': db_stats.get('max', 0),
                'total_queries': db_stats.get('count', 0)
            }
        
        # Cache performance analysis
        cache_hit_stats = self.cache_hits.get_statistics()
        cache_miss_stats = self.cache_misses.get_statistics()
        
        if cache_hit_stats and cache_miss_stats:
            total_hits = cache_hit_stats.get('sum', 0)
            total_misses = cache_miss_stats.get('sum', 0)
            total_accesses = total_hits + total_misses
            
            if total_accesses > 0:
                insights['cache_performance'] = {
                    'hit_rate': total_hits / total_accesses,
                    'total_accesses': total_accesses,
                    'total_hits': total_hits,
                    'total_misses': total_misses
                }
        
        # Performance recommendations
        insights['recommendations'] = self._generate_recommendations(insights)
        
        # Store insights
        insight_record = {
            'timestamp': time.time(),
            'insights': insights
        }
        self.insights_history.append(insight_record)
        
        # Keep only recent insights
        if len(self.insights_history) > 100:
            self.insights_history = self.insights_history[-100:]
        
        return insights
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total_requests = self.request_count.get_statistics().get('sum', 0)
        total_errors = self.error_count.get_statistics().get('sum', 0)
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Request performance recommendations
        if 'request_performance' in insights:
            perf = insights['request_performance']
            
            if perf['average_response_time'] > 1.0:
                recommendations.append("Consider optimizing slow endpoints - average response time > 1s")
            
            if perf['p95_response_time'] > 5.0:
                recommendations.append("High tail latency detected - investigate slowest 5% of requests")
            
            if perf['error_rate'] > 0.05:
                recommendations.append("High error rate detected - review error handling and monitoring")
        
        # Database recommendations
        if 'database_performance' in insights:
            db_perf = insights['database_performance']
            
            if db_perf['average_query_time'] > 0.1:
                recommendations.append("Database queries are slow - consider query optimization or indexing")
            
            if db_perf['slowest_query_time'] > 2.0:
                recommendations.append("Very slow database queries detected - review and optimize")
        
        # Cache recommendations
        if 'cache_performance' in insights:
            cache_perf = insights['cache_performance']
            
            if cache_perf['hit_rate'] < 0.8:
                recommendations.append("Low cache hit rate - review caching strategy and TTL settings")
            
            if cache_perf['total_accesses'] > 10000 and cache_perf['hit_rate'] < 0.9:
                recommendations.append("High-volume caching with suboptimal hit rate - optimize cache keys and eviction policy")
        
        return recommendations
    
    def get_application_summary(self) -> Dict[str, Any]:
        """Get application performance summary."""
        summary = {
            'request_metrics': self.request_duration.export_data(),
            'error_metrics': self.error_count.export_data(),
            'database_metrics': self.db_query_duration.export_data(),
            'cache_metrics': {
                'hits': self.cache_hits.export_data(),
                'misses': self.cache_misses.export_data()
            },
            'custom_metrics': {
                name: collector.export_data() 
                for name, collector in self.custom_metrics.items()
            },
            'performance_insights': self.insights_history[-5:] if self.insights_history else []
        }
        
        return summary


class PerformanceMetricsRegistry:
    """Central registry for all performance metrics."""
    
    def __init__(self):
        self.collectors: Dict[str, PerformanceCollector] = {}
        self.system_monitor = SystemResourceMonitor()
        self.app_monitor = ApplicationPerformanceMonitor()
        
        # Global alert handlers
        self.global_alert_handlers: List[Callable] = []
        
        # Metrics export settings
        self.export_interval = 60.0  # Export metrics every minute
        self.export_enabled = False
        self.export_thread: Optional[threading.Thread] = None
    
    def register_collector(self, collector: PerformanceCollector) -> None:
        """Register a performance collector."""
        self.collectors[collector.name] = collector
        
        # Add global alert handlers to collector
        for handler in self.global_alert_handlers:
            collector.add_alert_callback(handler)
    
    def get_collector(self, name: str) -> Optional[PerformanceCollector]:
        """Get collector by name."""
        return self.collectors.get(name)
    
    def create_collector(self, name: str, metric_type: MetricType) -> PerformanceCollector:
        """Create and register new collector."""
        collector = PerformanceCollector(name, metric_type)
        self.register_collector(collector)
        return collector
    
    def add_global_alert_handler(self, handler: Callable) -> None:
        """Add global alert handler."""
        self.global_alert_handlers.append(handler)
        
        # Add to existing collectors
        for collector in self.collectors.values():
            collector.add_alert_callback(handler)
    
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        
        if self.export_enabled:
            self._start_export_thread()
        
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        
        if self.export_thread:
            self.export_enabled = False
            self.export_thread.join(timeout=2.0)
        
        logger.info("Stopped performance monitoring")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'system_summary': self.system_monitor.get_system_summary(),
            'application_summary': self.app_monitor.get_application_summary(),
            'custom_collectors': {
                name: collector.export_data()
                for name, collector in self.collectors.items()
            },
            'performance_insights': self.app_monitor.analyze_performance_trends()
        }
        
        return report
    
    def export_metrics_json(self) -> str:
        """Export all metrics as JSON."""
        report = self.get_comprehensive_report()
        return json.dumps(report, indent=2, default=str)
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        current_time = int(time.time() * 1000)  # Prometheus uses milliseconds
        
        # System metrics
        system_summary = self.system_monitor.get_system_summary()
        
        for metric_name, stats in system_summary.items():
            if isinstance(stats, dict) and 'current' in stats:
                prometheus_name = f"system_{metric_name.replace('.', '_')}"
                lines.append(f"{prometheus_name} {stats['current']} {current_time}")
        
        # Application metrics
        app_summary = self.app_monitor.get_application_summary()
        
        for metric_group, metrics in app_summary.items():
            if isinstance(metrics, dict) and 'statistics' in metrics:
                stats = metrics['statistics']
                if 'current' in stats:
                    prometheus_name = f"app_{metric_group.replace('.', '_')}"
                    lines.append(f"{prometheus_name} {stats['current']} {current_time}")
        
        return '\n'.join(lines)
    
    def _start_export_thread(self) -> None:
        """Start metrics export thread."""
        self.export_thread = threading.Thread(target=self._export_loop)
        self.export_thread.daemon = True
        self.export_thread.start()
    
    def _export_loop(self) -> None:
        """Export metrics periodically."""
        while self.export_enabled:
            try:
                # Export metrics (placeholder - implement actual export logic)
                report = self.get_comprehensive_report()
                logger.debug(f"Exported {len(report)} metric groups")
                
                time.sleep(self.export_interval)
            except Exception as e:
                logger.error(f"Error in metrics export: {e}")
                time.sleep(self.export_interval)


# Performance decorators for easy instrumentation
def measure_execution_time(metric_name: str, registry: Optional[PerformanceMetricsRegistry] = None):
    """Decorator to measure function execution time."""
    def decorator(func):
        # Get or create collector
        if registry:
            collector = registry.get_collector(metric_name)
            if not collector:
                collector = registry.create_collector(metric_name, MetricType.TIMER)
        else:
            collector = PerformanceCollector(metric_name, MetricType.TIMER)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return collector.time_it(func)(*args, **kwargs)
        
        return wrapper
    return decorator


def count_invocations(metric_name: str, registry: Optional[PerformanceMetricsRegistry] = None):
    """Decorator to count function invocations."""
    def decorator(func):
        # Get or create collector
        if registry:
            collector = registry.get_collector(metric_name)
            if not collector:
                collector = registry.create_collector(metric_name, MetricType.COUNTER)
        else:
            collector = PerformanceCollector(metric_name, MetricType.COUNTER)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                collector.increment(labels={'status': 'success'})
                return result
            except Exception as e:
                collector.increment(labels={'status': 'error', 'error_type': type(e).__name__})
                raise
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize performance monitoring
    registry = PerformanceMetricsRegistry()
    
    # Add alert handler
    def alert_handler(alert_data):
        print(f"ALERT: {alert_data['level']} - {alert_data['message']} (value: {alert_data['value']})")
    
    registry.add_global_alert_handler(alert_handler)
    
    # Start monitoring
    registry.start_monitoring()
    
    # Demo custom metrics
    custom_metric = registry.create_collector("demo.operations", MetricType.COUNTER)
    timing_metric = registry.create_collector("demo.timing", MetricType.TIMER)
    
    # Demo functions with decorators
    @measure_execution_time("demo.function.timing", registry)
    @count_invocations("demo.function.calls", registry)
    def demo_function(duration=0.1):
        time.sleep(duration)
        return "completed"
    
    try:
        print("Running performance monitoring demo...")
        
        # Generate some metrics
        for i in range(20):
            # Record custom metrics
            custom_metric.increment()
            
            # Call demo function
            if i % 5 == 0:
                demo_function(0.2)  # Slower execution
            else:
                demo_function(0.05)  # Normal execution
            
            # Record application metrics
            registry.app_monitor.record_request(
                duration=np.random.uniform(0.01, 0.5),
                status_code=200 if i % 10 != 0 else 500,
                endpoint=f"/api/endpoint{i % 3}",
                method="GET"
            )
            
            time.sleep(0.1)
        
        # Wait for some metrics collection
        time.sleep(2)
        
        # Generate comprehensive report
        report = registry.get_comprehensive_report()
        
        print("\n=== Performance Report ===")
        print(f"System CPU Usage: {report['system_summary']['cpu'].get('current', 0):.1f}%")
        print(f"System Memory Usage: {report['system_summary']['memory']['usage_percent'].get('current', 0):.1f}%")
        
        if 'performance_insights' in report:
            insights = report['performance_insights']
            if 'recommendations' in insights:
                print("\nRecommendations:")
                for rec in insights['recommendations']:
                    print(f"  - {rec}")
        
        # Export metrics
        print("\n=== Metrics Export ===")
        json_export = registry.export_metrics_json()
        print(f"JSON export size: {len(json_export)} characters")
        
        prometheus_export = registry.export_metrics_prometheus()
        print(f"Prometheus export size: {len(prometheus_export)} characters")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        registry.stop_monitoring()
        print("Performance monitoring stopped")