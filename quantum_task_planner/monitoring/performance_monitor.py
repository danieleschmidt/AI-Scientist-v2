"""
Performance Monitoring and Alerting for Quantum Task Planner

Real-time performance monitoring, alerting, and optimization
recommendations for quantum task planning operations.
"""

import time
import statistics
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Performance alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput" 
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUANTUM_METRIC = "quantum_metric"
    SUCCESS_RATE = "success_rate"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "greater_than"  # "greater_than", "less_than"
    window_size: int = 10  # Number of samples to consider
    min_samples: int = 5   # Minimum samples needed


@dataclass 
class PerformanceAlert:
    """Performance alert notification."""
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_name: str
    direction: str  # "increasing", "decreasing", "stable", "volatile"
    rate_of_change: float
    confidence: float
    prediction_1h: Optional[float] = None
    prediction_24h: Optional[float] = None


class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting system.
    
    Tracks quantum planning performance metrics, detects anomalies,
    and provides optimization recommendations.
    """
    
    def __init__(self,
                 metric_retention: int = 10000,
                 alert_retention: int = 1000,
                 trend_analysis_window: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            metric_retention: Number of metric samples to retain
            alert_retention: Number of alerts to retain in history
            trend_analysis_window: Window size for trend analysis
        """
        self.metric_retention = metric_retention
        self.alert_retention = alert_retention
        self.trend_analysis_window = trend_analysis_window
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metric_retention))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Thresholds and alerting
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alert_history: deque = deque(maxlen=alert_retention)
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance tracking
        self.operation_start_times: Dict[str, float] = {}
        self.performance_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0
        })
        
        # Trend analysis
        self.trend_cache: Dict[str, PerformanceTrend] = {}
        self.trend_update_interval = 60.0  # seconds
        self.last_trend_update = 0.0
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Set up default thresholds
        self._setup_default_thresholds()
        
        logger.info("Initialized PerformanceMonitor with real-time alerting")
    
    def _setup_default_thresholds(self) -> None:
        """Set up default performance thresholds."""
        
        # Planning operation latency
        self.add_threshold(PerformanceThreshold(
            metric_name="planning_latency",
            warning_threshold=30.0,  # 30 seconds
            critical_threshold=60.0,  # 60 seconds
            comparison="greater_than"
        ))
        
        # Scheduling operation latency
        self.add_threshold(PerformanceThreshold(
            metric_name="scheduling_latency", 
            warning_threshold=20.0,
            critical_threshold=45.0,
            comparison="greater_than"
        ))
        
        # Optimization convergence time
        self.add_threshold(PerformanceThreshold(
            metric_name="optimization_time",
            warning_threshold=15.0,
            critical_threshold=30.0,
            comparison="greater_than"
        ))
        
        # Success rate
        self.add_threshold(PerformanceThreshold(
            metric_name="success_rate",
            warning_threshold=0.80,  # 80%
            critical_threshold=0.70,  # 70%
            comparison="less_than"
        ))
        
        # Quantum coherence
        self.add_threshold(PerformanceThreshold(
            metric_name="quantum_coherence",
            warning_threshold=0.30,
            critical_threshold=0.20,
            comparison="less_than"
        ))
        
        # Error rate
        self.add_threshold(PerformanceThreshold(
            metric_name="error_rate",
            warning_threshold=0.05,  # 5%
            critical_threshold=0.10,  # 10%
            comparison="greater_than"
        ))
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add performance threshold."""
        self.thresholds[threshold.metric_name] = threshold
        logger.debug(f"Added performance threshold for {threshold.metric_name}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
        logger.debug("Added performance alert callback")
    
    def start_operation(self, operation_id: str, operation_type: str) -> None:
        """Start timing an operation."""
        self.operation_start_times[operation_id] = time.time()
    
    def end_operation(self, 
                     operation_id: str, 
                     operation_type: str,
                     success: bool = True,
                     metadata: Dict[str, Any] = None) -> Optional[float]:
        """
        End timing an operation and record metrics.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            success: Whether operation succeeded
            metadata: Additional operation metadata
            
        Returns:
            Operation duration in seconds
        """
        if operation_id not in self.operation_start_times:
            logger.warning(f"No start time found for operation {operation_id}")
            return None
        
        # Calculate duration
        start_time = self.operation_start_times.pop(operation_id)
        duration = time.time() - start_time
        
        # Update performance stats
        stats = self.performance_stats[operation_type]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        if not success:
            stats['error_count'] += 1
        
        # Record latency metric
        self.record_metric(PerformanceMetric(
            name=f"{operation_type}_latency",
            value=duration,
            metric_type=MetricType.LATENCY,
            timestamp=time.time(),
            unit="seconds",
            tags={"operation_type": operation_type, "success": str(success)}
        ))
        
        # Calculate and record success rate
        success_rate = (stats['count'] - stats['error_count']) / stats['count']
        self.record_metric(PerformanceMetric(
            name=f"{operation_type}_success_rate",
            value=success_rate,
            metric_type=MetricType.SUCCESS_RATE,
            timestamp=time.time(),
            unit="ratio",
            tags={"operation_type": operation_type}
        ))
        
        # Calculate and record error rate
        error_rate = stats['error_count'] / stats['count']
        self.record_metric(PerformanceMetric(
            name=f"{operation_type}_error_rate",
            value=error_rate,
            metric_type=MetricType.ERROR_RATE,
            timestamp=time.time(),
            unit="ratio",
            tags={"operation_type": operation_type}
        ))
        
        logger.debug(f"Recorded {operation_type} operation: {duration:.3f}s, success: {success}")
        return duration
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        # Store metric
        self.metrics[metric.name].append(metric)
        
        # Store metadata
        if metric.name not in self.metric_metadata:
            self.metric_metadata[metric.name] = {
                'metric_type': metric.metric_type.value,
                'unit': metric.unit,
                'first_recorded': metric.timestamp
            }
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Update trends periodically
        current_time = time.time()
        if current_time - self.last_trend_update > self.trend_update_interval:
            self._update_trends()
            self.last_trend_update = current_time
    
    def record_quantum_metrics(self, quantum_metrics: Dict[str, float]) -> None:
        """Record quantum-specific metrics."""
        current_time = time.time()
        
        for metric_name, value in quantum_metrics.items():
            self.record_metric(PerformanceMetric(
                name=f"quantum_{metric_name}",
                value=value,
                metric_type=MetricType.QUANTUM_METRIC,
                timestamp=current_time,
                tags={"category": "quantum"}
            ))
    
    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric violates any thresholds."""
        threshold = self.thresholds.get(metric.name)
        if not threshold:
            return
        
        # Get recent values for the metric
        recent_metrics = list(self.metrics[metric.name])[-threshold.window_size:]
        
        if len(recent_metrics) < threshold.min_samples:
            return  # Not enough samples
        
        # Calculate aggregate value (mean for now)
        recent_values = [m.value for m in recent_metrics]
        avg_value = statistics.mean(recent_values)
        
        # Check thresholds
        alert = None
        
        if threshold.comparison == "greater_than":
            if avg_value > threshold.critical_threshold:
                alert = PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    metric_name=metric.name,
                    current_value=avg_value,
                    threshold_value=threshold.critical_threshold,
                    message=f"{metric.name} critically high: {avg_value:.3f} > {threshold.critical_threshold}",
                    timestamp=time.time()
                )
            elif avg_value > threshold.warning_threshold:
                alert = PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    metric_name=metric.name,
                    current_value=avg_value,
                    threshold_value=threshold.warning_threshold,
                    message=f"{metric.name} above warning threshold: {avg_value:.3f} > {threshold.warning_threshold}",
                    timestamp=time.time()
                )
        
        elif threshold.comparison == "less_than":
            if avg_value < threshold.critical_threshold:
                alert = PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    metric_name=metric.name,
                    current_value=avg_value,
                    threshold_value=threshold.critical_threshold,
                    message=f"{metric.name} critically low: {avg_value:.3f} < {threshold.critical_threshold}",
                    timestamp=time.time()
                )
            elif avg_value < threshold.warning_threshold:
                alert = PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    metric_name=metric.name,
                    current_value=avg_value,
                    threshold_value=threshold.warning_threshold,
                    message=f"{metric.name} below warning threshold: {avg_value:.3f} < {threshold.warning_threshold}",
                    timestamp=time.time()
                )
        
        if alert:
            self._send_alert(alert)
    
    def _send_alert(self, alert: PerformanceAlert) -> None:
        """Send performance alert."""
        # Store in history
        self.alert_history.append(alert)
        
        # Log alert
        if alert.severity == AlertSeverity.CRITICAL:
            logger.error(f"CRITICAL PERFORMANCE ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"PERFORMANCE WARNING: {alert.message}")
        else:
            logger.info(f"PERFORMANCE INFO: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Performance alert callback failed: {e}")
    
    def _update_trends(self) -> None:
        """Update performance trend analysis."""
        for metric_name, metric_deque in self.metrics.items():
            if len(metric_deque) < 10:  # Need minimum data for trends
                continue
            
            try:
                trend = self._analyze_trend(metric_name, list(metric_deque))
                self.trend_cache[metric_name] = trend
            except Exception as e:
                logger.error(f"Trend analysis failed for {metric_name}: {e}")
    
    def _analyze_trend(self, metric_name: str, metrics: List[PerformanceMetric]) -> PerformanceTrend:
        """Analyze trend for a metric."""
        # Use recent data for trend analysis
        recent_metrics = metrics[-self.trend_analysis_window:]
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        if len(values) < 10:
            return PerformanceTrend(
                metric_name=metric_name,
                direction="insufficient_data",
                rate_of_change=0.0,
                confidence=0.0
            )
        
        # Simple linear regression for trend
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))
        
        # Calculate slope (rate of change)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate correlation coefficient for confidence
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
        denom_x = sum((i - mean_x) ** 2 for i in range(n))
        denom_y = sum((v - mean_y) ** 2 for v in values)
        
        if denom_x > 0 and denom_y > 0:
            correlation = numerator / (denom_x * denom_y) ** 0.5
            confidence = abs(correlation)
        else:
            correlation = 0.0
            confidence = 0.0
        
        # Determine trend direction
        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Add volatility check
        if len(values) > 5:
            volatility = statistics.stdev(values) / abs(statistics.mean(values)) if statistics.mean(values) != 0 else 0
            if volatility > 0.5:  # High relative standard deviation
                direction = "volatile"
        
        # Simple prediction (linear extrapolation)
        current_value = values[-1]
        prediction_1h = current_value + slope * 3600  # 1 hour ahead
        prediction_24h = current_value + slope * 86400  # 24 hours ahead
        
        return PerformanceTrend(
            metric_name=metric_name,
            direction=direction,
            rate_of_change=slope,
            confidence=confidence,
            prediction_1h=prediction_1h,
            prediction_24h=prediction_24h
        )
    
    def get_performance_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Args:
            time_window: Time window in seconds (None for all data)
            
        Returns:
            Performance summary dictionary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        summary = {
            'timestamp': current_time,
            'time_window_hours': time_window / 3600 if time_window else 'all',
            'operation_stats': {},
            'metric_summaries': {},
            'recent_alerts': [],
            'trends': {},
            'recommendations': []
        }
        
        # Operation statistics
        for op_type, stats in self.performance_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                error_rate = stats['error_count'] / stats['count']
                success_rate = (stats['count'] - stats['error_count']) / stats['count']
                
                summary['operation_stats'][op_type] = {
                    'total_operations': stats['count'],
                    'avg_duration': avg_time,
                    'min_duration': stats['min_time'],
                    'max_duration': stats['max_time'],
                    'success_rate': success_rate,
                    'error_rate': error_rate
                }
        
        # Metric summaries
        for metric_name, metric_deque in self.metrics.items():
            recent_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary['metric_summaries'][metric_name] = {
                    'count': len(values),
                    'current': values[-1],
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'unit': self.metric_metadata.get(metric_name, {}).get('unit', '')
                }
        
        # Recent alerts
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        summary['recent_alerts'] = [
            {
                'severity': alert.severity.value,
                'metric': alert.metric_name,
                'message': alert.message,
                'timestamp': alert.timestamp
            }
            for alert in recent_alerts[-10:]  # Last 10 alerts
        ]
        
        # Trends
        summary['trends'] = {
            name: {
                'direction': trend.direction,
                'rate_of_change': trend.rate_of_change,
                'confidence': trend.confidence,
                'prediction_1h': trend.prediction_1h,
                'prediction_24h': trend.prediction_24h
            }
            for name, trend in self.trend_cache.items()
        }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check for high latency operations
        for op_type, stats in summary.get('operation_stats', {}).items():
            if stats.get('avg_duration', 0) > 30.0:
                recommendations.append(
                    f"Consider optimizing {op_type} operations - average duration is {stats['avg_duration']:.1f}s"
                )
        
        # Check for low success rates
        for op_type, stats in summary.get('operation_stats', {}).items():
            if stats.get('success_rate', 1.0) < 0.85:
                recommendations.append(
                    f"Investigate {op_type} failures - success rate is {stats['success_rate']:.1%}"
                )
        
        # Check for concerning trends
        for metric_name, trend in summary.get('trends', {}).items():
            if trend['direction'] == 'increasing' and 'error' in metric_name:
                recommendations.append(
                    f"Rising {metric_name} trend detected - investigate root cause"
                )
            elif trend['direction'] == 'decreasing' and 'success' in metric_name:
                recommendations.append(
                    f"Declining {metric_name} trend detected - review system performance"
                )
        
        # Check for volatile metrics
        for metric_name, trend in summary.get('trends', {}).items():
            if trend['direction'] == 'volatile':
                recommendations.append(
                    f"Metric {metric_name} shows high volatility - consider system stability"
                )
        
        return recommendations
    
    def get_metric_data(self, 
                       metric_name: str, 
                       time_window: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_name: Name of metric
            time_window: Time window in seconds
            
        Returns:
            List of (timestamp, value) tuples
        """
        if metric_name not in self.metrics:
            return []
        
        metrics = list(self.metrics[metric_name])
        
        if time_window:
            cutoff_time = time.time() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return [(m.timestamp, m.value) for m in metrics]
    
    def export_metrics(self, filepath: str, time_window: Optional[float] = None) -> None:
        """Export performance metrics to JSON file."""
        summary = self.get_performance_summary(time_window)
        
        # Add raw metric data
        summary['raw_metrics'] = {}
        for metric_name in self.metrics.keys():
            summary['raw_metrics'][metric_name] = self.get_metric_data(metric_name, time_window)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Exported performance metrics to {filepath}")
    
    def clear_metrics(self) -> None:
        """Clear all metrics and reset counters (use with caution)."""
        self.metrics.clear()
        self.metric_metadata.clear()
        self.alert_history.clear()
        self.operation_start_times.clear()
        self.performance_stats.clear()
        self.trend_cache.clear()
        
        logger.warning("Cleared all performance metrics and statistics")