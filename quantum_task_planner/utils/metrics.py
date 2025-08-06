"""
Performance Metrics and Monitoring for Quantum Task Planner

Comprehensive metrics collection, analysis, and reporting system
for quantum-inspired task planning operations.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: float
    operation_type: str
    duration: float
    success: bool
    quantum_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics over time period."""
    operation_type: str
    count: int
    success_rate: float
    avg_duration: float
    median_duration: float
    p95_duration: float
    p99_duration: float
    quantum_coherence_avg: float
    quantum_entanglement_avg: float
    throughput_per_second: float


class PlannerMetrics:
    """
    Comprehensive metrics collection and analysis for quantum task planner.
    
    Tracks performance, quantum-specific metrics, resource utilization,
    and system health indicators.
    """
    
    def __init__(self, max_history: int = 10000, aggregation_window: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of snapshots to keep in memory
            aggregation_window: Window size for rolling averages
        """
        self.max_history = max_history
        self.aggregation_window = aggregation_window
        
        # Performance tracking
        self.snapshots: deque = deque(maxlen=max_history)
        self.operation_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.duration_history = defaultdict(list)
        
        # Quantum-specific metrics
        self.quantum_metrics_history = defaultdict(list)
        self.coherence_measurements = deque(maxlen=1000)
        self.entanglement_measurements = deque(maxlen=1000)
        
        # Resource utilization
        self.resource_utilization_history = {}
        self.memory_usage_history = deque(maxlen=1000)
        self.cpu_usage_history = deque(maxlen=1000)
        
        # System health
        self.error_counts = defaultdict(int)
        self.warning_counts = defaultdict(int)
        self.anomaly_detections = []
        
        # Real-time monitoring
        self.current_operations = {}  # Track ongoing operations
        self.alerts_enabled = True
        self.performance_thresholds = {
            'max_duration': 60.0,  # seconds
            'min_success_rate': 0.85,
            'max_memory_usage': 0.90,  # 90% of available
            'min_quantum_coherence': 0.3
        }
        
        logger.info(f"Initialized PlannerMetrics with {max_history} max history")
    
    def start_operation(self, operation_id: str, operation_type: str, metadata: Dict[str, Any] = None) -> float:
        """
        Start tracking an operation.
        
        Args:
            operation_id: Unique identifier for operation
            operation_type: Type of operation (e.g., 'planning', 'scheduling')
            metadata: Additional metadata
            
        Returns:
            Start timestamp
        """
        start_time = time.time()
        self.current_operations[operation_id] = {
            'type': operation_type,
            'start_time': start_time,
            'metadata': metadata or {}
        }
        
        logger.debug(f"Started operation {operation_id} of type {operation_type}")
        return start_time
    
    def end_operation(self, 
                     operation_id: str, 
                     success: bool = True,
                     quantum_metrics: Dict[str, float] = None) -> PerformanceSnapshot:
        """
        End tracking an operation and record metrics.
        
        Args:
            operation_id: Operation identifier
            success: Whether operation succeeded
            quantum_metrics: Quantum-specific measurements
            
        Returns:
            Performance snapshot
        """
        end_time = time.time()
        
        if operation_id not in self.current_operations:
            logger.warning(f"Operation {operation_id} not found in current operations")
            return None
        
        operation = self.current_operations.pop(operation_id)
        duration = end_time - operation['start_time']
        
        snapshot = PerformanceSnapshot(
            timestamp=end_time,
            operation_type=operation['type'],
            duration=duration,
            success=success,
            quantum_metrics=quantum_metrics or {},
            metadata=operation['metadata']
        )
        
        self.record_snapshot(snapshot)
        logger.debug(f"Completed operation {operation_id} in {duration:.3f}s")
        
        return snapshot
    
    def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Record a performance snapshot."""
        self.snapshots.append(snapshot)
        
        # Update counters
        self.operation_counts[snapshot.operation_type] += 1
        if snapshot.success:
            self.success_counts[snapshot.operation_type] += 1
        
        # Track duration
        self.duration_history[snapshot.operation_type].append(snapshot.duration)
        if len(self.duration_history[snapshot.operation_type]) > self.max_history:
            self.duration_history[snapshot.operation_type].pop(0)
        
        # Track quantum metrics
        for metric_name, value in snapshot.quantum_metrics.items():
            self.quantum_metrics_history[metric_name].append(value)
            if len(self.quantum_metrics_history[metric_name]) > self.max_history:
                self.quantum_metrics_history[metric_name].pop(0)
        
        # Special handling for key quantum metrics
        if 'coherence' in snapshot.quantum_metrics:
            self.coherence_measurements.append(snapshot.quantum_metrics['coherence'])
        
        if 'entanglement' in snapshot.quantum_metrics:
            self.entanglement_measurements.append(snapshot.quantum_metrics['entanglement'])
        
        # Check for anomalies and alerts
        self._check_performance_alerts(snapshot)
    
    def record_optimization_time(self, duration: float) -> None:
        """Record optimization operation time."""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            operation_type='optimization',
            duration=duration,
            success=True,
            quantum_metrics={'optimization_time': duration}
        )
        self.record_snapshot(snapshot)
    
    def record_planning_result(self, result: Dict[str, Any]) -> None:
        """Record complete planning result."""
        planning_time = result.get('planning_time', 0.0)
        quantum_metrics = result.get('quantum_metrics', {})
        
        # Add derived metrics
        if 'selected_tasks' in result and 'total_tasks' in result:
            selection_ratio = result['selected_tasks'] / max(result['total_tasks'], 1)
            quantum_metrics['task_selection_ratio'] = selection_ratio
        
        if 'energy' in result:
            quantum_metrics['final_energy'] = result['energy']
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            operation_type='planning',
            duration=planning_time,
            success=result.get('success', True),
            quantum_metrics=quantum_metrics
        )
        self.record_snapshot(snapshot)
    
    def record_scheduling_result(self, result: Any) -> None:
        """Record scheduling operation result."""
        # Handle different result types (assuming it has scheduling_time and quantum_metrics)
        if hasattr(result, 'scheduling_time'):
            duration = result.scheduling_time
        else:
            duration = 0.0
        
        if hasattr(result, 'quantum_metrics'):
            quantum_metrics = result.quantum_metrics
        else:
            quantum_metrics = {}
        
        if hasattr(result, 'success_rate'):
            quantum_metrics['success_rate'] = result.success_rate
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            operation_type='scheduling',
            duration=duration,
            success=True,  # Assume success for now
            quantum_metrics=quantum_metrics
        )
        self.record_snapshot(snapshot)
    
    def get_aggregated_metrics(self, 
                              operation_type: Optional[str] = None,
                              time_window: Optional[float] = None) -> Dict[str, AggregatedMetrics]:
        """
        Get aggregated metrics for analysis.
        
        Args:
            operation_type: Filter by operation type
            time_window: Time window in seconds (None for all time)
            
        Returns:
            Dictionary of aggregated metrics by operation type
        """
        current_time = time.time()
        
        # Filter snapshots
        filtered_snapshots = []
        for snapshot in self.snapshots:
            if operation_type and snapshot.operation_type != operation_type:
                continue
            if time_window and current_time - snapshot.timestamp > time_window:
                continue
            filtered_snapshots.append(snapshot)
        
        # Group by operation type
        grouped_snapshots = defaultdict(list)
        for snapshot in filtered_snapshots:
            grouped_snapshots[snapshot.operation_type].append(snapshot)
        
        # Calculate aggregated metrics
        aggregated = {}
        for op_type, snapshots in grouped_snapshots.items():
            if not snapshots:
                continue
            
            durations = [s.duration for s in snapshots]
            success_count = sum(1 for s in snapshots if s.success)
            
            # Quantum metrics aggregation
            coherence_values = []
            entanglement_values = []
            for s in snapshots:
                if 'coherence' in s.quantum_metrics:
                    coherence_values.append(s.quantum_metrics['coherence'])
                if 'entanglement' in s.quantum_metrics:
                    entanglement_values.append(s.quantum_metrics['entanglement'])
            
            # Calculate throughput
            if time_window:
                throughput = len(snapshots) / time_window
            else:
                time_span = max(s.timestamp for s in snapshots) - min(s.timestamp for s in snapshots)
                throughput = len(snapshots) / max(time_span, 1.0)
            
            aggregated[op_type] = AggregatedMetrics(
                operation_type=op_type,
                count=len(snapshots),
                success_rate=success_count / len(snapshots),
                avg_duration=np.mean(durations),
                median_duration=np.median(durations),
                p95_duration=np.percentile(durations, 95),
                p99_duration=np.percentile(durations, 99),
                quantum_coherence_avg=np.mean(coherence_values) if coherence_values else 0.0,
                quantum_entanglement_avg=np.mean(entanglement_values) if entanglement_values else 0.0,
                throughput_per_second=throughput
            )
        
        return aggregated
    
    def get_quantum_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive quantum metrics summary."""
        summary = {}
        
        # Coherence statistics
        if self.coherence_measurements:
            coherence_array = np.array(self.coherence_measurements)
            summary['coherence'] = {
                'mean': np.mean(coherence_array),
                'std': np.std(coherence_array),
                'min': np.min(coherence_array),
                'max': np.max(coherence_array),
                'trend': self._calculate_trend(coherence_array)
            }
        
        # Entanglement statistics
        if self.entanglement_measurements:
            entanglement_array = np.array(self.entanglement_measurements)
            summary['entanglement'] = {
                'mean': np.mean(entanglement_array),
                'std': np.std(entanglement_array),
                'min': np.min(entanglement_array),
                'max': np.max(entanglement_array),
                'trend': self._calculate_trend(entanglement_array)
            }
        
        # Other quantum metrics
        for metric_name, values in self.quantum_metrics_history.items():
            if metric_name not in ['coherence', 'entanglement'] and values:
                values_array = np.array(values)
                summary[metric_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array)
                }
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = time.time()
        
        # Recent metrics (last hour)
        recent_aggregated = self.get_aggregated_metrics(time_window=3600)
        
        # All-time metrics
        all_time_aggregated = self.get_aggregated_metrics()
        
        # System health indicators
        total_operations = sum(self.operation_counts.values())
        total_successes = sum(self.success_counts.values())
        overall_success_rate = total_successes / max(total_operations, 1)
        
        report = {
            'timestamp': current_time,
            'system_health': {
                'total_operations': total_operations,
                'overall_success_rate': overall_success_rate,
                'active_operations': len(self.current_operations),
                'error_count': sum(self.error_counts.values()),
                'warning_count': sum(self.warning_counts.values())
            },
            'recent_performance': recent_aggregated,
            'all_time_performance': all_time_aggregated,
            'quantum_metrics': self.get_quantum_metrics_summary(),
            'resource_utilization': self._get_resource_summary(),
            'anomalies': self.anomaly_detections[-10:]  # Last 10 anomalies
        }
        
        return report
    
    def get_summary(self) -> Dict[str, Any]:
        """Get concise metrics summary."""
        aggregated = self.get_aggregated_metrics()
        quantum_summary = self.get_quantum_metrics_summary()
        
        total_operations = sum(self.operation_counts.values())
        total_successes = sum(self.success_counts.values())
        
        return {
            'total_operations': total_operations,
            'success_rate': total_successes / max(total_operations, 1),
            'operations_by_type': dict(self.operation_counts),
            'recent_aggregated': aggregated,
            'quantum_metrics': quantum_summary,
            'active_operations': len(self.current_operations)
        }
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export metrics to file."""
        report = self.get_performance_report()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported metrics to {filepath}")
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance anomalies and generate alerts."""
        if not self.alerts_enabled:
            return
        
        alerts = []
        
        # Duration threshold check
        if snapshot.duration > self.performance_thresholds['max_duration']:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"Operation {snapshot.operation_type} took {snapshot.duration:.2f}s (threshold: {self.performance_thresholds['max_duration']}s)",
                'timestamp': snapshot.timestamp
            })
        
        # Quantum coherence check
        if 'coherence' in snapshot.quantum_metrics:
            coherence = snapshot.quantum_metrics['coherence']
            if coherence < self.performance_thresholds['min_quantum_coherence']:
                alerts.append({
                    'type': 'quantum',
                    'severity': 'warning',
                    'message': f"Low quantum coherence detected: {coherence:.3f} (threshold: {self.performance_thresholds['min_quantum_coherence']})",
                    'timestamp': snapshot.timestamp
                })
        
        # Success rate check (for recent operations)
        op_type = snapshot.operation_type
        recent_snapshots = [s for s in list(self.snapshots)[-50:] if s.operation_type == op_type]
        if len(recent_snapshots) >= 10:
            recent_success_rate = sum(1 for s in recent_snapshots if s.success) / len(recent_snapshots)
            if recent_success_rate < self.performance_thresholds['min_success_rate']:
                alerts.append({
                    'type': 'reliability',
                    'severity': 'error',
                    'message': f"Low success rate for {op_type}: {recent_success_rate:.2f} (threshold: {self.performance_thresholds['min_success_rate']})",
                    'timestamp': snapshot.timestamp
                })
        
        # Log alerts
        for alert in alerts:
            if alert['severity'] == 'error':
                logger.error(f"ALERT: {alert['message']}")
                self.error_counts[alert['type']] += 1
            else:
                logger.warning(f"ALERT: {alert['message']}")
                self.warning_counts[alert['type']] += 1
        
        # Store anomalies
        if alerts:
            self.anomaly_detections.extend(alerts)
            # Keep only recent anomalies
            self.anomaly_detections = self.anomaly_detections[-100:]
    
    def _calculate_trend(self, values: np.ndarray, window_size: int = 20) -> str:
        """Calculate trend direction for time series values."""
        if len(values) < window_size:
            return 'insufficient_data'
        
        # Compare recent vs older values
        recent_avg = np.mean(values[-window_size:])
        older_avg = np.mean(values[-2*window_size:-window_size])
        
        if recent_avg > older_avg * 1.05:
            return 'increasing'
        elif recent_avg < older_avg * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_resource_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary."""
        # Simplified resource tracking for v1
        return {
            'memory_samples': len(self.memory_usage_history),
            'cpu_samples': len(self.cpu_usage_history),
            'resource_types_tracked': len(self.resource_utilization_history)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)."""
        self.snapshots.clear()
        self.operation_counts.clear()
        self.success_counts.clear()
        self.duration_history.clear()
        self.quantum_metrics_history.clear()
        self.coherence_measurements.clear()
        self.entanglement_measurements.clear()
        self.resource_utilization_history.clear()
        self.memory_usage_history.clear()
        self.cpu_usage_history.clear()
        self.error_counts.clear()
        self.warning_counts.clear()
        self.anomaly_detections.clear()
        self.current_operations.clear()
        
        logger.info("All metrics have been reset")