"""
Tests for monitoring and observability components.

Comprehensive test suite for health monitoring, performance monitoring,
and quantum-specific monitoring systems.
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from quantum_task_planner.monitoring.health_monitor import (
    HealthMonitor, SystemHealth, HealthStatus, HealthCheck, HealthMetric
)
from quantum_task_planner.monitoring.performance_monitor import (
    PerformanceMonitor, PerformanceAlert, AlertSeverity, MetricType,
    PerformanceThreshold, PerformanceMetric, PerformanceTrend
)
from quantum_task_planner.monitoring.quantum_monitor import (
    QuantumMetricsMonitor, QuantumState, QuantumCircuitMetrics,
    DecoherenceEvent, QuantumMetricType
)


class TestHealthMonitor:
    """Test suite for HealthMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor(
            check_interval=1.0,  # Fast for testing
            metric_history_size=100
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'monitor') and self.monitor.is_running:
            self.monitor.stop_monitoring()
    
    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(check_interval=30.0, metric_history_size=500)
        
        assert monitor.check_interval == 30.0
        assert monitor.metric_history_size == 500
        assert not monitor.is_running
        assert len(monitor.health_checks) > 0  # Should have default checks
        assert len(monitor.alert_callbacks) == 0
    
    def test_add_health_check(self):
        """Test adding custom health checks."""
        def custom_check():
            return True
        
        health_check = HealthCheck(
            name="custom_check",
            check_function=custom_check,
            description="Custom test check",
            check_interval=15.0,
            critical=True
        )
        
        initial_count = len(self.monitor.health_checks)
        self.monitor.add_health_check(health_check)
        
        assert len(self.monitor.health_checks) == initial_count + 1
        assert "custom_check" in self.monitor.health_checks
        assert self.monitor.health_checks["custom_check"].critical == True
    
    def test_add_alert_callback(self):
        """Test adding alert callbacks."""
        callback = Mock()
        
        self.monitor.add_alert_callback(callback)
        assert callback in self.monitor.alert_callbacks
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor.is_running
        
        self.monitor.start_monitoring()
        assert self.monitor.is_running
        assert self.monitor.monitor_thread is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_running
    
    @patch('psutil.Process')
    def test_memory_usage_check(self, mock_process):
        """Test memory usage health check."""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 500  # 500MB
        
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.percent = 75.0
            
            # Run check
            result = self.monitor._check_memory_usage()
            assert result == True  # Should pass with normal usage
    
    @patch('psutil.Process')
    def test_cpu_usage_check(self, mock_process):
        """Test CPU usage health check."""
        # Mock process CPU usage
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        # Mock system CPU
        with patch('psutil.cpu_percent', return_value=60.0):
            result = self.monitor._check_cpu_usage()
            assert result == True  # Should pass with normal usage
    
    @patch('psutil.disk_usage')
    def test_disk_space_check(self, mock_disk_usage):
        """Test disk space health check."""
        # Mock disk usage with plenty of space
        mock_usage = Mock()
        mock_usage.total = 1024 ** 4  # 1TB
        mock_usage.used = 1024 ** 4 // 2  # 50% used
        mock_usage.free = 1024 ** 4 // 2  # 50% free
        
        mock_disk_usage.return_value = mock_usage
        
        result = self.monitor._check_disk_space()
        assert result == True
    
    def test_get_system_health(self):
        """Test system health snapshot generation."""
        health = self.monitor.get_system_health()
        
        assert isinstance(health, SystemHealth)
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert health.timestamp > 0
        assert health.uptime >= 0
        assert isinstance(health.metrics, dict)
        assert isinstance(health.failing_checks, list)
        assert isinstance(health.warnings, list)
    
    def test_health_check_execution(self):
        """Test health check execution with timeout and retries."""
        # Mock slow check that times out
        def slow_check():
            time.sleep(2.0)  # Longer than typical timeout
            return True
        
        health_check = HealthCheck(
            name="slow_check",
            check_function=slow_check,
            description="Slow check",
            timeout=0.1,  # Very short timeout
            retries=1
        )
        
        result = self.monitor._run_health_check(health_check)
        assert result == False  # Should fail due to timeout
    
    def test_failing_health_check(self):
        """Test health check that fails."""
        def failing_check():
            raise Exception("Check failed")
        
        health_check = HealthCheck(
            name="failing_check",
            check_function=failing_check,
            description="Failing check",
            retries=2
        )
        
        result = self.monitor._run_health_check(health_check)
        assert result == False  # Should fail after retries
    
    def test_metric_history_tracking(self):
        """Test metric history tracking."""
        # Get initial metric count
        initial_count = len(self.monitor.current_metrics)
        
        # Trigger a health check that records metrics
        self.monitor._check_memory_usage()
        
        # Should have recorded metrics
        assert len(self.monitor.current_metrics) > initial_count
    
    def test_health_summary(self):
        """Test health summary generation."""
        summary = self.monitor.get_health_summary()
        
        assert "overall_status" in summary
        assert "uptime_hours" in summary
        assert "failing_checks" in summary
        assert "warnings" in summary
        assert "metrics" in summary
        assert "monitoring_active" in summary
        assert summary["monitoring_active"] == self.monitor.is_running
    
    def test_alert_queue_management(self):
        """Test alert queue management."""
        # Create mock system health with warnings
        health = SystemHealth(
            status=HealthStatus.WARNING,
            timestamp=time.time(),
            metrics={},
            failing_checks=[],
            warnings=["Test warning"],
            uptime=3600.0
        )
        
        # Send alerts
        self.monitor._send_alerts(health)
        
        # Check alert queue
        assert not self.monitor.alert_queue.empty()
    
    def test_restart_monitoring(self):
        """Test monitoring restart functionality."""
        self.monitor.start_monitoring()
        assert self.monitor.is_running
        
        restart_time_before = self.monitor.last_restart
        
        self.monitor.restart_monitoring()
        
        assert self.monitor.is_running
        assert self.monitor.last_restart != restart_time_before


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(
            metric_retention=1000,
            alert_retention=100
        )
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(
            metric_retention=5000,
            alert_retention=500,
            trend_analysis_window=200
        )
        
        assert monitor.metric_retention == 5000
        assert monitor.alert_retention == 500
        assert monitor.trend_analysis_window == 200
        assert len(monitor.thresholds) > 0  # Should have default thresholds
        assert len(monitor.metrics) == 0
    
    def test_add_threshold(self):
        """Test adding performance thresholds."""
        threshold = PerformanceThreshold(
            metric_name="custom_metric",
            warning_threshold=10.0,
            critical_threshold=20.0,
            comparison="greater_than"
        )
        
        self.monitor.add_threshold(threshold)
        
        assert "custom_metric" in self.monitor.thresholds
        assert self.monitor.thresholds["custom_metric"].warning_threshold == 10.0
    
    def test_add_alert_callback(self):
        """Test adding alert callbacks."""
        callback = Mock()
        
        self.monitor.add_alert_callback(callback)
        assert callback in self.monitor.alert_callbacks
    
    def test_start_end_operation(self):
        """Test operation timing."""
        operation_id = "test_op_123"
        operation_type = "test_operation"
        
        # Start operation
        self.monitor.start_operation(operation_id, operation_type)
        assert operation_id in self.monitor.operation_start_times
        
        time.sleep(0.01)  # Brief delay
        
        # End operation
        duration = self.monitor.end_operation(operation_id, operation_type, success=True)
        
        assert duration is not None
        assert duration > 0
        assert operation_id not in self.monitor.operation_start_times
        
        # Check statistics update
        stats = self.monitor.performance_stats[operation_type]
        assert stats['count'] == 1
        assert stats['total_time'] > 0
        assert stats['error_count'] == 0
    
    def test_failed_operation_tracking(self):
        """Test failed operation tracking."""
        operation_id = "failed_op"
        operation_type = "failing_operation"
        
        self.monitor.start_operation(operation_id, operation_type)
        duration = self.monitor.end_operation(operation_id, operation_type, success=False)
        
        assert duration is not None
        
        # Check error tracking
        stats = self.monitor.performance_stats[operation_type]
        assert stats['error_count'] == 1
        assert stats['count'] == 1
    
    def test_record_metric(self):
        """Test metric recording."""
        metric = PerformanceMetric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.LATENCY,
            timestamp=time.time(),
            unit="seconds",
            tags={"component": "test"}
        )
        
        self.monitor.record_metric(metric)
        
        assert "test_metric" in self.monitor.metrics
        assert len(self.monitor.metrics["test_metric"]) == 1
        assert "test_metric" in self.monitor.metric_metadata
    
    def test_record_quantum_metrics(self):
        """Test quantum metrics recording."""
        quantum_metrics = {
            "coherence": 0.85,
            "entanglement": 0.67,
            "fidelity": 0.92
        }
        
        self.monitor.record_quantum_metrics(quantum_metrics)
        
        # Check that quantum metrics were recorded
        assert "quantum_coherence" in self.monitor.metrics
        assert "quantum_entanglement" in self.monitor.metrics
        assert "quantum_fidelity" in self.monitor.metrics
        
        for name in ["quantum_coherence", "quantum_entanglement", "quantum_fidelity"]:
            assert len(self.monitor.metrics[name]) == 1
    
    def test_threshold_violation_detection(self):
        """Test threshold violation detection and alerting."""
        # Add callback to capture alerts
        alert_callback = Mock()
        self.monitor.add_alert_callback(alert_callback)
        
        # Create metric that violates threshold
        violation_metric = PerformanceMetric(
            name="planning_latency",
            value=45.0,  # Above critical threshold (30.0)
            metric_type=MetricType.LATENCY,
            timestamp=time.time()
        )
        
        # Record multiple times to trigger threshold check
        for _ in range(10):  # Above min_samples threshold
            self.monitor.record_metric(violation_metric)
        
        # Should have triggered alert
        assert len(self.monitor.alert_history) > 0
        alert_callback.assert_called()
        
        # Check alert properties
        alert = self.monitor.alert_history[-1]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.metric_name == "planning_latency"
        assert alert.current_value >= 45.0
    
    def test_trend_analysis(self):
        """Test performance trend analysis."""
        # Generate trend data
        base_time = time.time()
        for i in range(20):
            metric = PerformanceMetric(
                name="trend_metric",
                value=i * 0.5,  # Increasing trend
                metric_type=MetricType.LATENCY,
                timestamp=base_time + i
            )
            self.monitor.record_metric(metric)
        
        # Update trends
        self.monitor._update_trends()
        
        # Check trend analysis
        assert "trend_metric" in self.monitor.trend_cache
        trend = self.monitor.trend_cache["trend_metric"]
        assert trend.direction == "increasing"
        assert trend.rate_of_change > 0
        assert 0 <= trend.confidence <= 1
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some performance data
        for i in range(5):
            self.monitor.start_operation(f"op_{i}", "test_operation")
            time.sleep(0.01)
            self.monitor.end_operation(f"op_{i}", "test_operation", success=True)
        
        # Record some metrics
        self.monitor.record_quantum_metrics({"coherence": 0.8, "fidelity": 0.9})
        
        summary = self.monitor.get_performance_summary()
        
        assert "timestamp" in summary
        assert "operation_stats" in summary
        assert "metric_summaries" in summary
        assert "trends" in summary
        assert "recommendations" in summary
        
        # Check operation stats
        assert "test_operation" in summary["operation_stats"]
        op_stats = summary["operation_stats"]["test_operation"]
        assert op_stats["total_operations"] == 5
        assert op_stats["success_rate"] == 1.0
    
    def test_recommendations_generation(self):
        """Test performance recommendations generation."""
        # Create performance data that should trigger recommendations
        summary = {
            "operation_stats": {
                "slow_operation": {
                    "avg_duration": 35.0,  # High latency
                    "success_rate": 0.7    # Low success rate
                }
            },
            "trends": {
                "error_rate": {
                    "direction": "increasing"
                }
            }
        }
        
        recommendations = self.monitor._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("slow_operation" in rec for rec in recommendations)
        assert any("success rate" in rec for rec in recommendations)
        assert any("error_rate" in rec for rec in recommendations)
    
    def test_metric_data_retrieval(self):
        """Test metric data retrieval."""
        # Add metric data
        base_time = time.time()
        for i in range(10):
            metric = PerformanceMetric(
                name="data_metric",
                value=i,
                metric_type=MetricType.THROUGHPUT,
                timestamp=base_time + i
            )
            self.monitor.record_metric(metric)
        
        # Get all data
        all_data = self.monitor.get_metric_data("data_metric")
        assert len(all_data) == 10
        
        # Get recent data
        recent_data = self.monitor.get_metric_data("data_metric", time_window=5.0)
        assert len(recent_data) <= 10
        assert all(timestamp >= base_time + 5 for timestamp, _ in recent_data)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        import tempfile
        import os
        import json
        
        # Add some data
        self.monitor.record_metric(PerformanceMetric(
            "export_test", 123.0, MetricType.LATENCY, time.time()
        ))
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.monitor.export_metrics(filepath)
            
            # Verify file was created and contains data
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "timestamp" in data
            assert "raw_metrics" in data
            assert "export_test" in data["raw_metrics"]
            
        finally:
            os.unlink(filepath)


class TestQuantumMetricsMonitor:
    """Test suite for QuantumMetricsMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = QuantumMetricsMonitor(
            coherence_threshold=0.3,
            entanglement_threshold=0.5,
            max_history=100
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'monitor') and self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
    
    def test_initialization(self):
        """Test quantum metrics monitor initialization."""
        monitor = QuantumMetricsMonitor(
            coherence_threshold=0.4,
            entanglement_threshold=0.6,
            max_history=500
        )
        
        assert monitor.coherence_threshold == 0.4
        assert monitor.entanglement_threshold == 0.6
        assert monitor.max_history == 500
        assert not monitor.is_monitoring
        assert len(monitor.quantum_states) == 0
        assert len(monitor.circuit_metrics) == 0
    
    def test_add_callbacks(self):
        """Test adding monitoring callbacks."""
        coherence_callback = Mock()
        decoherence_callback = Mock()
        
        self.monitor.add_coherence_alert(coherence_callback)
        self.monitor.add_decoherence_alert(decoherence_callback)
        
        assert coherence_callback in self.monitor.coherence_alerts
        assert decoherence_callback in self.monitor.decoherence_alerts
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping quantum monitoring."""
        assert not self.monitor.is_monitoring
        
        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring
        assert self.monitor.monitor_thread is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_monitoring
    
    def test_record_quantum_state(self):
        """Test quantum state recording."""
        # Create test quantum state vector
        state_vector = np.array([0.6, 0.8]) / np.sqrt(0.36 + 0.64)  # Normalized
        metadata = {"experiment_id": "test_123"}
        
        self.monitor.record_quantum_state(state_vector, metadata)
        
        # Check that state was recorded
        assert len(self.monitor.quantum_states) == 1
        assert self.monitor.current_quantum_state is not None
        
        quantum_state = self.monitor.quantum_states[0]
        assert np.array_equal(quantum_state.state_vector, state_vector)
        assert quantum_state.coherence >= 0
        assert quantum_state.entanglement_measure >= 0
    
    def test_record_circuit_execution(self):
        """Test quantum circuit execution recording."""
        circuit_metrics = QuantumCircuitMetrics(
            circuit_depth=5,
            gate_count=12,
            two_qubit_gate_count=3,
            execution_time=0.15,
            fidelity=0.94,
            success_probability=0.87,
            error_rate=0.03
        )
        
        self.monitor.record_quantum_circuit_execution("QAOA", circuit_metrics)
        
        # Check recording
        assert len(self.monitor.circuit_metrics) == 1
        assert "QAOA" in self.monitor.algorithm_performance
        
        algo_stats = self.monitor.algorithm_performance["QAOA"]
        assert algo_stats["executions"] == 1
        assert algo_stats["avg_fidelity"] == 0.94
    
    def test_coherence_calculation(self):
        """Test quantum coherence calculation."""
        # Maximum coherence (uniform superposition)
        uniform_state = np.ones(4) / 2.0
        coherence = self.monitor._calculate_coherence(uniform_state)
        assert coherence > 0.9  # Should be high
        
        # Minimum coherence (single state)
        single_state = np.array([1.0, 0.0, 0.0, 0.0])
        coherence = self.monitor._calculate_coherence(single_state)
        assert coherence < 0.1  # Should be low
        
        # Empty state
        empty_state = np.array([])
        coherence = self.monitor._calculate_coherence(empty_state)
        assert coherence == 0.0
    
    def test_entanglement_measurement(self):
        """Test entanglement measurement calculation."""
        # Single qubit (no entanglement)
        single_qubit = np.array([0.6, 0.8])
        entanglement = self.monitor._calculate_entanglement_measure(single_qubit)
        assert entanglement == 0.0
        
        # Two qubit product state (separable)
        product_state = np.array([0.6, 0.0, 0.0, 0.8])  # |0⟩⊗|0⟩ + |1⟩⊗|1⟩ (unnormalized)
        product_state = product_state / np.linalg.norm(product_state)
        entanglement = self.monitor._calculate_entanglement_measure(product_state)
        assert entanglement >= 0.0
        
        # Bell state (maximally entangled)
        bell_state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        entanglement = self.monitor._calculate_entanglement_measure(bell_state)
        assert entanglement > 0.0
    
    def test_concurrence_calculation(self):
        """Test concurrence calculation for 2-qubit states."""
        # Bell state (maximally entangled)
        bell_state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        concurrence = self.monitor._calculate_concurrence(bell_state)
        assert 0.8 <= concurrence <= 1.0  # Should be high for Bell state
        
        # Product state (separable)
        product_state = np.array([1.0, 0.0, 0.0, 0.0])  # |00⟩
        concurrence = self.monitor._calculate_concurrence(product_state)
        assert concurrence == 0.0  # No entanglement
    
    def test_phase_extraction(self):
        """Test quantum phase information extraction."""
        # State with phase information
        state_vector = np.array([0.5, 0.5j, 0.5, 0.5*np.exp(1j*np.pi/4)])
        phase_info = self.monitor._extract_phase_info(state_vector)
        
        assert len(phase_info) > 0
        assert all(isinstance(amp, complex) for amp in phase_info.values())
    
    def test_measurement_probabilities(self):
        """Test measurement probabilities calculation."""
        # Uniform superposition
        state_vector = np.ones(4) / 2.0
        probs = self.monitor._calculate_measurement_probabilities(state_vector)
        
        assert len(probs) == 4
        assert all(prob == 0.25 for prob in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)
    
    def test_superposition_strength(self):
        """Test superposition strength calculation."""
        # Maximum superposition (uniform)
        uniform_state = np.ones(4) / 2.0
        strength = self.monitor._calculate_superposition_strength(uniform_state)
        assert strength > 0.9
        
        # Minimum superposition (single state)
        single_state = np.array([1.0, 0.0, 0.0, 0.0])
        strength = self.monitor._calculate_superposition_strength(single_state)
        assert strength < 0.1
    
    def test_quantum_volume_estimation(self):
        """Test quantum volume estimation."""
        # 2-qubit state
        two_qubit_state = np.ones(4) / 2.0
        volume = self.monitor._estimate_quantum_volume(two_qubit_state)
        assert volume > 0
        assert volume <= 4  # Should be reasonable for 2 qubits
        
        # Single qubit
        single_qubit = np.array([0.6, 0.8])
        volume = self.monitor._estimate_quantum_volume(single_qubit)
        assert 0 <= volume <= 2  # Should be reasonable for 1 qubit
    
    def test_decoherence_detection(self):
        """Test decoherence event detection."""
        # Record states with decreasing coherence
        coherence_values = [0.9, 0.7, 0.5, 0.3, 0.1]  # Decreasing
        
        for i, coherence in enumerate(coherence_values):
            # Create state with specific coherence
            if coherence > 0.5:
                state_vector = np.ones(4) / 2.0 * coherence + np.array([1,0,0,0]) * (1-coherence)
            else:
                state_vector = np.array([1,0,0,0])  # Collapsed state
            
            state_vector = state_vector / np.linalg.norm(state_vector)
            
            self.monitor.record_quantum_state(state_vector)
            time.sleep(0.01)  # Small delay to create time progression
        
        # Trigger decoherence detection
        self.monitor._detect_decoherence_events()
        
        # Should detect decoherence trend
        assert len(self.monitor.decoherence_events) > 0
    
    def test_coherence_alert_triggering(self):
        """Test coherence alert triggering."""
        # Add alert callback
        alert_callback = Mock()
        self.monitor.add_coherence_alert(alert_callback)
        
        # Record state with low coherence
        low_coherence_state = np.array([1.0, 0.0, 0.0, 0.0])  # Single state
        self.monitor.record_quantum_state(low_coherence_state)
        
        # Should trigger coherence alert
        alert_callback.assert_called()
        args = alert_callback.call_args[0]
        assert args[0] < self.monitor.coherence_threshold  # Coherence value
        assert args[1] == self.monitor.coherence_threshold  # Threshold
    
    def test_quantum_metrics_summary(self):
        """Test quantum metrics summary generation."""
        # Record some quantum data
        state_vector = np.ones(4) / 2.0
        self.monitor.record_quantum_state(state_vector)
        
        circuit_metrics = QuantumCircuitMetrics(
            circuit_depth=3, gate_count=8, two_qubit_gate_count=2,
            execution_time=0.1, fidelity=0.9, success_probability=0.85, error_rate=0.05
        )
        self.monitor.record_quantum_circuit_execution("test_algorithm", circuit_metrics)
        
        summary = self.monitor.get_quantum_metrics_summary()
        
        assert "timestamp" in summary
        assert "current_state" in summary
        assert "algorithm_performance" in summary
        assert "decoherence_events" in summary
        assert "monitoring_active" in summary
        
        # Check current state info
        current_state = summary["current_state"]
        assert "coherence" in current_state
        assert "entanglement" in current_state
        assert "n_qubits" in current_state
    
    def test_quantum_data_export(self):
        """Test quantum data export."""
        import tempfile
        import os
        import json
        
        # Record some data
        state_vector = np.array([0.6, 0.8]) / np.sqrt(0.36 + 0.64)
        self.monitor.record_quantum_state(state_vector)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.monitor.export_quantum_data(filepath)
            
            # Verify export
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "quantum_metrics_summary" in data
            assert "quantum_states" in data
            assert "decoherence_events" in data
            
        finally:
            os.unlink(filepath)
    
    def test_algorithm_performance_tracking(self):
        """Test algorithm performance tracking over multiple executions."""
        algorithm_name = "VQE"
        
        # Record multiple circuit executions
        fidelities = [0.85, 0.90, 0.88, 0.92, 0.87]
        execution_times = [0.1, 0.12, 0.09, 0.11, 0.13]
        
        for fidelity, exec_time in zip(fidelities, execution_times):
            circuit_metrics = QuantumCircuitMetrics(
                circuit_depth=4, gate_count=10, two_qubit_gate_count=3,
                execution_time=exec_time, fidelity=fidelity,
                success_probability=fidelity, error_rate=1.0-fidelity
            )
            self.monitor.record_quantum_circuit_execution(algorithm_name, circuit_metrics)
        
        # Check algorithm performance tracking
        algo_stats = self.monitor.algorithm_performance[algorithm_name]
        assert algo_stats["executions"] == 5
        assert abs(algo_stats["avg_fidelity"] - np.mean(fidelities)) < 1e-10
        assert abs(algo_stats["avg_execution_time"] - np.mean(execution_times)) < 1e-10


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_health_performance_integration(self):
        """Test integration between health and performance monitoring."""
        health_monitor = HealthMonitor(check_interval=0.1)
        performance_monitor = PerformanceMonitor()
        
        # Add performance alert callback to health monitor
        def performance_health_callback(system_health):
            if system_health.status == HealthStatus.CRITICAL:
                # Record performance alert
                alert = PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    metric_name="system_health",
                    current_value=0.0,  # Critical = 0
                    threshold_value=0.5,
                    message="System health critical",
                    timestamp=time.time()
                )
                performance_monitor.alert_history.append(alert)
        
        health_monitor.add_alert_callback(performance_health_callback)
        
        # Start monitoring
        health_monitor.start_monitoring()
        
        try:
            # Wait briefly for monitoring
            time.sleep(0.2)
            
            # Get health summary
            health_summary = health_monitor.get_health_summary()
            performance_summary = performance_monitor.get_performance_summary()
            
            assert "overall_status" in health_summary
            assert "timestamp" in performance_summary
            
        finally:
            health_monitor.stop_monitoring()
    
    def test_quantum_performance_integration(self):
        """Test integration between quantum and performance monitoring."""
        quantum_monitor = QuantumMetricsMonitor()
        performance_monitor = PerformanceMonitor()
        
        # Record quantum data that should impact performance metrics
        state_vector = np.ones(4) / 2.0
        quantum_monitor.record_quantum_state(state_vector)
        
        # Quantum monitor should record performance metrics
        quantum_summary = quantum_monitor.get_quantum_metrics_summary()
        
        # Check that quantum metrics exist
        assert "current_state" in quantum_summary
        if "coherence" in quantum_summary["current_state"]:
            coherence = quantum_summary["current_state"]["coherence"]
            
            # Record in performance monitor
            performance_monitor.record_metric(PerformanceMonitor.PerformanceMetric(
                "quantum_coherence",
                coherence,
                MetricType.QUANTUM_METRIC,
                time.time()
            ))
            
            # Verify integration
            assert "quantum_coherence" in performance_monitor.metrics
    
    def test_monitoring_stack_alerts(self):
        """Test alert propagation through monitoring stack."""
        # Create monitoring stack
        health_monitor = HealthMonitor()
        performance_monitor = PerformanceMonitor()
        quantum_monitor = QuantumMetricsMonitor()
        
        # Collect alerts from all monitors
        all_alerts = []
        
        def collect_alert(alert):
            all_alerts.append(("performance", alert))
        
        def collect_health_alert(health):
            if health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                all_alerts.append(("health", health))
        
        def collect_coherence_alert(coherence, threshold):
            all_alerts.append(("quantum_coherence", (coherence, threshold)))
        
        # Add callbacks
        performance_monitor.add_alert_callback(collect_alert)
        health_monitor.add_alert_callback(collect_health_alert)
        quantum_monitor.add_coherence_alert(collect_coherence_alert)
        
        # Trigger alerts
        # 1. Performance alert
        high_latency_metric = PerformanceMonitor.PerformanceMetric(
            "test_latency", 100.0, MetricType.LATENCY, time.time()
        )
        for _ in range(10):  # Trigger threshold
            performance_monitor.record_metric(high_latency_metric)
        
        # 2. Quantum alert
        low_coherence_state = np.array([1.0, 0.0, 0.0, 0.0])
        quantum_monitor.record_quantum_state(low_coherence_state)
        
        # Check that alerts were collected
        assert len(all_alerts) > 0
        alert_types = [alert[0] for alert in all_alerts]
        assert "quantum_coherence" in alert_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])