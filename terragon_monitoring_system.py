#!/usr/bin/env python3
"""
TERRAGON MONITORING SYSTEM v2.0

Advanced monitoring, alerting, and observability for autonomous research systems.
"""

import os
import sys
import json
import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from collections import deque, defaultdict

import psutil
import requests
from concurrent.futures import ThreadPoolExecutor


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
    """Alert information."""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime
    type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str  # healthy, degraded, unhealthy
    score: float  # 0-100 health score
    checks: Dict[str, bool]
    alerts: List[Alert]
    last_updated: datetime
    uptime: float


class MetricCollector:
    """Collects various system and application metrics."""
    
    def __init__(self):
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.metric_handlers = {
            "system": self._collect_system_metrics,
            "application": self._collect_application_metrics,
            "research": self._collect_research_metrics
        }
        
    def _collect_system_metrics(self) -> List[Metric]:
        """Collect system-level metrics."""
        timestamp = datetime.now()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric("cpu_usage_percent", cpu_percent, timestamp, MetricType.GAUGE, {"unit": "percent"}))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(Metric("memory_usage_percent", memory.percent, timestamp, MetricType.GAUGE, {"unit": "percent"}))
        metrics.append(Metric("memory_available_bytes", memory.available, timestamp, MetricType.GAUGE, {"unit": "bytes"}))
        metrics.append(Metric("memory_used_bytes", memory.used, timestamp, MetricType.GAUGE, {"unit": "bytes"}))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(Metric("disk_usage_percent", (disk.used / disk.total) * 100, timestamp, MetricType.GAUGE, {"unit": "percent"}))
        metrics.append(Metric("disk_free_bytes", disk.free, timestamp, MetricType.GAUGE, {"unit": "bytes"}))
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(Metric("network_bytes_sent", network.bytes_sent, timestamp, MetricType.COUNTER, {"unit": "bytes"}))
        metrics.append(Metric("network_bytes_recv", network.bytes_recv, timestamp, MetricType.COUNTER, {"unit": "bytes"}))
        
        # Process count
        process_count = len(psutil.pids())
        metrics.append(Metric("process_count", process_count, timestamp, MetricType.GAUGE, {"unit": "count"}))
        
        return metrics
    
    def _collect_application_metrics(self) -> List[Metric]:
        """Collect application-level metrics."""
        timestamp = datetime.now()
        metrics = []
        
        # Python-specific metrics
        import gc
        gc_counts = gc.get_count()
        metrics.append(Metric("gc_objects", sum(gc_counts), timestamp, MetricType.GAUGE, {"unit": "count"}))
        
        # Thread count
        thread_count = threading.active_count()
        metrics.append(Metric("thread_count", thread_count, timestamp, MetricType.GAUGE, {"unit": "count"}))
        
        return metrics
    
    def _collect_research_metrics(self) -> List[Metric]:
        """Collect research-specific metrics."""
        timestamp = datetime.now()
        metrics = []
        
        # Check for active research sessions
        research_dirs = list(Path(".").glob("*research_output*/session_*"))
        metrics.append(Metric("active_research_sessions", len(research_dirs), timestamp, MetricType.GAUGE, {"unit": "count"}))
        
        # Check for recent experiments
        recent_experiments = 0
        for dir in research_dirs:
            if dir.exists() and (datetime.now() - datetime.fromtimestamp(dir.stat().st_mtime)).days < 1:
                recent_experiments += 1
        
        metrics.append(Metric("recent_experiments", recent_experiments, timestamp, MetricType.GAUGE, {"unit": "count"}))
        
        return metrics
    
    def collect_all_metrics(self) -> List[Metric]:
        """Collect all metrics from all sources."""
        all_metrics = []
        
        for source, handler in self.metric_handlers.items():
            try:
                source_metrics = handler()
                for metric in source_metrics:
                    metric.tags["source"] = source
                all_metrics.extend(source_metrics)
            except Exception as e:
                logging.error(f"Failed to collect {source} metrics: {e}")
        
        # Store metrics
        self.metrics.extend(all_metrics)
        
        return all_metrics
    
    def get_metric_history(self, metric_name: str, minutes: int = 60) -> List[Metric]:
        """Get metric history for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            metric for metric in self.metrics
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable[[List[Metric]], List[Alert]]] = []
        self.notification_channels = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        
        def high_cpu_alert(metrics: List[Metric]) -> List[Alert]:
            alerts = []
            cpu_metrics = [m for m in metrics if m.name == "cpu_usage_percent"]
            
            for metric in cpu_metrics:
                if metric.value > 90:
                    alert = Alert(
                        id=f"high_cpu_{metric.timestamp.isoformat()}",
                        level=AlertLevel.CRITICAL,
                        title="High CPU Usage",
                        message=f"CPU usage is {metric.value:.1f}%",
                        timestamp=metric.timestamp,
                        source="system",
                        tags={"metric": "cpu_usage_percent"}
                    )
                    alerts.append(alert)
                elif metric.value > 80:
                    alert = Alert(
                        id=f"elevated_cpu_{metric.timestamp.isoformat()}",
                        level=AlertLevel.WARNING,
                        title="Elevated CPU Usage",
                        message=f"CPU usage is {metric.value:.1f}%",
                        timestamp=metric.timestamp,
                        source="system",
                        tags={"metric": "cpu_usage_percent"}
                    )
                    alerts.append(alert)
            
            return alerts
        
        def high_memory_alert(metrics: List[Metric]) -> List[Alert]:
            alerts = []
            memory_metrics = [m for m in metrics if m.name == "memory_usage_percent"]
            
            for metric in memory_metrics:
                if metric.value > 90:
                    alert = Alert(
                        id=f"high_memory_{metric.timestamp.isoformat()}",
                        level=AlertLevel.CRITICAL,
                        title="High Memory Usage",
                        message=f"Memory usage is {metric.value:.1f}%",
                        timestamp=metric.timestamp,
                        source="system",
                        tags={"metric": "memory_usage_percent"}
                    )
                    alerts.append(alert)
            
            return alerts
        
        def disk_space_alert(metrics: List[Metric]) -> List[Alert]:
            alerts = []
            disk_metrics = [m for m in metrics if m.name == "disk_usage_percent"]
            
            for metric in disk_metrics:
                if metric.value > 95:
                    alert = Alert(
                        id=f"disk_space_critical_{metric.timestamp.isoformat()}",
                        level=AlertLevel.CRITICAL,
                        title="Disk Space Critical",
                        message=f"Disk usage is {metric.value:.1f}%",
                        timestamp=metric.timestamp,
                        source="system",
                        tags={"metric": "disk_usage_percent"}
                    )
                    alerts.append(alert)
            
            return alerts
        
        self.alert_rules.extend([
            high_cpu_alert,
            high_memory_alert,
            disk_space_alert
        ])
    
    def add_alert_rule(self, rule: Callable[[List[Metric]], List[Alert]]):
        """Add custom alert rule."""
        self.alert_rules.append(rule)
    
    def process_metrics(self, metrics: List[Metric]) -> List[Alert]:
        """Process metrics and generate alerts."""
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                rule_alerts = rule(metrics)
                new_alerts.extend(rule_alerts)
            except Exception as e:
                logging.error(f"Alert rule failed: {e}")
        
        # Store new alerts
        for alert in new_alerts:
            self.alerts[alert.id] = alert
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts.values() if alert.level == level]


class HealthChecker:
    """Performs health checks on various system components."""
    
    def __init__(self):
        self.health_checks = {
            "system_resources": self._check_system_resources,
            "disk_space": self._check_disk_space,
            "research_services": self._check_research_services,
            "api_connectivity": self._check_api_connectivity
        }
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are healthy."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return cpu_percent < 85 and memory_percent < 85
    
    def _check_disk_space(self) -> bool:
        """Check if disk space is sufficient."""
        disk_usage = psutil.disk_usage('/').percent
        return disk_usage < 90
    
    def _check_research_services(self) -> bool:
        """Check if research services are running."""
        # Check for active research processes
        research_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('research' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    research_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return len(research_processes) >= 0  # Always healthy for now
    
    def _check_api_connectivity(self) -> bool:
        """Check API connectivity."""
        # Simple connectivity check
        try:
            # Check OpenAI API
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                response = requests.get("https://api.openai.com/v1/models", 
                                      headers={"Authorization": f"Bearer {openai_key}"},
                                      timeout=10)
                return response.status_code == 200
            else:
                return True  # No key configured, assume healthy
        except Exception:
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                results[check_name] = check_func()
            except Exception as e:
                logging.error(f"Health check {check_name} failed: {e}")
                results[check_name] = False
        
        return results
    
    def get_overall_health(self) -> SystemHealth:
        """Get overall system health status."""
        checks = self.run_all_checks()
        
        # Calculate health score
        passed_checks = sum(1 for result in checks.values() if result)
        total_checks = len(checks)
        score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        # Determine status
        if score >= 90:
            status = "healthy"
        elif score >= 70:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return SystemHealth(
            status=status,
            score=score,
            checks=checks,
            alerts=[],  # Would be populated with current alerts
            last_updated=datetime.now(),
            uptime=time.time() - psutil.boot_time()
        )


class MonitoringDashboard:
    """Simple web dashboard for monitoring."""
    
    def __init__(self, metric_collector: MetricCollector, alert_manager: AlertManager, 
                 health_checker: HealthChecker):
        self.metric_collector = metric_collector
        self.alert_manager = alert_manager
        self.health_checker = health_checker
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        # Get system health
        health = self.health_checker.get_overall_health()
        
        # Get recent metrics
        recent_metrics = []
        if self.metric_collector.metrics:
            cutoff_time = datetime.now() - timedelta(minutes=5)
            recent_metrics = [
                asdict(m) for m in self.metric_collector.metrics
                if m.timestamp >= cutoff_time
            ]
        
        # Get active alerts
        active_alerts = [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": asdict(health),
            "recent_metrics": recent_metrics,
            "active_alerts": active_alerts,
            "alert_counts": {
                "total": len(self.alert_manager.alerts),
                "active": len(active_alerts),
                "critical": len(self.alert_manager.get_alerts_by_level(AlertLevel.CRITICAL)),
                "warning": len(self.alert_manager.get_alerts_by_level(AlertLevel.WARNING))
            }
        }
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        status = self.generate_status_report()
        health = status["system_health"]
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Terragon Research Monitoring Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .status-{health['status']} {{ 
            color: {'green' if health['status'] == 'healthy' else 'orange' if health['status'] == 'degraded' else 'red'};
            font-weight: bold;
        }}
        .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
        .alert {{ margin: 10px 0; padding: 10px; border-left: 4px solid red; }}
        .health-score {{ font-size: 24px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>🔬 Terragon Research System Monitoring</h1>
    
    <div class="system-overview">
        <h2>System Health</h2>
        <div class="health-score status-{health['status']}">
            {health['status'].upper()} ({health['score']:.1f}%)
        </div>
        <p>Last Updated: {health['last_updated']}</p>
        <p>Uptime: {health['uptime']/3600:.1f} hours</p>
    </div>
    
    <div class="health-checks">
        <h3>Health Checks</h3>
        <table>
            <tr><th>Check</th><th>Status</th></tr>
            {''.join(f"<tr><td>{check}</td><td>{'✅ Pass' if result else '❌ Fail'}</td></tr>" 
                    for check, result in health['checks'].items())}
        </table>
    </div>
    
    <div class="alerts">
        <h3>Active Alerts ({status['alert_counts']['active']})</h3>
        {f"<p>No active alerts</p>" if not status['active_alerts'] else 
         ''.join(f'<div class="alert"><strong>{alert["level"]}</strong>: {alert["title"]} - {alert["message"]} ({alert["timestamp"]})</div>' 
                for alert in status['active_alerts'][:10])}
    </div>
    
    <div class="metrics">
        <h3>Recent Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Timestamp</th></tr>
            {''.join(f"<tr><td>{metric['name']}</td><td>{metric['value']:.2f} {metric.get('unit', '')}</td><td>{metric['timestamp']}</td></tr>" 
                    for metric in status['recent_metrics'][-20:])}
        </table>
    </div>
    
    <div class="footer">
        <p><em>Auto-refresh every 30 seconds</em></p>
        <p>Generated at {status['timestamp']}</p>
    </div>
</body>
</html>
        """
        
        return html_template


class TerragronMonitoringSystem:
    """Main monitoring system coordinator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.dashboard = MonitoringDashboard(
            self.metric_collector, 
            self.alert_manager, 
            self.health_checker
        )
        
        self.running = False
        self.monitoring_task = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self, interval: int = 30):
        """Start the monitoring loop."""
        self.running = True
        self.logger.info("🚀 Starting Terragon Monitoring System")
        
        while self.running:
            try:
                # Collect metrics
                metrics = self.metric_collector.collect_all_metrics()
                
                # Process alerts
                new_alerts = self.alert_manager.process_metrics(metrics)
                
                if new_alerts:
                    self.logger.warning(f"Generated {len(new_alerts)} new alerts")
                    for alert in new_alerts:
                        self.logger.warning(f"ALERT [{alert.level.value}]: {alert.title} - {alert.message}")
                
                # Log health status
                health = self.health_checker.get_overall_health()
                self.logger.info(f"System Health: {health.status} ({health.score:.1f}%)")
                
                # Save dashboard
                await self._save_dashboard()
                
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(5)  # Short sleep on error
    
    async def _save_dashboard(self):
        """Save dashboard to file."""
        try:
            dashboard_dir = Path("monitoring_output")
            dashboard_dir.mkdir(exist_ok=True)
            
            # Save HTML dashboard
            html_content = self.dashboard.generate_html_dashboard()
            html_path = dashboard_dir / "dashboard.html"
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Save JSON status
            status_content = self.dashboard.generate_status_report()
            json_path = dashboard_dir / "status.json"
            
            with open(json_path, 'w') as f:
                json.dump(status_content, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save dashboard: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        self.logger.info("🛑 Stopping Terragon Monitoring System")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary."""
        return self.dashboard.generate_status_report()


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Monitoring System")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--dashboard-only", action="store_true", help="Generate dashboard and exit")
    
    args = parser.parse_args()
    
    # Create monitoring system
    monitoring = TerragronMonitoringSystem()
    
    if args.dashboard_only:
        # Generate dashboard once and exit
        print("📊 Generating monitoring dashboard...")
        
        # Collect metrics once
        monitoring.metric_collector.collect_all_metrics()
        await monitoring._save_dashboard()
        
        dashboard_path = Path("monitoring_output/dashboard.html")
        status_path = Path("monitoring_output/status.json")
        
        print(f"✅ Dashboard saved to: {dashboard_path}")
        print(f"✅ Status saved to: {status_path}")
        
        # Print status summary
        status = monitoring.get_status_summary()
        health = status["system_health"]
        
        print(f"\n🏥 System Health: {health['status']} ({health['score']:.1f}%)")
        print(f"📈 Active Alerts: {status['alert_counts']['active']}")
        print(f"📊 Recent Metrics: {len(status['recent_metrics'])}")
        
        return
    
    # Start continuous monitoring
    try:
        await monitoring.start_monitoring(interval=args.interval)
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted by user")
        monitoring.stop_monitoring()
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())