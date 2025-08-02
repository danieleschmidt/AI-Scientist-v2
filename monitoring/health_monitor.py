#!/usr/bin/env python3
"""
Comprehensive health monitoring system for AI Scientist v2.
Monitors system health, performance metrics, and sends alerts.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import requests
import psutil


class HealthMonitor:
    """Comprehensive health monitoring for AI Scientist v2."""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path("/app/monitoring/config.json")
        self.root_dir = Path.cwd()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Health status
        self.health_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "metrics": {},
            "alerts": [],
            "services": {}
        }
    
    def _load_config(self) -> Dict:
        """Load monitoring configuration."""
        default_config = {
            "monitoring": {
                "interval_seconds": 60,
                "retention_days": 30,
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_percent": 90,
                    "response_time_ms": 5000,
                    "error_rate_percent": 5
                }
            },
            "alerts": {
                "enabled": True,
                "webhook_url": "",
                "email_recipients": [],
                "severity_levels": ["critical", "warning", "info"]
            },
            "services": {
                "api_endpoint": "http://localhost:8000/health",
                "database_url": "",
                "external_apis": []
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    return {**default_config, **loaded_config}
            except json.JSONDecodeError:
                self.logger.warning("Invalid config file, using defaults")
        
        return default_config
    
    def collect_system_metrics(self) -> Dict:
        """Collect system performance metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            metrics["cpu"] = {
                "percent": cpu_percent,
                "count": cpu_count,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["memory"] = {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "percent": memory.percent,
                "free_bytes": memory.free
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics["disk"] = {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics["network"] = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process metrics
            current_process = psutil.Process()
            metrics["process"] = {
                "cpu_percent": current_process.cpu_percent(),
                "memory_mb": current_process.memory_info().rss / 1024 / 1024,
                "num_threads": current_process.num_threads(),
                "open_files": len(current_process.open_files()),
                "connections": len(current_process.connections())
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def collect_application_metrics(self) -> Dict:
        """Collect application-specific metrics."""
        metrics = {}
        
        try:
            # AI Scientist specific metrics
            metrics["ai_scientist"] = self._collect_ai_scientist_metrics()
            
            # Experiment metrics
            metrics["experiments"] = self._collect_experiment_metrics()
            
            # Model metrics
            metrics["models"] = self._collect_model_metrics()
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _collect_ai_scientist_metrics(self) -> Dict:
        """Collect AI Scientist specific metrics."""
        metrics = {
            "status": "unknown",
            "experiments_running": 0,
            "papers_generated": 0,
            "total_experiments": 0
        }
        
        try:
            # Check experiments directory
            experiments_dir = Path("/app/experiments")
            if experiments_dir.exists():
                experiments = list(experiments_dir.iterdir())
                metrics["total_experiments"] = len(experiments)
                
                # Count running experiments (simplified check)
                running_experiments = [
                    exp for exp in experiments
                    if (exp / "status.json").exists()
                ]
                metrics["experiments_running"] = len(running_experiments)
            
            # Check for generated papers
            papers_dir = Path("/app/final_papers")
            if papers_dir.exists():
                papers = list(papers_dir.glob("*.pdf"))
                metrics["papers_generated"] = len(papers)
            
            metrics["status"] = "healthy"
            
        except Exception as e:
            metrics["error"] = str(e)
            metrics["status"] = "error"
        
        return metrics
    
    def _collect_experiment_metrics(self) -> Dict:
        """Collect experiment execution metrics."""
        metrics = {
            "success_rate": 0,
            "average_duration_minutes": 0,
            "recent_experiments": 0
        }
        
        try:
            experiments_dir = Path("/app/experiments")
            if not experiments_dir.exists():
                return metrics
            
            recent_experiments = []
            successful_experiments = 0
            total_duration = 0
            
            # Check experiments from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for exp_dir in experiments_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                try:
                    # Check modification time
                    mod_time = datetime.fromtimestamp(exp_dir.stat().st_mtime)
                    if mod_time > cutoff_time:
                        recent_experiments.append(exp_dir)
                        
                        # Check if experiment was successful
                        status_file = exp_dir / "status.json"
                        if status_file.exists():
                            with open(status_file, 'r') as f:
                                status = json.load(f)
                                if status.get("status") == "completed":
                                    successful_experiments += 1
                                
                                # Add duration if available
                                duration = status.get("duration_minutes", 0)
                                total_duration += duration
                
                except Exception:
                    continue
            
            metrics["recent_experiments"] = len(recent_experiments)
            
            if recent_experiments:
                metrics["success_rate"] = (successful_experiments / len(recent_experiments)) * 100
                metrics["average_duration_minutes"] = total_duration / len(recent_experiments)
            
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def _collect_model_metrics(self) -> Dict:
        """Collect ML model performance metrics."""
        metrics = {
            "gpu_available": False,
            "gpu_utilization": 0,
            "gpu_memory_used": 0,
            "models_loaded": 0
        }
        
        try:
            # Check GPU availability
            import torch
            if torch.cuda.is_available():
                metrics["gpu_available"] = True
                metrics["gpu_count"] = torch.cuda.device_count()
                
                # Get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    metrics["gpu_memory_total"] = info.total
                    metrics["gpu_memory_used"] = info.used
                    metrics["gpu_memory_percent"] = (info.used / info.total) * 100
                    metrics["gpu_utilization"] = utilization.gpu
                    
                except ImportError:
                    # pynvml not available
                    pass
            
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def check_service_health(self) -> Dict:
        """Check health of configured services."""
        services = {}
        
        # Check API endpoint
        api_endpoint = self.config["services"]["api_endpoint"]
        if api_endpoint:
            services["api"] = self._check_http_service(api_endpoint)
        
        # Check external APIs
        for api_name, api_url in self.config["services"].get("external_apis", {}).items():
            services[api_name] = self._check_http_service(api_url)
        
        return services
    
    def _check_http_service(self, url: str, timeout: int = 10) -> Dict:
        """Check HTTP service health."""
        service_health = {
            "status": "unknown",
            "response_time_ms": 0,
            "status_code": 0
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            end_time = time.time()
            
            service_health["status_code"] = response.status_code
            service_health["response_time_ms"] = round((end_time - start_time) * 1000, 2)
            
            if 200 <= response.status_code < 300:
                service_health["status"] = "healthy"
            elif 400 <= response.status_code < 500:
                service_health["status"] = "client_error"
            else:
                service_health["status"] = "server_error"
                
        except requests.exceptions.Timeout:
            service_health["status"] = "timeout"
        except requests.exceptions.ConnectionError:
            service_health["status"] = "connection_error"
        except Exception as e:
            service_health["status"] = "error"
            service_health["error"] = str(e)
        
        return service_health
    
    def evaluate_alerts(self) -> List[Dict]:
        """Evaluate metrics against thresholds and generate alerts."""
        alerts = []
        thresholds = self.config["monitoring"]["alert_thresholds"]
        
        # System metric alerts
        system_metrics = self.health_data["metrics"].get("system", {})
        
        # CPU alert
        cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > thresholds["cpu_percent"]:
            alerts.append({
                "severity": "warning" if cpu_percent < 90 else "critical",
                "message": f"High CPU usage: {cpu_percent}%",
                "metric": "cpu_percent",
                "value": cpu_percent,
                "threshold": thresholds["cpu_percent"]
            })
        
        # Memory alert
        memory_percent = system_metrics.get("memory", {}).get("percent", 0)
        if memory_percent > thresholds["memory_percent"]:
            alerts.append({
                "severity": "warning" if memory_percent < 95 else "critical",
                "message": f"High memory usage: {memory_percent}%",
                "metric": "memory_percent",
                "value": memory_percent,
                "threshold": thresholds["memory_percent"]
            })
        
        # Disk alert
        disk_percent = system_metrics.get("disk", {}).get("percent", 0)
        if disk_percent > thresholds["disk_percent"]:
            alerts.append({
                "severity": "warning" if disk_percent < 95 else "critical",
                "message": f"High disk usage: {disk_percent}%",
                "metric": "disk_percent",
                "value": disk_percent,
                "threshold": thresholds["disk_percent"]
            })
        
        # Service response time alerts
        for service_name, service_data in self.health_data["services"].items():
            response_time = service_data.get("response_time_ms", 0)
            if response_time > thresholds["response_time_ms"]:
                alerts.append({
                    "severity": "warning",
                    "message": f"Slow response from {service_name}: {response_time}ms",
                    "metric": "response_time_ms",
                    "value": response_time,
                    "threshold": thresholds["response_time_ms"],
                    "service": service_name
                })
        
        return alerts
    
    def send_alerts(self, alerts: List[Dict]):
        """Send alerts via configured channels."""
        if not self.config["alerts"]["enabled"] or not alerts:
            return
        
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert['message']}")
            
            # Send webhook notification
            webhook_url = self.config["alerts"]["webhook_url"]
            if webhook_url:
                self._send_webhook_alert(webhook_url, alert)
    
    def _send_webhook_alert(self, webhook_url: str, alert: Dict):
        """Send alert via webhook."""
        try:
            payload = {
                "text": f"🚨 AI Scientist Alert: {alert['message']}",
                "alert": alert,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def run_health_check(self) -> Dict:
        """Run comprehensive health check."""
        self.logger.info("Running health check...")
        
        # Collect all metrics
        self.health_data["metrics"]["system"] = self.collect_system_metrics()
        self.health_data["metrics"]["application"] = self.collect_application_metrics()
        self.health_data["services"] = self.check_service_health()
        
        # Evaluate alerts
        self.health_data["alerts"] = self.evaluate_alerts()
        
        # Determine overall health status
        critical_alerts = [a for a in self.health_data["alerts"] if a["severity"] == "critical"]
        warning_alerts = [a for a in self.health_data["alerts"] if a["severity"] == "warning"]
        
        if critical_alerts:
            self.health_data["status"] = "critical"
        elif warning_alerts:
            self.health_data["status"] = "warning"
        else:
            self.health_data["status"] = "healthy"
        
        # Send alerts
        self.send_alerts(self.health_data["alerts"])
        
        return self.health_data
    
    def save_metrics(self, output_file: Path = None):
        """Save metrics to file."""
        if not output_file:
            output_file = Path("/app/monitoring/health_data.json")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.health_data, f, indent=2)
    
    def start_monitoring(self, interval_seconds: int = None):
        """Start continuous monitoring."""
        interval = interval_seconds or self.config["monitoring"]["interval_seconds"]
        
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                # Run health check
                health_data = self.run_health_check()
                
                # Save metrics
                self.save_metrics()
                
                # Log status
                status = health_data["status"]
                alert_count = len(health_data["alerts"])
                self.logger.info(f"Health status: {status}, Alerts: {alert_count}")
                
                # Wait for next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise


def main():
    """Main entry point for health monitoring."""
    parser = argparse.ArgumentParser(description="AI Scientist Health Monitor")
    
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--output", help="Output file for health data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize monitor
    config_file = Path(args.config) if args.config else None
    monitor = HealthMonitor(config_file)
    
    try:
        if args.once:
            # Run single health check
            health_data = monitor.run_health_check()
            
            # Save results
            if args.output:
                monitor.save_metrics(Path(args.output))
            else:
                monitor.save_metrics()
            
            # Print summary
            print(f"Health Status: {health_data['status']}")
            print(f"Alerts: {len(health_data['alerts'])}")
            
            for alert in health_data['alerts']:
                severity_emoji = "🚨" if alert['severity'] == 'critical' else "⚠️"
                print(f"{severity_emoji} {alert['message']}")
            
            # Exit with appropriate code
            if health_data['status'] == 'critical':
                sys.exit(2)
            elif health_data['status'] == 'warning':
                sys.exit(1)
            else:
                sys.exit(0)
        else:
            # Start continuous monitoring
            monitor.start_monitoring(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()