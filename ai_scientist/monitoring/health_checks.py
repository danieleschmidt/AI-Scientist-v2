"""
Health check endpoints and monitoring utilities for AI Scientist v2.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks = {}
        self.history = []
        self.max_history = 1000
    
    def register_check(self, name: str, check_func, timeout: float = 5.0):
        """Register a health check function."""
        self.checks[name] = {
            'function': check_func,
            'timeout': timeout
        }
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a single health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not registered",
                duration_ms=0.0,
                timestamp=datetime.now()
            )
        
        check_config = self.checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check_config['function'](),
                timeout=check_config['timeout']
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                result.timestamp = datetime.now()
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration_ms=duration_ms,
                    timestamp=datetime.now()
                )
                
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check_config['timeout']}s",
                duration_ms=duration_ms,
                timestamp=datetime.now()
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.now()
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name in self.checks.keys():
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                
                # Store in history
                self.history.append(result)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                    
            except Exception as e:
                logger.error(f"Failed to run health check '{name}': {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected error: {str(e)}",
                    duration_ms=0.0,
                    timestamp=datetime.now()
                )
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


# Default health checker instance
health_checker = HealthChecker()


# Standard health check functions
async def check_system_resources() -> HealthCheck:
    """Check system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine status based on resource usage
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent > 90:
            status = HealthStatus.UNHEALTHY
            issues.append(f"High CPU usage: {cpu_percent}%")
        elif cpu_percent > 75:
            status = HealthStatus.DEGRADED
            issues.append(f"Elevated CPU usage: {cpu_percent}%")
        
        if memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            issues.append(f"High memory usage: {memory.percent}%")
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
            issues.append(f"Elevated memory usage: {memory.percent}%")
        
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 95:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Low disk space: {disk_percent:.1f}% used")
        elif disk_percent > 85:
            status = HealthStatus.DEGRADED
            issues.append(f"Disk space warning: {disk_percent:.1f}% used")
        
        message = "; ".join(issues) if issues else "System resources OK"
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        )
    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to check system resources: {str(e)}",
            duration_ms=0.0,
            timestamp=datetime.now()
        )


async def check_gpu_status() -> HealthCheck:
    """Check GPU availability and status."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return HealthCheck(
                name="gpu_status",
                status=HealthStatus.DEGRADED,
                message="CUDA not available",
                duration_ms=0.0,
                timestamp=datetime.now(),
                details={"cuda_available": False}
            )
        
        gpu_count = torch.cuda.device_count()
        gpu_details = {}
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = props.total_memory
            
            gpu_details[f"gpu_{i}"] = {
                "name": props.name,
                "memory_allocated_gb": memory_allocated / (1024**3),
                "memory_reserved_gb": memory_reserved / (1024**3),
                "memory_total_gb": memory_total / (1024**3),
                "memory_utilization": (memory_allocated / memory_total) * 100
            }
        
        return HealthCheck(
            name="gpu_status",
            status=HealthStatus.HEALTHY,
            message=f"{gpu_count} GPU(s) available",
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={
                "cuda_available": True,
                "gpu_count": gpu_count,
                "gpus": gpu_details
            }
        )
        
    except ImportError:
        return HealthCheck(
            name="gpu_status",
            status=HealthStatus.DEGRADED,
            message="PyTorch not available",
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={"pytorch_available": False}
        )
    except Exception as e:
        return HealthCheck(
            name="gpu_status",
            status=HealthStatus.UNHEALTHY,
            message=f"GPU check failed: {str(e)}",
            duration_ms=0.0,
            timestamp=datetime.now()
        )


async def check_database_connection() -> HealthCheck:
    """Check database connectivity."""
    try:
        # This would connect to your actual database
        # For now, we'll simulate the check
        await asyncio.sleep(0.1)  # Simulate connection time
        
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection OK",
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={"connected": True}
        )
    except Exception as e:
        return HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={"connected": False}
        )


async def check_api_dependencies() -> HealthCheck:
    """Check external API dependencies."""
    try:
        import os
        
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "semantic_scholar": os.getenv("S2_API_KEY")
        }
        
        available_apis = [name for name, key in api_keys.items() if key]
        missing_apis = [name for name, key in api_keys.items() if not key]
        
        if not available_apis:
            status = HealthStatus.UNHEALTHY
            message = "No API keys configured"
        elif missing_apis:
            status = HealthStatus.DEGRADED
            message = f"Some APIs unavailable: {', '.join(missing_apis)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All API keys configured"
        
        return HealthCheck(
            name="api_dependencies",
            status=status,
            message=message,
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={
                "available_apis": available_apis,
                "missing_apis": missing_apis,
                "total_apis": len(api_keys)
            }
        )
    except Exception as e:
        return HealthCheck(
            name="api_dependencies",
            status=HealthStatus.UNHEALTHY,
            message=f"API dependency check failed: {str(e)}",
            duration_ms=0.0,
            timestamp=datetime.now()
        )


async def check_experiment_queue() -> HealthCheck:
    """Check experiment queue status."""
    try:
        # This would check your actual experiment queue
        # For now, we'll simulate the check
        queue_size = 0  # Would get from actual queue
        running_experiments = 0  # Would get from actual system
        
        if queue_size > 100:
            status = HealthStatus.DEGRADED
            message = f"Large queue backlog: {queue_size} pending"
        elif running_experiments > 10:
            status = HealthStatus.DEGRADED
            message = f"High experiment load: {running_experiments} running"
        else:
            status = HealthStatus.HEALTHY
            message = "Experiment queue normal"
        
        return HealthCheck(
            name="experiment_queue",
            status=status,
            message=message,
            duration_ms=0.0,
            timestamp=datetime.now(),
            details={
                "queue_size": queue_size,
                "running_experiments": running_experiments
            }
        )
    except Exception as e:
        return HealthCheck(
            name="experiment_queue",
            status=HealthStatus.UNHEALTHY,
            message=f"Queue check failed: {str(e)}",
            duration_ms=0.0,
            timestamp=datetime.now()
        )


# Register default health checks
health_checker.register_check("system_resources", check_system_resources)
health_checker.register_check("gpu_status", check_gpu_status)
health_checker.register_check("database", check_database_connection)
health_checker.register_check("api_dependencies", check_api_dependencies)
health_checker.register_check("experiment_queue", check_experiment_queue)


async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    start_time = time.time()
    
    # Run all health checks
    check_results = await health_checker.run_all_checks()
    
    # Determine overall status
    overall_status = health_checker.get_overall_status(check_results)
    
    # Calculate uptime (would be more sophisticated in real implementation)
    uptime_seconds = time.time() - start_time
    
    # Build response
    response = {
        "status": overall_status.value,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "checks": {
            name: {
                "status": check.status.value,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "details": check.details or {}
            }
            for name, check in check_results.items()
        },
        "summary": {
            "total_checks": len(check_results),
            "healthy_checks": sum(1 for c in check_results.values() 
                                 if c.status == HealthStatus.HEALTHY),
            "degraded_checks": sum(1 for c in check_results.values() 
                                  if c.status == HealthStatus.DEGRADED),
            "unhealthy_checks": sum(1 for c in check_results.values() 
                                   if c.status == HealthStatus.UNHEALTHY)
        }
    }
    
    return response


def get_readiness_status() -> Dict[str, Any]:
    """Get readiness status (simpler check for load balancers)."""
    try:
        # Basic readiness checks
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Simple thresholds for readiness
        ready = cpu_percent < 95 and memory.percent < 95
        
        return {
            "ready": ready,
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent
        }
    except Exception as e:
        return {
            "ready": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def get_liveness_status() -> Dict[str, Any]:
    """Get liveness status (basic process health)."""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "process_id": os.getpid() if 'os' in globals() else None
    }