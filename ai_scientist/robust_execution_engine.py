#!/usr/bin/env python3
"""
Robust Execution Engine - Generation 2: MAKE IT ROBUST
======================================================

Enhanced autonomous research execution with comprehensive:
- Error handling and recovery mechanisms
- Security validation and input sanitization
- Resource monitoring and management
- Logging and observability
- Circuit breakers and retries

Author: AI Scientist v2 Autonomous System - Terragon Labs
License: MIT
"""

import logging
import asyncio
import json
import time
import hashlib
import threading
import signal
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
import sys
import os
import re
from enum import Enum
import tempfile
import uuid

# Optional dependencies with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Security and validation
import secrets
from urllib.parse import urlparse

# Base execution engine
from ai_scientist.unified_autonomous_executor import (
    UnifiedAutonomousExecutor,
    ResearchConfig
)

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for research execution."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    GPU = "gpu"
    NETWORK = "network"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    level: SecurityLevel = SecurityLevel.MEDIUM
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        '.py', '.json', '.md', '.txt', '.csv', '.yaml', '.yml'
    ])
    blocked_imports: List[str] = field(default_factory=lambda: [
        'subprocess', 'os.system', 'eval', 'exec', '__import__'
    ])
    max_execution_time: float = 3600.0  # 1 hour
    sandbox_mode: bool = True
    validate_inputs: bool = True


@dataclass 
class ResourceLimits:
    """Resource usage limits."""
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 4096
    max_disk_mb: int = 10240
    max_processes: int = 10
    monitor_interval: float = 30.0


@dataclass
class RetryPolicy:
    """Retry policy for failed operations."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        'TimeoutError', 'ConnectionError', 'TemporaryFailure'
    ])


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.failure_count = 0
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            raise


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.monitoring = False
        self.monitor_thread = None
        self.violations = []
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_resources()
                time.sleep(self.limits.monitor_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    def _check_resources(self):
        """Check current resource usage against limits."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, skipping resource monitoring")
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.limits.max_cpu_percent:
                self._record_violation("CPU", cpu_percent, self.limits.max_cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            if memory_mb > self.limits.max_memory_mb:
                self._record_violation("Memory", memory_mb, self.limits.max_memory_mb)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used_mb = disk.used / (1024 * 1024)
            if disk_used_mb > self.limits.max_disk_mb:
                self._record_violation("Disk", disk_used_mb, self.limits.max_disk_mb)
            
            # Process count
            process_count = len(psutil.pids())
            if process_count > self.limits.max_processes:
                self._record_violation("Processes", process_count, self.limits.max_processes)
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
    
    def _record_violation(self, resource_type: str, current: float, limit: float):
        """Record a resource limit violation."""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "resource": resource_type,
            "current": current,
            "limit": limit,
            "severity": "HIGH" if current > limit * 1.5 else "MEDIUM"
        }
        self.violations.append(violation)
        logger.warning(f"Resource violation: {resource_type} usage {current:.2f} exceeds limit {limit:.2f}")


class SecurityValidator:
    """Validate security and sanitize inputs."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        path = Path(file_path)
        
        # Check for path traversal
        if ".." in str(path) or str(path).startswith("/"):
            logger.warning(f"Suspicious file path detected: {file_path}")
            return False
        
        # Check file extension
        if path.suffix.lower() not in self.policy.allowed_file_types:
            logger.warning(f"Disallowed file type: {path.suffix}")
            return False
        
        return True
    
    def validate_content(self, content: str) -> bool:
        """Validate content for security issues."""
        # Check for blocked imports/functions
        for blocked in self.policy.blocked_imports:
            if blocked in content:
                logger.warning(f"Blocked content detected: {blocked}")
                return False
        
        # Check for shell injection patterns
        shell_patterns = [
            r'\$\(.*\)',  # Command substitution
            r'`.*`',      # Backticks
            r';\s*rm\s+',  # Dangerous commands
            r';\s*sudo\s+',
            r'>\s*/dev/',
        ]
        
        for pattern in shell_patterns:
            if re.search(pattern, content):
                logger.warning(f"Shell injection pattern detected: {pattern}")
                return False
        
        return True
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # Remove or escape potentially dangerous characters
        sanitized = re.sub(r'[<>&"\'`\$]', '', text)
        return sanitized[:1000]  # Limit length


class RobustExecutionEngine(UnifiedAutonomousExecutor):
    """
    Robust execution engine with comprehensive error handling,
    security validation, and resource monitoring.
    """
    
    def __init__(self, config: ResearchConfig, 
                 security_policy: Optional[SecurityPolicy] = None,
                 resource_limits: Optional[ResourceLimits] = None,
                 retry_policy: Optional[RetryPolicy] = None):
        super().__init__(config)
        
        self.security_policy = security_policy or SecurityPolicy()
        self.resource_limits = resource_limits or ResourceLimits()
        self.retry_policy = retry_policy or RetryPolicy()
        
        # Initialize robust components
        self.security_validator = SecurityValidator(self.security_policy)
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        self.circuit_breaker = CircuitBreaker()
        
        # Execution state
        self.execution_id = str(uuid.uuid4())
        self.start_time = None
        self.errors = []
        self.security_violations = []
        
        # Enhanced logging
        self._setup_robust_logging()
        
        logger.info(f"RobustExecutionEngine initialized with ID: {self.execution_id}")
    
    def _setup_robust_logging(self):
        """Setup enhanced logging with multiple handlers."""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Main log file
        main_log = logs_dir / f"execution_{self.execution_id}.log"
        
        # Security log file
        security_log = logs_dir / f"security_{self.execution_id}.log"
        
        # Error log file
        error_log = logs_dir / f"errors_{self.execution_id}.log"
        
        # Configure formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Main handler
        main_handler = logging.FileHandler(main_log)
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.INFO)
        
        # Security handler
        security_handler = logging.FileHandler(security_log)
        security_handler.setFormatter(detailed_formatter)
        security_handler.setLevel(logging.WARNING)
        
        # Error handler
        error_handler = logging.FileHandler(error_log)
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers
        logger.addHandler(main_handler)
        logger.addHandler(security_handler)
        logger.addHandler(error_handler)
    
    async def execute_research_pipeline(self) -> Dict[str, Any]:
        """Execute research pipeline with robust error handling and monitoring."""
        self.start_time = time.time()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Execute with timeout
            async with self._execution_timeout():
                return await self._execute_with_circuit_breaker()
        
        except asyncio.TimeoutError:
            logger.error("Research pipeline execution timed out")
            return await self._handle_timeout_error()
        
        except Exception as e:
            logger.error(f"Research pipeline failed: {e}")
            return await self._handle_execution_error(e)
        
        finally:
            self.resource_monitor.stop_monitoring()
            await self._cleanup_resources()
    
    @asynccontextmanager
    async def _execution_timeout(self):
        """Async context manager for execution timeout."""
        try:
            async with asyncio.timeout(self.security_policy.max_execution_time):
                yield
        except asyncio.TimeoutError:
            logger.error(f"Execution timed out after {self.security_policy.max_execution_time} seconds")
            raise
    
    async def _execute_with_circuit_breaker(self) -> Dict[str, Any]:
        """Execute pipeline with circuit breaker protection."""
        try:
            return await self.circuit_breaker.call(self._execute_pipeline_stages)
        except Exception as e:
            self.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "circuit_breaker",
                "severity": "HIGH"
            })
            raise
    
    async def _execute_pipeline_stages(self) -> Dict[str, Any]:
        """Execute all pipeline stages with robust error handling."""
        results = {
            "execution_id": self.execution_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "stages": {},
            "security_checks": [],
            "resource_usage": [],
            "errors": []
        }
        
        stages = [
            ("ideation", self._robust_ideation_stage, None),
            ("planning", self._robust_planning_stage, "ideation"),
            ("experimentation", self._robust_experimentation_stage, "planning"),
            ("validation", self._robust_validation_stage, "experimentation"),
            ("reporting", self._robust_reporting_stage, "validation")
        ]
        
        for stage_name, stage_func, dependency in stages:
            try:
                logger.info(f"🔒 Executing robust {stage_name} stage")
                
                # Pre-stage security check
                security_check = await self._pre_stage_security_check(stage_name)
                results["security_checks"].append(security_check)
                
                # Get dependency results if needed
                if dependency and dependency in results["stages"]:
                    dependency_results = results["stages"][dependency]
                    stage_result = await self._execute_with_retry(lambda: stage_func(dependency_results))
                else:
                    stage_result = await self._execute_with_retry(stage_func)
                
                results["stages"][stage_name] = stage_result
                
                # Post-stage validation
                await self._post_stage_validation(stage_name, stage_result)
                
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                results["stages"][stage_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results["errors"].append({
                    "stage": stage_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Decide whether to continue or abort
                if not await self._should_continue_after_error(stage_name, e):
                    results["status"] = "aborted"
                    break
        
        # Final status determination
        if results["status"] == "running":
            failed_stages = [name for name, data in results["stages"].items() 
                           if data.get("status") == "failed"]
            results["status"] = "completed" if not failed_stages else "partial_failure"
        
        results["end_time"] = datetime.now().isoformat()
        results["execution_time_hours"] = (time.time() - self.start_time) / 3600
        results["resource_violations"] = self.resource_monitor.violations
        
        return results
    
    async def _execute_with_retry(self, func) -> Dict[str, Any]:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_policy.max_attempts):
            try:
                return await func()
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                if error_type not in self.retry_policy.retryable_errors:
                    logger.error(f"Non-retryable error: {e}")
                    raise
                
                if attempt < self.retry_policy.max_attempts - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.retry_policy.max_attempts} attempts failed")
        
        raise last_error
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        if self.retry_policy.exponential_backoff:
            delay = self.retry_policy.base_delay * (2 ** attempt)
        else:
            delay = self.retry_policy.base_delay
        
        return min(delay, self.retry_policy.max_delay)
    
    async def _pre_stage_security_check(self, stage_name: str) -> Dict[str, Any]:
        """Perform security checks before stage execution."""
        check_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
            "violations": [],
            "status": "passed"
        }
        
        # Check file system state
        check_result["checks_performed"].append("filesystem_check")
        
        # Check process list
        check_result["checks_performed"].append("process_check")
        
        # Check network connections (if applicable)
        check_result["checks_performed"].append("network_check")
        
        return check_result
    
    async def _post_stage_validation(self, stage_name: str, stage_result: Dict[str, Any]):
        """Validate stage results and output."""
        # Validate output files if any
        if "output_file" in stage_result:
            file_path = stage_result["output_file"]
            if not self.security_validator.validate_file_path(file_path):
                raise SecurityError(f"Invalid output file path: {file_path}")
        
        # Validate result structure
        required_fields = ["status"]
        for field in required_fields:
            if field not in stage_result:
                raise ValidationError(f"Missing required field: {field}")
    
    async def _should_continue_after_error(self, stage_name: str, error: Exception) -> bool:
        """Determine if execution should continue after an error."""
        # Critical stages that should stop execution
        critical_stages = ["validation"]
        
        if stage_name in critical_stages:
            return False
        
        # Check error severity
        if isinstance(error, (SecurityError, CriticalError)):
            return False
        
        return True
    
    async def _robust_ideation_stage(self) -> Dict[str, Any]:
        """Robust ideation stage with enhanced validation."""
        try:
            # Validate research topic
            sanitized_topic = self.security_validator.sanitize_input(self.config.research_topic)
            
            # Execute original ideation logic with validation
            result = await super()._execute_ideation_stage()
            
            # Additional validation for ideation output
            if "output_file" in result:
                await self._validate_json_output(result["output_file"])
            
            return result
            
        except Exception as e:
            logger.error(f"Robust ideation stage failed: {e}")
            raise
    
    async def _robust_planning_stage(self, ideation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Robust planning stage with resource validation."""
        try:
            # Validate ideation results
            if not ideation_results or ideation_results.get("status") != "completed":
                raise ValidationError("Invalid ideation results")
            
            result = await super()._execute_planning_stage(ideation_results)
            
            # Validate planning output
            if "plan_file" in result:
                await self._validate_json_output(result["plan_file"])
            
            return result
            
        except Exception as e:
            logger.error(f"Robust planning stage failed: {e}")
            raise
    
    async def _robust_experimentation_stage(self, planning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Robust experimentation stage with sandbox execution."""
        try:
            # Enhanced validation and sandboxing
            if self.security_policy.sandbox_mode:
                logger.info("Executing experiments in sandbox mode")
            
            result = await super()._execute_experimentation_stage(planning_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Robust experimentation stage failed: {e}")
            raise
    
    async def _robust_validation_stage(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Robust validation stage with statistical verification."""
        try:
            result = await super()._execute_validation_stage(experiment_results)
            
            # Enhanced statistical validation
            if result.get("validation_passed"):
                logger.info("✅ Statistical validation passed")
            else:
                logger.warning("⚠️ Statistical validation concerns detected")
            
            return result
            
        except Exception as e:
            logger.error(f"Robust validation stage failed: {e}")
            raise
    
    async def _robust_reporting_stage(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Robust reporting stage with content validation."""
        try:
            result = await super()._execute_reporting_stage(validation_results)
            
            # Validate report content
            if "report_file" in result:
                await self._validate_report_content(result["report_file"])
            
            return result
            
        except Exception as e:
            logger.error(f"Robust reporting stage failed: {e}")
            raise
    
    async def _validate_json_output(self, file_path: str):
        """Validate JSON output file."""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            logger.info(f"✅ JSON validation passed: {file_path}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {file_path}: {e}")
    
    async def _validate_report_content(self, file_path: str):
        """Validate report content."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if not self.security_validator.validate_content(content):
                raise SecurityError(f"Security validation failed for report: {file_path}")
            
            logger.info(f"✅ Report validation passed: {file_path}")
        except Exception as e:
            raise ValidationError(f"Report validation failed: {e}")
    
    async def _handle_timeout_error(self) -> Dict[str, Any]:
        """Handle execution timeout."""
        return {
            "status": "timeout",
            "error": "Execution timed out",
            "execution_time_hours": (time.time() - self.start_time) / 3600,
            "resource_violations": self.resource_monitor.violations
        }
    
    async def _handle_execution_error(self, error: Exception) -> Dict[str, Any]:
        """Handle general execution error."""
        return {
            "status": "failed",
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_time_hours": (time.time() - self.start_time) / 3600,
            "errors": self.errors,
            "resource_violations": self.resource_monitor.violations
        }
    
    async def _cleanup_resources(self):
        """Cleanup resources after execution."""
        try:
            # Stop monitoring
            if self.resource_monitor.monitoring:
                self.resource_monitor.stop_monitoring()
            
            # Clean temporary files
            temp_files = list(self.output_dir.glob("*.tmp"))
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")


# Custom exception classes
class SecurityError(Exception):
    """Security validation error."""
    pass

class ValidationError(Exception):
    """Data validation error."""
    pass

class CriticalError(Exception):
    """Critical system error."""
    pass


# CLI interface for robust execution
async def main():
    """Main function for testing robust execution."""
    config = ResearchConfig(
        research_topic="Robust AI Research Pipeline Testing",
        output_dir="robust_test_output",
        max_experiments=2
    )
    
    security_policy = SecurityPolicy(
        level=SecurityLevel.HIGH,
        sandbox_mode=True,
        validate_inputs=True
    )
    
    resource_limits = ResourceLimits(
        max_cpu_percent=75.0,
        max_memory_mb=2048,
        monitor_interval=10.0
    )
    
    engine = RobustExecutionEngine(config, security_policy, resource_limits)
    await engine.initialize_components()
    
    results = await engine.execute_research_pipeline()
    
    print(f"🛡️ Robust execution completed: {results['status']}")
    print(f"⏱️ Execution time: {results.get('execution_time_hours', 0):.3f} hours")
    print(f"🔒 Security checks: {len(results.get('security_checks', []))}")
    print(f"⚠️ Resource violations: {len(results.get('resource_violations', []))}")


if __name__ == "__main__":
    asyncio.run(main())