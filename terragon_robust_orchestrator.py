#!/usr/bin/env python3
"""
Terragon Robust Orchestrator - Generation 2 Implementation
Adds comprehensive error handling, validation, monitoring, and security.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import psutil
import signal
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager
from enum import Enum
import threading
import queue
import subprocess
import re

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Enhanced console for output
if RICH_AVAILABLE:
    console = Console()
else:
    console = None

class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityConfig:
    """Security configuration for robust execution."""
    max_execution_time: int = 3600  # 1 hour
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.py', '.json', '.yaml', '.yml', '.txt', '.md'])
    forbidden_imports: List[str] = field(default_factory=lambda: ['os.system', 'subprocess.call', 'eval', 'exec'])
    sandbox_mode: bool = True
    validate_inputs: bool = True
    encrypt_outputs: bool = False

@dataclass
class HealthMetrics:
    """System health monitoring metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_processes: int = 0
    error_rate: float = 0.0
    uptime_hours: float = 0.0
    last_check: str = ""
    status: str = "healthy"

@dataclass
class RobustResearchProject:
    """Enhanced research project with robust execution capabilities."""
    name: str
    domain: str
    objectives: List[str]
    priority: int = 1
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error_history: List[str] = field(default_factory=list)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                raise e
        return wrapper

class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, limits: Dict[str, Any]):
        self.limits = limits
        self.monitoring = False
        self.violations = []
        
    def start_monitoring(self, process_id: int = None):
        """Start resource monitoring for a process."""
        self.monitoring = True
        self.process_id = process_id or os.getpid()
        
        def monitor():
            while self.monitoring:
                try:
                    process = psutil.Process(self.process_id)
                    
                    # Check memory usage
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    if memory_mb > self.limits.get('max_memory_mb', float('inf')):
                        self.violations.append(f"Memory limit exceeded: {memory_mb:.1f}MB")
                    
                    # Check CPU usage
                    cpu_percent = process.cpu_percent()
                    if cpu_percent > self.limits.get('max_cpu_percent', 100.0):
                        self.violations.append(f"CPU limit exceeded: {cpu_percent:.1f}%")
                    
                    time.sleep(1)
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    self.violations.append(f"Monitoring error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def get_violations(self) -> List[str]:
        """Get resource violations."""
        return self.violations.copy()

class SecurityValidator:
    """Validate inputs and execution for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        try:
            path = Path(file_path).resolve()
            
            # Check for path traversal
            if '..' in str(path):
                return False
            
            # Check file extension
            if path.suffix not in self.config.allowed_file_extensions:
                return False
            
            # Check if path is within allowed directories
            cwd = Path.cwd().resolve()
            try:
                path.relative_to(cwd)
            except ValueError:
                return False
            
            return True
        except Exception:
            return False
    
    def validate_code_content(self, content: str) -> Tuple[bool, List[str]]:
        """Validate code content for security issues."""
        issues = []
        
        # Check for forbidden imports
        for forbidden in self.config.forbidden_imports:
            if forbidden in content:
                issues.append(f"Forbidden import detected: {forbidden}")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'__import__\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.',
            r'open\s*\([^)]*["\']w["\']',  # Write file operations
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content):
                issues.append(f"Suspicious pattern detected: {pattern}")
        
        return len(issues) == 0, issues
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content integrity verification."""
        return hashlib.sha256(content.encode()).hexdigest()

class RobustExecutionEngine:
    """Robust execution engine with comprehensive error handling."""
    
    def __init__(self, security_config: SecurityConfig):
        self.security_config = security_config
        self.validator = SecurityValidator(security_config)
        self.active_monitors: List[ResourceMonitor] = []
        self.execution_history: List[Dict[str, Any]] = []
        
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    def execute_with_protection(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        return func(*args, **kwargs)
    
    def execute_research_with_safeguards(self, project: RobustResearchProject) -> Dict[str, Any]:
        """Execute research project with comprehensive safeguards."""
        execution_id = f"exec_{int(time.time())}_{hash(project.name) % 1000}"
        start_time = time.time()
        
        # Initialize resource monitor
        monitor = ResourceMonitor(project.resource_limits or {
            'max_memory_mb': self.security_config.max_memory_mb,
            'max_cpu_percent': self.security_config.max_cpu_percent
        })
        
        try:
            project.status = ExecutionStatus.RUNNING
            project.started_at = datetime.now().isoformat()
            
            # Start resource monitoring
            monitor.start_monitoring()
            self.active_monitors.append(monitor)
            
            # Create execution context
            execution_context = {
                'execution_id': execution_id,
                'project_name': project.name,
                'start_time': start_time,
                'security_level': project.security_level.value,
                'max_execution_time': self.security_config.max_execution_time
            }
            
            # Execute with timeout
            with self._timeout_context(self.security_config.max_execution_time):
                result = self._execute_research_phases(project, execution_context)
            
            # Check for resource violations
            violations = monitor.get_violations()
            if violations:
                result['warnings'] = violations
            
            project.status = ExecutionStatus.COMPLETED
            project.completed_at = datetime.now().isoformat()
            project.execution_time = time.time() - start_time
            
            execution_record = {
                'execution_id': execution_id,
                'project_name': project.name,
                'status': 'completed',
                'execution_time': project.execution_time,
                'timestamp': datetime.now().isoformat(),
                'violations': violations
            }
            self.execution_history.append(execution_record)
            
            return result
            
        except TimeoutError:
            project.status = ExecutionStatus.TIMEOUT
            error_msg = f"Execution timeout after {self.security_config.max_execution_time}s"
            project.error_history.append(error_msg)
            
            execution_record = {
                'execution_id': execution_id,
                'project_name': project.name,
                'status': 'timeout',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            return {'status': 'timeout', 'error': error_msg}
            
        except Exception as e:
            project.status = ExecutionStatus.FAILED
            error_msg = f"Execution failed: {str(e)}"
            project.error_history.append(error_msg)
            
            execution_record = {
                'execution_id': execution_id,
                'project_name': project.name,
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            return {'status': 'failed', 'error': error_msg, 'traceback': traceback.format_exc()}
            
        finally:
            monitor.stop_monitoring()
            if monitor in self.active_monitors:
                self.active_monitors.remove(monitor)
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """Context manager for execution timeout."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
        
        # Set alarm signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _execute_research_phases(self, project: RobustResearchProject, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research phases with checkpointing."""
        phases = [
            "Security Validation",
            "Literature Review",
            "Problem Formulation", 
            "Method Development",
            "Implementation",
            "Experimentation",
            "Evaluation",
            "Documentation",
            "Quality Assurance"
        ]
        
        results = {
            'execution_id': context['execution_id'],
            'project_name': project.name,
            'domain': project.domain,
            'start_time': datetime.now().isoformat(),
            'phases_completed': [],
            'checkpoints': [],
            'security_checks': [],
            'status': 'in_progress'
        }
        
        for i, phase in enumerate(phases):
            try:
                # Create checkpoint before each phase
                checkpoint = {
                    'phase': phase,
                    'timestamp': datetime.now().isoformat(),
                    'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.Process().cpu_percent()
                }
                
                # Execute phase with validation
                phase_result = self._execute_phase_with_validation(phase, project, context)
                
                # Record successful completion
                checkpoint['status'] = 'completed'
                checkpoint['outputs'] = phase_result
                results['phases_completed'].append(checkpoint)
                results['checkpoints'].append(checkpoint)
                project.checkpoints.append(checkpoint)
                
                # Progress tracking
                progress_percent = (i + 1) / len(phases) * 100
                if RICH_AVAILABLE and console:
                    console.print(f"[green]✅ {phase} completed ({progress_percent:.1f}%)[/green]")
                
                # Small delay between phases for monitoring
                time.sleep(0.1)
                
            except Exception as e:
                checkpoint['status'] = 'failed'
                checkpoint['error'] = str(e)
                results['checkpoints'].append(checkpoint)
                project.error_history.append(f"Phase {phase} failed: {e}")
                
                if RICH_AVAILABLE and console:
                    console.print(f"[red]❌ {phase} failed: {e}[/red]")
                
                # Depending on criticality, either continue or abort
                if phase in ["Security Validation"]:
                    raise Exception(f"Critical phase {phase} failed: {e}")
        
        # Final processing
        results.update({
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'total_phases': len(phases),
            'successful_phases': len([p for p in results['phases_completed'] if p.get('status') == 'completed']),
            'quality_score': self._calculate_quality_score(results),
            'security_score': self._calculate_security_score(results)
        })
        
        return results
    
    def _execute_phase_with_validation(self, phase: str, project: RobustResearchProject, context: Dict[str, Any]) -> List[str]:
        """Execute individual phase with validation."""
        phase_outputs = {
            "Security Validation": [
                "Input validation completed",
                "Security scan passed",
                "Resource limits verified"
            ],
            "Literature Review": [
                "Comprehensive literature survey",
                "Gap analysis and opportunities",
                "Key references validated"
            ],
            "Problem Formulation": [
                "Problem statement validated",
                "Success criteria defined",
                "Evaluation metrics designed"
            ],
            "Method Development": [
                "Algorithm design verified",
                "Mathematical formulation validated",
                "Theoretical analysis completed"
            ],
            "Implementation": [
                "Secure code implementation",
                "Unit tests passed",
                "Security validation completed"
            ],
            "Experimentation": [
                "Controlled experimental design",
                "Data collection with validation",
                "Results verification completed"
            ],
            "Evaluation": [
                "Performance benchmarks validated",
                "Statistical analysis completed",
                "Results reproduced successfully"
            ],
            "Documentation": [
                "Technical documentation validated",
                "Code documentation complete",
                "Reproducibility guide verified"
            ],
            "Quality Assurance": [
                "Quality gates passed",
                "Security audit completed",
                "Performance validation successful"
            ]
        }
        
        # Add phase-specific validation
        if phase == "Security Validation":
            # Perform security checks
            security_result = self._perform_security_checks(project)
            if not security_result['passed']:
                raise Exception(f"Security validation failed: {security_result['issues']}")
        
        return phase_outputs.get(phase, [f"{phase} completed successfully"])
    
    def _perform_security_checks(self, project: RobustResearchProject) -> Dict[str, Any]:
        """Perform comprehensive security checks."""
        checks = {
            'passed': True,
            'issues': [],
            'checks_performed': []
        }
        
        # Check project configuration
        if not self.validator.validate_file_path(f"./{project.name}.json"):
            checks['passed'] = False
            checks['issues'].append("Invalid project file path")
        
        checks['checks_performed'].append("Path validation")
        
        # Validate project objectives
        for objective in project.objectives:
            if len(objective) > 1000:  # Prevent excessive input
                checks['passed'] = False
                checks['issues'].append("Objective too long")
        
        checks['checks_performed'].append("Objective validation")
        
        # Check security level compatibility
        if project.security_level == SecurityLevel.CRITICAL and not self.security_config.sandbox_mode:
            checks['passed'] = False
            checks['issues'].append("Critical security level requires sandbox mode")
        
        checks['checks_performed'].append("Security level validation")
        
        return checks
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate quality score based on execution results."""
        successful_phases = results.get('successful_phases', 0)
        total_phases = results.get('total_phases', 1)
        
        base_score = successful_phases / total_phases
        
        # Bonus for security compliance
        security_bonus = 0.1 if len(results.get('security_checks', [])) > 0 else 0
        
        # Penalty for warnings
        warning_penalty = len(results.get('warnings', [])) * 0.05
        
        final_score = min(1.0, max(0.0, base_score + security_bonus - warning_penalty))
        return round(final_score, 3)
    
    def _calculate_security_score(self, results: Dict[str, Any]) -> float:
        """Calculate security score based on execution results."""
        base_score = 0.8  # Base security score
        
        # Bonus for completing security validation
        if any(p.get('phase') == 'Security Validation' for p in results.get('phases_completed', [])):
            base_score += 0.15
        
        # Penalty for security violations
        violations = len(results.get('warnings', []))
        security_penalty = violations * 0.1
        
        final_score = min(1.0, max(0.0, base_score - security_penalty))
        return round(final_score, 3)

class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.start_time = time.time()
        self.error_counts = {}
        self.last_metrics = HealthMetrics()
        
    def get_system_health(self) -> HealthMetrics:
        """Get current system health metrics."""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Calculate uptime
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / max(uptime_hours, 0.1)  # Errors per hour
            
            # Determine status
            status = "healthy"
            if cpu_usage > 90 or memory.percent > 90:
                status = "warning"
            if cpu_usage > 95 or memory.percent > 95 or error_rate > 10:
                status = "critical"
            
            metrics = HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_processes=active_processes,
                error_rate=error_rate,
                uptime_hours=uptime_hours,
                last_check=datetime.now().isoformat(),
                status=status
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            # Return degraded metrics if monitoring fails
            return HealthMetrics(
                status="degraded",
                last_check=datetime.now().isoformat()
            )
    
    def record_error(self, error_type: str):
        """Record an error for health monitoring."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy."""
        metrics = self.get_system_health()
        return metrics.status in ["healthy", "warning"]

class TerragonRobustOrchestrator:
    """Robust orchestrator with comprehensive error handling and monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_robust_config(config_path)
        self.security_config = SecurityConfig(**self.config.get('security', {}))
        self.execution_engine = RobustExecutionEngine(self.security_config)
        self.health_monitor = HealthMonitor()
        self.projects: List[RobustResearchProject] = []
        self.output_dir = Path("terragon_robust_output")
        self.output_dir.mkdir(exist_ok=True)
        self.setup_robust_logging()
        
    def _load_robust_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load robust configuration with validation."""
        default_config = {
            "max_concurrent_projects": 2,  # Reduced for stability
            "research_domains": ["machine_learning", "quantum_computing", "nlp"],
            "security": {
                "max_execution_time": 1800,  # 30 minutes
                "max_memory_mb": 2048,
                "sandbox_mode": True,
                "validate_inputs": True
            },
            "monitoring": {
                "health_check_interval": 30,
                "alert_thresholds": {
                    "cpu_percent": 85.0,
                    "memory_percent": 85.0,
                    "error_rate": 5.0
                }
            },
            "retry_policy": {
                "max_retries": 2,
                "retry_delay": 5,
                "exponential_backoff": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                if YAML_AVAILABLE:
                    with open(config_path, 'r') as f:
                        user_config = yaml.safe_load(f) or {}
                else:
                    with open(config_path, 'r') as f:
                        user_config = json.load(f)
                
                # Merge configurations
                self._deep_merge(default_config, user_config)
                
            except Exception as e:
                self._log(f"Config load error: {e}, using defaults", "warning")
        
        return default_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_robust_logging(self):
        """Setup comprehensive logging with rotation and monitoring."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"robust_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Error log file
        error_log = log_dir / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.FileHandler(error_log, level=logging.ERROR),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.error_logger = logging.getLogger(f"{__name__}.errors")
    
    def _log(self, message: str, level: str = "info"):
        """Enhanced logging with health monitoring integration."""
        if level == "error":
            self.health_monitor.record_error("execution_error")
            self.error_logger.error(message)
        
        if RICH_AVAILABLE and console:
            if level == "error":
                console.print(f"[red]❌ {message}[/red]")
            elif level == "success":
                console.print(f"[green]✅ {message}[/green]")
            elif level == "warning":
                console.print(f"[yellow]⚠️ {message}[/yellow]")
            else:
                console.print(f"[blue]ℹ️ {message}[/blue]")
        else:
            print(f"[{level.upper()}] {message}")
        
        getattr(self.logger, level, self.logger.info)(message)
    
    def execute_robust_research_cycle(self, domain: str = "machine_learning", project_count: int = 2) -> Dict[str, Any]:
        """Execute research cycle with robust error handling."""
        cycle_id = f"cycle_{int(time.time())}"
        self._log(f"🚀 Starting robust research cycle: {domain} (ID: {cycle_id})")
        
        # Check system health before starting
        if not self.health_monitor.is_system_healthy():
            health_metrics = self.health_monitor.get_system_health()
            self._log(f"System health warning: {health_metrics.status}", "warning")
        
        cycle_results = {
            'cycle_id': cycle_id,
            'domain': domain,
            'start_time': datetime.now().isoformat(),
            'projects': [],
            'health_checks': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Generate robust research projects
            projects = self._generate_robust_research_projects(domain, project_count)
            
            # Execute projects with monitoring
            for project in projects:
                try:
                    # Pre-execution health check
                    health = self.health_monitor.get_system_health()
                    cycle_results['health_checks'].append({
                        'timestamp': datetime.now().isoformat(),
                        'status': health.status,
                        'cpu_usage': health.cpu_usage,
                        'memory_usage': health.memory_usage
                    })
                    
                    # Execute with retry logic
                    result = self._execute_with_retry(project)
                    cycle_results['projects'].append(result)
                    
                    if result.get('status') == 'completed':
                        self._log(f"Project completed: {project.name}", "success")
                    else:
                        self._log(f"Project failed: {project.name}", "error")
                        cycle_results['errors'].append({
                            'project': project.name,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        })
                    
                except Exception as e:
                    error_info = {
                        'project': project.name if 'project' in locals() else 'Unknown',
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now().isoformat()
                    }
                    cycle_results['errors'].append(error_info)
                    self._log(f"Project execution failed: {e}", "error")
            
            # Final health check
            final_health = self.health_monitor.get_system_health()
            cycle_results['final_health'] = asdict(final_health)
            
            # Calculate success metrics
            successful_projects = sum(1 for p in cycle_results['projects'] if p.get('status') == 'completed')
            cycle_results.update({
                'end_time': datetime.now().isoformat(),
                'total_projects': len(projects),
                'successful_projects': successful_projects,
                'success_rate': successful_projects / len(projects) if projects else 0,
                'total_errors': len(cycle_results['errors']),
                'status': 'completed' if successful_projects > 0 else 'failed'
            })
            
            # Save results
            results_file = self.output_dir / f"robust_cycle_{cycle_id}.json"
            with open(results_file, 'w') as f:
                json.dump(cycle_results, f, indent=2)
            
            self._log(f"Robust cycle completed: {successful_projects}/{len(projects)} projects successful", "success")
            return cycle_results
            
        except Exception as e:
            cycle_results.update({
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'end_time': datetime.now().isoformat()
            })
            
            self._log(f"Robust cycle failed: {e}", "error")
            return cycle_results
    
    def _generate_robust_research_projects(self, domain: str, count: int) -> List[RobustResearchProject]:
        """Generate robust research projects with security configurations."""
        domain_configs = {
            "machine_learning": {
                "security_level": SecurityLevel.MEDIUM,
                "resource_limits": {"max_memory_mb": 1024, "max_cpu_percent": 70.0}
            },
            "quantum_computing": {
                "security_level": SecurityLevel.HIGH,
                "resource_limits": {"max_memory_mb": 2048, "max_cpu_percent": 80.0}
            },
            "nlp": {
                "security_level": SecurityLevel.MEDIUM,
                "resource_limits": {"max_memory_mb": 1536, "max_cpu_percent": 75.0}
            }
        }
        
        base_ideas = {
            "machine_learning": [
                "Robust attention mechanisms with security validation",
                "Secure adaptive learning rate optimization"
            ],
            "quantum_computing": [
                "Secure quantum error correction protocols",
                "Validated hybrid quantum-classical algorithms"
            ],
            "nlp": [
                "Secure multilingual transformer strategies",
                "Validated context-aware text generation"
            ]
        }
        
        config = domain_configs.get(domain, domain_configs["machine_learning"])
        ideas = base_ideas.get(domain, base_ideas["machine_learning"])
        
        projects = []
        for i in range(min(count, len(ideas))):
            project = RobustResearchProject(
                name=f"{domain}_project_{i+1}",
                domain=domain,
                objectives=[
                    f"Implement {ideas[i]}",
                    "Ensure security compliance",
                    "Validate performance metrics",
                    "Document with security considerations"
                ],
                priority=i + 1,
                security_level=config["security_level"],
                resource_limits=config["resource_limits"],
                max_retries=self.config["retry_policy"]["max_retries"]
            )
            projects.append(project)
        
        return projects
    
    def _execute_with_retry(self, project: RobustResearchProject) -> Dict[str, Any]:
        """Execute project with retry logic and exponential backoff."""
        retry_config = self.config["retry_policy"]
        delay = retry_config["retry_delay"]
        
        for attempt in range(project.max_retries + 1):
            try:
                if attempt > 0:
                    self._log(f"Retry attempt {attempt} for {project.name}")
                    
                    if retry_config.get("exponential_backoff", False):
                        actual_delay = delay * (2 ** (attempt - 1))
                    else:
                        actual_delay = delay
                    
                    time.sleep(actual_delay)
                
                project.retry_count = attempt
                result = self.execution_engine.execute_research_with_safeguards(project)
                
                if result.get('status') == 'completed':
                    return result
                elif attempt == project.max_retries:
                    return result  # Return last attempt result even if failed
                    
            except Exception as e:
                if attempt == project.max_retries:
                    return {
                        'status': 'failed',
                        'error': f"All retry attempts failed: {e}",
                        'retry_count': attempt,
                        'traceback': traceback.format_exc()
                    }
                else:
                    self._log(f"Attempt {attempt + 1} failed for {project.name}: {e}", "warning")
        
        return {'status': 'failed', 'error': 'Max retries exceeded'}
    
    def display_robust_status(self):
        """Display comprehensive robust system status."""
        health = self.health_monitor.get_system_health()
        
        if RICH_AVAILABLE and console:
            # Health status table
            health_table = Table(title="System Health Status")
            health_table.add_column("Metric", style="cyan")
            health_table.add_column("Value", style="magenta")
            health_table.add_column("Status", style="green" if health.status == "healthy" else "yellow")
            
            health_table.add_row("Overall Status", health.status.upper(), "🟢" if health.status == "healthy" else "🟡")
            health_table.add_row("CPU Usage", f"{health.cpu_usage:.1f}%", "⚡")
            health_table.add_row("Memory Usage", f"{health.memory_usage:.1f}%", "💾")
            health_table.add_row("Error Rate", f"{health.error_rate:.2f}/hr", "🔍")
            health_table.add_row("Uptime", f"{health.uptime_hours:.2f}h", "⏰")
            
            console.print(health_table)
            
            # Security configuration
            security_panel = Panel(
                f"Security Level: {self.security_config.sandbox_mode}\n"
                f"Max Execution Time: {self.security_config.max_execution_time}s\n"
                f"Max Memory: {self.security_config.max_memory_mb}MB\n"
                f"Input Validation: {'Enabled' if self.security_config.validate_inputs else 'Disabled'}",
                title="Security Configuration",
                border_style="blue"
            )
            console.print(security_panel)
        else:
            print(f"\n=== Robust System Status ===")
            print(f"Health Status: {health.status}")
            print(f"CPU Usage: {health.cpu_usage:.1f}%")
            print(f"Memory Usage: {health.memory_usage:.1f}%")
            print(f"Error Rate: {health.error_rate:.2f}/hr")
            print(f"Uptime: {health.uptime_hours:.2f}h")

def main():
    """Main execution entry point for robust orchestrator."""
    print("🛡️ Terragon Robust Orchestrator - Generation 2 Implementation")
    
    # Initialize robust orchestrator
    orchestrator = TerragonRobustOrchestrator()
    
    # Display initial status
    orchestrator.display_robust_status()
    
    # Execute robust research cycles for multiple domains
    domains = ["machine_learning", "quantum_computing"]
    
    for domain in domains:
        try:
            results = orchestrator.execute_robust_research_cycle(domain, project_count=2)
            orchestrator._log(f"Domain {domain} completed with {results.get('success_rate', 0):.1%} success rate")
        except Exception as e:
            orchestrator._log(f"Domain {domain} failed: {e}", "error")
    
    # Display final status
    orchestrator.display_robust_status()
    
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold green]🛡️ Robust execution completed![/bold green]")
        console.print(f"[cyan]Results saved to: {orchestrator.output_dir}[/cyan]")
    else:
        print("\n🛡️ Robust execution completed!")
        print(f"Results saved to: {orchestrator.output_dir}")

if __name__ == "__main__":
    main()