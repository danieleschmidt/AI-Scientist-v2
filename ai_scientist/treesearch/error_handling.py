"""
Enhanced error handling and recovery mechanisms for AI Scientist tree search.

This module provides comprehensive error categorization, recovery strategies,
and resource management to improve the robustness of the tree search system.
"""

import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, wait
import functools

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Categories of node execution failures."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error" 
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    METRIC_PARSING = "metric_parsing"
    PLOTTING_ERROR = "plotting_error"
    GPU_ERROR = "gpu_error"
    SECURITY_VIOLATION = "security_violation"
    FILE_ACCESS_ERROR = "file_access_error"
    UNKNOWN = "unknown"


class NodeExecutionError(Exception):
    """Base class for node execution errors with categorization."""
    
    def __init__(self, message: str, failure_type: FailureType, 
                 node_id: str = None, recoverable: bool = True):
        super().__init__(message)
        self.failure_type = failure_type
        self.node_id = node_id
        self.recoverable = recoverable
        self.timestamp = time.time()


class TimeoutError(NodeExecutionError):
    """Execution exceeded time limit."""
    
    def __init__(self, message: str, node_id: str = None):
        super().__init__(message, FailureType.TIMEOUT, node_id, recoverable=True)


class SecurityViolationError(NodeExecutionError):
    """Code violated security policies."""
    
    def __init__(self, message: str, node_id: str = None):
        super().__init__(message, FailureType.SECURITY_VIOLATION, node_id, recoverable=False)


@dataclass
class ExecutionMetrics:
    """Metrics for tracking execution performance and failures."""
    total_nodes: int = 0
    successful_nodes: int = 0
    failed_nodes: int = 0
    timeout_nodes: int = 0
    debug_nodes: int = 0
    avg_execution_time: float = 0.0
    failure_by_type: Dict[FailureType, int] = None
    
    def __post_init__(self):
        if self.failure_by_type is None:
            self.failure_by_type = {ft: 0 for ft in FailureType}
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful_nodes / self.total_nodes * 100) if self.total_nodes > 0 else 0.0
    
    def record_success(self, execution_time: float):
        """Record a successful node execution."""
        self.total_nodes += 1
        self.successful_nodes += 1
        self._update_avg_time(execution_time)
    
    def record_failure(self, failure_type: FailureType, execution_time: float = 0.0):
        """Record a failed node execution."""
        self.total_nodes += 1
        self.failed_nodes += 1
        self.failure_by_type[failure_type] += 1
        
        if failure_type == FailureType.TIMEOUT:
            self.timeout_nodes += 1
            
        self._update_avg_time(execution_time)
    
    def _update_avg_time(self, execution_time: float):
        """Update average execution time."""
        if self.total_nodes == 1:
            self.avg_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 2.0 / (self.total_nodes + 1)
            self.avg_execution_time = alpha * execution_time + (1 - alpha) * self.avg_execution_time


class NodeClassifier:
    """Classifies node failures for targeted recovery strategies."""
    
    @staticmethod
    def classify_failure(node) -> FailureType:
        """Classify the type of failure from node information."""
        # Check exception type first
        if hasattr(node, 'exc_type') and node.exc_type:
            exc_type_str = str(node.exc_type).lower()
            
            if "syntaxerror" in exc_type_str:
                return FailureType.SYNTAX_ERROR
            elif "timeouterror" in exc_type_str or "timeout" in exc_type_str:
                return FailureType.TIMEOUT
            elif "memoryerror" in exc_type_str or "cuda out of memory" in exc_type_str:
                return FailureType.MEMORY_EXCEEDED
            elif "securityviolation" in exc_type_str or "securityerror" in exc_type_str:
                return FailureType.SECURITY_VIOLATION
        
        # Check exception info for more details
        if hasattr(node, 'exc_info') and node.exc_info:
            exc_info_str = str(node.exc_info).lower()
            
            if "cuda" in exc_info_str or "gpu" in exc_info_str:
                return FailureType.GPU_ERROR
            elif "permission denied" in exc_info_str or "file not found" in exc_info_str:
                return FailureType.FILE_ACCESS_ERROR
        
        # Check parsing errors
        if hasattr(node, 'parse_exc_type') and node.parse_exc_type:
            return FailureType.METRIC_PARSING
        
        # Check plotting errors
        if hasattr(node, 'plot_exc_type') and node.plot_exc_type:
            return FailureType.PLOTTING_ERROR
            
        return FailureType.UNKNOWN


class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on execution history and node complexity."""
    
    def __init__(self, base_timeout: int = 300):
        self.base_timeout = base_timeout
        self.execution_history: List[float] = []
        self.timeout_history: List[float] = []
    
    def get_timeout(self, node, stage_name: str = None) -> int:
        """Calculate timeout based on node complexity and execution history."""
        complexity_factor = 1.0
        
        # Increase timeout for debugging nodes
        if stage_name == "debug" or (hasattr(node, 'debug_depth') and node.debug_depth > 0):
            complexity_factor *= 1.5
        
        # Increase timeout based on code length
        if hasattr(node, 'code') and node.code:
            code_length = len(node.code)
            if code_length > 2000:
                complexity_factor *= 1.5
            elif code_length > 1000:
                complexity_factor *= 1.2
        
        # Adjust based on historical execution times
        if self.execution_history:
            avg_time = sum(self.execution_history[-10:]) / len(self.execution_history[-10:])
            if avg_time > self.base_timeout * 0.8:
                complexity_factor *= 1.3
        
        # Adjust based on recent timeouts
        recent_timeouts = self.timeout_history[-5:]
        if len(recent_timeouts) > 2:  # Many recent timeouts
            complexity_factor *= 1.4
        
        timeout = int(self.base_timeout * complexity_factor)
        return min(timeout, 1800)  # Cap at 30 minutes
    
    def record_execution(self, execution_time: float, timed_out: bool = False):
        """Record execution time and timeout status."""
        self.execution_history.append(execution_time)
        if timed_out:
            self.timeout_history.append(time.time())
        
        # Keep history manageable
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
        if len(self.timeout_history) > 20:
            self.timeout_history = self.timeout_history[-10:]


class NodeRecoveryStrategy:
    """Strategies for recovering from various types of node failures."""
    
    @staticmethod
    def recover_from_timeout(node) -> Optional[Any]:
        """Attempt to salvage partial results from timeout."""
        if not hasattr(node, 'term_out') or not node.term_out:
            return None
        
        # Look for partial results in output
        partial_results = {}
        for line in node.term_out:
            line_str = str(line).lower()
            if "epoch" in line_str or "step" in line_str:
                # Extract training progress
                try:
                    if "loss" in line_str:
                        # Simple regex to extract loss value
                        import re
                        loss_match = re.search(r'loss[:\s=]*([\d.]+)', line_str)
                        if loss_match:
                            partial_results['partial_loss'] = float(loss_match.group(1))
                    if "accuracy" in line_str or "acc" in line_str:
                        acc_match = re.search(r'acc[uracy]*[:\s=]*([\d.]+)', line_str)
                        if acc_match:
                            partial_results['partial_accuracy'] = float(acc_match.group(1))
                except (ValueError, AttributeError):
                    pass
        
        if partial_results:
            logger.info(f"Recovered partial results from timeout: {partial_results}")
            return partial_results
        
        return None
    
    @staticmethod
    def recover_from_gpu_failure(node) -> str:
        """Generate CPU fallback code for GPU failures."""
        if not hasattr(node, 'code') or not node.code:
            return None
        
        # Simple GPU to CPU conversion
        cpu_code = node.code
        gpu_replacements = [
            ('.cuda()', '.cpu()'),
            ('.to(device)', '.to("cpu")'),
            ('device="cuda"', 'device="cpu"'),
            ("device='cuda'", "device='cpu'"),
            ('torch.cuda', 'torch'),
        ]
        
        for old, new in gpu_replacements:
            cpu_code = cpu_code.replace(old, new)
        
        # Add CPU device override at the beginning
        cpu_override = """
# GPU failure recovery: Force CPU execution
import torch
if torch.cuda.is_available():
    print("GPU available but forcing CPU execution due to previous GPU failure")
device = "cpu"
torch.cuda.is_available = lambda: False

"""
        cpu_code = cpu_override + cpu_code
        
        logger.info("Generated CPU fallback code for GPU failure")
        return cpu_code
    
    @staticmethod
    def recover_from_syntax_error(node) -> str:
        """Attempt to fix common syntax errors."""
        if not hasattr(node, 'code') or not node.code:
            return None
        
        # Common syntax fixes
        fixed_code = node.code
        
        # Fix common indentation issues (very basic)
        lines = fixed_code.split('\n')
        fixed_lines = []
        for line in lines:
            # Remove extra spaces at the beginning if not part of logical indentation
            if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                fixed_lines.append(line.lstrip())
            else:
                fixed_lines.append(line)
        
        fixed_code = '\n'.join(fixed_lines)
        
        # Add missing imports if obvious
        if 'plt.' in fixed_code and 'import matplotlib.pyplot as plt' not in fixed_code:
            fixed_code = 'import matplotlib.pyplot as plt\n' + fixed_code
        
        if 'np.' in fixed_code and 'import numpy as np' not in fixed_code:
            fixed_code = 'import numpy as np\n' + fixed_code
        
        if fixed_code != node.code:
            logger.info("Applied basic syntax error fixes")
            return fixed_code
        
        return None


class ResourceManager:
    """Manages and cleans up resources used during node execution."""
    
    def __init__(self):
        self.active_resources: Dict[str, Dict[str, Any]] = {}
        self.cleanup_timeout = 10
    
    def register_resource(self, resource_id: str, resource_type: str, 
                         cleanup_fn: Callable, metadata: Dict[str, Any] = None):
        """Register a resource with its cleanup function."""
        self.active_resources[resource_id] = {
            "type": resource_type,
            "cleanup": cleanup_fn,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        logger.debug(f"Registered resource {resource_id} of type {resource_type}")
    
    def cleanup_resource(self, resource_id: str) -> bool:
        """Clean up a specific resource."""
        if resource_id not in self.active_resources:
            return False
        
        resource = self.active_resources[resource_id]
        try:
            resource["cleanup"]()
            del self.active_resources[resource_id]
            logger.debug(f"Successfully cleaned up resource {resource_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clean up resource {resource_id}: {e}")
            return False
    
    def cleanup_all(self, timeout: int = None) -> Dict[str, bool]:
        """Clean up all registered resources."""
        if timeout is None:
            timeout = self.cleanup_timeout
        
        results = {}
        
        if not self.active_resources:
            return results
        
        # Use thread pool for parallel cleanup
        with ThreadPoolExecutor(max_workers=5) as executor:
            cleanup_futures = {
                resource_id: executor.submit(self._cleanup_resource_safe, resource_id, resource)
                for resource_id, resource in self.active_resources.items()
            }
            
            # Wait for all cleanups with timeout
            done, not_done = wait(cleanup_futures.values(), timeout=timeout)
            
            # Collect results
            for resource_id, future in cleanup_futures.items():
                if future in done:
                    try:
                        results[resource_id] = future.result()
                    except Exception as e:
                        logger.error(f"Cleanup future failed for {resource_id}: {e}")
                        results[resource_id] = False
                else:
                    logger.warning(f"Cleanup timeout for resource {resource_id}")
                    results[resource_id] = False
                    future.cancel()
        
        # Clear all resources (even if cleanup failed)
        self.active_resources.clear()
        
        return results
    
    def _cleanup_resource_safe(self, resource_id: str, resource: Dict[str, Any]) -> bool:
        """Safe resource cleanup with error handling."""
        try:
            resource["cleanup"]()
            logger.debug(f"Cleaned up resource {resource_id}")
            return True
        except Exception as e:
            logger.error(f"Resource cleanup failed for {resource_id}: {e}")
            return False
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about active resources."""
        now = time.time()
        stats = {
            "total_resources": len(self.active_resources),
            "resource_types": {},
            "oldest_resource_age": 0
        }
        
        if self.active_resources:
            ages = [now - r["created_at"] for r in self.active_resources.values()]
            stats["oldest_resource_age"] = max(ages)
            
            for resource in self.active_resources.values():
                rtype = resource["type"]
                stats["resource_types"][rtype] = stats["resource_types"].get(rtype, 0) + 1
        
        return stats


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      max_delay: float = 60.0, backoff_factor: float = 2.0,
                      retryable_errors: tuple = None):
    """Decorator for retrying functions with exponential backoff."""
    
    if retryable_errors is None:
        retryable_errors = (TimeoutError, ConnectionError, OSError)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                        raise e
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class EnhancedErrorHandler:
    """Main error handler that coordinates all error handling components."""
    
    def __init__(self, base_timeout: int = 300):
        self.timeout_manager = AdaptiveTimeoutManager(base_timeout)
        self.resource_manager = ResourceManager()
        self.metrics = ExecutionMetrics()
        self.recovery_strategy = NodeRecoveryStrategy()
        self.classifier = NodeClassifier()
    
    def handle_node_failure(self, node, exception: Exception = None) -> Dict[str, Any]:
        """Comprehensive node failure handling with recovery attempts."""
        failure_type = self.classifier.classify_failure(node)
        
        # Record metrics
        execution_time = getattr(node, 'exec_time', 0.0)
        self.metrics.record_failure(failure_type, execution_time)
        
        # Log the failure
        logger.error(f"Node {getattr(node, 'id', 'unknown')} failed: {failure_type.value}",
                    extra={
                        "node_id": getattr(node, 'id', 'unknown'),
                        "failure_type": failure_type.value,
                        "execution_time": execution_time,
                        "debug_depth": getattr(node, 'debug_depth', 0)
                    })
        
        # Attempt recovery based on failure type
        recovery_result = self._attempt_recovery(node, failure_type)
        
        return {
            "failure_type": failure_type,
            "recoverable": self._is_recoverable(failure_type),
            "recovery_result": recovery_result,
            "should_debug": self._should_debug(node, failure_type)
        }
    
    def handle_node_success(self, node):
        """Handle successful node execution."""
        execution_time = getattr(node, 'exec_time', 0.0)
        self.metrics.record_success(execution_time)
        self.timeout_manager.record_execution(execution_time, timed_out=False)
    
    def get_timeout_for_node(self, node, stage_name: str = None) -> int:
        """Get adaptive timeout for a node."""
        return self.timeout_manager.get_timeout(node, stage_name)
    
    def _attempt_recovery(self, node, failure_type: FailureType) -> Optional[Any]:
        """Attempt recovery based on failure type."""
        try:
            if failure_type == FailureType.TIMEOUT:
                return self.recovery_strategy.recover_from_timeout(node)
            elif failure_type == FailureType.GPU_ERROR:
                return self.recovery_strategy.recover_from_gpu_failure(node)
            elif failure_type == FailureType.SYNTAX_ERROR:
                return self.recovery_strategy.recover_from_syntax_error(node)
            else:
                return None
        except Exception as e:
            logger.warning(f"Recovery attempt failed: {e}")
            return None
    
    def _is_recoverable(self, failure_type: FailureType) -> bool:
        """Determine if a failure type is potentially recoverable."""
        recoverable_types = {
            FailureType.TIMEOUT,
            FailureType.GPU_ERROR,
            FailureType.SYNTAX_ERROR,
            FailureType.MEMORY_EXCEEDED,
            FailureType.FILE_ACCESS_ERROR
        }
        return failure_type in recoverable_types
    
    def _should_debug(self, node, failure_type: FailureType) -> bool:
        """Determine if a failed node should be debugged."""
        # Don't debug security violations or unrecoverable errors
        if failure_type == FailureType.SECURITY_VIOLATION:
            return False
        
        # Check debug depth limit
        debug_depth = getattr(node, 'debug_depth', 0)
        max_debug_depth = 3  # This should come from config
        
        if debug_depth >= max_debug_depth:
            logger.info(f"Max debug depth {max_debug_depth} reached for node {getattr(node, 'id', 'unknown')}")
            return False
        
        return True
    
    def cleanup_and_get_stats(self) -> Dict[str, Any]:
        """Clean up all resources and return comprehensive statistics."""
        cleanup_results = self.resource_manager.cleanup_all()
        resource_stats = self.resource_manager.get_resource_stats()
        
        return {
            "execution_metrics": {
                "total_nodes": self.metrics.total_nodes,
                "successful_nodes": self.metrics.successful_nodes,
                "failed_nodes": self.metrics.failed_nodes,
                "success_rate": self.metrics.success_rate,
                "avg_execution_time": self.metrics.avg_execution_time,
                "failures_by_type": {ft.value: count for ft, count in self.metrics.failure_by_type.items()}
            },
            "resource_cleanup": cleanup_results,
            "resource_stats": resource_stats
        }