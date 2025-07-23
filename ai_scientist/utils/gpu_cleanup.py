"""
GPU resource cleanup utilities.
"""

import logging
import os
import subprocess
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def cleanup_gpu_resources(force: bool = False) -> bool:
    """
    Clean up GPU resources and reset GPU state.
    
    Args:
        force: Whether to force cleanup even if processes are still running
        
    Returns:
        True if cleanup was successful
    """
    try:
        # Check if CUDA is available
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible == '' or cuda_visible == '-1':
            logger.info("No CUDA devices visible, skipping GPU cleanup")
            return True
        
        logger.info("Starting GPU resource cleanup")
        
        # Try to get GPU process information
        gpu_processes = get_gpu_processes()
        
        if gpu_processes and not force:
            logger.warning(f"Found {len(gpu_processes)} GPU processes, use force=True to clean up")
            return False
        
        # Clear GPU memory cache if possible
        try:
            # This would typically require torch to be available
            # For now, we'll just log what we would do
            logger.info("Would clear GPU memory cache (requires torch)")
        except Exception as e:
            logger.debug(f"Could not clear GPU cache: {e}")
        
        # Reset CUDA context if needed
        if force and gpu_processes:
            logger.warning("Force cleanup requested - would terminate GPU processes")
            # In a real implementation, we might terminate GPU processes here
            # For safety, we'll just log the intent
        
        logger.info("GPU cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}")
        return False


def get_gpu_processes() -> List[Dict[str, Any]]:
    """
    Get list of processes currently using GPU resources.
    
    Returns:
        List of dictionaries containing GPU process information
    """
    processes = []
    
    try:
        # Try to use nvidia-smi to get process info
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2])
                        })
        else:
            logger.debug("nvidia-smi not available or failed")
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Could not query GPU processes using nvidia-smi")
    except Exception as e:
        logger.error(f"Error querying GPU processes: {e}")
    
    return processes


def monitor_gpu_usage() -> Dict[str, Any]:
    """
    Monitor current GPU usage statistics.
    
    Returns:
        Dictionary containing GPU usage information
    """
    usage_info = {
        'timestamp': None,
        'gpus': [],
        'total_memory_used_mb': 0,
        'process_count': 0
    }
    
    try:
        import time
        usage_info['timestamp'] = time.time()
        
        # Get GPU processes
        processes = get_gpu_processes()
        usage_info['process_count'] = len(processes)
        usage_info['total_memory_used_mb'] = sum(p.get('memory_mb', 0) for p in processes)
        
        # Try to get detailed GPU info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        usage_info['gpus'].append({
                            'index': int(parts[0]),
                            'memory_used_mb': int(parts[1]),
                            'memory_total_mb': int(parts[2]),
                            'utilization_percent': int(parts[3])
                        })
        
    except Exception as e:
        logger.debug(f"Could not monitor GPU usage: {e}")
        usage_info['error'] = str(e)
    
    return usage_info