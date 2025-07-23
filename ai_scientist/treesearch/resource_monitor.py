"""
Resource monitoring and leak detection for tree search operations.
"""

import psutil
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def detect_resource_leaks() -> Dict[str, Any]:
    """
    Detect potential resource leaks in the system.
    
    Returns:
        Dictionary containing resource leak information
    """
    leak_info = {
        'timestamp': time.time(),
        'memory_usage_mb': 0,
        'open_files': 0,
        'process_count': 0,
        'potential_leaks': []
    }
    
    try:
        # Get current process info
        current_process = psutil.Process()
        
        # Memory usage
        memory_info = current_process.memory_info()
        leak_info['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        
        # Open file handles
        try:
            open_files = current_process.open_files()
            leak_info['open_files'] = len(open_files)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        
        # Child process count
        children = current_process.children(recursive=True)
        leak_info['process_count'] = len(children)
        
        # Detect potential leaks
        if leak_info['memory_usage_mb'] > 1000:  # More than 1GB
            leak_info['potential_leaks'].append({
                'type': 'memory',
                'severity': 'high' if leak_info['memory_usage_mb'] > 2000 else 'medium',
                'value': leak_info['memory_usage_mb'],
                'description': f"High memory usage: {leak_info['memory_usage_mb']:.1f}MB"
            })
        
        if leak_info['open_files'] > 100:
            leak_info['potential_leaks'].append({
                'type': 'file_handles',
                'severity': 'high' if leak_info['open_files'] > 500 else 'medium',
                'value': leak_info['open_files'],
                'description': f"Many open files: {leak_info['open_files']}"
            })
        
        if leak_info['process_count'] > 10:
            leak_info['potential_leaks'].append({
                'type': 'processes',
                'severity': 'high' if leak_info['process_count'] > 50 else 'medium',
                'value': leak_info['process_count'],
                'description': f"Many child processes: {leak_info['process_count']}"
            })
            
    except Exception as e:
        logger.error(f"Error detecting resource leaks: {e}")
        leak_info['error'] = str(e)
    
    return leak_info