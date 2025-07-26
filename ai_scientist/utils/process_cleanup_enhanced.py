"""
Enhanced process cleanup and resource management utilities.
Implements the requirements from backlog item: process-cleanup (WSJF: 4.25)

Acceptance criteria:
- Ensure all child processes are properly terminated
- Add timeout handling for process cleanup
- Implement resource leak detection
- Add proper signal handling
"""

import os
import signal
import time
import multiprocessing
import psutil
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


def cleanup_child_processes(processes: List[multiprocessing.Process], timeout: int = 5) -> bool:
    """
    Clean up child processes with escalating termination strategy.
    
    Args:
        processes: List of processes to terminate
        timeout: Maximum time to wait for graceful termination
        
    Returns:
        True if all processes were successfully terminated
    """
    if not processes:
        return True
        
    logger.info(f"Cleaning up {len(processes)} child processes with {timeout}s timeout")
    
    # Step 1: Try graceful termination with SIGTERM
    alive_processes = []
    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
                alive_processes.append(proc)
        except (OSError, AttributeError):
            logger.warning(f"Failed to terminate process {proc.pid if hasattr(proc, 'pid') else 'unknown'}")
            continue
    
    if not alive_processes:
        logger.info("All processes terminated gracefully")
        return True
    
    # Step 2: Wait for graceful termination
    start_time = time.time()
    while alive_processes and (time.time() - start_time) < timeout:
        still_alive = []
        for proc in alive_processes:
            try:
                if proc.is_alive():
                    still_alive.append(proc)
                else:
                    logger.debug(f"Process {proc.pid if hasattr(proc, 'pid') else 'unknown'} terminated gracefully")
            except (OSError, AttributeError):
                # Process already dead
                continue
        alive_processes = still_alive
        if alive_processes:
            # Get cleanup interval from configuration
            try:
                from .config import get_timeout
                sleep_interval = get_timeout("process_cleanup_interval")
            except ImportError:
                sleep_interval = 0.1
            time.sleep(sleep_interval)
    
    # Step 3: Force kill remaining processes
    if alive_processes:
        logger.warning(f"Force killing {len(alive_processes)} stubborn processes")
        for proc in alive_processes:
            try:
                if proc.is_alive():
                    proc.kill()
                    logger.info(f"Force killed process {proc.pid if hasattr(proc, 'pid') else 'unknown'}")
            except (OSError, AttributeError):
                logger.warning(f"Failed to force kill process {proc.pid if hasattr(proc, 'pid') else 'unknown'}")
                continue
    
    # Step 4: Final verification
    final_check = []
    for proc in processes:
        try:
            if proc.is_alive():
                final_check.append(proc)
        except (OSError, AttributeError):
            continue
    
    success = len(final_check) == 0
    if success:
        logger.info("All child processes successfully cleaned up")
    else:
        logger.error(f"Failed to clean up {len(final_check)} processes")
    
    return success


def detect_orphaned_processes(keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Detect potentially orphaned processes related to the application.
    
    Args:
        keywords: List of keywords to search for in process command lines
        
    Returns:
        List of dictionaries containing orphaned process information
    """
    if keywords is None:
        keywords = ["python", "torch", "mp", "bfts", "experiment", "ai_scientist"]
    
    orphaned = []
    current_pid = os.getpid()
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
            try:
                proc_info = proc.info
                if proc_info['pid'] == current_pid:
                    continue
                    
                cmdline = " ".join(proc_info['cmdline'] or []).lower()
                if any(keyword in cmdline for keyword in keywords):
                    orphaned.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cmdline': cmdline,
                        'memory_mb': proc_info['memory_info'].rss / 1024 / 1024 if proc_info['memory_info'] else 0,
                        'age_seconds': time.time() - proc_info['create_time'] if proc_info['create_time'] else 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error detecting orphaned processes: {e}")
    
    if orphaned:
        logger.warning(f"Detected {len(orphaned)} potentially orphaned processes")
        for proc in orphaned:
            logger.warning(f"Orphaned: PID={proc['pid']}, Memory={proc['memory_mb']:.1f}MB, Age={proc['age_seconds']:.1f}s")
    
    return orphaned


def setup_cleanup_signal_handlers():
    """
    Set up signal handlers for graceful cleanup on system signals.
    """
    def cleanup_signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating cleanup")
        # This will be called when the signal is received
        # The actual cleanup logic should be implemented by the calling application
        pass
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, cleanup_signal_handler)
    signal.signal(signal.SIGINT, cleanup_signal_handler)
    
    logger.info("Signal handlers registered for SIGTERM and SIGINT")


def cleanup_with_timeout(cleanup_func, timeout: int = 30) -> bool:
    """
    Execute a cleanup function with a timeout.
    
    Args:
        cleanup_func: Function to execute for cleanup
        timeout: Maximum time to wait for cleanup completion
        
    Returns:
        True if cleanup completed within timeout
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cleanup_func)
            try:
                future.result(timeout=timeout)
                logger.info("Cleanup completed within timeout")
                return True
            except FutureTimeoutError:
                logger.error(f"Cleanup timed out after {timeout} seconds")
                return False
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False