"""
Signal handling utilities for graceful shutdown.
"""

import signal
import logging
import atexit
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Global cleanup function registry
_cleanup_functions = []


def register_cleanup_function(func: Callable):
    """Register a function to be called during cleanup."""
    global _cleanup_functions
    _cleanup_functions.append(func)
    logger.debug(f"Registered cleanup function: {func.__name__}")


def setup_signal_handlers(custom_handler: Optional[Callable] = None):
    """
    Set up signal handlers for graceful shutdown.
    
    Args:
        custom_handler: Optional custom signal handler function
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        
        # Execute registered cleanup functions
        for cleanup_func in _cleanup_functions:
            try:
                logger.debug(f"Executing cleanup function: {cleanup_func.__name__}")
                cleanup_func()
            except Exception as e:
                logger.error(f"Error in cleanup function {cleanup_func.__name__}: {e}")
        
        # Call custom handler if provided
        if custom_handler:
            try:
                custom_handler(signum, frame)
            except Exception as e:
                logger.error(f"Error in custom signal handler: {e}")
        
        logger.info("Graceful shutdown completed")
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Also register with atexit for additional cleanup
    atexit.register(lambda: [func() for func in _cleanup_functions])
    
    logger.info("Signal handlers registered for SIGTERM and SIGINT")


def cleanup_signal_handlers():
    """Reset signal handlers to default behavior."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    logger.info("Signal handlers reset to default")