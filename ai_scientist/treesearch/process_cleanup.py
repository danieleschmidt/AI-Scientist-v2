"""
Process cleanup utilities for tree search operations.
"""

from ai_scientist.utils.process_cleanup_enhanced import (
    cleanup_child_processes,
    detect_orphaned_processes,
    setup_cleanup_signal_handlers,
    cleanup_with_timeout
)

# Re-export functions for compatibility
__all__ = [
    'cleanup_child_processes',
    'detect_orphaned_processes', 
    'setup_cleanup_signal_handlers',
    'cleanup_with_timeout'
]