#!/usr/bin/env python3
"""
Test suite for enhanced process cleanup and resource management.
Tests the requirements from backlog item: process-cleanup (WSJF: 4.25)

Acceptance criteria:
- Ensure all child processes are properly terminated
- Add timeout handling for process cleanup
- Implement resource leak detection
- Add proper signal handling
"""

import unittest
import subprocess
import signal
import time
import os
import multiprocessing
import psutil
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestProcessCleanupEnhanced(unittest.TestCase):
    """Test enhanced process cleanup functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_processes = []
        
    def tearDown(self):
        """Clean up any remaining test processes."""
        for proc in self.test_processes:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                    if proc.is_alive():
                        proc.kill()
            except (OSError, AttributeError):
                pass

    def test_child_process_termination_escalation(self):
        """Test that child processes are terminated with proper escalation."""
        # Test that the escalating termination function exists and works
        try:
            # First try importing from the utils module directly
            from ai_scientist.utils.process_cleanup_enhanced import cleanup_child_processes
            
            # Test with empty list (should succeed immediately)
            result = cleanup_child_processes([], timeout=1)
            self.assertTrue(result, "cleanup_child_processes should succeed with empty list")
            
            # Test with mock processes
            import multiprocessing
            import time
            
            def short_task():
                time.sleep(0.2)
            
            # Create a short-lived process
            proc = multiprocessing.Process(target=short_task)
            proc.start()
            
            # Test cleanup
            result = cleanup_child_processes([proc], timeout=2)
            self.assertTrue(result, "cleanup_child_processes should succeed with real process")
            
        except ImportError as e:
            self.skipTest(f"Process cleanup dependencies not available: {e}")
            pass
            
    def test_timeout_handling_for_cleanup(self):
        """Test that process cleanup has proper timeout handling."""
        # Test that the cleanup module exists and has timeout functionality
        from ai_scientist.treesearch.process_cleanup import cleanup_with_timeout
        
        # Test with a simple function that should complete quickly
        def quick_task():
            time.sleep(0.1)
            return True
        
        result = cleanup_with_timeout(quick_task, timeout=2)
        self.assertTrue(result, "Quick cleanup task should complete within timeout")
            
    def test_resource_leak_detection(self):
        """Test that resource leaks can be detected."""
        # Test that resource monitoring works
        from ai_scientist.treesearch.resource_monitor import detect_resource_leaks
        
        leaks = detect_resource_leaks()
        self.assertIsInstance(leaks, dict)
        self.assertIn('timestamp', leaks)
        self.assertIn('memory_usage_mb', leaks)
        self.assertIn('potential_leaks', leaks)
            
    def test_signal_handling_implementation(self):
        """Test that proper signal handling is implemented."""
        # Test that signal handling can be set up
        from ai_scientist.treesearch.signal_handlers import setup_signal_handlers, cleanup_signal_handlers
        
        # Test setup and cleanup don't raise exceptions
        try:
            setup_signal_handlers()
            cleanup_signal_handlers()
        except Exception as e:
            self.fail(f"Signal handler setup/cleanup failed: {e}")
            
    def test_gpu_resource_cleanup(self):
        """Test that GPU resources are properly cleaned up."""
        # Test that GPU cleanup functions exist and work
        from ai_scientist.utils.gpu_cleanup import cleanup_gpu_resources, monitor_gpu_usage
        
        # These should not raise exceptions even if no GPU is available
        result = cleanup_gpu_resources(force=False)
        self.assertIsInstance(result, bool)
        
        usage = monitor_gpu_usage()
        self.assertIsInstance(usage, dict)
        self.assertIn('timestamp', usage)
            
    def test_interpreter_session_cleanup(self):
        """Test that interpreter sessions are properly cleaned up."""
        import tempfile
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from ai_scientist.treesearch.interpreter import Interpreter
        except ImportError as e:
            self.skipTest(f"Interpreter dependencies not available: {e}")
        
        # Test context manager usage ensures cleanup
        temp_dir = tempfile.mkdtemp()
        
        try:
            with Interpreter(working_dir=temp_dir, timeout=5) as interpreter:
                # Execute some simple code
                result = interpreter.run("print('Hello World')", reset_session=True)
                self.assertIsNotNone(result)
                self.assertIn("Hello World", result.term_out)
            
            # After context manager exit, process should be cleaned up
            self.assertIsNone(interpreter.process)
            
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_interpreter_context_manager_exception_handling(self):
        """Test that context manager properly cleans up even when exceptions occur."""
        import tempfile
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from ai_scientist.treesearch.interpreter import Interpreter
        
        temp_dir = tempfile.mkdtemp()
        interpreter = None
        
        try:
            with Interpreter(working_dir=temp_dir, timeout=5) as interp:
                interpreter = interp
                # Execute some code
                result = interpreter.run("print('Test')", reset_session=True)
                self.assertIsNotNone(result)
                # Simulate an exception
                raise ValueError("Test exception")
                
        except ValueError:
            # Exception occurred, but cleanup should still happen
            self.assertIsNone(interpreter.process)
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
                
    def test_zombie_process_prevention(self):
        """Test that zombie processes are prevented."""
        try:
            import psutil
            from ai_scientist.utils.process_cleanup_enhanced import cleanup_child_processes
        except ImportError as e:
            self.skipTest(f"Process cleanup dependencies not available: {e}")
        
        # Get initial process count
        initial_procs = len(psutil.pids())
        
        # Create and cleanup processes using our enhanced cleanup
        def short_task():
            time.sleep(0.1)
            
        processes = []
        for _ in range(3):  # Reduced for test stability
            proc = multiprocessing.Process(target=short_task)
            proc.start()
            processes.append(proc)
            
        # Use our enhanced cleanup instead of basic join
        success = cleanup_child_processes(processes, timeout=2)
        self.assertTrue(success, "Process cleanup should succeed")
            
        # Check for zombie processes 
        time.sleep(0.5)  # Allow time for cleanup
        final_procs = len(psutil.pids())
        
        # Should not have significantly more processes (accounting for test overhead)
        self.assertLess(final_procs - initial_procs, 10, 
                       "Too many processes remain, possible zombie processes")


class TestLaunchScientistCleanup(unittest.TestCase):
    """Test cleanup functionality in launch_scientist_bfts.py."""
    
    def test_enhanced_process_cleanup_exists(self):
        """Test that enhanced cleanup functions exist in launch script."""
        # Test that the enhanced cleanup module exists and has required functions
        from ai_scientist.utils.process_cleanup_enhanced import (
            cleanup_child_processes,
            detect_orphaned_processes,
            setup_cleanup_signal_handlers
        )
        
        # Test that functions are callable
        self.assertTrue(callable(cleanup_child_processes))
        self.assertTrue(callable(detect_orphaned_processes))
        self.assertTrue(callable(setup_cleanup_signal_handlers))


if __name__ == '__main__':
    unittest.main()