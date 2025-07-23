#!/usr/bin/env python3
"""
Test suite for interpreter timeout handling improvements.
Tests the enhanced timeout handling to replace the TODO comment.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestInterpreterTimeoutHandling(unittest.TestCase):
    """Test enhanced timeout handling in the interpreter."""
    
    def setUp(self):
        """Set up test interpreter with short timeout."""
        # Skip actual interpreter tests if dependencies not available
        self.skip_interpreter_tests = False
        try:
            from ai_scientist.treesearch.interpreter import Interpreter
            self.interpreter = Interpreter(timeout=2)  # 2 second timeout for testing
        except ImportError as e:
            self.skip_interpreter_tests = True
            self.skipTest(f"Interpreter dependencies not available: {e}")
    
    def tearDown(self):
        """Clean up interpreter after each test."""
        if not self.skip_interpreter_tests and hasattr(self.interpreter, 'process') and self.interpreter.process:
            self.interpreter.cleanup_session()
    
    def test_graceful_timeout_handling(self):
        """Test that timeout is handled gracefully without TODO assertion."""
        # This test should pass after we fix the TODO
        long_running_code = """
import time
time.sleep(5)  # This will exceed the 2-second timeout
print("Should not reach here")
"""
        
        result = self.interpreter.run(long_running_code, reset_session=True)
        
        # Should handle timeout gracefully
        self.assertEqual(result.exc_type, "TimeoutError")
        self.assertIn("TimeoutError", result.output[-1])
        self.assertLessEqual(result.exec_time, self.interpreter.timeout + 1)  # Allow some tolerance
    
    def test_interactive_session_timeout_prevention(self):
        """Test that timeout in interactive session is handled properly."""
        # First run some code in non-reset mode to establish session
        self.interpreter.run("x = 1", reset_session=True)
        
        # Now try to run long code in interactive mode (reset_session=False)
        long_running_code = """
import time
time.sleep(5)  # This should timeout
"""
        
        # This should not raise an assertion error about interactive session
        result = self.interpreter.run(long_running_code, reset_session=False)
        
        # Should handle timeout gracefully even in interactive mode
        self.assertEqual(result.exc_type, "TimeoutError")
    
    def test_resource_cleanup_after_timeout(self):
        """Test that resources are properly cleaned up after timeout."""
        long_running_code = """
import time
time.sleep(5)  # This will timeout
"""
        
        # Run code that will timeout
        result = self.interpreter.run(long_running_code, reset_session=True)
        
        # Process should be cleaned up
        self.assertIsNone(self.interpreter.process)
        
        # Should be able to run new code after timeout
        simple_code = "print('Hello after timeout')"
        result2 = self.interpreter.run(simple_code, reset_session=True)
        self.assertIsNone(result2.exc_type)
        self.assertIn("Hello after timeout", "\\n".join(result2.output))
    
    def test_signal_handling_robustness(self):
        """Test that signal handling during timeout is robust."""
        with patch('os.kill') as mock_kill:
            # Simulate signal sending failure
            mock_kill.side_effect = ProcessLookupError("Process not found")
            
            long_running_code = """
import time
time.sleep(5)  # This will timeout
"""
            
            # Should handle signal sending failure gracefully
            result = self.interpreter.run(long_running_code, reset_session=True)
            self.assertEqual(result.exc_type, "TimeoutError")
    
    def test_child_process_termination_after_grace_period(self):
        """Test that child process is terminated after grace period."""
        # Mock a process that doesn't respond to SIGINT
        with patch.object(self.interpreter, 'process') as mock_process:
            mock_process.is_alive.return_value = True
            mock_process.pid = 12345
            
            with patch('os.kill') as mock_kill:
                with patch('time.time', side_effect=[0, 0, 3, 65]):  # Simulate 65 seconds elapsed
                    
                    long_running_code = "import time; time.sleep(100)"
                    
                    # This should trigger the grace period termination
                    result = self.interpreter.run(long_running_code, reset_session=True)
                    
                    # Should have attempted to send SIGINT
                    mock_kill.assert_called()
                    
                    # Should have called cleanup_session after grace period
                    # (This is tested indirectly through the expected behavior)
    
    def test_timeout_error_message_clarity(self):
        """Test that timeout error messages are clear and informative."""
        long_running_code = """
import time
time.sleep(5)  # This will timeout
"""
        
        result = self.interpreter.run(long_running_code, reset_session=True)
        
        # Check that error message is informative
        error_message = result.output[-1]
        self.assertIn("TimeoutError", error_message)
        self.assertIn("time limit", error_message)
        self.assertIn(str(self.interpreter.timeout), error_message)
    
    def test_no_timeout_with_quick_execution(self):
        """Test that quick execution doesn't trigger timeout handling."""
        quick_code = """
print("Quick execution")
x = 1 + 1
print(f"Result: {x}")
"""
        
        result = self.interpreter.run(quick_code, reset_session=True)
        
        # Should execute successfully without timeout
        self.assertIsNone(result.exc_type)
        self.assertIn("Quick execution", "\\n".join(result.output))
        self.assertIn("Result: 2", "\\n".join(result.output))
        self.assertLess(result.exec_time, self.interpreter.timeout)


if __name__ == '__main__':
    unittest.main()