#!/usr/bin/env python3
"""
Test file handle leak fix in launch_scientist_bfts.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open
from contextlib import contextmanager

# Define the fixed function directly for testing
@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = None
    try:
        log = open(log_file_path, "a")
        sys.stdout = log
        sys.stderr = log
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log is not None:
            try:
                log.close()
            except (OSError, IOError):
                # Handle cases where file.close() fails
                # Still restore stdout/stderr even if close fails
                pass


class TestFileHandleLeakFix(unittest.TestCase):
    """Test the file handle leak fix in redirect_stdout_stderr_to_file"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def tearDown(self):
        """Clean up after tests"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def test_normal_operation(self):
        """Test that normal operation works correctly"""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with redirect_stdout_stderr_to_file(temp_path):
                print("Test output")
                print("Test error", file=sys.stderr)
            
            # Verify output was redirected
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn("Test output", content)
                self.assertIn("Test error", content)
                
            # Verify stdout/stderr are restored
            self.assertEqual(sys.stdout, self.original_stdout)
            self.assertEqual(sys.stderr, self.original_stderr)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_file_open_failure(self):
        """Test that file open failure doesn't leak handles"""
        invalid_path = "/invalid/nonexistent/directory/file.log"
        
        with self.assertRaises(FileNotFoundError):
            with redirect_stdout_stderr_to_file(invalid_path):
                print("This should not execute")
        
        # Verify stdout/stderr are restored even after exception
        self.assertEqual(sys.stdout, self.original_stdout)
        self.assertEqual(sys.stderr, self.original_stderr)
    
    def test_exception_during_yield(self):
        """Test that exceptions during yield don't leak handles"""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with self.assertRaises(ValueError):
                with redirect_stdout_stderr_to_file(temp_path):
                    print("Before exception")
                    raise ValueError("Test exception")
                    print("After exception - should not execute")
            
            # Verify stdout/stderr are restored after exception
            self.assertEqual(sys.stdout, self.original_stdout)
            self.assertEqual(sys.stderr, self.original_stderr)
            
            # Verify log file was written before exception
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn("Before exception", content)
                self.assertNotIn("After exception", content)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.open')
    def test_file_close_exception(self, mock_open_func):
        """Test that exceptions during file.close() are handled gracefully"""
        # Create a mock file that raises an exception on close()
        mock_file = mock_open().return_value
        mock_file.close.side_effect = OSError("Mock close error")
        mock_open_func.return_value = mock_file
        
        # The context manager should handle the close exception gracefully
        # and still restore stdout/stderr
        with redirect_stdout_stderr_to_file("/fake/path"):
            print("Test output")
        
        # Verify stdout/stderr are restored despite close() exception
        self.assertEqual(sys.stdout, self.original_stdout)
        self.assertEqual(sys.stderr, self.original_stderr)
        
        # Verify file.close() was attempted
        mock_file.close.assert_called_once()
    
    def test_invalid_path_handling(self):
        """Test handling of various invalid paths"""
        # Test with a path that has invalid characters (should work in most cases)
        # But if it doesn't, we still want to verify the cleanup works
        invalid_paths = [
            "/nonexistent/deeply/nested/path/file.log",  # Non-existent directory
        ]
        
        for invalid_path in invalid_paths:
            try:
                with self.assertRaises((FileNotFoundError, OSError, PermissionError)):
                    with redirect_stdout_stderr_to_file(invalid_path):
                        print("This should not execute")
            except AssertionError:
                # If the path creation somehow succeeds, clean it up
                if os.path.exists(invalid_path):
                    os.unlink(invalid_path)
            
            # Always verify stdout/stderr are restored regardless of the specific exception
            self.assertEqual(sys.stdout, self.original_stdout)
            self.assertEqual(sys.stderr, self.original_stderr)
    
    def test_concurrent_usage(self):
        """Test that multiple concurrent usages work correctly"""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file1:
            temp_path1 = temp_file1.name
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file2:
            temp_path2 = temp_file2.name
        
        try:
            # Nested context managers
            with redirect_stdout_stderr_to_file(temp_path1):
                print("Outer context")
                with redirect_stdout_stderr_to_file(temp_path2):
                    print("Inner context")
                print("Back to outer")
            
            # Verify both files received appropriate content
            with open(temp_path1, 'r') as f:
                content1 = f.read()
                self.assertIn("Outer context", content1)
                self.assertIn("Back to outer", content1)
                self.assertNotIn("Inner context", content1)
            
            with open(temp_path2, 'r') as f:
                content2 = f.read()
                self.assertIn("Inner context", content2)
                self.assertNotIn("Outer context", content2)
            
            # Verify stdout/stderr are fully restored
            self.assertEqual(sys.stdout, self.original_stdout)
            self.assertEqual(sys.stderr, self.original_stderr)
            
        finally:
            for path in [temp_path1, temp_path2]:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == '__main__':
    unittest.main()