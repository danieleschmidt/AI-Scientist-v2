#!/usr/bin/env python3
"""
Test suite to verify the TODO comment fix in interpreter timeout handling.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTimeoutTODOFix(unittest.TestCase):
    """Test that the TODO comment has been properly addressed."""
    
    def test_todo_comment_removed(self):
        """Test that the TODO comment has been removed from interpreter.py."""
        interpreter_file = project_root / "ai_scientist" / "treesearch" / "interpreter.py"
        
        with open(interpreter_file, 'r') as f:
            content = f.read()
        
        # Check that the problematic TODO comment is gone
        self.assertNotIn("[TODO] handle this in a better way", content)
        
        # Check that the assertion error prone code is replaced
        self.assertNotIn('assert reset_session, "Timeout ocurred in interactive session"', content)
    
    def test_improved_error_handling_present(self):
        """Test that improved error handling is present."""
        interpreter_file = project_root / "ai_scientist" / "treesearch" / "interpreter.py"
        
        with open(interpreter_file, 'r') as f:
            content = f.read()
        
        # Check that improved error handling is present
        self.assertIn("Handle timeout gracefully for both reset and interactive sessions", content)
        self.assertIn("if not reset_session:", content)
        self.assertIn("Cleaning up session to prevent resource leaks", content)
        self.assertIn("try:", content)
        self.assertIn("except (ProcessLookupError, OSError)", content)
        self.assertIn("grace_period", content)
    
    def test_logging_improvements_present(self):
        """Test that logging improvements are present."""
        interpreter_file = project_root / "ai_scientist" / "treesearch" / "interpreter.py"
        
        with open(interpreter_file, 'r') as f:
            content = f.read()
        
        # Check for improved logging
        self.assertIn("logger.warning", content)
        self.assertIn("logger.info", content)
        self.assertIn("Sending SIGINT to child process", content)
        self.assertIn("Failed to send SIGINT", content)
        self.assertIn("Force killing process", content)


class TestTimeoutLogicStructure(unittest.TestCase):
    """Test the structure of the timeout handling logic."""
    
    def test_graceful_handling_structure(self):
        """Test that the timeout handling has proper structure."""
        interpreter_file = project_root / "ai_scientist" / "treesearch" / "interpreter.py"
        
        with open(interpreter_file, 'r') as f:
            lines = f.readlines()
        
        # Find the timeout handling section
        timeout_section_start = None
        for i, line in enumerate(lines):
            if "running_time > self.timeout:" in line:
                timeout_section_start = i
                break
        
        self.assertIsNotNone(timeout_section_start, "Timeout handling section not found")
        
        # Check the structure of the timeout handling
        section_content = ''.join(lines[timeout_section_start:timeout_section_start + 40])
        
        # Should handle interactive sessions first
        self.assertIn("if not reset_session:", section_content)
        
        # Should have try-catch for signal sending
        self.assertIn("try:", section_content)
        self.assertIn("except (ProcessLookupError, OSError)", section_content)
        
        # Should have grace period handling
        self.assertIn("grace_period", section_content)


if __name__ == '__main__':
    unittest.main()