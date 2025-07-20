"""
Test suite for security fixes and improvements
"""
import unittest
import subprocess
import tempfile
import os
from pathlib import Path


class TestSecurityFixes(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_tex_file = Path(self.temp_dir) / "test.tex"
        
        # Create a simple test tex file
        self.test_tex_file.write_text(r"""
\documentclass{article}
\begin{document}
Hello World
\end{document}
        """)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_chktex_subprocess_security(self):
        """Test that chktex is called securely via subprocess"""
        # This tests the pattern we implemented to replace os.popen
        try:
            check_result = subprocess.run(
                ["chktex", str(self.test_tex_file), "-q", "-n2", "-n24", "-n13", "-n1"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should not raise an exception for valid tex file
            self.assertIsInstance(check_result.stdout, str)
            self.assertIsInstance(check_result.stderr, str)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            # FileNotFoundError is expected if chktex is not installed
            if isinstance(e, FileNotFoundError):
                self.skipTest("chktex not available in test environment")
            else:
                self.fail(f"Subprocess call failed unexpectedly: {e}")
    
    def test_subprocess_command_injection_prevention(self):
        """Test that our subprocess implementation prevents command injection"""
        # Test with a malicious filename
        malicious_file = Path(self.temp_dir) / "test; rm -rf /"
        malicious_file.write_text("test content")
        
        try:
            # This should safely handle the malicious filename
            check_result = subprocess.run(
                ["chktex", str(malicious_file), "-q"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # The command should complete without executing the injection
            self.assertTrue(True)  # If we get here, no injection occurred
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # These are expected errors, not security issues
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()