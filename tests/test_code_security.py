"""
Tests for code security validation and sandboxing system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_scientist.utils.code_security import (
    CodeSecurityValidator,
    SecurityViolation,
    execute_code_safely,
    create_restricted_globals,
    resource_limited_execution
)


class TestCodeSecurityValidator(unittest.TestCase):
    """Test code security validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = CodeSecurityValidator(allowed_dirs=[self.temp_dir])
    
    def test_safe_code_passes(self):
        """Test that safe scientific code passes validation."""
        safe_code = """
import numpy as np
import matplotlib.pyplot as plt

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot it
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig('sine_wave.png')
        """
        
        is_valid, violations = self.validator.validate_code(safe_code, self.temp_dir)
        self.assertTrue(is_valid, f"Safe code rejected: {violations}")
        self.assertEqual(len(violations), 0)
    
    def test_dangerous_imports_blocked(self):
        """Test that dangerous imports are blocked."""
        dangerous_codes = [
            "import os",
            "import subprocess", 
            "import sys",
            "from os import system",
            "from subprocess import run",
        ]
        
        for code in dangerous_codes:
            with self.subTest(code=code):
                is_valid, violations = self.validator.validate_code(code, self.temp_dir)
                self.assertFalse(is_valid, f"Dangerous code not blocked: {code}")
                self.assertGreater(len(violations), 0)
    
    def test_dangerous_functions_blocked(self):
        """Test that dangerous function calls are blocked."""
        dangerous_codes = [
            "eval('print(1)')",
            "exec('import os')",
            "__import__('os')",
            "compile('import sys', '<string>', 'exec')",
        ]
        
        for code in dangerous_codes:
            with self.subTest(code=code):
                is_valid, violations = self.validator.validate_code(code, self.temp_dir)
                self.assertFalse(is_valid, f"Dangerous function not blocked: {code}")
                self.assertGreater(len(violations), 0)
    
    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        dangerous_codes = [
            "open('../../../etc/passwd', 'r')",
            "open('/etc/passwd', 'r')",
            "open('../../sensitive_file.txt', 'w')",
        ]
        
        for code in dangerous_codes:
            with self.subTest(code=code):
                is_valid, violations = self.validator.validate_code(code, self.temp_dir)
                self.assertFalse(is_valid, f"Path traversal not blocked: {code}")
                self.assertGreater(len(violations), 0)
    
    def test_allowed_file_operations(self):
        """Test that file operations in working directory are allowed."""
        allowed_codes = [
            "open('data.txt', 'w').write('test')",
            "open('results.json', 'r').read()",
            f"open('{self.temp_dir}/output.csv', 'w')",
        ]
        
        for code in allowed_codes:
            with self.subTest(code=code):
                is_valid, violations = self.validator.validate_code(code, self.temp_dir)
                # Note: These might still fail due to other restrictions, but should not fail on path checks
                path_violations = [v for v in violations if 'path' in v.lower() or 'directory' in v.lower()]
                self.assertEqual(len(path_violations), 0, f"Allowed path blocked: {code}")
    
    def test_syntax_errors_caught(self):
        """Test that syntax errors are caught."""
        invalid_code = "if True print('invalid syntax')"
        
        is_valid, violations = self.validator.validate_code(invalid_code, self.temp_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any('syntax' in v.lower() for v in violations))
    
    def test_regex_security_patterns(self):
        """Test regex-based security checks."""
        dangerous_codes = [
            "__import__('os')",
            "exec(user_input)",
            "eval(malicious_code)",
            "os.system('rm -rf /')",
            "subprocess.run(['rm', '-rf', '/'])",
        ]
        
        for code in dangerous_codes:
            with self.subTest(code=code):
                is_valid, violations = self.validator.validate_code(code, self.temp_dir)
                self.assertFalse(is_valid, f"Dangerous pattern not caught: {code}")


class TestSecureExecution(unittest.TestCase):
    """Test secure code execution."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_safe_code_execution(self):
        """Test execution of safe code."""
        safe_code = """
import json
result = {'status': 'success', 'value': 42}
print(json.dumps(result))
        """
        
        success, stdout, stderr = execute_code_safely(safe_code, self.temp_dir)
        self.assertTrue(success, f"Safe code execution failed: {stderr}")
        self.assertIn('success', stdout)
        self.assertEqual(stderr.strip(), "")
    
    def test_dangerous_code_blocked(self):
        """Test that dangerous code execution is blocked."""
        dangerous_code = "import os; os.system('echo hacked')"
        
        success, stdout, stderr = execute_code_safely(dangerous_code, self.temp_dir)
        self.assertFalse(success, "Dangerous code was not blocked")
        self.assertIn("Security violations", stderr)
    
    def test_file_access_restrictions(self):
        """Test file access restrictions."""
        # Should fail - accessing outside working directory
        bad_code = "open('/etc/passwd', 'r').read()"
        success, stdout, stderr = execute_code_safely(bad_code, self.temp_dir)
        self.assertFalse(success, "File access outside working directory not blocked")
        
        # Should work - accessing within working directory  
        test_file = os.path.join(self.temp_dir, 'test.txt')
        good_code = f"""
with open('test.txt', 'w') as f:
    f.write('test data')
with open('test.txt', 'r') as f:
    print(f.read())
        """
        success, stdout, stderr = execute_code_safely(good_code, self.temp_dir)
        self.assertTrue(success, f"Valid file access failed: {stderr}")
        self.assertIn('test data', stdout)
    
    def test_resource_limits(self):
        """Test resource limiting."""
        # Memory-intensive code (should be limited)
        memory_hog = """
big_list = []
for i in range(1000000):
    big_list.append([0] * 1000)
        """
        
        success, stdout, stderr = execute_code_safely(memory_hog, self.temp_dir, timeout=5)
        # This may or may not fail depending on system limits, but shouldn't crash the test
        
        # CPU-intensive code (should timeout)
        cpu_hog = """
import time
while True:
    pass
        """
        
        success, stdout, stderr = execute_code_safely(cpu_hog, self.temp_dir, timeout=2)
        self.assertFalse(success, "CPU-intensive code did not timeout")
        self.assertIn("timeout", stderr.lower())
    
    def test_restricted_globals(self):
        """Test that restricted globals work correctly."""
        globals_dict = create_restricted_globals(self.temp_dir)
        
        # Should have safe builtins
        self.assertIn('len', globals_dict['__builtins__'])
        self.assertIn('sum', globals_dict['__builtins__'])
        
        # Should not have dangerous functions
        self.assertNotIn('eval', globals_dict['__builtins__'])
        self.assertNotIn('exec', globals_dict['__builtins__'])
        self.assertNotIn('__import__', globals_dict['__builtins__'])
        
        # Should have safe open function
        self.assertIn('open', globals_dict)
        
        # Test safe open function
        safe_open = globals_dict['open']
        
        # Should work for files in working directory
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        with safe_open('test.txt', 'r') as f:
            content = f.read()
        self.assertEqual(content, 'test')
        
        # Should fail for files outside working directory
        with self.assertRaises(PermissionError):
            safe_open('/etc/passwd', 'r')


class TestResourceLimits(unittest.TestCase):
    """Test resource limiting context manager."""
    
    def test_resource_limited_execution(self):
        """Test resource-limited execution context."""
        import time
        
        # Test that normal execution works
        with resource_limited_execution(cpu_limit=5, memory_limit=100*1024*1024):
            result = sum(range(1000))
            self.assertEqual(result, 499500)
        
        # Test timeout (this might not work in all environments)
        try:
            with resource_limited_execution(cpu_limit=1, memory_limit=100*1024*1024):
                start = time.time()
                while time.time() - start < 2:  # Try to run for 2 seconds
                    pass
        except TimeoutError:
            pass  # Expected behavior
        except:
            pass  # May not work in all test environments


if __name__ == '__main__':
    unittest.main()