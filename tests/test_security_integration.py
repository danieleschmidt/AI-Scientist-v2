#!/usr/bin/env python3
"""
Integration tests for security implementations
"""

import os
import sys
import tempfile
import unittest
import json
import zipfile

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ai_scientist.utils.input_validation import (
        is_code_safe, validate_zip_file, safe_extract_zip,
        validate_json_config, validate_file_path, sanitize_filename,
        SecurityError, ValidationError
    )
    HAS_VALIDATION = True
except ImportError as e:
    print(f"Could not import validation module: {e}")
    HAS_VALIDATION = False


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security implementations"""
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available")
    def test_safe_code_validation(self):
        """Test that safe code passes validation"""
        safe_codes = [
            "result = 2 + 2",
            "import pandas as pd",
            "x = [1, 2, 3]",
            "for i in range(10): print(i)"
        ]
        
        for code in safe_codes:
            with self.subTest(code=code):
                # Should not raise exception
                self.assertTrue(is_code_safe(code))
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available")
    def test_dangerous_code_rejection(self):
        """Test that dangerous code is rejected"""
        dangerous_codes = [
            "import os",
            "exec('malicious')",
            "__import__('subprocess')",
        ]
        
        for code in dangerous_codes:
            with self.subTest(code=code):
                with self.assertRaises(SecurityError):
                    is_code_safe(code)
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available")
    def test_file_path_validation(self):
        """Test file path validation"""
        # Safe paths
        safe_paths = ["data/file.txt", "experiments/result.json"]
        for path in safe_paths:
            with self.subTest(path=path):
                self.assertTrue(validate_file_path(path))
        
        # Dangerous paths
        dangerous_paths = ["../../../etc/passwd", "/etc/shadow"]
        for path in dangerous_paths:
            with self.subTest(path=path):
                with self.assertRaises(SecurityError):
                    validate_file_path(path)
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available") 
    def test_json_config_validation(self):
        """Test JSON configuration validation"""
        # Safe config
        safe_config = {"model": "gpt-4", "temperature": 0.7}
        result = validate_json_config(safe_config)
        self.assertEqual(result, safe_config)
        
        # Dangerous config
        with self.assertRaises(SecurityError):
            validate_json_config({"path": "../../../etc/passwd"})
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available")
    def test_filename_sanitization(self):
        """Test filename sanitization"""
        dangerous_name = "../../../evil<>file|.txt"
        safe_name = sanitize_filename(dangerous_name)
        
        # Should not contain dangerous characters
        self.assertNotIn('..', safe_name)
        self.assertNotIn('<', safe_name)
        self.assertNotIn('>', safe_name)
        self.assertNotIn('|', safe_name)
    
    @unittest.skipUnless(HAS_VALIDATION, "Validation module not available")
    def test_zip_file_security(self):
        """Test zip file security validation"""
        # Create a safe zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip.name, 'w') as zf:
                zf.writestr("safe_file.txt", "safe content")
            
            # Should validate successfully
            self.assertTrue(validate_zip_file(temp_zip.name))
            
            # Clean up
            os.unlink(temp_zip.name)


if __name__ == '__main__':
    unittest.main()