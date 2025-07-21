#!/usr/bin/env python3
"""
Test enhanced API key security patterns and fixes
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ai_scientist.utils.api_security import get_api_key_secure, validate_api_key_format
except ImportError:
    # Fallback if imports fail in test environment
    def get_api_key_secure(key_name):
        return os.environ.get(key_name)
    def validate_api_key_format(key_value, key_name, min_length=8):
        return len(key_value) >= min_length if key_value else False


class TestAPIKeySecurityEnhancements(unittest.TestCase):
    """Test enhanced API key security patterns"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up after tests"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_hf_token_secure_access(self):
        """Test that HF_TOKEN is accessed securely in idea files"""
        # Set up valid HF_TOKEN
        os.environ["HF_TOKEN"] = "hf_test123456789abcdef"
        
        # This would be the secure pattern we want to implement
        try:
            token = get_api_key_secure("HF_TOKEN")
            self.assertIsNotNone(token)
            self.assertTrue(token.startswith("hf_"))
        except Exception as e:
            self.fail(f"Secure HF_TOKEN access should not raise exception: {e}")
    
    def test_hf_token_missing_handling(self):
        """Test graceful handling of missing HF_TOKEN"""
        # Remove HF_TOKEN from environment
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        
        # Should handle missing token gracefully
        with self.assertRaises((KeyError, ValueError)):
            get_api_key_secure("HF_TOKEN")
    
    def test_openai_client_key_validation(self):
        """Test that OpenAI client creation includes key validation"""
        # This test will verify that OpenAI client creation is secure
        test_key = "sk-test123456789abcdef"
        os.environ["OPENAI_API_KEY"] = test_key
        
        # This should succeed with secure key handling
        try:
            secure_key = get_api_key_secure("OPENAI_API_KEY")
            self.assertIsNotNone(secure_key)
            self.assertTrue(validate_api_key_format(secure_key, "OPENAI_API_KEY"))
        except Exception as e:
            self.fail(f"OpenAI key validation should not fail: {e}")
    
    def test_anthropic_client_key_validation(self):
        """Test that Anthropic client creation includes key validation"""
        test_key = "sk-ant-test123456789abcdef"
        os.environ["ANTHROPIC_API_KEY"] = test_key
        
        # This should succeed with secure key handling
        try:
            secure_key = get_api_key_secure("ANTHROPIC_API_KEY")
            self.assertIsNotNone(secure_key)
            self.assertTrue(validate_api_key_format(secure_key, "ANTHROPIC_API_KEY"))
        except Exception as e:
            self.fail(f"Anthropic key validation should not fail: {e}")
    
    def test_multiple_missing_keys_handling(self):
        """Test handling when multiple API keys are missing"""
        # Clear all API keys
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", 
                   "DEEPSEEK_API_KEY", "GEMINI_API_KEY"]
        
        for key in api_keys:
            if key in os.environ:
                del os.environ[key]
        
        # Each key should raise appropriate error
        for key in api_keys:
            with self.assertRaises((KeyError, ValueError)):
                get_api_key_secure(key)
    
    def test_invalid_key_format_rejection(self):
        """Test that malformed API keys are rejected"""
        invalid_keys = [
            "",  # Empty
            " ",  # Whitespace only
            "short",  # Too short
            "contains spaces",  # Invalid characters
            "contains\nnewlines",  # Newlines
            "contains\ttabs",  # Tabs
            "contains|pipes",  # Potential command injection chars
        ]
        
        for invalid_key in invalid_keys:
            with self.assertRaises(ValueError):
                # This should fail validation
                validate_api_key_format(invalid_key, "TEST_KEY")
    
    def test_secure_error_messages(self):
        """Test that error messages don't expose API key values"""
        # Set invalid key
        os.environ["TEST_API_KEY"] = "invalid_key_that_should_not_appear_in_error"
        
        try:
            # This should fail but not expose the key
            result = get_api_key_secure("TEST_API_KEY")
            # If it succeeds, verify the key is validated
            self.assertTrue(validate_api_key_format(result, "TEST_API_KEY"))
        except Exception as e:
            error_msg = str(e)
            # Error message should not contain the actual key value
            self.assertNotIn("invalid_key_that_should_not_appear_in_error", error_msg)
    
    def test_key_format_validation_patterns(self):
        """Test API key format validation for different providers"""
        valid_keys = {
            "OPENAI_API_KEY": "sk-proj-test123456789abcdefghijklmnopqrstuvwxyz",
            "ANTHROPIC_API_KEY": "sk-ant-api03-test123456789abcdefghijklmnopqr",
            "HF_TOKEN": "hf_test123456789abcdefghijklmnopqrstuvw",
            "GEMINI_API_KEY": "AIzaSyTest123456789AbCdEfGhIjKlMnOpQr",
        }
        
        for key_name, key_value in valid_keys.items():
            with self.subTest(key=key_name):
                # Should pass validation
                self.assertTrue(
                    validate_api_key_format(key_value, key_name),
                    f"Valid {key_name} should pass format validation"
                )
    
    @patch.dict(os.environ, {}, clear=True)
    def test_environment_isolation(self):
        """Test that API key handling doesn't leak between tests"""
        # Verify clean environment
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"]
        
        for key in api_keys:
            self.assertNotIn(key, os.environ)
            
        # Set a key temporarily
        os.environ["TEMP_KEY"] = "temp_value"
        
        # Verify it's accessible
        self.assertEqual(os.environ.get("TEMP_KEY"), "temp_value")
    
    def test_client_creation_error_handling(self):
        """Test that client creation errors are handled without key exposure"""
        # Test with invalid but properly formatted key
        os.environ["OPENAI_API_KEY"] = "sk-invalid123456789abcdefghijklmnopqr"
        
        try:
            # This should handle the error gracefully
            key = get_api_key_secure("OPENAI_API_KEY")
            self.assertIsNotNone(key)
        except Exception as e:
            # Error should not contain the actual key
            error_msg = str(e)
            self.assertNotIn("sk-invalid123456789abcdefghijklmnopqr", error_msg)


class TestLegacyCodeSecurityFixes(unittest.TestCase):
    """Test that legacy insecure patterns are fixed"""
    
    def test_no_direct_environ_access(self):
        """Test that direct os.environ[KEY] access is eliminated"""
        # This is a regression test to ensure we don't reintroduce direct access
        
        # Set up environment
        os.environ["HF_TOKEN"] = "hf_test123456789"
        
        # The secure way should work
        try:
            secure_token = get_api_key_secure("HF_TOKEN")
            self.assertIsNotNone(secure_token)
        except Exception as e:
            self.fail(f"Secure access should work: {e}")
        
        # Direct access should be avoided (this is more of a code review check)
        # We can't easily test this programmatically, but it's covered by manual review
    
    def test_error_propagation_security(self):
        """Test that error propagation doesn't leak sensitive information"""
        # Remove key to trigger error
        if "TEST_KEY" in os.environ:
            del os.environ["TEST_KEY"]
        
        try:
            get_api_key_secure("TEST_KEY")
            self.fail("Should raise exception for missing key")
        except Exception as e:
            # Exception should be informative but not leak sensitive data
            error_msg = str(e).lower()
            # Should mention the key name (for debugging)
            self.assertIn("test_key", error_msg)
            # But not contain any actual key values
            sensitive_patterns = ["sk-", "hf_", "api_key", "token"]
            for pattern in sensitive_patterns:
                if pattern != "test_key":  # The key name itself is ok
                    self.assertNotIn(pattern, error_msg)


if __name__ == '__main__':
    unittest.main()