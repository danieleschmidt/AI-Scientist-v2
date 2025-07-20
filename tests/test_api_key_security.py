"""
Test suite for API key security and validation
"""
import unittest
import os
import sys
from unittest.mock import patch, Mock
import tempfile

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAPIKeyValidation(unittest.TestCase):
    """Test API key validation and security handling"""
    
    def setUp(self):
        """Set up test environment"""
        # Store original environment variables
        self.original_env = dict(os.environ)
        
        # Clear all API key environment variables for consistent testing
        api_keys = [
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HUGGINGFACE_API_KEY',
            'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'GEMINI_API_KEY', 'S2_API_KEY'
        ]
        for key in api_keys:
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_api_key_validation_utility(self):
        """Test a utility function for secure API key validation"""
        # This tests the pattern we want to implement
        def get_api_key_secure(key_name, required=True):
            """Secure API key retrieval with validation"""
            value = os.environ.get(key_name)
            if required and not value:
                raise ValueError(f"Required environment variable {key_name} is not set or empty")
            if value and not value.strip():
                raise ValueError(f"Environment variable {key_name} is empty or contains only whitespace")
            return value.strip() if value else None
        
        # Test missing required key
        with self.assertRaises(ValueError) as context:
            get_api_key_secure("NONEXISTENT_KEY", required=True)
        self.assertIn("NONEXISTENT_KEY", str(context.exception))
        
        # Test empty key
        os.environ["EMPTY_KEY"] = ""
        with self.assertRaises(ValueError) as context:
            get_api_key_secure("EMPTY_KEY", required=True)
        self.assertIn("empty", str(context.exception).lower())
        
        # Test whitespace-only key
        os.environ["WHITESPACE_KEY"] = "   "
        with self.assertRaises(ValueError) as context:
            get_api_key_secure("WHITESPACE_KEY", required=True)
        self.assertIn("whitespace", str(context.exception).lower())
        
        # Test valid key
        os.environ["VALID_KEY"] = "valid-api-key-123"
        result = get_api_key_secure("VALID_KEY")
        self.assertEqual(result, "valid-api-key-123")
        
        # Test optional missing key
        result = get_api_key_secure("OPTIONAL_KEY", required=False)
        self.assertIsNone(result)
    
    def test_api_key_patterns_security(self):
        """Test for potential API key exposure patterns"""
        # Test that API keys are not logged or exposed
        test_api_key = "test-api-key-12345"
        
        # Simulate logging function that should NOT expose keys
        def safe_log_api_key_usage(key_name):
            """Example of safe logging that doesn't expose the key"""
            key_value = os.environ.get(key_name)
            if key_value:
                # Safe: log that key exists, not the value
                masked_key = f"{key_value[:4]}..." if len(key_value) > 4 else "***"
                return f"Using {key_name}: {masked_key}"
            else:
                return f"{key_name} not configured"
        
        os.environ["TEST_API_KEY"] = test_api_key
        log_message = safe_log_api_key_usage("TEST_API_KEY")
        
        # Verify the actual key is not in the log message
        self.assertNotIn(test_api_key, log_message)
        # But verify it acknowledges the key exists
        self.assertIn("test...", log_message)
    
    def test_semantic_scholar_key_handling(self):
        """Test semantic scholar key handling (which already uses getenv)"""
        # This should work fine with missing key (optional)
        s2_key = os.environ.get("S2_API_KEY")
        self.assertIsNone(s2_key)  # Should be None, not raise exception
        
        # Should work with valid key
        os.environ["S2_API_KEY"] = "test-s2-key"
        s2_key = os.environ.get("S2_API_KEY")
        self.assertEqual(s2_key, "test-s2-key")
    
    def test_client_creation_security(self):
        """Test that client creation handles missing keys gracefully"""
        # This simulates what should happen when we fix the llm.py file
        def create_client_secure(model):
            """Secure version of create_client with proper validation"""
            def get_required_api_key(key_name):
                value = os.environ.get(key_name)
                if not value:
                    raise ValueError(f"Required API key {key_name} is not set. Please set this environment variable.")
                return value.strip()
            
            if model == "deepseek-coder-v2-0724":
                api_key = get_required_api_key("DEEPSEEK_API_KEY")
                return {"api_key": api_key, "base_url": "https://api.deepseek.com"}
            
            elif model == "deepcoder-14b":
                api_key = get_required_api_key("HUGGINGFACE_API_KEY")
                return {"api_key": api_key, "base_url": "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview"}
            
            else:
                raise ValueError(f"Unknown model: {model}")
        
        # Test missing API key
        with self.assertRaises(ValueError) as context:
            create_client_secure("deepseek-coder-v2-0724")
        self.assertIn("DEEPSEEK_API_KEY", str(context.exception))
        
        # Test valid API key
        os.environ["DEEPSEEK_API_KEY"] = "valid-deepseek-key"
        client_config = create_client_secure("deepseek-coder-v2-0724")
        self.assertEqual(client_config["api_key"], "valid-deepseek-key")


class TestAPIKeyExposurePrevention(unittest.TestCase):
    """Test prevention of API key exposure in logs and errors"""
    
    def test_error_messages_dont_expose_keys(self):
        """Test that error messages don't accidentally expose API keys"""
        # This is a common mistake - including the actual key in error messages
        def bad_validation(key_name):
            """Example of BAD key validation that exposes the key"""
            key = os.environ.get(key_name)
            if not key:
                raise ValueError(f"Key {key_name} with value '{key}' is invalid")
            return key
        
        def good_validation(key_name):
            """Example of GOOD key validation that doesn't expose the key"""
            key = os.environ.get(key_name)
            if not key:
                raise ValueError(f"Environment variable {key_name} is not set or empty")
            return key
        
        # Test that our good validation doesn't expose the missing key value
        with self.assertRaises(ValueError) as context:
            good_validation("MISSING_KEY")
        
        # Should mention the key name but not expose any value
        error_msg = str(context.exception)
        self.assertIn("MISSING_KEY", error_msg)
        self.assertNotIn("None", error_msg)  # Shouldn't expose the None value
    
    def test_api_key_length_validation(self):
        """Test validation of API key format/length"""
        def validate_api_key_format(key_value, key_name):
            """Validate API key format without exposing the actual key"""
            if not key_value:
                raise ValueError(f"{key_name} is required")
            
            # Basic format validation
            if len(key_value) < 8:
                raise ValueError(f"{key_name} appears to be too short (minimum 8 characters)")
            
            if not key_value.replace('-', '').replace('_', '').isalnum():
                raise ValueError(f"{key_name} contains invalid characters")
            
            return True
        
        # Test valid key
        self.assertTrue(validate_api_key_format("valid-api-key-123", "TEST_KEY"))
        
        # Test invalid keys
        with self.assertRaises(ValueError):
            validate_api_key_format("short", "TEST_KEY")
        
        with self.assertRaises(ValueError):
            validate_api_key_format("invalid@key#", "TEST_KEY")


if __name__ == '__main__':
    unittest.main()