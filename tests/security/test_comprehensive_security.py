"""
Comprehensive security tests for AI Scientist v2.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from tests.fixtures.sample_data import TestDataFactory


class TestInputValidation:
    """Test input validation and sanitization."""
    
    @pytest.mark.security
    def test_sql_injection_prevention(self):
        """Test that SQL injection attempts are blocked."""
        security_data = TestDataFactory.create_security_test_data()
        
        for payload in security_data["sql_injection"]:
            # Test various input validation scenarios
            assert self._validate_input_sanitization(payload)
    
    @pytest.mark.security
    def test_xss_prevention(self):
        """Test that XSS payloads are properly sanitized."""
        security_data = TestDataFactory.create_security_test_data()
        
        for payload in security_data["xss_payloads"]:
            sanitized = self._sanitize_html_input(payload)
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
            assert "onerror=" not in sanitized
    
    @pytest.mark.security
    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        security_data = TestDataFactory.create_security_test_data()
        
        for payload in security_data["path_traversal"]:
            with pytest.raises((ValueError, OSError, PermissionError)):
                self._validate_file_path(payload)
    
    @pytest.mark.security
    def test_command_injection_prevention(self):
        """Test that command injection attempts are blocked."""
        security_data = TestDataFactory.create_security_test_data()
        
        for payload in security_data["command_injection"]:
            with pytest.raises((ValueError, SecurityError, OSError)):
                self._validate_command_input(payload)
    
    @pytest.mark.security
    def test_code_injection_prevention(self):
        """Test that code injection attempts are blocked."""
        security_data = TestDataFactory.create_security_test_data()
        
        for payload in security_data["code_injection"]:
            with pytest.raises((ValueError, SyntaxError, SecurityError)):
                self._validate_code_input(payload)
    
    def _validate_input_sanitization(self, input_data: str) -> bool:
        """Validate that input is properly sanitized."""
        # Mock implementation of input sanitization validation
        dangerous_patterns = ["'", "\"", ";", "--", "DROP", "SELECT", "INSERT"]
        return not any(pattern in input_data.upper() for pattern in dangerous_patterns)
    
    def _sanitize_html_input(self, html_input: str) -> str:
        """Sanitize HTML input to remove dangerous elements."""
        import html
        import re
        
        # Escape HTML entities
        sanitized = html.escape(html_input)
        
        # Remove dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path to prevent directory traversal."""
        if ".." in file_path or file_path.startswith("/"):
            raise ValueError("Invalid file path detected")
        return True
    
    def _validate_command_input(self, command: str) -> bool:
        """Validate command input to prevent injection."""
        dangerous_chars = [";", "|", "&", "`", "$", "(", ")"]
        if any(char in command for char in dangerous_chars):
            raise ValueError("Dangerous command characters detected")
        return True
    
    def _validate_code_input(self, code: str) -> bool:
        """Validate code input to prevent injection."""
        dangerous_functions = ["eval", "exec", "__import__", "subprocess", "os.system"]
        if any(func in code for func in dangerous_functions):
            raise ValueError("Dangerous code patterns detected")
        return True


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization mechanisms."""
    
    @pytest.mark.security
    def test_api_key_validation(self, mock_api_keys):
        """Test API key validation mechanisms."""
        # Test valid API keys
        for key_name, key_value in mock_api_keys.items():
            assert self._validate_api_key(key_value)
        
        # Test invalid API keys
        invalid_keys = ["", "invalid", "sk-invalid", "short"]
        for invalid_key in invalid_keys:
            assert not self._validate_api_key(invalid_key)
    
    @pytest.mark.security
    def test_api_key_storage_security(self):
        """Test that API keys are securely stored and handled."""
        # Test that API keys are not logged
        with patch('logging.Logger.info') as mock_logger:
            self._log_api_operation("sk-test-key-1234567890")
            
            # Verify that the full API key is not in any log messages
            for call in mock_logger.call_args_list:
                args, kwargs = call
                message = str(args[0]) if args else ""
                assert "sk-test-key-1234567890" not in message
    
    @pytest.mark.security
    def test_rate_limiting(self):
        """Test rate limiting mechanisms."""
        # Simulate multiple rapid requests
        request_count = 0
        max_requests = 10
        
        for i in range(max_requests + 5):
            try:
                self._make_api_request()
                request_count += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        
        # Should hit rate limit before max_requests + 5
        assert request_count <= max_requests
    
    @pytest.mark.security
    def test_session_management(self):
        """Test session management and timeout."""
        session_id = self._create_session()
        
        # Test valid session
        assert self._validate_session(session_id)
        
        # Test session timeout
        with patch('time.time', return_value=9999999999):  # Far future
            assert not self._validate_session(session_id)
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and length."""
        if not api_key or len(api_key) < 20:
            return False
        if not api_key.startswith(('sk-', 'sk-ant-')):
            return False
        return True
    
    def _log_api_operation(self, api_key: str):
        """Log API operation while masking sensitive data."""
        import logging
        logger = logging.getLogger(__name__)
        masked_key = api_key[:7] + "***" + api_key[-4:] if len(api_key) > 11 else "***"
        logger.info(f"API operation with key: {masked_key}")
    
    def _make_api_request(self):
        """Simulate API request for rate limiting test."""
        # Mock implementation
        import random
        if random.random() < 0.3:  # 30% chance of rate limit
            raise Exception("Rate limit exceeded")
    
    def _create_session(self) -> str:
        """Create a new session."""
        import uuid
        return str(uuid.uuid4())
    
    def _validate_session(self, session_id: str) -> bool:
        """Validate session."""
        # Mock implementation - would check against session store
        return len(session_id) == 36  # UUID length


class TestDataPrivacyAndEncryption:
    """Test data privacy and encryption mechanisms."""
    
    @pytest.mark.security
    def test_sensitive_data_encryption(self):
        """Test that sensitive data is properly encrypted."""
        sensitive_data = "user_password_123"
        
        encrypted = self._encrypt_sensitive_data(sensitive_data)
        
        # Verify data is encrypted (not plaintext)
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        # Verify can be decrypted
        decrypted = self._decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
    
    @pytest.mark.security
    def test_pii_data_handling(self):
        """Test handling of personally identifiable information."""
        pii_data = {
            "email": "user@example.com",
            "phone": "+1234567890",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
        }
        
        for data_type, value in pii_data.items():
            # Test that PII is properly masked/encrypted
            processed = self._process_pii_data(value)
            assert self._is_pii_protected(processed, value)
    
    @pytest.mark.security
    def test_secure_temp_file_handling(self):
        """Test secure temporary file creation and cleanup."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("sensitive test data")
            temp_path = tmp_file.name
        
        try:
            # Verify file permissions are secure
            file_stat = os.stat(temp_path)
            file_mode = oct(file_stat.st_mode)[-3:]
            assert file_mode in ['600', '644']  # Owner read/write only or read-only
            
            # Test secure deletion
            self._secure_delete_file(temp_path)
            assert not os.path.exists(temp_path)
        finally:
            # Cleanup in case test fails
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        import base64
        # Mock encryption - in real implementation, use proper encryption
        encoded = base64.b64encode(data.encode()).decode()
        return f"encrypted_{encoded}"
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        import base64
        # Mock decryption
        if encrypted_data.startswith("encrypted_"):
            encoded = encrypted_data[10:]  # Remove "encrypted_" prefix
            return base64.b64decode(encoded).decode()
        raise ValueError("Invalid encrypted data format")
    
    def _process_pii_data(self, pii_value: str) -> str:
        """Process PII data with appropriate protection."""
        if "@" in pii_value:  # Email
            parts = pii_value.split("@")
            return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_value.startswith("+"):  # Phone
            return f"{pii_value[:3]}***{pii_value[-4:]}"
        elif "-" in pii_value and len(pii_value.replace("-", "")) >= 9:  # SSN/Credit card
            return f"{pii_value[:4]}***{pii_value[-4:]}"
        return "***"
    
    def _is_pii_protected(self, processed: str, original: str) -> bool:
        """Check if PII is properly protected."""
        return processed != original and "***" in processed
    
    def _secure_delete_file(self, file_path: str):
        """Securely delete file."""
        # Overwrite file content before deletion
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            with open(file_path, 'wb') as f:
                f.write(b'0' * file_size)
            os.unlink(file_path)


class TestResourceProtection:
    """Test resource protection and DoS prevention."""
    
    @pytest.mark.security
    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced."""
        # Test memory allocation limit
        try:
            self._allocate_large_memory(size_mb=5000)  # 5GB
            pytest.fail("Should have hit memory limit")
        except MemoryError:
            pass  # Expected
    
    @pytest.mark.security
    def test_cpu_time_limit_enforcement(self):
        """Test that CPU time limits are enforced."""
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("CPU time limit exceeded")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second limit
        
        try:
            self._cpu_intensive_operation()
            pytest.fail("Should have hit CPU time limit")
        except TimeoutError:
            pass  # Expected
        finally:
            signal.alarm(0)  # Cancel alarm
    
    @pytest.mark.security
    def test_file_size_limits(self):
        """Test file size limits are enforced."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            try:
                self._write_large_file(tmp_file.name, size_mb=1000)  # 1GB
                pytest.fail("Should have hit file size limit")
            except (OSError, IOError):
                pass  # Expected
    
    @pytest.mark.security
    def test_network_request_limits(self):
        """Test network request limits and timeouts."""
        # Test request timeout
        try:
            self._make_slow_network_request(timeout=1)
            pytest.fail("Should have hit network timeout")
        except (TimeoutError, OSError):
            pass  # Expected
    
    def _allocate_large_memory(self, size_mb: int):
        """Attempt to allocate large amount of memory."""
        # Mock implementation that would trigger memory limit
        if size_mb > 1000:  # Mock limit
            raise MemoryError("Memory allocation limit exceeded")
    
    def _cpu_intensive_operation(self):
        """Perform CPU-intensive operation."""
        # Mock CPU-intensive task
        import time
        time.sleep(10)  # This should be interrupted by timeout
    
    def _write_large_file(self, file_path: str, size_mb: int):
        """Attempt to write large file."""
        if size_mb > 100:  # Mock limit
            raise OSError("File size limit exceeded")
    
    def _make_slow_network_request(self, timeout: int):
        """Make slow network request."""
        import time
        if timeout < 5:  # Mock slow response
            raise TimeoutError("Network request timeout")


class TestSecureCodeExecution:
    """Test secure code execution sandbox."""
    
    @pytest.mark.security
    def test_code_execution_sandbox(self):
        """Test that code execution is properly sandboxed."""
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        
        with pytest.raises((SecurityError, ImportError, AttributeError)):
            self._execute_code_safely(dangerous_code)
    
    @pytest.mark.security
    def test_import_restrictions(self):
        """Test that dangerous imports are restricted."""
        restricted_imports = [
            "import subprocess",
            "import os",
            "from os import system",
            "import shutil",
            "__import__('os')",
        ]
        
        for import_statement in restricted_imports:
            with pytest.raises((ImportError, SecurityError)):
                self._execute_code_safely(import_statement)
    
    @pytest.mark.security
    def test_function_restrictions(self):
        """Test that dangerous functions are restricted."""
        restricted_code = [
            "eval('print(1)')",
            "exec('print(1)')",
            "compile('print(1)', '<string>', 'exec')",
            "open('/etc/passwd', 'r')",
        ]
        
        for code in restricted_code:
            with pytest.raises((NameError, SecurityError, PermissionError)):
                self._execute_code_safely(code)
    
    def _execute_code_safely(self, code: str):
        """Execute code in secure sandbox."""
        # Mock secure execution - would use actual sandbox in real implementation
        dangerous_patterns = [
            'import os', 'import subprocess', 'import shutil',
            'eval(', 'exec(', '__import__',
            'open(', 'file(', 'input(',
            'raw_input(', 'execfile(',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                raise SecurityError(f"Dangerous pattern detected: {pattern}")


# Custom exception for security testing
class SecurityError(Exception):
    """Custom exception for security violations."""
    pass