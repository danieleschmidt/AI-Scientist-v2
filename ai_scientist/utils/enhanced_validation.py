#!/usr/bin/env python3
"""
Enhanced Validation Framework for AI Scientist v2

Comprehensive validation, error handling, and security framework with 
enterprise-grade reliability features.
"""

import os
import re
import json
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class SecurityRisk(Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    security_risk: SecurityRisk = SecurityRisk.LOW
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FileValidationResult(ValidationResult):
    """Extended validation result for file operations."""
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    permissions: Optional[str] = None
    is_executable: bool = False

class EnhancedValidator:
    """Enterprise-grade validation framework with security focus."""
    
    # Security patterns for dangerous content
    DANGEROUS_PATTERNS = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.call',
        r'subprocess\.run',
        r'subprocess\.Popen',
        r'os\.system',
        r'os\.popen',
        r'import\s+subprocess',
        r'from\s+subprocess\s+import',
        r'__import__\s*\(',
        r'compile\s*\(',
        r'globals\s*\(\s*\)',
        r'locals\s*\(\s*\)',
        r'vars\s*\(\s*\)',
        r'dir\s*\(\s*\)',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'hasattr\s*\(',
        r'delattr\s*\(',
        r'open\s*\(.+[\'"]w',  # Write mode file operations
        r'\.write\s*\(',
        r'\.unlink\s*\(',
        r'\.rmdir\s*\(',
        r'shutil\.rmtree',
        r'rm\s+-rf',  # Shell commands
        r'curl\s+',
        r'wget\s+',
        r'bash\s+',
        r'/bin/',
        r'/usr/bin/',
        r'chmod\s+',
        r'chown\s+',
    ]
    
    # Allowed file extensions for different contexts
    ALLOWED_EXTENSIONS = {
        'config': {'.yaml', '.yml', '.json', '.toml', '.ini'},
        'code': {'.py', '.ipynb', '.r', '.R', '.m', '.jl'},
        'data': {'.csv', '.json', '.parquet', '.h5', '.pkl', '.pickle'},
        'text': {'.md', '.txt', '.rst', '.tex'},
        'image': {'.png', '.jpg', '.jpeg', '.svg', '.pdf'},
    }
    
    # Maximum file sizes (bytes)
    MAX_FILE_SIZES = {
        'config': 1024 * 1024,      # 1MB
        'code': 10 * 1024 * 1024,   # 10MB
        'data': 100 * 1024 * 1024,  # 100MB
        'text': 5 * 1024 * 1024,    # 5MB
        'image': 50 * 1024 * 1024,  # 50MB
    }
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
        
    def validate_api_key(self, key: str, provider: str) -> ValidationResult:
        """Validate API key format and security."""
        try:
            if not key or not isinstance(key, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="API key must be a non-empty string",
                    security_risk=SecurityRisk.MEDIUM
                )
            
            # Demo mode check
            if key.lower() == "demo":
                return ValidationResult(
                    is_valid=True,
                    warnings=["Using demo API key - not suitable for production"],
                    security_risk=SecurityRisk.LOW,
                    metadata={"demo_mode": True}
                )
            
            # Provider-specific validation
            patterns = {
                'openai': r'^sk-[a-zA-Z0-9]{32,}$',
                'anthropic': r'^sk-ant-[a-zA-Z0-9\-]{32,}$',
                'gemini': r'^[a-zA-Z0-9\-_]{32,}$',
            }
            
            provider_lower = provider.lower()
            if provider_lower in patterns:
                if not re.match(patterns[provider_lower], key):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid {provider} API key format",
                        security_risk=SecurityRisk.MEDIUM
                    )
            
            # Security checks
            warnings = []
            security_risk = SecurityRisk.LOW
            
            # Check for obvious test/placeholder keys
            test_patterns = ['test', 'dummy', 'placeholder', 'example', 'fake']
            if any(pattern in key.lower() for pattern in test_patterns):
                warnings.append("API key appears to be a test/placeholder value")
                security_risk = SecurityRisk.MEDIUM
            
            # Check key entropy (basic)
            unique_chars = len(set(key))
            if unique_chars < 10:
                warnings.append("API key has low entropy - may be weak")
                security_risk = max(security_risk, SecurityRisk.MEDIUM)
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                security_risk=security_risk,
                metadata={"provider": provider, "key_length": len(key)}
            )
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"API key validation error: {str(e)}",
                security_risk=SecurityRisk.HIGH
            )
    
    def validate_file_path(self, file_path: Union[str, Path], 
                          expected_type: str = None) -> FileValidationResult:
        """Validate file path with security checks."""
        try:
            path = Path(file_path)
            
            # Path traversal check
            if '..' in str(path) or str(path).startswith('/'):
                if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    return FileValidationResult(
                        is_valid=False,
                        error_message="Potential path traversal detected",
                        security_risk=SecurityRisk.HIGH
                    )
            
            # Check if file exists
            if not path.exists():
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"File does not exist: {file_path}",
                    security_risk=SecurityRisk.LOW
                )
            
            # Get file info
            stat = path.stat()
            file_size = stat.st_size
            permissions = oct(stat.st_mode)[-3:]
            is_executable = os.access(path, os.X_OK)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(path)
            
            # Extension validation
            suffix = path.suffix.lower()
            warnings = []
            security_risk = SecurityRisk.LOW
            
            if expected_type and expected_type in self.ALLOWED_EXTENSIONS:
                allowed_exts = self.ALLOWED_EXTENSIONS[expected_type]
                if suffix not in allowed_exts:
                    if self.validation_level == ValidationLevel.PARANOID:
                        return FileValidationResult(
                            is_valid=False,
                            error_message=f"File extension {suffix} not allowed for type {expected_type}",
                            security_risk=SecurityRisk.MEDIUM
                        )
                    else:
                        warnings.append(f"Unusual file extension {suffix} for type {expected_type}")
                        security_risk = SecurityRisk.LOW
            
            # Size validation
            if expected_type and expected_type in self.MAX_FILE_SIZES:
                max_size = self.MAX_FILE_SIZES[expected_type]
                if file_size > max_size:
                    return FileValidationResult(
                        is_valid=False,
                        error_message=f"File too large: {file_size} bytes (max: {max_size})",
                        security_risk=SecurityRisk.MEDIUM,
                        file_size=file_size,
                        file_hash=file_hash
                    )
            
            # Security checks for executable files
            if is_executable:
                warnings.append("File is executable")
                security_risk = max(security_risk, SecurityRisk.MEDIUM)
            
            return FileValidationResult(
                is_valid=True,
                warnings=warnings,
                security_risk=security_risk,
                file_size=file_size,
                file_hash=file_hash,
                permissions=permissions,
                is_executable=is_executable
            )
            
        except Exception as e:
            logger.error(f"File path validation failed: {e}")
            return FileValidationResult(
                is_valid=False,
                error_message=f"File validation error: {str(e)}",
                security_risk=SecurityRisk.HIGH
            )
    
    def validate_code_content(self, code: str, allow_execution: bool = False) -> ValidationResult:
        """Validate code content for security risks."""
        try:
            if not isinstance(code, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="Code must be a string",
                    security_risk=SecurityRisk.MEDIUM
                )
            
            warnings = []
            security_risk = SecurityRisk.LOW
            
            # Check for dangerous patterns
            dangerous_matches = []
            for pattern in self.compiled_patterns:
                matches = pattern.findall(code)
                dangerous_matches.extend(matches)
            
            if dangerous_matches:
                if not allow_execution or self.validation_level == ValidationLevel.PARANOID:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Potentially dangerous code detected: {dangerous_matches[:3]}",
                        security_risk=SecurityRisk.CRITICAL,
                        metadata={"dangerous_patterns": dangerous_matches}
                    )
                else:
                    warnings.append(f"Potentially risky code patterns found: {len(dangerous_matches)}")
                    security_risk = SecurityRisk.HIGH
            
            # Check for suspicious imports
            import_lines = [line.strip() for line in code.split('\\n') if line.strip().startswith('import') or line.strip().startswith('from')]
            suspicious_imports = []
            dangerous_modules = {'os', 'subprocess', 'sys', 'eval', 'exec', 'compile', 'importlib'}
            
            for line in import_lines:
                for module in dangerous_modules:
                    if module in line:
                        suspicious_imports.append(line)
            
            if suspicious_imports:
                if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    warnings.append(f"Suspicious imports detected: {len(suspicious_imports)}")
                    security_risk = max(security_risk, SecurityRisk.MEDIUM)
            
            # Check code complexity
            lines = len([l for l in code.split('\\n') if l.strip()])
            if lines > 1000:
                warnings.append(f"Large code block ({lines} lines) - consider review")
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                security_risk=security_risk,
                metadata={
                    "lines_of_code": lines,
                    "dangerous_patterns": len(dangerous_matches),
                    "suspicious_imports": suspicious_imports
                }
            )
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Code validation error: {str(e)}",
                security_risk=SecurityRisk.HIGH
            )
    
    def validate_json_data(self, data: Union[str, dict], max_depth: int = 10) -> ValidationResult:
        """Validate JSON data structure and content."""
        try:
            # Parse if string
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid JSON format: {str(e)}",
                        security_risk=SecurityRisk.LOW
                    )
            else:
                parsed_data = data
            
            warnings = []
            security_risk = SecurityRisk.LOW
            
            # Check depth
            actual_depth = self._calculate_json_depth(parsed_data)
            if actual_depth > max_depth:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"JSON too deep ({actual_depth} > {max_depth}) - possible attack",
                    security_risk=SecurityRisk.MEDIUM
                )
            
            # Check for suspicious keys
            suspicious_keys = {'eval', 'exec', '__import__', 'subprocess', 'system'}
            found_suspicious = self._find_suspicious_keys(parsed_data, suspicious_keys)
            
            if found_suspicious:
                warnings.append(f"Suspicious keys found: {found_suspicious}")
                security_risk = SecurityRisk.MEDIUM
            
            # Size check
            json_str = json.dumps(parsed_data) if not isinstance(data, str) else data
            size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            
            if size_mb > 100:  # 100MB limit
                return ValidationResult(
                    is_valid=False,
                    error_message=f"JSON too large: {size_mb:.2f}MB",
                    security_risk=SecurityRisk.MEDIUM
                )
            elif size_mb > 10:  # 10MB warning
                warnings.append(f"Large JSON data: {size_mb:.2f}MB")
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                security_risk=security_risk,
                metadata={
                    "depth": actual_depth,
                    "size_mb": size_mb,
                    "suspicious_keys": found_suspicious
                }
            )
            
        except Exception as e:
            logger.error(f"JSON validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"JSON validation error: {str(e)}",
                security_risk=SecurityRisk.HIGH
            )
    
    def validate_model_name(self, model_name: str) -> ValidationResult:
        """Validate LLM model name format."""
        try:
            if not model_name or not isinstance(model_name, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="Model name must be a non-empty string",
                    security_risk=SecurityRisk.LOW
                )
            
            # Known model patterns
            known_patterns = [
                r'^gpt-[0-9]+(\.?[0-9]*)?(-turbo)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?$',
                r'^claude-[0-9](\.[0-9])?-sonnet(-[0-9]{8})?(-v[0-9]:[0-9])?$',
                r'^gemini-(pro|ultra)(-[0-9]{4}-[0-9]{2}-[0-9]{2})?$',
                r'^o[0-9](-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?$',
            ]
            
            is_known = any(re.match(pattern, model_name, re.IGNORECASE) for pattern in known_patterns)
            
            warnings = []
            if not is_known:
                warnings.append(f"Unknown model name pattern: {model_name}")
            
            # Check for suspicious content
            if any(char in model_name for char in ['<', '>', '{', '}', ';', '|']):
                return ValidationResult(
                    is_valid=False,
                    error_message="Model name contains suspicious characters",
                    security_risk=SecurityRisk.MEDIUM
                )
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                security_risk=SecurityRisk.LOW,
                metadata={"is_known_model": is_known}
            )
            
        except Exception as e:
            logger.error(f"Model name validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Model validation error: {str(e)}",
                security_risk=SecurityRisk.HIGH
            )
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity checking."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception:
            return "unknown"
    
    def _calculate_json_depth(self, obj, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested JSON structure."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _find_suspicious_keys(self, obj, suspicious_keys: set, found: List[str] = None) -> List[str]:
        """Find suspicious keys in nested JSON structure."""
        if found is None:
            found = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in suspicious_keys:
                    found.append(key)
                self._find_suspicious_keys(value, suspicious_keys, found)
        elif isinstance(obj, list):
            for item in obj:
                self._find_suspicious_keys(item, suspicious_keys, found)
        
        return found

class RobustErrorHandler:
    """Enhanced error handling with recovery strategies."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_counts = {}
        
    def with_retry(self, operation_name: str = None):
        """Decorator for automatic retry with exponential backoff."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Track error counts
                        error_key = f"{func.__name__}:{type(e).__name__}"
                        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                        
                        if attempt < self.max_retries:
                            wait_time = (self.backoff_factor ** attempt)
                            logger.warning(f"Attempt {attempt + 1} failed for {operation_name or func.__name__}: {e}. Retrying in {wait_time}s...")
                            import time
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {self.max_retries + 1} attempts failed for {operation_name or func.__name__}")
                
                raise last_exception
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return dict(self.error_counts)
    
    def reset_error_counts(self):
        """Reset error statistics."""
        self.error_counts.clear()

# Global instances
enhanced_validator = EnhancedValidator()
robust_error_handler = RobustErrorHandler()

# Convenience functions
def validate_api_key(key: str, provider: str) -> ValidationResult:
    """Validate API key with enhanced security checks."""
    return enhanced_validator.validate_api_key(key, provider)

def validate_file_path(file_path: Union[str, Path], expected_type: str = None) -> FileValidationResult:
    """Validate file path with security and integrity checks."""
    return enhanced_validator.validate_file_path(file_path, expected_type)

def validate_code_content(code: str, allow_execution: bool = False) -> ValidationResult:
    """Validate code content for security risks."""
    return enhanced_validator.validate_code_content(code, allow_execution)

def validate_json_data(data: Union[str, dict], max_depth: int = 10) -> ValidationResult:
    """Validate JSON data structure and content."""
    return enhanced_validator.validate_json_data(data, max_depth)

def validate_model_name(model_name: str) -> ValidationResult:
    """Validate LLM model name format."""
    return enhanced_validator.validate_model_name(model_name)