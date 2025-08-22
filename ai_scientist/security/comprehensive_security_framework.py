"""
Comprehensive Security Validation Framework

Advanced security framework addressing critical vulnerabilities identified in analysis.
Provides sandboxed execution, input validation, and comprehensive threat detection.
"""

import os
import re
import ast
import sys
import time
import uuid
import json
import hmac
import base64
import hashlib
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
import inspect
import importlib.util
from contextlib import contextmanager
import signal

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security assessment levels."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    UNSAFE_EVAL = "unsafe_eval"
    UNSAFE_IMPORT = "unsafe_import"
    SENSITIVE_DATA = "sensitive_data"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    location: str
    remediation: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'threat_type': self.threat_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'location': self.location,
            'remediation': self.remediation,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'context': self.context
        }


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_safe: bool
    security_level: SecurityLevel
    threats: List[SecurityThreat]
    execution_time: float
    validator_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.""" 
        return {
            'is_safe': self.is_safe,
            'security_level': self.security_level.value,
            'threats': [threat.to_dict() for threat in self.threats],
            'execution_time': self.execution_time,
            'validator_version': self.validator_version
        }


class CodeAnalyzer:
    """Advanced static code analysis for security threats."""
    
    def __init__(self):
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
            'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir',
            'open', 'file', 'input', 'raw_input'
        }
        
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle',
            'marshal', 'shelve', 'socket', 'urllib', 'httplib', 'ftplib',
            'smtplib', 'telnetlib', 'webbrowser', '__builtin__', 'builtins'
        }
        
        self.dangerous_attributes = {
            '__class__', '__bases__', '__subclasses__', '__mro__',
            '__globals__', '__dict__', '__code__', '__func__',
            'func_globals', 'func_code', 'gi_frame', 'cr_frame'
        }
        
        # Regex patterns for various threats
        self.threat_patterns = {
            ThreatType.COMMAND_INJECTION: [
                re.compile(r'os\.system\s*\(', re.IGNORECASE),
                re.compile(r'subprocess\.(call|run|Popen)', re.IGNORECASE),
                re.compile(r'shell\s*=\s*True', re.IGNORECASE),
                re.compile(r'os\.popen\s*\(', re.IGNORECASE)
            ],
            ThreatType.PATH_TRAVERSAL: [
                re.compile(r'\.\./', re.IGNORECASE),
                re.compile(r'\.\.\\\\', re.IGNORECASE),
                re.compile(r'%2e%2e%2f', re.IGNORECASE),
                re.compile(r'%252e%252e%252f', re.IGNORECASE)
            ],
            ThreatType.SENSITIVE_DATA: [
                re.compile(r'password\s*=\s*["\'].*["\']', re.IGNORECASE),
                re.compile(r'api_key\s*=\s*["\'].*["\']', re.IGNORECASE),
                re.compile(r'secret\s*=\s*["\'].*["\']', re.IGNORECASE),
                re.compile(r'token\s*=\s*["\'][^"\']{20,}["\']', re.IGNORECASE)
            ]
        }
    
    def analyze_code(self, code: str, context: str = "unknown") -> List[SecurityThreat]:
        """Analyze code for security threats."""
        threats = []
        
        try:
            # Parse the code into AST
            tree = ast.parse(code)
            threats.extend(self._analyze_ast(tree, context))
        except SyntaxError as e:
            threats.append(SecurityThreat(
                threat_type=ThreatType.CODE_INJECTION,
                severity=SecurityLevel.HIGH_RISK,
                description=f"Syntax error in code: {str(e)}",
                location=context,
                remediation="Fix syntax errors in the code",
                confidence=1.0
            ))
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            threats.append(SecurityThreat(
                threat_type=ThreatType.CODE_INJECTION,
                severity=SecurityLevel.MEDIUM_RISK,
                description=f"Code analysis error: {str(e)}",
                location=context,
                remediation="Review code for potential issues",
                confidence=0.7
            ))
        
        # Pattern-based analysis
        threats.extend(self._analyze_patterns(code, context))
        
        return threats
    
    def _analyze_ast(self, tree: ast.AST, context: str) -> List[SecurityThreat]:
        """Analyze AST for security threats."""
        threats = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                threats.extend(self._analyze_function_call(node, context))
            elif isinstance(node, ast.Import):
                threats.extend(self._analyze_import(node, context))
            elif isinstance(node, ast.ImportFrom):
                threats.extend(self._analyze_import_from(node, context))
            elif isinstance(node, ast.Attribute):
                threats.extend(self._analyze_attribute_access(node, context))
            elif isinstance(node, ast.Str):
                threats.extend(self._analyze_string_literal(node, context))
        
        return threats
    
    def _analyze_function_call(self, node: ast.Call, context: str) -> List[SecurityThreat]:
        """Analyze function calls for dangerous operations."""
        threats = []
        
        # Check for dangerous function names
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.dangerous_functions:
                severity = SecurityLevel.HIGH_RISK if func_name in ['eval', 'exec'] else SecurityLevel.MEDIUM_RISK
                threats.append(SecurityThreat(
                    threat_type=ThreatType.UNSAFE_EVAL,
                    severity=severity,
                    description=f"Dangerous function call: {func_name}()",
                    location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                    remediation=f"Avoid using {func_name}() or implement proper validation",
                    confidence=0.9
                ))
        
        # Check for subprocess calls with shell=True
        elif isinstance(node.func, ast.Attribute):
            if (hasattr(node.func.value, 'id') and 
                node.func.value.id == 'subprocess' and 
                node.func.attr in ['call', 'run', 'Popen']):
                
                # Check for shell=True argument
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            threats.append(SecurityThreat(
                                threat_type=ThreatType.COMMAND_INJECTION,
                                severity=SecurityLevel.HIGH_RISK,
                                description="subprocess call with shell=True",
                                location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                                remediation="Use shell=False and pass arguments as list",
                                confidence=1.0
                            ))
        
        return threats
    
    def _analyze_import(self, node: ast.Import, context: str) -> List[SecurityThreat]:
        """Analyze import statements for dangerous modules."""
        threats = []
        
        for alias in node.names:
            if alias.name in self.dangerous_modules:
                severity = SecurityLevel.HIGH_RISK if alias.name in ['os', 'subprocess'] else SecurityLevel.MEDIUM_RISK
                threats.append(SecurityThreat(
                    threat_type=ThreatType.UNSAFE_IMPORT,
                    severity=severity,
                    description=f"Import of potentially dangerous module: {alias.name}",
                    location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                    remediation=f"Review usage of {alias.name} module for security implications",
                    confidence=0.7
                ))
        
        return threats
    
    def _analyze_import_from(self, node: ast.ImportFrom, context: str) -> List[SecurityThreat]:
        """Analyze from...import statements."""
        threats = []
        
        if node.module and node.module in self.dangerous_modules:
            for alias in node.names:
                threats.append(SecurityThreat(
                    threat_type=ThreatType.UNSAFE_IMPORT,
                    severity=SecurityLevel.MEDIUM_RISK,
                    description=f"Import from dangerous module: from {node.module} import {alias.name}",
                    location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                    remediation=f"Review {alias.name} usage for security implications",
                    confidence=0.7
                ))
        
        return threats
    
    def _analyze_attribute_access(self, node: ast.Attribute, context: str) -> List[SecurityThreat]:
        """Analyze attribute access for dangerous attributes."""
        threats = []
        
        if node.attr in self.dangerous_attributes:
            threats.append(SecurityThreat(
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                severity=SecurityLevel.HIGH_RISK,
                description=f"Access to dangerous attribute: {node.attr}",
                location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                remediation=f"Avoid accessing {node.attr} attribute",
                confidence=0.8
            ))
        
        return threats
    
    def _analyze_string_literal(self, node: ast.Str, context: str) -> List[SecurityThreat]:
        """Analyze string literals for sensitive data."""
        threats = []
        
        # Check for potential sensitive data in strings
        string_value = node.s
        
        # Check for potential API keys, passwords, etc.
        if len(string_value) > 20 and any(keyword in string_value.lower() 
                                        for keyword in ['key', 'token', 'secret', 'password']):
            threats.append(SecurityThreat(
                threat_type=ThreatType.SENSITIVE_DATA,
                severity=SecurityLevel.MEDIUM_RISK,
                description="Potential sensitive data in string literal",
                location=f"{context}:line_{getattr(node, 'lineno', 'unknown')}",
                remediation="Move sensitive data to environment variables or secure storage",
                confidence=0.6
            ))
        
        return threats
    
    def _analyze_patterns(self, code: str, context: str) -> List[SecurityThreat]:
        """Analyze code using regex patterns."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(code)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    threats.append(SecurityThreat(
                        threat_type=threat_type,
                        severity=SecurityLevel.MEDIUM_RISK,
                        description=f"Pattern match for {threat_type.value}: {match.group()}",
                        location=f"{context}:line_{line_num}",
                        remediation=f"Review and sanitize {threat_type.value} pattern",
                        confidence=0.7
                    ))
        
        return threats


class SandboxedExecutor:
    """Sandboxed code execution environment."""
    
    def __init__(self, timeout: float = 30.0, memory_limit: int = 128 * 1024 * 1024):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_imports = {
            'math', 'random', 'datetime', 'json', 'time', 'collections',
            'itertools', 'functools', 'operator', 'string', 're'
        }
        
    def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute code in sandboxed environment."""
        start_time = time.time()
        
        try:
            # Create restricted globals
            restricted_globals = self._create_restricted_globals()
            
            # Add context variables if provided
            if context:
                for key, value in context.items():
                    if self._is_safe_value(value):
                        restricted_globals[key] = value
            
            # Execute with timeout
            result = self._execute_with_timeout(code, restricted_globals)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'memory_usage': self._get_memory_usage(),
                'output': result.get('output', ''),
                'errors': []
            }
            
        except TimeoutError:
            return {
                'success': False,
                'result': None,
                'execution_time': time.time() - start_time,
                'memory_usage': 0,
                'output': '',
                'errors': ['Execution timeout']
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'execution_time': time.time() - start_time,
                'memory_usage': 0,
                'output': '',
                'errors': [str(e)]
            }
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global environment."""
        # Start with minimal builtins
        safe_builtins = {
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'frozenset', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sum', 'min', 'max', 'abs', 'round', 'sorted', 'reversed',
            'all', 'any', 'isinstance', 'issubclass', 'type', 'id',
            'print', 'repr', 'ord', 'chr', 'bin', 'oct', 'hex'
        }
        
        restricted_builtins = {}
        for name in safe_builtins:
            if hasattr(__builtins__, name):
                restricted_builtins[name] = getattr(__builtins__, name)
        
        # Add safe imports
        restricted_globals = {'__builtins__': restricted_builtins}
        
        for module_name in self.allowed_imports:
            try:
                module = __import__(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                pass  # Skip if module not available
        
        return restricted_globals
    
    def _execute_with_timeout(self, code: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code with timeout protection."""
        # This is a simplified timeout implementation
        # In production, you might want to use a more robust solution
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timeout")
        
        # Set up timeout handler (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout))
        
        try:
            # Compile and execute code
            compiled_code = compile(code, '<sandbox>', 'exec')
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            captured_output = io.StringIO()
            
            sys.stdout = captured_output
            sys.stderr = captured_output
            
            try:
                exec(compiled_code, globals_dict)
                output = captured_output.getvalue()
                
                return {
                    'output': output,
                    'globals': {k: v for k, v in globals_dict.items() 
                              if not k.startswith('__') and self._is_safe_value(v)}
                }
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        finally:
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def _is_safe_value(self, value: Any) -> bool:
        """Check if a value is safe to include in sandbox."""
        # Allow basic types
        if isinstance(value, (str, int, float, bool, list, dict, tuple, set)):
            return True
        
        # Allow None
        if value is None:
            return True
        
        # Reject functions, modules, classes
        if callable(value) or inspect.ismodule(value) or inspect.isclass(value):
            return False
        
        return False
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0


class SecurityValidator:
    """Main security validation engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.code_analyzer = CodeAnalyzer()
        self.sandboxed_executor = SandboxedExecutor(
            timeout=self.config.get('execution_timeout', 30.0),
            memory_limit=self.config.get('memory_limit', 128 * 1024 * 1024)
        )
        
        # Security thresholds
        self.security_thresholds = {
            SecurityLevel.SAFE: 0,
            SecurityLevel.LOW_RISK: 2,
            SecurityLevel.MEDIUM_RISK: 5,
            SecurityLevel.HIGH_RISK: 10,
            SecurityLevel.CRITICAL: 20
        }
        
        self.validation_history: List[ValidationResult] = []
    
    def validate_code(self, code: str, context: str = "unknown", 
                     allow_execution: bool = False) -> ValidationResult:
        """Comprehensive code validation."""
        start_time = time.time()
        
        # Static analysis
        threats = self.code_analyzer.analyze_code(code, context)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(threats)
        security_level = self._determine_security_level(risk_score)
        
        # Determine if code is safe to execute
        is_safe = security_level in [SecurityLevel.SAFE, SecurityLevel.LOW_RISK]
        
        # Execution analysis (if allowed and safe)
        if allow_execution and is_safe:
            execution_result = self.sandboxed_executor.execute_code(code)
            if not execution_result['success']:
                # Add execution errors as threats
                for error in execution_result['errors']:
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.CODE_INJECTION,
                        severity=SecurityLevel.MEDIUM_RISK,
                        description=f"Execution error: {error}",
                        location=context,
                        remediation="Fix code execution errors",
                        confidence=0.8
                    ))
                is_safe = False
                security_level = SecurityLevel.MEDIUM_RISK
        
        execution_time = time.time() - start_time
        
        result = ValidationResult(
            is_safe=is_safe,
            security_level=security_level,
            threats=threats,
            execution_time=execution_time
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        logger.info(f"Code validation completed: {security_level.value}, "
                   f"{len(threats)} threats, {execution_time:.3f}s")
        
        return result
    
    def validate_file(self, file_path: str, allow_execution: bool = False) -> ValidationResult:
        """Validate a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.validate_code(code, file_path, allow_execution)
        
        except Exception as e:
            threats = [SecurityThreat(
                threat_type=ThreatType.FILE_SYSTEM_ACCESS,
                severity=SecurityLevel.MEDIUM_RISK,
                description=f"Error reading file: {str(e)}",
                location=file_path,
                remediation="Check file permissions and path",
                confidence=1.0
            )]
            
            return ValidationResult(
                is_safe=False,
                security_level=SecurityLevel.MEDIUM_RISK,
                threats=threats,
                execution_time=0.0
            )
    
    def validate_directory(self, directory_path: str, 
                          file_patterns: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """Validate all Python files in a directory."""
        if file_patterns is None:
            file_patterns = ['*.py']
        
        results = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            return results
        
        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                try:
                    result = self.validate_file(str(file_path))
                    results[str(file_path)] = result
                except Exception as e:
                    logger.error(f"Error validating {file_path}: {e}")
        
        return results
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score from threats."""
        if not threats:
            return 0.0
        
        severity_weights = {
            SecurityLevel.SAFE: 0,
            SecurityLevel.LOW_RISK: 1,
            SecurityLevel.MEDIUM_RISK: 3,
            SecurityLevel.HIGH_RISK: 7,
            SecurityLevel.CRITICAL: 15
        }
        
        total_score = 0.0
        for threat in threats:
            weight = severity_weights.get(threat.severity, 1)
            total_score += weight * threat.confidence
        
        return total_score
    
    def _determine_security_level(self, risk_score: float) -> SecurityLevel:
        """Determine security level from risk score."""
        if risk_score == 0:
            return SecurityLevel.SAFE
        elif risk_score <= 2:
            return SecurityLevel.LOW_RISK
        elif risk_score <= 5:
            return SecurityLevel.MEDIUM_RISK
        elif risk_score <= 10:
            return SecurityLevel.HIGH_RISK
        else:
            return SecurityLevel.CRITICAL
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {}
        
        # Calculate statistics
        total_validations = len(self.validation_history)
        safe_validations = sum(1 for r in self.validation_history if r.is_safe)
        
        # Security level distribution
        level_counts = {}
        for level in SecurityLevel:
            level_counts[level.value] = sum(1 for r in self.validation_history 
                                          if r.security_level == level)
        
        # Threat type distribution
        threat_counts = {}
        for result in self.validation_history:
            for threat in result.threats:
                threat_type = threat.threat_type.value
                threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        # Average execution time
        avg_execution_time = sum(r.execution_time for r in self.validation_history) / total_validations
        
        return {
            'total_validations': total_validations,
            'safe_validations': safe_validations,
            'safety_rate': safe_validations / total_validations,
            'security_level_distribution': level_counts,
            'threat_type_distribution': threat_counts,
            'average_execution_time': avg_execution_time,
            'recent_validations': [r.to_dict() for r in self.validation_history[-10:]]
        }


class InputValidator:
    """Input validation for various data types."""
    
    def __init__(self):
        self.validation_rules = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'),
            'path': re.compile(r'^[a-zA-Z0-9._/-]+$')
        }
    
    def validate_input(self, input_data: Any, input_type: str, 
                      max_length: Optional[int] = None) -> Tuple[bool, str]:
        """Validate input data."""
        try:
            # Convert to string for validation
            input_str = str(input_data)
            
            # Check length
            if max_length and len(input_str) > max_length:
                return False, f"Input exceeds maximum length of {max_length}"
            
            # Check for null bytes
            if '\x00' in input_str:
                return False, "Input contains null bytes"
            
            # Check for path traversal
            if '../' in input_str or '..\\' in input_str:
                return False, "Input contains path traversal sequences"
            
            # Type-specific validation
            if input_type in self.validation_rules:
                if not self.validation_rules[input_type].match(input_str):
                    return False, f"Input does not match {input_type} format"
            
            # Check for common injection patterns
            injection_patterns = [
                r'<script.*?>.*?</script>',  # XSS
                r'javascript:',             # JavaScript injection
                r'vbscript:',              # VBScript injection
                r'on\w+\s*=',              # Event handlers
                r'(union|select|insert|update|delete|drop)\s+',  # SQL injection
                r'(eval|exec|system|shell_exec)\s*\(',  # Code injection
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return False, f"Input contains potential injection pattern: {pattern}"
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data."""
        # Remove null bytes
        sanitized = input_data.replace('\x00', '')
        
        # Remove or escape dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\n', '\r', '\t']
        escape_map = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char in dangerous_chars:
            if char in escape_map:
                sanitized = sanitized.replace(char, escape_map[char])
            else:
                sanitized = sanitized.replace(char, '')
        
        return sanitized


class SecureFileHandler:
    """Secure file handling with path validation."""
    
    def __init__(self, allowed_directories: Optional[List[str]] = None):
        self.allowed_directories = allowed_directories or []
        self.max_file_size = 100 * 1024 * 1024  # 100MB default
        
    def validate_path(self, file_path: str) -> Tuple[bool, str]:
        """Validate file path for security."""
        try:
            # Resolve absolute path
            abs_path = os.path.abspath(file_path)
            
            # Check for path traversal
            if '..' in file_path:
                return False, "Path contains traversal sequences"
            
            # Check against allowed directories
            if self.allowed_directories:
                allowed = False
                for allowed_dir in self.allowed_directories:
                    abs_allowed = os.path.abspath(allowed_dir)
                    if abs_path.startswith(abs_allowed):
                        allowed = True
                        break
                
                if not allowed:
                    return False, "Path not in allowed directories"
            
            # Check for dangerous file extensions
            dangerous_extensions = ['.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.js']
            if any(abs_path.lower().endswith(ext) for ext in dangerous_extensions):
                return False, "File extension not allowed"
            
            return True, "Path validation passed"
            
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    def secure_read(self, file_path: str) -> Tuple[bool, Union[str, bytes], str]:
        """Securely read a file."""
        # Validate path
        is_valid, message = self.validate_path(file_path)
        if not is_valid:
            return False, b'', message
        
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return False, b'', f"File too large: {file_size} bytes"
            
            # Read file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            return True, content, "File read successfully"
            
        except Exception as e:
            return False, b'', f"Error reading file: {str(e)}"
    
    def secure_write(self, file_path: str, content: Union[str, bytes]) -> Tuple[bool, str]:
        """Securely write to a file."""
        # Validate path
        is_valid, message = self.validate_path(file_path)
        if not is_valid:
            return False, message
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            mode = 'w' if isinstance(content, str) else 'wb'
            with open(file_path, mode) as f:
                f.write(content)
            
            return True, "File written successfully"
            
        except Exception as e:
            return False, f"Error writing file: {str(e)}"


class ComprehensiveSecurityFramework:
    """Main security framework orchestrating all security components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.validator = SecurityValidator(self.config.get('validator', {}))
        self.input_validator = InputValidator()
        self.file_handler = SecureFileHandler(
            self.config.get('allowed_directories', [])
        )
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        
        # Framework statistics
        self.statistics = {
            'total_validations': 0,
            'blocked_operations': 0,
            'security_events': 0,
            'start_time': time.time()
        }
    
    def validate_and_execute_code(self, code: str, context: str = "unknown",
                                 execution_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate and safely execute code."""
        self.statistics['total_validations'] += 1
        
        # Validate code
        validation_result = self.validator.validate_code(code, context, allow_execution=False)
        
        if not validation_result.is_safe:
            self.statistics['blocked_operations'] += 1
            self._log_security_event('code_blocked', {
                'context': context,
                'security_level': validation_result.security_level.value,
                'threat_count': len(validation_result.threats)
            })
            
            return {
                'success': False,
                'result': None,
                'validation_result': validation_result.to_dict(),
                'message': 'Code blocked due to security concerns'
            }
        
        # Execute code if safe
        if validation_result.security_level == SecurityLevel.SAFE:
            execution_result = self.validator.sandboxed_executor.execute_code(
                code, execution_context
            )
            
            return {
                'success': execution_result['success'],
                'result': execution_result,
                'validation_result': validation_result.to_dict(),
                'message': 'Code executed successfully' if execution_result['success'] else 'Execution failed'
            }
        else:
            return {
                'success': False,
                'result': None,
                'validation_result': validation_result.to_dict(),
                'message': 'Code has security risks and cannot be executed'
            }
    
    def validate_user_input(self, input_data: Any, input_type: str = 'text',
                           max_length: Optional[int] = None) -> Dict[str, Any]:
        """Validate user input."""
        is_valid, message = self.input_validator.validate_input(
            input_data, input_type, max_length
        )
        
        if not is_valid:
            self.statistics['blocked_operations'] += 1
            self._log_security_event('input_blocked', {
                'input_type': input_type,
                'message': message
            })
        
        return {
            'is_valid': is_valid,
            'message': message,
            'sanitized_input': self.input_validator.sanitize_input(str(input_data)) if is_valid else None
        }
    
    def secure_file_operation(self, operation: str, file_path: str, 
                             content: Optional[Union[str, bytes]] = None) -> Dict[str, Any]:
        """Perform secure file operations."""
        if operation == 'read':
            success, content, message = self.file_handler.secure_read(file_path)
        elif operation == 'write':
            if content is None:
                return {'success': False, 'message': 'Content required for write operation'}
            success, message = self.file_handler.secure_write(file_path, content)
            content = None  # Don't return content for write operations
        else:
            return {'success': False, 'message': f'Unknown operation: {operation}'}
        
        if not success:
            self.statistics['blocked_operations'] += 1
            self._log_security_event('file_operation_blocked', {
                'operation': operation,
                'file_path': file_path,
                'message': message
            })
        
        return {
            'success': success,
            'content': content if operation == 'read' else None,
            'message': message
        }
    
    def scan_directory_security(self, directory_path: str) -> Dict[str, Any]:
        """Scan directory for security issues."""
        results = self.validator.validate_directory(directory_path)
        
        # Aggregate results
        total_files = len(results)
        safe_files = sum(1 for r in results.values() if r.is_safe)
        
        threat_summary = {}
        for result in results.values():
            for threat in result.threats:
                threat_type = threat.threat_type.value
                threat_summary[threat_type] = threat_summary.get(threat_type, 0) + 1
        
        return {
            'directory': directory_path,
            'total_files': total_files,
            'safe_files': safe_files,
            'safety_rate': safe_files / total_files if total_files > 0 else 1.0,
            'threat_summary': threat_summary,
            'detailed_results': {path: result.to_dict() for path, result in results.items()}
        }
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for monitoring."""
        if not self.monitoring_enabled:
            return
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        self.statistics['security_events'] += 1
        
        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        logger.warning(f"Security event: {event_type} - {details}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        uptime = time.time() - self.statistics['start_time']
        
        # Get validation statistics
        validation_stats = self.validator.get_validation_statistics()
        
        # Recent security events
        recent_events = self.security_events[-50:] if self.security_events else []
        
        return {
            'framework_statistics': self.statistics,
            'uptime_seconds': uptime,
            'validation_statistics': validation_stats,
            'recent_security_events': recent_events,
            'security_status': 'active' if self.monitoring_enabled else 'inactive',
            'threat_detection_rate': len(self.security_events) / max(1, self.statistics['total_validations']),
            'blocked_operation_rate': self.statistics['blocked_operations'] / max(1, self.statistics['total_validations'])
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize security framework
    config = {
        'monitoring_enabled': True,
        'allowed_directories': ['/tmp', '/var/tmp'],
        'validator': {
            'execution_timeout': 30.0,
            'memory_limit': 128 * 1024 * 1024
        }
    }
    
    security_framework = ComprehensiveSecurityFramework(config)
    
    # Test code validation
    test_codes = [
        ("print('Hello World')", "safe_code"),
        ("import os; os.system('rm -rf /')", "dangerous_code"),
        ("eval(input())", "very_dangerous_code"),
        ("x = 1 + 1; print(x)", "simple_math"),
    ]
    
    print("Testing code validation:")
    for code, description in test_codes:
        result = security_framework.validate_and_execute_code(code, description)
        print(f"\n{description}:")
        print(f"  Success: {result['success']}")
        print(f"  Security Level: {result['validation_result']['security_level']}")
        print(f"  Threats: {len(result['validation_result']['threats'])}")
    
    # Test input validation
    print("\n\nTesting input validation:")
    test_inputs = [
        ("user@example.com", "email"),
        ("../../../etc/passwd", "path"),
        ("<script>alert('xss')</script>", "text"),
        ("normal text input", "text"),
    ]
    
    for input_data, input_type in test_inputs:
        result = security_framework.validate_user_input(input_data, input_type)
        print(f"\nInput: {input_data}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Message: {result['message']}")
    
    # Generate security report
    print("\n\nSecurity Report:")
    report = security_framework.get_security_report()
    print(f"Total validations: {report['framework_statistics']['total_validations']}")
    print(f"Blocked operations: {report['framework_statistics']['blocked_operations']}")
    print(f"Security events: {report['framework_statistics']['security_events']}")
    print(f"Threat detection rate: {report['threat_detection_rate']:.2%}")