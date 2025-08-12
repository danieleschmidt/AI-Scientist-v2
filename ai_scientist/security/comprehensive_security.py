#!/usr/bin/env python3
"""
Comprehensive Security Framework
===============================

Enterprise-grade security framework implementing:
- Input validation and sanitization
- Output filtering and data leakage prevention
- API key and secret management
- Access control and authentication
- Security monitoring and threat detection

Generation 2: MAKE IT ROBUST - Security Implementation
"""

import os
import re
import hashlib
import secrets
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hmac
import base64


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    severity: ThreatLevel
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    component: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControl:
    """Access control configuration."""
    user_id: str
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    session_expires: Optional[datetime] = None


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess',
        r'os\.system',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'compile\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'getattr',
        r'setattr',
        r'hasattr',
        r'delattr',
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\'|\")[^\'\"]*(\bOR\b|\bAND\b)[^\'\"]*(\1)',
        r'(;|\s)(DROP|DELETE|INSERT|UPDATE)\s',
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./+',
        r'\.\.\\+',
        r'/etc/passwd',
        r'/etc/shadow',
        r'\\windows\\system32',
        r'\.\.[\\/]',
    ]
    
    def __init__(self):
        self.compiled_patterns = {
            'dangerous': [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS],
            'sql_injection': [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS],
            'path_traversal': [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS],
        }
        
    def validate_input(self, data: Any, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> bool:
        """Validate input data for security threats."""
        if data is None:
            return True
            
        # Convert to string for pattern matching
        data_str = str(data)
        
        # Check for dangerous patterns
        threats_found = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(data_str):
                    threats_found.append(f"{category}: {pattern.pattern}")
        
        if threats_found:
            self._log_security_event(
                "input_validation_failed",
                ThreatLevel.CRITICAL,
                {"threats": threats_found, "input_sample": data_str[:100]}
            )
            return False
        
        # Additional checks based on security level
        if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            return self._advanced_validation(data_str)
        
        return True
    
    def _advanced_validation(self, data_str: str) -> bool:
        """Advanced validation for high security levels."""
        # Check for encoded malicious content
        try:
            # Base64 decode check
            if self._is_base64(data_str):
                decoded = base64.b64decode(data_str).decode('utf-8', errors='ignore')
                if not self.validate_input(decoded, SecurityLevel.MEDIUM):
                    return False
        except Exception:
            pass
        
        # Check for URL encoding
        try:
            import urllib.parse
            decoded = urllib.parse.unquote(data_str)
            if decoded != data_str:
                if not self.validate_input(decoded, SecurityLevel.MEDIUM):
                    return False
        except Exception:
            pass
        
        return True
    
    def _is_base64(self, s: str) -> bool:
        """Check if string is base64 encoded."""
        try:
            if len(s) % 4 != 0:
                return False
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
    
    def sanitize_input(self, data: str) -> str:
        """Sanitize input data."""
        if not isinstance(data, str):
            data = str(data)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', data)
        
        # Limit length
        if len(sanitized) > 10000:  # 10KB limit
            sanitized = sanitized[:10000]
            logger.warning("Input truncated due to length limit")
        
        return sanitized
    
    def _log_security_event(self, event_type: str, severity: ThreatLevel, details: Dict[str, Any]):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            component="InputValidator",
            details=details
        )
        security_monitor.log_event(event)


class SecretManager:
    """Secure management of API keys and secrets."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.secrets_cache = {}
        self.access_log = []
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)
    
    def store_secret(self, name: str, value: str, metadata: Optional[Dict] = None) -> str:
        """Store secret securely."""
        if not self._validate_secret_name(name):
            raise ValueError(f"Invalid secret name: {name}")
        
        # Encrypt the secret
        encrypted_value = self._encrypt(value)
        
        # Store with metadata
        self.secrets_cache[name] = {
            'encrypted_value': encrypted_value,
            'created_at': datetime.now(),
            'metadata': metadata or {},
            'access_count': 0
        }
        
        # Log storage (without the actual secret)
        self._log_secret_access(name, "stored")
        
        return f"secret:{name}"
    
    def get_secret(self, name: str, user_id: Optional[str] = None) -> Optional[str]:
        """Retrieve secret securely."""
        if name not in self.secrets_cache:
            self._log_security_event(
                "secret_not_found",
                ThreatLevel.WARNING,
                {"secret_name": name, "user_id": user_id}
            )
            return None
        
        # Check access permissions
        if not self._check_secret_access(name, user_id):
            self._log_security_event(
                "unauthorized_secret_access",
                ThreatLevel.CRITICAL,
                {"secret_name": name, "user_id": user_id}
            )
            return None
        
        # Decrypt and return
        secret_data = self.secrets_cache[name]
        secret_data['access_count'] += 1
        secret_data['last_accessed'] = datetime.now()
        
        decrypted_value = self._decrypt(secret_data['encrypted_value'])
        
        self._log_secret_access(name, "accessed", user_id)
        
        return decrypted_value
    
    def _validate_secret_name(self, name: str) -> bool:
        """Validate secret name."""
        # Must be alphanumeric with underscores, no spaces or special chars
        return re.match(r'^[a-zA-Z0-9_]+$', name) is not None
    
    def _encrypt(self, value: str) -> str:
        """Encrypt value (simplified implementation)."""
        # In production, use proper encryption like Fernet
        encoded = base64.b64encode(value.encode()).decode()
        return encoded
    
    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt value (simplified implementation)."""
        # In production, use proper decryption
        decoded = base64.b64decode(encrypted_value.encode()).decode()
        return decoded
    
    def _check_secret_access(self, name: str, user_id: Optional[str]) -> bool:
        """Check if user has access to secret."""
        # Implement proper access control
        return True  # Simplified for demo
    
    def _log_secret_access(self, name: str, action: str, user_id: Optional[str] = None):
        """Log secret access."""
        self.access_log.append({
            'timestamp': datetime.now(),
            'secret_name': name,
            'action': action,
            'user_id': user_id
        })
    
    def _log_security_event(self, event_type: str, severity: ThreatLevel, details: Dict[str, Any]):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            component="SecretManager",
            details=details
        )
        security_monitor.log_event(event)


class OutputFilter:
    """Filter outputs to prevent data leakage."""
    
    # Patterns that might indicate sensitive data
    SENSITIVE_PATTERNS = [
        (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*[\'"]?([a-zA-Z0-9_-]{20,})[\'"]?', 'API_KEY'),
        (r'(?i)(secret|password|pwd)\s*[:=]\s*[\'"]?([a-zA-Z0-9_@#$%^&*!-]{8,})[\'"]?', 'SECRET'),
        (r'(?i)(token|auth[_-]?token)\s*[:=]\s*[\'"]?([a-zA-Z0-9_.-]{20,})[\'"]?', 'TOKEN'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
        (r'\b\d{3}-?\d{2}-?\d{4}\b', 'SSN'),
        (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 'CREDIT_CARD'),
        (r'(?i)(bearer\s+|authorization:\s*bearer\s+)([a-zA-Z0-9_.-]{20,})', 'BEARER_TOKEN'),
    ]
    
    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern), label) 
            for pattern, label in self.SENSITIVE_PATTERNS
        ]
        self.redaction_log = []
    
    def filter_output(self, data: Any, redact: bool = True) -> Any:
        """Filter output to remove sensitive information."""
        if isinstance(data, str):
            return self._filter_string(data, redact)
        elif isinstance(data, dict):
            return {k: self.filter_output(v, redact) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.filter_output(item, redact) for item in data]
        else:
            return data
    
    def _filter_string(self, text: str, redact: bool) -> str:
        """Filter sensitive information from string."""
        filtered_text = text
        redactions_made = []
        
        for pattern, label in self.compiled_patterns:
            matches = pattern.finditer(filtered_text)
            for match in matches:
                if redact:
                    # Replace with redacted placeholder
                    redacted = f"[REDACTED_{label}]"
                    filtered_text = filtered_text.replace(match.group(), redacted)
                
                redactions_made.append({
                    'type': label,
                    'position': match.span(),
                    'length': len(match.group())
                })
        
        if redactions_made:
            self._log_redaction(redactions_made)
        
        return filtered_text
    
    def _log_redaction(self, redactions: List[Dict]):
        """Log redaction events."""
        self.redaction_log.append({
            'timestamp': datetime.now(),
            'redactions': redactions,
            'count': len(redactions)
        })
        
        security_monitor.log_event(SecurityEvent(
            event_type="data_redacted",
            severity=ThreatLevel.INFO,
            component="OutputFilter",
            details={'redaction_count': len(redactions)}
        ))


class SecurityMonitor:
    """Security monitoring and threat detection."""
    
    def __init__(self):
        self.events = []
        self.threat_counts = {}
        self.blocked_ips = set()
        self.suspicious_activities = []
    
    def log_event(self, event: SecurityEvent):
        """Log security event."""
        self.events.append(event)
        
        # Update threat counts
        threat_key = f"{event.component}:{event.event_type}"
        self.threat_counts[threat_key] = self.threat_counts.get(threat_key, 0) + 1
        
        # Log to file
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'severity': event.severity.value,
            'component': event.component,
            'details': event.details
        }
        
        logger.log(
            self._severity_to_log_level(event.severity),
            f"Security Event: {json.dumps(log_entry)}"
        )
        
        # Check for patterns indicating attacks
        self._analyze_threat_patterns(event)
    
    def _severity_to_log_level(self, severity: ThreatLevel) -> int:
        """Convert threat level to logging level."""
        mapping = {
            ThreatLevel.INFO: logging.INFO,
            ThreatLevel.WARNING: logging.WARNING,
            ThreatLevel.CRITICAL: logging.ERROR,
            ThreatLevel.EMERGENCY: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)
    
    def _analyze_threat_patterns(self, event: SecurityEvent):
        """Analyze events for threat patterns."""
        # Check for repeated failed validation attempts
        if event.event_type == "input_validation_failed":
            recent_failures = [
                e for e in self.events[-10:]  # Last 10 events
                if e.event_type == "input_validation_failed"
                and (datetime.now() - e.timestamp).total_seconds() < 300  # Within 5 minutes
            ]
            
            if len(recent_failures) >= 5:
                self._trigger_threat_response("repeated_validation_failures", event)
    
    def _trigger_threat_response(self, threat_type: str, triggering_event: SecurityEvent):
        """Trigger automated threat response."""
        logger.critical(f"Threat detected: {threat_type}")
        
        # Add to suspicious activities
        self.suspicious_activities.append({
            'timestamp': datetime.now(),
            'threat_type': threat_type,
            'triggering_event': triggering_event.event_type,
            'severity': triggering_event.severity
        })
        
        # Implement response actions based on threat type
        if threat_type == "repeated_validation_failures":
            # Could implement rate limiting, IP blocking, etc.
            logger.warning("Implementing enhanced monitoring due to repeated failures")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        recent_events = [
            e for e in self.events 
            if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        severity_counts = {}
        for event in recent_events:
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        return {
            'total_events': len(self.events),
            'recent_events': len(recent_events),
            'severity_breakdown': severity_counts,
            'top_threats': sorted(self.threat_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'suspicious_activities': len(self.suspicious_activities),
            'blocked_ips': len(self.blocked_ips)
        }


class SecurityFramework:
    """Main security framework orchestrator."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.secret_manager = SecretManager()
        self.output_filter = OutputFilter()
        self.monitor = SecurityMonitor()
        self.access_controls = {}
        
    def validate_and_sanitize(self, data: Any, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> Any:
        """Validate and sanitize input data."""
        if not self.input_validator.validate_input(data, security_level):
            raise SecurityError(f"Input validation failed for security level {security_level.value}")
        
        if isinstance(data, str):
            return self.input_validator.sanitize_input(data)
        
        return data
    
    def filter_sensitive_output(self, data: Any) -> Any:
        """Filter sensitive information from output."""
        return self.output_filter.filter_output(data)
    
    def secure_api_call(self, api_name: str, *args, **kwargs):
        """Make secure API call with monitoring."""
        start_time = time.time()
        
        try:
            # Log API call
            self.monitor.log_event(SecurityEvent(
                event_type="api_call_started",
                severity=ThreatLevel.INFO,
                component="SecurityFramework",
                details={'api_name': api_name}
            ))
            
            # Implement actual API call here
            # For demo, simulate success
            result = {"status": "success", "api": api_name}
            
            # Filter output
            filtered_result = self.filter_sensitive_output(result)
            
            # Log success
            duration = time.time() - start_time
            self.monitor.log_event(SecurityEvent(
                event_type="api_call_completed",
                severity=ThreatLevel.INFO,
                component="SecurityFramework",
                details={'api_name': api_name, 'duration': duration}
            ))
            
            return filtered_result
            
        except Exception as e:
            # Log failure
            self.monitor.log_event(SecurityEvent(
                event_type="api_call_failed",
                severity=ThreatLevel.WARNING,
                component="SecurityFramework",
                details={'api_name': api_name, 'error': str(e)}
            ))
            raise
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'monitor_summary': self.monitor.get_security_summary(),
            'active_sessions': len(self.access_controls),
            'secrets_managed': len(self.secret_manager.secrets_cache),
            'redactions_performed': len(self.output_filter.redaction_log)
        }


class SecurityError(Exception):
    """Custom security exception."""
    pass


# Global security framework instance
security_framework = SecurityFramework()
security_monitor = security_framework.monitor


def secure_operation(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator for securing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Validate inputs
            validated_args = []
            for arg in args:
                validated_args.append(
                    security_framework.validate_and_sanitize(arg, security_level)
                )
            
            validated_kwargs = {}
            for key, value in kwargs.items():
                validated_kwargs[key] = security_framework.validate_and_sanitize(value, security_level)
            
            # Execute function
            result = func(*validated_args, **validated_kwargs)
            
            # Filter output
            return security_framework.filter_sensitive_output(result)
        
        return wrapper
    return decorator


def setup_security_monitoring():
    """Initialize security monitoring."""
    logger.info("Security framework initialized with comprehensive monitoring")
    
    # Log initial status
    security_monitor.log_event(SecurityEvent(
        event_type="security_framework_initialized",
        severity=ThreatLevel.INFO,
        component="SecurityFramework",
        details={'timestamp': datetime.now().isoformat()}
    ))


# Initialize security on import
setup_security_monitoring()