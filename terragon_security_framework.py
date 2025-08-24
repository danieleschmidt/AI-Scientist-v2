#!/usr/bin/env python3
"""
TERRAGON SECURITY FRAMEWORK v2.0

Comprehensive security framework for autonomous research systems including
input validation, API security, file system protection, and threat detection.
"""

import os
import re
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import ast


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    SUSPICIOUS_API_ACTIVITY = "suspicious_api_activity"
    FILE_SYSTEM_VIOLATION = "file_system_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_CODE_DETECTED = "malicious_code_detected"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration_attempt"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source: str
    details: Dict[str, Any]
    blocked: bool = False
    resolved: bool = False


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    # File system security
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.md', '.txt', '.json', '.yaml', '.yml', '.py', '.csv', '.log'
    })
    blocked_file_extensions: Set[str] = field(default_factory=lambda: {
        '.exe', '.bat', '.sh', '.cmd', '.scr', '.vbs', '.js', '.jar'
    })
    max_file_size_mb: int = 100
    
    # Path security
    allowed_paths: Set[str] = field(default_factory=lambda: {
        '/tmp', '/var/tmp', './research_output', './autonomous_research_output'
    })
    blocked_paths: Set[str] = field(default_factory=lambda: {
        '/etc', '/usr', '/bin', '/sbin', '/sys', '/proc', '/root'
    })
    
    # API security
    api_rate_limit_per_minute: int = 100
    api_timeout_seconds: int = 30
    max_concurrent_requests: int = 10
    
    # Code execution security
    allow_code_execution: bool = False
    allowed_imports: Set[str] = field(default_factory=lambda: {
        'json', 'yaml', 'datetime', 'pathlib', 'typing', 'dataclasses',
        'numpy', 'pandas', 'matplotlib', 'torch', 'transformers'
    })
    blocked_functions: Set[str] = field(default_factory=lambda: {
        'exec', 'eval', 'compile', '__import__', 'open', 'input',
        'subprocess.run', 'subprocess.call', 'os.system', 'os.popen'
    })
    
    # Data protection
    enable_data_encryption: bool = True
    enable_api_key_validation: bool = True
    log_sensitive_data: bool = False


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.dangerous_patterns = [
            r'__import__\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'execfile\s*\(',
            r'\.read\s*\(',
            r'\.write\s*\(',
            r'\.delete\s*\(',
            r'urllib\.',
            r'requests\.',
            r'socket\.',
            r'pickle\.',
            r'marshal\.',
        ]
        
        self.sql_injection_patterns = [
            r'(\bunion\b.*\bselect\b)',
            r'(\bselect\b.*\bfrom\b)',
            r'(\binsert\b.*\binto\b)',
            r'(\bupdate\b.*\bset\b)',
            r'(\bdelete\b.*\bfrom\b)',
            r'(\bdrop\b.*\btable\b)',
            r'(\balter\b.*\btable\b)',
            r'(\bcreate\b.*\btable\b)',
            r'(--|#|/\*|\*/)',
            r"(\bor\b.*=.*|'\s*or\s*')",
        ]
    
    def validate_string_input(self, text: str, max_length: int = 10000) -> Tuple[bool, str]:
        """Validate string input for security threats."""
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length}"
        
        # Check for dangerous patterns
        text_lower = text.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}"
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, f"SQL injection pattern detected: {pattern}"
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-_()[]{}') / len(text)
        if special_char_ratio > 0.5:
            return False, "Excessive special characters detected"
        
        return True, "Input validation passed"
    
    def validate_json_input(self, json_str: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate JSON input for security threats."""
        try:
            # Basic string validation first
            is_valid, message = self.validate_string_input(json_str, max_length=50000)
            if not is_valid:
                return False, message, None
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Check for dangerous keys or values
            def check_dict_recursive(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # Check for dangerous keys
                        if key.lower() in ['__import__', 'eval', 'exec', 'compile']:
                            return False, f"Dangerous key detected: {current_path}"
                        
                        # Recurse into nested structures
                        is_valid, msg = check_dict_recursive(value, current_path)
                        if not is_valid:
                            return False, msg
                
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        is_valid, msg = check_dict_recursive(item, f"{path}[{i}]")
                        if not is_valid:
                            return False, msg
                
                elif isinstance(obj, str):
                    is_valid, msg = self.validate_string_input(obj, max_length=1000)
                    if not is_valid:
                        return False, f"Invalid string at {path}: {msg}"
                
                return True, "Valid"
            
            is_valid, message = check_dict_recursive(data)
            if not is_valid:
                return False, message, None
            
            return True, "JSON validation passed", data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None
    
    def validate_file_path(self, file_path: str) -> Tuple[bool, str]:
        """Validate file path for security."""
        path = Path(file_path)
        
        try:
            # Resolve path to check for traversal attacks
            resolved_path = path.resolve()
            current_dir = Path.cwd().resolve()
            
            # Check if path is within allowed directories
            is_allowed = False
            for allowed_path in self.policy.allowed_paths:
                allowed_resolved = Path(allowed_path).resolve()
                try:
                    resolved_path.relative_to(allowed_resolved)
                    is_allowed = True
                    break
                except ValueError:
                    continue
            
            if not is_allowed:
                return False, "Path not in allowed directories"
            
            # Check for blocked paths
            for blocked_path in self.policy.blocked_paths:
                blocked_resolved = Path(blocked_path).resolve()
                try:
                    resolved_path.relative_to(blocked_resolved)
                    return False, f"Path in blocked directory: {blocked_path}"
                except ValueError:
                    continue
            
            # Check file extension
            if path.suffix.lower() in self.policy.blocked_file_extensions:
                return False, f"Blocked file extension: {path.suffix}"
            
            if path.suffix.lower() not in self.policy.allowed_file_extensions:
                return False, f"File extension not in allowed list: {path.suffix}"
            
            # Check for path traversal patterns
            path_str = str(path)
            if '..' in path_str or '~' in path_str:
                return False, "Path traversal patterns detected"
            
            return True, "File path validation passed"
            
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    def validate_code_content(self, code: str) -> Tuple[bool, str]:
        """Validate code content for dangerous operations."""
        if not self.policy.allow_code_execution:
            return False, "Code execution is disabled by policy"
        
        try:
            # Parse the code to check for dangerous operations
            tree = ast.parse(code)
            
            # Check for dangerous function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.policy.blocked_functions:
                            return False, f"Blocked function call: {func_name}"
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.policy.allowed_imports:
                            return False, f"Blocked import: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.policy.allowed_imports:
                        return False, f"Blocked import from: {node.module}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Code validation error: {str(e)}"


class APISecurityManager:
    """Manage API security including rate limiting and authentication."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.rate_limits = {}  # Track rate limits per API/user
        self.active_requests = 0  # Track concurrent requests
        self.api_keys = {}  # Store validated API keys
        
    def validate_api_key(self, api_key: str, service: str) -> bool:
        """Validate API key format and store securely."""
        if not api_key or len(api_key) < 10:
            return False
        
        # Check format based on service
        format_patterns = {
            'openai': r'^sk-[a-zA-Z0-9]{48,}$',
            'anthropic': r'^sk-ant-[a-zA-Z0-9-]{36,}$',
            'gemini': r'^[a-zA-Z0-9-_]{39}$'
        }
        
        pattern = format_patterns.get(service.lower())
        if pattern and not re.match(pattern, api_key):
            return False
        
        # Store hash of API key (not the key itself)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[service] = key_hash
        
        return True
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.now()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Clean old requests (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier] 
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        if len(self.rate_limits[identifier]) >= self.policy.api_rate_limit_per_minute:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    def check_concurrent_limit(self) -> bool:
        """Check if concurrent request limit is exceeded."""
        return self.active_requests < self.policy.max_concurrent_requests
    
    def acquire_request_slot(self) -> bool:
        """Acquire a slot for concurrent request."""
        if self.check_concurrent_limit():
            self.active_requests += 1
            return True
        return False
    
    def release_request_slot(self):
        """Release a concurrent request slot."""
        if self.active_requests > 0:
            self.active_requests -= 1


class FileSystemProtector:
    """Protect file system operations."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.monitored_operations = []
        
    def validate_file_operation(self, operation: str, file_path: str, 
                               content: Optional[str] = None) -> Tuple[bool, str]:
        """Validate file system operation."""
        validator = InputValidator(self.policy)
        
        # Validate file path
        is_valid, message = validator.validate_file_path(file_path)
        if not is_valid:
            return False, f"File path validation failed: {message}"
        
        # Check file size for write operations
        if operation in ['write', 'append'] and content:
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if content_size_mb > self.policy.max_file_size_mb:
                return False, f"File content exceeds size limit: {content_size_mb:.1f}MB"
        
        # Validate content if provided
        if content:
            is_valid, message = validator.validate_string_input(content, max_length=50000000)
            if not is_valid:
                return False, f"Content validation failed: {message}"
        
        # Log the operation
        self.monitored_operations.append({
            'operation': operation,
            'file_path': file_path,
            'timestamp': datetime.now(),
            'content_size': len(content) if content else 0
        })
        
        return True, "File operation validated"
    
    def scan_file_for_threats(self, file_path: str) -> List[str]:
        """Scan file for potential security threats."""
        threats = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            validator = InputValidator(self.policy)
            
            # Check for dangerous patterns
            for pattern in validator.dangerous_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(f"Dangerous pattern found: {pattern}")
            
            # Check for suspicious strings
            suspicious_strings = [
                'password', 'secret', 'api_key', 'private_key', 'token',
                'credential', 'auth', 'login', 'admin'
            ]
            
            content_lower = content.lower()
            for suspicious in suspicious_strings:
                if suspicious in content_lower:
                    # Check if it's in a suspicious context
                    pattern = rf'{suspicious}\s*[:=]\s*["\'][^"\']+["\']'
                    if re.search(pattern, content_lower):
                        threats.append(f"Potential credential exposure: {suspicious}")
        
        except Exception as e:
            threats.append(f"File scan error: {str(e)}")
        
        return threats


class ThreatDetector:
    """Detect and classify security threats."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.threat_patterns = {
            ThreatLevel.CRITICAL: [
                r'rm\s+-rf\s+/',
                r'dd\s+if=/dev/zero',
                r':(){ :|:& };:',  # Fork bomb
                r'curl.*\|\s*bash',
                r'wget.*\|\s*bash',
            ],
            ThreatLevel.HIGH: [
                r'nc\s+-l',  # Netcat listener
                r'python.*-c.*exec',
                r'bash.*-i.*>&.*2>&1',
                r'/etc/passwd',
                r'/etc/shadow',
            ],
            ThreatLevel.MEDIUM: [
                r'sudo\s+',
                r'chmod\s+777',
                r'chown\s+root',
                r'crontab\s+-e',
            ]
        }
    
    def analyze_content(self, content: str, source: str = "unknown") -> List[SecurityEvent]:
        """Analyze content for security threats."""
        events = []
        event_id_base = hashlib.md5(f"{source}_{datetime.now()}".encode()).hexdigest()[:8]
        
        # Check threat patterns
        for threat_level, patterns in self.threat_patterns.items():
            for i, pattern in enumerate(patterns):
                if re.search(pattern, content, re.IGNORECASE):
                    event = SecurityEvent(
                        event_id=f"{event_id_base}_{threat_level.value}_{i}",
                        event_type=SecurityEventType.MALICIOUS_CODE_DETECTED,
                        threat_level=threat_level,
                        timestamp=datetime.now(),
                        source=source,
                        details={
                            "pattern_matched": pattern,
                            "content_preview": content[:200] + "..." if len(content) > 200 else content
                        },
                        blocked=True
                    )
                    events.append(event)
        
        return events
    
    def analyze_network_activity(self, requests: List[Dict]) -> List[SecurityEvent]:
        """Analyze network activity for suspicious patterns."""
        events = []
        
        # Look for unusual request patterns
        if len(requests) > 100:  # Too many requests
            event = SecurityEvent(
                event_id=f"net_{datetime.now().timestamp()}",
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.now(),
                source="network_monitor",
                details={"request_count": len(requests)},
                blocked=True
            )
            events.append(event)
        
        return events


class SecurityLogger:
    """Centralized security logging."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup security logger
        self.logger = logging.getLogger('security')
        self.logger.setLevel(logging.INFO)
        
        # Security log file
        security_log = self.log_dir / 'security.log'
        handler = logging.FileHandler(security_log)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.events = []
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event."""
        self.events.append(event)
        
        log_message = (
            f"SECURITY EVENT [{event.threat_level.value.upper()}] "
            f"{event.event_type.value}: {event.details.get('message', '')} "
            f"(Source: {event.source}, Blocked: {event.blocked})"
        )
        
        if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            self.logger.error(log_message)
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        now = datetime.now()
        
        # Count events by type and level
        event_counts = {}
        level_counts = {}
        
        for event in self.events:
            event_type = event.event_type.value
            threat_level = event.threat_level.value
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            level_counts[threat_level] = level_counts.get(threat_level, 0) + 1
        
        # Recent events (last 24 hours)
        recent_cutoff = now - timedelta(hours=24)
        recent_events = [
            event for event in self.events 
            if event.timestamp >= recent_cutoff
        ]
        
        return {
            "report_generated": now.isoformat(),
            "total_events": len(self.events),
            "recent_events": len(recent_events),
            "event_counts": event_counts,
            "threat_level_counts": level_counts,
            "blocked_events": sum(1 for event in self.events if event.blocked),
            "unresolved_events": sum(1 for event in self.events if not event.resolved),
            "top_sources": self._get_top_sources(),
            "recommendations": self._generate_recommendations()
        }
    
    def _get_top_sources(self) -> List[Dict[str, Any]]:
        """Get top sources of security events."""
        source_counts = {}
        for event in self.events:
            source_counts[event.source] = source_counts.get(event.source, 0) + 1
        
        return [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        critical_events = [e for e in self.events if e.threat_level == ThreatLevel.CRITICAL]
        if critical_events:
            recommendations.append("Investigate and resolve critical security events immediately")
        
        high_events = [e for e in self.events if e.threat_level == ThreatLevel.HIGH]
        if len(high_events) > 10:
            recommendations.append("High number of high-severity events detected - review security policies")
        
        unresolved = [e for e in self.events if not e.resolved]
        if len(unresolved) > 5:
            recommendations.append("Multiple unresolved security events - implement automated resolution")
        
        return recommendations


class TerragronSecurityFramework:
    """Main security framework coordinator."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None, 
                 log_dir: Path = Path("security_logs")):
        self.policy = policy or SecurityPolicy()
        self.input_validator = InputValidator(self.policy)
        self.api_security = APISecurityManager(self.policy)
        self.fs_protector = FileSystemProtector(self.policy)
        self.threat_detector = ThreatDetector(self.policy)
        self.security_logger = SecurityLogger(log_dir)
        
        self.logger = logging.getLogger(__name__)
    
    def validate_research_input(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Comprehensive validation of research input data."""
        try:
            # Convert to JSON string for validation
            json_str = json.dumps(data)
            is_valid, message, validated_data = self.input_validator.validate_json_input(json_str)
            
            if not is_valid:
                event = SecurityEvent(
                    event_id=f"input_val_{datetime.now().timestamp()}",
                    event_type=SecurityEventType.INPUT_VALIDATION_FAILED,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.now(),
                    source="research_input",
                    details={"validation_error": message},
                    blocked=True
                )
                self.security_logger.log_security_event(event)
                return False, f"Input validation failed: {message}"
            
            return True, "Research input validated successfully"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def secure_file_operation(self, operation: str, file_path: str, 
                             content: Optional[str] = None) -> Tuple[bool, str]:
        """Perform secure file operation with validation."""
        # Validate the operation
        is_valid, message = self.fs_protector.validate_file_operation(
            operation, file_path, content
        )
        
        if not is_valid:
            event = SecurityEvent(
                event_id=f"file_op_{datetime.now().timestamp()}",
                event_type=SecurityEventType.FILE_SYSTEM_VIOLATION,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.now(),
                source="file_system",
                details={
                    "operation": operation,
                    "file_path": file_path,
                    "validation_error": message
                },
                blocked=True
            )
            self.security_logger.log_security_event(event)
            return False, f"File operation blocked: {message}"
        
        # Scan content for threats if provided
        if content:
            threat_events = self.threat_detector.analyze_content(content, f"file:{file_path}")
            for event in threat_events:
                self.security_logger.log_security_event(event)
            
            if any(e.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] for e in threat_events):
                return False, "High-risk content detected - operation blocked"
        
        return True, "File operation approved"
    
    def validate_api_request(self, service: str, api_key: str, identifier: str) -> Tuple[bool, str]:
        """Validate API request with security checks."""
        # Check rate limits
        if not self.api_security.check_rate_limit(identifier):
            event = SecurityEvent(
                event_id=f"rate_limit_{datetime.now().timestamp()}",
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=datetime.now(),
                source="api_security",
                details={"service": service, "identifier": identifier},
                blocked=True
            )
            self.security_logger.log_security_event(event)
            return False, "Rate limit exceeded"
        
        # Check concurrent requests
        if not self.api_security.acquire_request_slot():
            return False, "Concurrent request limit exceeded"
        
        # Validate API key
        if self.policy.enable_api_key_validation:
            if not self.api_security.validate_api_key(api_key, service):
                event = SecurityEvent(
                    event_id=f"api_key_{datetime.now().timestamp()}",
                    event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.now(),
                    source="api_security",
                    details={"service": service, "key_format": "invalid"},
                    blocked=True
                )
                self.security_logger.log_security_event(event)
                self.api_security.release_request_slot()
                return False, "Invalid API key"
        
        return True, "API request validated"
    
    def complete_api_request(self):
        """Complete API request and release resources."""
        self.api_security.release_request_slot()
    
    def scan_research_session(self, session_dir: Path) -> Dict[str, Any]:
        """Comprehensive security scan of research session."""
        scan_results = {
            "scan_timestamp": datetime.now().isoformat(),
            "session_directory": str(session_dir),
            "files_scanned": 0,
            "threats_found": 0,
            "high_risk_files": [],
            "recommendations": []
        }
        
        if not session_dir.exists():
            scan_results["error"] = "Session directory not found"
            return scan_results
        
        # Scan all files in session
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                scan_results["files_scanned"] += 1
                
                # Check file extension
                if file_path.suffix in self.policy.blocked_file_extensions:
                    scan_results["high_risk_files"].append(str(file_path))
                    scan_results["threats_found"] += 1
                    continue
                
                # Scan file content
                threats = self.fs_protector.scan_file_for_threats(file_path)
                if threats:
                    scan_results["high_risk_files"].append({
                        "file": str(file_path),
                        "threats": threats
                    })
                    scan_results["threats_found"] += len(threats)
        
        # Generate recommendations
        if scan_results["threats_found"] > 0:
            scan_results["recommendations"].append("Review and quarantine high-risk files")
        
        if scan_results["files_scanned"] > 1000:
            scan_results["recommendations"].append("Large number of files - implement automated scanning")
        
        return scan_results
    
    def generate_security_dashboard(self) -> str:
        """Generate HTML security dashboard."""
        report = self.security_logger.generate_security_report()
        
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Terragon Security Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .critical {{ color: #dc3545; font-weight: bold; }}
        .high {{ color: #fd7e14; font-weight: bold; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #28a745; }}
        .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>🛡️ Terragon Security Dashboard</h1>
    
    <div class="security-overview">
        <h2>Security Overview</h2>
        <div class="metric">
            <strong>Total Security Events:</strong> {report['total_events']}
        </div>
        <div class="metric">
            <strong>Recent Events (24h):</strong> {report['recent_events']}
        </div>
        <div class="metric">
            <strong>Blocked Events:</strong> {report['blocked_events']}
        </div>
        <div class="metric">
            <strong>Unresolved Events:</strong> {report['unresolved_events']}
        </div>
    </div>
    
    <div class="threat-levels">
        <h3>Threat Levels</h3>
        <table>
            <tr><th>Level</th><th>Count</th></tr>
            {self._generate_threat_level_rows(report.get('threat_level_counts', {}))}
        </table>
    </div>
    
    <div class="event-types">
        <h3>Event Types</h3>
        <table>
            <tr><th>Type</th><th>Count</th></tr>
            {self._generate_event_type_rows(report.get('event_counts', {}))}
        </table>
    </div>
    
    <div class="recommendations">
        <h3>Security Recommendations</h3>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in report.get('recommendations', []))}
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated: {report['report_generated']}</p>
    </div>
</body>
</html>
        """
        
        return dashboard_html
    
    def _generate_threat_level_rows(self, counts: Dict[str, int]) -> str:
        """Generate HTML rows for threat levels."""
        level_classes = {
            'critical': 'critical',
            'high': 'high', 
            'medium': 'medium',
            'low': 'low'
        }
        
        rows = []
        for level in ['critical', 'high', 'medium', 'low']:
            count = counts.get(level, 0)
            class_name = level_classes.get(level, '')
            rows.append(f'<tr><td class="{class_name}">{level.title()}</td><td>{count}</td></tr>')
        
        return ''.join(rows)
    
    def _generate_event_type_rows(self, counts: Dict[str, int]) -> str:
        """Generate HTML rows for event types."""
        rows = []
        for event_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            formatted_type = event_type.replace('_', ' ').title()
            rows.append(f'<tr><td>{formatted_type}</td><td>{count}</td></tr>')
        
        return ''.join(rows)


async def main():
    """Main execution function for security framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Security Framework")
    parser.add_argument("--scan-dir", help="Directory to scan for security threats")
    parser.add_argument("--generate-dashboard", action="store_true", 
                       help="Generate security dashboard")
    
    args = parser.parse_args()
    
    # Create security framework
    security = TerragronSecurityFramework()
    
    if args.scan_dir:
        print(f"🔍 Scanning directory: {args.scan_dir}")
        scan_results = security.scan_research_session(Path(args.scan_dir))
        
        print(f"📊 Scan Results:")
        print(f"   Files scanned: {scan_results['files_scanned']}")
        print(f"   Threats found: {scan_results['threats_found']}")
        print(f"   High-risk files: {len(scan_results.get('high_risk_files', []))}")
        
        if scan_results.get('recommendations'):
            print(f"💡 Recommendations:")
            for rec in scan_results['recommendations']:
                print(f"   • {rec}")
    
    if args.generate_dashboard:
        print("📊 Generating security dashboard...")
        dashboard_html = security.generate_security_dashboard()
        
        dashboard_path = Path("security_dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"✅ Dashboard saved to: {dashboard_path}")
    
    print("\n🛡️ Terragon Security Framework ready for use!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())