#!/usr/bin/env python3
"""
Secure subprocess wrapper to prevent command injection and enforce security policies.
Implements the requirements from backlog item: subprocess-security-wrapper (WSJF: 4.8)

Acceptance criteria:
- Create secure subprocess wrapper module
- Replace all subprocess.run() calls with secure wrapper
- Add input validation and sanitization
- Write security tests for wrapper
"""

import os
import re
import shlex
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class SecureSubprocess:
    """
    Secure wrapper around subprocess operations.
    
    Provides security controls including:
    - Command injection prevention
    - Path traversal protection
    - Working directory validation
    - Environment variable sanitization
    - Command whitelist enforcement
    """
    
    # Allowed executables (expandable based on needs)
    ALLOWED_EXECUTABLES = {
        'echo', 'ls', 'cat', 'grep', 'find', 'head', 'tail', 'wc', 'sort',
        'python', 'python3', 'pip', 'pip3', 'git', 'pdflatex', 'bibtex',
        'makeindex', 'chktex', 'pdftocairo', 'pdftoppm', 'pdftotext',
        'biber', 'pdfcrop', 'ps2pdf', 'dvips', 'latex', 'tex', 'mkdir',
        'cp', 'mv', 'rm', 'touch', 'chmod', 'which', 'whereis', 'sleep'
    }
    
    # Dangerous command patterns to block
    DANGEROUS_PATTERNS = [
        r'[;&|`$]',  # Command injection characters
        r'\.\./|\.\.\\',  # Path traversal
        r'/etc/',  # System directories
        r'/bin/',
        r'/usr/bin/',
        r'/sbin/',
        r'/usr/sbin/',
        r'sudo\s',  # Privilege escalation
        r'su\s',
        r'chmod\s+[0-9]*7',  # Dangerous permissions
        r'rm\s+.*-rf',  # Dangerous deletion
        r'>(.*)/dev/',  # Device access
        r'exec\s*\(',  # Code execution
        r'eval\s*\(',
        r'import\s+os.*system',  # Dangerous Python imports
        r'__import__',
    ]
    
    # Dangerous environment variables to remove/sanitize
    DANGEROUS_ENV_VARS = {
        'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES',
        'DYLD_LIBRARY_PATH', 'PYTHONPATH', 'PERL5LIB', 'RUBYLIB'
    }
    
    # Safe system directories that can be used as working directories
    SAFE_WORKING_DIRS = {
        '/tmp', '/var/tmp', '/home', '/Users', '/workspace', '/workspaces'
    }
    
    @classmethod
    def _validate_command(cls, cmd: List[str]) -> None:
        """
        Validate that a command is safe to execute.
        
        Args:
            cmd: Command and arguments as list
            
        Raises:
            SecurityError: If command is deemed unsafe
            ValueError: If command format is invalid
        """
        if not cmd or not isinstance(cmd, list):
            raise ValueError("Command must be a non-empty list")
        
        if not cmd[0]:
            raise ValueError("Command executable cannot be empty")
        
        # Extract just the executable name (without path)
        executable = Path(cmd[0]).name
        
        # Check if executable is in whitelist
        if executable not in cls.ALLOWED_EXECUTABLES:
            raise SecurityError(f"Executable '{executable}' is not in whitelist")
        
        # Join command for pattern checking
        cmd_string = ' '.join(cmd)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, cmd_string, re.IGNORECASE):
                raise SecurityError(f"Command contains dangerous pattern: {pattern}")
        
        # Additional checks for specific commands
        if executable in ['python', 'python3']:
            cls._validate_python_command(cmd)
        elif executable == 'rm':
            cls._validate_rm_command(cmd)
        elif executable in ['chmod', 'chown']:
            cls._validate_permission_command(cmd)
    
    @classmethod
    def _validate_python_command(cls, cmd: List[str]) -> None:
        """Validate Python command for dangerous operations."""
        cmd_string = ' '.join(cmd)
        
        # Block dangerous Python operations
        dangerous_python = [
            r'-c.*import\s+os',
            r'-c.*subprocess',
            r'-c.*exec\s*\(',
            r'-c.*eval\s*\(',
            r'-c.*__import__',
            r'-c.*open\s*\(',
        ]
        
        for pattern in dangerous_python:
            if re.search(pattern, cmd_string, re.IGNORECASE):
                raise SecurityError(f"Python command contains dangerous operation: {pattern}")
    
    @classmethod
    def _validate_rm_command(cls, cmd: List[str]) -> None:
        """Validate rm command to prevent dangerous deletions."""
        if '-rf' in ' '.join(cmd):
            # Allow rm -rf only for specific safe patterns
            safe_rm_patterns = [
                r'/tmp/',
                r'/var/tmp/',
                r'\./',  # Current directory files
                r'[^/]*\.tmp',  # Temporary files
                r'[^/]*\.temp',
            ]
            
            cmd_string = ' '.join(cmd)
            is_safe = any(re.search(pattern, cmd_string) for pattern in safe_rm_patterns)
            
            if not is_safe:
                raise SecurityError("rm -rf command targets unsafe paths")
    
    @classmethod
    def _validate_permission_command(cls, cmd: List[str]) -> None:
        """Validate chmod/chown commands."""
        if len(cmd) >= 3 and 'chmod' in cmd[0]:
            mode = cmd[1]
            # Block dangerous permissions (7 in any position = execute for all)
            if '7' in mode and len(mode) == 3:
                raise SecurityError(f"chmod mode '{mode}' grants dangerous permissions")
    
    @classmethod
    def _validate_working_directory(cls, cwd: Optional[str]) -> None:
        """
        Validate that working directory is safe.
        
        Args:
            cwd: Working directory path
            
        Raises:
            SecurityError: If working directory is unsafe
        """
        if cwd is None:
            return
        
        cwd_path = Path(cwd).resolve()
        
        # Block dangerous system directories
        dangerous_dirs = {
            Path('/'), Path('/etc'), Path('/bin'), Path('/usr/bin'),
            Path('/sbin'), Path('/usr/sbin'), Path('/root')
        }
        
        for dangerous_dir in dangerous_dirs:
            try:
                if cwd_path == dangerous_dir or dangerous_dir in cwd_path.parents:
                    raise SecurityError(f"Working directory '{cwd}' is in dangerous system path")
            except (OSError, ValueError):
                # Path resolution failed, treat as unsafe
                raise SecurityError(f"Cannot resolve working directory '{cwd}'")
    
    @classmethod
    def _sanitize_environment(cls, env: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Sanitize environment variables.
        
        Args:
            env: Environment variables dictionary
            
        Returns:
            Sanitized environment variables
        """
        if env is None:
            env = os.environ.copy()
        else:
            env = env.copy()
        
        # Remove dangerous environment variables
        for var in cls.DANGEROUS_ENV_VARS:
            env.pop(var, None)
        
        # Sanitize PATH to remove dangerous directories
        if 'PATH' in env:
            path_parts = env['PATH'].split(os.pathsep)
            safe_path_parts = []
            
            for part in path_parts:
                # Skip empty parts and dangerous directories
                if part and not any(danger in part for danger in ['/tmp', '..', '~']):
                    safe_path_parts.append(part)
            
            env['PATH'] = os.pathsep.join(safe_path_parts)
        
        return env
    
    @classmethod
    def run(cls, 
            cmd: List[str],
            timeout: Optional[float] = 30,
            cwd: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            capture_output: bool = False,
            check: bool = False,
            **kwargs) -> subprocess.CompletedProcess:
        """
        Secure version of subprocess.run().
        
        Args:
            cmd: Command and arguments as list
            timeout: Maximum execution time in seconds
            cwd: Working directory
            env: Environment variables
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit
            **kwargs: Additional arguments for subprocess.run()
            
        Returns:
            CompletedProcess instance
            
        Raises:
            SecurityError: If command is deemed unsafe
            ValueError: If arguments are invalid
        """
        # Validate inputs
        cls._validate_command(cmd)
        cls._validate_working_directory(cwd)
        
        # Sanitize environment
        safe_env = cls._sanitize_environment(env)
        
        # Set default timeout if not provided
        if timeout is None:
            timeout = 30
        
        # Log the secure execution
        logger.info(f"Executing secure command: {cmd[:2]}... (cwd={cwd})")
        
        try:
            # Execute with security controls
            result = subprocess.run(
                cmd,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
                capture_output=capture_output,
                check=check,
                **kwargs
            )
            
            logger.debug(f"Command completed with return code: {result.returncode}")
            return result
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Command timed out after {timeout} seconds: {cmd[0]}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}: {cmd[0]}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing command: {e}")
            raise
    
    @classmethod
    def call(cls, cmd: List[str], **kwargs) -> int:
        """
        Secure version of subprocess.call().
        
        Args:
            cmd: Command and arguments as list
            **kwargs: Arguments for run()
            
        Returns:
            Return code
        """
        result = cls.run(cmd, **kwargs)
        return result.returncode
    
    @classmethod
    def check_output(cls, cmd: List[str], **kwargs) -> bytes:
        """
        Secure version of subprocess.check_output().
        
        Args:
            cmd: Command and arguments as list
            **kwargs: Arguments for run()
            
        Returns:
            Command output as bytes
        """
        kwargs['capture_output'] = True
        kwargs['check'] = True
        result = cls.run(cmd, **kwargs)
        return result.stdout


# Convenience functions for backward compatibility
def secure_run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Convenience wrapper for SecureSubprocess.run()."""
    return SecureSubprocess.run(cmd, **kwargs)

def secure_call(cmd: List[str], **kwargs) -> int:
    """Convenience wrapper for SecureSubprocess.call()."""
    return SecureSubprocess.call(cmd, **kwargs)

def secure_check_output(cmd: List[str], **kwargs) -> bytes:
    """Convenience wrapper for SecureSubprocess.check_output()."""
    return SecureSubprocess.check_output(cmd, **kwargs)