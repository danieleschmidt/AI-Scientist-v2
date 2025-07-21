#!/usr/bin/env python3
"""
Comprehensive input validation and sanitization utilities.

This module provides security functions for validating and sanitizing
various types of input including:
- LLM-generated code execution
- File path validation  
- Archive extraction security
- Configuration parsing
- External API data validation

All functions follow a secure-by-default approach.
"""

import os
import re
import ast
import json
import zipfile
import tempfile
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ValidationError(ValueError):
    """Raised when input validation fails"""
    pass


class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass


# Code Execution Security
# ======================

DANGEROUS_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'glob', 'socket', 'urllib', 
    'requests', 'http', 'ftplib', 'smtplib', 'telnetlib', 'webbrowser',
    'importlib', 'pkgutil', 'runpy', 'code', 'codeop', 'compileall'
}

DANGEROUS_BUILTINS = {
    'eval', 'exec', 'compile', 'open', '__import__', 'getattr', 'setattr',
    'delattr', 'hasattr', 'vars', 'dir', 'globals', 'locals', 'input',
    'raw_input', 'file', 'execfile', 'reload'
}

DANGEROUS_PATTERNS = [
    r'__[a-zA-Z_]+__',  # Dunder methods
    r'\.system\(',       # os.system calls
    r'\.call\(',         # subprocess.call
    r'\.Popen\(',        # subprocess.Popen
    r'\.urlopen\(',      # urllib.urlopen
    r'\.request\(',      # requests calls
    r'exec\s*\(',        # exec function
    r'eval\s*\(',        # eval function
    r'import\s+os',      # os import
    r'from\s+os',        # from os import
]


def is_code_safe(code: str) -> bool:
    """
    Check if code is safe to execute.
    
    Args:
        code (str): The code to validate
        
    Returns:
        bool: True if code appears safe, False otherwise
        
    Raises:
        SecurityError: If dangerous patterns are detected
    """
    if not isinstance(code, str):
        raise ValidationError("Code must be a string")
    
    # Parse the code to check for dangerous AST nodes
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise ValidationError("Invalid Python syntax")
    
    # Check for dangerous imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in DANGEROUS_IMPORTS:
                    raise SecurityError(f"Dangerous import detected: {alias.name}")
        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in DANGEROUS_IMPORTS:
                raise SecurityError(f"Dangerous import detected: from {node.module}")
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
                raise SecurityError(f"Dangerous builtin detected: {node.func.id}")
    
    # Check for dangerous patterns in code string
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            raise SecurityError(f"Dangerous pattern detected: {pattern}")
    
    return True


def execute_code_safely(code: str, globals_dict: Optional[Dict] = None, 
                       locals_dict: Optional[Dict] = None) -> Any:
    """
    Execute code in a restricted environment.
    
    Args:
        code (str): The code to execute
        globals_dict (dict): Global variables (will be filtered)
        locals_dict (dict): Local variables (will be filtered)
        
    Returns:
        Any: Result of code execution
        
    Raises:
        SecurityError: If code is unsafe
        ValidationError: If execution fails
    """
    # First validate the code
    is_code_safe(code)
    
    # Create restricted environment
    safe_globals = {
        '__builtins__': {
            # Safe builtins only
            'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'min': min, 'max': max, 'sum': sum, 'abs': abs,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'print': print, 'type': type, 'isinstance': isinstance,
        }
    }
    
    # Add safe modules
    import math
    import json as json_module
    safe_globals['math'] = math
    safe_globals['json'] = json_module
    
    # Add user globals if provided (filtered)
    if globals_dict:
        for key, value in globals_dict.items():
            if not key.startswith('_') and key not in DANGEROUS_BUILTINS:
                safe_globals[key] = value
    
    # Execute in restricted environment
    try:
        result = eval(compile(code, '<string>', 'eval'), safe_globals, locals_dict)
        return result
    except Exception as e:
        raise ValidationError(f"Code execution failed: {e}")


# Archive Extraction Security
# ===========================

MAX_EXTRACTED_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILES = 1000
MAX_PATH_LENGTH = 255


def validate_zip_file(zip_path: str) -> bool:
    """
    Validate a zip file for security issues.
    
    Args:
        zip_path (str): Path to the zip file
        
    Returns:
        bool: True if safe, False otherwise
        
    Raises:
        SecurityError: If security issues are detected
        ValidationError: If file is invalid
    """
    if not os.path.exists(zip_path):
        raise ValidationError(f"Zip file does not exist: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total_size = 0
            file_count = 0
            
            for member in zf.namelist():
                file_count += 1
                
                # Check file count limit
                if file_count > MAX_FILES:
                    raise SecurityError(f"Too many files in archive: {file_count}")
                
                # Check for path traversal
                if '..' in member or member.startswith('/') or member.startswith('\\'):
                    raise SecurityError(f"Path traversal detected: {member}")
                
                # Check path length
                if len(member) > MAX_PATH_LENGTH:
                    raise SecurityError(f"Path too long: {member}")
                
                # Check uncompressed size
                info = zf.getinfo(member)
                total_size += info.file_size
                
                if total_size > MAX_EXTRACTED_SIZE:
                    raise SecurityError(f"Archive too large when extracted: {total_size}")
                
                # Check compression ratio (zip bomb detection)
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > 100:  # Highly compressed files are suspicious
                        raise SecurityError(f"Suspicious compression ratio: {ratio}")
        
        return True
    
    except zipfile.BadZipFile:
        raise ValidationError("Invalid zip file")


def safe_extract_zip(zip_path: str, extract_dir: str) -> bool:
    """
    Safely extract a zip file.
    
    Args:
        zip_path (str): Path to the zip file
        extract_dir (str): Directory to extract to
        
    Returns:
        bool: True if extraction successful
        
    Raises:
        SecurityError: If security issues are detected
        ValidationError: If extraction fails
    """
    # First validate the zip file
    validate_zip_file(zip_path)
    
    # Ensure extract directory exists and is safe
    extract_path = Path(extract_dir).resolve()
    extract_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                # Double-check each member before extraction
                target_path = extract_path / member
                
                # Ensure the target is within the extract directory
                if not str(target_path.resolve()).startswith(str(extract_path)):
                    raise SecurityError(f"Path traversal attempt: {member}")
                
                # Extract the file
                zf.extract(member, extract_dir)
        
        return True
    
    except Exception as e:
        raise ValidationError(f"Zip extraction failed: {e}")


# File Path Security
# ==================

def validate_file_path(path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
    """
    Validate a file path for security issues.
    
    Args:
        path (str): The file path to validate
        allowed_dirs (list): List of allowed base directories
        
    Returns:
        bool: True if path is safe
        
    Raises:
        SecurityError: If path is unsafe
        ValidationError: If path is invalid
    """
    if not isinstance(path, str):
        raise ValidationError("Path must be a string")
    
    if not path:
        raise ValidationError("Path cannot be empty")
    
    # Check for path traversal
    if '..' in path:
        raise SecurityError("Path traversal detected")
    
    # Check for absolute paths (unless explicitly allowed)
    if os.path.isabs(path):
        if not allowed_dirs:
            raise SecurityError("Absolute paths not allowed")
        
        # Check if absolute path is within allowed directories
        abs_path = os.path.abspath(path)
        allowed = False
        for allowed_dir in allowed_dirs:
            if abs_path.startswith(os.path.abspath(allowed_dir)):
                allowed = True
                break
        
        if not allowed:
            raise SecurityError(f"Path not in allowed directories: {path}")
    
    # Check path length
    if len(path) > MAX_PATH_LENGTH:
        raise SecurityError("Path too long")
    
    # Check for dangerous characters
    dangerous_chars = ['<', '>', '|', '&', ';', '`', '$']
    for char in dangerous_chars:
        if char in path:
            raise SecurityError(f"Dangerous character in path: {char}")
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing dangerous characters.
    
    Args:
        filename (str): The filename to sanitize
        
    Returns:
        str: Sanitized filename
        
    Raises:
        ValidationError: If filename is invalid
    """
    if not isinstance(filename, str):
        raise ValidationError("Filename must be a string")
    
    if not filename.strip():
        raise ValidationError("Filename cannot be empty")
    
    # Remove path separators and traversal patterns
    filename = filename.replace('/', '_').replace('\\', '_').replace('..', '_')
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', ';', '&', '`', '$']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not a reserved name
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 
                     'LPT1', 'LPT2', 'LPT3']
    if filename.upper() in reserved_names:
        filename = f"file_{filename}"
    
    # Ensure minimum length
    if len(filename) < 1:
        filename = "unnamed_file"
    
    # Ensure maximum length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename


# Configuration Security
# ======================

def validate_json_config(config_data: Union[str, Dict], schema: Optional[Dict] = None) -> Dict:
    """
    Validate JSON configuration data.
    
    Args:
        config_data: JSON string or dictionary
        schema: Optional JSON schema for validation
        
    Returns:
        dict: Validated configuration
        
    Raises:
        SecurityError: If dangerous content detected
        ValidationError: If validation fails
    """
    # Parse JSON if string
    if isinstance(config_data, str):
        if len(config_data) > 1024 * 1024:  # 1MB limit
            raise ValidationError("Configuration too large")
        
        try:
            config = json.loads(config_data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
    else:
        config = config_data
    
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    # Check for dangerous content
    config_str = json.dumps(config, default=str)
    
    # Check for script injection
    dangerous_patterns = ['<script>', 'javascript:', 'data:text/html', 'eval(', 'Function(']
    for pattern in dangerous_patterns:
        if pattern in config_str:
            raise SecurityError(f"Dangerous content detected: {pattern}")
    
    # Check for path traversal
    if '..' in config_str:
        raise SecurityError("Path traversal detected in configuration")
    
    # Basic schema validation if provided
    if schema:
        for key, expected_type in schema.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    raise ValidationError(f"Invalid type for {key}: expected {expected_type}")
    
    return config


# External Data Validation
# ========================

def validate_api_response(response: Dict, max_size: int = 1024 * 1024) -> bool:
    """
    Validate external API response data.
    
    Args:
        response: API response dictionary
        max_size: Maximum response size in bytes
        
    Returns:
        bool: True if response is safe
        
    Raises:
        SecurityError: If dangerous content detected
        ValidationError: If response is invalid
    """
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")
    
    # Check size limit
    response_str = json.dumps(response, default=str)
    if len(response_str) > max_size:
        raise SecurityError("Response too large")
    
    # Check for script injection
    if '<script>' in response_str or 'javascript:' in response_str:
        raise SecurityError("Script injection detected")
    
    # Check for path traversal
    if '..' in response_str:
        raise SecurityError("Path traversal detected")
    
    # Check for potential SQL injection patterns
    sql_patterns = ['union select', 'drop table', 'delete from', '; --', '/*', '*/']
    response_lower = response_str.lower()
    for pattern in sql_patterns:
        if pattern in response_lower:
            raise SecurityError(f"SQL injection pattern detected: {pattern}")
    
    return True


def sanitize_text_content(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text content from external sources.
    
    Args:
        text: Text content to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
        
    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")
    
    # Length check
    if len(text) > max_length:
        raise ValidationError(f"Text too long: {len(text)} > {max_length}")
    
    # Remove dangerous HTML/script tags
    dangerous_tags = ['<script>', '</script>', '<iframe>', '</iframe>', 
                     '<object>', '</object>', '<embed>', '</embed>']
    
    for tag in dangerous_tags:
        text = text.replace(tag, '')
    
    # Remove potential XSS patterns
    xss_patterns = ['javascript:', 'data:text/html', 'vbscript:', 'onload=', 'onerror=']
    for pattern in xss_patterns:
        text = text.replace(pattern, '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text