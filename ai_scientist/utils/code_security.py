"""
Code security utilities for validating and sanitizing LLM-generated code before execution.

This module provides comprehensive security measures to prevent dangerous operations
in LLM-generated code while allowing legitimate scientific computing tasks.
"""

import ast
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Set, List, Tuple, Dict, Any, Optional
import resource
import signal
import sys
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class SecurityViolation(Exception):
    """Raised when code violates security policies."""
    pass

class CodeSecurityValidator:
    """Validates and sanitizes LLM-generated code for safe execution."""
    
    # Dangerous operations that should be blocked
    DANGEROUS_IMPORTS = {
        'subprocess', 'os', 'sys', 'shutil', 'glob', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'telnetlib', 'smtplib', 'poplib',
        'imaplib', 'ssl', 'multiprocessing', 'threading', 'ctypes',
        'importlib', 'pkgutil', 'runpy', 'code', 'codeop', 'compile',
        'eval', 'exec', '__import__'
    }
    
    # Allowed safe imports for scientific computing
    ALLOWED_IMPORTS = {
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
        'torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras',
        'PIL', 'cv2', 'skimage', 'plotly', 'bokeh', 'altair',
        'json', 'pickle', 'csv', 'gzip', 'zipfile', 'tarfile',
        'datetime', 'time', 'random', 'math', 'statistics', 'collections',
        'itertools', 'functools', 'operator', 'copy', 'deepcopy',
        'pathlib', 'tempfile', 'logging', 're', 'string', 'unicodedata',
        'base64', 'hashlib', 'hmac', 'secrets', 'uuid', 'warnings'
    }
    
    # Dangerous function calls
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'exit', 'quit', 'help', 'copyright',
        'credits', 'license', 'reload', 'vars', 'locals', 'globals',
        'dir', 'delattr', 'getattr', 'setattr', 'hasattr'
    }
    
    # Dangerous attributes that access system functionality
    DANGEROUS_ATTRIBUTES = {
        '__import__', '__builtins__', '__globals__', '__locals__',
        '__dict__', '__class__', '__bases__', '__subclasses__',
        '__reduce__', '__reduce_ex__'
    }
    
    # File operations that should be restricted
    RESTRICTED_FILE_OPERATIONS = {
        'open', 'file', 'input', 'raw_input'
    }
    
    def __init__(self, allowed_dirs: Optional[List[str]] = None, max_file_size: int = 10_000_000):
        """Initialize validator with security policies.
        
        Args:
            allowed_dirs: List of directories where file operations are allowed
            max_file_size: Maximum allowed file size for operations
        """
        self.allowed_dirs = set(allowed_dirs or [])
        self.max_file_size = max_file_size
        
    def validate_code(self, code: str, working_dir: str) -> Tuple[bool, List[str]]:
        """Validate code for security violations.
        
        Args:
            code: Python code to validate
            working_dir: Working directory for execution
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return False, violations
            
        # Add working directory to allowed paths
        allowed_paths = self.allowed_dirs.copy()
        allowed_paths.add(working_dir)
        
        # Analyze AST for security violations
        analyzer = SecurityASTAnalyzer(
            dangerous_imports=self.DANGEROUS_IMPORTS,
            allowed_imports=self.ALLOWED_IMPORTS,
            dangerous_functions=self.DANGEROUS_FUNCTIONS,
            dangerous_attributes=self.DANGEROUS_ATTRIBUTES,
            allowed_paths=allowed_paths
        )
        
        analyzer.visit(tree)
        violations.extend(analyzer.violations)
        
        # Additional regex-based checks for patterns AST might miss
        violations.extend(self._regex_security_check(code))
        
        is_valid = len(violations) == 0
        return is_valid, violations
        
    def _regex_security_check(self, code: str) -> List[str]:
        """Additional regex-based security checks."""
        violations = []
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'__import__\s*\(', "Dynamic import detected"),
            (r'exec\s*\(', "Exec call detected"),
            (r'eval\s*\(', "Eval call detected"),
            (r'os\.system\s*\(', "OS system call detected"),
            (r'subprocess\.', "Subprocess usage detected"),
            (r'open\s*\([\'"][\/\\]', "Absolute path file access detected"),
            (r'\.\./', "Path traversal attempt detected"),
            (r'%s|%d|%.*s', "String formatting injection risk"),
            (r'\.format\s*\(.*\{', "Format string injection risk")
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(message)
                
        return violations
        
    def sanitize_environment(self) -> Dict[str, str]:
        """Create sanitized environment for code execution."""
        # Start with minimal environment
        safe_env = {
            'PYTHONPATH': '',
            'PATH': '/usr/bin:/bin',
            'HOME': tempfile.gettempdir(),
            'TMPDIR': tempfile.gettempdir()
        }
        
        # Copy only safe environment variables
        safe_vars = {'LANG', 'LC_ALL', 'TZ'}
        for var in safe_vars:
            if var in os.environ:
                safe_env[var] = os.environ[var]
                
        return safe_env

class SecurityASTAnalyzer(ast.NodeVisitor):
    """AST analyzer to detect security violations."""
    
    def __init__(self, dangerous_imports: Set[str], allowed_imports: Set[str],
                 dangerous_functions: Set[str], dangerous_attributes: Set[str],
                 allowed_paths: Set[str]):
        self.dangerous_imports = dangerous_imports
        self.allowed_imports = allowed_imports
        self.dangerous_functions = dangerous_functions
        self.dangerous_attributes = dangerous_attributes
        self.allowed_paths = allowed_paths
        self.violations = []
        
    def visit_Import(self, node):
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name in self.dangerous_imports:
                self.violations.append(f"Dangerous import: {module_name}")
                
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Check from-import statements."""
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name in self.dangerous_imports:
                self.violations.append(f"Dangerous import from: {module_name}")
                
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Check function calls."""
        # Check direct function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.dangerous_functions:
                self.violations.append(f"Dangerous function call: {node.func.id}")
                
        # Check attribute calls (like os.system)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.dangerous_functions:
                self.violations.append(f"Dangerous method call: {node.func.attr}")
                
        # Check for file operations
        if isinstance(node.func, ast.Name) and node.func.id == 'open':
            if node.args:
                self._check_file_access(node.args[0])
                
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        """Check attribute access."""
        if node.attr in self.dangerous_attributes:
            self.violations.append(f"Dangerous attribute access: {node.attr}")
            
        self.generic_visit(node)
        
    def _check_file_access(self, path_node):
        """Check if file access is to allowed paths."""
        if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
            file_path = path_node.value
        elif hasattr(path_node, 's'):  # Legacy ast.Str support
            file_path = path_node.s
        else:
            # Can't analyze dynamic paths
            self.violations.append("Dynamic file path detected - cannot validate")
            return
            
        # Check if path is allowed
        path_obj = Path(file_path)
        
        # Check for absolute paths outside allowed directories
        if path_obj.is_absolute():
            allowed = any(
                str(path_obj).startswith(allowed_dir) 
                for allowed_dir in self.allowed_paths
            )
            if not allowed:
                self.violations.append(f"File access outside allowed directories: {file_path}")
                
        # Check for path traversal
        if '..' in file_path:
            self.violations.append(f"Path traversal attempt: {file_path}")

@contextmanager
def resource_limited_execution(cpu_limit: int = 30, memory_limit: int = 512 * 1024 * 1024):
    """Context manager for resource-limited code execution.
    
    Args:
        cpu_limit: CPU time limit in seconds
        memory_limit: Memory limit in bytes
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
        
    # Set resource limits
    old_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
    old_mem_limit = resource.getrlimit(resource.RLIMIT_AS)
    old_alarm = signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        signal.alarm(cpu_limit + 5)  # Wall clock limit slightly higher than CPU limit
        yield
    finally:
        # Restore original limits
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        resource.setrlimit(resource.RLIMIT_CPU, old_cpu_limit)
        resource.setrlimit(resource.RLIMIT_AS, old_mem_limit)

def create_restricted_globals(working_dir: str) -> Dict[str, Any]:
    """Create restricted global namespace for code execution."""
    import builtins
    
    # Start with safe builtins
    safe_builtins = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
        'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float',
        'format', 'frozenset', 'hex', 'int', 'isinstance', 'issubclass',
        'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord',
        'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
        'str', 'sum', 'tuple', 'type', 'zip'
    }
    
    restricted_builtins = {}
    for name in safe_builtins:
        if hasattr(builtins, name):
            restricted_builtins[name] = getattr(builtins, name)
    
    # Add safe modules
    safe_modules = {}
    
    # Restricted file operations
    def safe_open(file, mode='r', **kwargs):
        """Restricted open function that only allows access to working directory."""
        file_path = Path(file)
        working_path = Path(working_dir)
        
        # Resolve paths to prevent traversal
        try:
            resolved_file = file_path.resolve()
            resolved_working = working_path.resolve()
            
            # Check if file is within working directory
            if not str(resolved_file).startswith(str(resolved_working)):
                raise PermissionError(f"File access denied: {file}")
                
            # Check file size for write operations
            if 'w' in mode or 'a' in mode:
                if resolved_file.exists() and resolved_file.stat().st_size > 10_000_000:
                    raise PermissionError("File too large")
                    
            return open(file, mode, **kwargs)
        except Exception as e:
            raise PermissionError(f"File access error: {e}")
    
    return {
        '__builtins__': restricted_builtins,
        'open': safe_open,
        '__name__': '__main__',
        '__file__': '<string>',
        '__cached__': None,
    }

def execute_code_safely(code: str, working_dir: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Execute code with comprehensive security measures.
    
    Args:
        code: Python code to execute
        working_dir: Working directory for execution
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    validator = CodeSecurityValidator(allowed_dirs=[working_dir])
    
    # Validate code first
    is_valid, violations = validator.validate_code(code, working_dir)
    if not is_valid:
        error_msg = "Security violations detected:\n" + "\n".join(violations)
        return False, "", error_msg
    
    # Create restricted execution environment
    restricted_globals = create_restricted_globals(working_dir)
    
    # Capture output
    import io
    import contextlib
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with resource_limited_execution(timeout, 512 * 1024 * 1024):
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Change to working directory
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(working_dir)
                        exec(compile(code, '<string>', 'exec'), restricted_globals)
                    finally:
                        os.chdir(old_cwd)
                        
        return True, stdout_capture.getvalue(), stderr_capture.getvalue()
        
    except TimeoutError:
        return False, stdout_capture.getvalue(), "Code execution timed out"
    except MemoryError:
        return False, stdout_capture.getvalue(), "Code execution exceeded memory limit"
    except Exception as e:
        return False, stdout_capture.getvalue(), f"Execution error: {str(e)}"