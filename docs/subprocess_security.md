# Secure Subprocess Wrapper

The AI Scientist v2 includes a secure subprocess wrapper to prevent command injection and enforce security policies across all subprocess operations.

## Overview

The `SecureSubprocess` wrapper provides:
- Command injection prevention
- Path traversal protection
- Working directory validation
- Environment variable sanitization
- Command whitelist enforcement
- Timeout management

## Usage

### Basic Usage

```python
from ai_scientist.utils.secure_subprocess import SecureSubprocess

# Secure subprocess execution
result = SecureSubprocess.run(['echo', 'hello'], timeout=30)
print(result.stdout)
```

### Convenience Functions

```python
from ai_scientist.utils.secure_subprocess import secure_run, secure_call

# Convenience wrappers
result = secure_run(['ls', '-la'], capture_output=True)
exit_code = secure_call(['git', 'status'])
```

## Security Features

### Command Whitelist

Only approved executables are allowed:
- Standard utilities: `echo`, `ls`, `cat`, `grep`, `find`, `head`, `tail`, `wc`, `sort`
- Development tools: `python`, `python3`, `pip`, `git`
- LaTeX tools: `pdflatex`, `bibtex`, `makeindex`, `chktex`
- PDF tools: `pdftocairo`, `pdftoppm`, `pdftotext`
- File operations: `mkdir`, `cp`, `mv`, `rm`, `touch`, `chmod`

### Command Injection Prevention

Dangerous patterns are blocked:
- Shell metacharacters: `;`, `&`, `|`, `` ` ``, `$`
- Path traversal: `../`, `..\\`
- System directory access: `/etc/`, `/bin/`, `/usr/bin/`
- Privilege escalation: `sudo`, `su`
- Code execution: `exec()`, `eval()`, `__import__`

### Example Blocked Commands

```python
# These will raise SecurityError
SecureSubprocess.run(['echo', 'test; rm -rf /'])  # Command injection
SecureSubprocess.run(['cat', '../../../etc/passwd'])  # Path traversal
SecureSubprocess.run(['/bin/sh', '-c', 'malicious'])  # Shell execution
SecureSubprocess.run(['python', '-c', 'import os; os.system("rm -rf /")'])  # Dangerous Python
```

### Working Directory Validation

Dangerous working directories are blocked:
- Root directory: `/`
- System directories: `/etc`, `/bin`, `/usr/bin`, `/sbin`
- User root directory: `/root`

### Environment Variable Sanitization

Dangerous environment variables are removed:
- `LD_PRELOAD`, `LD_LIBRARY_PATH`
- `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`
- `PYTHONPATH`, `PERL5LIB`, `RUBYLIB`

The `PATH` variable is sanitized to remove dangerous directories.

## Error Handling

The wrapper raises specific exceptions:

```python
from ai_scientist.utils.secure_subprocess import SecurityError

try:
    SecureSubprocess.run(['dangerous_command'])
except SecurityError as e:
    print(f"Security violation: {e}")
except subprocess.TimeoutExpired as e:
    print(f"Command timed out: {e}")
```

## Integration Guide

### Current State

The secure subprocess wrapper is implemented and tested. The following files have been identified for future integration:

- `ai_scientist/perform_icbinb_writeup.py`
- `ai_scientist/perform_plotting.py`
- `ai_scientist/perform_writeup.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/utils/gpu_cleanup.py`

### Migration Steps

1. Import the secure wrapper:
   ```python
   from ai_scientist.utils.secure_subprocess import SecureSubprocess
   ```

2. Replace `subprocess.run()` calls:
   ```python
   # Before
   result = subprocess.run(['git', 'status'], capture_output=True)
   
   # After
   result = SecureSubprocess.run(['git', 'status'], capture_output=True)
   ```

3. Handle security exceptions:
   ```python
   try:
       result = SecureSubprocess.run(cmd)
   except SecurityError as e:
       logger.error(f"Command blocked for security: {e}")
       # Handle securely or fail safely
   ```

## Configuration

### Adding Allowed Executables

To add new executables to the whitelist, update `ALLOWED_EXECUTABLES` in `secure_subprocess.py`:

```python
ALLOWED_EXECUTABLES = {
    # ... existing executables ...
    'new_tool',  # Add new approved executable
}
```

### Customizing Security Policies

Security patterns can be customized by modifying:
- `DANGEROUS_PATTERNS`: Regex patterns for dangerous commands
- `DANGEROUS_ENV_VARS`: Environment variables to sanitize
- `SAFE_WORKING_DIRS`: Allowed working directory prefixes

## Testing

The security wrapper includes comprehensive tests:

```bash
python3 tests/test_subprocess_security.py
```

Test coverage includes:
- Command injection prevention
- Path traversal protection
- Working directory validation
- Timeout enforcement
- Environment variable sanitization
- Safe command execution

## Security Considerations

1. **Whitelist Approach**: Only explicitly allowed executables can run
2. **Defense in Depth**: Multiple layers of validation (command, args, environment, working directory)
3. **Fail Secure**: Unknown or suspicious commands are blocked by default
4. **Audit Trail**: All security decisions are logged
5. **Timeout Protection**: Prevents resource exhaustion attacks

## Best Practices

1. **Use the secure wrapper** for all subprocess operations
2. **Handle SecurityError exceptions** gracefully
3. **Log security violations** for monitoring
4. **Test command validation** when adding new subprocess calls
5. **Review whitelist additions** carefully before approval