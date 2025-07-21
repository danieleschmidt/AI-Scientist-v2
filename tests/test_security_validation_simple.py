#!/usr/bin/env python3
"""
Simple tests for code security validation.
"""

import os
import sys
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_scientist.utils.code_security import CodeSecurityValidator

def test_basic_security():
    """Test basic security validation."""
    print("Testing basic security validation...")
    
    temp_dir = tempfile.mkdtemp()
    validator = CodeSecurityValidator(allowed_dirs=[temp_dir])
    
    # Test 1: Safe code should pass
    safe_code = """
import numpy as np
x = np.array([1, 2, 3])
print(f"Array: {x}")
    """
    
    is_valid, violations = validator.validate_code(safe_code, temp_dir)
    print(f"Safe code test: {'PASS' if is_valid else 'FAIL'}")
    if violations:
        print(f"  Violations: {violations}")
    
    # Test 2: Dangerous import should be blocked
    dangerous_code = "import os; os.system('echo test')"
    
    is_valid, violations = validator.validate_code(dangerous_code, temp_dir)
    print(f"Dangerous import test: {'PASS' if not is_valid else 'FAIL'}")
    if violations:
        print(f"  Violations: {violations}")
    
    # Test 3: Path traversal should be blocked
    path_traversal = "open('../../../etc/passwd', 'r')"
    
    is_valid, violations = validator.validate_code(path_traversal, temp_dir)
    print(f"Path traversal test: {'PASS' if not is_valid else 'FAIL'}")
    if violations:
        print(f"  Violations: {violations}")
    
    # Test 4: Eval should be blocked
    eval_code = "eval('print(1)')"
    
    is_valid, violations = validator.validate_code(eval_code, temp_dir)
    print(f"Eval blocking test: {'PASS' if not is_valid else 'FAIL'}")
    if violations:
        print(f"  Violations: {violations}")
    
    print("Security validation tests completed!")

if __name__ == '__main__':
    test_basic_security()