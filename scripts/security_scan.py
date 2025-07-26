#!/usr/bin/env python3
"""
Security scanning script for dependency management.
Part of the autonomous backlog management system.
"""

import subprocess
import sys
from pathlib import Path


def run_safety_check():
    """Run safety check for known vulnerabilities."""
    print("🔍 Running safety check for known vulnerabilities...")
    try:
        result = subprocess.run([
            'python', '-m', 'pip', 'install', 'safety', '--quiet'
        ], capture_output=True, text=True)
        
        result = subprocess.run([
            'safety', 'check', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ No known vulnerabilities found")
        else:
            print("⚠️  Security vulnerabilities detected:")
            print(result.stdout)
            return False
            
    except FileNotFoundError:
        print("⚠️  Safety tool not available. Install with: pip install safety")
        return False
    
    return True


def run_bandit_scan():
    """Run bandit security linter on code."""
    print("🔍 Running bandit security scan...")
    try:
        result = subprocess.run([
            'python', '-m', 'pip', 'install', 'bandit', '--quiet'
        ], capture_output=True, text=True)
        
        result = subprocess.run([
            'bandit', '-r', 'ai_scientist/', '-f', 'txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ No security issues found in code")
        else:
            print("⚠️  Security issues detected:")
            print(result.stdout)
            return False
            
    except FileNotFoundError:
        print("⚠️  Bandit tool not available. Install with: pip install bandit")
        return False
    
    return True


def main():
    """Run all security scans."""
    print("🛡️  AI Scientist Security Scan")
    print("=" * 40)
    
    safety_ok = run_safety_check()
    bandit_ok = run_bandit_scan()
    
    print("\n" + "=" * 40)
    if safety_ok and bandit_ok:
        print("✅ All security scans passed")
        sys.exit(0)
    else:
        print("❌ Security issues detected - please review")
        sys.exit(1)


if __name__ == '__main__':
    main()