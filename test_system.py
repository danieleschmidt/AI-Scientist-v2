#!/usr/bin/env python3
"""
Simple test script to validate the autonomous system structure.
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        'autonomous_backlog_manager.py',
        'tdd_security_framework.py',
        'metrics_reporter.py',
        'autonomous_launcher.py',
        'DOCS/backlog.yml',
        '.gitattributes',
        'scripts/git_hooks/prepare-commit-msg',
        'scripts/git_hooks/pre-push',
        'docs/autonomous_backlog_management_guide.md',
        'docs/ci_supply_chain_security.md',
        'AUTONOMOUS_SYSTEM_README.md'
    ]
    
    print("Autonomous Backlog Management System - Structure Test")
    print("=" * 55)
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    print(f"\nOverall Status: {'✓ PASS' if all_exist else '✗ FAIL'}")
    return all_exist

def test_git_configuration():
    """Test git configuration."""
    print("\nGit Configuration Test")
    print("-" * 25)
    
    try:
        import subprocess
        
        # Test git rerere
        result = subprocess.run(['git', 'config', 'rerere.enabled'], 
                              capture_output=True, text=True)
        rerere_enabled = result.stdout.strip() == 'true'
        print(f"  {'✓' if rerere_enabled else '✗'} Git rerere enabled: {rerere_enabled}")
        
        # Test merge drivers
        result = subprocess.run(['git', 'config', 'merge.theirs.driver'], 
                              capture_output=True, text=True)
        theirs_driver = bool(result.stdout.strip())
        print(f"  {'✓' if theirs_driver else '✗'} Merge driver (theirs) configured: {theirs_driver}")
        
        result = subprocess.run(['git', 'config', 'merge.union.driver'], 
                              capture_output=True, text=True)
        union_driver = bool(result.stdout.strip())
        print(f"  {'✓' if union_driver else '✗'} Merge driver (union) configured: {union_driver}")
        
        return rerere_enabled and theirs_driver and union_driver
        
    except Exception as e:
        print(f"  ✗ Git configuration test failed: {e}")
        return False

def test_python_modules():
    """Test Python module availability."""
    print("\nPython Dependencies Test")
    print("-" * 28)
    
    # Test core Python modules
    core_modules = ['json', 'os', 'sys', 'pathlib', 'datetime', 'subprocess', 'logging']
    all_core_available = True
    
    for module in core_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module}")
            all_core_available = False
    
    # Test optional modules (nice to have but not required for basic functionality)
    optional_modules = ['yaml', 'asyncio', 'dataclasses', 'enum', 'statistics']
    optional_available = 0
    
    print("\nOptional Dependencies:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
            optional_available += 1
        except ImportError:
            print(f"  ✗ {module} (install with: pip install PyYAML)")
    
    print(f"\nCore modules: {'✓ PASS' if all_core_available else '✗ FAIL'}")
    print(f"Optional modules: {optional_available}/{len(optional_modules)} available")
    
    return all_core_available

def test_backlog_structure():
    """Test backlog file structure."""
    print("\nBacklog Structure Test")
    print("-" * 22)
    
    backlog_file = Path('DOCS/backlog.yml')
    if not backlog_file.exists():
        print("  ✗ Backlog file not found")
        return False
    
    try:
        with open(backlog_file, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = ['metadata:', 'scoring_criteria:', 'backlog_items:']
        all_sections_present = True
        
        for section in required_sections:
            if section in content:
                print(f"  ✓ {section}")
            else:
                print(f"  ✗ {section}")
                all_sections_present = False
        
        print(f"\nBacklog structure: {'✓ PASS' if all_sections_present else '✗ FAIL'}")
        return all_sections_present
        
    except Exception as e:
        print(f"  ✗ Error reading backlog file: {e}")
        return False

def generate_installation_guide():
    """Generate installation instructions based on test results."""
    print("\n" + "=" * 55)
    print("INSTALLATION GUIDE")
    print("=" * 55)
    
    print("\n1. Install Python dependencies:")
    print("   pip install PyYAML")
    print("   pip install bandit safety pytest-cov  # Optional security tools")
    
    print("\n2. Configure Git (if not already done):")
    print("   git config rerere.enabled true")
    print("   git config rerere.autoupdate true")
    print("   git config merge.theirs.name 'Prefer incoming'")
    print("   git config merge.theirs.driver \"cp -f '%B' '%A'\"")
    print("   git config merge.union.name 'Line union'")
    print("   git config merge.union.driver \"git merge-file -p %A %O %B > %A\"")
    
    print("\n3. Install Git hooks:")
    print("   cp scripts/git_hooks/* .git/hooks/")
    print("   chmod +x .git/hooks/*")
    
    print("\n4. Test the system:")
    print("   python3 test_system.py  # Run this test again")
    
    print("\n5. Basic usage:")
    print("   python3 autonomous_launcher.py --mode health")
    print("   python3 autonomous_launcher.py --mode discovery")
    print("   python3 autonomous_launcher.py --mode single")

def main():
    """Main test function."""
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print()
    
    tests = [
        test_file_structure,
        test_git_configuration, 
        test_python_modules,
        test_backlog_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append(False)
    
    # Overall summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ System is ready for autonomous execution!")
        print("\nNext steps:")
        print("  python3 autonomous_launcher.py --mode health")
    else:
        print("✗ System needs configuration")
        generate_installation_guide()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)