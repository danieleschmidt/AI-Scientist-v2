#!/usr/bin/env python3
"""
Comprehensive test runner for AI Scientist v2.
Runs different test suites with appropriate configurations.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and handle output."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} - FAILED")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False
    
    return True


def run_unit_tests(coverage=True, verbose=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if coverage:
        cmd.extend(["--cov=ai_scientist", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "unit"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Integration Tests")


def run_security_tests(verbose=False):
    """Run security tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "security"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Security Tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Performance Tests")


def run_gpu_tests(verbose=False):
    """Run GPU tests if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            cmd = ["python", "-m", "pytest", "tests/", "-m", "gpu"]
            
            if verbose:
                cmd.append("-v")
            
            return run_command(cmd, "GPU Tests")
        else:
            print("⚠️  GPU not available, skipping GPU tests")
            return True
    except ImportError:
        print("⚠️  PyTorch not available, skipping GPU tests")
        return True


def run_lint_checks():
    """Run linting checks."""
    checks = [
        (["python", "-m", "black", "--check", "."], "Black formatting check"),
        (["python", "-m", "isort", "--check-only", "."], "Import sorting check"),
        (["python", "-m", "flake8", "."], "Flake8 linting"),
        (["python", "-m", "mypy", "ai_scientist/"], "MyPy type checking"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def run_security_scanning():
    """Run security scanning tools."""
    checks = [
        (["python", "-m", "bandit", "-r", "ai_scientist/", "-f", "json"], "Bandit security scan"),
        (["python", "-m", "safety", "check"], "Safety dependency scan"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="AI Scientist v2 Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--scan", action="store_true", help="Run security scanning")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (unit + lint)")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    print("🧪 AI Scientist v2 Test Runner")
    print(f"📁 Working directory: {repo_root}")
    
    success = True
    
    if args.all or args.fast or args.unit:
        success &= run_unit_tests(
            coverage=not args.no_coverage, 
            verbose=args.verbose
        )
    
    if args.all or args.lint:
        success &= run_lint_checks()
    
    if args.all or args.integration:
        success &= run_integration_tests(verbose=args.verbose)
    
    if args.all or args.security:
        success &= run_security_tests(verbose=args.verbose)
    
    if args.all or args.scan:
        success &= run_security_scanning()
    
    if args.all or args.performance:
        success &= run_performance_tests(verbose=args.verbose)
    
    if args.all or args.gpu:
        success &= run_gpu_tests(verbose=args.verbose)
    
    # If no specific test type is requested, run fast tests
    if not any([args.unit, args.integration, args.security, args.performance, 
                args.gpu, args.lint, args.scan, args.all, args.fast]):
        print("No test type specified, running fast tests (unit + lint)")
        success &= run_unit_tests(
            coverage=not args.no_coverage, 
            verbose=args.verbose
        )
        success &= run_lint_checks()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()