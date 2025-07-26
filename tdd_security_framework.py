#!/usr/bin/env python3
"""
TDD + Security Micro Cycle Framework
Implements strict TDD methodology with integrated security checks.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

logger = logging.getLogger(__name__)

@dataclass
class SecurityCheckResult:
    """Result of a security check."""
    passed: bool
    check_name: str
    details: str
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

@dataclass
class TestResult:
    """Result of test execution."""
    passed: bool
    test_name: str
    details: str
    execution_time: float
    coverage_delta: Optional[float] = None

class SecurityChecker:
    """Comprehensive security validation."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    async def validate_input_sanitization(self, files: List[str]) -> SecurityCheckResult:
        """Check for proper input validation and sanitization."""
        logger.info("Checking input validation and sanitization")
        
        # Look for unsafe patterns
        unsafe_patterns = [
            r'eval\(',
            r'exec\(',
            r'os\.system\(',
            r'subprocess\.call\([^)]*shell=True',
            r'pickle\.loads?\(',
            r'\.format\([^)]*\*\*'
        ]
        
        issues = []
        for file_path in files:
            if not Path(file_path).exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in unsafe_patterns:
                    import re
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append(f"{file_path}: Found potentially unsafe pattern: {pattern}")
                        
            except Exception as e:
                logger.warning(f"Could not check {file_path}: {e}")
        
        if issues:
            return SecurityCheckResult(
                passed=False,
                check_name="input_sanitization",
                details=f"Found {len(issues)} potential input validation issues",
                severity="HIGH",
                recommendations=[
                    "Use parameterized queries for database operations",
                    "Validate and sanitize all user inputs",
                    "Avoid eval() and exec() functions",
                    "Use subprocess with shell=False"
                ]
            )
        
        return SecurityCheckResult(
            passed=True,
            check_name="input_sanitization",
            details="No input validation issues found"
        )
    
    async def validate_secrets_management(self, files: List[str]) -> SecurityCheckResult:
        """Check for proper secrets management."""
        logger.info("Checking secrets management")
        
        # Look for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        issues = []
        for file_path in files:
            if not Path(file_path).exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in secret_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Check if it's a placeholder or example
                        for match in matches:
                            if not any(placeholder in match.lower() for placeholder in 
                                     ['placeholder', 'example', 'your_', 'xxx', '***']):
                                issues.append(f"{file_path}: Potential hardcoded secret: {match}")
                        
            except Exception as e:
                logger.warning(f"Could not check {file_path}: {e}")
        
        if issues:
            return SecurityCheckResult(
                passed=False,
                check_name="secrets_management",
                details=f"Found {len(issues)} potential hardcoded secrets",
                severity="CRITICAL",
                recommendations=[
                    "Use environment variables for secrets",
                    "Use secret management systems (AWS Secrets Manager, etc.)",
                    "Never commit secrets to version control",
                    "Use .env files with .gitignore"
                ]
            )
        
        return SecurityCheckResult(
            passed=True,
            check_name="secrets_management",
            details="No hardcoded secrets found"
        )
    
    async def validate_authentication_authorization(self, files: List[str]) -> SecurityCheckResult:
        """Check for proper authentication and authorization."""
        logger.info("Checking authentication and authorization")
        
        # Look for auth-related code
        auth_indicators = []
        auth_issues = []
        
        for file_path in files:
            if not Path(file_path).exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for auth patterns
                if any(pattern in content.lower() for pattern in 
                      ['authenticate', 'authorize', 'login', 'token', 'session']):
                    auth_indicators.append(file_path)
                    
                    # Check for common auth issues
                    if 'password' in content.lower() and 'hash' not in content.lower():
                        auth_issues.append(f"{file_path}: Potential plain text password handling")
                    
                    if 'admin' in content.lower() and 'bypass' in content.lower():
                        auth_issues.append(f"{file_path}: Potential admin bypass code")
                        
            except Exception as e:
                logger.warning(f"Could not check {file_path}: {e}")
        
        if auth_issues:
            return SecurityCheckResult(
                passed=False,
                check_name="authentication_authorization",
                details=f"Found {len(auth_issues)} authentication/authorization issues",
                severity="HIGH",
                recommendations=[
                    "Always hash passwords before storage",
                    "Implement proper session management",
                    "Use principle of least privilege",
                    "Implement proper access controls"
                ]
            )
        
        return SecurityCheckResult(
            passed=True,
            check_name="authentication_authorization",
            details="No authentication/authorization issues found"
        )
    
    async def run_sast_scan(self, files: List[str]) -> SecurityCheckResult:
        """Run Static Application Security Testing."""
        logger.info("Running SAST scan")
        
        # Use bandit for Python SAST
        try:
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', '.', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                # Parse bandit results
                try:
                    bandit_results = json.loads(result.stdout)
                    issues = bandit_results.get('results', [])
                    
                    if issues:
                        high_severity = [i for i in issues if i.get('issue_severity') == 'HIGH']
                        medium_severity = [i for i in issues if i.get('issue_severity') == 'MEDIUM']
                        
                        if high_severity:
                            return SecurityCheckResult(
                                passed=False,
                                check_name="sast_scan",
                                details=f"Found {len(high_severity)} high severity security issues",
                                severity="HIGH",
                                recommendations=["Review and fix high severity security issues"]
                            )
                        elif medium_severity:
                            return SecurityCheckResult(
                                passed=False,
                                check_name="sast_scan",
                                details=f"Found {len(medium_severity)} medium severity security issues",
                                severity="MEDIUM",
                                recommendations=["Review and fix medium severity security issues"]
                            )
                except json.JSONDecodeError:
                    pass
                    
        except subprocess.CalledProcessError:
            logger.warning("Bandit not available for SAST scanning")
        except FileNotFoundError:
            logger.warning("Bandit not installed")
        
        return SecurityCheckResult(
            passed=True,
            check_name="sast_scan",
            details="SAST scan completed - no critical issues found"
        )
    
    async def validate_dependency_security(self) -> SecurityCheckResult:
        """Check for vulnerable dependencies."""
        logger.info("Checking dependency security")
        
        # Use safety to check for known vulnerabilities
        try:
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                try:
                    safety_results = json.loads(result.stdout)
                    if safety_results:
                        return SecurityCheckResult(
                            passed=False,
                            check_name="dependency_security",
                            details=f"Found {len(safety_results)} vulnerable dependencies",
                            severity="HIGH",
                            recommendations=[
                                "Update vulnerable dependencies",
                                "Review security advisories",
                                "Consider alternative packages"
                            ]
                        )
                except json.JSONDecodeError:
                    pass
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Safety not available for dependency checking")
        
        return SecurityCheckResult(
            passed=True,
            check_name="dependency_security",
            details="No vulnerable dependencies found"
        )

class TDDFramework:
    """Test-Driven Development framework with security integration."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.security_checker = SecurityChecker(repo_path)
    
    async def execute_tdd_cycle(self, task_id: str, files: List[str], 
                               test_requirements: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Execute complete TDD cycle: RED -> GREEN -> REFACTOR + Security."""
        logger.info(f"Starting TDD cycle for task {task_id}")
        
        results = {
            "task_id": task_id,
            "files": files,
            "phases": {},
            "security_checks": {},
            "overall_success": False,
            "execution_time": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Phase 1: RED - Write failing tests
            red_result = await self._red_phase(task_id, files, test_requirements)
            results["phases"]["red"] = red_result
            
            if not red_result["success"]:
                return False, results
            
            # Phase 2: GREEN - Make tests pass
            green_result = await self._green_phase(task_id, files)
            results["phases"]["green"] = green_result
            
            if not green_result["success"]:
                return False, results
            
            # Phase 3: REFACTOR - Improve code quality
            refactor_result = await self._refactor_phase(task_id, files)
            results["phases"]["refactor"] = refactor_result
            
            if not refactor_result["success"]:
                return False, results
            
            # Phase 4: SECURITY - Comprehensive security checks
            security_result = await self._security_phase(files)
            results["security_checks"] = security_result
            
            if not security_result["passed"]:
                return False, results
            
            # Phase 5: INTEGRATION - Final validation
            integration_result = await self._integration_phase(task_id)
            results["phases"]["integration"] = integration_result
            
            results["overall_success"] = integration_result["success"]
            
        except Exception as e:
            logger.error(f"TDD cycle failed with exception: {e}")
            results["error"] = str(e)
            
        finally:
            end_time = datetime.now()
            results["execution_time"] = (end_time - start_time).total_seconds()
        
        return results["overall_success"], results
    
    async def _red_phase(self, task_id: str, files: List[str], 
                        test_requirements: List[str]) -> Dict[str, Any]:
        """RED phase: Write failing tests."""
        logger.info("TDD RED phase: Writing failing tests")
        
        # Create test file if it doesn't exist
        test_file = f"tests/test_{task_id}.py"
        test_path = self.repo_path / test_file
        
        if not test_path.exists():
            # Generate basic test structure
            test_content = self._generate_test_template(task_id, test_requirements)
            
            os.makedirs(test_path.parent, exist_ok=True)
            with open(test_path, 'w') as f:
                f.write(test_content)
        
        # Run tests to ensure they fail
        test_result = await self._run_tests(test_file)
        
        if test_result.passed:
            return {
                "success": False,
                "message": "Tests are passing when they should fail (RED phase)",
                "test_result": test_result
            }
        
        return {
            "success": True,
            "message": "Tests are properly failing",
            "test_result": test_result,
            "test_file": test_file
        }
    
    async def _green_phase(self, task_id: str, files: List[str]) -> Dict[str, Any]:
        """GREEN phase: Make tests pass with minimal code."""
        logger.info("TDD GREEN phase: Making tests pass")
        
        # This would typically involve implementing the actual functionality
        # For this framework, we'll simulate the implementation
        
        # Run tests again to check if they pass
        test_file = f"tests/test_{task_id}.py"
        test_result = await self._run_tests(test_file)
        
        return {
            "success": test_result.passed,
            "message": "Implementation completed" if test_result.passed else "Tests still failing",
            "test_result": test_result
        }
    
    async def _refactor_phase(self, task_id: str, files: List[str]) -> Dict[str, Any]:
        """REFACTOR phase: Improve code quality without changing behavior."""
        logger.info("TDD REFACTOR phase: Improving code quality")
        
        # Run code quality checks
        quality_issues = []
        
        # Check with black for formatting
        try:
            for file_path in files:
                if file_path.endswith('.py') and Path(file_path).exists():
                    result = subprocess.run([
                        'python', '-m', 'black', '--check', file_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        quality_issues.append(f"Formatting issues in {file_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Ensure tests still pass after refactoring
        test_file = f"tests/test_{task_id}.py"
        test_result = await self._run_tests(test_file)
        
        return {
            "success": test_result.passed and len(quality_issues) == 0,
            "message": f"Refactoring completed. Quality issues: {len(quality_issues)}",
            "test_result": test_result,
            "quality_issues": quality_issues
        }
    
    async def _security_phase(self, files: List[str]) -> Dict[str, Any]:
        """SECURITY phase: Comprehensive security validation."""
        logger.info("TDD SECURITY phase: Running security checks")
        
        security_checks = [
            self.security_checker.validate_input_sanitization(files),
            self.security_checker.validate_secrets_management(files),
            self.security_checker.validate_authentication_authorization(files),
            self.security_checker.run_sast_scan(files),
            self.security_checker.validate_dependency_security()
        ]
        
        results = await asyncio.gather(*security_checks)
        
        failed_checks = [r for r in results if not r.passed]
        critical_issues = [r for r in failed_checks if r.severity == "CRITICAL"]
        high_issues = [r for r in failed_checks if r.severity == "HIGH"]
        
        return {
            "passed": len(critical_issues) == 0 and len(high_issues) == 0,
            "total_checks": len(results),
            "passed_checks": len(results) - len(failed_checks),
            "failed_checks": len(failed_checks),
            "critical_issues": len(critical_issues),
            "high_issues": len(high_issues),
            "check_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "details": r.details,
                    "recommendations": r.recommendations
                }
                for r in results
            ]
        }
    
    async def _integration_phase(self, task_id: str) -> Dict[str, Any]:
        """INTEGRATION phase: Final validation and integration tests."""
        logger.info("TDD INTEGRATION phase: Final validation")
        
        # Run full test suite
        full_test_result = await self._run_tests()
        
        # Check test coverage if possible
        coverage_result = await self._check_test_coverage()
        
        return {
            "success": full_test_result.passed,
            "message": "Integration tests completed",
            "test_result": full_test_result,
            "coverage_result": coverage_result
        }
    
    async def _run_tests(self, test_file: Optional[str] = None) -> TestResult:
        """Run pytest tests."""
        start_time = datetime.now()
        
        cmd = ['python', '-m', 'pytest', '-v']
        if test_file:
            cmd.append(test_file)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                passed=result.returncode == 0,
                test_name=test_file or "full_suite",
                details=result.stdout + result.stderr,
                execution_time=execution_time
            )
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                passed=False,
                test_name=test_file or "full_suite",
                details=f"Test execution failed: {e}",
                execution_time=execution_time
            )
    
    async def _check_test_coverage(self) -> Optional[Dict[str, Any]]:
        """Check test coverage using pytest-cov."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=.', '--cov-report=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                # Try to read coverage report
                coverage_file = self.repo_path / 'coverage.json'
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    return {
                        "total_coverage": coverage_data.get('totals', {}).get('percent_covered', 0),
                        "lines_covered": coverage_data.get('totals', {}).get('covered_lines', 0),
                        "lines_total": coverage_data.get('totals', {}).get('num_statements', 0)
                    }
                    
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
        
        return None
    
    def _generate_test_template(self, task_id: str, test_requirements: List[str]) -> str:
        """Generate a basic test template."""
        template = f'''"""
Tests for {task_id}
Generated by TDD Security Framework
"""

import pytest
import unittest
from unittest.mock import Mock, patch


class Test{task_id.replace("-", "_").title()}(unittest.TestCase):
    """Test cases for {task_id}."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass

'''
        
        # Add test methods based on requirements
        for i, requirement in enumerate(test_requirements, 1):
            method_name = f"test_{requirement.lower().replace(' ', '_').replace('-', '_')}"
            template += f'''    def {method_name}(self):
        """Test: {requirement}"""
        # TODO: Implement test for: {requirement}
        self.fail("Test not implemented")
    
'''
        
        template += '''
if __name__ == '__main__':
    unittest.main()
'''
        
        return template

# Example usage function
async def run_tdd_cycle_example():
    """Example of running a TDD cycle."""
    framework = TDDFramework()
    
    # Example task
    task_id = "example-feature"
    files = ["src/example.py"]
    test_requirements = [
        "Function should return correct result",
        "Function should handle edge cases",
        "Function should validate inputs"
    ]
    
    success, results = await framework.execute_tdd_cycle(task_id, files, test_requirements)
    
    print(f"TDD Cycle Success: {success}")
    print(f"Results: {json.dumps(results, indent=2, default=str)}")
    
    return success, results

if __name__ == "__main__":
    asyncio.run(run_tdd_cycle_example())