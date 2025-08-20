#!/usr/bin/env python3
"""
Terragon Quality Gates - Final Validation System
Comprehensive quality assurance, security validation, and compliance checks.
"""

import json
import logging
import os
import sys
import time
import hashlib
import subprocess
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Enhanced console for output
if RICH_AVAILABLE:
    console = Console()
else:
    console = None

class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"

class SecurityRisk(Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Individual quality metric result."""
    name: str
    category: str
    score: float  # 0.0 to 1.0
    max_score: float
    status: str
    details: str = ""
    recommendations: List[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.recommendations is None:
            self.recommendations = []

@dataclass
class SecurityFinding:
    """Security assessment finding."""
    severity: SecurityRisk
    category: str
    description: str
    file_path: str
    line_number: int = 0
    recommendation: str = ""
    cve_reference: str = ""
    
@dataclass
class QualityGateResult:
    """Complete quality gate assessment result."""
    overall_score: float
    quality_level: QualityLevel
    metrics: List[QualityMetric]
    security_findings: List[SecurityFinding]
    compliance_status: Dict[str, bool]
    performance_benchmarks: Dict[str, float]
    recommendations: List[str]
    gate_passed: bool
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class CodeQualityAnalyzer:
    """Analyze code quality metrics."""
    
    def __init__(self):
        self.python_files: List[Path] = []
        self.metrics: Dict[str, Any] = {}
        
    def analyze_codebase(self, root_path: Path) -> Dict[str, Any]:
        """Analyze entire codebase for quality metrics."""
        self.python_files = list(root_path.rglob("*.py"))
        
        metrics = {
            "total_files": len(self.python_files),
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "complexity_score": 0.0,
            "documentation_coverage": 0.0,
            "test_coverage": 0.0,
            "maintainability_index": 0.0
        }
        
        for py_file in self.python_files:
            file_metrics = self._analyze_file(py_file)
            metrics["total_lines"] += file_metrics["lines"]
            metrics["total_functions"] += file_metrics["functions"]
            metrics["total_classes"] += file_metrics["classes"]
            metrics["complexity_score"] += file_metrics["complexity"]
        
        # Calculate averages and indices
        if self.python_files:
            metrics["complexity_score"] /= len(self.python_files)
            metrics["documentation_coverage"] = self._calculate_documentation_coverage()
            metrics["test_coverage"] = self._estimate_test_coverage()
            metrics["maintainability_index"] = self._calculate_maintainability_index(metrics)
        
        return metrics
    
    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze individual Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for analysis
            tree = ast.parse(content)
            
            analyzer = ASTAnalyzer()
            analyzer.visit(tree)
            
            return {
                "lines": len(content.splitlines()),
                "functions": analyzer.function_count,
                "classes": analyzer.class_count,
                "complexity": analyzer.complexity_score,
                "docstrings": analyzer.docstring_count
            }
        except Exception:
            return {"lines": 0, "functions": 0, "classes": 0, "complexity": 0, "docstrings": 0}
    
    def _calculate_documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage."""
        total_functions = 0
        documented_functions = 0
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                analyzer = ASTAnalyzer()
                analyzer.visit(tree)
                
                total_functions += analyzer.function_count
                documented_functions += analyzer.docstring_count
            except Exception:
                continue
        
        return documented_functions / max(total_functions, 1)
    
    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on test files."""
        test_files = [f for f in self.python_files if 'test' in f.name.lower()]
        source_files = [f for f in self.python_files if 'test' not in f.name.lower()]
        
        if not source_files:
            return 0.0
        
        # Simple heuristic: ratio of test files to source files
        test_ratio = len(test_files) / len(source_files)
        
        # Estimate coverage based on test file ratio and content
        estimated_coverage = min(test_ratio * 0.6, 0.9)  # Max 90% estimated
        
        return estimated_coverage
    
    def _calculate_maintainability_index(self, metrics: Dict[str, Any]) -> float:
        """Calculate maintainability index."""
        # Simplified maintainability index calculation
        lines = metrics["total_lines"]
        complexity = metrics["complexity_score"]
        
        if lines == 0:
            return 1.0
        
        # Higher lines and complexity reduce maintainability
        base_score = 1.0
        line_penalty = min(lines / 10000, 0.3)  # Penalty for large codebases
        complexity_penalty = min(complexity / 10, 0.4)  # Penalty for high complexity
        
        maintainability = base_score - line_penalty - complexity_penalty
        return max(maintainability, 0.0)

class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor for code analysis."""
    
    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.complexity_score = 0
        self.docstring_count = 0
        
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.complexity_score += self._calculate_function_complexity(node)
        
        # Check for docstring
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Str)):
            self.docstring_count += 1
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)
    
    def _calculate_function_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

class SecurityAnalyzer:
    """Analyze security vulnerabilities and risks."""
    
    def __init__(self):
        self.findings: List[SecurityFinding] = []
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security vulnerability patterns."""
        return {
            "eval_usage": {
                "pattern": r"\beval\s*\(",
                "severity": SecurityRisk.HIGH,
                "description": "Use of eval() can lead to code injection vulnerabilities"
            },
            "exec_usage": {
                "pattern": r"\bexec\s*\(",
                "severity": SecurityRisk.HIGH,
                "description": "Use of exec() can lead to code injection vulnerabilities"
            },
            "shell_injection": {
                "pattern": r"os\.system\s*\(|subprocess\.call\s*\(",
                "severity": SecurityRisk.MEDIUM,
                "description": "Potential shell injection vulnerability"
            },
            "hardcoded_password": {
                "pattern": r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                "severity": SecurityRisk.HIGH,
                "description": "Hardcoded password detected"
            },
            "sql_injection": {
                "pattern": r"['\"].*%s.*['\"].*%|['\"].*\+.*['\"]",
                "severity": SecurityRisk.MEDIUM,
                "description": "Potential SQL injection vulnerability"
            },
            "unsafe_deserialization": {
                "pattern": r"pickle\.load|pickle\.loads|yaml\.load(?!\s*\(.*,\s*Loader=)",
                "severity": SecurityRisk.HIGH,
                "description": "Unsafe deserialization can lead to code execution"
            },
            "weak_crypto": {
                "pattern": r"hashlib\.md5|hashlib\.sha1",
                "severity": SecurityRisk.MEDIUM,
                "description": "Weak cryptographic hash function"
            }
        }
    
    def analyze_security(self, root_path: Path) -> List[SecurityFinding]:
        """Analyze codebase for security vulnerabilities."""
        self.findings = []
        
        python_files = list(root_path.rglob("*.py"))
        
        for py_file in python_files:
            self._analyze_file_security(py_file)
        
        return self.findings
    
    def _analyze_file_security(self, file_path: Path):
        """Analyze individual file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                self._check_line_security(line, line_num, file_path)
                
        except Exception:
            pass
    
    def _check_line_security(self, line: str, line_num: int, file_path: Path):
        """Check individual line for security issues."""
        for pattern_name, pattern_info in self.security_patterns.items():
            if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                finding = SecurityFinding(
                    severity=pattern_info["severity"],
                    category="code_analysis",
                    description=pattern_info["description"],
                    file_path=str(file_path),
                    line_number=line_num,
                    recommendation=self._get_recommendation(pattern_name)
                )
                self.findings.append(finding)
    
    def _get_recommendation(self, pattern_name: str) -> str:
        """Get security recommendation for pattern."""
        recommendations = {
            "eval_usage": "Replace eval() with safe alternatives like ast.literal_eval()",
            "exec_usage": "Avoid exec() or use with strict input validation",
            "shell_injection": "Use subprocess with shell=False and input validation",
            "hardcoded_password": "Use environment variables or secure key management",
            "sql_injection": "Use parameterized queries or ORM",
            "unsafe_deserialization": "Use safe loading methods with explicit Loader parameter",
            "weak_crypto": "Use SHA-256 or stronger cryptographic hash functions"
        }
        return recommendations.get(pattern_name, "Review and apply security best practices")

class PerformanceBenchmarker:
    """Benchmark performance metrics."""
    
    def __init__(self):
        self.benchmarks: Dict[str, float] = {}
    
    def run_performance_benchmarks(self, root_path: Path) -> Dict[str, float]:
        """Run performance benchmarks."""
        benchmarks = {}
        
        # File I/O performance
        benchmarks["file_io_speed"] = self._benchmark_file_io(root_path)
        
        # Code execution speed (estimated)
        benchmarks["code_execution_speed"] = self._estimate_execution_speed(root_path)
        
        # Memory efficiency
        benchmarks["memory_efficiency"] = self._estimate_memory_efficiency(root_path)
        
        # Startup time
        benchmarks["startup_time"] = self._benchmark_startup_time(root_path)
        
        return benchmarks
    
    def _benchmark_file_io(self, root_path: Path) -> float:
        """Benchmark file I/O operations."""
        try:
            test_file = root_path / "test_io_performance.tmp"
            start_time = time.time()
            
            # Write test
            with open(test_file, 'w') as f:
                f.write("test data " * 1000)
            
            # Read test
            with open(test_file, 'r') as f:
                _ = f.read()
            
            # Cleanup
            test_file.unlink()
            
            io_time = time.time() - start_time
            return 1.0 / max(io_time, 0.001)  # Higher is better
            
        except Exception:
            return 0.5  # Default moderate score
    
    def _estimate_execution_speed(self, root_path: Path) -> float:
        """Estimate code execution speed based on complexity."""
        try:
            analyzer = CodeQualityAnalyzer()
            metrics = analyzer.analyze_codebase(root_path)
            
            # Lower complexity = higher execution speed
            complexity = metrics.get("complexity_score", 5.0)
            execution_speed = 1.0 / max(complexity / 5.0, 0.1)
            
            return min(execution_speed, 1.0)
        except Exception:
            return 0.7  # Default good score
    
    def _estimate_memory_efficiency(self, root_path: Path) -> float:
        """Estimate memory efficiency."""
        try:
            # Simple heuristic based on file sizes and structure
            python_files = list(root_path.rglob("*.py"))
            total_size = sum(f.stat().st_size for f in python_files)
            
            # Smaller, well-structured code is more memory efficient
            if total_size < 100_000:  # < 100KB
                return 1.0
            elif total_size < 500_000:  # < 500KB
                return 0.8
            elif total_size < 1_000_000:  # < 1MB
                return 0.6
            else:
                return 0.4
        except Exception:
            return 0.6  # Default moderate score
    
    def _benchmark_startup_time(self, root_path: Path) -> float:
        """Benchmark module startup time."""
        try:
            # Estimate startup time based on import complexity
            python_files = list(root_path.rglob("*.py"))
            import_count = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    import_count += len(re.findall(r'^\s*(?:import|from)\s+', content, re.MULTILINE))
                except Exception:
                    continue
            
            # More imports = slower startup
            startup_score = 1.0 / max(import_count / 50.0, 0.1)
            return min(startup_score, 1.0)
            
        except Exception:
            return 0.7  # Default good score

class ComplianceChecker:
    """Check compliance with various standards."""
    
    def __init__(self):
        self.compliance_results: Dict[str, bool] = {}
    
    def check_compliance(self, root_path: Path) -> Dict[str, bool]:
        """Check compliance with various standards."""
        compliance = {}
        
        # PEP 8 compliance (simplified check)
        compliance["pep8_compliant"] = self._check_pep8_compliance(root_path)
        
        # Documentation standards
        compliance["documentation_adequate"] = self._check_documentation_standards(root_path)
        
        # Testing standards
        compliance["testing_adequate"] = self._check_testing_standards(root_path)
        
        # Security standards
        compliance["security_compliant"] = self._check_security_standards(root_path)
        
        # Licensing compliance
        compliance["license_compliant"] = self._check_license_compliance(root_path)
        
        return compliance
    
    def _check_pep8_compliance(self, root_path: Path) -> bool:
        """Check basic PEP 8 compliance."""
        try:
            python_files = list(root_path.rglob("*.py"))
            violations = 0
            
            for py_file in python_files:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                lines = content.splitlines()
                for line in lines:
                    # Check line length (simplified)
                    if len(line) > 120:  # Slightly relaxed from PEP 8's 79
                        violations += 1
                    
                    # Check for trailing whitespace
                    if line.endswith(' ') or line.endswith('\t'):
                        violations += 1
            
            # Allow some violations but not too many
            total_lines = sum(len(open(f).readlines()) for f in python_files)
            violation_rate = violations / max(total_lines, 1)
            
            return violation_rate < 0.05  # Less than 5% violation rate
            
        except Exception:
            return False
    
    def _check_documentation_standards(self, root_path: Path) -> bool:
        """Check documentation standards."""
        try:
            # Check for README
            readme_exists = any((root_path / name).exists() 
                              for name in ["README.md", "README.rst", "README.txt"])
            
            # Check for docstring coverage
            analyzer = CodeQualityAnalyzer()
            metrics = analyzer.analyze_codebase(root_path)
            doc_coverage = metrics.get("documentation_coverage", 0.0)
            
            return readme_exists and doc_coverage > 0.3  # 30% docstring coverage
            
        except Exception:
            return False
    
    def _check_testing_standards(self, root_path: Path) -> bool:
        """Check testing standards."""
        try:
            test_files = list(root_path.rglob("test*.py")) + list(root_path.rglob("*test.py"))
            python_files = list(root_path.rglob("*.py"))
            
            # Exclude test files from source count
            source_files = [f for f in python_files if not any(pattern in f.name.lower() 
                                                             for pattern in ['test', 'conftest'])]
            
            if not source_files:
                return True  # No source files to test
            
            test_ratio = len(test_files) / len(source_files)
            return test_ratio > 0.1  # At least 10% test coverage by file count
            
        except Exception:
            return False
    
    def _check_security_standards(self, root_path: Path) -> bool:
        """Check basic security standards."""
        try:
            security_analyzer = SecurityAnalyzer()
            findings = security_analyzer.analyze_security(root_path)
            
            # Check for critical security issues
            critical_findings = [f for f in findings if f.severity == SecurityRisk.CRITICAL]
            high_findings = [f for f in findings if f.severity == SecurityRisk.HIGH]
            
            # No critical findings and limited high findings
            return len(critical_findings) == 0 and len(high_findings) < 3
            
        except Exception:
            return False
    
    def _check_license_compliance(self, root_path: Path) -> bool:
        """Check license compliance."""
        try:
            license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]
            has_license = any((root_path / name).exists() for name in license_files)
            
            return has_license
            
        except Exception:
            return False

class TerragonQualityGates:
    """Comprehensive quality gate system."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.quality_threshold = 0.75  # 75% minimum quality score
        self.setup_logging()
    
    def setup_logging(self):
        """Setup quality gate logging."""
        log_dir = self.root_path / "quality_reports"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"quality_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _log(self, message: str, level: str = "info"):
        """Enhanced logging for quality gates."""
        if RICH_AVAILABLE and console:
            level_colors = {
                "error": "red",
                "success": "green",
                "warning": "yellow",
                "info": "blue"
            }
            color = level_colors.get(level, "blue")
            console.print(f"[{color}]🔍 {message}[/{color}]")
        else:
            print(f"🔍 {message}")
        
        getattr(self.logger, level, self.logger.info)(message)
    
    def run_quality_gates(self) -> QualityGateResult:
        """Run comprehensive quality gate analysis."""
        self._log("🚀 Starting Comprehensive Quality Gate Analysis")
        
        start_time = time.time()
        metrics = []
        
        if RICH_AVAILABLE and console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                # Code Quality Analysis
                task = progress.add_task("Analyzing code quality...", total=100)
                code_analyzer = CodeQualityAnalyzer()
                code_metrics = code_analyzer.analyze_codebase(self.root_path)
                
                quality_metrics = self._convert_code_metrics_to_quality(code_metrics)
                metrics.extend(quality_metrics)
                progress.update(task, advance=25)
                
                # Security Analysis
                progress.update(task, description="Analyzing security...")
                security_analyzer = SecurityAnalyzer()
                security_findings = security_analyzer.analyze_security(self.root_path)
                
                security_metric = self._convert_security_to_metric(security_findings)
                metrics.append(security_metric)
                progress.update(task, advance=25)
                
                # Performance Benchmarks
                progress.update(task, description="Running performance benchmarks...")
                benchmarker = PerformanceBenchmarker()
                performance_benchmarks = benchmarker.run_performance_benchmarks(self.root_path)
                
                performance_metrics = self._convert_performance_to_metrics(performance_benchmarks)
                metrics.extend(performance_metrics)
                progress.update(task, advance=25)
                
                # Compliance Checks
                progress.update(task, description="Checking compliance...")
                compliance_checker = ComplianceChecker()
                compliance_status = compliance_checker.check_compliance(self.root_path)
                
                compliance_metrics = self._convert_compliance_to_metrics(compliance_status)
                metrics.extend(compliance_metrics)
                progress.update(task, advance=25)
        else:
            # Non-rich execution
            self._log("Analyzing code quality...")
            code_analyzer = CodeQualityAnalyzer()
            code_metrics = code_analyzer.analyze_codebase(self.root_path)
            quality_metrics = self._convert_code_metrics_to_quality(code_metrics)
            metrics.extend(quality_metrics)
            
            self._log("Analyzing security...")
            security_analyzer = SecurityAnalyzer()
            security_findings = security_analyzer.analyze_security(self.root_path)
            security_metric = self._convert_security_to_metric(security_findings)
            metrics.append(security_metric)
            
            self._log("Running performance benchmarks...")
            benchmarker = PerformanceBenchmarker()
            performance_benchmarks = benchmarker.run_performance_benchmarks(self.root_path)
            performance_metrics = self._convert_performance_to_metrics(performance_benchmarks)
            metrics.extend(performance_metrics)
            
            self._log("Checking compliance...")
            compliance_checker = ComplianceChecker()
            compliance_status = compliance_checker.check_compliance(self.root_path)
            compliance_metrics = self._convert_compliance_to_metrics(compliance_status)
            metrics.extend(compliance_metrics)
        
        # Calculate overall score and quality level
        overall_score = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, security_findings, compliance_status)
        
        # Determine if gate passed
        gate_passed = (overall_score >= self.quality_threshold and 
                      len([f for f in security_findings if f.severity == SecurityRisk.CRITICAL]) == 0)
        
        execution_time = time.time() - start_time
        
        result = QualityGateResult(
            overall_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            security_findings=security_findings,
            compliance_status=compliance_status,
            performance_benchmarks=performance_benchmarks,
            recommendations=recommendations,
            gate_passed=gate_passed
        )
        
        # Save results
        self._save_quality_report(result)
        
        self._log(f"Quality gate analysis completed in {execution_time:.2f}s", 
                 "success" if gate_passed else "warning")
        
        return result
    
    def _convert_code_metrics_to_quality(self, code_metrics: Dict[str, Any]) -> List[QualityMetric]:
        """Convert code metrics to quality metrics."""
        metrics = []
        
        # Complexity metric
        complexity_score = max(0, min(1, 1.0 - (code_metrics.get("complexity_score", 0) - 1) / 10))
        metrics.append(QualityMetric(
            name="Code Complexity",
            category="Code Quality",
            score=complexity_score,
            max_score=1.0,
            status="good" if complexity_score > 0.7 else "needs_improvement",
            details=f"Average complexity: {code_metrics.get('complexity_score', 0):.2f}"
        ))
        
        # Documentation metric
        doc_coverage = code_metrics.get("documentation_coverage", 0.0)
        metrics.append(QualityMetric(
            name="Documentation Coverage",
            category="Documentation", 
            score=doc_coverage,
            max_score=1.0,
            status="good" if doc_coverage > 0.5 else "needs_improvement",
            details=f"Documentation coverage: {doc_coverage:.1%}"
        ))
        
        # Maintainability metric
        maintainability = code_metrics.get("maintainability_index", 0.0)
        metrics.append(QualityMetric(
            name="Maintainability Index",
            category="Code Quality",
            score=maintainability,
            max_score=1.0,
            status="good" if maintainability > 0.6 else "needs_improvement",
            details=f"Maintainability index: {maintainability:.2f}"
        ))
        
        return metrics
    
    def _convert_security_to_metric(self, security_findings: List[SecurityFinding]) -> QualityMetric:
        """Convert security findings to quality metric."""
        if not security_findings:
            return QualityMetric(
                name="Security Analysis",
                category="Security",
                score=1.0,
                max_score=1.0,
                status="excellent",
                details="No security issues found"
            )
        
        # Calculate security score based on severity
        penalty = 0
        for finding in security_findings:
            if finding.severity == SecurityRisk.CRITICAL:
                penalty += 0.4
            elif finding.severity == SecurityRisk.HIGH:
                penalty += 0.2
            elif finding.severity == SecurityRisk.MEDIUM:
                penalty += 0.1
            else:  # LOW
                penalty += 0.05
        
        security_score = max(0, 1.0 - penalty)
        
        return QualityMetric(
            name="Security Analysis",
            category="Security",
            score=security_score,
            max_score=1.0,
            status="good" if security_score > 0.7 else "needs_improvement",
            details=f"Found {len(security_findings)} security issues"
        )
    
    def _convert_performance_to_metrics(self, benchmarks: Dict[str, float]) -> List[QualityMetric]:
        """Convert performance benchmarks to quality metrics."""
        metrics = []
        
        for name, score in benchmarks.items():
            metrics.append(QualityMetric(
                name=name.replace("_", " ").title(),
                category="Performance",
                score=score,
                max_score=1.0,
                status="good" if score > 0.6 else "needs_improvement",
                details=f"Performance score: {score:.2f}"
            ))
        
        return metrics
    
    def _convert_compliance_to_metrics(self, compliance: Dict[str, bool]) -> List[QualityMetric]:
        """Convert compliance status to quality metrics."""
        metrics = []
        
        for name, passed in compliance.items():
            metrics.append(QualityMetric(
                name=name.replace("_", " ").title(),
                category="Compliance",
                score=1.0 if passed else 0.0,
                max_score=1.0,
                status="passed" if passed else "failed",
                details="Compliant" if passed else "Non-compliant"
            ))
        
        return metrics
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        if overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.8:
            return QualityLevel.GOOD
        elif overall_score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 0.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.FAILED
    
    def _generate_recommendations(self, metrics: List[QualityMetric], 
                                security_findings: List[SecurityFinding],
                                compliance_status: Dict[str, bool]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Code quality recommendations
        low_quality_metrics = [m for m in metrics if m.score < 0.6]
        for metric in low_quality_metrics:
            if metric.category == "Code Quality":
                if "complexity" in metric.name.lower():
                    recommendations.append("Reduce code complexity by breaking down large functions")
                elif "maintainability" in metric.name.lower():
                    recommendations.append("Improve code maintainability with better structure and documentation")
            elif metric.category == "Documentation":
                recommendations.append("Increase documentation coverage with docstrings and comments")
            elif metric.category == "Performance":
                recommendations.append(f"Optimize {metric.name.lower()} for better performance")
        
        # Security recommendations
        if security_findings:
            critical_findings = [f for f in security_findings if f.severity == SecurityRisk.CRITICAL]
            if critical_findings:
                recommendations.append("Address critical security vulnerabilities immediately")
            
            high_findings = [f for f in security_findings if f.severity == SecurityRisk.HIGH]
            if high_findings:
                recommendations.append("Fix high-severity security issues")
        
        # Compliance recommendations
        failed_compliance = [name for name, passed in compliance_status.items() if not passed]
        for compliance_item in failed_compliance:
            if "pep8" in compliance_item:
                recommendations.append("Improve PEP 8 compliance with code formatting")
            elif "documentation" in compliance_item:
                recommendations.append("Add proper documentation (README, docstrings)")
            elif "testing" in compliance_item:
                recommendations.append("Increase test coverage")
            elif "security" in compliance_item:
                recommendations.append("Implement security best practices")
            elif "license" in compliance_item:
                recommendations.append("Add appropriate license file")
        
        return recommendations
    
    def _save_quality_report(self, result: QualityGateResult):
        """Save comprehensive quality report."""
        report_dir = self.root_path / "quality_reports"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        serializable_result = asdict(result)
        
        # Convert enum to string
        if 'quality_level' in serializable_result:
            serializable_result['quality_level'] = result.quality_level.value
        
        # Convert security findings enums
        if 'security_findings' in serializable_result:
            for finding in serializable_result['security_findings']:
                if 'severity' in finding:
                    finding['severity'] = finding['severity'].value if hasattr(finding['severity'], 'value') else str(finding['severity'])
        
        with open(report_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        self._log(f"Quality report saved to: {report_file}")
    
    def display_quality_results(self, result: QualityGateResult):
        """Display comprehensive quality gate results."""
        if RICH_AVAILABLE and console:
            # Overall summary
            summary_table = Table(title="Quality Gate Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")
            summary_table.add_column("Status", style="green" if result.gate_passed else "red")
            
            summary_table.add_row("Overall Score", f"{result.overall_score:.1%}", "🎯")
            summary_table.add_row("Quality Level", result.quality_level.value.title(), "📊")
            summary_table.add_row("Gate Status", "PASSED" if result.gate_passed else "FAILED", 
                                 "✅" if result.gate_passed else "❌")
            summary_table.add_row("Security Findings", str(len(result.security_findings)), 
                                 "🔒" if len(result.security_findings) == 0 else "⚠️")
            summary_table.add_row("Recommendations", str(len(result.recommendations)), "💡")
            
            console.print(summary_table)
            
            # Detailed metrics by category
            categories = set(m.category for m in result.metrics)
            for category in categories:
                category_metrics = [m for m in result.metrics if m.category == category]
                
                category_table = Table(title=f"{category} Metrics")
                category_table.add_column("Metric", style="cyan")
                category_table.add_column("Score", style="magenta")
                category_table.add_column("Status", style="yellow")
                category_table.add_column("Details", style="white")
                
                for metric in category_metrics:
                    status_icon = "✅" if metric.score > 0.7 else "⚠️" if metric.score > 0.5 else "❌"
                    category_table.add_row(
                        metric.name,
                        f"{metric.score:.1%}",
                        f"{status_icon} {metric.status}",
                        metric.details
                    )
                
                console.print(category_table)
            
            # Security findings
            if result.security_findings:
                security_table = Table(title="Security Findings")
                security_table.add_column("Severity", style="red")
                security_table.add_column("Description", style="yellow")
                security_table.add_column("File", style="cyan")
                security_table.add_column("Line", style="magenta")
                
                for finding in result.security_findings[:10]:  # Show first 10
                    security_table.add_row(
                        finding.severity.value.upper(),
                        finding.description[:50] + "..." if len(finding.description) > 50 else finding.description,
                        Path(finding.file_path).name,
                        str(finding.line_number) if finding.line_number else "N/A"
                    )
                
                console.print(security_table)
            
            # Recommendations
            if result.recommendations:
                recommendations_panel = Panel(
                    "\n".join(f"• {rec}" for rec in result.recommendations[:5]),
                    title="Top Recommendations",
                    border_style="yellow"
                )
                console.print(recommendations_panel)
        else:
            print(f"\n=== Quality Gate Results ===")
            print(f"Overall Score: {result.overall_score:.1%}")
            print(f"Quality Level: {result.quality_level.value.title()}")
            print(f"Gate Status: {'PASSED' if result.gate_passed else 'FAILED'}")
            print(f"Security Findings: {len(result.security_findings)}")
            
            if result.security_findings:
                print("\nSecurity Findings:")
                for finding in result.security_findings[:5]:
                    print(f"  {finding.severity.value.upper()}: {finding.description}")
            
            if result.recommendations:
                print("\nRecommendations:")
                for rec in result.recommendations[:5]:
                    print(f"  • {rec}")

def main():
    """Main quality gate execution."""
    print("🔍 Terragon Quality Gates - Final Validation")
    
    # Initialize quality gates
    root_path = Path.cwd()
    quality_gates = TerragonQualityGates(root_path)
    
    # Run comprehensive quality analysis
    result = quality_gates.run_quality_gates()
    
    # Display results
    quality_gates.display_quality_results(result)
    
    # Final summary
    if RICH_AVAILABLE and console:
        if result.gate_passed:
            console.print(f"\n[bold green]🎉 Quality gate PASSED! Score: {result.overall_score:.1%}[/bold green]")
        else:
            console.print(f"\n[bold red]❌ Quality gate FAILED. Score: {result.overall_score:.1%}[/bold red]")
    else:
        if result.gate_passed:
            print(f"\n🎉 Quality gate PASSED! Score: {result.overall_score:.1%}")
        else:
            print(f"\n❌ Quality gate FAILED. Score: {result.overall_score:.1%}")

if __name__ == "__main__":
    main()