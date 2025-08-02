#!/usr/bin/env python3
"""
Repository health monitoring and automated maintenance script for AI Scientist v2.
Monitors repository health metrics, code quality, and performs automated maintenance tasks.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import ast


class RepositoryHealthMonitor:
    """Monitors and maintains repository health metrics."""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path.cwd()
        self.metrics_file = self.root_dir / ".github" / "project-metrics.json"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load project metrics configuration
        self.metrics_config = self._load_metrics_config()
        
        # Health check results
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "checks": {},
            "metrics": {},
            "recommendations": [],
            "alerts": []
        }
    
    def _load_metrics_config(self) -> Dict:
        """Load project metrics configuration."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _run_command(self, command: List[str], cwd: Path = None, 
                    capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run command and return result."""
        cwd = cwd or self.root_dir
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.debug(f"Command failed: {' '.join(command)}")
            raise
    
    def check_git_repository_health(self) -> Dict:
        """Check Git repository health metrics."""
        self.logger.info("🔍 Checking Git repository health...")
        
        git_health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Check if repo is clean
            result = self._run_command(["git", "status", "--porcelain"])
            if result.stdout.strip():
                git_health["issues"].append("Repository has uncommitted changes")
                git_health["status"] = "warning"
            
            # Check for untracked files
            untracked_files = [
                line[3:] for line in result.stdout.split('\n') 
                if line.startswith('??')
            ]
            if untracked_files:
                git_health["metrics"]["untracked_files"] = len(untracked_files)
                if len(untracked_files) > 10:
                    git_health["issues"].append(f"Too many untracked files: {len(untracked_files)}")
            
            # Check branch status
            result = self._run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            current_branch = result.stdout.strip()
            git_health["metrics"]["current_branch"] = current_branch
            
            # Check if branch is ahead/behind main
            try:
                result = self._run_command([
                    "git", "rev-list", "--count", "--left-right", "HEAD...main"
                ])
                ahead, behind = result.stdout.strip().split('\t')
                git_health["metrics"]["commits_ahead"] = int(ahead)
                git_health["metrics"]["commits_behind"] = int(behind)
                
                if int(behind) > 50:
                    git_health["issues"].append(f"Branch is {behind} commits behind main")
                    
            except subprocess.CalledProcessError:
                pass  # main branch might not exist
            
            # Check last commit age
            result = self._run_command([
                "git", "log", "-1", "--format=%ct"
            ])
            last_commit_timestamp = int(result.stdout.strip())
            last_commit_age = datetime.now().timestamp() - last_commit_timestamp
            git_health["metrics"]["last_commit_age_days"] = last_commit_age / 86400
            
            if last_commit_age > 86400 * 30:  # 30 days
                git_health["issues"].append("No commits in the last 30 days")
                git_health["status"] = "warning"
            
        except subprocess.CalledProcessError as e:
            git_health["status"] = "error"
            git_health["issues"].append(f"Git command failed: {e}")
        
        return git_health
    
    def check_code_quality_metrics(self) -> Dict:
        """Check code quality metrics."""
        self.logger.info("📊 Checking code quality metrics...")
        
        quality_metrics = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Count lines of code
        code_stats = self._count_lines_of_code()
        quality_metrics["metrics"].update(code_stats)
        
        # Check test coverage
        coverage_data = self._check_test_coverage()
        if coverage_data:
            quality_metrics["metrics"].update(coverage_data)
            
            target_coverage = self.metrics_config.get("metrics", {}).get(
                "code_quality", {}
            ).get("coverage_target", 80)
            
            if coverage_data.get("coverage_percent", 0) < target_coverage:
                quality_metrics["issues"].append(
                    f"Test coverage below target: {coverage_data['coverage_percent']}% < {target_coverage}%"
                )
                quality_metrics["status"] = "warning"
        
        # Check complexity metrics
        complexity_data = self._check_code_complexity()
        if complexity_data:
            quality_metrics["metrics"].update(complexity_data)
            
            max_complexity = self.metrics_config.get("metrics", {}).get(
                "code_quality", {}
            ).get("complexity_max", 10)
            
            if complexity_data.get("max_complexity", 0) > max_complexity:
                quality_metrics["issues"].append(
                    f"Code complexity too high: {complexity_data['max_complexity']} > {max_complexity}"
                )
                quality_metrics["status"] = "warning"
        
        # Check for TODO/FIXME comments
        todo_count = self._count_todo_comments()
        quality_metrics["metrics"]["todo_comments"] = todo_count
        
        if todo_count > 50:
            quality_metrics["issues"].append(f"High number of TODO comments: {todo_count}")
            quality_metrics["status"] = "warning"
        
        return quality_metrics
    
    def _count_lines_of_code(self) -> Dict:
        """Count lines of code by type."""
        stats = {
            "total_files": 0,
            "python_files": 0,
            "test_files": 0,
            "total_lines": 0,
            "python_lines": 0,
            "test_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0
        }
        
        # Define file patterns
        python_patterns = ["**/*.py"]
        test_patterns = ["**/test_*.py", "**/tests/**/*.py"]
        
        for pattern in python_patterns:
            for file_path in self.root_dir.glob(pattern):
                if file_path.is_file():
                    stats["total_files"] += 1
                    stats["python_files"] += 1
                    
                    is_test_file = any(
                        file_path.match(test_pattern) 
                        for test_pattern in test_patterns
                    )
                    
                    if is_test_file:
                        stats["test_files"] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        file_lines = len(lines)
                        file_blank = sum(1 for line in lines if not line.strip())
                        file_comment = sum(
                            1 for line in lines 
                            if line.strip().startswith('#')
                        )
                        
                        stats["total_lines"] += file_lines
                        stats["python_lines"] += file_lines
                        stats["blank_lines"] += file_blank
                        stats["comment_lines"] += file_comment
                        
                        if is_test_file:
                            stats["test_lines"] += file_lines
                            
                    except (UnicodeDecodeError, PermissionError):
                        continue
        
        # Calculate ratios
        if stats["python_lines"] > 0:
            stats["comment_ratio"] = stats["comment_lines"] / stats["python_lines"]
            stats["test_ratio"] = stats["test_lines"] / stats["python_lines"]
        
        return stats
    
    def _check_test_coverage(self) -> Optional[Dict]:
        """Check test coverage using pytest-cov."""
        try:
            result = self._run_command([
                sys.executable, "-m", "pytest",
                "--cov=ai_scientist",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--tb=no",
                "-q"
            ])
            
            # Try to read coverage.json
            coverage_file = self.root_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                return {
                    "coverage_percent": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "covered_lines": coverage_data.get("totals", {}).get("covered_lines", 0),
                    "total_lines": coverage_data.get("totals", {}).get("num_statements", 0),
                    "missing_lines": coverage_data.get("totals", {}).get("missing_lines", 0)
                }
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            self.logger.debug("Could not determine test coverage")
        
        return None
    
    def _check_code_complexity(self) -> Optional[Dict]:
        """Check code complexity using radon or basic analysis."""
        complexity_data = {
            "max_complexity": 0,
            "avg_complexity": 0,
            "complex_functions": []
        }
        
        try:
            # Try using radon if available
            result = self._run_command([
                "radon", "cc", "ai_scientist/", "--json"
            ])
            
            radon_data = json.loads(result.stdout)
            complexities = []
            
            for file_path, functions in radon_data.items():
                for func in functions:
                    complexity = func.get("complexity", 0)
                    complexities.append(complexity)
                    
                    if complexity > 10:
                        complexity_data["complex_functions"].append({
                            "file": file_path,
                            "function": func.get("name", ""),
                            "complexity": complexity
                        })
            
            if complexities:
                complexity_data["max_complexity"] = max(complexities)
                complexity_data["avg_complexity"] = sum(complexities) / len(complexities)
            
            return complexity_data
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            # Fallback: basic AST analysis
            return self._basic_complexity_analysis()
    
    def _basic_complexity_analysis(self) -> Dict:
        """Basic complexity analysis using AST."""
        complexity_data = {
            "max_complexity": 0,
            "avg_complexity": 0,
            "complex_functions": []
        }
        
        complexities = []
        
        for py_file in self.root_dir.glob("ai_scientist/**/*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Simple complexity: count decision points
                            complexity = 1  # Base complexity
                            
                            for child in ast.walk(node):
                                if isinstance(child, (ast.If, ast.While, ast.For, 
                                                    ast.AsyncFor, ast.With, ast.AsyncWith,
                                                    ast.Try, ast.ExceptHandler)):
                                    complexity += 1
                                elif isinstance(child, ast.BoolOp):
                                    complexity += len(child.values) - 1
                            
                            complexities.append(complexity)
                            
                            if complexity > 10:
                                complexity_data["complex_functions"].append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "function": node.name,
                                    "complexity": complexity
                                })
                
                except (SyntaxError, UnicodeDecodeError):
                    continue
        
        if complexities:
            complexity_data["max_complexity"] = max(complexities)
            complexity_data["avg_complexity"] = sum(complexities) / len(complexities)
        
        return complexity_data
    
    def _count_todo_comments(self) -> int:
        """Count TODO/FIXME comments in code."""
        todo_count = 0
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|HACK|XXX)', re.IGNORECASE)
        
        for py_file in self.root_dir.glob("**/*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if todo_pattern.search(line):
                                todo_count += 1
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        return todo_count
    
    def check_security_health(self) -> Dict:
        """Check security health metrics."""
        self.logger.info("🔒 Checking security health...")
        
        security_health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check for secrets in code
        secrets_issues = self._check_for_secrets()
        if secrets_issues:
            security_health["issues"].extend(secrets_issues)
            security_health["status"] = "warning"
        
        # Check dependency vulnerabilities
        vuln_count = self._check_dependency_vulnerabilities()
        security_health["metrics"]["vulnerability_count"] = vuln_count
        
        if vuln_count > 0:
            security_health["issues"].append(f"Found {vuln_count} dependency vulnerabilities")
            security_health["status"] = "critical" if vuln_count > 5 else "warning"
        
        # Check file permissions
        permission_issues = self._check_file_permissions()
        if permission_issues:
            security_health["issues"].extend(permission_issues)
            if security_health["status"] == "healthy":
                security_health["status"] = "warning"
        
        return security_health
    
    def _check_for_secrets(self) -> List[str]:
        """Check for potential secrets in code."""
        issues = []
        
        # Patterns that might indicate secrets
        secret_patterns = [
            r'password\s*=\s*[\'"][^\'"\s]{8,}[\'"]',
            r'api_key\s*=\s*[\'"][^\'"\s]{20,}[\'"]',
            r'secret\s*=\s*[\'"][^\'"\s]{16,}[\'"]',
            r'token\s*=\s*[\'"][^\'"\s]{20,}[\'"]',
        ]
        
        for pattern in secret_patterns:
            try:
                result = self._run_command([
                    "grep", "-r", "-E", pattern, 
                    "ai_scientist/", "--exclude-dir=__pycache__"
                ])
                
                if result.stdout:
                    issues.append(f"Potential secret found matching pattern: {pattern}")
                    
            except subprocess.CalledProcessError:
                pass  # No matches found
        
        return issues
    
    def _check_dependency_vulnerabilities(self) -> int:
        """Check for dependency vulnerabilities."""
        try:
            result = self._run_command([
                sys.executable, "-m", "safety", "check", "--json"
            ])
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                return len(safety_data)
                
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
        
        return 0
    
    def _check_file_permissions(self) -> List[str]:
        """Check for overly permissive file permissions."""
        issues = []
        
        # Check for executable files that shouldn't be
        for file_path in self.root_dir.glob("**/*.py"):
            if file_path.is_file() and os.access(file_path, os.X_OK):
                # Python files generally shouldn't be executable unless they're scripts
                if not self._is_script_file(file_path):
                    issues.append(f"Executable Python file: {file_path}")
        
        return issues
    
    def _is_script_file(self, file_path: Path) -> bool:
        """Check if a Python file is a script (has shebang)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                return first_line.startswith('#!')
        except (UnicodeDecodeError, PermissionError):
            return False
    
    def check_performance_health(self) -> Dict:
        """Check performance-related health metrics."""
        self.logger.info("⚡ Checking performance health...")
        
        performance_health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check repository size
        repo_size = self._get_repository_size()
        performance_health["metrics"]["repository_size_mb"] = repo_size
        
        if repo_size > 1000:  # 1GB
            performance_health["issues"].append(f"Large repository size: {repo_size}MB")
            performance_health["status"] = "warning"
        
        # Check for large files
        large_files = self._find_large_files()
        if large_files:
            performance_health["metrics"]["large_files_count"] = len(large_files)
            performance_health["issues"].append(f"Found {len(large_files)} large files (>10MB)")
            if performance_health["status"] == "healthy":
                performance_health["status"] = "warning"
        
        # Check for binary files in git
        binary_files = self._find_binary_files_in_git()
        if binary_files:
            performance_health["metrics"]["binary_files_count"] = len(binary_files)
            if len(binary_files) > 10:
                performance_health["issues"].append(f"Many binary files tracked: {len(binary_files)}")
                if performance_health["status"] == "healthy":
                    performance_health["status"] = "warning"
        
        return performance_health
    
    def _get_repository_size(self) -> float:
        """Get repository size in MB."""
        try:
            result = self._run_command(["du", "-sm", str(self.root_dir)])
            size_str = result.stdout.split()[0]
            return float(size_str)
        except (subprocess.CalledProcessError, ValueError):
            return 0.0
    
    def _find_large_files(self, size_limit_mb: int = 10) -> List[str]:
        """Find files larger than size limit."""
        large_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        for file_path in self.root_dir.glob("**/*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size > size_limit_bytes:
                        large_files.append(str(file_path.relative_to(self.root_dir)))
                except OSError:
                    continue
        
        return large_files
    
    def _find_binary_files_in_git(self) -> List[str]:
        """Find binary files tracked by git."""
        try:
            result = self._run_command(["git", "ls-files"])
            tracked_files = result.stdout.strip().split('\n')
            
            binary_files = []
            for file_path in tracked_files:
                full_path = self.root_dir / file_path
                if full_path.exists() and self._is_binary_file(full_path):
                    binary_files.append(file_path)
            
            return binary_files
            
        except subprocess.CalledProcessError:
            return []
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                return b'\0' in chunk
        except (OSError, PermissionError):
            return False
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report."""
        self.logger.info("📋 Generating repository health report...")
        
        # Run all health checks
        self.health_report["checks"]["git"] = self.check_git_repository_health()
        self.health_report["checks"]["code_quality"] = self.check_code_quality_metrics()
        self.health_report["checks"]["security"] = self.check_security_health()
        self.health_report["checks"]["performance"] = self.check_performance_health()
        
        # Determine overall health
        statuses = [check["status"] for check in self.health_report["checks"].values()]
        
        if "critical" in statuses:
            self.health_report["overall_health"] = "critical"
        elif "warning" in statuses:
            self.health_report["overall_health"] = "warning"
        else:
            self.health_report["overall_health"] = "healthy"
        
        # Collect all issues and create recommendations
        all_issues = []
        for category, check in self.health_report["checks"].items():
            for issue in check.get("issues", []):
                all_issues.append(f"[{category.upper()}] {issue}")
        
        self.health_report["alerts"] = all_issues
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save report
        report_file = self.root_dir / "repository-health-report.json"
        with open(report_file, 'w') as f:
            json.dump(self.health_report, f, indent=2)
        
        self.logger.info(f"✅ Health report saved: {report_file}")
        return self.health_report
    
    def _generate_recommendations(self):
        """Generate recommendations based on health checks."""
        recommendations = []
        
        # Code quality recommendations
        code_quality = self.health_report["checks"].get("code_quality", {})
        if "warning" in code_quality.get("status", ""):
            metrics = code_quality.get("metrics", {})
            
            if metrics.get("coverage_percent", 100) < 80:
                recommendations.append({
                    "category": "code_quality",
                    "priority": "high",
                    "action": "Increase test coverage",
                    "description": "Write more unit tests to reach 80% coverage target"
                })
            
            if metrics.get("todo_comments", 0) > 50:
                recommendations.append({
                    "category": "code_quality", 
                    "priority": "medium",
                    "action": "Address TODO comments",
                    "description": "Review and resolve outstanding TODO/FIXME comments"
                })
        
        # Security recommendations
        security = self.health_report["checks"].get("security", {})
        if security.get("metrics", {}).get("vulnerability_count", 0) > 0:
            recommendations.append({
                "category": "security",
                "priority": "critical",
                "action": "Update vulnerable dependencies",
                "description": "Run dependency updates to fix security vulnerabilities"
            })
        
        # Performance recommendations
        performance = self.health_report["checks"].get("performance", {})
        if performance.get("metrics", {}).get("repository_size_mb", 0) > 1000:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "action": "Reduce repository size",
                "description": "Remove large files or use Git LFS for binary assets"
            })
        
        self.health_report["recommendations"] = recommendations
    
    def create_maintenance_tasks(self) -> List[Dict]:
        """Create automated maintenance tasks based on health report."""
        tasks = []
        
        for recommendation in self.health_report.get("recommendations", []):
            if recommendation["category"] == "security" and recommendation["priority"] == "critical":
                tasks.append({
                    "type": "dependency_update",
                    "description": "Update vulnerable dependencies",
                    "command": [
                        sys.executable, "scripts/dependency_update.py", 
                        "--security-only", "--create-pr"
                    ]
                })
            
            elif recommendation["category"] == "code_quality":
                tasks.append({
                    "type": "code_cleanup",
                    "description": "Run code formatting and linting",
                    "command": [
                        "pre-commit", "run", "--all-files"
                    ]
                })
        
        return tasks


def main():
    """Main entry point for repository health monitoring."""
    parser = argparse.ArgumentParser(description="Repository health monitoring")
    
    parser.add_argument("--check", choices=["all", "git", "quality", "security", "performance"],
                       default="all", help="Specific health check to run")
    parser.add_argument("--report", action="store_true", 
                       help="Generate comprehensive health report")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to automatically fix issues")
    parser.add_argument("--output", help="Output file for health report")
    parser.add_argument("--format", choices=["json", "text"], default="json",
                       help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    monitor = RepositoryHealthMonitor()
    
    try:
        if args.check == "all" or args.report:
            report = monitor.generate_health_report()
        else:
            # Run specific check
            if args.check == "git":
                result = monitor.check_git_repository_health()
            elif args.check == "quality":
                result = monitor.check_code_quality_metrics()
            elif args.check == "security":
                result = monitor.check_security_health()
            elif args.check == "performance":
                result = monitor.check_performance_health()
            
            report = {"check": args.check, "result": result}
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Health report saved to: {args.output}")
        else:
            if args.format == "json":
                print(json.dumps(report, indent=2))
            else:
                # Text format
                overall_health = report.get("overall_health", "unknown")
                print(f"\n🏥 Repository Health: {overall_health.upper()}")
                
                for category, check in report.get("checks", {}).items():
                    status = check.get("status", "unknown")
                    status_emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
                    print(f"{status_emoji} {category.title()}: {status}")
                    
                    for issue in check.get("issues", []):
                        print(f"   - {issue}")
                
                if report.get("recommendations"):
                    print(f"\n💡 Recommendations:")
                    for rec in report["recommendations"]:
                        priority_emoji = "🚨" if rec["priority"] == "critical" else "⚠️" if rec["priority"] == "high" else "💡"
                        print(f"{priority_emoji} {rec['action']}: {rec['description']}")
        
        # Attempt fixes if requested
        if args.fix:
            tasks = monitor.create_maintenance_tasks()
            print(f"\n🔧 Running {len(tasks)} maintenance tasks...")
            
            for task in tasks:
                print(f"Running: {task['description']}")
                try:
                    subprocess.run(task["command"], check=True)
                    print("✅ Task completed")
                except subprocess.CalledProcessError:
                    print("❌ Task failed")
        
        # Exit with appropriate code
        overall_health = report.get("overall_health", "unknown")
        if overall_health == "critical":
            sys.exit(2)
        elif overall_health == "warning":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        monitor.logger.info("⚠️ Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        monitor.logger.error(f"💥 Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()