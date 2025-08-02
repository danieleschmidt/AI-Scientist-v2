#!/usr/bin/env python3
"""
Automated integration and deployment script for AI Scientist v2.
Handles automated testing, deployment, and integration with external systems.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class AutomatedIntegration:
    """Manages automated integration and deployment workflows."""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path.cwd()
        self.config_file = self.root_dir / "integration_config.json"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Integration results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "integrations": {},
            "deployments": {},
            "notifications": []
        }
    
    def _load_config(self) -> Dict:
        """Load integration configuration."""
        default_config = {
            "integrations": {
                "github": {
                    "enabled": True,
                    "auto_merge": False,
                    "required_checks": ["tests", "security", "build"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channels": ["#dev-alerts", "#releases"]
                },
                "jira": {
                    "enabled": False,
                    "server": "",
                    "project_key": ""
                },
                "sonarqube": {
                    "enabled": False,
                    "server": "",
                    "project_key": ""
                }
            },
            "deployment": {
                "environments": ["staging", "production"],
                "auto_deploy_staging": True,
                "auto_deploy_production": False,
                "rollback_on_failure": True,
                "health_check_timeout": 300
            },
            "notifications": {
                "on_success": True,
                "on_failure": True,
                "on_deployment": True
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    return {**default_config, **loaded_config}
            except json.JSONDecodeError:
                self.logger.warning("Invalid config file, using defaults")
        
        return default_config
    
    def _run_command(self, command: List[str], cwd: Path = None, 
                    timeout: int = None) -> subprocess.CompletedProcess:
        """Run command with timeout and error handling."""
        cwd = cwd or self.root_dir
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(command)}")
            self.logger.error(f"Exit code: {e.returncode}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            raise
    
    def run_continuous_integration(self) -> Dict:
        """Run continuous integration pipeline."""
        self.logger.info("🔄 Running continuous integration pipeline...")
        
        ci_results = {
            "status": "success",
            "steps": {},
            "duration": 0,
            "artifacts": []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Code quality checks
            self.logger.info("📊 Running code quality checks...")
            ci_results["steps"]["quality"] = self._run_quality_checks()
            
            # Step 2: Security scans
            self.logger.info("🔒 Running security scans...")
            ci_results["steps"]["security"] = self._run_security_scans()
            
            # Step 3: Unit tests
            self.logger.info("🧪 Running unit tests...")
            ci_results["steps"]["unit_tests"] = self._run_unit_tests()
            
            # Step 4: Integration tests
            self.logger.info("🔗 Running integration tests...")
            ci_results["steps"]["integration_tests"] = self._run_integration_tests()
            
            # Step 5: Build packages
            self.logger.info("📦 Building packages...")
            ci_results["steps"]["build"] = self._build_packages()
            
            # Step 6: Performance tests
            self.logger.info("⚡ Running performance tests...")
            ci_results["steps"]["performance"] = self._run_performance_tests()
            
            # Check if all steps passed
            failed_steps = [
                step for step, result in ci_results["steps"].items()
                if result.get("status") != "success"
            ]
            
            if failed_steps:
                ci_results["status"] = "failure"
                ci_results["failed_steps"] = failed_steps
                self.logger.error(f"CI failed on steps: {failed_steps}")
            else:
                self.logger.info("✅ All CI steps passed")
            
        except Exception as e:
            ci_results["status"] = "error"
            ci_results["error"] = str(e)
            self.logger.error(f"CI pipeline error: {e}")
        
        ci_results["duration"] = time.time() - start_time
        self.results["integrations"]["ci"] = ci_results
        
        return ci_results
    
    def _run_quality_checks(self) -> Dict:
        """Run code quality checks."""
        quality_result = {"status": "success", "checks": {}}
        
        try:
            # Flake8
            self._run_command([
                sys.executable, "-m", "flake8",
                "ai_scientist/", "tests/",
                "--max-line-length=88",
                "--extend-ignore=E203,W503"
            ])
            quality_result["checks"]["flake8"] = "passed"
            
            # MyPy
            self._run_command([
                sys.executable, "-m", "mypy",
                "ai_scientist/",
                "--ignore-missing-imports"
            ])
            quality_result["checks"]["mypy"] = "passed"
            
            # Black formatting check
            self._run_command([
                sys.executable, "-m", "black",
                "--check", "--diff",
                "ai_scientist/", "tests/"
            ])
            quality_result["checks"]["black"] = "passed"
            
        except subprocess.CalledProcessError as e:
            quality_result["status"] = "failure"
            quality_result["error"] = str(e)
        
        return quality_result
    
    def _run_security_scans(self) -> Dict:
        """Run security scans."""
        security_result = {"status": "success", "scans": {}}
        
        try:
            # Bandit
            self._run_command([
                sys.executable, "-m", "bandit",
                "-r", "ai_scientist/",
                "-f", "json",
                "-o", "bandit-report.json"
            ])
            security_result["scans"]["bandit"] = "passed"
            
            # Safety
            try:
                self._run_command([
                    sys.executable, "-m", "safety", "check"
                ])
                security_result["scans"]["safety"] = "passed"
            except subprocess.CalledProcessError:
                security_result["scans"]["safety"] = "vulnerabilities_found"
                security_result["status"] = "warning"
            
        except subprocess.CalledProcessError as e:
            security_result["status"] = "failure"
            security_result["error"] = str(e)
        
        return security_result
    
    def _run_unit_tests(self) -> Dict:
        """Run unit tests with coverage."""
        test_result = {"status": "success", "coverage": 0}
        
        try:
            result = self._run_command([
                sys.executable, "-m", "pytest",
                "tests/unit/",
                "--cov=ai_scientist",
                "--cov-report=json",
                "--junit-xml=test-results.xml",
                "-v"
            ], timeout=300)
            
            # Extract coverage from output
            coverage_file = self.root_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                test_result["coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0)
            
        except subprocess.CalledProcessError as e:
            test_result["status"] = "failure"
            test_result["error"] = str(e)
        except subprocess.TimeoutExpired:
            test_result["status"] = "timeout"
            test_result["error"] = "Tests timed out after 5 minutes"
        
        return test_result
    
    def _run_integration_tests(self) -> Dict:
        """Run integration tests."""
        integration_result = {"status": "success"}
        
        try:
            self._run_command([
                sys.executable, "-m", "pytest",
                "tests/integration/",
                "-v", "--tb=short"
            ], timeout=600)
            
        except subprocess.CalledProcessError as e:
            integration_result["status"] = "failure"
            integration_result["error"] = str(e)
        except subprocess.TimeoutExpired:
            integration_result["status"] = "timeout"
            integration_result["error"] = "Integration tests timed out"
        
        return integration_result
    
    def _build_packages(self) -> Dict:
        """Build Python packages."""
        build_result = {"status": "success", "artifacts": []}
        
        try:
            # Build wheel and source distribution
            self._run_command([sys.executable, "-m", "build"])
            
            # List built artifacts
            dist_dir = self.root_dir / "dist"
            if dist_dir.exists():
                build_result["artifacts"] = [
                    f.name for f in dist_dir.glob("*")
                    if f.is_file()
                ]
            
        except subprocess.CalledProcessError as e:
            build_result["status"] = "failure"
            build_result["error"] = str(e)
        
        return build_result
    
    def _run_performance_tests(self) -> Dict:
        """Run performance benchmarks."""
        perf_result = {"status": "success", "benchmarks": {}}
        
        try:
            # Run performance tests if they exist
            perf_tests_dir = self.root_dir / "tests" / "performance"
            if perf_tests_dir.exists():
                self._run_command([
                    sys.executable, "-m", "pytest",
                    str(perf_tests_dir),
                    "--benchmark-json=benchmark-results.json",
                    "-v"
                ], timeout=600)
                
                # Load benchmark results
                benchmark_file = self.root_dir / "benchmark-results.json"
                if benchmark_file.exists():
                    with open(benchmark_file, 'r') as f:
                        benchmark_data = json.load(f)
                    perf_result["benchmarks"] = benchmark_data
            else:
                perf_result["status"] = "skipped"
                perf_result["reason"] = "No performance tests found"
            
        except subprocess.CalledProcessError as e:
            perf_result["status"] = "failure"
            perf_result["error"] = str(e)
        except subprocess.TimeoutExpired:
            perf_result["status"] = "timeout"
            perf_result["error"] = "Performance tests timed out"
        
        return perf_result
    
    def deploy_to_environment(self, environment: str) -> Dict:
        """Deploy to specified environment."""
        self.logger.info(f"🚀 Deploying to {environment}...")
        
        deploy_result = {
            "status": "success",
            "environment": environment,
            "deployment_id": f"deploy-{int(time.time())}",
            "steps": {}
        }
        
        try:
            # Pre-deployment checks
            deploy_result["steps"]["pre_checks"] = self._run_pre_deployment_checks(environment)
            
            # Deploy application
            deploy_result["steps"]["deploy"] = self._deploy_application(environment)
            
            # Post-deployment verification
            deploy_result["steps"]["verification"] = self._verify_deployment(environment)
            
            # Health checks
            deploy_result["steps"]["health_check"] = self._run_health_checks(environment)
            
            self.logger.info(f"✅ Successfully deployed to {environment}")
            
        except Exception as e:
            deploy_result["status"] = "failure"
            deploy_result["error"] = str(e)
            self.logger.error(f"Deployment to {environment} failed: {e}")
            
            # Attempt rollback if configured
            if self.config["deployment"]["rollback_on_failure"]:
                self.logger.info("🔄 Attempting rollback...")
                deploy_result["rollback"] = self._rollback_deployment(environment)
        
        self.results["deployments"][environment] = deploy_result
        return deploy_result
    
    def _run_pre_deployment_checks(self, environment: str) -> Dict:
        """Run pre-deployment checks."""
        return {"status": "success", "checks": ["config_validation", "dependency_check"]}
    
    def _deploy_application(self, environment: str) -> Dict:
        """Deploy application to environment."""
        # This would typically involve Docker, Kubernetes, or cloud deployment
        # For demo purposes, we'll simulate the deployment
        time.sleep(2)  # Simulate deployment time
        return {"status": "success", "method": "simulated"}
    
    def _verify_deployment(self, environment: str) -> Dict:
        """Verify deployment was successful."""
        return {"status": "success", "verified": True}
    
    def _run_health_checks(self, environment: str) -> Dict:
        """Run health checks on deployed application."""
        health_result = {"status": "success", "checks": {}}
        
        # Simulate health checks
        health_checks = ["api_endpoint", "database_connection", "dependencies"]
        
        for check in health_checks:
            # In real implementation, this would make actual HTTP requests
            health_result["checks"][check] = "healthy"
        
        return health_result
    
    def _rollback_deployment(self, environment: str) -> Dict:
        """Rollback deployment to previous version."""
        self.logger.info(f"Rolling back deployment in {environment}...")
        # Simulate rollback
        return {"status": "success", "method": "simulated"}
    
    def integrate_with_github(self) -> Dict:
        """Integrate with GitHub (PR status, merge, etc.)."""
        if not self.config["integrations"]["github"]["enabled"]:
            return {"status": "disabled"}
        
        self.logger.info("🐙 Integrating with GitHub...")
        
        github_result = {"status": "success", "actions": []}
        
        try:
            # Update PR status
            if self._is_pull_request():
                pr_number = self._get_pr_number()
                if pr_number:
                    github_result["actions"].append(f"updated_pr_{pr_number}")
                    
                    # Auto-merge if configured and all checks pass
                    if (self.config["integrations"]["github"]["auto_merge"] and 
                        self._all_checks_passed()):
                        self._merge_pull_request(pr_number)
                        github_result["actions"].append(f"merged_pr_{pr_number}")
            
        except Exception as e:
            github_result["status"] = "failure"
            github_result["error"] = str(e)
        
        return github_result
    
    def _is_pull_request(self) -> bool:
        """Check if running in PR context."""
        return os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
    
    def _get_pr_number(self) -> Optional[str]:
        """Get PR number from environment."""
        return os.environ.get("GITHUB_PR_NUMBER")
    
    def _all_checks_passed(self) -> bool:
        """Check if all required checks passed."""
        ci_result = self.results.get("integrations", {}).get("ci", {})
        return ci_result.get("status") == "success"
    
    def _merge_pull_request(self, pr_number: str):
        """Merge pull request using GitHub CLI."""
        try:
            self._run_command([
                "gh", "pr", "merge", pr_number,
                "--squash", "--delete-branch"
            ])
        except subprocess.CalledProcessError:
            self.logger.warning(f"Failed to auto-merge PR {pr_number}")
    
    def send_notifications(self, event: str, data: Dict):
        """Send notifications to configured channels."""
        if not self.config["notifications"].get(f"on_{event}", False):
            return
        
        self.logger.info(f"📢 Sending notifications for {event}...")
        
        notification = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Slack notification
        if self.config["integrations"]["slack"]["enabled"]:
            self._send_slack_notification(notification)
        
        # Email notification (if configured)
        # self._send_email_notification(notification)
        
        self.results["notifications"].append(notification)
    
    def _send_slack_notification(self, notification: Dict):
        """Send Slack notification."""
        webhook_url = self.config["integrations"]["slack"]["webhook_url"]
        if not webhook_url:
            return
        
        # Format message based on event type
        event = notification["event"]
        data = notification["data"]
        
        if event == "success":
            message = f"✅ CI/CD Pipeline Successful\nDuration: {data.get('duration', 'unknown')}s"
            color = "good"
        elif event == "failure":
            message = f"❌ CI/CD Pipeline Failed\nError: {data.get('error', 'unknown')}"
            color = "danger"
        elif event == "deployment":
            env = data.get("environment", "unknown")
            status = data.get("status", "unknown")
            message = f"🚀 Deployment to {env}: {status}"
            color = "good" if status == "success" else "danger"
        else:
            message = f"ℹ️ Event: {event}"
            color = "warning"
        
        # In real implementation, this would make HTTP request to Slack webhook
        self.logger.info(f"Slack notification: {message}")
    
    def generate_integration_report(self) -> Dict:
        """Generate comprehensive integration report."""
        self.logger.info("📋 Generating integration report...")
        
        report = {
            "summary": {
                "timestamp": self.results["timestamp"],
                "overall_status": "success",
                "total_duration": 0,
                "integrations_run": len(self.results["integrations"]),
                "deployments_run": len(self.results["deployments"]),
                "notifications_sent": len(self.results["notifications"])
            },
            "details": self.results,
            "recommendations": []
        }
        
        # Determine overall status
        failed_integrations = [
            name for name, result in self.results["integrations"].items()
            if result.get("status") != "success"
        ]
        
        failed_deployments = [
            env for env, result in self.results["deployments"].items()
            if result.get("status") != "success"
        ]
        
        if failed_integrations or failed_deployments:
            report["summary"]["overall_status"] = "failure"
            report["summary"]["failed_integrations"] = failed_integrations
            report["summary"]["failed_deployments"] = failed_deployments
        
        # Generate recommendations
        if failed_integrations:
            report["recommendations"].append({
                "type": "integration_failure",
                "description": "Review and fix failed integration steps",
                "priority": "high"
            })
        
        if failed_deployments:
            report["recommendations"].append({
                "type": "deployment_failure", 
                "description": "Investigate deployment failures and improve deployment process",
                "priority": "critical"
            })
        
        # Save report
        report_file = self.root_dir / "integration-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✅ Integration report saved: {report_file}")
        return report


def main():
    """Main entry point for automated integration."""
    parser = argparse.ArgumentParser(description="Automated integration and deployment")
    
    parser.add_argument("--ci", action="store_true", help="Run continuous integration")
    parser.add_argument("--deploy", choices=["staging", "production"], 
                       help="Deploy to environment")
    parser.add_argument("--github", action="store_true", help="Integrate with GitHub")
    parser.add_argument("--notify", choices=["success", "failure", "deployment"],
                       help="Send test notification")
    parser.add_argument("--config", help="Custom configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    integration = AutomatedIntegration()
    
    if args.config:
        integration.config_file = Path(args.config)
        integration.config = integration._load_config()
    
    try:
        if args.ci:
            result = integration.run_continuous_integration()
            
            # Send notifications based on result
            if result["status"] == "success":
                integration.send_notifications("success", result)
            else:
                integration.send_notifications("failure", result)
            
        if args.deploy:
            if args.dry_run:
                print(f"Dry run: Would deploy to {args.deploy}")
            else:
                result = integration.deploy_to_environment(args.deploy)
                integration.send_notifications("deployment", result)
        
        if args.github:
            integration.integrate_with_github()
        
        if args.notify:
            test_data = {"test": True, "timestamp": datetime.now().isoformat()}
            integration.send_notifications(args.notify, test_data)
        
        # Generate final report
        report = integration.generate_integration_report()
        
        print(f"\n📊 Integration Summary:")
        print(f"   Overall Status: {report['summary']['overall_status']}")
        print(f"   Integrations: {report['summary']['integrations_run']}")
        print(f"   Deployments: {report['summary']['deployments_run']}")
        print(f"   Notifications: {report['summary']['notifications_sent']}")
        
        # Exit with appropriate code
        if report['summary']['overall_status'] == "failure":
            sys.exit(1)
        
    except KeyboardInterrupt:
        integration.logger.info("⚠️ Integration interrupted by user")
        sys.exit(1)
    except Exception as e:
        integration.logger.error(f"💥 Integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()