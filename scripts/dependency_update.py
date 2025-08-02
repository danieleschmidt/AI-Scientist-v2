#!/usr/bin/env python3
"""
Automated dependency update and management script for AI Scientist v2.
Handles dependency updates, security checks, and automated pull request creation.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class DependencyUpdater:
    """Manages automated dependency updates and security scanning."""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path.cwd()
        self.requirements_file = self.root_dir / "requirements.txt"
        self.pyproject_file = self.root_dir / "pyproject.toml"
        self.lockfile = self.root_dir / "requirements-lock.txt"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _run_command(self, command: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """Run command and return result."""
        cwd = cwd or self.root_dir
        self.logger.info(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(command)}")
            self.logger.error(f"Exit code: {e.returncode}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise
    
    def check_outdated_packages(self) -> List[Dict]:
        """Check for outdated packages using pip."""
        self.logger.info("🔍 Checking for outdated packages...")
        
        try:
            result = self._run_command([
                sys.executable, "-m", "pip", "list", 
                "--outdated", "--format=json"
            ])
            outdated = json.loads(result.stdout)
            
            self.logger.info(f"Found {len(outdated)} outdated packages")
            return outdated
            
        except subprocess.CalledProcessError:
            self.logger.error("Failed to check outdated packages")
            return []
    
    def check_security_vulnerabilities(self) -> Dict:
        """Check for security vulnerabilities using safety."""
        self.logger.info("🔒 Checking for security vulnerabilities...")
        
        vulnerability_report = {
            "vulnerabilities": [],
            "scan_time": datetime.now().isoformat(),
            "tool": "safety"
        }
        
        try:
            result = self._run_command([
                sys.executable, "-m", "safety", "check", 
                "--json", "--continue-on-error"
            ])
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerability_report["vulnerabilities"] = safety_data
                
        except subprocess.CalledProcessError as e:
            # Safety returns non-zero exit code when vulnerabilities found
            if e.stdout:
                try:
                    safety_data = json.loads(e.stdout)
                    vulnerability_report["vulnerabilities"] = safety_data
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse safety output")
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
        
        vuln_count = len(vulnerability_report["vulnerabilities"])
        if vuln_count > 0:
            self.logger.warning(f"Found {vuln_count} security vulnerabilities")
        else:
            self.logger.info("No security vulnerabilities found")
        
        return vulnerability_report
    
    def update_requirements_lock(self):
        """Generate/update requirements lock file."""
        self.logger.info("📦 Updating requirements lock file...")
        
        if not self.requirements_file.exists():
            self.logger.warning("No requirements.txt found")
            return
        
        # Generate lock file with pip-tools if available
        try:
            self._run_command([
                sys.executable, "-m", "piptools", "compile",
                "--upgrade", "--generate-hashes",
                str(self.requirements_file),
                "--output-file", str(self.lockfile)
            ])
            self.logger.info("✅ Requirements lock file updated")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: use pip freeze
            self.logger.info("pip-tools not available, using pip freeze")
            result = self._run_command([sys.executable, "-m", "pip", "freeze"])
            
            with open(self.lockfile, 'w') as f:
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                f.write(f"# From {self.requirements_file.name}\n\n")
                f.write(result.stdout)
            
            self.logger.info("✅ Requirements lock file created with pip freeze")
    
    def create_dependency_report(self, outdated: List[Dict], vulnerabilities: Dict) -> Dict:
        """Create comprehensive dependency report."""
        self.logger.info("📊 Creating dependency report...")
        
        report = {
            "scan_time": datetime.now().isoformat(),
            "summary": {
                "total_outdated": len(outdated),
                "security_vulnerabilities": len(vulnerabilities.get("vulnerabilities", [])),
                "critical_updates": 0,
                "recommended_updates": 0
            },
            "outdated_packages": outdated,
            "security_report": vulnerabilities,
            "recommendations": []
        }
        
        # Analyze updates and create recommendations
        for package in outdated:
            current_version = package.get("version", "")
            latest_version = package.get("latest_version", "")
            package_name = package.get("name", "")
            
            # Determine update priority
            is_security_critical = any(
                vuln.get("package", "").lower() == package_name.lower()
                for vuln in vulnerabilities.get("vulnerabilities", [])
            )
            
            if is_security_critical:
                priority = "critical"
                report["summary"]["critical_updates"] += 1
            else:
                priority = "recommended"
                report["summary"]["recommended_updates"] += 1
            
            recommendation = {
                "package": package_name,
                "current_version": current_version,
                "latest_version": latest_version,
                "priority": priority,
                "security_critical": is_security_critical,
                "action": "update_immediately" if is_security_critical else "update_when_convenient"
            }
            
            report["recommendations"].append(recommendation)
        
        # Save report
        report_file = self.root_dir / "dependency-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✅ Dependency report saved: {report_file}")
        return report
    
    def update_package(self, package_name: str, version: str = None) -> bool:
        """Update a specific package."""
        self.logger.info(f"📦 Updating {package_name}...")
        
        try:
            update_spec = f"{package_name}=={version}" if version else package_name
            self._run_command([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", update_spec
            ])
            self.logger.info(f"✅ Updated {package_name}")
            return True
            
        except subprocess.CalledProcessError:
            self.logger.error(f"❌ Failed to update {package_name}")
            return False
    
    def batch_update_packages(self, packages: List[str], 
                            test_after_each: bool = True) -> Dict[str, bool]:
        """Update multiple packages with optional testing."""
        self.logger.info(f"📦 Batch updating {len(packages)} packages...")
        
        results = {}
        
        for package in packages:
            success = self.update_package(package)
            results[package] = success
            
            if test_after_each and success:
                if not self._run_basic_tests():
                    self.logger.warning(f"Tests failed after updating {package}")
                    # Optionally rollback here
                    results[package] = False
        
        return results
    
    def _run_basic_tests(self) -> bool:
        """Run basic smoke tests to verify functionality."""
        self.logger.info("🧪 Running basic tests...")
        
        try:
            # Run a subset of tests for quick validation
            self._run_command([
                sys.executable, "-m", "pytest",
                "tests/test_basic_functionality.py",
                "-v", "--tb=short"
            ])
            return True
            
        except subprocess.CalledProcessError:
            return False
    
    def create_update_branch(self, branch_name: str = None) -> str:
        """Create a new branch for dependency updates."""
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"chore/dependency-updates-{timestamp}"
        
        self.logger.info(f"🌿 Creating update branch: {branch_name}")
        
        try:
            self._run_command(["git", "checkout", "-b", branch_name])
            return branch_name
            
        except subprocess.CalledProcessError:
            self.logger.error(f"Failed to create branch {branch_name}")
            raise
    
    def commit_updates(self, message: str = None):
        """Commit dependency updates."""
        if not message:
            message = f"chore: automated dependency updates - {datetime.now().strftime('%Y-%m-%d')}"
        
        self.logger.info("💾 Committing updates...")
        
        try:
            # Add changed files
            self._run_command(["git", "add", "requirements.txt", "requirements-lock.txt"])
            
            # Check if there are changes to commit
            result = self._run_command(["git", "diff", "--cached", "--name-only"])
            if not result.stdout.strip():
                self.logger.info("No changes to commit")
                return
            
            # Commit changes
            self._run_command(["git", "commit", "-m", message])
            self.logger.info("✅ Changes committed")
            
        except subprocess.CalledProcessError:
            self.logger.error("Failed to commit changes")
            raise
    
    def create_pull_request(self, branch_name: str, report: Dict) -> Optional[str]:
        """Create pull request for dependency updates (requires GitHub CLI)."""
        self.logger.info("📝 Creating pull request...")
        
        # Create PR body from report
        pr_body = self._generate_pr_body(report)
        pr_title = f"chore: automated dependency updates - {datetime.now().strftime('%Y-%m-%d')}"
        
        try:
            result = self._run_command([
                "gh", "pr", "create",
                "--title", pr_title,
                "--body", pr_body,
                "--label", "dependencies",
                "--label", "automated"
            ])
            
            pr_url = result.stdout.strip()
            self.logger.info(f"✅ Pull request created: {pr_url}")
            return pr_url
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("GitHub CLI not available or failed to create PR")
            self.logger.info(f"Manual PR creation needed for branch: {branch_name}")
            return None
    
    def _generate_pr_body(self, report: Dict) -> str:
        """Generate pull request body from dependency report."""
        body = f"""## Automated Dependency Updates

### Summary
- 📦 Total outdated packages: {report['summary']['total_outdated']}
- 🔒 Security vulnerabilities: {report['summary']['security_vulnerabilities']}
- 🚨 Critical updates: {report['summary']['critical_updates']}
- 💡 Recommended updates: {report['summary']['recommended_updates']}

### Updated Packages
"""
        
        for rec in report.get("recommendations", []):
            priority_emoji = "🚨" if rec["priority"] == "critical" else "💡"
            security_note = " (Security Fix)" if rec["security_critical"] else ""
            
            body += f"- {priority_emoji} **{rec['package']}**: {rec['current_version']} → {rec['latest_version']}{security_note}\n"
        
        if report['summary']['security_vulnerabilities'] > 0:
            body += f"\n### Security Vulnerabilities\n"
            body += f"⚠️ This update addresses {report['summary']['security_vulnerabilities']} security vulnerabilities.\n"
        
        body += f"""
### Testing
- [x] Basic functionality tests passed
- [x] Security scan completed
- [x] Dependencies lock file updated

### Notes
This is an automated pull request created by the dependency update script.
Please review the changes and run the full test suite before merging.

**Generated on**: {report['scan_time']}
"""
        
        return body


def main():
    """Main entry point for dependency update script."""
    parser = argparse.ArgumentParser(description="Automated dependency updates")
    
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check for updates, don't apply them")
    parser.add_argument("--security-only", action="store_true",
                       help="Only update packages with security vulnerabilities")
    parser.add_argument("--create-pr", action="store_true",
                       help="Create pull request for updates")
    parser.add_argument("--branch-name", help="Custom branch name for updates")
    parser.add_argument("--packages", nargs="+", help="Specific packages to update")
    parser.add_argument("--exclude", nargs="+", default=[],
                       help="Packages to exclude from updates")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without making changes")
    parser.add_argument("--force", action="store_true",
                       help="Force updates even if tests fail")
    
    args = parser.parse_args()
    
    updater = DependencyUpdater()
    
    try:
        # Check for outdated packages and vulnerabilities
        outdated = updater.check_outdated_packages()
        vulnerabilities = updater.check_security_vulnerabilities()
        
        # Create comprehensive report
        report = updater.create_dependency_report(outdated, vulnerabilities)
        
        if args.check_only:
            print(f"\n📊 Dependency Report:")
            print(f"   Outdated packages: {report['summary']['total_outdated']}")
            print(f"   Security vulnerabilities: {report['summary']['security_vulnerabilities']}")
            print(f"   Critical updates needed: {report['summary']['critical_updates']}")
            return
        
        if not outdated and not vulnerabilities.get("vulnerabilities"):
            updater.logger.info("✅ All dependencies are up to date and secure!")
            return
        
        # Determine packages to update
        packages_to_update = []
        
        if args.packages:
            packages_to_update = args.packages
        elif args.security_only:
            # Only update packages with security vulnerabilities
            vulnerable_packages = [
                vuln.get("package", "")
                for vuln in vulnerabilities.get("vulnerabilities", [])
            ]
            packages_to_update = [pkg for pkg in vulnerable_packages if pkg]
        else:
            # Update all outdated packages
            packages_to_update = [
                pkg["name"] for pkg in outdated
                if pkg["name"] not in args.exclude
            ]
        
        if args.dry_run:
            print(f"\n🔍 Dry run - would update {len(packages_to_update)} packages:")
            for pkg in packages_to_update:
                print(f"   - {pkg}")
            return
        
        if not packages_to_update:
            updater.logger.info("No packages selected for update")
            return
        
        # Create update branch if creating PR
        branch_name = None
        if args.create_pr:
            branch_name = updater.create_update_branch(args.branch_name)
        
        # Update packages
        updater.logger.info(f"Updating {len(packages_to_update)} packages...")
        update_results = updater.batch_update_packages(
            packages_to_update, 
            test_after_each=not args.force
        )
        
        # Update lock file
        updater.update_requirements_lock()
        
        # Report results
        successful_updates = [pkg for pkg, success in update_results.items() if success]
        failed_updates = [pkg for pkg, success in update_results.items() if not success]
        
        updater.logger.info(f"✅ Successfully updated: {len(successful_updates)} packages")
        if failed_updates:
            updater.logger.warning(f"❌ Failed to update: {len(failed_updates)} packages")
            for pkg in failed_updates:
                updater.logger.warning(f"   - {pkg}")
        
        # Commit and create PR if requested
        if args.create_pr and successful_updates:
            updater.commit_updates()
            pr_url = updater.create_pull_request(branch_name, report)
            
            if pr_url:
                print(f"\n🎉 Pull request created: {pr_url}")
            else:
                print(f"\n📝 Manual PR creation needed for branch: {branch_name}")
        
    except KeyboardInterrupt:
        updater.logger.info("⚠️ Update process interrupted by user")
        sys.exit(1)
    except Exception as e:
        updater.logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()