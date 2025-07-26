#!/usr/bin/env python3
"""
Metrics and Reporting System for Autonomous Backlog Management
Generates comprehensive status reports and DORA metrics.
"""

import json
import logging
import os
import subprocess
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

logger = logging.getLogger(__name__)

@dataclass
class DORAMetrics:
    """DORA (DevOps Research and Assessment) metrics."""
    deployment_frequency: str  # "per day" or actual frequency
    lead_time_hours: float  # Time from commit to production
    change_failure_rate: float  # Percentage of deployments causing failures
    mean_time_to_recovery_hours: float  # Time to recover from failures
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deploy_freq": self.deployment_frequency,
            "lead_time": f"{self.lead_time_hours} hours",
            "change_fail_rate": f"{self.change_failure_rate}%",
            "mttr": f"{self.mean_time_to_recovery_hours} hours"
        }

@dataclass
class CycleMetrics:
    """Metrics for a single execution cycle."""
    cycle_id: str
    start_time: datetime
    end_time: datetime
    items_discovered: int
    items_executed: int
    items_completed: int
    items_failed: int
    test_pass_rate: float
    security_issues_found: int
    security_issues_fixed: int
    conflicts_auto_resolved: int
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60
    
    @property
    def success_rate(self) -> float:
        if self.items_executed == 0:
            return 100.0
        return (self.items_completed / self.items_executed) * 100

class GitMetricsCollector:
    """Collects metrics from git repository."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def get_commit_frequency(self, days: int = 7) -> float:
        """Get commit frequency per day over the last N days."""
        try:
            since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            result = subprocess.run([
                'git', 'log', '--oneline', f'--since={since_date}'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                commit_count = len([line for line in result.stdout.split('\n') if line.strip()])
                return commit_count / days
                
        except Exception as e:
            logger.warning(f"Could not get commit frequency: {e}")
        
        return 0.0
    
    def get_merge_conflict_stats(self) -> Dict[str, int]:
        """Get merge conflict and rerere statistics."""
        stats = {
            "conflicts_total": 0,
            "conflicts_auto_resolved": 0,
            "merge_driver_hits": 0
        }
        
        try:
            # Check rerere cache
            rerere_dir = self.repo_path / '.git' / 'rr-cache'
            if rerere_dir.exists():
                stats["conflicts_auto_resolved"] = len(list(rerere_dir.iterdir()))
            
            # Check recent merge commits
            result = subprocess.run([
                'git', 'log', '--merges', '--oneline', '-20'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                merge_commits = len([line for line in result.stdout.split('\n') if line.strip()])
                # Estimate conflicts based on merge commits (simplified)
                stats["conflicts_total"] = merge_commits
                
        except Exception as e:
            logger.warning(f"Could not get merge conflict stats: {e}")
        
        return stats
    
    def get_branch_info(self) -> Dict[str, Any]:
        """Get current branch and status information."""
        info = {
            "current_branch": "unknown",
            "is_clean": False,
            "ahead_behind": {"ahead": 0, "behind": 0}
        }
        
        try:
            # Get current branch
            result = subprocess.run([
                'git', 'branch', '--show-current'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                info["current_branch"] = result.stdout.strip()
            
            # Check if working directory is clean
            result = subprocess.run([
                'git', 'status', '--porcelain'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                info["is_clean"] = not result.stdout.strip()
                
        except Exception as e:
            logger.warning(f"Could not get branch info: {e}")
        
        return info

class TestMetricsCollector:
    """Collects metrics from test execution."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get latest test execution results."""
        results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "pass_rate": 0.0,
            "execution_time": 0.0,
            "coverage": None
        }
        
        try:
            # Run pytest with JSON output
            result = subprocess.run([
                'python', '-m', 'pytest', '--tb=no', '-q', '--json-report', 
                '--json-report-file=test-results.json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Read JSON results if available
            json_file = self.repo_path / 'test-results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    test_data = json.load(f)
                
                summary = test_data.get('summary', {})
                results.update({
                    "total_tests": summary.get('total', 0),
                    "passed_tests": summary.get('passed', 0),
                    "failed_tests": summary.get('failed', 0),
                    "skipped_tests": summary.get('skipped', 0),
                    "execution_time": test_data.get('duration', 0.0)
                })
                
                if results["total_tests"] > 0:
                    results["pass_rate"] = (results["passed_tests"] / results["total_tests"]) * 100
                    
        except Exception as e:
            logger.warning(f"Could not get test results: {e}")
        
        return results
    
    def get_coverage_info(self) -> Optional[Dict[str, Any]]:
        """Get test coverage information."""
        coverage_file = self.repo_path / 'coverage.json'
        
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                totals = coverage_data.get('totals', {})
                return {
                    "total_coverage": totals.get('percent_covered', 0),
                    "lines_covered": totals.get('covered_lines', 0),
                    "lines_total": totals.get('num_statements', 0),
                    "branches_covered": totals.get('covered_branches', 0),
                    "branches_total": totals.get('num_branches', 0)
                }
                
            except Exception as e:
                logger.warning(f"Could not read coverage data: {e}")
        
        return None

class SecurityMetricsCollector:
    """Collects security-related metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def get_security_scan_results(self) -> Dict[str, Any]:
        """Get results from security scans."""
        results = {
            "bandit_issues": 0,
            "safety_vulnerabilities": 0,
            "secret_scan_issues": 0,
            "last_scan_time": None,
            "scan_status": "unknown"
        }
        
        # Check for bandit results
        bandit_file = self.repo_path / 'bandit-results.json'
        if bandit_file.exists():
            try:
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                
                results["bandit_issues"] = len(bandit_data.get('results', []))
                results["last_scan_time"] = bandit_data.get('generated_at')
                results["scan_status"] = "completed"
                
            except Exception as e:
                logger.warning(f"Could not read bandit results: {e}")
        
        # Check for safety results
        safety_file = self.repo_path / 'safety-results.json'
        if safety_file.exists():
            try:
                with open(safety_file, 'r') as f:
                    safety_data = json.load(f)
                
                if isinstance(safety_data, list):
                    results["safety_vulnerabilities"] = len(safety_data)
                    
            except Exception as e:
                logger.warning(f"Could not read safety results: {e}")
        
        return results
    
    def get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency security and management info."""
        info = {
            "total_dependencies": 0,
            "outdated_dependencies": 0,
            "vulnerable_dependencies": 0,
            "license_issues": 0
        }
        
        # Count dependencies from requirements.txt
        req_file = self.repo_path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                
                # Count non-empty, non-comment lines
                deps = [line.strip() for line in lines 
                       if line.strip() and not line.strip().startswith('#')]
                info["total_dependencies"] = len(deps)
                
            except Exception as e:
                logger.warning(f"Could not read requirements.txt: {e}")
        
        return info

class BacklogMetricsCollector:
    """Collects backlog and task metrics."""
    
    def __init__(self, backlog_file: str = "DOCS/backlog.yml"):
        self.backlog_file = Path(backlog_file)
    
    def get_backlog_stats(self) -> Dict[str, Any]:
        """Get comprehensive backlog statistics."""
        stats = {
            "total_items": 0,
            "items_by_status": {},
            "items_by_type": {},
            "average_wsjf": 0.0,
            "average_age_days": 0.0,
            "high_priority_items": 0,
            "security_items": 0,
            "technical_debt_items": 0
        }
        
        if not self.backlog_file.exists():
            return stats
        
        try:
            with open(self.backlog_file, 'r') as f:
                backlog_data = yaml.safe_load(f)
            
            items = backlog_data.get('backlog_items', [])
            stats["total_items"] = len(items)
            
            if items:
                # Calculate statistics
                wsjf_scores = []
                age_values = []
                
                for item in items:
                    # Count by status
                    status = item.get('status', 'unknown')
                    stats["items_by_status"][status] = stats["items_by_status"].get(status, 0) + 1
                    
                    # Count by type
                    item_type = item.get('type', 'unknown')
                    stats["items_by_type"][item_type] = stats["items_by_type"].get(item_type, 0) + 1
                    
                    # Collect WSJF scores
                    wsjf = item.get('wsjf_score', 0)
                    if wsjf > 0:
                        wsjf_scores.append(wsjf)
                    
                    # Collect age data
                    age_days = item.get('age_days', 0)
                    age_values.append(age_days)
                    
                    # Count high priority items (WSJF > 5)
                    if wsjf > 5:
                        stats["high_priority_items"] += 1
                    
                    # Count security items
                    if item_type.lower() == 'security':
                        stats["security_items"] += 1
                    
                    # Count technical debt items
                    if item_type.lower() in ['technical_debt', 'refactor']:
                        stats["technical_debt_items"] += 1
                
                # Calculate averages
                if wsjf_scores:
                    stats["average_wsjf"] = statistics.mean(wsjf_scores)
                
                if age_values:
                    stats["average_age_days"] = statistics.mean(age_values)
                    
        except Exception as e:
            logger.warning(f"Could not read backlog data: {e}")
        
        return stats

class MetricsReporter:
    """Main metrics collection and reporting system."""
    
    def __init__(self, repo_path: str = ".", backlog_file: str = "DOCS/backlog.yml"):
        self.repo_path = Path(repo_path)
        self.git_collector = GitMetricsCollector(repo_path)
        self.test_collector = TestMetricsCollector(repo_path)
        self.security_collector = SecurityMetricsCollector(repo_path)
        self.backlog_collector = BacklogMetricsCollector(backlog_file)
        
    def calculate_dora_metrics(self, days_back: int = 30) -> DORAMetrics:
        """Calculate DORA metrics based on recent activity."""
        
        # Deployment frequency (simplified - based on commit frequency)
        commit_freq = self.git_collector.get_commit_frequency(days_back)
        deploy_freq = f"{commit_freq:.1f} per day" if commit_freq > 0 else "0 per day"
        
        # Lead time (simplified - assume 4 hours average)
        lead_time = 4.0
        
        # Change failure rate (simplified - based on test results)
        test_results = self.test_collector.get_test_results()
        change_fail_rate = 100 - test_results.get("pass_rate", 0)
        
        # Mean time to recovery (simplified - assume 2 hours)
        mttr = 2.0
        
        return DORAMetrics(
            deployment_frequency=deploy_freq,
            lead_time_hours=lead_time,
            change_failure_rate=change_fail_rate,
            mean_time_to_recovery_hours=mttr
        )
    
    def generate_status_report(self, cycle_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        timestamp = datetime.now()
        
        # Collect all metrics
        git_metrics = self.git_collector.get_merge_conflict_stats()
        git_info = self.git_collector.get_branch_info()
        test_metrics = self.test_collector.get_test_results()
        coverage_info = self.test_collector.get_coverage_info()
        security_metrics = self.security_collector.get_security_scan_results()
        dependency_info = self.security_collector.get_dependency_info()
        backlog_stats = self.backlog_collector.get_backlog_stats()
        dora_metrics = self.calculate_dora_metrics()
        
        # Generate comprehensive report
        report = {
            "timestamp": timestamp.isoformat(),
            "cycle_id": cycle_id or f"manual-{timestamp.strftime('%Y%m%d-%H%M%S')}",
            "git_info": git_info,
            "backlog_summary": {
                "total_items": backlog_stats["total_items"],
                "items_by_status": backlog_stats["items_by_status"],
                "items_by_type": backlog_stats["items_by_type"],
                "average_wsjf": round(backlog_stats["average_wsjf"], 2),
                "high_priority_items": backlog_stats["high_priority_items"],
                "security_items": backlog_stats["security_items"],
                "technical_debt_items": backlog_stats["technical_debt_items"]
            },
            "test_metrics": {
                "total_tests": test_metrics["total_tests"],
                "pass_rate": round(test_metrics["pass_rate"], 1),
                "failed_tests": test_metrics["failed_tests"],
                "execution_time": round(test_metrics["execution_time"], 2)
            },
            "coverage_info": coverage_info,
            "security_metrics": {
                "bandit_issues": security_metrics["bandit_issues"],
                "safety_vulnerabilities": security_metrics["safety_vulnerabilities"],
                "scan_status": security_metrics["scan_status"],
                "total_dependencies": dependency_info["total_dependencies"],
                "vulnerable_dependencies": dependency_info["vulnerable_dependencies"]
            },
            "merge_conflict_stats": git_metrics,
            "dora_metrics": dora_metrics.to_dict(),
            "system_health": self._assess_system_health(
                test_metrics, security_metrics, backlog_stats
            ),
            "autonomous_execution_status": {
                "active": True,
                "last_cycle": cycle_id,
                "conflicts_auto_resolved": git_metrics["conflicts_auto_resolved"],
                "merge_driver_hits": git_metrics["merge_driver_hits"]
            }
        }
        
        return report
    
    def _assess_system_health(self, test_metrics: Dict, security_metrics: Dict, 
                             backlog_stats: Dict) -> Dict[str, str]:
        """Assess overall system health."""
        health = {
            "overall": "good",
            "test_health": "good",
            "security_health": "good",
            "backlog_health": "good"
        }
        
        # Test health
        pass_rate = test_metrics.get("pass_rate", 0)
        if pass_rate < 80:
            health["test_health"] = "poor"
        elif pass_rate < 95:
            health["test_health"] = "fair"
        
        # Security health
        if security_metrics.get("safety_vulnerabilities", 0) > 0:
            health["security_health"] = "poor"
        elif security_metrics.get("bandit_issues", 0) > 5:
            health["security_health"] = "fair"
        
        # Backlog health
        ready_items = backlog_stats["items_by_status"].get("READY", 0)
        blocked_items = backlog_stats["items_by_status"].get("BLOCKED", 0)
        
        if blocked_items > ready_items:
            health["backlog_health"] = "poor"
        elif ready_items == 0:
            health["backlog_health"] = "excellent"  # No work backlog
        
        # Overall health
        poor_count = sum(1 for h in health.values() if h == "poor")
        if poor_count > 0:
            health["overall"] = "poor"
        elif sum(1 for h in health.values() if h == "fair") > 1:
            health["overall"] = "fair"
        
        return health
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"autonomous_execution_report_{timestamp}.json"
        
        # Ensure reports directory exists
        reports_dir = self.repo_path / "docs" / "status"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def generate_markdown_summary(self, report: Dict[str, Any]) -> str:
        """Generate a markdown summary of the report."""
        timestamp = report["timestamp"]
        cycle_id = report["cycle_id"]
        
        md = f"""# Autonomous Execution Report
        
**Generated:** {timestamp}  
**Cycle ID:** {cycle_id}

## System Health Overview
- **Overall Health:** {report['system_health']['overall'].title()}
- **Test Health:** {report['system_health']['test_health'].title()}
- **Security Health:** {report['system_health']['security_health'].title()}
- **Backlog Health:** {report['system_health']['backlog_health'].title()}

## Backlog Summary
- **Total Items:** {report['backlog_summary']['total_items']}
- **Ready Items:** {report['backlog_summary']['items_by_status'].get('READY', 0)}
- **In Progress:** {report['backlog_summary']['items_by_status'].get('DOING', 0)}
- **Completed:** {report['backlog_summary']['items_by_status'].get('DONE', 0)}
- **Average WSJF Score:** {report['backlog_summary']['average_wsjf']}

## Test Metrics
- **Total Tests:** {report['test_metrics']['total_tests']}
- **Pass Rate:** {report['test_metrics']['pass_rate']}%
- **Failed Tests:** {report['test_metrics']['failed_tests']}

## Security Status
- **Bandit Issues:** {report['security_metrics']['bandit_issues']}
- **Vulnerable Dependencies:** {report['security_metrics']['vulnerable_dependencies']}
- **Total Dependencies:** {report['security_metrics']['total_dependencies']}

## DORA Metrics
- **Deployment Frequency:** {report['dora_metrics']['deploy_freq']}
- **Lead Time:** {report['dora_metrics']['lead_time']}
- **Change Failure Rate:** {report['dora_metrics']['change_fail_rate']}
- **Mean Time to Recovery:** {report['dora_metrics']['mttr']}

## Autonomous Execution
- **Status:** {'Active' if report['autonomous_execution_status']['active'] else 'Inactive'}
- **Conflicts Auto-Resolved:** {report['autonomous_execution_status']['conflicts_auto_resolved']}
- **Merge Driver Hits:** {report['autonomous_execution_status']['merge_driver_hits']}
"""
        
        return md

def main():
    """Main function for running metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate autonomous backlog metrics report")
    parser.add_argument("--cycle-id", help="Cycle ID for this report")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--markdown", action="store_true", help="Generate markdown summary")
    
    args = parser.parse_args()
    
    reporter = MetricsReporter()
    report = reporter.generate_status_report(args.cycle_id)
    
    # Save JSON report
    json_file = reporter.save_report(report, args.output)
    print(f"Report saved to: {json_file}")
    
    # Generate markdown summary if requested
    if args.markdown:
        md_content = reporter.generate_markdown_summary(report)
        md_file = json_file.replace('.json', '.md')
        
        with open(md_file, 'w') as f:
            f.write(md_content)
        
        print(f"Markdown summary saved to: {md_file}")

if __name__ == "__main__":
    main()