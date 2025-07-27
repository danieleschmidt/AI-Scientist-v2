#!/usr/bin/env python3
"""
Comprehensive maintenance and lifecycle automation for AI Scientist v2.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaintenanceManager:
    """Handles automated maintenance tasks."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load maintenance configuration."""
        config_file = self.repo_path / "maintenance_config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                return yaml.safe_load(f)
        
        return {
            "cleanup": {
                "retention_days": 30,
                "cleanup_paths": [
                    "experiments/*/logs/*.log",
                    "aisci_outputs/*/temp/*",
                    "results/*/cache/*",
                    "cache/*/*",
                    "**/__pycache__",
                    "**/*.pyc",
                    "**/*.pyo"
                ]
            },
            "monitoring": {
                "disk_usage_threshold": 85,
                "memory_usage_threshold": 80,
                "log_file_size_limit_mb": 100
            },
            "backup": {
                "enabled": True,
                "backup_paths": [
                    "ai_scientist/",
                    "docs/",
                    "pyproject.toml",
                    "requirements.txt",
                    ".github/",
                    "monitoring/",
                    "security/"
                ],
                "retention_count": 5
            }
        }
    
    def cleanup_old_files(self) -> Dict:
        """Clean up old files and artifacts."""
        logger.info("Starting file cleanup...")
        
        cleanup_config = self.config["cleanup"]
        retention_days = cleanup_config["retention_days"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleaned_files = []
        total_size_freed = 0
        
        for pattern in cleanup_config["cleanup_paths"]:
            pattern_path = self.repo_path / pattern
            
            # Handle glob patterns
            if "*" in pattern:
                from glob import glob
                for file_path in glob(str(pattern_path), recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file():
                        file_stat = file_path.stat()
                        file_date = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        if file_date < cutoff_date:
                            file_size = file_stat.st_size
                            try:
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                                total_size_freed += file_size
                                logger.debug(f"Deleted: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to delete {file_path}: {e}")
            else:
                # Handle direct paths
                if pattern_path.exists():
                    if pattern_path.is_file():
                        file_stat = pattern_path.stat()
                        file_date = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        if file_date < cutoff_date:
                            file_size = file_stat.st_size
                            try:
                                pattern_path.unlink()
                                cleaned_files.append(str(pattern_path))
                                total_size_freed += file_size
                            except Exception as e:
                                logger.error(f"Failed to delete {pattern_path}: {e}")
                    elif pattern_path.is_dir():
                        try:
                            shutil.rmtree(pattern_path)
                            cleaned_files.append(str(pattern_path))
                        except Exception as e:
                            logger.error(f"Failed to delete directory {pattern_path}: {e}")
        
        # Clean empty directories
        self._clean_empty_directories()
        
        result = {
            "files_cleaned": len(cleaned_files),
            "size_freed_mb": total_size_freed / (1024 * 1024),
            "cleaned_files": cleaned_files[:100]  # Limit output
        }
        
        logger.info(f"Cleanup complete: {result['files_cleaned']} files, {result['size_freed_mb']:.2f}MB freed")
        return result
    
    def _clean_empty_directories(self):
        """Remove empty directories."""
        for root, dirs, files in os.walk(self.repo_path, topdown=False):
            for directory in dirs:
                dir_path = Path(root) / directory
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logger.debug(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or permission denied
    
    def check_system_health(self) -> Dict:
        """Check system health metrics."""
        import psutil
        
        monitoring_config = self.config["monitoring"]
        health_issues = []
        
        # Check disk usage
        disk_usage = psutil.disk_usage(str(self.repo_path))
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        if disk_percent > monitoring_config["disk_usage_threshold"]:
            health_issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > monitoring_config["memory_usage_threshold"]:
            health_issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        # Check log file sizes
        log_files = list(self.repo_path.glob("**/*.log"))
        large_logs = []
        
        for log_file in log_files:
            if log_file.exists():
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > monitoring_config["log_file_size_limit_mb"]:
                    large_logs.append(f"{log_file}: {size_mb:.1f}MB")
        
        if large_logs:
            health_issues.extend([f"Large log file: {log}" for log in large_logs])
        
        return {
            "disk_usage_percent": disk_percent,
            "memory_usage_percent": memory.percent,
            "health_issues": health_issues,
            "status": "healthy" if not health_issues else "warning"
        }
    
    def update_dependencies(self) -> Dict:
        """Update and check dependencies."""
        logger.info("Checking dependency updates...")
        
        results = {
            "outdated_packages": [],
            "security_vulnerabilities": [],
            "update_available": False
        }
        
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                results["outdated_packages"] = outdated
                results["update_available"] = len(outdated) > 0
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check outdated packages: {e}")
        
        try:
            # Run safety check for security vulnerabilities
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                results["security_vulnerabilities"] = vulnerabilities
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Safety check not available")
        
        return results
    
    def create_backup(self) -> Dict:
        """Create system backup."""
        if not self.config["backup"]["enabled"]:
            return {"status": "disabled"}
        
        logger.info("Creating backup...")
        
        backup_config = self.config["backup"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.repo_path / "backups" / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backed_up_paths = []
        total_size = 0
        
        for path_pattern in backup_config["backup_paths"]:
            source_path = self.repo_path / path_pattern
            
            if source_path.exists():
                if source_path.is_file():
                    dest_path = backup_dir / path_pattern
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    backed_up_paths.append(path_pattern)
                    total_size += source_path.stat().st_size
                elif source_path.is_dir():
                    dest_path = backup_dir / path_pattern
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    backed_up_paths.append(path_pattern)
                    total_size += sum(f.stat().st_size for f in source_path.rglob('*') if f.is_file())
        
        # Clean old backups
        self._cleanup_old_backups(backup_config["retention_count"])
        
        # Create backup manifest
        manifest = {
            "timestamp": timestamp,
            "paths": backed_up_paths,
            "total_size_mb": total_size / (1024 * 1024),
            "git_commit": self._get_git_commit()
        }
        
        with open(backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Backup created: {backup_dir}")
        return {
            "status": "success",
            "backup_path": str(backup_dir),
            "size_mb": total_size / (1024 * 1024),
            "files_backed_up": len(backed_up_paths)
        }
    
    def _cleanup_old_backups(self, retention_count: int):
        """Remove old backups beyond retention limit."""
        backups_dir = self.repo_path / "backups"
        if not backups_dir.exists():
            return
        
        backup_dirs = sorted([d for d in backups_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_backup in backup_dirs[retention_count:]:
            try:
                shutil.rmtree(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.error(f"Failed to remove old backup {old_backup}: {e}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
                cwd=self.repo_path
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def rotate_logs(self) -> Dict:
        """Rotate and compress log files."""
        logger.info("Rotating logs...")
        
        log_files = list(self.repo_path.glob("**/*.log"))
        rotated_files = []
        
        for log_file in log_files:
            if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_name = f"{log_file.stem}_{timestamp}.log"
                rotated_path = log_file.parent / rotated_name
                
                try:
                    shutil.move(log_file, rotated_path)
                    
                    # Compress the rotated log
                    import gzip
                    with open(rotated_path, 'rb') as f_in:
                        with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    rotated_path.unlink()  # Remove uncompressed version
                    rotated_files.append(str(rotated_path))
                    
                    # Create new empty log file
                    log_file.touch()
                    
                except Exception as e:
                    logger.error(f"Failed to rotate log {log_file}: {e}")
        
        return {
            "rotated_files": len(rotated_files),
            "files": rotated_files
        }
    
    def generate_maintenance_report(self) -> Dict:
        """Generate comprehensive maintenance report."""
        logger.info("Generating maintenance report...")
        
        # Run all maintenance checks
        cleanup_result = self.cleanup_old_files()
        health_result = self.check_system_health()
        dependency_result = self.update_dependencies()
        backup_result = self.create_backup()
        log_rotation_result = self.rotate_logs()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "cleanup": cleanup_result,
            "system_health": health_result,
            "dependencies": dependency_result,
            "backup": backup_result,
            "log_rotation": log_rotation_result,
            "recommendations": self._generate_recommendations(
                health_result, dependency_result
            )
        }
        
        # Save report
        report_file = self.repo_path / "maintenance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Maintenance report saved: {report_file}")
        return report
    
    def _generate_recommendations(self, health_result: Dict, dependency_result: Dict) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        if health_result["health_issues"]:
            recommendations.append("Address system health issues identified")
        
        if dependency_result["update_available"]:
            recommendations.append("Update outdated dependencies")
        
        if dependency_result["security_vulnerabilities"]:
            recommendations.append("URGENT: Address security vulnerabilities in dependencies")
        
        if health_result["disk_usage_percent"] > 80:
            recommendations.append("Consider expanding disk space or cleaning more files")
        
        recommendations.append("Run maintenance weekly for optimal performance")
        recommendations.append("Monitor system metrics regularly")
        
        return recommendations


def main():
    """CLI interface for maintenance manager."""
    parser = argparse.ArgumentParser(description="AI Scientist v2 Maintenance Manager")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--action", choices=[
        "cleanup", "health", "dependencies", "backup", "logs", "report", "all"
    ], default="all", help="Maintenance action to perform")
    parser.add_argument("--config", help="Path to maintenance config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize maintenance manager
    repo_path = Path(args.repo_path).resolve()
    manager = MaintenanceManager(repo_path)
    
    # Perform requested action
    try:
        if args.action == "cleanup":
            result = manager.cleanup_old_files()
        elif args.action == "health":
            result = manager.check_system_health()
        elif args.action == "dependencies":
            result = manager.update_dependencies()
        elif args.action == "backup":
            result = manager.create_backup()
        elif args.action == "logs":
            result = manager.rotate_logs()
        elif args.action == "report":
            result = manager.generate_maintenance_report()
        else:  # all
            result = manager.generate_maintenance_report()
        
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        if args.action == "health" and result.get("status") == "warning":
            sys.exit(1)
        elif args.action == "dependencies" and result.get("security_vulnerabilities"):
            sys.exit(2)
        
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()