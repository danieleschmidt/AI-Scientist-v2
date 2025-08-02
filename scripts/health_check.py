#!/usr/bin/env python3
"""
Health check script for AI Scientist v2 Docker containers.
Verifies system health and readiness for scientific research operations.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional


class HealthChecker:
    """Performs comprehensive health checks for AI Scientist v2."""
    
    def __init__(self):
        self.health_status = {
            "timestamp": time.time(),
            "status": "healthy",
            "checks": {},
            "errors": []
        }
    
    def check_python_environment(self) -> bool:
        """Check Python environment and core dependencies."""
        try:
            # Check Python version
            if sys.version_info < (3, 11):
                self.health_status["errors"].append("Python version must be 3.11 or higher")
                return False
            
            # Check core imports
            import torch
            import numpy as np
            import requests
            
            self.health_status["checks"]["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"
            self.health_status["checks"]["torch_available"] = True
            self.health_status["checks"]["numpy_available"] = True
            self.health_status["checks"]["requests_available"] = True
            
            # Check GPU availability
            if torch.cuda.is_available():
                self.health_status["checks"]["gpu_available"] = True
                self.health_status["checks"]["gpu_count"] = torch.cuda.device_count()
            else:
                self.health_status["checks"]["gpu_available"] = False
            
            return True
            
        except ImportError as e:
            self.health_status["errors"].append(f"Missing dependency: {e}")
            return False
        except Exception as e:
            self.health_status["errors"].append(f"Python environment error: {e}")
            return False
    
    def check_file_system(self) -> bool:
        """Check file system accessibility and permissions."""
        try:
            # Check if we can write to required directories
            required_dirs = ["/app", "/app/experiments", "/app/data", "/app/logs"]
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    self.health_status["errors"].append(f"Required directory missing: {dir_path}")
                    return False
                
                # Test write permissions
                test_file = path / ".health_check_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except (PermissionError, OSError):
                    self.health_status["errors"].append(f"No write access to: {dir_path}")
                    return False
            
            self.health_status["checks"]["filesystem_writable"] = True
            return True
            
        except Exception as e:
            self.health_status["errors"].append(f"Filesystem error: {e}")
            return False
    
    def check_ai_scientist_module(self) -> bool:
        """Check if AI Scientist module is importable and functional."""
        try:
            # Try to import main AI Scientist components
            from ai_scientist import llm
            from ai_scientist.utils import config
            
            self.health_status["checks"]["ai_scientist_importable"] = True
            
            # Check if configuration can be loaded
            try:
                # This would check if basic configuration works
                config_path = Path("/app/ai_scientist_config.yaml")
                if config_path.exists():
                    self.health_status["checks"]["config_file_exists"] = True
                else:
                    self.health_status["checks"]["config_file_exists"] = False
            except Exception:
                self.health_status["checks"]["config_file_exists"] = False
            
            return True
            
        except ImportError as e:
            self.health_status["errors"].append(f"AI Scientist module import error: {e}")
            return False
        except Exception as e:
            self.health_status["errors"].append(f"AI Scientist module error: {e}")
            return False
    
    def check_external_dependencies(self) -> bool:
        """Check external tools and dependencies."""
        try:
            import subprocess
            
            # Check LaTeX installation
            try:
                result = subprocess.run(
                    ["pdflatex", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                self.health_status["checks"]["latex_available"] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.health_status["checks"]["latex_available"] = False
            
            # Check Git
            try:
                result = subprocess.run(
                    ["git", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self.health_status["checks"]["git_available"] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.health_status["checks"]["git_available"] = False
            
            return True
            
        except Exception as e:
            self.health_status["errors"].append(f"External dependencies error: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        try:
            # Check for API keys (don't log the values for security)
            api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "S2_API_KEY"]
            available_keys = []
            
            for key in api_keys:
                if os.getenv(key):
                    available_keys.append(key)
            
            self.health_status["checks"]["api_keys_configured"] = len(available_keys)
            self.health_status["checks"]["available_api_providers"] = available_keys
            
            # Check other environment variables
            env_vars = {
                "PYTHONPATH": os.getenv("PYTHONPATH"),
                "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES")
            }
            
            self.health_status["checks"]["environment_variables"] = env_vars
            
            return True
            
        except Exception as e:
            self.health_status["errors"].append(f"Environment variables error: {e}")
            return False
    
    def check_memory_and_resources(self) -> bool:
        """Check system resources and memory."""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            self.health_status["checks"]["memory_total_gb"] = round(memory.total / (1024**3), 2)
            self.health_status["checks"]["memory_available_gb"] = round(memory.available / (1024**3), 2)
            self.health_status["checks"]["memory_percent_used"] = memory.percent
            
            # Disk space check
            disk = psutil.disk_usage('/')
            self.health_status["checks"]["disk_total_gb"] = round(disk.total / (1024**3), 2)
            self.health_status["checks"]["disk_free_gb"] = round(disk.free / (1024**3), 2)
            self.health_status["checks"]["disk_percent_used"] = round((disk.used / disk.total) * 100, 2)
            
            # CPU check
            self.health_status["checks"]["cpu_count"] = psutil.cpu_count()
            self.health_status["checks"]["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # Check if resources are sufficient
            if memory.available < 2 * 1024**3:  # Less than 2GB available
                self.health_status["errors"].append("Low memory: less than 2GB available")
                return False
            
            if disk.free < 5 * 1024**3:  # Less than 5GB free
                self.health_status["errors"].append("Low disk space: less than 5GB free")
                return False
            
            return True
            
        except ImportError:
            # psutil might not be available
            self.health_status["checks"]["resource_monitoring"] = "unavailable"
            return True
        except Exception as e:
            self.health_status["errors"].append(f"Resource check error: {e}")
            return False
    
    def run_all_checks(self) -> Dict:
        """Run all health checks and return comprehensive status."""
        checks = [
            ("python_environment", self.check_python_environment),
            ("file_system", self.check_file_system),
            ("ai_scientist_module", self.check_ai_scientist_module),
            ("external_dependencies", self.check_external_dependencies),
            ("environment_variables", self.check_environment_variables),
            ("memory_and_resources", self.check_memory_and_resources)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                passed = check_func()
                self.health_status["checks"][f"{check_name}_status"] = "passed" if passed else "failed"
                if not passed:
                    all_passed = False
            except Exception as e:
                self.health_status["checks"][f"{check_name}_status"] = "error"
                self.health_status["errors"].append(f"{check_name} check failed: {e}")
                all_passed = False
        
        # Set overall status
        if not all_passed:
            self.health_status["status"] = "unhealthy"
        elif self.health_status["errors"]:
            self.health_status["status"] = "degraded"
        else:
            self.health_status["status"] = "healthy"
        
        return self.health_status


def main():
    """Main health check entry point."""
    checker = HealthChecker()
    
    try:
        # Run health checks
        status = checker.run_all_checks()
        
        # Output results
        if os.getenv("HEALTH_CHECK_JSON", "false").lower() == "true":
            # JSON output for programmatic use
            print(json.dumps(status, indent=2))
        else:
            # Human-readable output
            print(f"Health Status: {status['status'].upper()}")
            
            if status['status'] == 'healthy':
                print("✅ All health checks passed")
            elif status['status'] == 'degraded':
                print("⚠️  System is running but has issues:")
                for error in status['errors']:
                    print(f"   - {error}")
            else:
                print("❌ Health checks failed:")
                for error in status['errors']:
                    print(f"   - {error}")
        
        # Exit with appropriate code for Docker health check
        if status['status'] == 'healthy':
            sys.exit(0)
        elif status['status'] == 'degraded':
            sys.exit(0)  # Still considered healthy for Docker
        else:
            sys.exit(1)  # Unhealthy
            
    except KeyboardInterrupt:
        print("Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Health check error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()