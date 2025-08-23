#!/usr/bin/env python3
"""
AI Scientist v2 Health Check Script
Comprehensive health monitoring for production deployment

This script performs various health checks including:
- Application status
- Database connectivity
- Cache availability
- GPU resources
- Memory usage
- Disk space
- Network connectivity
"""

import sys
import os
import json
import time
import psutil
import logging
import argparse
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the application path
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for AI Scientist v2"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or '/app/config/production.yaml'
        self.results = {}
        self.overall_status = True
        
    def check_application_status(self) -> Dict[str, Any]:
        """Check if the main application is responding"""
        try:
            # Check if the application port is open
            port = int(os.environ.get('PORT', '8000'))
            response = requests.get(
                f'http://localhost:{port}/health',
                timeout=10,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'details': response.json() if response.content else {}
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f'HTTP {response.status_code}',
                    'details': response.text[:200]
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f'Unexpected error: {str(e)}'
            }
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity if configured"""
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return {
                    'status': 'not_configured',
                    'message': 'Database not configured'
                }
            
            import psycopg2
            from urllib.parse import urlparse
            
            parsed = urlparse(database_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return {
                'status': 'healthy',
                'message': 'Database connection successful'
            }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'psycopg2 not installed'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_redis_connectivity(self) -> Dict[str, Any]:
        """Check Redis connectivity if configured"""
        try:
            redis_url = os.environ.get('REDIS_URL')
            if not redis_url:
                return {
                    'status': 'not_configured',
                    'message': 'Redis not configured'
                }
            
            import redis
            
            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            
            info = r.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            
            return {
                'status': 'healthy',
                'memory_usage': memory_usage,
                'connected_clients': info.get('connected_clients', 0)
            }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'redis not installed'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_gpu_resources(self) -> Dict[str, Any]:
        """Check GPU availability and usage"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'status': 'not_available',
                    'message': 'CUDA not available'
                }
            
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                gpu_info.append({
                    'device_id': i,
                    'name': props.name,
                    'memory_allocated_mb': memory_allocated // (1024 * 1024),
                    'memory_reserved_mb': memory_reserved // (1024 * 1024),
                    'memory_total_mb': memory_total // (1024 * 1024),
                    'memory_usage_percent': (memory_reserved / memory_total) * 100
                })
            
            return {
                'status': 'healthy',
                'gpu_count': gpu_count,
                'gpus': gpu_info
            }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'PyTorch not installed'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, memory, disk)"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/app')
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = os.getloadavg()
            except AttributeError:
                pass  # Not available on Windows
            
            # Check for resource constraints
            warnings = []
            if cpu_percent > 90:
                warnings.append(f'High CPU usage: {cpu_percent}%')
            if memory.percent > 90:
                warnings.append(f'High memory usage: {memory.percent}%')
            if disk.percent > 90:
                warnings.append(f'High disk usage: {disk.percent}%')
            
            status = 'healthy'
            if warnings:
                status = 'warning' if len(warnings) <= 2 else 'unhealthy'
            
            return {
                'status': status,
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_average': load_avg
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'available_mb': memory.available // (1024 * 1024),
                    'total_mb': memory.total // (1024 * 1024)
                },
                'disk': {
                    'usage_percent': disk.percent,
                    'free_gb': disk.free // (1024 * 1024 * 1024),
                    'total_gb': disk.total // (1024 * 1024 * 1024)
                },
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_application_modules(self) -> Dict[str, Any]:
        """Check if critical application modules can be imported"""
        critical_modules = [
            'ai_scientist',
            'ai_scientist.autonomous_sdlc_orchestrator',
            'ai_scientist.research.adaptive_tree_search',
            'ai_scientist.monitoring.health_checks',
        ]
        
        module_status = {}
        failed_modules = []
        
        for module in critical_modules:
            try:
                __import__(module)
                module_status[module] = 'ok'
            except ImportError as e:
                module_status[module] = f'import_error: {str(e)}'
                failed_modules.append(module)
            except Exception as e:
                module_status[module] = f'error: {str(e)}'
                failed_modules.append(module)
        
        if failed_modules:
            return {
                'status': 'unhealthy',
                'failed_modules': failed_modules,
                'module_status': module_status
            }
        else:
            return {
                'status': 'healthy',
                'module_status': module_status
            }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check critical file and directory permissions"""
        critical_paths = [
            '/app/data',
            '/app/logs',
            '/app/cache',
            '/app/experiments'
        ]
        
        permission_issues = []
        
        for path in critical_paths:
            path_obj = Path(path)
            try:
                if not path_obj.exists():
                    permission_issues.append(f'{path}: does not exist')
                    continue
                
                if not path_obj.is_dir():
                    permission_issues.append(f'{path}: not a directory')
                    continue
                
                # Test write permissions
                test_file = path_obj / '.health_check_test'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                except PermissionError:
                    permission_issues.append(f'{path}: not writable')
                except Exception as e:
                    permission_issues.append(f'{path}: write test failed - {str(e)}')
                    
            except Exception as e:
                permission_issues.append(f'{path}: access error - {str(e)}')
        
        if permission_issues:
            return {
                'status': 'unhealthy',
                'issues': permission_issues
            }
        else:
            return {
                'status': 'healthy',
                'message': 'All critical paths accessible'
            }
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check for required environment variables"""
        required_vars = [
            'PYTHONPATH',
        ]
        
        optional_vars = [
            'ANTHROPIC_API_KEY',
            'OPENAI_API_KEY',
            'REDIS_URL',
            'DATABASE_URL'
        ]
        
        missing_required = []
        missing_optional = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_required.append(var)
        
        for var in optional_vars:
            if not os.environ.get(var):
                missing_optional.append(var)
        
        status = 'healthy'
        if missing_required:
            status = 'unhealthy'
        elif missing_optional:
            status = 'warning'
        
        return {
            'status': status,
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'configured_vars': len([v for v in required_vars + optional_vars if os.environ.get(v)])
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = {
            'application': self.check_application_status,
            'database': self.check_database_connectivity,
            'redis': self.check_redis_connectivity,
            'gpu': self.check_gpu_resources,
            'system_resources': self.check_system_resources,
            'application_modules': self.check_application_modules,
            'file_permissions': self.check_file_permissions,
            'environment_variables': self.check_environment_variables,
        }
        
        results = {}
        overall_healthy = True
        
        for check_name, check_func in checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                # Determine if this check affects overall health
                status = result.get('status', 'unknown')
                if status in ['unhealthy', 'error']:
                    overall_healthy = False
                    
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'error': f'Check failed: {str(e)}'
                }
                overall_healthy = False
        
        return {
            'timestamp': time.time(),
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'summary': {
                'total_checks': len(checks),
                'passed': len([r for r in results.values() if r.get('status') == 'healthy']),
                'warnings': len([r for r in results.values() if r.get('status') == 'warning']),
                'failed': len([r for r in results.values() if r.get('status') in ['unhealthy', 'error']]),
                'not_configured': len([r for r in results.values() if r.get('status') == 'not_configured'])
            }
        }

def main():
    parser = argparse.ArgumentParser(description='AI Scientist v2 Health Check')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (minimal output)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--check', help='Run specific check only')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    health_checker = HealthChecker(args.config)
    
    if args.check:
        # Run specific check
        check_methods = {
            'app': health_checker.check_application_status,
            'application': health_checker.check_application_status,
            'db': health_checker.check_database_connectivity,
            'database': health_checker.check_database_connectivity,
            'redis': health_checker.check_redis_connectivity,
            'gpu': health_checker.check_gpu_resources,
            'system': health_checker.check_system_resources,
            'modules': health_checker.check_application_modules,
            'permissions': health_checker.check_file_permissions,
            'env': health_checker.check_environment_variables,
        }
        
        check_func = check_methods.get(args.check.lower())
        if not check_func:
            print(f"Unknown check: {args.check}")
            print(f"Available checks: {', '.join(check_methods.keys())}")
            sys.exit(1)
        
        result = check_func()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Check '{args.check}': {result.get('status', 'unknown')}")
            if not args.quiet:
                print(json.dumps(result, indent=2))
        
        sys.exit(0 if result.get('status') == 'healthy' else 1)
    
    else:
        # Run all checks
        results = health_checker.run_all_checks()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            overall_status = results['overall_status']
            summary = results['summary']
            
            print(f"Overall Health: {overall_status.upper()}")
            print(f"Checks: {summary['passed']} passed, {summary['warnings']} warnings, {summary['failed']} failed, {summary['not_configured']} not configured")
            
            if not args.quiet:
                print("\nDetailed Results:")
                for check_name, check_result in results['checks'].items():
                    status = check_result.get('status', 'unknown')
                    print(f"  {check_name}: {status}")
                    
                    if status in ['unhealthy', 'error', 'warning'] and not args.quiet:
                        error = check_result.get('error')
                        if error:
                            print(f"    Error: {error}")
                        
                        warnings = check_result.get('warnings')
                        if warnings:
                            for warning in warnings:
                                print(f"    Warning: {warning}")
        
        sys.exit(0 if results['overall_status'] == 'healthy' else 1)

if __name__ == '__main__':
    main()