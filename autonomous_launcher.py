#!/usr/bin/env python3
"""
Autonomous Backlog Management System Launcher
Main entry point for running the complete autonomous system.
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Import our modules
from autonomous_backlog_manager import AutonomousExecutor
from metrics_reporter import MetricsReporter
from tdd_security_framework import TDDFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutonomousSystem:
    """Main autonomous system coordinator."""
    
    def __init__(self, max_cycles: int = 10, pr_limit: int = 5):
        self.max_cycles = max_cycles
        self.pr_limit = pr_limit
        self.executor = AutonomousExecutor()
        self.metrics_reporter = MetricsReporter()
        self.tdd_framework = TDDFramework()
        
    async def run_full_autonomous_cycle(self) -> bool:
        """Run a complete autonomous cycle with full integration."""
        cycle_id = f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting autonomous cycle: {cycle_id}")
        
        try:
            # Generate pre-cycle metrics
            pre_metrics = self.metrics_reporter.generate_status_report(f"{cycle_id}-pre")
            logger.info(f"Pre-cycle backlog size: {pre_metrics['backlog_summary']['total_items']}")
            
            # Execute the autonomous backlog cycle
            cycle_result = await self.executor.execute_macro_cycle()
            
            # Generate post-cycle metrics  
            post_metrics = self.metrics_reporter.generate_status_report(f"{cycle_id}-post")
            
            # Save comprehensive report
            report_file = self.metrics_reporter.save_report(post_metrics, f"{cycle_id}-report.json")
            
            # Generate markdown summary
            md_content = self.metrics_reporter.generate_markdown_summary(post_metrics)
            md_file = report_file.replace('.json', '.md')
            with open(md_file, 'w') as f:
                f.write(md_content)
            
            logger.info(f"Cycle {cycle_id} completed successfully")
            logger.info(f"Reports saved: {report_file}, {md_file}")
            
            return cycle_result.get('task_success', False)
            
        except Exception as e:
            logger.error(f"Autonomous cycle {cycle_id} failed: {e}")
            return False
    
    async def run_discovery_only(self) -> dict:
        """Run only the discovery phase."""
        logger.info("Running discovery-only mode")
        
        # Load existing backlog
        self.executor.backlog_manager.load_backlog()
        
        # Run discovery
        new_items = await self.executor.backlog_manager.discover_and_add_items()
        
        # Update aging and save
        self.executor.backlog_manager.update_aging_multipliers()
        self.executor.backlog_manager.save_backlog()
        
        # Generate discovery report
        report = self.metrics_reporter.generate_status_report("discovery")
        report['discovery_results'] = {
            'new_items_found': new_items,
            'discovery_sources': [
                'TODO/FIXME scanning',
                'failing test analysis', 
                'security scanning'
            ]
        }
        
        return report
    
    async def validate_system_health(self) -> dict:
        """Validate overall system health."""
        logger.info("Validating system health")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': {},
            'recommendations': []
        }
        
        # Validate backlog file
        try:
            self.executor.backlog_manager.load_backlog()
            health_report['validation_results']['backlog_valid'] = True
        except Exception as e:
            health_report['validation_results']['backlog_valid'] = False
            health_report['recommendations'].append(f"Fix backlog file: {e}")
        
        # Validate git configuration
        try:
            import subprocess
            result = subprocess.run(['git', 'config', 'rerere.enabled'], 
                                  capture_output=True, text=True)
            health_report['validation_results']['git_rerere_enabled'] = result.stdout.strip() == 'true'
            if not health_report['validation_results']['git_rerere_enabled']:
                health_report['recommendations'].append("Enable git rerere: git config rerere.enabled true")
        except Exception:
            health_report['validation_results']['git_rerere_enabled'] = False
            health_report['recommendations'].append("Configure git rerere for merge conflict resolution")
        
        # Validate test framework
        try:
            import subprocess
            result = subprocess.run(['python', '-m', 'pytest', '--version'], 
                                  capture_output=True, text=True)
            health_report['validation_results']['pytest_available'] = result.returncode == 0
        except Exception:
            health_report['validation_results']['pytest_available'] = False
            health_report['recommendations'].append("Install pytest for test framework")
        
        # Generate comprehensive metrics
        metrics = self.metrics_reporter.generate_status_report("health-check")
        health_report['system_metrics'] = metrics
        
        return health_report
    
    async def run_continuous_mode(self) -> None:
        """Run in continuous autonomous mode."""
        logger.info(f"Starting continuous autonomous mode (max {self.max_cycles} cycles)")
        
        cycles_completed = 0
        consecutive_failures = 0
        
        # Check for stop file
        stop_file = Path('.autonomous_stop')
        
        while cycles_completed < self.max_cycles and not stop_file.exists():
            try:
                # Run autonomous cycle
                success = await self.run_full_autonomous_cycle()
                cycles_completed += 1
                
                if success:
                    consecutive_failures = 0
                    logger.info(f"Cycle {cycles_completed} completed successfully")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Cycle {cycles_completed} failed (consecutive failures: {consecutive_failures})")
                
                # Stop if too many consecutive failures
                if consecutive_failures >= 3:
                    logger.error("Too many consecutive failures, stopping autonomous execution")
                    break
                
                # Check if we have more work
                ready_items = self.executor.backlog_manager.get_prioritized_items()
                if not ready_items:
                    logger.info("No more ready items in backlog, stopping execution")
                    break
                
                # Brief pause between cycles
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Autonomous execution interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous mode: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
        
        logger.info(f"Continuous mode completed: {cycles_completed} cycles executed")

async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management System")
    parser.add_argument('--mode', choices=['continuous', 'single', 'discovery', 'health'], 
                       default='single', help='Execution mode')
    parser.add_argument('--max-cycles', type=int, default=10, 
                       help='Maximum cycles in continuous mode')
    parser.add_argument('--pr-limit', type=int, default=5,
                       help='Daily PR limit')
    parser.add_argument('--output-dir', default='docs/status',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize system
    system = AutonomousSystem(max_cycles=args.max_cycles, pr_limit=args.pr_limit)
    
    try:
        if args.mode == 'continuous':
            await system.run_continuous_mode()
            
        elif args.mode == 'single':
            success = await system.run_full_autonomous_cycle()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'discovery':
            report = await system.run_discovery_only()
            print(f"Discovery completed. Found {report['discovery_results']['new_items_found']} new items.")
            
        elif args.mode == 'health':
            health = await system.validate_system_health()
            
            print("System Health Validation:")
            for check, result in health['validation_results'].items():
                status = "✓" if result else "✗"
                print(f"  {status} {check.replace('_', ' ').title()}")
            
            if health['recommendations']:
                print("\nRecommendations:")
                for rec in health['recommendations']:
                    print(f"  - {rec}")
            
            # Exit with error code if any validation failed
            all_valid = all(health['validation_results'].values())
            sys.exit(0 if all_valid else 1)
            
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        sys.exit(1)

def cli_status():
    """Quick status check function."""
    print("Autonomous Backlog Management System")
    print("=" * 40)
    
    # Check if system files exist
    required_files = [
        'autonomous_backlog_manager.py',
        'tdd_security_framework.py', 
        'metrics_reporter.py',
        'DOCS/backlog.yml'
    ]
    
    print("System Files:")
    for file in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Check git configuration
    try:
        import subprocess
        result = subprocess.run(['git', 'config', 'rerere.enabled'], 
                              capture_output=True, text=True)
        rerere_enabled = result.stdout.strip() == 'true'
        status = "✓" if rerere_enabled else "✗"
        print(f"  {status} Git rerere enabled")
    except Exception:
        print("  ✗ Git rerere configuration")
    
    print(f"\nUsage:")
    print(f"  python {__file__} --mode continuous  # Run continuously")
    print(f"  python {__file__} --mode single      # Run one cycle")
    print(f"  python {__file__} --mode discovery   # Discovery only")
    print(f"  python {__file__} --mode health      # Health check")

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']):
        cli_status()
    else:
        asyncio.run(main())