#!/usr/bin/env python3
"""
Simple CLI for Robust Research Execution Engine
==============================================

Easy-to-use command line interface for running robust AI research pipelines.

Usage:
    python3 run_robust_research.py --topic "Your Research Topic"
    python3 run_robust_research.py --topic "Neural Architecture Search" --max-experiments 10
    python3 run_robust_research.py --help

Author: AI Scientist v2 Autonomous System - Terragon Labs
License: MIT
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from robust_research_execution_engine import (
        RobustResearchExecutionEngine,
        RobustConfig,
        ResearchConfig
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Could not import robust execution engine: {e}")
    print("Please ensure robust_research_execution_engine.py is in the current directory.")
    ENGINE_AVAILABLE = False


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Robust Research Execution Engine - Enterprise-grade AI research automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic research execution
    python3 run_robust_research.py --topic "Deep Learning Optimization"
    
    # Advanced configuration
    python3 run_robust_research.py \\
        --topic "Federated Learning with Privacy" \\
        --output-dir "federated_research" \\
        --max-experiments 8 \\
        --timeout-hours 12 \\
        --max-cpu 70 \\
        --max-memory 6 \\
        --log-level DEBUG
    
    # Production mode
    python3 run_robust_research.py \\
        --topic "Production ML Pipeline" \\
        --production-mode \\
        --enable-all-features
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--topic", 
        required=True, 
        help="Research topic to investigate"
    )
    
    # Basic configuration
    parser.add_argument(
        "--output-dir", 
        default="robust_research_output", 
        help="Output directory for results (default: robust_research_output)"
    )
    
    parser.add_argument(
        "--max-experiments", 
        type=int, 
        default=5, 
        help="Maximum number of experiments to run (default: 5)"
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4o-2024-11-20", 
        help="AI model to use (default: gpt-4o-2024-11-20)"
    )
    
    # Robustness configuration
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum retry attempts for failed operations (default: 3)"
    )
    
    parser.add_argument(
        "--stage-timeout", 
        type=float, 
        default=60.0, 
        help="Timeout for individual stages in minutes (default: 60.0)"
    )
    
    parser.add_argument(
        "--timeout-hours", 
        type=float, 
        default=24.0, 
        help="Total execution timeout in hours (default: 24.0)"
    )
    
    # Resource limits
    parser.add_argument(
        "--max-cpu", 
        type=float, 
        default=80.0, 
        help="Maximum CPU usage percentage (default: 80.0)"
    )
    
    parser.add_argument(
        "--max-memory", 
        type=float, 
        default=8.0, 
        help="Maximum memory usage in GB (default: 8.0)"
    )
    
    parser.add_argument(
        "--max-disk", 
        type=float, 
        default=50.0, 
        help="Maximum disk usage in GB (default: 50.0)"
    )
    
    parser.add_argument(
        "--max-gpu-memory", 
        type=float, 
        default=12.0, 
        help="Maximum GPU memory in GB (default: 12.0)"
    )
    
    # Feature toggles
    parser.add_argument(
        "--disable-circuit-breaker", 
        action="store_true", 
        help="Disable circuit breaker protection"
    )
    
    parser.add_argument(
        "--disable-checkpoints", 
        action="store_true", 
        help="Disable checkpoint system"
    )
    
    parser.add_argument(
        "--disable-backups", 
        action="store_true", 
        help="Disable backup system"
    )
    
    parser.add_argument(
        "--disable-monitoring", 
        action="store_true", 
        help="Disable resource monitoring"
    )
    
    parser.add_argument(
        "--disable-input-validation", 
        action="store_true", 
        help="Disable input validation"
    )
    
    parser.add_argument(
        "--enable-sandbox", 
        action="store_true", 
        help="Enable sandbox mode for enhanced security"
    )
    
    # Monitoring configuration
    parser.add_argument(
        "--health-check-interval", 
        type=float, 
        default=30.0, 
        help="Health check interval in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--metrics-interval", 
        type=float, 
        default=10.0, 
        help="Metrics collection interval in seconds (default: 10.0)"
    )
    
    # API configuration
    parser.add_argument(
        "--api-requests-per-minute", 
        type=int, 
        default=60, 
        help="API requests per minute limit (default: 60)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO", 
        help="Logging level (default: INFO)"
    )
    
    # Quality gates
    parser.add_argument(
        "--min-success-rate", 
        type=float, 
        default=0.8, 
        help="Minimum success rate for quality gate (default: 0.8)"
    )
    
    parser.add_argument(
        "--max-error-rate", 
        type=float, 
        default=0.2, 
        help="Maximum error rate for quality gate (default: 0.2)"
    )
    
    # Presets
    parser.add_argument(
        "--production-mode", 
        action="store_true", 
        help="Enable production mode with enhanced reliability settings"
    )
    
    parser.add_argument(
        "--development-mode", 
        action="store_true", 
        help="Enable development mode with faster execution and more logging"
    )
    
    parser.add_argument(
        "--enable-all-features", 
        action="store_true", 
        help="Enable all robustness features"
    )
    
    # Utilities
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show configuration without executing"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="Robust Research Execution Engine v2.0"
    )
    
    return parser


def apply_preset_configurations(args):
    """Apply preset configurations based on mode flags."""
    if args.production_mode:
        # Production mode: Maximum reliability
        args.max_retries = 5
        args.stage_timeout = 120.0
        args.timeout_hours = 48.0
        args.max_cpu = 70.0
        args.max_memory = 6.0
        args.health_check_interval = 15.0
        args.metrics_interval = 5.0
        args.min_success_rate = 0.95
        args.max_error_rate = 0.05
        args.log_level = "INFO"
        args.enable_sandbox = True
        print("🏭 Production mode enabled: Enhanced reliability settings activated")
    
    elif args.development_mode:
        # Development mode: Fast execution with detailed logging
        args.max_retries = 2
        args.stage_timeout = 30.0
        args.timeout_hours = 6.0
        args.health_check_interval = 60.0
        args.metrics_interval = 30.0
        args.log_level = "DEBUG"
        print("🔧 Development mode enabled: Fast execution with detailed logging")
    
    if args.enable_all_features:
        # Enable all robustness features
        args.disable_circuit_breaker = False
        args.disable_checkpoints = False
        args.disable_backups = False
        args.disable_monitoring = False
        args.disable_input_validation = False
        args.enable_sandbox = True
        print("🌟 All robustness features enabled")


def create_configurations(args):
    """Create research and robust configurations from arguments."""
    # Create research configuration
    research_config = ResearchConfig(
        research_topic=args.topic,
        output_dir=args.output_dir,
        max_experiments=args.max_experiments,
        model_name=args.model,
        timeout_hours=args.timeout_hours
    )
    
    # Create robust configuration
    robust_config = RobustConfig(
        research_config=research_config,
        max_retries=args.max_retries,
        circuit_breaker_enabled=not args.disable_circuit_breaker,
        checkpoint_enabled=not args.disable_checkpoints,
        backup_enabled=not args.disable_backups,
        stage_timeout_minutes=args.stage_timeout,
        total_timeout_hours=args.timeout_hours,
        max_cpu_percent=args.max_cpu,
        max_memory_gb=args.max_memory,
        max_disk_gb=args.max_disk,
        max_gpu_memory_gb=args.max_gpu_memory,
        api_requests_per_minute=args.api_requests_per_minute,
        health_check_interval_seconds=args.health_check_interval,
        metrics_collection_interval_seconds=args.metrics_interval,
        log_level=args.log_level,
        enable_input_validation=not args.disable_input_validation,
        sandbox_mode=args.enable_sandbox,
        min_success_rate=args.min_success_rate,
        max_error_rate=args.max_error_rate
    )
    
    return research_config, robust_config


def print_configuration_summary(research_config, robust_config):
    """Print a summary of the configuration."""
    print("\n" + "="*70)
    print("🚀 ROBUST RESEARCH EXECUTION ENGINE - CONFIGURATION SUMMARY")
    print("="*70)
    
    print(f"📋 Research Configuration:")
    print(f"   Topic: {research_config.research_topic}")
    print(f"   Output Directory: {research_config.output_dir}")
    print(f"   Max Experiments: {research_config.max_experiments}")
    print(f"   Model: {research_config.model_name}")
    print(f"   Timeout: {research_config.timeout_hours} hours")
    
    print(f"\n🛡️ Robustness Configuration:")
    print(f"   Max Retries: {robust_config.max_retries}")
    print(f"   Circuit Breaker: {'✅ Enabled' if robust_config.circuit_breaker_enabled else '❌ Disabled'}")
    print(f"   Checkpoints: {'✅ Enabled' if robust_config.checkpoint_enabled else '❌ Disabled'}")
    print(f"   Backups: {'✅ Enabled' if robust_config.backup_enabled else '❌ Disabled'}")
    print(f"   Input Validation: {'✅ Enabled' if robust_config.enable_input_validation else '❌ Disabled'}")
    print(f"   Sandbox Mode: {'✅ Enabled' if robust_config.sandbox_mode else '❌ Disabled'}")
    
    print(f"\n📊 Resource Limits:")
    print(f"   Max CPU: {robust_config.max_cpu_percent}%")
    print(f"   Max Memory: {robust_config.max_memory_gb} GB")
    print(f"   Max Disk: {robust_config.max_disk_gb} GB")
    print(f"   Max GPU Memory: {robust_config.max_gpu_memory_gb} GB")
    
    print(f"\n⏱️ Timeouts and Intervals:")
    print(f"   Stage Timeout: {robust_config.stage_timeout_minutes} minutes")
    print(f"   Total Timeout: {robust_config.total_timeout_hours} hours")
    print(f"   Health Check Interval: {robust_config.health_check_interval_seconds} seconds")
    print(f"   Metrics Interval: {robust_config.metrics_collection_interval_seconds} seconds")
    
    print(f"\n🎯 Quality Gates:")
    print(f"   Min Success Rate: {robust_config.min_success_rate:.1%}")
    print(f"   Max Error Rate: {robust_config.max_error_rate:.1%}")
    print(f"   Log Level: {robust_config.log_level}")
    
    print("="*70)


async def run_robust_research(research_config, robust_config):
    """Execute the robust research pipeline."""
    print(f"\n🚀 Starting robust research execution...")
    print(f"📝 Research Topic: {research_config.research_topic}")
    
    # Create execution engine
    engine = RobustResearchExecutionEngine(robust_config)
    
    try:
        # Execute research pipeline
        results = await engine.execute_research_pipeline()
        
        # Print results summary
        print_results_summary(results)
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n🛑 Execution interrupted by user")
        return None
    except Exception as e:
        print(f"\n💥 Execution failed: {e}")
        return None


def print_results_summary(results):
    """Print a comprehensive results summary."""
    print(f"\n" + "="*70)
    print(f"🎉 EXECUTION COMPLETED")
    print(f"="*70)
    
    print(f"📊 Execution Summary:")
    print(f"   Status: {results.get('status', 'Unknown')}")
    print(f"   Execution ID: {results.get('execution_id', 'N/A')}")
    
    if 'execution_time_hours' in results:
        print(f"   Duration: {results['execution_time_hours']:.2f} hours")
    
    # Stage results
    stages = results.get('stages', {})
    if stages:
        completed_stages = len([s for s in stages.values() if s.get('status') == 'completed'])
        total_stages = len(stages)
        print(f"   Stages: {completed_stages}/{total_stages} completed")
        
        for stage_name, stage_data in stages.items():
            status = stage_data.get('status', 'unknown')
            status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⚠️"
            print(f"     {status_icon} {stage_name.title()}: {status}")
    
    # Error and recovery information
    errors = results.get('errors', [])
    recoveries = results.get('recovery_info', [])
    
    if errors or recoveries:
        print(f"\n🔧 Reliability Summary:")
        print(f"   Errors Handled: {len(errors)}")
        print(f"   Recovery Operations: {len(recoveries)}")
        
        if recoveries:
            successful_recoveries = len([r for r in recoveries if r.get('success')])
            print(f"   Successful Recoveries: {successful_recoveries}/{len(recoveries)}")
    
    # System metrics
    system_metrics = results.get('system_metrics', {})
    if system_metrics:
        health_status = system_metrics.get('health_status', {})
        resource_usage = system_metrics.get('resource_usage', {})
        
        print(f"\n📈 System Metrics:")
        if health_status:
            print(f"   Final Health Score: {health_status.get('overall_health_score', 'N/A')}/100")
            print(f"   System State: {health_status.get('system_state', 'Unknown')}")
        
        if resource_usage:
            if 'cpu_percent' in resource_usage:
                print(f"   Peak CPU: {resource_usage['cpu_percent']:.1f}%")
            if 'memory_percent' in resource_usage:
                print(f"   Peak Memory: {resource_usage['memory_percent']:.1f}%")
            if 'gpu_utilization' in resource_usage:
                print(f"   GPU Utilization: {resource_usage['gpu_utilization']:.1f}%")
        
        resource_alerts = system_metrics.get('resource_alerts', [])
        if resource_alerts:
            print(f"   Resource Alerts: {len(resource_alerts)}")
    
    # Output information
    if 'stages' in results and 'reporting' in results['stages']:
        reporting = results['stages']['reporting']
        if 'report_file' in reporting:
            print(f"\n📄 Output Files:")
            print(f"   Report: {reporting['report_file']}")
    
    print(f"="*70)


async def main():
    """Main function."""
    if not ENGINE_AVAILABLE:
        sys.exit(1)
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Apply preset configurations
    apply_preset_configurations(args)
    
    # Create configurations
    research_config, robust_config = create_configurations(args)
    
    # Print configuration summary
    if not args.dry_run:
        print_configuration_summary(research_config, robust_config)
        
        # Ask for confirmation in production mode
        if args.production_mode:
            response = input("\n🏭 Production mode enabled. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return
    
    # Dry run - just show configuration
    if args.dry_run:
        print("\n🔍 DRY RUN - Configuration Preview:")
        print_configuration_summary(research_config, robust_config)
        print("\n✅ Configuration validated. Use without --dry-run to execute.")
        return
    
    # Execute research
    results = await run_robust_research(research_config, robust_config)
    
    if results:
        if results.get('status') in ['completed', 'partial_failure']:
            print(f"\n✅ Research execution completed successfully!")
            print(f"📁 Output directory: {args.output_dir}")
            sys.exit(0)
        else:
            print(f"\n❌ Research execution failed")
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)