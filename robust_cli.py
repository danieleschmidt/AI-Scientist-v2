#!/usr/bin/env python3
"""
Robust CLI - Enhanced Command Line Interface
===========================================

Generation 2: MAKE IT ROBUST
Enhanced CLI with comprehensive error handling, security validation,
and resource monitoring for autonomous AI research execution.

Usage:
    python robust_cli.py --topic "Your Research Topic" --security high
    python robust_cli.py --config config.json --monitor-resources
    python robust_cli.py --interactive --sandbox

Author: AI Scientist v2 Autonomous System - Terragon Labs
License: MIT
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from ai_scientist.robust_execution_engine import (
    RobustExecutionEngine,
    SecurityPolicy,
    SecurityLevel,
    ResourceLimits,
    RetryPolicy
)
from ai_scientist.unified_autonomous_executor import ResearchConfig


def create_robust_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🛡️ Robust Autonomous AI Research Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🛡️ Security & Robustness Features:
  - Input validation and sanitization
  - Resource monitoring and limits
  - Circuit breaker pattern for fault tolerance
  - Comprehensive error handling and recovery
  - Security policy enforcement
  - Audit logging and compliance tracking

Examples:
  # High security research execution
  python robust_cli.py --topic "ML Security" --security high --sandbox
  
  # Resource-constrained execution
  python robust_cli.py --topic "Edge AI" --max-cpu 50 --max-memory 1024
  
  # Interactive robust setup
  python robust_cli.py --interactive --monitor-resources
  
  # Configuration with retry policies
  python robust_cli.py --config robust_config.json --max-retries 5
        """
    )
    
    # Basic research parameters
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic for autonomous investigation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="robust_research_output",
        help="Output directory for research results (default: robust_research_output)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="JSON configuration file path"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Security options
    security_group = parser.add_argument_group("🔒 Security Options")
    security_group.add_argument(
        "--security",
        type=str,
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Security level (default: medium)"
    )
    
    security_group.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable sandbox mode for safer execution"
    )
    
    security_group.add_argument(
        "--validate-inputs",
        action="store_true",
        default=True,
        help="Enable input validation (default: True)"
    )
    
    security_group.add_argument(
        "--max-file-size",
        type=int,
        default=100,
        help="Maximum file size in MB (default: 100)"
    )
    
    # Resource monitoring options
    resource_group = parser.add_argument_group("📊 Resource Monitoring")
    resource_group.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Enable resource monitoring"
    )
    
    resource_group.add_argument(
        "--max-cpu",
        type=float,
        default=80.0,
        help="Maximum CPU usage percentage (default: 80.0)"
    )
    
    resource_group.add_argument(
        "--max-memory",
        type=int,
        default=4096,
        help="Maximum memory usage in MB (default: 4096)"
    )
    
    resource_group.add_argument(
        "--max-disk",
        type=int,
        default=10240,
        help="Maximum disk usage in MB (default: 10240)"
    )
    
    resource_group.add_argument(
        "--monitor-interval",
        type=float,
        default=30.0,
        help="Resource monitoring interval in seconds (default: 30.0)"
    )
    
    # Retry and fault tolerance options
    retry_group = parser.add_argument_group("🔄 Retry & Fault Tolerance")
    retry_group.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)"
    )
    
    retry_group.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Base retry delay in seconds (default: 1.0)"
    )
    
    retry_group.add_argument(
        "--exponential-backoff",
        action="store_true",
        default=True,
        help="Enable exponential backoff for retries (default: True)"
    )
    
    # Execution options
    exec_group = parser.add_argument_group("⚙️ Execution Options")
    exec_group.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Execution timeout in seconds (default: 3600)"
    )
    
    exec_group.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="LLM model to use (default: gpt-4o-2024-11-20)"
    )
    
    exec_group.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Maximum number of experiments to run (default: 5)"
    )
    
    return parser


def interactive_robust_config() -> tuple:
    """Create robust configuration through interactive prompts."""
    print("🛡️ Robust Autonomous AI Research System - Interactive Setup")
    print("=" * 60)
    
    # Basic research config
    topic = input("Enter research topic: ").strip()
    if not topic:
        print("Research topic is required!")
        sys.exit(1)
    
    output_dir = input("Output directory (default: robust_research_output): ").strip()
    if not output_dir:
        output_dir = "robust_research_output"
    
    # Security configuration
    print("\n🔒 Security Configuration:")
    security_level = input("Security level (low/medium/high/critical) [medium]: ").strip().lower()
    if security_level not in ["low", "medium", "high", "critical"]:
        security_level = "medium"
    
    sandbox_mode = input("Enable sandbox mode? (y/N): ").strip().lower() == 'y'
    validate_inputs = input("Enable input validation? (Y/n): ").strip().lower() != 'n'
    
    # Resource limits
    print("\n📊 Resource Configuration:")
    monitor_resources = input("Enable resource monitoring? (Y/n): ").strip().lower() != 'n'
    
    max_cpu = 80.0
    max_memory = 4096
    if monitor_resources:
        try:
            max_cpu = float(input("Maximum CPU usage % (default: 80.0): ") or "80.0")
            max_memory = int(input("Maximum memory MB (default: 4096): ") or "4096")
        except ValueError:
            print("Using default resource limits")
    
    # Retry configuration
    print("\n🔄 Retry Configuration:")
    try:
        max_retries = int(input("Maximum retry attempts (default: 3): ") or "3")
        retry_delay = float(input("Base retry delay seconds (default: 1.0): ") or "1.0")
    except ValueError:
        max_retries = 3
        retry_delay = 1.0
    
    # Execution configuration
    print("\n⚙️ Execution Configuration:")
    try:
        timeout = float(input("Execution timeout seconds (default: 3600): ") or "3600")
        max_experiments = int(input("Maximum experiments (default: 5): ") or "5")
    except ValueError:
        timeout = 3600.0
        max_experiments = 5
    
    model = input("LLM model (default: gpt-4o-2024-11-20): ").strip()
    if not model:
        model = "gpt-4o-2024-11-20"
    
    # Create configurations
    research_config = ResearchConfig(
        research_topic=topic,
        output_dir=output_dir,
        max_experiments=max_experiments,
        model_name=model,
        timeout_hours=timeout / 3600.0
    )
    
    security_policy = SecurityPolicy(
        level=getattr(SecurityLevel, security_level.upper()),
        sandbox_mode=sandbox_mode,
        validate_inputs=validate_inputs,
        max_execution_time=timeout
    )
    
    resource_limits = ResourceLimits(
        max_cpu_percent=max_cpu,
        max_memory_mb=max_memory
    ) if monitor_resources else None
    
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        base_delay=retry_delay,
        exponential_backoff=True
    )
    
    return research_config, security_policy, resource_limits, retry_policy


def create_config_from_args(args) -> tuple:
    """Create configuration from command line arguments."""
    if not args.topic:
        print("Research topic is required when not using config file or interactive mode!")
        sys.exit(1)
    
    research_config = ResearchConfig(
        research_topic=args.topic,
        output_dir=args.output,
        max_experiments=args.max_experiments,
        model_name=args.model,
        timeout_hours=args.timeout / 3600.0
    )
    
    security_policy = SecurityPolicy(
        level=getattr(SecurityLevel, args.security.upper()),
        sandbox_mode=args.sandbox,
        validate_inputs=args.validate_inputs,
        max_file_size_mb=args.max_file_size,
        max_execution_time=args.timeout
    )
    
    resource_limits = ResourceLimits(
        max_cpu_percent=args.max_cpu,
        max_memory_mb=args.max_memory,
        max_disk_mb=args.max_disk,
        monitor_interval=args.monitor_interval
    ) if args.monitor_resources else None
    
    retry_policy = RetryPolicy(
        max_attempts=args.max_retries,
        base_delay=args.retry_delay,
        exponential_backoff=args.exponential_backoff
    )
    
    return research_config, security_policy, resource_limits, retry_policy


def print_robust_banner():
    """Print robust startup banner."""
    print("""
🛡️ AI Scientist v2 - Robust Autonomous Research System
======================================================
🔒 Enhanced security and input validation
📊 Resource monitoring and limits
🔄 Fault tolerance with circuit breakers
⚠️ Comprehensive error handling
🧪 Sandbox execution environment
📋 Audit logging and compliance

Powered by Terragon Labs - Generation 2: MAKE IT ROBUST
""")


def print_config_summary(research_config, security_policy, resource_limits, retry_policy):
    """Print configuration summary."""
    print(f"\n🎯 Research Configuration:")
    print(f"   Topic: {research_config.research_topic}")
    print(f"   Output: {research_config.output_dir}")
    print(f"   Model: {research_config.model_name}")
    print(f"   Max Experiments: {research_config.max_experiments}")
    print(f"   Timeout: {research_config.timeout_hours} hours")
    
    print(f"\n🔒 Security Policy:")
    print(f"   Level: {security_policy.level.value}")
    print(f"   Sandbox Mode: {security_policy.sandbox_mode}")
    print(f"   Input Validation: {security_policy.validate_inputs}")
    print(f"   Max File Size: {security_policy.max_file_size_mb} MB")
    
    if resource_limits:
        print(f"\n📊 Resource Limits:")
        print(f"   Max CPU: {resource_limits.max_cpu_percent}%")
        print(f"   Max Memory: {resource_limits.max_memory_mb} MB")
        print(f"   Max Disk: {resource_limits.max_disk_mb} MB")
        print(f"   Monitor Interval: {resource_limits.monitor_interval} seconds")
    
    print(f"\n🔄 Retry Policy:")
    print(f"   Max Attempts: {retry_policy.max_attempts}")
    print(f"   Base Delay: {retry_policy.base_delay} seconds")
    print(f"   Exponential Backoff: {retry_policy.exponential_backoff}")


async def main():
    """Main robust CLI execution function."""
    parser = create_robust_parser()
    args = parser.parse_args()
    
    print_robust_banner()
    
    # Determine configuration source
    if args.config:
        print(f"📁 Loading configuration from: {args.config}")
        # Would load from file - simplified for this demo
        print("⚠️ Config file loading not implemented in this demo")
        sys.exit(1)
    elif args.interactive:
        research_config, security_policy, resource_limits, retry_policy = interactive_robust_config()
    else:
        research_config, security_policy, resource_limits, retry_policy = create_config_from_args(args)
    
    # Display configuration
    print_config_summary(research_config, security_policy, resource_limits, retry_policy)
    
    # Confirm execution
    if args.interactive:
        confirm = input("\n🚀 Proceed with robust autonomous research execution? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Execution cancelled.")
            return
    
    print(f"\n🛡️ Starting robust autonomous research execution...")
    print(f"⏰ Maximum execution time: {research_config.timeout_hours} hours")
    print(f"🔒 Security level: {security_policy.level.value}")
    
    try:
        # Initialize robust execution engine
        engine = RobustExecutionEngine(
            research_config, 
            security_policy, 
            resource_limits, 
            retry_policy
        )
        await engine.initialize_components()
        
        print("🔄 Executing robust research pipeline...")
        results = await engine.execute_research_pipeline()
        
        # Display results
        print(f"\n✨ Robust research execution completed!")
        print(f"📊 Status: {results['status']}")
        
        if results['status'] in ['completed', 'partial_failure']:
            print(f"⏱️ Execution time: {results.get('execution_time_hours', 0):.3f} hours")
            print(f"📁 Results saved to: {research_config.output_dir}")
            print(f"🔒 Security checks: {len(results.get('security_checks', []))}")
            print(f"⚠️ Resource violations: {len(results.get('resource_violations', []))}")
            print(f"❌ Errors encountered: {len(results.get('errors', []))}")
            
            # Display stage summaries
            stages = results.get('stages', {})
            for stage_name, stage_data in stages.items():
                status = stage_data.get('status', 'unknown')
                if status == "completed":
                    emoji = "✅"
                elif status == "failed":
                    emoji = "❌"
                else:
                    emoji = "⚠️"
                print(f"   {emoji} {stage_name.title()}: {status}")
            
            # Show warnings for any issues
            if results.get('resource_violations'):
                print(f"\n⚠️ Resource violations detected. Check logs for details.")
            
            if results.get('errors'):
                print(f"⚠️ Errors occurred during execution. Check logs for details.")
        
        elif results['status'] == 'failed':
            print(f"❌ Execution failed: {results.get('error', 'Unknown error')}")
            print(f"🔍 Error type: {results.get('error_type', 'Unknown')}")
            return 1
        
        elif results['status'] == 'timeout':
            print(f"⏰ Execution timed out after {results.get('execution_time_hours', 0):.3f} hours")
            return 1
        
        elif results['status'] == 'aborted':
            print(f"🛑 Execution was aborted due to critical errors")
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)