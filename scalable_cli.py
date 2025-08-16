#!/usr/bin/env python3
"""
Scalable CLI - High-Performance Command Line Interface
=====================================================

Generation 3: MAKE IT SCALE
Advanced CLI with parallel processing, intelligent caching,
auto-scaling, and performance optimization for autonomous AI research.

Usage:
    python scalable_cli.py --topic "Your Research Topic" --parallel --cache
    python scalable_cli.py --config config.json --max-concurrent 10 --auto-scale
    python scalable_cli.py --interactive --optimize-performance

Author: AI Scientist v2 Autonomous System - Terragon Labs
License: MIT
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from ai_scientist.scalable_execution_engine import (
    ScalableExecutionEngine,
    PerformanceConfig,
    ScalingConfig,
    CacheConfig,
    ScalingStrategy,
    CacheStrategy
)
from ai_scientist.robust_execution_engine import (
    SecurityPolicy,
    SecurityLevel,
    ResourceLimits,
    RetryPolicy
)
from ai_scientist.unified_autonomous_executor import ResearchConfig


def create_scalable_parser() -> argparse.ArgumentParser:
    """Create high-performance command line argument parser."""
    parser = argparse.ArgumentParser(
        description="⚡ Scalable Autonomous AI Research Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚡ Performance & Scalability Features:
  - Parallel experiment execution with auto-scaling
  - Intelligent multi-layer caching system
  - Real-time performance monitoring and optimization
  - Concurrent stage processing
  - Memory and resource optimization
  - Load balancing and fault tolerance

Examples:
  # High-performance parallel execution
  python scalable_cli.py --topic "ML Optimization" --parallel --max-concurrent 8
  
  # Auto-scaling with intelligent caching
  python scalable_cli.py --topic "AI Performance" --auto-scale --cache hybrid
  
  # Interactive performance optimization
  python scalable_cli.py --interactive --performance-monitoring
  
  # Resource-optimized execution
  python scalable_cli.py --config scalable_config.json --optimize-memory --gpu
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
        default="scalable_research_output",
        help="Output directory for research results (default: scalable_research_output)"
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
    
    # Performance options
    perf_group = parser.add_argument_group("⚡ Performance Options")
    perf_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel stage execution"
    )
    
    perf_group.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent experiments (default: 10)"
    )
    
    perf_group.add_argument(
        "--performance-monitoring",
        action="store_true",
        default=True,
        help="Enable real-time performance monitoring (default: True)"
    )
    
    perf_group.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Enable memory optimization"
    )
    
    perf_group.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (if available)"
    )
    
    # Caching options
    cache_group = parser.add_argument_group("📋 Caching Options")
    cache_group.add_argument(
        "--cache",
        type=str,
        choices=["memory", "disk", "distributed", "hybrid"],
        default="hybrid",
        help="Caching strategy (default: hybrid)"
    )
    
    cache_group.add_argument(
        "--cache-size",
        type=int,
        default=1024,
        help="Cache size in MB (default: 1024)"
    )
    
    cache_group.add_argument(
        "--cache-ttl",
        type=float,
        default=3600.0,
        help="Cache TTL in seconds (default: 3600)"
    )
    
    cache_group.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching completely"
    )
    
    # Auto-scaling options
    scaling_group = parser.add_argument_group("📈 Auto-Scaling Options")
    scaling_group.add_argument(
        "--auto-scale",
        action="store_true",
        help="Enable auto-scaling"
    )
    
    scaling_group.add_argument(
        "--scaling-strategy",
        type=str,
        choices=["conservative", "aggressive", "adaptive", "predictive"],
        default="adaptive",
        help="Scaling strategy (default: adaptive)"
    )
    
    scaling_group.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum worker count (default: 1)"
    )
    
    scaling_group.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker count (default: CPU count)"
    )
    
    scaling_group.add_argument(
        "--scale-threshold",
        type=float,
        default=0.8,
        help="Scale up threshold (default: 0.8)"
    )
    
    # Security and robustness options (inherited)
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
        help="Enable sandbox mode"
    )
    
    # Resource monitoring (inherited)
    resource_group = parser.add_argument_group("📊 Resource Monitoring")
    resource_group.add_argument(
        "--monitor-resources",
        action="store_true",
        default=True,
        help="Enable resource monitoring (default: True)"
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
    
    # Execution options
    exec_group = parser.add_argument_group("⚙️ Execution Options")
    exec_group.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="Execution timeout in seconds (default: 7200)"
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
        default=10,
        help="Maximum number of experiments to run (default: 10)"
    )
    
    return parser


def interactive_scalable_config() -> tuple:
    """Create scalable configuration through interactive prompts."""
    print("⚡ Scalable Autonomous AI Research System - Interactive Setup")
    print("=" * 65)
    
    # Basic research config
    topic = input("Enter research topic: ").strip()
    if not topic:
        print("Research topic is required!")
        sys.exit(1)
    
    output_dir = input("Output directory (default: scalable_research_output): ").strip()
    if not output_dir:
        output_dir = "scalable_research_output"
    
    # Performance configuration
    print("\n⚡ Performance Configuration:")
    enable_parallel = input("Enable parallel stage execution? (Y/n): ").strip().lower() != 'n'
    
    try:
        max_concurrent = int(input("Maximum concurrent experiments (default: 10): ") or "10")
    except ValueError:
        max_concurrent = 10
    
    performance_monitoring = input("Enable performance monitoring? (Y/n): ").strip().lower() != 'n'
    optimize_memory = input("Enable memory optimization? (y/N): ").strip().lower() == 'y'
    gpu_acceleration = input("Enable GPU acceleration? (y/N): ").strip().lower() == 'y'
    
    # Caching configuration
    print("\n📋 Caching Configuration:")
    enable_caching = input("Enable intelligent caching? (Y/n): ").strip().lower() != 'n'
    
    cache_strategy = "hybrid"
    cache_size = 1024
    if enable_caching:
        cache_strategy = input("Cache strategy (memory/disk/distributed/hybrid) [hybrid]: ").strip().lower()
        if cache_strategy not in ["memory", "disk", "distributed", "hybrid"]:
            cache_strategy = "hybrid"
        
        try:
            cache_size = int(input("Cache size MB (default: 1024): ") or "1024")
        except ValueError:
            cache_size = 1024
    
    # Auto-scaling configuration
    print("\n📈 Auto-Scaling Configuration:")
    enable_auto_scaling = input("Enable auto-scaling? (Y/n): ").strip().lower() != 'n'
    
    scaling_strategy = "adaptive"
    min_workers = 1
    max_workers = None
    if enable_auto_scaling:
        scaling_strategy = input("Scaling strategy (conservative/aggressive/adaptive/predictive) [adaptive]: ").strip().lower()
        if scaling_strategy not in ["conservative", "aggressive", "adaptive", "predictive"]:
            scaling_strategy = "adaptive"
        
        try:
            min_workers = int(input("Minimum workers (default: 1): ") or "1")
            max_workers_input = input("Maximum workers (default: auto): ").strip()
            max_workers = int(max_workers_input) if max_workers_input else None
        except ValueError:
            min_workers = 1
            max_workers = None
    
    # Security configuration
    print("\n🔒 Security Configuration:")
    security_level = input("Security level (low/medium/high/critical) [medium]: ").strip().lower()
    if security_level not in ["low", "medium", "high", "critical"]:
        security_level = "medium"
    
    sandbox_mode = input("Enable sandbox mode? (y/N): ").strip().lower() == 'y'
    
    # Resource limits
    print("\n📊 Resource Configuration:")
    try:
        max_cpu = float(input("Maximum CPU usage % (default: 80.0): ") or "80.0")
        max_memory = int(input("Maximum memory MB (default: 4096): ") or "4096")
        timeout = float(input("Execution timeout seconds (default: 7200): ") or "7200")
        max_experiments = int(input("Maximum experiments (default: 10): ") or "10")
    except ValueError:
        max_cpu = 80.0
        max_memory = 4096
        timeout = 7200.0
        max_experiments = 10
    
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
    
    performance_config = PerformanceConfig(
        max_concurrent_experiments=max_concurrent,
        enable_parallel_stages=enable_parallel,
        enable_caching=enable_caching,
        cache_strategy=getattr(CacheStrategy, cache_strategy.upper()),
        cache_size_mb=cache_size,
        enable_auto_scaling=enable_auto_scaling,
        scaling_strategy=getattr(ScalingStrategy, scaling_strategy.upper()),
        performance_monitoring=performance_monitoring,
        optimize_memory=optimize_memory,
        enable_gpu_acceleration=gpu_acceleration
    )
    
    scaling_config = ScalingConfig(
        min_workers=min_workers,
        max_workers=max_workers or mp.cpu_count(),
        scale_up_threshold=0.8
    ) if enable_auto_scaling else None
    
    cache_config = CacheConfig(
        memory_cache_size=100,
        cache_ttl=3600.0,
        enable_compression=True
    ) if enable_caching else None
    
    security_policy = SecurityPolicy(
        level=getattr(SecurityLevel, security_level.upper()),
        sandbox_mode=sandbox_mode,
        max_execution_time=timeout
    )
    
    resource_limits = ResourceLimits(
        max_cpu_percent=max_cpu,
        max_memory_mb=max_memory
    )
    
    return research_config, performance_config, scaling_config, cache_config, security_policy, resource_limits


def create_config_from_args(args) -> tuple:
    """Create configuration from command line arguments."""
    if not args.topic:
        print("Research topic is required when not using config file or interactive mode!")
        sys.exit(1)
    
    import multiprocessing as mp
    
    research_config = ResearchConfig(
        research_topic=args.topic,
        output_dir=args.output,
        max_experiments=args.max_experiments,
        model_name=args.model,
        timeout_hours=args.timeout / 3600.0
    )
    
    performance_config = PerformanceConfig(
        max_concurrent_experiments=args.max_concurrent,
        enable_parallel_stages=args.parallel,
        enable_caching=not args.disable_cache,
        cache_strategy=getattr(CacheStrategy, args.cache.upper()),
        cache_size_mb=args.cache_size,
        enable_auto_scaling=args.auto_scale,
        scaling_strategy=getattr(ScalingStrategy, args.scaling_strategy.upper()),
        performance_monitoring=args.performance_monitoring,
        optimize_memory=args.optimize_memory,
        enable_gpu_acceleration=args.gpu
    )
    
    scaling_config = ScalingConfig(
        min_workers=args.min_workers,
        max_workers=args.max_workers or mp.cpu_count(),
        scale_up_threshold=args.scale_threshold
    ) if args.auto_scale else None
    
    cache_config = CacheConfig(
        memory_cache_size=100,
        cache_ttl=args.cache_ttl,
        enable_compression=True
    ) if not args.disable_cache else None
    
    security_policy = SecurityPolicy(
        level=getattr(SecurityLevel, args.security.upper()),
        sandbox_mode=args.sandbox,
        max_execution_time=args.timeout
    )
    
    resource_limits = ResourceLimits(
        max_cpu_percent=args.max_cpu,
        max_memory_mb=args.max_memory
    ) if args.monitor_resources else None
    
    return research_config, performance_config, scaling_config, cache_config, security_policy, resource_limits


def print_scalable_banner():
    """Print scalable startup banner."""
    print("""
⚡ AI Scientist v2 - Scalable Autonomous Research System
=======================================================
🔄 Parallel experiment execution with auto-scaling
📋 Intelligent multi-layer caching system
📊 Real-time performance monitoring and optimization
🧠 Concurrent stage processing and load balancing
⚡ High-performance resource optimization
🚀 Enterprise-grade scalability and fault tolerance

Powered by Terragon Labs - Generation 3: MAKE IT SCALE
""")


def print_scalable_config_summary(research_config, performance_config, scaling_config, 
                                cache_config, security_policy, resource_limits):
    """Print scalable configuration summary."""
    print(f"\n🎯 Research Configuration:")
    print(f"   Topic: {research_config.research_topic}")
    print(f"   Output: {research_config.output_dir}")
    print(f"   Model: {research_config.model_name}")
    print(f"   Max Experiments: {research_config.max_experiments}")
    print(f"   Timeout: {research_config.timeout_hours} hours")
    
    print(f"\n⚡ Performance Configuration:")
    print(f"   Max Concurrent: {performance_config.max_concurrent_experiments}")
    print(f"   Parallel Stages: {performance_config.enable_parallel_stages}")
    print(f"   Performance Monitoring: {performance_config.performance_monitoring}")
    print(f"   Memory Optimization: {performance_config.optimize_memory}")
    print(f"   GPU Acceleration: {performance_config.enable_gpu_acceleration}")
    
    if performance_config.enable_caching and cache_config:
        print(f"\n📋 Caching Configuration:")
        print(f"   Strategy: {performance_config.cache_strategy.value}")
        print(f"   Size: {performance_config.cache_size_mb} MB")
        print(f"   TTL: {cache_config.cache_ttl} seconds")
        print(f"   Compression: {cache_config.enable_compression}")
    
    if performance_config.enable_auto_scaling and scaling_config:
        print(f"\n📈 Auto-Scaling Configuration:")
        print(f"   Strategy: {performance_config.scaling_strategy.value}")
        print(f"   Worker Range: {scaling_config.min_workers} - {scaling_config.max_workers}")
        print(f"   Scale Threshold: {scaling_config.scale_up_threshold}")
    
    print(f"\n🔒 Security Policy:")
    print(f"   Level: {security_policy.level.value}")
    print(f"   Sandbox Mode: {security_policy.sandbox_mode}")
    
    if resource_limits:
        print(f"\n📊 Resource Limits:")
        print(f"   Max CPU: {resource_limits.max_cpu_percent}%")
        print(f"   Max Memory: {resource_limits.max_memory_mb} MB")


async def main():
    """Main scalable CLI execution function."""
    parser = create_scalable_parser()
    args = parser.parse_args()
    
    print_scalable_banner()
    
    # Determine configuration source
    if args.config:
        print(f"📁 Loading configuration from: {args.config}")
        print("⚠️ Config file loading not implemented in this demo")
        sys.exit(1)
    elif args.interactive:
        configs = interactive_scalable_config()
        research_config, performance_config, scaling_config, cache_config, security_policy, resource_limits = configs
    else:
        configs = create_config_from_args(args)
        research_config, performance_config, scaling_config, cache_config, security_policy, resource_limits = configs
    
    # Display configuration
    print_scalable_config_summary(research_config, performance_config, scaling_config, 
                                cache_config, security_policy, resource_limits)
    
    # Confirm execution
    if args.interactive:
        confirm = input("\n🚀 Proceed with scalable autonomous research execution? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Execution cancelled.")
            return
    
    print(f"\n⚡ Starting scalable autonomous research execution...")
    print(f"⏰ Maximum execution time: {research_config.timeout_hours} hours")
    print(f"🔄 Max concurrent experiments: {performance_config.max_concurrent_experiments}")
    
    if performance_config.enable_auto_scaling:
        print(f"📈 Auto-scaling: {performance_config.scaling_strategy.value}")
    
    if performance_config.enable_caching:
        print(f"📋 Caching: {performance_config.cache_strategy.value}")
    
    try:
        # Initialize scalable execution engine
        engine = ScalableExecutionEngine(
            research_config,
            security_policy,
            resource_limits,
            None,  # retry_policy
            performance_config,
            scaling_config,
            cache_config
        )
        await engine.initialize_components()
        
        print("🔄 Executing scalable research pipeline...")
        results = await engine.execute_research_pipeline()
        
        # Display results
        print(f"\n✨ Scalable research execution completed!")
        print(f"📊 Status: {results['status']}")
        
        if results['status'] in ['completed', 'partial_failure']:
            print(f"⏱️ Execution time: {results.get('execution_time_hours', 0):.3f} hours")
            print(f"📁 Results saved to: {research_config.output_dir}")
            
            # Performance metrics
            perf_metrics = results.get('performance_metrics', {})
            if perf_metrics:
                print(f"📊 Avg CPU usage: {perf_metrics.get('avg_cpu_usage', 0):.1f}%")
                print(f"💾 Avg memory usage: {perf_metrics.get('avg_memory_usage', 0):.1f}%")
                print(f"🧵 Avg active threads: {perf_metrics.get('avg_active_threads', 0):.1f}")
            
            # Cache statistics
            cache_stats = results.get('cache_stats', {})
            if cache_stats:
                print(f"📋 Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
                print(f"🎯 Cache hits: {cache_stats.get('total_hits', 0)}")
                print(f"❌ Cache misses: {cache_stats.get('total_misses', 0)}")
            
            # Scaling information
            scaling_info = results.get('scaling_info', {})
            if scaling_info:
                print(f"👥 Workers used: {scaling_info.get('current_workers', 1)}")
                print(f"⚡ Max concurrent: {scaling_info.get('max_concurrent', 1)}")
            
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
                
                extra_info = ""
                if stage_data.get('parallel_execution'):
                    extra_info = f" (parallel: {stage_data.get('worker_count', 1)} workers)"
                elif stage_data.get('experiments_completed'):
                    extra_info = f" ({stage_data.get('experiments_completed')} experiments)"
                
                print(f"   {emoji} {stage_name.title()}: {status}{extra_info}")
        
        elif results['status'] == 'failed':
            print(f"❌ Execution failed: {results.get('error', 'Unknown error')}")
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