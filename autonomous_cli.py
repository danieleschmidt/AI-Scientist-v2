#!/usr/bin/env python3
"""
Autonomous CLI - Simple Command Line Interface
=============================================

Generation 1: MAKE IT WORK
Simple CLI interface for autonomous AI research execution.

Usage:
    python autonomous_cli.py --topic "Your Research Topic"
    python autonomous_cli.py --config config.json
    python autonomous_cli.py --interactive

Author: AI Scientist v2 Autonomous System - Terragon Labs
License: MIT
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from ai_scientist.unified_autonomous_executor import (
    UnifiedAutonomousExecutor,
    ResearchConfig
)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Autonomous AI Research Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autonomous_cli.py --topic "Deep Learning Optimization"
  python autonomous_cli.py --topic "Neural Architecture Search" --output research_output
  python autonomous_cli.py --config my_research_config.json
  python autonomous_cli.py --interactive
        """
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic for autonomous investigation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="autonomous_research_output",
        help="Output directory for research results (default: autonomous_research_output)"
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
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="LLM model to use (default: gpt-4o-2024-11-20)"
    )
    
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Maximum number of experiments to run (default: 5)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=24.0,
        help="Timeout in hours (default: 24.0)"
    )
    
    parser.add_argument(
        "--disable-novel-discovery",
        action="store_true",
        help="Disable novel algorithm discovery"
    )
    
    parser.add_argument(
        "--disable-autonomous-sdlc",
        action="store_true",
        help="Disable autonomous SDLC orchestration"
    )
    
    return parser


def load_config_from_file(config_path: str) -> ResearchConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return ResearchConfig(**config_data)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def interactive_config() -> ResearchConfig:
    """Create configuration through interactive prompts."""
    print("🧠 Autonomous AI Research System - Interactive Setup")
    print("=" * 50)
    
    topic = input("Enter research topic: ").strip()
    if not topic:
        print("Research topic is required!")
        sys.exit(1)
    
    output_dir = input("Output directory (default: autonomous_research_output): ").strip()
    if not output_dir:
        output_dir = "autonomous_research_output"
    
    try:
        max_experiments = int(input("Maximum experiments (default: 5): ") or "5")
    except ValueError:
        max_experiments = 5
    
    model = input("LLM model (default: gpt-4o-2024-11-20): ").strip()
    if not model:
        model = "gpt-4o-2024-11-20"
    
    try:
        timeout = float(input("Timeout in hours (default: 24.0): ") or "24.0")
    except ValueError:
        timeout = 24.0
    
    enable_novel = input("Enable novel algorithm discovery? (y/N): ").strip().lower() == 'y'
    enable_sdlc = input("Enable autonomous SDLC? (y/N): ").strip().lower() == 'y'
    
    return ResearchConfig(
        research_topic=topic,
        output_dir=output_dir,
        max_experiments=max_experiments,
        model_name=model,
        enable_novel_discovery=enable_novel,
        enable_autonomous_sdlc=enable_sdlc,
        timeout_hours=timeout
    )


def create_config_from_args(args) -> ResearchConfig:
    """Create configuration from command line arguments."""
    if not args.topic:
        print("Research topic is required when not using config file or interactive mode!")
        sys.exit(1)
    
    return ResearchConfig(
        research_topic=args.topic,
        output_dir=args.output,
        max_experiments=args.max_experiments,
        model_name=args.model,
        enable_novel_discovery=not args.disable_novel_discovery,
        enable_autonomous_sdlc=not args.disable_autonomous_sdlc,
        timeout_hours=args.timeout
    )


def print_banner():
    """Print startup banner."""
    print("""
🧠 AI Scientist v2 - Autonomous Research System
================================================
🔬 Autonomous experimentation and discovery
⚗️ Novel algorithm development  
📊 Statistical validation and analysis
📝 Publication-ready research reports

Powered by Terragon Labs
""")


async def main():
    """Main CLI execution function."""
    parser = create_parser()
    args = parser.parse_args()
    
    print_banner()
    
    # Determine configuration source
    if args.config:
        print(f"📁 Loading configuration from: {args.config}")
        config = load_config_from_file(args.config)
    elif args.interactive:
        config = interactive_config()
    else:
        config = create_config_from_args(args)
    
    # Display configuration
    print(f"\n🎯 Research Configuration:")
    print(f"   Topic: {config.research_topic}")
    print(f"   Output: {config.output_dir}")
    print(f"   Model: {config.model_name}")
    print(f"   Max Experiments: {config.max_experiments}")
    print(f"   Timeout: {config.timeout_hours} hours")
    print(f"   Novel Discovery: {config.enable_novel_discovery}")
    print(f"   Autonomous SDLC: {config.enable_autonomous_sdlc}")
    
    # Confirm execution
    if args.interactive:
        confirm = input("\n🚀 Proceed with autonomous research execution? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Execution cancelled.")
            return
    
    print(f"\n🚀 Starting autonomous research execution...")
    print(f"⏰ Maximum execution time: {config.timeout_hours} hours")
    
    try:
        # Initialize and execute
        executor = UnifiedAutonomousExecutor(config)
        await executor.initialize_components()
        
        print("🔄 Executing research pipeline...")
        results = await executor.execute_research_pipeline()
        
        # Display results
        print(f"\n✨ Research execution completed!")
        print(f"📊 Status: {results['status']}")
        
        if results['status'] == 'completed':
            print(f"⏱️ Execution time: {results.get('execution_time_hours', 0):.2f} hours")
            print(f"📁 Results saved to: {config.output_dir}")
            
            # Display stage summaries
            stages = results.get('stages', {})
            for stage_name, stage_data in stages.items():
                status = stage_data.get('status', 'unknown')
                emoji = "✅" if status == "completed" else "❌"
                print(f"   {emoji} {stage_name.title()}: {status}")
        
        elif results['status'] == 'failed':
            print(f"❌ Execution failed: {results.get('error', 'Unknown error')}")
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)