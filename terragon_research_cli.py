#!/usr/bin/env python3
"""
TERRAGON RESEARCH CLI v1.0

Comprehensive command-line interface for autonomous research execution.
"""

import os
import sys
import asyncio
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from terragon_autonomous_research_engine import (
    AutonomousResearchEngine,
    ResearchConfig,
    load_research_config
)


class ResearchCLI:
    """Command-line interface for autonomous research operations."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser."""
        parser = argparse.ArgumentParser(
            description="Terragon Autonomous Research System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Quick start with example topic
  python terragon_research_cli.py run --quick-start

  # Run with custom topic
  python terragon_research_cli.py run --topic my_research.md
  
  # Advanced configuration
  python terragon_research_cli.py run --config advanced_config.yaml --max-ideas 10
  
  # Create research topic template
  python terragon_research_cli.py create-topic --output ai_research.md
  
  # View session results
  python terragon_research_cli.py status --session 20250824_143022
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Execute research pipeline')
        self._add_run_arguments(run_parser)
        
        # Create topic command
        topic_parser = subparsers.add_parser('create-topic', help='Create research topic template')
        self._add_topic_arguments(topic_parser)
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Check session status')
        self._add_status_arguments(status_parser)
        
        # Config command
        config_parser = subparsers.add_parser('create-config', help='Create configuration template')
        self._add_config_arguments(config_parser)
        
        return parser
    
    def _add_run_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for run command."""
        parser.add_argument('--quick-start', action='store_true',
                          help='Quick start with example configuration')
        parser.add_argument('--config', default='research_config.yaml',
                          help='Configuration file path')
        parser.add_argument('--topic', help='Research topic description file')
        parser.add_argument('--output-dir', help='Output directory for results')
        
        # Model configurations
        parser.add_argument('--ideation-model', help='Model for ideation phase')
        parser.add_argument('--experiment-model', help='Model for experimentation')
        parser.add_argument('--writeup-model', help='Model for paper writing')
        parser.add_argument('--review-model', help='Model for paper review')
        
        # Execution parameters
        parser.add_argument('--max-ideas', type=int, help='Maximum number of ideas to generate')
        parser.add_argument('--writeup-type', choices=['normal', 'icbinb'], 
                          help='Type of writeup (normal=8 pages, icbinb=4 pages)')
        parser.add_argument('--skip-writeup', action='store_true', 
                          help='Skip paper writeup phase')
        parser.add_argument('--skip-review', action='store_true',
                          help='Skip paper review phase')
        
        # Tree search parameters
        parser.add_argument('--num-workers', type=int, help='Number of parallel workers')
        parser.add_argument('--max-steps', type=int, help='Maximum tree search steps')
    
    def _add_topic_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for create-topic command."""
        parser.add_argument('--output', '-o', required=True,
                          help='Output path for topic template')
        parser.add_argument('--template', choices=['basic', 'advanced', 'nlp', 'cv', 'ml'],
                          default='basic', help='Topic template type')
    
    def _add_status_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for status command."""
        parser.add_argument('--session', help='Session ID to check status')
        parser.add_argument('--list-sessions', action='store_true',
                          help='List all available sessions')
        parser.add_argument('--output-dir', default='autonomous_research_output',
                          help='Research output directory')
    
    def _add_config_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for create-config command."""
        parser.add_argument('--output', '-o', default='research_config.yaml',
                          help='Output path for configuration template')
        parser.add_argument('--template', choices=['basic', 'advanced', 'production'],
                          default='basic', help='Configuration template type')
    
    async def execute(self, args: Optional[list] = None) -> None:
        """Execute CLI commands."""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        if parsed_args.command == 'run':
            await self._execute_run(parsed_args)
        elif parsed_args.command == 'create-topic':
            await self._execute_create_topic(parsed_args)
        elif parsed_args.command == 'status':
            await self._execute_status(parsed_args)
        elif parsed_args.command == 'create-config':
            await self._execute_create_config(parsed_args)
    
    async def _execute_run(self, args) -> None:
        """Execute research pipeline."""
        print("🚀 Starting Terragon Autonomous Research System")
        
        if args.quick_start:
            print("📋 Quick start mode - using example configuration")
            await self._quick_start_setup()
        
        # Load or create configuration
        try:
            config = await load_research_config(args.config)
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return
        
        # Apply command line overrides
        self._apply_config_overrides(config, args)
        
        # Validate configuration
        if not self._validate_config(config):
            return
        
        # Execute research pipeline
        engine = AutonomousResearchEngine(config)
        
        try:
            print(f"🎯 Research session: {engine.session_id}")
            results = await engine.execute_full_pipeline()
            
            print("\n" + "="*60)
            print("🎉 RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Session ID: {results['session_id']}")
            print(f"Output Directory: {engine.output_dir}")
            print(f"Total Phases: {len(results['phases'])}")
            
            # Print phase summaries
            for phase_name, phase_data in results['phases'].items():
                status = phase_data['status']
                emoji = "✅" if status == "completed" else "❌"
                print(f"{emoji} {phase_name.title()}: {status}")
            
            if 'token_usage' in results:
                token_usage = results['token_usage']
                print(f"\n💰 Token Usage:")
                for model, usage in token_usage.items():
                    if isinstance(usage, dict) and 'total_cost' in usage:
                        print(f"  {model}: ${usage['total_cost']:.2f}")
            
            print(f"\n📁 Full results saved to: {engine.output_dir}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Research pipeline interrupted by user")
            
        except Exception as e:
            print(f"\n❌ Research pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def _execute_create_topic(self, args) -> None:
        """Create research topic template."""
        output_path = Path(args.output)
        
        if output_path.exists():
            response = input(f"File {output_path} exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        template_content = self._get_topic_template(args.template)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        print(f"✅ Research topic template created: {output_path}")
        print(f"📝 Template type: {args.template}")
        print(f"💡 Edit the file with your research ideas and run:")
        print(f"   python terragon_research_cli.py run --topic {output_path}")
    
    async def _execute_status(self, args) -> None:
        """Check session status."""
        output_dir = Path(args.output_dir)
        
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return
        
        if args.list_sessions:
            self._list_sessions(output_dir)
        elif args.session:
            self._show_session_status(output_dir, args.session)
        else:
            print("Please specify --session ID or use --list-sessions")
    
    async def _execute_create_config(self, args) -> None:
        """Create configuration template."""
        output_path = Path(args.output)
        
        if output_path.exists():
            response = input(f"File {output_path} exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        config_content = self._get_config_template(args.template)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Configuration template created: {output_path}")
        print(f"📝 Template type: {args.template}")
    
    async def _quick_start_setup(self) -> None:
        """Setup quick start configuration."""
        # Create example topic if not exists
        topic_path = Path("example_research_topic.md")
        if not topic_path.exists():
            topic_content = self._get_topic_template("basic")
            with open(topic_path, 'w') as f:
                f.write(topic_content)
            print(f"✅ Created example topic: {topic_path}")
        
        # Create basic config if not exists
        config_path = Path("research_config.yaml")
        if not config_path.exists():
            config_content = self._get_config_template("basic")
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ Created basic configuration: {config_path}")
    
    def _apply_config_overrides(self, config: ResearchConfig, args) -> None:
        """Apply command line overrides to configuration."""
        if args.topic:
            config.topic_description_path = args.topic
        if args.output_dir:
            config.output_directory = args.output_dir
        if args.ideation_model:
            config.ideation_model = args.ideation_model
        if args.experiment_model:
            config.experiment_model = args.experiment_model
        if args.writeup_model:
            config.writeup_model = args.writeup_model
        if args.review_model:
            config.review_model = args.review_model
        if args.max_ideas:
            config.max_ideas = args.max_ideas
        if args.writeup_type:
            config.writeup_type = args.writeup_type
        if args.skip_writeup:
            config.skip_writeup = args.skip_writeup
        if args.skip_review:
            config.skip_review = args.skip_review
        if args.num_workers:
            config.num_workers = args.num_workers
        if args.max_steps:
            config.max_steps = args.max_steps
    
    def _validate_config(self, config: ResearchConfig) -> bool:
        """Validate configuration."""
        if not os.path.exists(config.topic_description_path):
            print(f"❌ Topic description file not found: {config.topic_description_path}")
            print("💡 Create one with: python terragon_research_cli.py create-topic --output research_topic.md")
            return False
        
        # Check for required API keys
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        missing_keys = [key for key in api_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
            print("Some functionality may be limited.")
        
        return True
    
    def _get_topic_template(self, template_type: str) -> str:
        """Get research topic template content."""
        templates = {
            "basic": """# Research Topic

## Title
Novel Approach to [Your Research Area]

## Keywords
machine learning, artificial intelligence, [your keywords]

## TL;DR
Brief description of your research focus and main objectives.

## Abstract
Detailed description of the research problem, approach, and expected contributions.
This should be 200-300 words describing:
- The problem you're addressing
- Your proposed approach
- Expected outcomes and contributions

## Research Objectives
1. Primary objective
2. Secondary objective  
3. Validation and evaluation

## Expected Contributions
- Novel algorithmic contribution
- Empirical evaluation and analysis
- Open-source implementation
- Reproducible experimental framework
""",
            "advanced": """# Advanced Research Topic

## Title
[Specific and Descriptive Research Title]

## Keywords
[5-8 relevant keywords]

## TL;DR
[50-word summary of the research]

## Abstract
[300-word detailed abstract covering background, problem, approach, and contributions]

## Background and Motivation
- Current state of the field
- Identified gaps or limitations
- Motivation for this specific research

## Research Questions
1. Primary research question
2. Secondary research questions
3. Evaluation metrics and validation approach

## Methodology
- Proposed approach and techniques
- Experimental design
- Datasets and evaluation metrics
- Baseline comparisons

## Expected Contributions
- Theoretical contributions
- Algorithmic innovations
- Empirical insights
- Practical applications

## Timeline and Milestones
- Phase 1: Literature review and baseline implementation
- Phase 2: Novel algorithm development
- Phase 3: Experimental evaluation
- Phase 4: Paper writing and publication
""",
            "nlp": """# Natural Language Processing Research

## Title
Advanced Techniques in [NLP Subdomain]

## Keywords
natural language processing, transformers, language models, [specific techniques]

## TL;DR
Research focused on improving [specific NLP task] through novel architectural or algorithmic innovations.

## Abstract
This research investigates novel approaches to [specific NLP problem]. We propose [your approach] 
that addresses current limitations in [existing methods]. Our approach leverages [techniques] to 
achieve improved performance on [tasks/metrics]. Expected contributions include new model 
architectures, training techniques, and comprehensive evaluation on standard benchmarks.

## Research Focus Areas
- Model architecture innovations
- Training methodology improvements
- Evaluation and benchmarking
- Efficiency and scalability

## Datasets and Evaluation
- Primary datasets: [list key datasets]
- Evaluation metrics: [accuracy, BLEU, etc.]
- Baseline comparisons: [state-of-the-art models]

## Expected Contributions
- Novel model architecture
- Improved training techniques
- Comprehensive empirical evaluation
- Open-source implementation
""",
            "cv": """# Computer Vision Research

## Title
Advanced Computer Vision for [Specific Application]

## Keywords
computer vision, deep learning, neural networks, [specific techniques]

## TL;DR
Research on novel computer vision techniques for [specific task] with improved accuracy and efficiency.

## Abstract
This research develops advanced computer vision techniques for [specific problem]. We investigate 
[approach] to address limitations in current methods. Our approach combines [techniques] to achieve 
better performance on [metrics]. The research includes novel architectures, training strategies, 
and comprehensive evaluation on standard datasets.

## Technical Approach
- Novel neural network architectures
- Advanced data augmentation techniques
- Multi-scale feature processing
- Attention mechanisms

## Datasets and Benchmarks
- Primary datasets: [ImageNet, COCO, etc.]
- Evaluation metrics: [accuracy, mAP, etc.]
- Baseline models: [ResNet, EfficientNet, etc.]

## Expected Contributions
- Novel CNN/Vision Transformer architectures
- Improved training methodologies
- State-of-the-art benchmark results
- Efficient implementation for deployment
""",
            "ml": """# Machine Learning Research

## Title
Novel Machine Learning Approaches for [Problem Domain]

## Keywords
machine learning, optimization, algorithms, [specific techniques]

## TL;DR
Research on fundamental machine learning algorithms with improved theoretical properties and practical performance.

## Abstract
This research investigates novel machine learning algorithms for [problem type]. We develop [approach] 
that provides better [theoretical guarantees/empirical performance] compared to existing methods. 
Our work includes theoretical analysis, algorithmic innovations, and extensive experimental validation 
across multiple domains and datasets.

## Research Components
- Algorithm development and theoretical analysis
- Optimization and computational efficiency
- Empirical evaluation and comparison
- Real-world applications and case studies

## Evaluation Framework
- Theoretical analysis (convergence, complexity)
- Synthetic data experiments
- Real-world dataset evaluation
- Computational efficiency benchmarks

## Expected Contributions
- Novel algorithm with theoretical guarantees
- Comprehensive empirical evaluation
- Open-source implementation
- Application to real-world problems
"""
        }
        
        return templates.get(template_type, templates["basic"])
    
    def _get_config_template(self, template_type: str) -> str:
        """Get configuration template content."""
        templates = {
            "basic": """# Basic Research Configuration
topic_description_path: "example_research_topic.md"
output_directory: "autonomous_research_output"
max_ideas: 3
idea_reflections: 2

# Model configurations
ideation_model: "gpt-4o-2024-05-13"
experiment_model: "claude-3-5-sonnet"
writeup_model: "o1-preview-2024-09-12"
citation_model: "gpt-4o-2024-11-20"
review_model: "gpt-4o-2024-11-20"
plotting_model: "o3-mini-2025-01-31"

# Execution parameters
writeup_type: "icbinb"  # "normal" (8 pages) or "icbinb" (4 pages)
writeup_retries: 2
citation_rounds: 15
skip_writeup: false
skip_review: false

# Tree search parameters
num_workers: 2
max_steps: 15
max_debug_depth: 2
debug_probability: 0.6
num_drafts: 1
""",
            "advanced": """# Advanced Research Configuration
topic_description_path: "research_topic.md"
output_directory: "advanced_research_output"
max_ideas: 8
idea_reflections: 5

# Model configurations
ideation_model: "gpt-4o-2024-05-13"
experiment_model: "claude-3-5-sonnet"
writeup_model: "o1-preview-2024-09-12"
citation_model: "gpt-4o-2024-11-20"
review_model: "gpt-4o-2024-11-20"
plotting_model: "o3-mini-2025-01-31"

# Execution parameters
writeup_type: "normal"  # Full 8-page papers
writeup_retries: 3
citation_rounds: 25
skip_writeup: false
skip_review: false

# Advanced tree search parameters
num_workers: 4
max_steps: 30
max_debug_depth: 5
debug_probability: 0.8
num_drafts: 3

# Quality control
enable_quality_gates: true
minimum_experiment_success_rate: 0.7
require_statistical_significance: true
""",
            "production": """# Production Research Configuration
topic_description_path: "production_research_topic.md"
output_directory: "production_research_output"
max_ideas: 10
idea_reflections: 8

# Production model configurations
ideation_model: "gpt-4o-2024-05-13"
experiment_model: "claude-3-5-sonnet"
writeup_model: "o1-preview-2024-09-12"
citation_model: "gpt-4o-2024-11-20"
review_model: "gpt-4o-2024-11-20"
plotting_model: "o3-mini-2025-01-31"

# Production execution parameters
writeup_type: "normal"
writeup_retries: 5
citation_rounds: 30
skip_writeup: false
skip_review: false

# Production tree search parameters
num_workers: 6
max_steps: 50
max_debug_depth: 8
debug_probability: 0.9
num_drafts: 5

# Production features
enable_quality_gates: true
enable_monitoring: true
enable_backup: true
enable_distributed_execution: true
minimum_experiment_success_rate: 0.8
require_statistical_significance: true
enable_automated_paper_submission: false

# Resource limits
max_token_budget: 1000000
max_execution_time_hours: 24
max_concurrent_experiments: 3
"""
        }
        
        return templates.get(template_type, templates["basic"])
    
    def _list_sessions(self, output_dir: Path) -> None:
        """List all research sessions."""
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return
        
        session_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('session_')]
        
        if not session_dirs:
            print("📭 No research sessions found")
            return
        
        print(f"📋 Found {len(session_dirs)} research sessions:")
        print("="*60)
        
        for session_dir in sorted(session_dirs, reverse=True):
            session_id = session_dir.name.replace('session_', '')
            
            # Load session results if available
            results_file = session_dir / f"session_results_{session_id}.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    status = results.get('status', 'unknown')
                    phases = len(results.get('phases', {}))
                    start_time = results.get('start_time', 'unknown')
                    
                    emoji = "✅" if status == "completed" else "🔄" if status == "running" else "❌"
                    print(f"{emoji} {session_id}")
                    print(f"   Status: {status}")
                    print(f"   Phases: {phases}")
                    print(f"   Started: {start_time}")
                    print(f"   Path: {session_dir}")
                    
                except Exception as e:
                    print(f"❓ {session_id} (error reading results: {e})")
            else:
                print(f"❓ {session_id} (no results file)")
            
            print()
    
    def _show_session_status(self, output_dir: Path, session_id: str) -> None:
        """Show detailed status for a specific session."""
        session_dir = output_dir / f"session_{session_id}"
        
        if not session_dir.exists():
            print(f"❌ Session not found: {session_id}")
            return
        
        results_file = session_dir / f"session_results_{session_id}.json"
        
        if not results_file.exists():
            print(f"❌ Session results not found for: {session_id}")
            return
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"📊 Session Status: {session_id}")
            print("="*60)
            print(f"Status: {results.get('status', 'unknown')}")
            print(f"Started: {results.get('start_time', 'unknown')}")
            print(f"Ended: {results.get('end_time', 'ongoing')}")
            print(f"Configuration: {results.get('config', {}).get('topic_description_path', 'unknown')}")
            
            phases = results.get('phases', {})
            print(f"\nPhases Completed: {len(phases)}")
            
            for phase_name, phase_data in phases.items():
                status = phase_data['status']
                emoji = "✅" if status == "completed" else "❌"
                timestamp = phase_data.get('timestamp', 'unknown')
                print(f"  {emoji} {phase_name.title()}: {status} ({timestamp})")
            
            # Token usage summary
            if 'token_usage' in results:
                print(f"\n💰 Token Usage Summary:")
                token_usage = results['token_usage']
                total_cost = 0
                for model, usage in token_usage.items():
                    if isinstance(usage, dict) and 'total_cost' in usage:
                        cost = usage['total_cost']
                        total_cost += cost
                        print(f"  {model}: ${cost:.2f}")
                print(f"  Total: ${total_cost:.2f}")
            
            print(f"\n📁 Full results: {session_dir}")
            
        except Exception as e:
            print(f"❌ Error reading session results: {e}")


async def main():
    """Main CLI entry point."""
    cli = ResearchCLI()
    await cli.execute()


if __name__ == "__main__":
    asyncio.run(main())