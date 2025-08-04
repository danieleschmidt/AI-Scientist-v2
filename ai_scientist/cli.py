#!/usr/bin/env python3
"""
AI Scientist v2 - Command Line Interface

Enterprise-grade CLI for autonomous scientific discovery via agentic tree search.
Supports multiple workflows: ideation, experimentation, paper writing, and analysis.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ai_scientist.utils.config import load_config, validate_config
from ai_scientist.utils.api_security import validate_api_keys
from ai_scientist.monitoring.health_checks import HealthChecker
from ai_scientist.perform_ideation_temp_free import main as ideation_main
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import main as experiment_main
from ai_scientist.perform_writeup import main as writeup_main

console = Console()
logger = logging.getLogger(__name__)


class AIScientistCLI:
    """Advanced CLI for AI Scientist v2 with comprehensive workflow management."""
    
    def __init__(self):
        self.config = {}
        self.health_checker = HealthChecker()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure rich logging with appropriate levels."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )
    
    def load_configuration(self, config_path: Optional[str] = None) -> bool:
        """Load and validate configuration files."""
        try:
            config_file = config_path or "ai_scientist_config.yaml"
            self.config = load_config(config_file)
            
            # Validate configuration schema
            if not validate_config(self.config):
                console.print("[red]❌ Configuration validation failed[/red]")
                return False
                
            console.print("[green]✅ Configuration loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Configuration loading failed: {e}[/red]")
            return False
    
    def validate_environment(self) -> bool:
        """Validate API keys and system requirements."""
        try:
            # Check API keys
            required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
            optional_keys = ["S2_API_KEY", "GEMINI_API_KEY"]
            
            missing_keys = []
            for key in required_keys:
                if not os.getenv(key):
                    missing_keys.append(key)
            
            if missing_keys:
                console.print(f"[red]❌ Missing required API keys: {', '.join(missing_keys)}[/red]")
                return False
            
            # Validate API key formats
            if not validate_api_keys():
                console.print("[red]❌ API key validation failed[/red]")
                return False
            
            console.print("[green]✅ Environment validation passed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Environment validation failed: {e}[/red]")
            return False
    
    def health_check(self) -> bool:
        """Perform comprehensive system health check."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running health checks...", total=None)
                
                health_status = self.health_checker.check_all()
                
                if health_status.get("overall_health", False):
                    console.print("[green]✅ System health check passed[/green]")
                    return True
                else:
                    console.print("[red]❌ System health check failed[/red]")
                    self._display_health_details(health_status)
                    return False
                    
        except Exception as e:
            console.print(f"[red]❌ Health check failed: {e}[/red]")
            return False
    
    def _display_health_details(self, health_status: Dict):
        """Display detailed health check results."""
        table = Table(title="System Health Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="white")
        
        for component, details in health_status.items():
            if component == "overall_health":
                continue
                
            status = "✅ Healthy" if details.get("healthy", False) else "❌ Unhealthy"
            info = details.get("details", "No details available")
            table.add_row(component, status, str(info))
        
        console.print(table)
    
    def run_ideation(self, args) -> bool:
        """Execute research ideation workflow."""
        try:
            console.print("[bold blue]🧠 Starting Research Ideation[/bold blue]")
            
            # Validate workshop file exists
            if not Path(args.workshop_file).exists():
                console.print(f"[red]❌ Workshop file not found: {args.workshop_file}[/red]")
                return False
            
            # Build arguments for ideation
            ideation_args = [
                "--workshop-file", args.workshop_file,
                "--model", args.model,
                "--max-num-generations", str(args.max_generations),
                "--num-reflections", str(args.num_reflections)
            ]
            
            if args.verbose:
                ideation_args.append("--verbose")
            
            # Execute ideation
            result = ideation_main(ideation_args)
            
            if result:
                console.print("[green]✅ Research ideation completed successfully[/green]")
                return True
            else:
                console.print("[red]❌ Research ideation failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Ideation workflow failed: {e}[/red]")
            return False
    
    def run_experiments(self, args) -> bool:
        """Execute experimental research workflow."""
        try:
            console.print("[bold blue]🧪 Starting Experimental Research[/bold blue]")
            
            # Validate ideas file exists
            if not Path(args.ideas_file).exists():
                console.print(f"[red]❌ Ideas file not found: {args.ideas_file}[/red]")
                return False
            
            # Build arguments for experiments
            experiment_args = [
                "--load_ideas", args.ideas_file,
                "--model_writeup", args.model_writeup,
                "--model_citation", args.model_citation,
                "--model_review", args.model_review,
                "--num_cite_rounds", str(args.num_cite_rounds)
            ]
            
            if args.load_code:
                experiment_args.append("--load_code")
            if args.add_dataset_ref:
                experiment_args.append("--add_dataset_ref")
            if args.verbose:
                experiment_args.append("--verbose")
            
            # Execute experiments
            result = experiment_main(experiment_args)
            
            if result:
                console.print("[green]✅ Experimental research completed successfully[/green]")
                return True
            else:
                console.print("[red]❌ Experimental research failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Experimental workflow failed: {e}[/red]")
            return False
    
    def run_writeup(self, args) -> bool:
        """Execute paper writing workflow."""
        try:
            console.print("[bold blue]📝 Starting Paper Writing[/bold blue]")
            
            # Build arguments for writeup
            writeup_args = [
                "--experiment_dir", args.experiment_dir,
                "--model", args.model,
                "--citation_model", args.citation_model,
                "--review_model", args.review_model
            ]
            
            if args.verbose:
                writeup_args.append("--verbose")
            
            # Execute writeup
            result = writeup_main(writeup_args)
            
            if result:
                console.print("[green]✅ Paper writing completed successfully[/green]")
                return True
            else:
                console.print("[red]❌ Paper writing failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Writeup workflow failed: {e}[/red]")
            return False
    
    def interactive_mode(self):
        """Launch interactive mode for guided workflow execution."""
        console.print("[bold magenta]🤖 AI Scientist v2 - Interactive Mode[/bold magenta]")
        console.print("Welcome to the interactive research assistant!")
        
        while True:
            console.print("\n[bold cyan]Available Workflows:[/bold cyan]")
            console.print("1. 🧠 Research Ideation")
            console.print("2. 🧪 Experimental Research") 
            console.print("3. 📝 Paper Writing")
            console.print("4. 🔍 System Health Check")
            console.print("5. ⚙️  Configuration Status")
            console.print("6. 🚪 Exit")
            
            choice = Prompt.ask("Select workflow", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                self._interactive_ideation()
            elif choice == "2":
                self._interactive_experiments()
            elif choice == "3":
                self._interactive_writeup()
            elif choice == "4":
                self.health_check()
            elif choice == "5":
                self._show_configuration_status()
            elif choice == "6":
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
    
    def _interactive_ideation(self):
        """Interactive ideation workflow."""
        workshop_file = Prompt.ask("Enter workshop file path", default="ai_scientist/ideas/my_research_topic.md")
        model = Prompt.ask("Select model", default="gpt-4o-2024-05-13")
        max_generations = int(Prompt.ask("Max generations", default="20"))
        num_reflections = int(Prompt.ask("Number of reflections", default="5"))
        
        # Create mock args object
        class Args:
            def __init__(self):
                self.workshop_file = workshop_file
                self.model = model
                self.max_generations = max_generations
                self.num_reflections = num_reflections
                self.verbose = True
        
        self.run_ideation(Args())
    
    def _interactive_experiments(self):
        """Interactive experiments workflow."""
        ideas_file = Prompt.ask("Enter ideas JSON file path")
        model_writeup = Prompt.ask("Writeup model", default="o1-preview-2024-09-12")
        model_citation = Prompt.ask("Citation model", default="gpt-4o-2024-11-20")
        model_review = Prompt.ask("Review model", default="gpt-4o-2024-11-20")
        load_code = Confirm.ask("Load code snippets?", default=True)
        add_dataset_ref = Confirm.ask("Add dataset references?", default=True)
        
        # Create mock args object
        class Args:
            def __init__(self):
                self.ideas_file = ideas_file
                self.model_writeup = model_writeup
                self.model_citation = model_citation
                self.model_review = model_review
                self.num_cite_rounds = 20
                self.load_code = load_code
                self.add_dataset_ref = add_dataset_ref
                self.verbose = True
        
        self.run_experiments(Args())
    
    def _interactive_writeup(self):
        """Interactive writeup workflow."""
        experiment_dir = Prompt.ask("Enter experiment directory path")
        model = Prompt.ask("Writeup model", default="o1-preview-2024-09-12")
        citation_model = Prompt.ask("Citation model", default="gpt-4o-2024-11-20")
        review_model = Prompt.ask("Review model", default="gpt-4o-2024-11-20")
        
        # Create mock args object
        class Args:
            def __init__(self):
                self.experiment_dir = experiment_dir
                self.model = model
                self.citation_model = citation_model
                self.review_model = review_model
                self.verbose = True
        
        self.run_writeup(Args())
    
    def _show_configuration_status(self):
        """Display current configuration status."""
        table = Table(title="Configuration Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Show key configuration items
        for key, value in self.config.items():
            if isinstance(value, (str, int, float, bool)):
                table.add_row(key, str(value))
            elif isinstance(value, list):
                table.add_row(key, f"{len(value)} items")
            elif isinstance(value, dict):
                table.add_row(key, f"{len(value)} settings")
        
        console.print(table)


def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Scientist v2 - Autonomous Scientific Discovery System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research ideation
  ai-scientist ideate --workshop-file my_topic.md --model gpt-4o-2024-05-13
  
  # Run experiments  
  ai-scientist experiment --ideas-file ideas.json --model-writeup o1-preview
  
  # Write paper
  ai-scientist writeup --experiment-dir experiments/results/
  
  # Interactive mode
  ai-scientist interactive
  
  # System health check
  ai-scientist health-check
        """
    )
    
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ideation command
    ideate_parser = subparsers.add_parser("ideate", help="Generate research ideas")
    ideate_parser.add_argument("--workshop-file", required=True, help="Workshop topic file (.md)")
    ideate_parser.add_argument("--model", default="gpt-4o-2024-05-13", help="LLM model for ideation")
    ideate_parser.add_argument("--max-generations", type=int, default=20, help="Maximum idea generations")
    ideate_parser.add_argument("--num-reflections", type=int, default=5, help="Number of refinement steps")
    
    # Experiment command
    experiment_parser = subparsers.add_parser("experiment", help="Run experimental research")
    experiment_parser.add_argument("--ideas-file", required=True, help="Generated ideas JSON file")
    experiment_parser.add_argument("--model-writeup", default="o1-preview-2024-09-12", help="Writeup model")
    experiment_parser.add_argument("--model-citation", default="gpt-4o-2024-11-20", help="Citation model")
    experiment_parser.add_argument("--model-review", default="gpt-4o-2024-11-20", help="Review model")
    experiment_parser.add_argument("--num-cite-rounds", type=int, default=20, help="Citation rounds")
    experiment_parser.add_argument("--load-code", action="store_true", help="Load code snippets")
    experiment_parser.add_argument("--add-dataset-ref", action="store_true", help="Add dataset references")
    
    # Writeup command
    writeup_parser = subparsers.add_parser("writeup", help="Generate research paper")
    writeup_parser.add_argument("--experiment-dir", required=True, help="Experiment results directory")
    writeup_parser.add_argument("--model", default="o1-preview-2024-09-12", help="Writeup model")
    writeup_parser.add_argument("--citation-model", default="gpt-4o-2024-11-20", help="Citation model")
    writeup_parser.add_argument("--review-model", default="gpt-4o-2024-11-20", help="Review model")
    
    # Utility commands
    subparsers.add_parser("interactive", help="Launch interactive mode")
    subparsers.add_parser("health-check", help="Run system health check")
    subparsers.add_parser("validate", help="Validate configuration and environment")
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point with comprehensive error handling."""
    try:
        parser = create_parser()
        parsed_args = parser.parse_args(args)
        
        # Initialize CLI
        cli = AIScientistCLI()
        
        # Setup debug logging if requested
        if getattr(parsed_args, 'debug', False):
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load configuration
        if not cli.load_configuration(getattr(parsed_args, 'config', None)):
            return 1
        
        # Validate environment for non-utility commands
        if parsed_args.command not in ['health-check', 'validate']:
            if not cli.validate_environment():
                return 1
        
        # Execute command
        if parsed_args.command == "ideate":
            success = cli.run_ideation(parsed_args)
        elif parsed_args.command == "experiment":
            success = cli.run_experiments(parsed_args)
        elif parsed_args.command == "writeup":
            success = cli.run_writeup(parsed_args)
        elif parsed_args.command == "interactive":
            cli.interactive_mode()
            success = True
        elif parsed_args.command == "health-check":
            success = cli.health_check()
        elif parsed_args.command == "validate":
            success = cli.validate_environment()
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        if getattr(parsed_args, 'debug', False):
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())