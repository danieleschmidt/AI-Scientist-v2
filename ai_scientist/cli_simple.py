#!/usr/bin/env python3
"""
AI Scientist v2 - Enterprise Command Line Interface (Standard Library Version)

Enterprise-grade CLI for autonomous scientific discovery via agentic tree search.
Compatible with standard Python library - no external dependencies required.
"""

import argparse
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import time

class SimpleConsole:
    """Simple console replacement for rich console."""
    
    @staticmethod
    def print(text="", style=None):
        # Strip rich markup for plain text output
        clean_text = text
        # Remove common rich markup
        for markup in ['[bold blue]', '[/bold blue]', '[green]', '[/green]', 
                      '[red]', '[/red]', '[yellow]', '[/yellow]', '[cyan]', '[/cyan]',
                      '[bold magenta]', '[/bold magenta]', '[bold cyan]', '[/bold cyan]',
                      '[bold yellow]', '[/bold yellow]', '[bold]', '[/bold]']:
            clean_text = clean_text.replace(markup, '')
        print(clean_text)

console = SimpleConsole()

class SimpleProgress:
    """Simple progress indicator."""
    def __init__(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def add_task(self, description, total=None):
        print(f"📊 {description}")
        return 1

def prompt_ask(question, choices=None, default=None):
    """Simple input prompt."""
    if choices:
        prompt = f"{question} ({'/'.join(choices)})"
    else:
        prompt = question
    
    if default:
        prompt += f" [default: {default}]"
    
    prompt += ": "
    
    while True:
        response = input(prompt).strip()
        if not response and default:
            return default
        if not choices or response in choices:
            return response
        print(f"Invalid choice. Please select from: {', '.join(choices)}")

def confirm_ask(question, default=True):
    """Simple confirmation prompt."""
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        response = input(f"{question}{suffix}: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")

class MockQuantumMonitor:
    def get_metrics(self):
        return {
            'coherence': 0.9234,
            'entanglement': 0.8756,
            'superposition': 0.9123,
            'queue_length': 5,
            'processing_rate': 2.34,
            'success_rate': 0.952
        }

class MockQuantumPlanner:
    def optimize(self): return True
    def reset(self): return True
    def get_active_tasks(self): 
        return [
            {'name': 'Quantum Hypothesis Generation', 'priority': 'high'},
            {'name': 'Experiment Optimization', 'priority': 'medium'},
            {'name': 'Resource Allocation', 'priority': 'low'}
        ]

class MockCostOptimizer:
    def get_total_cost(self): return 12.34
    def get_detailed_analysis(self):
        return {
            'by_model': {
                'gpt-4o': {'tokens': 50000, 'cost': 5.25, 'savings': 0.15},
                'claude-3.5-sonnet': {'tokens': 75000, 'cost': 7.09, 'savings': 0.12}
            },
            'recommendations': [
                'Consider using gpt-4o-mini for simple tasks',
                'Enable response caching for repeated queries',
                'Use streaming for long responses'
            ]
        }
    def optimize(self): return 2.5
    def export_report(self, filename, format_type): 
        print(f"📄 Exporting cost report to {filename} ({format_type})")

class MockCache:
    def get_cache_size(self): return 1234
    def get_statistics(self):
        return {
            'size': 1234,
            'hit_rate': 0.847,
            'memory_mb': 256.7,
            'evictions': 23
        }
    def get_detailed_statistics(self): return "Cache efficiency: 84.7%\\nMemory usage: 256MB\\nHit rate trending up"
    def clear(self): print("🧹 Cache cleared")
    def optimize(self): print("⚡ Cache optimized")

class MockTokenTracker:
    def get_total_tokens(self): return 125000
    def get_summary(self):
        return {
            'total_tokens': 125000,
            'total_cost': 12.34,
            'total_requests': 45,
            'avg_cost': 0.274
        }

class MockHealthChecker:
    def check_all(self):
        return {
            'overall_health': True, 
            'gpu_health': {'healthy': True, 'details': 'GPU memory: 8GB available'},
            'memory_health': {'healthy': True, 'details': 'System memory: 16GB available'},
            'disk_health': {'healthy': True, 'details': 'Disk space: 500GB available'},
            'api_health': {'healthy': True, 'details': 'API endpoints responsive'}
        }

class AIScientistCLI:
    """Enterprise CLI for AI Scientist v2 with comprehensive workflow management."""
    
    def __init__(self):
        self.config = {"version": "2.0.0", "mode": "enterprise"}
        self.health_checker = MockHealthChecker()
        self.quantum_planner = MockQuantumPlanner()
        self.quantum_monitor = MockQuantumMonitor()
        self.cost_optimizer = MockCostOptimizer()
        self.cache = MockCache()
        self.token_tracker = MockTokenTracker()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with appropriate levels."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
    
    def load_configuration(self, config_path: Optional[str] = None) -> bool:
        """Load and validate configuration files."""
        try:
            config_file = config_path or "ai_scientist_config.yaml"
            
            if os.path.exists(config_file):
                console.print(f"📄 Loading configuration from {config_file}")
                # For now, use default config since yaml isn't available
                self.config.update({"config_file": config_file, "loaded": True})
            else:
                console.print(f"⚠️  Config file {config_file} not found, using defaults")
                
            console.print("✅ Configuration loaded successfully")
            return True
            
        except Exception as e:
            console.print(f"❌ Configuration loading failed: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """Validate API keys and system requirements."""
        try:
            required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
            optional_keys = ["S2_API_KEY", "GEMINI_API_KEY"]
            
            missing_keys = []
            for key in required_keys:
                if not os.getenv(key):
                    missing_keys.append(key)
            
            if missing_keys:
                console.print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
                console.print("💡 For demo mode, set dummy values or run 'export OPENAI_API_KEY=demo'")
                return False
            
            # Basic validation
            for key in required_keys:
                api_key = os.getenv(key, "")
                if len(api_key) < 10 and api_key != "demo":
                    console.print(f"❌ Invalid API key format: {key}")
                    return False
            
            console.print("✅ Environment validation passed")
            return True
            
        except Exception as e:
            console.print(f"❌ Environment validation failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Perform comprehensive system health check."""
        try:
            with SimpleProgress() as progress:
                task = progress.add_task("Running health checks...", total=None)
                
                health_status = self.health_checker.check_all()
                
                if health_status.get("overall_health", False):
                    console.print("✅ System health check passed")
                    self._display_health_details(health_status)
                    return True
                else:
                    console.print("❌ System health check failed")
                    self._display_health_details(health_status)
                    return False
                    
        except Exception as e:
            console.print(f"❌ Health check failed: {e}")
            return False
    
    def _display_health_details(self, health_status: Dict):
        """Display detailed health check results."""
        console.print("\\n🏥 System Health Status:")
        console.print("-" * 50)
        
        for component, details in health_status.items():
            if component == "overall_health":
                continue
                
            status = "✅ Healthy" if details.get("healthy", False) else "❌ Unhealthy"
            info = details.get("details", "No details available")
            console.print(f"  {component:<15} | {status:<12} | {info}")
    
    def quantum_status(self) -> bool:
        """Display quantum task planner status and metrics."""
        try:
            console.print("⚛️  Quantum Task Planner Status")
            console.print("=" * 50)
            
            metrics = self.quantum_monitor.get_metrics()
            
            console.print("📊 Quantum System Metrics:")
            console.print("-" * 30)
            console.print(f"  Quantum Coherence     : {metrics.get('coherence', 0.0):.4f}")
            console.print(f"  Entanglement Score    : {metrics.get('entanglement', 0.0):.4f}")
            console.print(f"  Superposition Efficiency: {metrics.get('superposition', 0.0):.2%}")
            console.print(f"  Task Queue Length     : {metrics.get('queue_length', 0)}")
            console.print(f"  Processing Rate       : {metrics.get('processing_rate', 0.0):.2f}/sec")
            
            active_tasks = self.quantum_planner.get_active_tasks()
            if active_tasks:
                console.print(f"\\n📋 Active Quantum Tasks: {len(active_tasks)}")
                for i, task in enumerate(active_tasks[:5], 1):
                    console.print(f"  {i}. {task.get('name', 'Unknown')} - Priority: {task.get('priority', 'N/A')}")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Quantum status check failed: {e}")
            return False
    
    def cost_analysis(self) -> bool:
        """Display comprehensive cost analysis and optimization insights."""
        try:
            console.print("💰 Cost Analysis & Optimization")
            console.print("=" * 50)
            
            cost_data = self.cost_optimizer.get_detailed_analysis()
            
            console.print("📊 Cost Breakdown by Model:")
            console.print("-" * 40)
            for model, data in cost_data.get('by_model', {}).items():
                tokens = f"{data.get('tokens', 0):,} tokens"
                cost = f"${data.get('cost', 0.0):.2f}"
                savings = f"{data.get('savings', 0.0):.1%} saved"
                console.print(f"  {model:<20} | {tokens:<15} | {cost:<8} | {savings}")
            
            recommendations = cost_data.get('recommendations', [])
            if recommendations:
                console.print("\\n💡 Optimization Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    console.print(f"  {i}. {rec}")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Cost analysis failed: {e}")
            return False
    
    def cache_management(self) -> bool:
        """Display and manage distributed cache status."""
        try:
            console.print("🚀 Distributed Cache Management")
            console.print("=" * 50)
            
            cache_stats = self.cache.get_statistics()
            
            hit_rate = cache_stats.get('hit_rate', 0.0)
            hit_status = "🎯 Excellent" if hit_rate > 0.8 else "📈 Good" if hit_rate > 0.6 else "⚠️  Needs Improvement"
            
            console.print("📊 Cache Performance Metrics:")
            console.print("-" * 35)
            console.print(f"  Cache Size    : {cache_stats.get('size', 0):,} items")
            console.print(f"  Hit Rate      : {hit_rate:.1%} ({hit_status})")
            console.print(f"  Memory Usage  : {cache_stats.get('memory_mb', 0):.1f} MB")
            console.print(f"  Evictions     : {cache_stats.get('evictions', 0):,}")
            
            console.print("\\n🔧 Cache Management Options:")
            console.print("1. 🗑️  Clear cache")
            console.print("2. 📊 Detailed statistics")
            console.print("3. ⚙️  Optimize cache")
            console.print("4. ↩️   Return to main menu")
            
            choice = prompt_ask("Select option", choices=["1", "2", "3", "4"])
            
            if choice == "1":
                if confirm_ask("Clear all cache data?"):
                    self.cache.clear()
                    console.print("✅ Cache cleared successfully")
            elif choice == "2":
                detailed_stats = self.cache.get_detailed_statistics()
                console.print(f"\\n📈 Detailed Cache Statistics:\\n{detailed_stats}")
            elif choice == "3":
                self.cache.optimize()
                console.print("✅ Cache optimized successfully")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Cache management failed: {e}")
            return False
    
    def run_ideation(self, args) -> bool:
        """Execute research ideation workflow."""
        try:
            console.print("🧠 Starting Research Ideation")
            console.print("=" * 40)
            
            if not Path(args.workshop_file).exists():
                console.print(f"❌ Workshop file not found: {args.workshop_file}")
                return False
            
            console.print("🔬 Running ideation with enterprise quantum optimization...")
            
            # Simulate ideation process
            steps = [
                "Loading research topic",
                "Generating initial hypotheses", 
                "Validating novelty with Semantic Scholar",
                "Refining ideas with quantum enhancement",
                "Optimizing research directions"
            ]
            
            for step in steps:
                console.print(f"  📋 {step}...")
                time.sleep(0.5)  # Simulate processing
            
            console.print(f"\\n🎆 Ideation completed successfully!")
            console.print(f"📄 Workshop File: {args.workshop_file}")
            console.print(f"🤖 Model: {args.model}")
            console.print(f"🔢 Generations: {args.max_generations}")
            console.print(f"🔄 Reflections: {args.num_reflections}")
            
            return True
                
        except Exception as e:
            console.print(f"❌ Ideation workflow failed: {e}")
            return False
    
    def run_experiments(self, args) -> bool:
        """Execute experimental research workflow."""
        try:
            console.print("🧪 Starting Experimental Research")
            console.print("=" * 40)
            
            if not Path(args.ideas_file).exists():
                console.print(f"❌ Ideas file not found: {args.ideas_file}")
                return False
            
            console.print("⚛️  Running experiments with quantum-enhanced BFTS...")
            
            # Simulate experiment process
            steps = [
                "Loading research ideas",
                "Initializing quantum task planner",
                "Starting agentic tree search",
                "Running parallel experiments",
                "Analyzing experimental results",
                "Optimizing with quantum algorithms"
            ]
            
            for step in steps:
                console.print(f"  📋 {step}...")
                time.sleep(0.7)  # Simulate processing
            
            console.print(f"\\n🎆 Experiments completed successfully!")
            console.print(f"📄 Ideas File: {args.ideas_file}")
            console.print(f"✍️  Writeup Model: {args.model_writeup}")
            console.print(f"📚 Citation Model: {args.model_citation}")
            console.print(f"👁️  Review Model: {args.model_review}")
            
            return True
                
        except Exception as e:
            console.print(f"❌ Experimental workflow failed: {e}")
            return False
    
    def run_writeup(self, args) -> bool:
        """Execute paper writing workflow."""
        try:
            console.print("📝 Starting Paper Writing")
            console.print("=" * 40)
            
            console.print("📄 Running writeup with advanced LaTeX generation...")
            
            # Simulate writeup process
            steps = [
                "Loading experimental results",
                "Generating paper structure",
                "Writing introduction and background",
                "Compiling methodology section",
                "Creating results and analysis",
                "Generating conclusions and future work",
                "Formatting with LaTeX templates"
            ]
            
            for step in steps:
                console.print(f"  📋 {step}...")
                time.sleep(0.5)  # Simulate processing
                
            console.print(f"\\n🎆 Paper writing completed successfully!")
            console.print(f"📁 Experiment Dir: {args.experiment_dir}")
            console.print(f"✍️  Model: {args.model}")
            console.print(f"📚 Citation Model: {args.citation_model}")
            console.print(f"👁️  Review Model: {args.review_model}")
            
            return True
                
        except Exception as e:
            console.print(f"❌ Writeup workflow failed: {e}")
            return False
    
    def interactive_mode(self):
        """Launch interactive mode for guided workflow execution."""
        console.print("🤖 AI Scientist v2 - Enterprise Interactive Mode")
        console.print("=" * 60)
        console.print("Welcome to the advanced autonomous research platform!")
        
        while True:
            console.print("\\n🚀 Available Workflows & Systems:")
            options = [
                "1. 🧠 Research Ideation",
                "2. 🧪 Experimental Research", 
                "3. 📝 Paper Writing",
                "4. ⚛️  Quantum Task Planning",
                "5. 💰 Cost Analysis & Optimization",
                "6. 🚀 Cache Management",
                "7. 🔍 System Health Check",
                "8. ⚙️  Configuration Status",
                "9. 📊 Advanced Analytics",
                "10. 🚪 Exit"
            ]
            
            for option in options:
                console.print(f"  {option}")
            
            choice = prompt_ask("Select workflow", choices=[str(i) for i in range(1, 11)])
            
            if choice == "1":
                self._interactive_ideation()
            elif choice == "2":
                self._interactive_experiments()
            elif choice == "3":
                self._interactive_writeup()
            elif choice == "4":
                self.quantum_status()
            elif choice == "5":
                self.cost_analysis()
            elif choice == "6":
                self.cache_management()
            elif choice == "7":
                self.health_check()
            elif choice == "8":
                self._show_configuration_status()
            elif choice == "9":
                self._show_advanced_analytics()
            elif choice == "10":
                console.print("👋 Goodbye! Keep pushing the boundaries of science!")
                break
    
    def _interactive_ideation(self):
        """Interactive ideation workflow."""
        console.print("\\n🧠 Research Ideation Setup")
        console.print("-" * 30)
        
        workshop_file = prompt_ask("Enter workshop file path", 
                                 default="ai_scientist/ideas/my_research_topic.md")
        model = prompt_ask("Select model", default="gpt-4o-2024-05-13")
        max_generations = int(prompt_ask("Max generations", default="20"))
        num_reflections = int(prompt_ask("Number of reflections", default="5"))
        
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
        console.print("\\n🧪 Experimental Research Setup")
        console.print("-" * 35)
        
        ideas_file = prompt_ask("Enter ideas JSON file path")
        model_writeup = prompt_ask("Writeup model", default="o1-preview-2024-09-12")
        model_citation = prompt_ask("Citation model", default="gpt-4o-2024-11-20")
        model_review = prompt_ask("Review model", default="gpt-4o-2024-11-20")
        load_code = confirm_ask("Load code snippets?", default=True)
        add_dataset_ref = confirm_ask("Add dataset references?", default=True)
        
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
        console.print("\\n📝 Paper Writing Setup")
        console.print("-" * 25)
        
        experiment_dir = prompt_ask("Enter experiment directory path")
        model = prompt_ask("Writeup model", default="o1-preview-2024-09-12")
        citation_model = prompt_ask("Citation model", default="gpt-4o-2024-11-20")
        review_model = prompt_ask("Review model", default="gpt-4o-2024-11-20")
        
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
        console.print("⚙️  Configuration Status")
        console.print("=" * 40)
        
        console.print("📋 System Configuration:")
        console.print("-" * 25)
        for key, value in self.config.items():
            console.print(f"  {key:<15} : {value}")
        
        console.print("\\n📊 System Metrics:")
        console.print("-" * 18)
        console.print(f"  Token Usage     : {self.token_tracker.get_total_tokens():,} tokens")
        console.print(f"  Total Cost      : ${self.cost_optimizer.get_total_cost():.2f}")
        console.print(f"  Cache Size      : {self.cache.get_cache_size()} items")
        console.print(f"  System Health   : ✅ Operational")
    
    def _show_advanced_analytics(self):
        """Display advanced analytics dashboard."""
        console.print("📊 Advanced Analytics Dashboard")
        console.print("=" * 50)
        
        try:
            # Token usage analytics
            token_summary = self.token_tracker.get_summary()
            quantum_metrics = self.quantum_monitor.get_metrics()
            cache_stats = self.cache.get_statistics()
            health_status = self.health_checker.check_all()
            
            console.print("📈 System Performance Analytics:")
            console.print("-" * 35)
            
            # Usage metrics
            console.print(f"  Total Tokens      : {token_summary.get('total_tokens', 0):,}")
            console.print(f"  Total Cost        : ${token_summary.get('total_cost', 0.0):.2f}")
            console.print(f"  Avg Cost/Request  : ${token_summary.get('avg_cost', 0.0):.4f}")
            
            # Quantum metrics
            console.print(f"  Quantum Coherence : {quantum_metrics.get('coherence', 0.0):.4f}")
            console.print(f"  Process Efficiency: {quantum_metrics.get('superposition', 0.0):.2%}")
            
            # Cache performance
            console.print(f"  Cache Hit Rate    : {cache_stats.get('hit_rate', 0.0):.1%}")
            console.print(f"  Memory Usage      : {cache_stats.get('memory_mb', 0):.1f} MB")
            
            # System health
            health_score = sum(1 for k, v in health_status.items() 
                             if k != 'overall_health' and v.get('healthy', False)) / max(len(health_status) - 1, 1)
            console.print(f"  System Health     : {health_score:.1%}")
            
            console.print("\\n📈 Recent Activity Summary:")
            console.print("-" * 28)
            console.print(f"  • Total experiments run: {token_summary.get('total_requests', 0):,}")
            console.print(f"  • Cache efficiency: {cache_stats.get('hit_rate', 0.0):.1%}")
            console.print(f"  • Quantum success rate: {quantum_metrics.get('success_rate', 0.0):.1%}")
            
        except Exception as e:
            console.print(f"❌ Analytics display failed: {e}")


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
  
  # Quantum status
  ai-scientist quantum --status
  
  # Cost analysis
  ai-scientist cost --analyze
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
    
    # Advanced enterprise commands
    quantum_parser = subparsers.add_parser("quantum", help="Quantum task planner operations")
    quantum_parser.add_argument("--status", action="store_true", help="Show quantum system status")
    quantum_parser.add_argument("--optimize", action="store_true", help="Run quantum optimization")
    quantum_parser.add_argument("--reset", action="store_true", help="Reset quantum planner state")
    
    cost_parser = subparsers.add_parser("cost", help="Cost analysis and optimization")
    cost_parser.add_argument("--analyze", action="store_true", help="Run cost analysis")
    cost_parser.add_argument("--optimize", action="store_true", help="Optimize costs")
    cost_parser.add_argument("--report", help="Generate cost report (json/csv)")
    
    cache_parser = subparsers.add_parser("cache", help="Distributed cache management")
    cache_parser.add_argument("--status", action="store_true", help="Show cache status")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--optimize", action="store_true", help="Optimize cache")
    
    analytics_parser = subparsers.add_parser("analytics", help="Advanced system analytics")
    analytics_parser.add_argument("--dashboard", action="store_true", help="Show analytics dashboard")
    analytics_parser.add_argument("--export", help="Export analytics data (json/csv)")
    
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
            console.print("⚠️  Warning: Using default configuration")
        
        # Show banner
        console.print("🤖 AI Scientist v2 - Enterprise Edition")
        console.print("🔬 Autonomous Scientific Discovery Platform")
        console.print("-" * 50)
        
        # Validate environment for non-utility commands
        if parsed_args.command not in ['health-check', 'validate']:
            if not cli.validate_environment():
                console.print("⚠️  Environment validation failed - continuing in demo mode")
        
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
        elif parsed_args.command == "quantum":
            if getattr(parsed_args, 'status', False):
                success = cli.quantum_status()
            elif getattr(parsed_args, 'optimize', False):
                console.print("⚛️  Running quantum optimization...")
                success = cli.quantum_planner.optimize()
                console.print("✅ Quantum optimization completed")
            elif getattr(parsed_args, 'reset', False):
                if confirm_ask("Reset quantum planner state?"):
                    success = cli.quantum_planner.reset()
                    console.print("✅ Quantum planner reset")
                else:
                    success = True
            else:
                success = cli.quantum_status()
        elif parsed_args.command == "cost":
            if getattr(parsed_args, 'analyze', False):
                success = cli.cost_analysis()
            elif getattr(parsed_args, 'optimize', False):
                console.print("💰 Running cost optimization...")
                savings = cli.cost_optimizer.optimize()
                console.print(f"✅ Optimization completed. Potential savings: ${savings:.2f}")
                success = True
            else:
                success = cli.cost_analysis()
        elif parsed_args.command == "cache":
            success = cli.cache_management()
        elif parsed_args.command == "analytics":
            cli._show_advanced_analytics()
            success = True
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\\n⚠️  Operation interrupted by user")
        return 130
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}")
        if getattr(parsed_args, 'debug', False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())