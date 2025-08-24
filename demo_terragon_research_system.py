#!/usr/bin/env python3
"""
TERRAGON RESEARCH SYSTEM DEMO

Comprehensive demonstration of the autonomous research system capabilities.
"""

import os
import sys
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terragon_autonomous_research_engine import (
    ResearchConfig,
    AutonomousResearchEngine,
    load_research_config
)
from terragon_research_cli import ResearchCLI


class ResearchSystemDemo:
    """Comprehensive demo of the research system."""
    
    def __init__(self):
        self.demo_dir = Path("terragon_demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
    def print_header(self, title: str) -> None:
        """Print demo section header."""
        print("\n" + "="*80)
        print(f"🎯 {title}")
        print("="*80)
    
    def print_step(self, step: str) -> None:
        """Print demo step."""
        print(f"\n📋 {step}")
        print("-" * 60)
    
    async def run_full_demo(self) -> None:
        """Run complete demonstration."""
        self.print_header("TERRAGON AUTONOMOUS RESEARCH SYSTEM DEMO")
        
        print("""
This demonstration showcases the autonomous research system capabilities:
1. Configuration management and templates
2. Research topic creation
3. CLI interface functionality  
4. Autonomous research pipeline simulation
5. Results analysis and reporting
        """)
        
        # Demo 1: Configuration Management
        await self._demo_configuration_management()
        
        # Demo 2: Topic Creation
        await self._demo_topic_creation()
        
        # Demo 3: CLI Interface
        await self._demo_cli_interface()
        
        # Demo 4: Pipeline Simulation
        await self._demo_pipeline_simulation()
        
        # Demo 5: Results Analysis
        await self._demo_results_analysis()
        
        self.print_header("DEMO COMPLETED SUCCESSFULLY!")
        print(f"📁 Demo files created in: {self.demo_dir}")
        print("🎉 Terragon Autonomous Research System is ready for use!")
    
    async def _demo_configuration_management(self) -> None:
        """Demo configuration management."""
        self.print_header("1. CONFIGURATION MANAGEMENT")
        
        self.print_step("Creating different configuration templates")
        
        cli = ResearchCLI()
        
        # Create basic config
        basic_config_path = self.demo_dir / "basic_config.yaml"
        basic_content = cli._get_config_template("basic")
        with open(basic_config_path, 'w') as f:
            f.write(basic_content)
        print(f"✅ Created basic configuration: {basic_config_path}")
        
        # Create advanced config
        advanced_config_path = self.demo_dir / "advanced_config.yaml"
        advanced_content = cli._get_config_template("advanced")
        with open(advanced_config_path, 'w') as f:
            f.write(advanced_content)
        print(f"✅ Created advanced configuration: {advanced_config_path}")
        
        # Create production config
        production_config_path = self.demo_dir / "production_config.yaml"
        production_content = cli._get_config_template("production")
        with open(production_config_path, 'w') as f:
            f.write(production_content)
        print(f"✅ Created production configuration: {production_config_path}")
        
        self.print_step("Loading and validating configurations")
        
        # Load configurations
        try:
            basic_config = await load_research_config(str(basic_config_path))
            print(f"✅ Basic config loaded - Max ideas: {basic_config.max_ideas}")
            
            advanced_config = await load_research_config(str(advanced_config_path))
            print(f"✅ Advanced config loaded - Max ideas: {advanced_config.max_ideas}")
            
            # Show configuration differences
            print("\n📊 Configuration Comparison:")
            print(f"Basic    - Ideas: {basic_config.max_ideas}, Workers: {basic_config.num_workers}")
            print(f"Advanced - Ideas: {advanced_config.max_ideas}, Workers: {advanced_config.num_workers}")
            
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
    
    async def _demo_topic_creation(self) -> None:
        """Demo research topic creation."""
        self.print_header("2. RESEARCH TOPIC CREATION")
        
        self.print_step("Creating different research topic templates")
        
        cli = ResearchCLI()
        
        topics = {
            "basic": "basic_research_topic.md",
            "nlp": "nlp_research_topic.md", 
            "cv": "computer_vision_topic.md",
            "ml": "machine_learning_topic.md",
            "advanced": "advanced_research_topic.md"
        }
        
        for template_type, filename in topics.items():
            topic_path = self.demo_dir / filename
            topic_content = cli._get_topic_template(template_type)
            
            with open(topic_path, 'w') as f:
                f.write(topic_content)
            
            # Show preview
            lines = topic_content.split('\\n')[:8]  # First 8 lines
            preview = '\\n'.join(lines)
            
            print(f"✅ Created {template_type} topic: {topic_path}")
            print(f"   Preview: {preview[:100]}...")
        
        self.print_step("Validating topic files")
        
        for template_type, filename in topics.items():
            topic_path = self.demo_dir / filename
            if topic_path.exists():
                size = topic_path.stat().st_size
                print(f"✅ {filename}: {size} bytes")
    
    async def _demo_cli_interface(self) -> None:
        """Demo CLI interface functionality."""
        self.print_header("3. CLI INTERFACE DEMONSTRATION")
        
        self.print_step("Testing CLI argument parsing")
        
        cli = ResearchCLI()
        
        # Test different command parsing
        test_commands = [
            ['run', '--quick-start'],
            ['run', '--topic', 'test.md', '--max-ideas', '5'],
            ['create-topic', '--output', 'my_topic.md', '--template', 'nlp'],
            ['status', '--list-sessions'],
            ['create-config', '--template', 'advanced']
        ]
        
        for cmd in test_commands:
            try:
                args = cli.parser.parse_args(cmd)
                print(f"✅ Command parsed: {' '.join(cmd)}")
                print(f"   → {args.command} with args: {vars(args)}")
            except SystemExit:
                print(f"❌ Command parsing failed: {' '.join(cmd)}")
        
        self.print_step("Testing help generation")
        
        try:
            # Capture help output
            import io
            import contextlib
            
            help_output = io.StringIO()
            with contextlib.redirect_stdout(help_output):
                try:
                    cli.parser.parse_args(['--help'])
                except SystemExit:
                    pass  # Expected for help
            
            help_text = help_output.getvalue()
            print(f"✅ Help system working ({len(help_text)} characters)")
            print(f"   Help includes: {'Examples:' in help_text}")
            
        except Exception as e:
            print(f"❌ Help generation failed: {e}")
    
    async def _demo_pipeline_simulation(self) -> None:
        """Demo autonomous research pipeline simulation."""
        self.print_header("4. AUTONOMOUS RESEARCH PIPELINE SIMULATION")
        
        self.print_step("Creating test research configuration")
        
        # Create test topic
        test_topic_path = self.demo_dir / "test_research_topic.md"
        test_topic_content = """# Autonomous Research Pipeline Test

## Title
Novel Techniques for Automated Research Discovery

## Keywords
automation, research, artificial intelligence, machine learning

## TL;DR
Testing the autonomous research pipeline with simulated components.

## Abstract
This research topic serves as a test case for the autonomous research system.
We investigate the effectiveness of automated research discovery and validation
through systematic experimentation and evaluation. The system demonstrates
end-to-end research capabilities including ideation, experimentation, and
paper generation.

## Research Objectives
1. Validate autonomous research pipeline functionality
2. Demonstrate end-to-end research automation
3. Test system reliability and error handling
4. Evaluate result quality and completeness

## Expected Contributions
- Proof of concept for autonomous research
- Comprehensive system validation
- Performance benchmarking results
- Documentation and best practices
"""
        
        with open(test_topic_path, 'w') as f:
            f.write(test_topic_content)
        
        print(f"✅ Created test topic: {test_topic_path}")
        
        self.print_step("Configuring test research pipeline")
        
        # Create minimal test configuration
        test_config = ResearchConfig(
            topic_description_path=str(test_topic_path),
            output_directory=str(self.demo_dir / "test_research_output"),
            max_ideas=2,
            idea_reflections=1,
            skip_writeup=True,  # Skip for demo
            skip_review=True,   # Skip for demo
            num_workers=1,
            max_steps=5
        )
        
        print(f"✅ Configuration created:")
        print(f"   Topic: {test_config.topic_description_path}")
        print(f"   Max ideas: {test_config.max_ideas}")
        print(f"   Output: {test_config.output_directory}")
        
        self.print_step("Initializing research engine")
        
        engine = AutonomousResearchEngine(test_config)
        
        print(f"✅ Engine initialized:")
        print(f"   Session ID: {engine.session_id}")
        print(f"   Output directory: {engine.output_dir}")
        print(f"   Status: {engine.results['status']}")
        
        self.print_step("Simulating ideation phase")
        
        # Simulate ideation
        ideation_args = {
            "workshop_file": str(test_topic_path),
            "model": test_config.ideation_model,
            "max_num_generations": test_config.max_ideas,
            "num_reflections": test_config.idea_reflections,
            "output_dir": str(engine.output_dir / "ideation")
        }
        
        try:
            ideas = await engine._generate_research_ideas(ideation_args)
            print(f"✅ Generated {len(ideas)} research ideas:")
            
            for i, idea in enumerate(ideas):
                print(f"   {i+1}. {idea['Title']}")
                print(f"      Interestingness: {idea['Interestingness']}/10")
                print(f"      Feasibility: {idea['Feasibility']}/10")
            
        except Exception as e:
            print(f"❌ Ideation simulation failed: {e}")
            ideas = []
        
        self.print_step("Simulating BFTS configuration")
        
        if ideas:
            try:
                # Create test experiment directory
                experiment_dir = engine.output_dir / "experiments" / f"idea_0_{ideas[0]['Name']}"
                experiment_dir.mkdir(parents=True, exist_ok=True)
                
                # Create idea JSON
                idea_json_path = experiment_dir / "idea.json"
                with open(idea_json_path, 'w') as f:
                    json.dump(ideas[0], f, indent=2)
                
                # Create BFTS config
                bfts_config_path = await engine._create_bfts_config(experiment_dir, idea_json_path)
                
                print(f"✅ BFTS configuration created: {bfts_config_path}")
                
                # Show config content
                with open(bfts_config_path, 'r') as f:
                    config_lines = f.readlines()[:10]  # First 10 lines
                print(f"   Configuration preview: {len(config_lines)} lines")
                
            except Exception as e:
                print(f"❌ BFTS configuration failed: {e}")
        
        # Save simulated results
        await engine._save_session_results()
        print(f"✅ Simulation results saved")
    
    async def _demo_results_analysis(self) -> None:
        """Demo results analysis functionality."""
        self.print_header("5. RESULTS ANALYSIS")
        
        self.print_step("Creating sample results data")
        
        # Create sample research results
        sample_results = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": {
                "topic_description_path": "demo_topic.md",
                "max_ideas": 3,
                "writeup_type": "icbinb",
                "ideation_model": "gpt-4o-2024-05-13"
            },
            "phases": {
                "ideation": {
                    "status": "completed",
                    "ideas_generated": 3,
                    "timestamp": datetime.now().isoformat()
                },
                "experimentation": {
                    "status": "completed", 
                    "experiments_completed": 3,
                    "timestamp": datetime.now().isoformat()
                },
                "writeup": {
                    "status": "completed",
                    "papers_generated": 2,
                    "timestamp": datetime.now().isoformat()
                }
            },
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "token_usage": {
                "gpt-4o": {"total_cost": 15.75},
                "claude-3-5-sonnet": {"total_cost": 28.50},
                "o1-preview": {"total_cost": 45.25}
            }
        }
        
        # Save sample results
        results_path = self.demo_dir / f"sample_results_{sample_results['session_id']}.json"
        with open(results_path, 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        print(f"✅ Sample results created: {results_path}")
        
        self.print_step("Analyzing results data")
        
        # Analyze the results
        total_phases = len(sample_results["phases"])
        completed_phases = sum(1 for phase in sample_results["phases"].values() 
                             if phase["status"] == "completed")
        
        total_cost = sum(usage.get("total_cost", 0) 
                        for usage in sample_results["token_usage"].values())
        
        print(f"📊 Results Analysis:")
        print(f"   Session Status: {sample_results['status']}")
        print(f"   Phases Completed: {completed_phases}/{total_phases}")
        print(f"   Total Token Cost: ${total_cost:.2f}")
        print(f"   Ideas Generated: {sample_results['phases']['ideation']['ideas_generated']}")
        print(f"   Papers Created: {sample_results['phases']['writeup']['papers_generated']}")
        
        self.print_step("Generating summary report")
        
        # Create summary report
        summary_report = f"""# Research Session Summary Report

## Session Information
- **Session ID**: {sample_results['session_id']}
- **Status**: {sample_results['status']}
- **Start Time**: {sample_results['start_time']}
- **End Time**: {sample_results['end_time']}

## Configuration
- **Topic**: {sample_results['config']['topic_description_path']}
- **Max Ideas**: {sample_results['config']['max_ideas']}
- **Writeup Type**: {sample_results['config']['writeup_type']}
- **Model**: {sample_results['config']['ideation_model']}

## Results Summary
- **Phases Completed**: {completed_phases}/{total_phases}
- **Ideas Generated**: {sample_results['phases']['ideation']['ideas_generated']}
- **Experiments Completed**: {sample_results['phases']['experimentation']['experiments_completed']}
- **Papers Generated**: {sample_results['phases']['writeup']['papers_generated']}

## Cost Analysis
- **Total Cost**: ${total_cost:.2f}
- **GPT-4o**: ${sample_results['token_usage']['gpt-4o']['total_cost']:.2f}
- **Claude-3.5-Sonnet**: ${sample_results['token_usage']['claude-3-5-sonnet']['total_cost']:.2f}
- **O1-Preview**: ${sample_results['token_usage']['o1-preview']['total_cost']:.2f}

## Recommendations
1. Research pipeline completed successfully
2. Cost efficiency within acceptable range
3. High completion rate achieved
4. System ready for production use
"""
        
        report_path = self.demo_dir / f"summary_report_{sample_results['session_id']}.md"
        with open(report_path, 'w') as f:
            f.write(summary_report)
        
        print(f"✅ Summary report generated: {report_path}")
        print(f"   Report size: {len(summary_report)} characters")
    
    def create_demo_readme(self) -> None:
        """Create demo README file."""
        readme_content = f"""# Terragon Autonomous Research System Demo

This directory contains demonstration files for the Terragon Autonomous Research System.

## Demo Components

### Configuration Files
- `basic_config.yaml` - Basic research configuration
- `advanced_config.yaml` - Advanced configuration with more features
- `production_config.yaml` - Production-ready configuration

### Research Topics
- `basic_research_topic.md` - Basic research topic template
- `nlp_research_topic.md` - Natural Language Processing topic
- `computer_vision_topic.md` - Computer Vision research topic
- `machine_learning_topic.md` - Machine Learning topic
- `advanced_research_topic.md` - Advanced topic with detailed structure

### Test Files
- `test_research_topic.md` - Test topic for pipeline simulation
- `sample_results_*.json` - Sample research results
- `summary_report_*.md` - Generated summary reports

## Running the Demo

To run the complete demo:

```bash
python demo_terragon_research_system.py
```

To run specific components:

```bash
# Test the CLI interface
python terragon_research_cli.py --help

# Create a new research topic
python terragon_research_cli.py create-topic --output my_topic.md --template nlp

# Run a quick research simulation
python terragon_research_cli.py run --quick-start
```

## Demo Output

The demo creates:
1. ✅ Configuration templates (3 types)
2. ✅ Research topic templates (5 types)  
3. ✅ CLI interface validation
4. ✅ Pipeline simulation results
5. ✅ Analysis reports and summaries

## Next Steps

After running the demo:
1. Edit research topic files with your specific research ideas
2. Adjust configuration files for your requirements
3. Set up API keys for LLM services
4. Run the full autonomous research pipeline

## Files Created

Generated on: {datetime.now().isoformat()}
Total demo files: [Will be updated after demo completion]
"""
        
        readme_path = self.demo_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Demo README created: {readme_path}")


async def main():
    """Main demo execution function."""
    print("🎯 Starting Terragon Research System Demo...")
    
    demo = ResearchSystemDemo()
    
    try:
        await demo.run_full_demo()
        demo.create_demo_readme()
        
        print("\n" + "🎉" * 20)
        print("TERRAGON AUTONOMOUS RESEARCH SYSTEM")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        # Count demo files
        demo_files = list(demo.demo_dir.glob("*"))
        print(f"\n📊 Demo Statistics:")
        print(f"   Files created: {len(demo_files)}")
        print(f"   Directory size: {sum(f.stat().st_size for f in demo_files if f.is_file())} bytes")
        print(f"   Demo location: {demo.demo_dir}")
        
        print(f"\n🚀 Ready to start autonomous research!")
        print(f"💡 Next: python terragon_research_cli.py run --quick-start")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())