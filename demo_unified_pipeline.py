#!/usr/bin/env python3
"""
Demo script for the Unified Research Pipeline
Shows the core capabilities and immediate value of the AI Scientist v2 system.
"""

import os
import shutil
from pathlib import Path
from unified_research_pipeline import UnifiedResearchPipeline, PipelineConfig


def demo_complete_pipeline():
    """Demonstrate the complete research pipeline from topic to paper."""
    
    print("=" * 70)
    print("AI SCIENTIST v2 - UNIFIED RESEARCH PIPELINE DEMO")
    print("=" * 70)
    print()
    print("This demonstrates the core value proposition:")
    print("  🎯 Input:  Research topic")
    print("  🔄 Process: Automated idea generation → experiments → paper")
    print("  📄 Output: Complete scientific paper")
    print()
    
    # Setup demo output directory
    demo_dir = Path("demo_research_output")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    
    # Demo 1: Complete pipeline
    print("🚀 DEMO 1: Complete Research Pipeline")
    print("-" * 50)
    print("Topic: 'Graph Neural Networks for Time Series Forecasting'")
    print()
    
    config = PipelineConfig(
        topic="Graph Neural Networks for Time Series Forecasting",
        output_dir=demo_dir / "complete_pipeline",
        model="gpt-4o-2024-11-20",
        big_model="o1-2024-12-17",
        max_ideas=2,
        num_reflections=3,
        verbose=False
    )
    
    pipeline = UnifiedResearchPipeline(config)
    results = pipeline.run_pipeline()
    
    print()
    print("✅ Complete pipeline finished!")
    print(f"   Session: {results['session_id']}")
    print(f"   Ideas:   {results['final_outputs']['ideas_count']}")
    print(f"   Paper:   {Path(results['final_outputs']['paper_path']).name}")
    print(f"   Time:    {calculate_duration(results)}")
    print()
    
    # Demo 2: Modular usage
    print("🔧 DEMO 2: Modular Pipeline Usage")
    print("-" * 50)
    print("Step 1: Generate ideas only")
    
    config2 = PipelineConfig(
        topic="Federated Learning Privacy Mechanisms",
        output_dir=demo_dir / "ideas_only",
        max_ideas=3,
        skip_experiments=True,
        skip_writeup=True,
        verbose=False
    )
    
    pipeline2 = UnifiedResearchPipeline(config2)
    results2 = pipeline2.run_pipeline()
    
    print(f"✅ Generated {results2['final_outputs']['ideas_count']} research ideas")
    ideas_file = Path(results2['final_outputs']['session_directory']) / "ideation" / "ideas.json"
    
    print("Step 2: Continue with selected ideas")
    
    config3 = PipelineConfig(
        topic="",  # Not needed when using existing ideas
        output_dir=demo_dir / "continue_from_ideas", 
        skip_ideation=True,
        idea_file=ideas_file,
        verbose=False
    )
    
    pipeline3 = UnifiedResearchPipeline(config3)
    results3 = pipeline3.run_pipeline()
    
    print(f"✅ Completed experiments and paper generation")
    print(f"   Paper: {Path(results3['final_outputs']['paper_path']).name}")
    print()
    
    # Demo 3: Different research areas
    print("🌟 DEMO 3: Multiple Research Areas")
    print("-" * 50)
    
    research_areas = [
        "Quantum Machine Learning Algorithms",
        "Efficient Vision Transformers", 
        "Causal Inference in Deep Learning"
    ]
    
    for i, topic in enumerate(research_areas, 1):
        print(f"Area {i}: {topic}")
        
        config = PipelineConfig(
            topic=topic,
            output_dir=demo_dir / f"research_area_{i}",
            max_ideas=1,
            num_reflections=2,
            verbose=False
        )
        
        pipeline = UnifiedResearchPipeline(config)
        results = pipeline.run_pipeline()
        
        print(f"   ✅ Generated paper: {Path(results['final_outputs']['paper_path']).name}")
    
    print()
    
    # Show final results
    print("📊 DEMO SUMMARY")
    print("-" * 50)
    print(f"Output Directory: {demo_dir.absolute()}")
    print("Generated:")
    
    total_papers = 0
    total_ideas = 0
    
    for session_dir in demo_dir.rglob("session_*"):
        if (session_dir / "pipeline_results.json").exists():
            import json
            with open(session_dir / "pipeline_results.json") as f:
                session_results = json.load(f)
                if session_results.get("status") == "completed":
                    total_ideas += session_results["final_outputs"]["ideas_count"]
                    if session_results["final_outputs"].get("paper_path"):
                        total_papers += 1
    
    print(f"  📝 {total_ideas} Research Ideas")
    print(f"  📄 {total_papers} Research Papers") 
    print(f"  🗂️  {len(list(demo_dir.glob('**/session_*')))} Research Sessions")
    print()
    
    print("🎉 DEMONSTRATION COMPLETE!")
    print()
    print("Key Benefits Demonstrated:")
    print("  ✅ Immediate value - topic to paper in minutes")
    print("  ✅ Flexible workflow - run complete or modular")
    print("  ✅ Multiple domains - works across research areas")
    print("  ✅ Production ready - proper logging and state management")
    print("  ✅ Clear integration - uses existing AI Scientist components")
    print()
    print("Next Steps:")
    print("  🔗 Install full dependencies for real LLM integration")
    print("  🎛️  Configure API keys for your preferred models")
    print("  🚀 Run on your actual research topics")
    print("  📈 Scale with distributed computing capabilities")
    print()
    print(f"📁 Explore outputs in: {demo_dir.absolute()}")
    print("=" * 70)


def calculate_duration(results):
    """Calculate duration from start to end time."""
    from datetime import datetime
    
    try:
        start = datetime.fromisoformat(results["start_time"])
        end = datetime.fromisoformat(results["end_time"])
        duration = end - start
        return f"{duration.total_seconds():.1f}s"
    except:
        return "N/A"


if __name__ == "__main__":
    demo_complete_pipeline()