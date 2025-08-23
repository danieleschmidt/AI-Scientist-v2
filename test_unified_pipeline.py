#!/usr/bin/env python3
"""
Test script for the Unified Research Pipeline
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unified_research_pipeline import UnifiedResearchPipeline, PipelineConfig


def test_pipeline_demo_mode():
    """Test the pipeline in demo mode (no external dependencies)"""
    
    print("Testing Unified Research Pipeline in Demo Mode...")
    
    # Create temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_output"
        
        # Create configuration
        config = PipelineConfig(
            topic="test machine learning topic",
            output_dir=output_dir,
            model="gpt-4o-2024-11-20",
            big_model="o1-2024-12-17",
            max_ideas=2,
            num_reflections=3,
            skip_ideation=False,
            skip_experiments=False,
            skip_writeup=False,
            verbose=True
        )
        
        # Create pipeline
        pipeline = UnifiedResearchPipeline(config)
        
        # Run pipeline
        results = pipeline.run_pipeline()
        
        # Verify results
        assert results["status"] == "completed", f"Pipeline failed: {results.get('final_error')}"
        assert results["final_outputs"]["ideas_count"] == 2, "Should generate 2 ideas"
        assert "paper_path" in results["final_outputs"], "Should generate paper path"
        
        # Check directory structure
        session_dir = Path(results["final_outputs"]["session_directory"])
        assert session_dir.exists(), "Session directory should exist"
        assert (session_dir / "ideation" / "ideas.json").exists(), "Ideas file should exist"
        assert (session_dir / "experiments" / "research_idea.md").exists(), "Research idea should exist"
        assert (session_dir / "writeup").exists(), "Writeup directory should exist"
        assert (session_dir / "pipeline_results.json").exists(), "Results file should exist"
        
        # Verify ideas file format
        with open(session_dir / "ideation" / "ideas.json") as f:
            ideas = json.load(f)
            assert len(ideas) == 2, "Should have 2 ideas"
            for idea in ideas:
                assert "Name" in idea, "Idea should have Name"
                assert "Title" in idea, "Idea should have Title"
                assert "Short Hypothesis" in idea, "Idea should have Hypothesis"
                assert "Abstract" in idea, "Idea should have Abstract"
        
        # Verify pipeline results format
        with open(session_dir / "pipeline_results.json") as f:
            pipeline_results = json.load(f)
            assert "session_id" in pipeline_results, "Should have session ID"
            assert "config" in pipeline_results, "Should have config"
            assert "steps" in pipeline_results, "Should have steps"
            assert "ideation" in pipeline_results["steps"], "Should have ideation step"
            assert "experiments" in pipeline_results["steps"], "Should have experiments step"
            assert "writeup" in pipeline_results["steps"], "Should have writeup step"
        
        print(f"✓ Pipeline completed successfully!")
        print(f"  Session ID: {results['session_id']}")
        print(f"  Ideas Generated: {results['final_outputs']['ideas_count']}")
        print(f"  Status: {results['status']}")
        print(f"  Output Directory: {results['final_outputs']['session_directory']}")
        
        return True


def test_pipeline_skip_options():
    """Test pipeline with various skip options"""
    
    print("\nTesting Pipeline Skip Options...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Skip experiments
        print("  Testing --skip-experiments...")
        output_dir = Path(temp_dir) / "test_skip_exp"
        config = PipelineConfig(
            topic="test topic",
            output_dir=output_dir,
            max_ideas=1,
            skip_experiments=True,
            verbose=False
        )
        pipeline = UnifiedResearchPipeline(config)
        results = pipeline.run_pipeline()
        assert results["status"] == "completed"
        assert results["final_outputs"]["experiment_results"]["status"] == "skipped"
        print("    ✓ Skip experiments works")
        
        # Test 2: Skip writeup
        print("  Testing --skip-writeup...")
        output_dir = Path(temp_dir) / "test_skip_writeup"
        config = PipelineConfig(
            topic="test topic",
            output_dir=output_dir,
            max_ideas=1,
            skip_writeup=True,
            verbose=False
        )
        pipeline = UnifiedResearchPipeline(config)
        results = pipeline.run_pipeline()
        assert results["status"] == "completed"
        assert results["final_outputs"]["paper_path"] == ""
        print("    ✓ Skip writeup works")
        
        # Test 3: Use existing ideas file
        print("  Testing --idea-file with --skip-ideation...")
        
        # First create ideas file
        ideas_file = Path(temp_dir) / "test_ideas.json"
        sample_ideas = [
            {
                "Name": "test_idea",
                "Title": "Test Research Idea",
                "Short Hypothesis": "Test hypothesis",
                "Abstract": "Test abstract",
                "Experiments": "Test experiments",
                "Related Work": "Test related work",
                "Risk Factors and Limitations": "Test risks"
            }
        ]
        with open(ideas_file, "w") as f:
            json.dump(sample_ideas, f, indent=2)
        
        # Run pipeline with existing ideas
        output_dir = Path(temp_dir) / "test_existing_ideas"
        config = PipelineConfig(
            topic="",  # Not used when skip_ideation=True
            output_dir=output_dir,
            skip_ideation=True,
            idea_file=ideas_file,
            verbose=False
        )
        pipeline = UnifiedResearchPipeline(config)
        results = pipeline.run_pipeline()
        assert results["status"] == "completed"
        assert results["final_outputs"]["ideas_count"] == 1
        print("    ✓ Existing ideas file works")
        
        print("  ✓ All skip options work correctly")


def test_error_handling():
    """Test pipeline error handling"""
    
    print("\nTesting Error Handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with non-existent idea file
        print("  Testing non-existent idea file...")
        output_dir = Path(temp_dir) / "test_error"
        config = PipelineConfig(
            topic="",
            output_dir=output_dir,
            skip_ideation=True,
            idea_file=Path("non_existent_file.json"),
            verbose=False
        )
        pipeline = UnifiedResearchPipeline(config)
        
        try:
            results = pipeline.run_pipeline()
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "non_existent_file.json" in str(e) or "No such file" in str(e)
            print("    ✓ Correctly handles missing idea file")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING UNIFIED RESEARCH PIPELINE")
    print("=" * 60)
    
    try:
        test_pipeline_demo_mode()
        test_pipeline_skip_options() 
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("The Unified Research Pipeline is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)