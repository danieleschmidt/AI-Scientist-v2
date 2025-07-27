"""
End-to-end integration tests for the research pipeline.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

from ai_scientist.perform_ideation_temp_free import generate_research_ideas
from ai_scientist.launch_scientist_bfts import run_experiment_pipeline
from ai_scientist.perform_writeup import generate_paper


@pytest.mark.integration
@pytest.mark.slow
class TestE2EResearchPipeline:
    """End-to-end tests for the complete research pipeline."""
    
    @pytest.fixture
    def research_topic_file(self, temp_dir):
        """Create a test research topic file."""
        topic_content = """# Test Research Topic

## Title
Automated Testing Strategies for AI Research Systems

## Keywords
artificial intelligence, automated testing, research methodology, validation

## TL;DR
Develop comprehensive testing frameworks for AI research automation systems to ensure reliability, reproducibility, and scientific validity.

## Abstract
This research focuses on developing robust testing methodologies for automated AI research systems. We aim to create frameworks that can validate research outputs, ensure reproducibility, and maintain scientific rigor in automated discovery processes. The work addresses critical challenges in verification of AI-generated research including hypothesis validation, experimental design verification, and result interpretation accuracy.

## Research Questions
1. How can we ensure the reliability of AI-generated research hypotheses?
2. What testing frameworks are most effective for validating experimental designs?
3. How do we measure the scientific validity of automated research outputs?

## Methodology
- Systematic analysis of existing testing frameworks
- Development of domain-specific validation metrics
- Implementation of automated verification systems
- Comparative evaluation across different research domains

## Expected Contributions
- Novel testing framework for AI research systems
- Metrics for measuring research quality and validity
- Guidelines for implementing verification in automated research
- Open-source tools for research validation
"""
        
        topic_file = temp_dir / "test_research_topic.md"
        topic_file.write_text(topic_content)
        return topic_file
    
    @pytest.fixture
    def mock_bfts_config(self, temp_dir):
        """Create a test BFTS configuration."""
        config = {
            "agent": {
                "num_workers": 2,
                "steps": 10,
                "num_seeds": 2,
                "k_fold_validation": False,
                "expose_prediction": False,
                "data_preview": False
            },
            "search": {
                "max_debug_depth": 3,
                "debug_prob": 0.5,
                "num_drafts": 2
            },
            "models": {
                "experimentation": "gpt-4",
                "writeup": "gpt-4",
                "review": "gpt-4",
                "citation": "gpt-4"
            },
            "timeouts": {
                "experiment": 300,
                "writeup": 600,
                "review": 300
            }
        }
        
        config_file = temp_dir / "test_bfts_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        return config_file
    
    @patch('ai_scientist.llm.get_llm_response')
    def test_ideation_pipeline(self, mock_llm, research_topic_file, temp_dir):
        """Test the research ideation pipeline."""
        # Mock LLM responses for ideation
        mock_responses = [
            {
                "title": "Test Idea 1",
                "abstract": "First test research idea",
                "hypothesis": "Testing hypothesis 1",
                "methodology": "Test methodology 1",
                "novelty_score": 0.8,
                "feasibility_score": 0.9
            },
            {
                "title": "Test Idea 2", 
                "abstract": "Second test research idea",
                "hypothesis": "Testing hypothesis 2",
                "methodology": "Test methodology 2",
                "novelty_score": 0.7,
                "feasibility_score": 0.8
            }
        ]
        
        mock_llm.side_effect = [json.dumps(idea) for idea in mock_responses]
        
        # Mock Semantic Scholar API
        with patch('ai_scientist.tools.semantic_scholar.search_papers') as mock_search:
            mock_search.return_value = {
                "data": [],
                "total": 0
            }
            
            # Run ideation
            output_file = temp_dir / "generated_ideas.json"
            
            result = generate_research_ideas(
                workshop_file=str(research_topic_file),
                output_file=str(output_file),
                model="gpt-4",
                max_num_generations=2,
                num_reflections=1
            )
            
            # Verify output
            assert output_file.exists()
            with open(output_file, 'r') as f:
                ideas = json.load(f)
            
            assert len(ideas) == 2
            assert ideas[0]["title"] == "Test Idea 1"
            assert ideas[1]["title"] == "Test Idea 2"
            assert all("novelty_score" in idea for idea in ideas)
    
    @patch('ai_scientist.treesearch.parallel_agent.ParallelAgent')
    @patch('ai_scientist.llm.get_llm_response')
    def test_experiment_pipeline(self, mock_llm, mock_agent, temp_dir, mock_bfts_config):
        """Test the experiment execution pipeline."""
        # Create mock ideas file
        ideas = [
            {
                "title": "Test Experiment",
                "abstract": "Test experiment for pipeline validation",
                "hypothesis": "Our test will succeed",
                "methodology": "Run automated tests",
                "code_snippet": "print('Hello, Test!')",
                "expected_results": "Successful execution"
            }
        ]
        
        ideas_file = temp_dir / "test_ideas.json"
        with open(ideas_file, 'w') as f:
            json.dump(ideas, f)
        
        # Mock agent responses
        mock_agent_instance = Mock()
        mock_agent_instance.run.return_value = {
            "success": True,
            "results": {
                "experiment_output": "Test completed successfully",
                "metrics": {"accuracy": 0.95, "precision": 0.88},
                "plots": ["test_plot.png"],
                "logs": ["Experiment started", "Test passed", "Experiment completed"]
            },
            "execution_time": 45.2,
            "resource_usage": {
                "memory_mb": 512,
                "cpu_percent": 25.5
            }
        }
        mock_agent.return_value = mock_agent_instance
        
        # Mock LLM for result analysis
        mock_llm.return_value = json.dumps({
            "analysis": "The experiment completed successfully with good performance metrics.",
            "insights": ["Test framework works correctly", "Performance is acceptable"],
            "next_steps": ["Expand test coverage", "Optimize performance"]
        })
        
        # Run experiment pipeline
        results = run_experiment_pipeline(
            ideas_file=str(ideas_file),
            config_file=str(mock_bfts_config),
            output_dir=str(temp_dir / "experiments")
        )
        
        # Verify results
        assert results["success"] is True
        assert "experiment_output" in results["results"]
        assert results["results"]["metrics"]["accuracy"] == 0.95
        assert mock_agent_instance.run.called
    
    @patch('ai_scientist.llm.get_llm_response')
    @patch('ai_scientist.tools.semantic_scholar.search_papers')
    def test_paper_generation_pipeline(self, mock_search, mock_llm, temp_dir):
        """Test the paper generation pipeline."""
        # Create mock experiment results
        experiment_results = {
            "title": "Test Paper Generation",
            "abstract": "Testing automated paper generation",
            "methodology": "Automated testing approach",
            "results": {
                "key_findings": ["Finding 1", "Finding 2"],
                "metrics": {"accuracy": 0.92, "f1_score": 0.89},
                "figures": ["figure1.png", "figure2.png"]
            },
            "analysis": "The results demonstrate effective performance",
            "conclusions": "Our approach shows promise for automated research"
        }
        
        results_file = temp_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f)
        
        # Mock literature search
        mock_search.return_value = {
            "data": [
                {
                    "paperId": "12345",
                    "title": "Related Work 1",
                    "authors": [{"name": "Author 1"}],
                    "year": 2023,
                    "abstract": "Related research abstract",
                    "citationCount": 15,
                    "url": "https://example.com/paper1"
                }
            ],
            "total": 1
        }
        
        # Mock LLM responses for paper sections
        paper_sections = {
            "introduction": "This paper presents a novel approach to automated testing...",
            "related_work": "Previous work in this area includes...",
            "methodology": "Our methodology consists of the following steps...",
            "experiments": "We conducted experiments to validate our approach...",
            "results": "The results demonstrate significant improvements...",
            "discussion": "These findings have important implications...",
            "conclusion": "In conclusion, we have successfully demonstrated..."
        }
        
        mock_llm.side_effect = [
            paper_sections[section] for section in paper_sections.keys()
        ] + ["Generated LaTeX content for the complete paper"]
        
        # Run paper generation
        paper_output = temp_dir / "generated_paper"
        paper_output.mkdir(exist_ok=True)
        
        result = generate_paper(
            results_file=str(results_file),
            output_dir=str(paper_output),
            model_writeup="gpt-4",
            model_citation="gpt-4",
            num_cite_rounds=2
        )
        
        # Verify paper generation
        assert result["success"] is True
        assert "paper_content" in result
        assert len(mock_llm.call_args_list) >= len(paper_sections)
        assert mock_search.called
    
    @patch('ai_scientist.treesearch.parallel_agent.ParallelAgent')
    @patch('ai_scientist.llm.get_llm_response')
    @patch('ai_scientist.tools.semantic_scholar.search_papers')
    def test_full_research_pipeline(self, mock_search, mock_llm, mock_agent, 
                                   research_topic_file, temp_dir, mock_bfts_config):
        """Test the complete research pipeline from idea to paper."""
        # Setup mocks for each stage
        
        # 1. Ideation stage
        ideation_responses = [
            json.dumps({
                "title": "Comprehensive Testing Framework",
                "abstract": "A framework for testing AI research systems",
                "hypothesis": "Automated testing improves research reliability",
                "methodology": "Develop and validate testing metrics",
                "novelty_score": 0.85,
                "feasibility_score": 0.90,
                "code_snippet": "def test_research_system(): pass"
            })
        ]
        
        # 2. Experiment stage
        mock_agent_instance = Mock()
        mock_agent_instance.run.return_value = {
            "success": True,
            "results": {
                "experiment_output": "Framework validation successful",
                "metrics": {"reliability": 0.94, "coverage": 0.87},
                "plots": ["results.png"],
                "code": "# Generated test framework code"
            }
        }
        mock_agent.return_value = mock_agent_instance
        
        # 3. Literature search
        mock_search.return_value = {
            "data": [
                {
                    "paperId": "67890",
                    "title": "Testing Methodologies for AI Systems",
                    "authors": [{"name": "Expert Researcher"}],
                    "year": 2024,
                    "citationCount": 25
                }
            ]
        }
        
        # 4. Paper writing
        paper_content = "\\documentclass{article}\n\\begin{document}\nGenerated research paper content\n\\end{document}"
        
        # Configure LLM responses for all stages
        mock_llm.side_effect = (
            ideation_responses +  # Ideation
            [json.dumps({"analysis": "Experiment successful"})] +  # Experiment analysis
            ["Introduction section...", "Methodology section...", 
             "Results section...", "Conclusion section..."] +  # Paper sections
            [paper_content]  # Final paper
        )
        
        # Run complete pipeline
        pipeline_output = temp_dir / "full_pipeline"
        pipeline_output.mkdir(exist_ok=True)
        
        # Stage 1: Generate ideas
        ideas_file = pipeline_output / "ideas.json"
        generate_research_ideas(
            workshop_file=str(research_topic_file),
            output_file=str(ideas_file),
            model="gpt-4",
            max_num_generations=1,
            num_reflections=1
        )
        
        # Stage 2: Run experiments
        experiments_dir = pipeline_output / "experiments"
        experiment_results = run_experiment_pipeline(
            ideas_file=str(ideas_file),
            config_file=str(mock_bfts_config),
            output_dir=str(experiments_dir)
        )
        
        # Stage 3: Generate paper
        paper_dir = pipeline_output / "paper"
        paper_dir.mkdir(exist_ok=True)
        
        # Create mock results for paper generation
        results_for_paper = {
            "title": "Comprehensive Testing Framework",
            "results": experiment_results["results"],
            "methodology": "Automated testing methodology",
            "analysis": "Comprehensive analysis of results"
        }
        
        results_file = paper_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results_for_paper, f)
        
        paper_result = generate_paper(
            results_file=str(results_file),
            output_dir=str(paper_dir),
            model_writeup="gpt-4",
            model_citation="gpt-4"
        )
        
        # Verify complete pipeline
        assert ideas_file.exists()
        assert experiment_results["success"] is True
        assert paper_result["success"] is True
        
        # Verify all stages were called
        assert mock_llm.call_count >= 6  # Minimum calls across all stages
        assert mock_agent_instance.run.called
        assert mock_search.called
        
        # Check pipeline artifacts
        assert len(list(pipeline_output.rglob("*.json"))) >= 2  # Ideas and results
        assert experiment_results["results"]["metrics"]["reliability"] == 0.94


@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test error handling in the research pipeline."""
    
    @patch('ai_scientist.llm.get_llm_response')
    def test_ideation_with_llm_failure(self, mock_llm, research_topic_file, temp_dir):
        """Test ideation pipeline handles LLM failures gracefully."""
        # Simulate LLM failure
        mock_llm.side_effect = Exception("LLM API timeout")
        
        output_file = temp_dir / "failed_ideas.json"
        
        with pytest.raises(Exception):
            generate_research_ideas(
                workshop_file=str(research_topic_file),
                output_file=str(output_file),
                model="gpt-4",
                max_num_generations=1
            )
        
        # Verify no partial output files are left
        assert not output_file.exists()
    
    @patch('ai_scientist.treesearch.parallel_agent.ParallelAgent')
    def test_experiment_with_execution_failure(self, mock_agent, temp_dir):
        """Test experiment pipeline handles execution failures."""
        # Create mock ideas
        ideas = [{"title": "Failing Test", "code_snippet": "raise Exception('Test failure')"}]
        ideas_file = temp_dir / "failing_ideas.json"
        with open(ideas_file, 'w') as f:
            json.dump(ideas, f)
        
        # Mock agent failure
        mock_agent_instance = Mock()
        mock_agent_instance.run.side_effect = RuntimeError("Experiment execution failed")
        mock_agent.return_value = mock_agent_instance
        
        # Run should handle failure gracefully
        with pytest.raises(RuntimeError):
            run_experiment_pipeline(
                ideas_file=str(ideas_file),
                config_file="dummy_config.yaml",
                output_dir=str(temp_dir / "failed_experiments")
            )
    
    def test_paper_generation_with_missing_results(self, temp_dir):
        """Test paper generation handles missing experiment results."""
        missing_results_file = temp_dir / "nonexistent_results.json"
        paper_output = temp_dir / "paper"
        
        with pytest.raises(FileNotFoundError):
            generate_paper(
                results_file=str(missing_results_file),
                output_dir=str(paper_output),
                model_writeup="gpt-4"
            )


@pytest.mark.integration
@pytest.mark.performance
class TestPipelinePerformance:
    """Test performance characteristics of the research pipeline."""
    
    @pytest.mark.slow
    def test_pipeline_memory_usage(self, temp_dir):
        """Test that pipeline doesn't exceed memory limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run a lightweight version of the pipeline
        with patch('ai_scientist.llm.get_llm_response') as mock_llm:
            mock_llm.return_value = json.dumps({"title": "Memory Test", "abstract": "Testing memory usage"})
            
            # Create multiple test files to simulate larger workload
            for i in range(10):
                test_file = temp_dir / f"test_topic_{i}.md"
                test_file.write_text(f"# Test Topic {i}\nTesting memory usage in pipeline")
                
                output_file = temp_dir / f"ideas_{i}.json"
                try:
                    generate_research_ideas(
                        workshop_file=str(test_file),
                        output_file=str(output_file),
                        model="gpt-4",
                        max_num_generations=1
                    )
                except Exception:
                    pass  # Ignore errors, focus on memory
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f} MB"
    
    @pytest.mark.slow
    def test_pipeline_timeout_handling(self, temp_dir):
        """Test that pipeline respects timeout limits."""
        import time
        
        start_time = time.time()
        
        with patch('ai_scientist.llm.get_llm_response') as mock_llm:
            # Simulate slow LLM response
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # Small delay to simulate processing
                return json.dumps({"title": "Timeout Test", "abstract": "Testing timeouts"})
            
            mock_llm.side_effect = slow_response
            
            topic_file = temp_dir / "timeout_test.md"
            topic_file.write_text("# Timeout Test\nTesting pipeline timeouts")
            
            output_file = temp_dir / "timeout_ideas.json"
            
            # Run with short timeout
            try:
                generate_research_ideas(
                    workshop_file=str(topic_file),
                    output_file=str(output_file),
                    model="gpt-4",
                    max_num_generations=5,
                    timeout=2  # Short timeout
                )
            except Exception:
                pass  # Timeout expected
        
        elapsed_time = time.time() - start_time
        
        # Should not take much longer than timeout
        assert elapsed_time < 5, f"Pipeline took {elapsed_time:.2f}s, expected to timeout around 2s"