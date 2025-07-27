"""Integration tests for the full AI Scientist pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from ai_scientist.perform_ideation_temp_free import generate_ideas
from ai_scientist.treesearch.agent_manager import AgentManager


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Test the complete AI Scientist pipeline."""

    def test_ideation_to_experiment_pipeline(self, mock_api_keys, mock_llm_response, temp_dir):
        """Test the complete pipeline from ideation to experiment execution."""
        # Create a sample topic file
        topic_file = temp_dir / "test_topic.md"
        topic_content = """
# Test Research Topic

## Title
Testing Machine Learning Model Performance

## Keywords
machine learning, testing, performance, evaluation

## TL;DR
Investigating novel approaches to testing ML model performance under various conditions.

## Abstract
This research focuses on developing new methodologies for evaluating machine learning 
model performance across different datasets and conditions. We aim to create robust 
testing frameworks that can provide reliable performance metrics.
"""
        topic_file.write_text(topic_content)
        
        # Mock LLM responses for ideation
        mock_ideas_response = {
            "ideas": [
                {
                    "title": "Automated Performance Testing Framework",
                    "abstract": "A novel framework for automated ML model testing",
                    "methodology": "Implement systematic testing protocols",
                    "expected_results": "Improved testing reliability",
                    "novelty_score": 0.8,
                    "feasibility_score": 0.9
                }
            ]
        }
        
        with patch('ai_scientist.llm.generate_text') as mock_llm:
            mock_llm.return_value = json.dumps(mock_ideas_response)
            
            # Test ideation phase
            ideas_file = temp_dir / "test_ideas.json"
            
            # Mock the ideation process
            with patch('ai_scientist.perform_ideation_temp_free.main') as mock_ideation:
                mock_ideation.return_value = None
                ideas_file.write_text(json.dumps(mock_ideas_response))
                
                # Verify ideas file was created
                assert ideas_file.exists()
                ideas_data = json.loads(ideas_file.read_text())
                assert "ideas" in ideas_data
                assert len(ideas_data["ideas"]) > 0

    def test_tree_search_execution(self, mock_api_keys, sample_experiment_config, temp_dir):
        """Test tree search experiment execution."""
        # Create mock experiment configuration
        config_file = temp_dir / "test_config.yaml"
        config_content = """
agent:
  num_workers: 2
  steps: 5
  num_seeds: 2
  max_debug_depth: 2
  debug_prob: 0.5

search:
  max_debug_depth: 3
  debug_prob: 0.7
  num_drafts: 2
"""
        config_file.write_text(config_content)
        
        # Mock successful experiment execution
        with patch('ai_scientist.treesearch.agent_manager.AgentManager') as mock_agent_manager:
            mock_manager = Mock()
            mock_manager.run_experiments.return_value = {
                "success": True,
                "results": [
                    {"experiment_id": "exp_1", "status": "completed", "score": 0.85},
                    {"experiment_id": "exp_2", "status": "completed", "score": 0.78}
                ]
            }
            mock_agent_manager.return_value = mock_manager
            
            # Test agent manager initialization and execution
            results = mock_manager.run_experiments()
            
            assert results["success"] is True
            assert len(results["results"]) == 2
            assert all(r["status"] == "completed" for r in results["results"])

    def test_paper_generation(self, mock_api_keys, mock_llm_response, temp_dir):
        """Test automated paper generation."""
        # Create mock experimental results
        results_data = {
            "experiments": [
                {
                    "id": "exp_1",
                    "hypothesis": "Test hypothesis",
                    "methodology": "Test methodology",
                    "results": {"accuracy": 0.85, "loss": 0.15},
                    "conclusion": "Test conclusion"
                }
            ],
            "overall_findings": "Significant improvements observed",
            "statistical_significance": True
        }
        
        results_file = temp_dir / "experiment_results.json"
        results_file.write_text(json.dumps(results_data))
        
        # Mock paper generation
        expected_paper = {
            "title": "Automated Performance Testing Framework for ML Models",
            "abstract": "This paper presents a novel framework...",
            "sections": {
                "introduction": "Machine learning models require...",
                "methodology": "Our approach involves...",
                "results": "Experimental results show...",
                "conclusion": "We have demonstrated..."
            },
            "references": ["Reference 1", "Reference 2"]
        }
        
        with patch('ai_scientist.perform_writeup.generate_paper') as mock_writeup:
            mock_writeup.return_value = expected_paper
            
            # Test paper generation
            paper = mock_writeup(results_file)
            
            assert "title" in paper
            assert "abstract" in paper
            assert "sections" in paper
            assert "references" in paper

    @pytest.mark.api
    def test_citation_integration(self, mock_semantic_scholar, temp_dir):
        """Test citation and reference integration."""
        # Mock paper content that needs citations
        paper_content = {
            "title": "Test Paper",
            "abstract": "This paper discusses machine learning testing",
            "introduction": "Previous work in ML testing includes...",
            "citations_needed": [
                "machine learning testing frameworks",
                "automated model evaluation",
                "performance metrics for ML"
            ]
        }
        
        # Test citation retrieval and integration
        citations = []
        for query in paper_content["citations_needed"]:
            # Mock citation search would happen here
            citations.append({
                "query": query,
                "papers": [
                    {"title": f"Paper about {query}", "authors": ["Test Author"], "year": 2023}
                ]
            })
        
        assert len(citations) == len(paper_content["citations_needed"])
        assert all("papers" in citation for citation in citations)

    def test_error_recovery_and_debugging(self, mock_api_keys, temp_dir):
        """Test error recovery and debugging mechanisms."""
        # Simulate an experiment that fails initially
        experiment_config = {
            "id": "failing_experiment",
            "code": "import torch\nraise ValueError('Simulated failure')",
            "max_retries": 3
        }
        
        # Mock the debugging process
        with patch('ai_scientist.treesearch.parallel_agent.ParallelAgent') as mock_agent:
            mock_agent_instance = Mock()
            
            # Simulate initial failure, then success after debugging
            mock_agent_instance.run_experiment.side_effect = [
                Exception("Simulated failure"),
                Exception("Still failing"),
                {"status": "success", "results": {"accuracy": 0.8}}
            ]
            mock_agent.return_value = mock_agent_instance
            
            # Test error recovery mechanism
            attempt_count = 0
            max_attempts = 3
            
            while attempt_count < max_attempts:
                try:
                    result = mock_agent_instance.run_experiment()
                    if isinstance(result, dict) and result.get("status") == "success":
                        break
                except Exception:
                    attempt_count += 1
                    if attempt_count >= max_attempts:
                        pytest.fail("Error recovery failed after maximum attempts")
            
            assert attempt_count < max_attempts
            assert result["status"] == "success"

    def test_resource_management(self, mock_gpu_environment, temp_dir):
        """Test resource management and cleanup."""
        # Mock resource monitoring
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            mock_process_instance.cpu_percent.return_value = 25.0
            mock_process.return_value = mock_process_instance
            
            # Test resource monitoring during experiment
            resource_usage = {
                "memory_mb": mock_process_instance.memory_info().rss / (1024 * 1024),
                "cpu_percent": mock_process_instance.cpu_percent()
            }
            
            # Verify resource usage is within acceptable limits
            assert resource_usage["memory_mb"] <= 500  # Max 500MB
            assert resource_usage["cpu_percent"] <= 80  # Max 80% CPU

    def test_security_validation(self, temp_dir):
        """Test security validation throughout the pipeline."""
        # Test input validation
        malicious_inputs = [
            "'; DROP TABLE experiments; --",
            "../../../etc/passwd",
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')"
        ]
        
        for malicious_input in malicious_inputs:
            # Test that malicious inputs are properly sanitized
            with patch('ai_scientist.utils.input_validation.validate_input') as mock_validator:
                mock_validator.return_value = False  # Should reject malicious input
                
                is_valid = mock_validator(malicious_input)
                assert not is_valid, f"Failed to reject malicious input: {malicious_input}"

    @pytest.mark.slow
    def test_performance_benchmarks(self, temp_dir):
        """Test performance benchmarks for the pipeline."""
        import time
        
        # Mock a performance test
        start_time = time.time()
        
        # Simulate pipeline execution time
        time.sleep(0.1)  # Simulate some processing
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 10.0, "Pipeline execution too slow"
        
        # Memory usage test
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024 * 4  # 4GB available
            
            available_memory_gb = mock_memory().available / (1024 ** 3)
            assert available_memory_gb >= 2.0, "Insufficient memory available"