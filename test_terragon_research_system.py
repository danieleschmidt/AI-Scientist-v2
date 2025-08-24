#!/usr/bin/env python3
"""
Comprehensive test suite for Terragon Autonomous Research System.
"""

import os
import sys
import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terragon_autonomous_research_engine import (
    ResearchConfig,
    AutonomousResearchEngine,
    load_research_config
)
from terragon_research_cli import ResearchCLI


class TestResearchConfig:
    """Test ResearchConfig functionality."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = ResearchConfig(topic_description_path="test_topic.md")
        
        assert config.topic_description_path == "test_topic.md"
        assert config.output_directory == "autonomous_research_output"
        assert config.max_ideas == 5
        assert config.ideation_model == "gpt-4o-2024-05-13"
        assert config.writeup_type == "icbinb"
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ResearchConfig(
            topic_description_path="test.md",
            max_ideas=10,
            writeup_type="normal"
        )
        
        config_dict = asdict(config)
        assert config_dict["topic_description_path"] == "test.md"
        assert config_dict["max_ideas"] == 10
        assert config_dict["writeup_type"] == "normal"


class TestAutonomousResearchEngine:
    """Test AutonomousResearchEngine functionality."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Topic\n\nTest research topic content")
            topic_path = f.name
        
        config = ResearchConfig(
            topic_description_path=topic_path,
            max_ideas=2,
            skip_writeup=True,
            skip_review=True
        )
        
        yield config
        
        # Cleanup
        os.unlink(topic_path)
    
    def test_engine_initialization(self, temp_config):
        """Test engine initialization."""
        engine = AutonomousResearchEngine(temp_config)
        
        assert engine.config == temp_config
        assert engine.session_id is not None
        assert engine.output_dir.exists()
        assert "session_" in engine.session_id
    
    @pytest.mark.asyncio
    async def test_research_idea_generation(self, temp_config):
        """Test research idea generation."""
        engine = AutonomousResearchEngine(temp_config)
        
        # Mock the ideation args
        ideation_args = {
            "workshop_file": temp_config.topic_description_path,
            "model": temp_config.ideation_model,
            "max_num_generations": temp_config.max_ideas,
            "num_reflections": temp_config.idea_reflections,
            "output_dir": str(engine.output_dir / "ideation")
        }
        
        ideas = await engine._generate_research_ideas(ideation_args)
        
        assert isinstance(ideas, list)
        assert len(ideas) == temp_config.max_ideas
        
        for idea in ideas:
            assert "Name" in idea
            assert "Title" in idea
            assert "Experiment" in idea
            assert "generated_timestamp" in idea
    
    @pytest.mark.asyncio
    async def test_bfts_config_creation(self, temp_config):
        """Test BFTS configuration creation."""
        engine = AutonomousResearchEngine(temp_config)
        
        experiment_dir = engine.output_dir / "test_experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        idea_json_path = experiment_dir / "idea.json"
        with open(idea_json_path, 'w') as f:
            json.dump({"Name": "test_idea"}, f)
        
        config_path = await engine._create_bfts_config(experiment_dir, idea_json_path)
        
        assert config_path.exists()
        assert config_path.suffix == ".yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert "agent" in config_data
        assert "search" in config_data
        assert "paths" in config_data
        assert config_data["agent"]["model"] == temp_config.experiment_model
        assert config_data["agent"]["num_workers"] == temp_config.num_workers


@pytest.mark.asyncio
class TestResearchCLI:
    """Test ResearchCLI functionality."""
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        cli = ResearchCLI()
        
        assert cli.parser is not None
        assert hasattr(cli.parser, 'parse_args')
    
    def test_argument_parsing_run_command(self):
        """Test argument parsing for run command."""
        cli = ResearchCLI()
        
        args = cli.parser.parse_args([
            'run',
            '--topic', 'test_topic.md',
            '--max-ideas', '3',
            '--writeup-type', 'normal',
            '--skip-writeup'
        ])
        
        assert args.command == 'run'
        assert args.topic == 'test_topic.md'
        assert args.max_ideas == 3
        assert args.writeup_type == 'normal'
        assert args.skip_writeup is True
    
    def test_argument_parsing_create_topic_command(self):
        """Test argument parsing for create-topic command."""
        cli = ResearchCLI()
        
        args = cli.parser.parse_args([
            'create-topic',
            '--output', 'my_topic.md',
            '--template', 'nlp'
        ])
        
        assert args.command == 'create-topic'
        assert args.output == 'my_topic.md'
        assert args.template == 'nlp'
    
    def test_topic_template_generation(self):
        """Test topic template generation."""
        cli = ResearchCLI()
        
        basic_template = cli._get_topic_template("basic")
        nlp_template = cli._get_topic_template("nlp")
        
        assert "# Research Topic" in basic_template
        assert "## Title" in basic_template
        assert "## Keywords" in basic_template
        
        assert "Natural Language Processing" in nlp_template
        assert "transformers" in nlp_template
    
    def test_config_template_generation(self):
        """Test configuration template generation."""
        cli = ResearchCLI()
        
        basic_config = cli._get_config_template("basic")
        advanced_config = cli._get_config_template("advanced")
        
        assert "topic_description_path" in basic_config
        assert "ideation_model" in basic_config
        assert "max_ideas: 3" in basic_config
        
        assert "max_ideas: 8" in advanced_config
        assert "enable_quality_gates" in advanced_config
    
    async def test_quick_start_setup(self):
        """Test quick start setup functionality."""
        cli = ResearchCLI()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            await cli._quick_start_setup()
            
            # Check if files were created
            assert Path("example_research_topic.md").exists()
            assert Path("research_config.yaml").exists()
            
            # Verify content
            with open("example_research_topic.md", 'r') as f:
                topic_content = f.read()
            assert "# Research Topic" in topic_content
            
            with open("research_config.yaml", 'r') as f:
                config_content = f.read()
            assert "topic_description_path" in config_content


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    @patch('terragon_autonomous_research_engine.perform_experiments_bfts')
    @patch('terragon_autonomous_research_engine.aggregate_plots')
    async def test_minimal_pipeline_execution(self, mock_plots, mock_experiments):
        """Test minimal pipeline execution with mocked components."""
        
        # Create temporary topic file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Test Research Topic

## Title
Test Machine Learning Research

## Keywords  
machine learning, neural networks

## TL;DR
Testing the autonomous research pipeline.

## Abstract
This is a test research topic for validating the autonomous research system functionality.
""")
            topic_path = f.name
        
        try:
            # Create test configuration
            config = ResearchConfig(
                topic_description_path=topic_path,
                max_ideas=1,
                skip_writeup=True,
                skip_review=True,
                num_workers=1,
                max_steps=5
            )
            
            # Mock external calls
            mock_experiments.return_value = None
            mock_plots.return_value = None
            
            # Execute pipeline
            engine = AutonomousResearchEngine(config)
            results = await engine.execute_full_pipeline()
            
            # Verify results
            assert results["status"] == "completed"
            assert "phases" in results
            assert "ideation" in results["phases"]
            assert "experimentation" in results["phases"]
            
            # Verify ideation phase
            ideation_phase = results["phases"]["ideation"]
            assert ideation_phase["status"] == "completed"
            assert ideation_phase["ideas_generated"] == 1
            
            # Verify experimentation phase
            experiment_phase = results["phases"]["experimentation"]
            assert experiment_phase["status"] == "completed"
            assert experiment_phase["experiments_completed"] == 1
            
        finally:
            # Cleanup
            os.unlink(topic_path)
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with non-existent topic file
        config = ResearchConfig(topic_description_path="nonexistent.md")
        
        with pytest.raises(FileNotFoundError):
            engine = AutonomousResearchEngine(config)
            # This should fail when trying to read the topic file
            asyncio.run(engine._generate_research_ideas({
                "workshop_file": "nonexistent.md",
                "model": "test-model",
                "max_num_generations": 1,
                "num_reflections": 1,
                "output_dir": "/tmp/test"
            }))


class TestConfigurationManagement:
    """Test configuration loading and management."""
    
    @pytest.mark.asyncio
    async def test_config_file_loading(self):
        """Test configuration file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "topic_description_path": "test.md",
                "max_ideas": 8,
                "ideation_model": "custom-model",
                "writeup_type": "normal"
            }, f)
            config_path = f.name
        
        try:
            config = await load_research_config(config_path)
            
            assert config.topic_description_path == "test.md"
            assert config.max_ideas == 8
            assert config.ideation_model == "custom-model"
            assert config.writeup_type == "normal"
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_default_config_creation(self):
        """Test default configuration creation when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nonexistent_config.yaml")
            
            config = await load_research_config(config_path)
            
            # Should create default config
            assert os.path.exists(config_path)
            assert config.topic_description_path == "research_topic.md"
            assert config.max_ideas == 5


class TestUtilities:
    """Test utility functions."""
    
    def test_session_id_generation(self):
        """Test session ID generation uniqueness."""
        config = ResearchConfig(topic_description_path="test.md")
        
        engine1 = AutonomousResearchEngine(config)
        engine2 = AutonomousResearchEngine(config)
        
        # Session IDs should be unique
        assert engine1.session_id != engine2.session_id
        assert len(engine1.session_id) > 10  # Should be a reasonable timestamp
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ResearchConfig(
                topic_description_path="test.md",
                output_directory=os.path.join(temp_dir, "research_output")
            )
            
            engine = AutonomousResearchEngine(config)
            
            # Directory should be created
            assert engine.output_dir.exists()
            assert engine.output_dir.is_dir()
            assert "session_" in str(engine.output_dir)


# Test fixtures and utilities
@pytest.fixture
def sample_research_results():
    """Sample research results for testing."""
    return {
        "session_id": "20250824_143022",
        "config": {
            "topic_description_path": "test_topic.md",
            "max_ideas": 3,
            "writeup_type": "icbinb"
        },
        "phases": {
            "ideation": {
                "status": "completed",
                "ideas_generated": 3,
                "timestamp": "2025-08-24T14:30:22"
            },
            "experimentation": {
                "status": "completed",
                "experiments_completed": 3,
                "timestamp": "2025-08-24T15:45:30"
            }
        },
        "start_time": "2025-08-24T14:30:00",
        "end_time": "2025-08-24T16:00:00",
        "status": "completed",
        "token_usage": {
            "gpt-4o": {"total_cost": 15.50},
            "claude-3-5-sonnet": {"total_cost": 25.30}
        }
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])