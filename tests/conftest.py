"""Pytest configuration and shared fixtures for AI Scientist v2 tests."""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import shutil
import logging


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'GEMINI_API_KEY': 'test-gemini-key',
        'S2_API_KEY': 'test-s2-key',
        'AWS_ACCESS_KEY_ID': 'test-aws-key',
        'AWS_SECRET_ACCESS_KEY': 'test-aws-secret',
        'AWS_REGION_NAME': 'us-east-1'
    }):
        yield


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    def _mock_response(content="Test response", model="test-model"):
        mock_response = Mock()
        mock_response.content = content
        mock_response.model = model
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        return mock_response
    return _mock_response


@pytest.fixture
def sample_research_idea():
    """Sample research idea for testing."""
    return {
        "title": "Test Research Idea",
        "abstract": "This is a test research idea for unit testing.",
        "keywords": ["test", "research", "ml"],
        "methodology": "Test methodology description",
        "expected_results": "Expected test results",
        "novelty_score": 0.8,
        "feasibility_score": 0.9
    }


@pytest.fixture
def sample_experiment_config():
    """Sample experiment configuration for testing."""
    return {
        "model": "test-model",
        "max_workers": 2,
        "max_steps": 5,
        "timeout": 300,
        "enable_debug": True
    }


@pytest.fixture
def mock_gpu_environment():
    """Mock GPU environment for testing."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_name', return_value='Test GPU'):
        yield


@pytest.fixture
def disable_logging():
    """Disable logging during tests to reduce noise."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def sample_paper_content():
    """Sample paper content for testing."""
    return {
        "title": "Test Paper Title",
        "abstract": "This is a test abstract for the paper.",
        "introduction": "This is the introduction section.",
        "methodology": "This is the methodology section.",
        "results": "This is the results section.",
        "conclusion": "This is the conclusion section.",
        "references": ["Test Reference 1", "Test Reference 2"]
    }


@pytest.fixture
def mock_semantic_scholar():
    """Mock Semantic Scholar API responses."""
    def _mock_search(query, limit=10):
        return {
            "data": [
                {
                    "paperId": f"test-paper-{i}",
                    "title": f"Test Paper {i}",
                    "abstract": f"Abstract for test paper {i}",
                    "authors": [{"name": f"Author {i}"}],
                    "year": 2023,
                    "citationCount": i * 10
                }
                for i in range(limit)
            ]
        }
    
    with patch('ai_scientist.tools.semantic_scholar.search_papers', side_effect=_mock_search):
        yield


@pytest.fixture
def isolated_filesystem():
    """Provide an isolated filesystem for tests that modify files."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield Path(tmpdir)
        finally:
            os.chdir(original_cwd)


@pytest.fixture
def mock_tree_search_state():
    """Mock tree search state for testing."""
    return {
        "nodes": [],
        "current_depth": 0,
        "max_depth": 5,
        "explored_paths": set(),
        "successful_experiments": [],
        "failed_experiments": []
    }


@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import psutil
    import time
    
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss
    
    yield
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance metrics for slow tests
    if duration > 5.0 or memory_delta > 100 * 1024 * 1024:  # 100MB
        print(f"\nPerformance warning:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory delta: {memory_delta / 1024 / 1024:.1f}MB")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and name."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in str(item.fspath) or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in str(item.fspath) or item.name.startswith("test_unit"):
            item.add_marker(pytest.mark.unit)
        
        # Mark security tests
        if "security" in item.name or "test_security" in item.name:
            item.add_marker(pytest.mark.security)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark API tests
        if "api" in item.name or "test_api" in item.name:
            item.add_marker(pytest.mark.api)


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state between tests."""
    # Store original environment
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)