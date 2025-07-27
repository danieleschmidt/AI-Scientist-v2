"""
Pytest configuration and shared fixtures for AI Scientist v2 tests.
"""

import os
import tempfile
import pytest
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_env_setup() -> Dict[str, str]:
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "SANDBOX_ENABLED": "true",
        "MAX_EXECUTION_TIME": "30",
        "MAX_MEMORY_USAGE": "1024",
        "NETWORK_ACCESS_ENABLED": "false",
        "COST_TRACKING_ENABLED": "false",
        "AUTO_BACKUP_ENABLED": "false",
    }
    
    # Set environment variables for the test session
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_experiment_dir(temp_dir: Path) -> Path:
    """Create a mock experiment directory structure."""
    exp_dir = temp_dir / "experiments" / "test_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create basic experiment structure
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    
    return exp_dir


@pytest.fixture
def mock_api_keys() -> Dict[str, str]:
    """Provide mock API keys for testing."""
    return {
        "OPENAI_API_KEY": "sk-test-openai-key-1234567890abcdef",
        "ANTHROPIC_API_KEY": "sk-ant-test-anthropic-key-1234567890abcdef",
        "GEMINI_API_KEY": "test-gemini-key-1234567890abcdef",
        "S2_API_KEY": "test-s2-key-1234567890abcdef",
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _create_response(content: str = "Test response", model: str = "gpt-4"):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        mock_response.model = model
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response
    
    return _create_response


@pytest.fixture
def mock_openai_client(mock_llm_response):
    """Mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock_client:
        client_instance = Mock()
        client_instance.chat.completions.create.return_value = mock_llm_response()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_anthropic_client(mock_llm_response):
    """Mock Anthropic client for testing."""
    with patch("anthropic.Anthropic") as mock_client:
        client_instance = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        client_instance.messages.create.return_value = mock_response
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def sample_research_idea() -> Dict[str, Any]:
    """Provide a sample research idea for testing."""
    return {
        "title": "Test Research Idea",
        "abstract": "This is a test research idea for unit testing.",
        "hypothesis": "We hypothesize that this test will pass successfully.",
        "methodology": "Use pytest to verify functionality.",
        "expected_results": "All tests should pass with green status.",
        "novelty_score": 0.8,
        "feasibility_score": 0.9,
        "impact_score": 0.7,
        "keywords": ["testing", "ai", "automation", "research"],
        "related_work": ["Paper 1", "Paper 2", "Paper 3"],
    }


@pytest.fixture
def sample_experiment_config() -> Dict[str, Any]:
    """Provide sample experiment configuration."""
    return {
        "model": "gpt-4",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 300,
        "retry_attempts": 3,
        "experiment_type": "unit_test",
        "data_sources": ["test_data.json"],
        "evaluation_metrics": ["accuracy", "precision", "recall"],
    }


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information for testing."""
    return {
        "available": True,
        "count": 1,
        "memory_total": 8192,  # MB
        "memory_free": 6144,   # MB
        "memory_used": 2048,   # MB
        "utilization": 25,     # %
        "temperature": 65,     # Celsius
        "driver_version": "525.147.05",
        "cuda_version": "12.4",
    }


@pytest.fixture
def mock_semantic_scholar_response():
    """Mock Semantic Scholar API response."""
    return {
        "data": [
            {
                "paperId": "123456789",
                "title": "Test Paper 1",
                "authors": [{"name": "Test Author 1"}, {"name": "Test Author 2"}],
                "year": 2024,
                "abstract": "This is a test paper abstract.",
                "citationCount": 10,
                "influentialCitationCount": 5,
                "venue": "Test Conference 2024",
                "url": "https://example.com/paper1",
            },
            {
                "paperId": "987654321",
                "title": "Test Paper 2",
                "authors": [{"name": "Test Author 3"}],
                "year": 2023,
                "abstract": "Another test paper abstract.",
                "citationCount": 25,
                "influentialCitationCount": 12,
                "venue": "Test Journal",
                "url": "https://example.com/paper2",
            },
        ],
        "total": 2,
        "offset": 0,
    }


@pytest.fixture
def mock_file_system(temp_dir: Path):
    """Mock file system operations for testing."""
    # Create test files and directories
    test_files = {
        "config.yaml": "test: true\nmode: testing",
        "data.json": '{"test": "data", "items": [1, 2, 3]}',
        "script.py": "#!/usr/bin/env python3\nprint('Hello, Test!')",
        "requirements.txt": "pytest>=7.0.0\npytest-cov>=4.0.0",
    }
    
    for filename, content in test_files.items():
        (temp_dir / filename).write_text(content)
    
    # Create subdirectories
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "results").mkdir(exist_ok=True)
    (temp_dir / "cache").mkdir(exist_ok=True)
    
    return temp_dir


@pytest.fixture
def mock_process_monitor():
    """Mock process monitoring for testing."""
    with patch("psutil.Process") as mock_process:
        process_instance = Mock()
        process_instance.pid = 12345
        process_instance.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        process_instance.cpu_percent.return_value = 15.5
        process_instance.status.return_value = "running"
        process_instance.create_time.return_value = 1640995200.0
        mock_process.return_value = process_instance
        yield process_instance


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up imports between tests to avoid module state issues."""
    import sys
    modules_to_clean = [
        mod for mod in sys.modules.keys() 
        if mod.startswith("ai_scientist")
    ]
    
    yield
    
    # Clean up after test
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch("logging.getLogger") as mock_get_logger:
        logger = Mock()
        mock_get_logger.return_value = logger
        yield logger


@pytest.fixture
def disable_network():
    """Disable network access during tests."""
    import socket
    
    def disabled_socket(*args, **kwargs):
        raise OSError("Network access disabled during testing")
    
    original_socket = socket.socket
    socket.socket = disabled_socket
    
    yield
    
    socket.socket = original_socket


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
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "performance", "security"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that are likely slow
        if "integration" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add api marker to tests that use API fixtures
        if any(fixture in item.fixturenames 
               for fixture in ["mock_openai_client", "mock_anthropic_client"]):
            item.add_marker(pytest.mark.api)


# Test data constants
TEST_CONSTANTS = {
    "DEFAULT_TIMEOUT": 30,
    "MAX_RETRIES": 3,
    "TEST_MODEL": "gpt-4",
    "TEST_TEMPERATURE": 0.1,
    "TEST_MAX_TOKENS": 1000,
}