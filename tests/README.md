# AI Scientist v2 - Testing Documentation

## Overview

This directory contains the comprehensive test suite for AI Scientist v2, implementing multiple testing strategies to ensure code quality, security, and performance.

## Test Organization

### Directory Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Pytest configuration and fixtures
├── fixtures/                # Test data and mock fixtures
├── integration/             # End-to-end integration tests
├── performance/             # Performance and load tests
├── security/                # Security-focused tests
├── unit/                    # Unit tests for individual components
└── test_*.py               # General test modules
```

### Test Categories

#### 🔧 Unit Tests (`test_*.py`, `unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Scope**: Single component, mocked dependencies
- **Speed**: Fast (< 1s per test)
- **Markers**: `@pytest.mark.unit`

#### 🔗 Integration Tests (`integration/`)
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components, real dependencies where possible
- **Speed**: Medium (1-10s per test)
- **Markers**: `@pytest.mark.integration`, `@pytest.mark.slow`

#### ⚡ Performance Tests (`performance/`)
- **Purpose**: Benchmark performance and resource usage
- **Scope**: System-wide performance characteristics
- **Speed**: Slow (10s+ per test)
- **Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`

#### 🔒 Security Tests (`test_*security*.py`)
- **Purpose**: Validate security controls and detect vulnerabilities
- **Scope**: Security-critical components and workflows
- **Speed**: Variable
- **Markers**: `@pytest.mark.security`

## Running Tests

### Basic Commands

```bash
# Run all tests
make test

# Run tests with coverage
pytest --cov=ai_scientist --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_llm_simple.py

# Run with verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

### Test Categories

```bash
# Unit tests only
pytest -m "unit and not slow"

# Integration tests
pytest -m integration

# Security tests
pytest -m security

# Performance tests
pytest -m performance

# GPU-dependent tests
pytest -m gpu

# API-dependent tests
pytest -m api

# Exclude slow tests
pytest -m "not slow"

# Run tests requiring specific markers
pytest -m "unit or integration"
```

### Environment-Specific Testing

```bash
# Development environment
ENVIRONMENT=development pytest

# CI/CD environment
ENVIRONMENT=ci pytest --cov=ai_scientist --cov-fail-under=80

# Production validation
ENVIRONMENT=production pytest -m "security or performance"
```

## Writing Tests

### Test Naming Conventions

- **File naming**: `test_<component_name>.py`
- **Class naming**: `Test<ComponentName>`
- **Method naming**: `test_<functionality>_<condition>_<expected_result>`

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from ai_scientist.some_module import SomeClass


class TestSomeClass:
    """Test suite for SomeClass functionality."""
    
    def test_method_with_valid_input_returns_expected_result(self):
        """Test that method returns expected result with valid input."""
        # Arrange
        instance = SomeClass()
        input_data = "test_input"
        expected = "expected_output"
        
        # Act
        result = instance.method(input_data)
        
        # Assert
        assert result == expected
    
    @pytest.mark.slow
    def test_method_with_large_dataset_performs_within_limits(self, large_dataset):
        """Test performance with large dataset."""
        # Performance test implementation
        pass
    
    @pytest.mark.security
    def test_method_rejects_malicious_input(self):
        """Test that method properly handles malicious input."""
        # Security test implementation
        pass
```

### Using Fixtures

```python
def test_with_temp_directory(temp_dir):
    """Test using temporary directory fixture."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.read_text() == "test content"

def test_with_mock_api(mock_openai_client):
    """Test using mocked API client."""
    response = mock_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}]
    )
    assert response.choices[0].message.content == "Test response"
```

## Available Fixtures

### Environment Fixtures
- `test_env_setup`: Test environment configuration
- `temp_dir`: Temporary directory for test files
- `mock_experiment_dir`: Mock experiment directory structure

### API Mocking Fixtures
- `mock_api_keys`: Mock API keys for testing
- `mock_openai_client`: Mocked OpenAI client
- `mock_anthropic_client`: Mocked Anthropic client
- `mock_llm_response`: Factory for creating mock LLM responses

### Data Fixtures
- `sample_research_idea`: Sample research idea for testing
- `sample_experiment_config`: Sample experiment configuration
- `mock_semantic_scholar_response`: Mock literature search response

### System Fixtures
- `mock_gpu_info`: Mock GPU information
- `mock_process_monitor`: Mock process monitoring
- `mock_file_system`: Mock file system operations
- `disable_network`: Disable network access during tests

## Test Guidelines

### Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Clarity**: Test names should clearly describe what is being tested
3. **Coverage**: Aim for high code coverage, especially for critical paths
4. **Speed**: Keep unit tests fast; mark slow tests appropriately
5. **Determinism**: Tests should produce consistent results across runs

### Security Testing Guidelines

1. **Input Validation**: Test all input validation functions with malicious data
2. **Access Controls**: Verify authorization and authentication mechanisms
3. **Data Sanitization**: Test output sanitization and encoding
4. **Resource Limits**: Verify resource consumption limits are enforced

### Performance Testing Guidelines

1. **Benchmarks**: Establish baseline performance metrics
2. **Resource Usage**: Monitor memory, CPU, and GPU utilization
3. **Scalability**: Test behavior under increasing load
4. **Regression**: Detect performance regressions in CI/CD

### Integration Testing Guidelines

1. **End-to-End**: Test complete workflows from start to finish
2. **Error Handling**: Test error conditions and recovery mechanisms
3. **Configuration**: Test with different configuration combinations
4. **External Services**: Test integration with external APIs and services

## Continuous Integration

### CI Test Pipeline

1. **Static Analysis**: Code quality checks (lint, type check)
2. **Security Scan**: Static security analysis
3. **Unit Tests**: Fast unit test execution
4. **Integration Tests**: Core integration tests
5. **Performance Tests**: Key performance benchmarks
6. **Coverage Report**: Code coverage analysis

### Coverage Requirements

- **Minimum Coverage**: 80% overall code coverage
- **Critical Components**: 95% coverage for security-critical code
- **Test Coverage**: Tests themselves should be reviewed for quality

### Test Data Management

- **Fixtures**: Use fixtures for reusable test data
- **Factories**: Use factory patterns for generating test objects
- **Cleanup**: Ensure proper cleanup after tests complete
- **Isolation**: Tests should not share mutable state

## Debugging Tests

### Common Issues

1. **Flaky Tests**: Tests that pass/fail inconsistently
   - Check for timing dependencies
   - Verify proper mocking
   - Ensure test isolation

2. **Resource Leaks**: Tests that don't clean up properly
   - Use fixtures for setup/teardown
   - Monitor resource usage
   - Implement proper cleanup

3. **Slow Tests**: Tests that take too long
   - Optimize test logic
   - Use appropriate mocking
   - Mark as slow if necessary

### Debugging Commands

```bash
# Run with detailed output
pytest -vvv --tb=long

# Run with debugging
pytest --pdb

# Show test coverage gaps
pytest --cov=ai_scientist --cov-report=term-missing

# Profile test execution
pytest --profile-svg

# Run only failed tests
pytest --lf

# Run tests matching pattern
pytest -k "test_security"
```

## Contributing to Tests

### Adding New Tests

1. **Identify Test Category**: Unit, integration, performance, or security
2. **Create Test File**: Follow naming conventions
3. **Write Test Cases**: Cover happy path, edge cases, and error conditions
4. **Add Markers**: Use appropriate pytest markers
5. **Update Documentation**: Update this README if needed

### Test Review Checklist

- [ ] Tests are properly categorized and marked
- [ ] Test names clearly describe functionality
- [ ] Proper use of fixtures and mocking
- [ ] Adequate coverage of edge cases
- [ ] Performance tests have reasonable time limits
- [ ] Security tests cover threat model
- [ ] Tests are deterministic and isolated

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)