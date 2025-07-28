# 🛠️ Developer Onboarding Guide

Welcome to the AI Scientist v2 development team! This guide will help you get up and running quickly with our development environment and processes.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.4+ (for GPU acceleration)

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2

# Create development branch
git checkout -b feature/your-feature-name
```

### 2. Development Environment

#### Option A: Dev Container (Recommended)
```bash
# Open in VS Code with Dev Container extension
code .
# Follow prompts to reopen in container
```

#### Option B: Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -r dev-requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Configure your API keys in .env
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export S2_API_KEY="your-semantic-scholar-key"
```

### 4. Verify Installation
```bash
# Run tests
pytest tests/

# Run linting
make lint

# Run type checking
make type-check

# Start development server
make dev
```

## 🏗️ Project Structure

```
AI-Scientist-v2/
├── ai_scientist/                # Main package
│   ├── __init__.py
│   ├── llm.py                  # LLM interfaces
│   ├── vlm.py                  # Vision-language models
│   ├── tools/                  # Research tools
│   ├── treesearch/             # Tree search algorithms
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── performance/            # Performance tests
├── docs/                       # Documentation
├── scripts/                    # Build and utility scripts
├── monitoring/                 # Monitoring configs
├── security/                   # Security configurations
└── templates/                  # CI/CD templates
```

## 🔄 Development Workflow

### 1. Feature Development
```bash
# Start new feature
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Make changes
# ... develop your feature ...

# Run tests locally
make test

# Commit changes
git add .
git commit -m "feat: add new feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Code Quality Checks
We enforce strict code quality standards:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security scanning
make security-scan

# Run all checks
make check-all
```

### 3. Testing Strategy

#### Unit Tests
```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=ai_scientist --cov-report=html
```

#### Integration Tests
```bash
# Run integration tests (requires API keys)
pytest tests/integration/ -v

# Run specific test
pytest tests/integration/test_research_pipeline.py::test_full_pipeline -v
```

#### Performance Tests
```bash
# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only
```

## 🧪 Testing Best Practices

### Writing Tests
1. **Use descriptive test names**: `test_llm_generates_valid_research_idea_from_prompt`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies**: Use `pytest-mock` for API calls
4. **Test edge cases**: Empty inputs, rate limits, network failures

### Example Test
```python
def test_research_idea_generation():
    # Arrange
    prompt = "Generate ideas about machine learning"
    mock_llm = Mock()
    mock_llm.generate.return_value = "Novel ML idea"
    
    # Act
    result = generate_research_idea(prompt, llm=mock_llm)
    
    # Assert
    assert result is not None
    assert "idea" in result
    mock_llm.generate.assert_called_once_with(prompt)
```

## 🔒 Security Guidelines

### Code Security
- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow principle of least privilege

### Dependencies
- Pin dependency versions in `requirements.txt`
- Regularly update dependencies: `make update-deps`
- Scan for vulnerabilities: `make security-scan`

### Data Handling
- Encrypt sensitive data at rest and in transit
- Use secure random number generation
- Implement proper access controls
- Log security events

## 📏 Coding Standards

### Python Style Guide
We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints for all functions
def process_research_data(data: List[Dict[str, Any]]) -> ProcessedData:
    """Process research data for analysis.
    
    Args:
        data: Raw research data from experiments
        
    Returns:
        Processed data ready for analysis
        
    Raises:
        ValueError: If data format is invalid
    """
    pass

# Use dataclasses for structured data
@dataclass
class ResearchResult:
    hypothesis: str
    experiments: List[Experiment]
    conclusion: str
    confidence: float
```

### Documentation Standards
- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints
- Document complex algorithms with inline comments

## 🔧 Available Make Commands

```bash
# Development
make dev              # Start development server
make install          # Install dependencies
make install-dev      # Install dev dependencies

# Code Quality
make format           # Format code with black and isort
make lint             # Run linting (flake8, mypy)
make type-check       # Run type checking
make security-scan    # Run security scans

# Testing
make test             # Run all tests
make test-unit        # Run unit tests only
make test-integration # Run integration tests only
make test-performance # Run performance tests
make coverage         # Generate coverage report

# Build and Release
make build            # Build package
make clean            # Clean build artifacts
make release          # Create release

# Utilities
make update-deps      # Update dependencies
make check-all        # Run all quality checks
make docs             # Generate documentation
```

## 🐛 Debugging Tips

### Common Issues

#### GPU Memory Issues
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

#### API Rate Limits
```python
# Implement exponential backoff
import backoff

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=3)
def api_call():
    return llm.generate(prompt)
```

#### Performance Issues
```bash
# Profile your code
python -m cProfile -o profile.prof script.py
python -m pstats profile.prof

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py
```

## 🤝 Contributing Guidelines

### Pull Request Process
1. Create feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Create pull request with descriptive title
6. Address review feedback
7. Squash commits before merge

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] No secrets or API keys in code
- [ ] Performance impact considered
- [ ] Security implications reviewed

## 📚 Additional Resources

### Documentation
- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](api/README.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Security Policy](../SECURITY.md)

### Tools and Extensions
- **VS Code Extensions**:
  - Python
  - Pylance
  - GitLens
  - Docker
  - Remote-Containers

### Learning Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Python Best Practices](https://docs.python-guide.org/)

## ❓ Getting Help

### Internal Support
- **Slack**: #ai-scientist-dev
- **Team Lead**: @team-lead
- **Architecture Questions**: @senior-architect

### Office Hours
- **Daily Standup**: 9:00 AM EST
- **Tech Talk Friday**: 3:00 PM EST
- **Open Office Hours**: Tuesdays 2-4 PM EST

### External Resources
- [GitHub Issues](https://github.com/SakanaAI/AI-Scientist-v2/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/ai-scientist)
- [Documentation Site](https://docs.terragonlabs.ai/ai-scientist-v2)