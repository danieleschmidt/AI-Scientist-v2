# Contributing to AI Scientist v2

Thank you for your interest in contributing to AI Scientist v2! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix issues
- 🚀 **Feature Requests**: Suggest new functionality
- 📝 **Documentation**: Improve or add documentation
- 🧪 **Testing**: Enhance test coverage
- 💡 **Ideas**: Share ideas for improvements
- 🔧 **Code**: Submit bug fixes and new features

### Before You Start

1. Check [existing issues](https://github.com/SakanaAI/AI-Scientist-v2/issues) to avoid duplicates
2. Read our [architecture documentation](ARCHITECTURE.md)
3. Join our [Discord community](https://discord.gg/terragon-labs) for discussions

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- Docker (optional but recommended)
- CUDA-capable GPU (for full functionality)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2

# Set up development environment
make setup

# Run tests to verify setup
make test
```

### Development Environment Options

#### Option 1: Local Development
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

#### Option 2: Docker Development
```bash
docker-compose up -d
docker-compose exec ai-scientist bash
```

#### Option 3: Dev Containers (VS Code)
Open the project in VS Code and use the "Reopen in Container" option.

## Contributing Guidelines

### Issue Guidelines

#### Bug Reports
Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and logs

#### Feature Requests
Use the feature request template and include:
- Clear description of the proposed feature
- Use case and motivation
- Proposed implementation approach
- Breaking change considerations

### Code Contributions

#### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

#### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(ideation): add multi-domain research support`
- `fix(llm): handle API timeout errors gracefully`
- `docs(api): update authentication documentation`
- `test(integration): add e2e pipeline tests`

## Pull Request Process

### Before Submitting

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes following our code standards
4. **Add** tests for new functionality
5. **Update** documentation as needed
6. **Run** the full test suite
7. **Ensure** CI checks pass

### PR Guidelines

1. **Title**: Use descriptive titles following conventional commits
2. **Description**: Use the PR template and include:
   - Summary of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)
3. **Size**: Keep PRs focused and reasonably sized
4. **Tests**: Include appropriate test coverage
5. **Documentation**: Update relevant documentation

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Manual testing for significant changes
4. **Documentation**: Verify documentation accuracy
5. **Merge**: Squash and merge after approval

## Code Standards

### Python Code Style

We use the following tools (configured in `pyproject.toml`):

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security analysis

#### Code Quality Checklist

- [ ] Code follows PEP 8 standards
- [ ] All functions have type hints
- [ ] Docstrings for all public functions
- [ ] No hardcoded secrets or API keys
- [ ] Error handling implemented
- [ ] Logging added where appropriate
- [ ] Performance considerations addressed

### Security Requirements

- **Input Validation**: Sanitize all external inputs
- **Error Handling**: Don't expose sensitive information
- **Dependencies**: Keep dependencies updated
- **Secrets**: Use environment variables
- **Code Execution**: Ensure proper sandboxing

## Testing

### Test Types

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance
- **Security Tests**: Test security measures

### Running Tests

```bash
# All tests
make test

# Specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# With coverage
make test-coverage

# Fast tests only (skip slow tests)
pytest -m "not slow"
```

### Test Requirements

- **Coverage**: Minimum 80% code coverage
- **Quality**: Tests should be deterministic and fast
- **Isolation**: Tests should not depend on external services
- **Documentation**: Complex tests should be well-commented

## Documentation

### Types of Documentation

- **API Documentation**: Auto-generated from docstrings
- **User Guides**: Step-by-step tutorials
- **Developer Docs**: Architecture and design documents
- **Examples**: Practical usage examples

### Documentation Standards

- **Clarity**: Write for your intended audience
- **Completeness**: Cover all important aspects
- **Examples**: Include practical examples
- **Updates**: Keep documentation current with code changes

### Building Documentation

```bash
# Build documentation locally
make docs

# Serve documentation
make docs-serve
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time community chat
- **Twitter**: Updates and announcements

### Getting Help

- Check the [documentation](https://docs.terragonlabs.ai/ai-scientist-v2)
- Search [existing issues](https://github.com/SakanaAI/AI-Scientist-v2/issues)
- Join our [Discord community](https://discord.gg/terragon-labs)
- Ask in [GitHub Discussions](https://github.com/SakanaAI/AI-Scientist-v2/discussions)

### Recognition

Contributors are recognized in:
- Release notes
- Contributors section in README
- Annual contributor highlights
- Special contributor badges

## Development Workflow

### Typical Contribution Flow

1. **Discuss** the change (for major features)
2. **Create** an issue (if one doesn't exist)
3. **Fork** and create a branch
4. **Develop** the feature/fix
5. **Test** thoroughly
6. **Submit** a pull request
7. **Address** review feedback
8. **Celebrate** when merged! 🎉

### Release Process

- We follow [Semantic Versioning](https://semver.org/)
- Releases are automated via GitHub Actions
- Release notes are automatically generated
- Contributors are credited in release notes

## Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](https://docs.terragonlabs.ai/ai-scientist-v2/api)
- [Development Roadmap](docs/ROADMAP.md)
- [Design Decisions](docs/adr/)
- [Performance Benchmarks](docs/benchmarks.md)

---

Thank you for contributing to AI Scientist v2! Together we're advancing automated scientific discovery. 🚀

For questions about contributing, reach out to us at contribute@terragonlabs.ai