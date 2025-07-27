# Contributing to AI Scientist v2

Thank you for your interest in contributing to AI Scientist v2! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include detailed information about the bug or feature request
- Provide steps to reproduce for bugs
- Check if the issue already exists

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2

# Install dependencies
make install-dev

# Set up pre-commit hooks
make setup-hooks

# Run tests
make test
```

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints
- Write comprehensive tests
- Document your code
- Follow conventional commit messages

### Testing

- Write unit tests for new functionality
- Ensure integration tests pass
- Add performance tests for critical paths
- Maintain test coverage above 80%

## Development Workflow

1. Check existing issues and PRs
2. Create an issue for discussion (for large changes)
3. Fork and create a branch
4. Implement changes with tests
5. Run quality checks: `make quality-check`
6. Submit a pull request
7. Address review feedback
8. Celebrate when merged! 🎉

## Release Process

Releases are automated using semantic-release based on conventional commits.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.