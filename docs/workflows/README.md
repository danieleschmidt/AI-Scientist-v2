# GitHub Workflows Requirements

## Overview

This document outlines the GitHub Actions workflows required for the AI Scientist v2 project.

## Required Workflows

### 1. Continuous Integration (CI)
- **Path**: `.github/workflows/ci.yml`
- **Triggers**: Pull requests, pushes to main
- **Requirements**:
  - Python 3.11+ testing matrix
  - Dependency installation and caching
  - Code quality checks (black, isort, flake8, mypy)
  - Security scanning (bandit, safety)
  - Unit and integration test execution
  - Coverage reporting

### 2. Continuous Deployment (CD)
- **Path**: `.github/workflows/cd.yml`
- **Triggers**: Releases, tags
- **Requirements**:
  - Automated semantic versioning
  - Package building and publishing
  - Docker image creation and registry push
  - Release notes generation

### 3. Security Scanning
- **Path**: `.github/workflows/security.yml`
- **Triggers**: Schedule (daily), manual dispatch
- **Requirements**:
  - Dependency vulnerability scanning
  - Secret detection in codebase
  - Container security scanning
  - SAST (Static Application Security Testing)

## Manual Setup Required

Due to repository permissions, these workflows must be created manually:

1. Copy templates from `templates/github-workflows/`
2. Configure repository secrets and variables
3. Set up branch protection rules
4. Configure required status checks

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Security Workflow Examples](https://github.com/actions/starter-workflows/tree/main/code-scanning)