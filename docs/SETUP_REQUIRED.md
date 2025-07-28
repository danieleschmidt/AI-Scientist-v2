# Manual Setup Requirements

## Overview

This document lists items that require manual setup due to permission restrictions.

## GitHub Repository Settings

### Branch Protection Rules
- Enable branch protection for `main` branch
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Dismiss stale reviews when new commits are pushed
- Restrict pushes to main branch

### Repository Secrets
Configure the following secrets in repository settings:
- `OPENAI_API_KEY`: OpenAI API access
- `ANTHROPIC_API_KEY`: Anthropic Claude API access
- `S2_API_KEY`: Semantic Scholar API access
- `DOCKER_REGISTRY_TOKEN`: Container registry access

### Repository Variables
- `PYTHON_VERSION`: "3.11"
- `DOCKER_REGISTRY`: "ghcr.io"

## GitHub Actions Workflows

### Required Workflow Files
Create these files in `.github/workflows/`:
1. Copy `templates/github-workflows/ci.yml` to `.github/workflows/ci.yml`
2. Copy `templates/github-workflows/cd.yml` to `.github/workflows/cd.yml`

### Status Checks
Enable the following required status checks:
- `test-python-3.11`
- `lint-and-format`
- `security-scan`
- `type-check`

## External Integrations

### Container Registry
- Set up GHCR (GitHub Container Registry) access
- Configure registry permissions

### Monitoring Setup
- Configure Prometheus metrics endpoint
- Set up Grafana dashboards
- Enable alerting rules

## Security Configuration

### Code Scanning
- Enable CodeQL analysis
- Configure Dependabot alerts
- Set up secret scanning alerts

## References

- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Repository Security](https://docs.github.com/en/code-security)