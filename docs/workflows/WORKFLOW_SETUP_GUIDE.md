# GitHub Actions Workflow Setup Guide

This directory contains the GitHub Actions workflows designed for AI Scientist v2. Due to repository permissions, these files need to be manually added to the `.github/workflows/` directory.

## Required Manual Setup

### 1. Copy Workflow Files

Copy the following files from `docs/workflows/github-actions/` to `.github/workflows/`:

```bash
cp docs/workflows/github-actions/ci.yml .github/workflows/
cp docs/workflows/github-actions/release.yml .github/workflows/
cp docs/workflows/github-actions/security.yml .github/workflows/
```

### 2. Workflow Descriptions

#### `ci.yml` - Continuous Integration
- **Purpose**: Comprehensive CI pipeline with code quality, security, and testing
- **Triggers**: Push to main/develop, PRs, nightly builds
- **Features**: 
  - Multi-OS testing (Ubuntu/macOS)
  - Python 3.11/3.12 matrix testing
  - Code quality checks (black, isort, flake8, mypy)
  - Security scanning (bandit, safety, semgrep)
  - Performance benchmarking
  - Docker build validation
  - Documentation deployment

#### `release.yml` - Release Automation
- **Purpose**: Automated release creation and publishing
- **Triggers**: Git tags (v*.*.*), manual workflow dispatch
- **Features**:
  - Semantic version validation
  - Multi-platform Docker builds
  - PyPI/TestPyPI publishing
  - GitHub release creation
  - Artifact management

#### `security.yml` - Security Scanning
- **Purpose**: Comprehensive security analysis
- **Triggers**: Push to main/develop, PRs, daily schedule
- **Features**:
  - CodeQL analysis
  - Container security scanning (Trivy)
  - Secret detection (GitLeaks)
  - Supply chain security (OSV Scanner)
  - SBOM generation
  - SARIF reporting

### 3. Required Repository Configuration

#### GitHub Environments
Create the following environments in GitHub repository settings:

1. **pypi** - For PyPI publishing
   - Add PyPI OIDC configuration
   - Enable deployment protection rules

2. **testpypi** - For Test PyPI publishing
   - Add Test PyPI OIDC configuration

#### Repository Secrets
Configure the following secrets if needed:
- `GITLEAKS_LICENSE` (optional, for GitLeaks Pro features)

#### Branch Protection
Configure branch protection rules as described in `docs/BRANCH_PROTECTION_GUIDE.md`

### 4. Permissions Required

The workflows require the following permissions:
- `contents: write` - For release creation
- `security-events: write` - For SARIF uploads
- `id-token: write` - For OIDC authentication
- `packages: write` - For container registry

### 5. Verification Steps

After copying the workflows:

1. **Syntax Check**: GitHub will automatically validate YAML syntax
2. **Test Run**: Create a test PR to verify CI workflow
3. **Security Scan**: Verify security workflow runs on schedule
4. **Release Test**: Test release workflow with a pre-release tag

### 6. Troubleshooting

#### Common Issues

**Workflow not triggering:**
- Check file location is `.github/workflows/`
- Verify YAML syntax is correct
- Ensure repository has required permissions

**Security scans failing:**
- Check if all security tools are properly configured
- Verify SARIF upload permissions
- Review security baseline files

**Release workflow issues:**
- Verify tag format matches semantic versioning
- Check PyPI environment configuration
- Ensure Docker registry permissions

#### Debug Commands

```bash
# Validate workflow syntax
gh workflow list

# Check workflow runs
gh run list --workflow=ci.yml

# View workflow logs
gh run view <run-id> --log
```

## Workflow Dependencies

These workflows depend on:
- Repository configuration in `pyproject.toml`
- Test configuration in `pytest.ini`
- Security configuration in existing security tools
- Pre-commit hooks in `.pre-commit-config.yaml`

## Next Steps

1. Copy workflow files to `.github/workflows/`
2. Configure repository environments and secrets
3. Set up branch protection rules
4. Test workflows with a sample PR
5. Monitor workflow execution and adjust as needed

The workflows are designed to be production-ready and follow GitHub Actions best practices for security, performance, and maintainability.