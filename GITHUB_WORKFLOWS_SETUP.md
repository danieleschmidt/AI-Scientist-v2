# 🚀 GitHub Actions Workflows Setup Guide

Due to GitHub App permissions, the workflow files need to be manually created. This guide provides all the workflow files needed for the comprehensive SDLC automation.

## 📁 Required Directory Structure

Create the following directory structure in your repository:

```
.github/
├── workflows/
│   ├── ci.yml
│   ├── cd.yml
│   ├── security.yml
│   ├── performance.yml
│   ├── release.yml
│   └── dependabot.yml
└── dependabot.yml
```

## 🔧 Workflow Files

### 1. Continuous Integration (`.github/workflows/ci.yml`)

```yaml
# AI Scientist v2 - Continuous Integration Workflow Template
name: 🚀 Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  # Security scanning
  security-scan:
    name: 🔒 Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  # Code quality checks
  code-quality:
    name: 🔍 Code Quality
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        pip install flake8 mypy bandit black isort
    
    - name: Run code quality checks
      run: |
        black --check ai_scientist/ tests/
        isort --check-only ai_scientist/ tests/
        flake8 ai_scientist/ tests/
        mypy ai_scientist/ --ignore-missing-imports
        bandit -r ai_scientist/

  # Test suite
  test:
    name: 🧪 Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.11", "3.12"]
        include:
          - os: ubuntu-latest
            python-version: "3.11"
            coverage: true
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short
    
    - name: Upload coverage
      if: matrix.coverage
      uses: codecov/codecov-action@v3
      with:
        fail_ci_if_error: false

  # Build verification
  build:
    name: 🏗️ Build Package
    runs-on: ubuntu-latest
    needs: [code-quality, test]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build twine
        python -m build
        python -m twine check dist/*
```

### 2. Continuous Deployment (`.github/workflows/cd.yml`)

```yaml
# AI Scientist v2 - Continuous Deployment Workflow
name: 🚀 Continuous Deployment

on:
  push:
    tags:
      - 'v*.*.*'
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io

jobs:
  validate:
    name: 🔍 Pre-deployment Validation
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      environment: ${{ steps.environment.outputs.environment }}
      
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Extract version and environment
      id: version
      run: |
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          VERSION=${GITHUB_REF#refs/tags/v}
        else
          VERSION="dev-$(git rev-parse --short HEAD)"
        fi
        echo "version=$VERSION" >> $GITHUB_OUTPUT

  build-images:
    name: 🐳 Build & Push Images
    runs-on: ubuntu-latest
    needs: validate
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ github.repository }}:${{ needs.validate.outputs.version }}

  publish-package:
    name: 📦 Publish Package
    runs-on: ubuntu-latest
    needs: validate
    if: github.event_name == 'release'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Build and publish package
      run: |
        python -m pip install --upgrade pip build twine
        python -m build
        python -m twine upload dist/*
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

### 3. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: 🔒 Security Scanning

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  security-hardening:
    name: Security Hardening Check
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety semgrep
        pip install -r requirements.txt
    
    - name: Run Bandit security scanner
      run: |
        bandit -r ai_scientist/ -f json -o bandit-report.json
        bandit -r ai_scientist/ -f txt
    
    - name: Run Safety dependency checker
      run: |
        safety check --json --output safety-report.json
        safety check
    
    - name: Run Semgrep security scanner
      run: |
        semgrep --config=auto ai_scientist/ --json --output=semgrep-report.json
        semgrep --config=auto ai_scientist/
```

### 4. Performance Monitoring (`.github/workflows/performance.yml`)

```yaml
name: 📊 Performance Monitoring

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  performance-tests:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        pip install pytest-benchmark memory-profiler py-spy
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark-results.json
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: benchmark-results.json
```

### 5. Release Management (`.github/workflows/release.yml`)

```yaml
name: 🚀 Release Management

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v2.1.0)'
        required: true
        type: string

env:
  PYTHON_VERSION: "3.11"

jobs:
  validate-release:
    name: 🔍 Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Validate version format
      id: version
      run: |
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          VERSION=${GITHUB_REF#refs/tags/}
        else
          VERSION=${{ github.event.inputs.version }}
        fi
        
        if [[ ! $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
          echo "Invalid version format: $VERSION"
          exit 1
        fi
        
        echo "version=$VERSION" >> $GITHUB_OUTPUT

  create-release:
    name: 📦 Create Release
    runs-on: ubuntu-latest
    needs: validate-release
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ needs.validate-release.outputs.version }}
        release_name: AI Scientist v2 ${{ needs.validate-release.outputs.version }}
        body: |
          ## AI Scientist v2 Release
          
          Version: ${{ needs.validate-release.outputs.version }}
          
          ### Installation
          ```bash
          pip install ai-scientist-v2==${{ needs.validate-release.outputs.version }}
          ```
        draft: false
        prerelease: false
```

### 6. Dependency Management (`.github/workflows/dependabot.yml`)

```yaml
name: 🔄 Dependency Management

on:
  schedule:
    - cron: '0 8 * * 1'  # Weekly on Monday at 8 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    name: Update Dependencies
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Check for outdated packages
      run: |
        pip list --outdated --format=json > outdated-packages.json
        pip list --outdated
    
    - name: Upload outdated packages report
      uses: actions/upload-artifact@v3
      with:
        name: outdated-packages
        path: outdated-packages.json
```

## 📋 Dependabot Configuration

Also create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "08:00"
    open-pull-requests-limit: 10
    commit-message:
      prefix: "chore"
      include: "scope"
    labels:
      - "dependencies"
      - "python"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "ci"
      include: "scope"
    labels:
      - "dependencies"
      - "github-actions"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "10:00"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "chore"
      include: "scope"
    labels:
      - "dependencies"
      - "docker"
```

## 🔐 Required Secrets

To enable full functionality, add these secrets in your GitHub repository settings:

### Repository Secrets
- `PYPI_API_TOKEN` - For PyPI package publishing
- `CODECOV_TOKEN` - For code coverage reporting (optional)
- `SLACK_WEBHOOK_URL` - For notifications (optional)

### Environment Variables
- `OPENAI_API_KEY` - For testing with OpenAI
- `ANTHROPIC_API_KEY` - For testing with Anthropic
- `GEMINI_API_KEY` - For testing with Gemini

## 🚀 Setup Instructions

1. **Create the directory structure**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy each workflow file** into the respective location

3. **Add the dependabot configuration**:
   ```bash
   # Copy the dependabot.yml content to .github/dependabot.yml
   ```

4. **Configure repository secrets** in GitHub Settings > Secrets and variables > Actions

5. **Enable GitHub Actions** in your repository if not already enabled

6. **Test the workflows** by creating a pull request or pushing to main

## ✅ Verification

After setup, verify that:
- [ ] All workflow files are in `.github/workflows/`
- [ ] Dependabot configuration is in `.github/dependabot.yml`
- [ ] Required secrets are configured
- [ ] GitHub Actions are enabled
- [ ] First workflow run completes successfully

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot)
- [GitHub Secrets Management](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

---

**Note**: These workflows provide comprehensive CI/CD automation including security scanning, quality checks, performance monitoring, and automated releases. Adjust the configurations based on your specific requirements and security policies.