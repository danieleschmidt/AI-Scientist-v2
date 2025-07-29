# 🚀 GitHub Workflows Implementation Guide
## AI Scientist v2 - Enterprise SDLC Automation

This guide provides step-by-step instructions for implementing the advanced GitHub Actions workflows provided as templates.

---

## 📋 Prerequisites

### Repository Setup
- [ ] GitHub repository with appropriate branch protection rules
- [ ] GitHub Advanced Security enabled (for CodeQL and secret scanning)
- [ ] Repository admin access for workflow creation

### Required Secrets
Configure the following secrets in your repository settings:

#### Essential Secrets
- `GITHUB_TOKEN` - Automatically provided by GitHub (no action needed)

#### Optional Enhancement Secrets  
- `SEMGREP_APP_TOKEN` - For enhanced Semgrep security scanning
- `CODECOV_TOKEN` - For code coverage reporting
- `GITLEAKS_LICENSE` - For GitLeaks Pro features
- `S2_API_KEY` - For Semantic Scholar API access
- `OPENAI_API_KEY` - For AI model integration

#### External Service Integration
- `HONEYCOMB_API_KEY` - For telemetry export to Honeycomb
- `GRAFANA_CLOUD_OTLP_ENDPOINT` - For Grafana Cloud integration
- `GRAFANA_CLOUD_API_KEY` - For Grafana Cloud authentication

---

## 🔧 Implementation Steps

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Copy Template Files
Copy the template files from `docs/workflows/templates/` to `.github/workflows/`:

```bash
# Copy CI workflow
cp docs/workflows/templates/github-ci.yml .github/workflows/ci.yml

# Copy CD workflow  
cp docs/workflows/templates/github-cd.yml .github/workflows/cd.yml

# Copy Security workflow
cp docs/workflows/templates/github-security.yml .github/workflows/security.yml
```

### Step 3: Configure Repository Secrets
1. Navigate to your repository on GitHub
2. Go to Settings → Secrets and variables → Actions
3. Add the required secrets listed above

### Step 4: Enable GitHub Advanced Security
1. Go to Settings → Code security and analysis
2. Enable:
   - [ ] Dependency graph
   - [ ] Dependabot alerts
   - [ ] Dependabot security updates
   - [ ] Code scanning (CodeQL)
   - [ ] Secret scanning

### Step 5: Set Up Branch Protection
1. Go to Settings → Branches
2. Add protection rule for `main` branch:
   - [ ] Require status checks to pass
   - [ ] Require branches to be up to date
   - [ ] Required status checks:
     - `CI Success`
     - `Security Scanning`
     - `Container Security`

---

## 🔍 Workflow Descriptions

### 🔄 CI Workflow (`ci.yml`)
**Triggers**: Push to main/develop, PR to main/develop, manual dispatch

**Features**:
- Multi-OS testing matrix (Ubuntu, Windows, macOS)
- Python 3.11 & 3.12 compatibility
- Code quality checks (Black, isort, Flake8, MyPy)
- Security scanning integration
- Container vulnerability scanning
- Coverage reporting

**Duration**: ~15-20 minutes
**Resources**: Uses 3 parallel jobs for efficiency

### 🚀 CD Workflow (`cd.yml`)
**Triggers**: Release published, manual dispatch

**Features**:
- Multi-platform container builds (AMD64, ARM64)
- SLSA provenance generation
- Security attestation
- Staged deployment (staging → production)
- Release artifact signing

**Duration**: ~25-30 minutes
**Resources**: Builds for multiple architectures

### 🔐 Security Workflow (`security.yml`)
**Triggers**: Daily schedule, push to main, PR to main, manual dispatch

**Features**:
- CodeQL analysis for multiple languages
- Dependency vulnerability scanning
- SAST with multiple tools (Bandit, Semgrep)
- Container security scanning (Trivy)
- Secret detection (TruffleHog, GitLeaks)
- Compliance checking automation

**Duration**: ~20-25 minutes
**Resources**: Comprehensive security scanning

---

## ⚙️ Customization Options

### Environment Variables
Update the following in each workflow file to match your needs:

```yaml
env:
  PYTHON_VERSION: "3.11"      # Your Python version
  REGISTRY: ghcr.io           # Your container registry
  IMAGE_NAME: ${{ github.repository }}  # Your image name
```

### Testing Configuration
Modify test execution parameters:

```yaml
# In ci.yml
- name: 🧪 Run Tests
  run: |
    pytest tests/ \
      --maxfail=5 \              # Stop after 5 failures
      --tb=short \               # Short traceback format
      --strict-markers \         # Require registered markers
      --cov=ai_scientist \       # Coverage for your package
      --cov-report=xml
```

### Security Scanning Customization
Adjust security scanner configurations:

```yaml
# Semgrep rules
config: >-
  p/security-audit
  p/secrets  
  p/python
  p/owasp-top-ten
  p/docker
  # Add custom rules: p/custom-rules
```

### Container Platforms
Modify supported platforms:

```yaml
strategy:
  matrix:
    platform:
      - linux/amd64
      - linux/arm64
      # - linux/arm/v7  # Add ARM v7 support
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Permission Denied for Workflow Creation
**Error**: `refusing to allow a GitHub App to create or update workflow`
**Solution**: Ensure you have admin access to the repository

#### 2. Missing Required Secrets
**Error**: `Secret SEMGREP_APP_TOKEN not found`
**Solution**: Add the secret in repository settings or mark as optional

#### 3. CodeQL Analysis Fails
**Error**: `No source code was seen during the build`
**Solution**: Ensure proper language detection in CodeQL configuration

#### 4. Container Build Fails
**Error**: `failed to solve: process "/bin/sh -c pip install` 
**Solution**: Check Dockerfile compatibility with multi-platform builds

### Performance Optimization

#### Reduce CI Runtime
```yaml
# Use parallel test execution
env:
  PYTEST_XDIST_WORKER_COUNT: auto

# Cache dependencies effectively
- uses: actions/setup-python@v5
  with:
    cache: pip
    cache-dependency-path: |
      requirements.txt
      pyproject.toml
```

#### Optimize Security Scans
```yaml
# Run security scans only on changes
- name: 🔍 Run Selective Scanning
  if: contains(github.event.head_commit.modified, 'ai_scientist/')
```

---

## 📊 Monitoring and Alerts

### GitHub Actions Monitoring
1. Navigate to Actions tab in your repository
2. Monitor workflow runs and success rates
3. Set up email notifications for failures

### Security Dashboard
1. Go to Security tab in your repository
2. Review CodeQL alerts and dependency vulnerabilities
3. Set up automated issue creation for security findings

### Performance Metrics
Track the following metrics:
- Average workflow runtime
- Success/failure rates
- Security scan coverage
- Deployment frequency

---

## 🔮 Advanced Configuration

### Environment-Specific Deployments
Create environment protection rules:
1. Go to Settings → Environments
2. Create `staging` and `production` environments
3. Add protection rules and required reviewers

### Custom Deployment Strategies
Implement blue-green or canary deployments:

```yaml
# Add to cd.yml
deploy-canary:
  name: 🐤 Canary Deployment
  runs-on: ubuntu-latest
  needs: security-scan
  environment: production
  
  steps:
    - name: 🚀 Deploy Canary
      run: |
        echo "Deploying 10% traffic to canary"
        # Add canary deployment logic
```

### Integration with External Tools
Connect to monitoring and alerting systems:

```yaml
# Add to workflows
- name: 📊 Update Monitoring Dashboard
  run: |
    curl -X POST "${{ secrets.GRAFANA_WEBHOOK_URL }}" \
      -H "Content-Type: application/json" \
      -d '{"deployment": "success", "version": "${{ needs.build.outputs.version }}"}'
```

---

## ✅ Validation Checklist

After implementation, verify:

- [ ] All workflows appear in Actions tab
- [ ] CI workflow runs successfully on PR
- [ ] Security scans complete without critical issues  
- [ ] Container builds and pushes to registry
- [ ] Deployment workflows execute properly
- [ ] Branch protection rules enforce CI checks
- [ ] Security alerts appear in Security tab
- [ ] Artifacts are properly uploaded and retained

---

*This implementation guide ensures enterprise-grade SDLC automation for AI Scientist v2. For questions or issues, refer to the GitHub Actions documentation or create an issue in this repository.*