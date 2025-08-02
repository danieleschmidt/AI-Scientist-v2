# AI Scientist v2 - Manual Setup Requirements

## Overview

Due to GitHub App permission limitations, some configuration must be completed manually by repository maintainers. This document provides step-by-step instructions for completing the SDLC implementation.

## Required Permissions

The following permissions are needed for full SDLC implementation:

### Repository Permissions
- **Administration**: Required for branch protection rules
- **Actions**: Required for GitHub Actions workflow management
- **Secrets**: Required for environment variable management
- **Settings**: Required for repository configuration

### Organization Permissions (if applicable)
- **Security**: Required for security policy enforcement
- **Audit logs**: Required for compliance monitoring
- **Member management**: Required for team access controls

## Manual Setup Tasks

### 1. GitHub Actions Workflows

**Priority: HIGH** | **Estimated Time: 30 minutes**

Copy the workflow files from `docs/workflows/templates/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/templates/github-ci.yml .github/workflows/ci.yml
cp docs/workflows/templates/github-cd.yml .github/workflows/cd.yml  
cp docs/workflows/templates/github-security.yml .github/workflows/security.yml

# Commit the workflows
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows"
git push
```

### 2. Repository Settings Configuration

**Priority: HIGH** | **Estimated Time: 15 minutes**

#### Branch Protection Rules

Navigate to **Settings > Branches** and configure protection for `main` branch:

- [x] Require pull request reviews before merging
  - Required approving reviews: 2
  - Dismiss stale reviews when new commits are pushed
- [x] Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Required status checks:
    - `test` (from CI workflow)
    - `lint` (from CI workflow)
    - `security-scan` (from security workflow)
- [x] Require conversation resolution before merging
- [x] Require signed commits
- [x] Include administrators
- [x] Restrict pushes that create files

## Completion Checklist

Mark as complete when all items are configured:

- [ ] GitHub Actions workflows deployed
- [ ] Branch protection rules configured
- [ ] Repository settings updated
- [ ] Security policies enabled
- [ ] All validation tests pass

---

**Estimated Total Time**: 1.5-2 hours
**Required Permissions**: Repository admin
**Prerequisites**: Completed SDLC checkpoint implementation