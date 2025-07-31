# Branch Protection Strategy Guide

This document outlines the recommended branch protection rules for AI Scientist v2 to ensure code quality, security, and proper release management.

## Overview

Branch protection rules help maintain code quality by requiring specific conditions before changes can be merged into protected branches. This guide provides configuration recommendations for different branch types.

## Recommended Branch Protection Rules

### Main Branch (`main`)

The main branch should have the strictest protection rules as it represents the production-ready code.

**Required Settings:**

```yaml
Protection Rules for 'main':
  - Require pull request reviews before merging: ✅
    - Required number of reviewers: 2
    - Dismiss stale reviews when new commits are pushed: ✅
    - Require review from code owners: ✅
    - Restrict reviews to users with write permissions: ✅
    
  - Require status checks to pass before merging: ✅
    - Require branches to be up to date before merging: ✅
    - Required status checks:
      - "Code Quality" (from ci.yml)
      - "Security Scanning" (from ci.yml) 
      - "Test Suite (ubuntu-latest, 3.11)" (from ci.yml)
      - "Build Package" (from ci.yml)
      - "Docker Build Test" (from ci.yml)
      - "Security Report Summary" (from security.yml)
      
  - Require conversation resolution before merging: ✅
  
  - Require signed commits: ✅ (recommended for security)
  
  - Require linear history: ✅ (enforces clean git history)
  
  - Restrict pushes that create files: ✅
    - Restricted paths:
      - "experiments/**"
      - "aisci_outputs/**" 
      - "results/**"
      - "cache/**"
      - "final_papers/**"
      - "*.log"
      - "*.tmp"
      
  - Allow force pushes: ❌
  
  - Allow deletions: ❌
  
  - Block creations: ❌ (allow branch creation for hotfixes)
```

### Develop Branch (`develop`)

The develop branch has slightly relaxed rules to allow for faster iteration during development.

**Required Settings:**

```yaml
Protection Rules for 'develop':
  - Require pull request reviews before merging: ✅
    - Required number of reviewers: 1
    - Dismiss stale reviews when new commits are pushed: ✅
    - Require review from code owners: ✅
    
  - Require status checks to pass before merging: ✅
    - Require branches to be up to date before merging: ❌
    - Required status checks:
      - "Code Quality" (from ci.yml)
      - "Test Suite (ubuntu-latest, 3.11)" (from ci.yml)
      - "Security Scanning" (from ci.yml)
      
  - Require conversation resolution before merging: ✅
  
  - Require signed commits: ❌ (optional for development)
  
  - Require linear history: ❌ (allow merge commits)
  
  - Allow force pushes: ❌
  
  - Allow deletions: ❌
```

### Release Branches (`release/*`)

Release branches require strict validation but allow maintainer overrides for urgent fixes.

**Required Settings:**

```yaml
Protection Rules for 'release/*':
  - Require pull request reviews before merging: ✅
    - Required number of reviewers: 2
    - Dismiss stale reviews when new commits are pushed: ✅
    - Require review from code owners: ✅
    - Restrict reviews to users with write permissions: ✅
    
  - Require status checks to pass before merging: ✅
    - Require branches to be up to date before merging: ✅
    - Required status checks:
      - "Code Quality" (from ci.yml)
      - "Security Scanning" (from ci.yml)
      - "Test Suite" (all matrix combinations)
      - "Build Package" (from ci.yml)
      - "Performance Tests" (from ci.yml)
      
  - Require conversation resolution before merging: ✅
  
  - Require signed commits: ✅
  
  - Require linear history: ✅
  
  - Allow force pushes: ❌
  
  - Allow deletions: ❌
  
  - Restrict pushes that create files: ✅
    - Same restrictions as main branch
```

### Feature Branches (`feature/*`)

Feature branches have minimal protection to allow rapid development while maintaining basic quality checks.

**Required Settings:**

```yaml
Protection Rules for 'feature/*':
  - Require pull request reviews before merging: ❌
    (Allow direct pushes for feature development)
    
  - Require status checks to pass before merging: ✅
    - Required status checks:
      - "Code Quality" (from ci.yml)
      - "Test Suite (ubuntu-latest, 3.11)" (from ci.yml)
      
  - Allow force pushes: ✅ (for rebasing during development)
  
  - Allow deletions: ✅ (allow feature branch cleanup)
```

## Implementation Steps

### 1. Configure Branch Protection via GitHub UI

1. Go to your repository's Settings → Branches
2. Click "Add rule" for each branch pattern
3. Configure the settings as outlined above
4. Save each rule

### 2. Configure Branch Protection via GitHub CLI

```bash
# Main branch protection
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Code Quality","Security Report Summary","Test Suite (ubuntu-latest, 3.11)","Build Package"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null

# Develop branch protection  
gh api repos/:owner/:repo/branches/develop/protection \
  --method PUT \
  --field required_status_checks='{"strict":false,"contexts":["Code Quality","Test Suite (ubuntu-latest, 3.11)"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"require_code_owner_reviews":true}' \
  --field restrictions=null
```

### 3. Configure Branch Protection via Terraform

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.ai_scientist.node_id
  pattern       = "main"
  
  required_status_checks {
    strict = true
    contexts = [
      "Code Quality",
      "Security Report Summary", 
      "Test Suite (ubuntu-latest, 3.11)",
      "Build Package",
      "Docker Build Test"
    ]
  }
  
  required_pull_request_reviews {
    required_approving_review_count = 2
    dismiss_stale_reviews          = true
    require_code_owner_reviews     = true
    restrict_reviews_to_team       = true
  }
  
  enforce_admins = true
}
```

## Security Considerations

### Required Security Checks

All protected branches must pass these security-related status checks:

1. **Bandit Security Scan** - Static security analysis for Python
2. **Safety Dependency Check** - Known vulnerability scanning
3. **Semgrep Analysis** - Advanced static analysis
4. **Secret Detection** - Prevent credential leaks
5. **Container Security Scan** - Docker image vulnerability scanning

### Code Owner Requirements

Critical files require approval from specific teams:

- **Security files**: `@terragon-labs/security-team`
- **Infrastructure**: `@terragon-labs/devops-team`  
- **Core AI modules**: `@terragon-labs/ai-ml-team`
- **Release workflows**: `@terragon-labs/release-team`

### Signed Commits

For production branches (`main`, `release/*`), signed commits are required to ensure:

- Commit authenticity
- Non-repudiation 
- Compliance with security policies

## Emergency Procedures

### Hotfix Process

For critical security or production issues:

1. Create hotfix branch from `main`: `hotfix/critical-fix-name`
2. Implement minimal fix with tests
3. Create PR with "HOTFIX" label
4. Require emergency review from:
   - Security team (for security issues)
   - Core maintainers (for production issues)
5. After merge, immediately create release

### Branch Protection Override

Repository administrators can temporarily disable protection rules for:

- Emergency hotfixes
- Critical security patches
- Major refactoring with maintainer consensus

**Override Process:**
1. Document reason in GitHub issue
2. Get approval from 2+ core maintainers
3. Implement changes with additional review
4. Re-enable protection immediately after merge
5. Post-mortem review of override usage

## Monitoring and Compliance

### Branch Protection Compliance Dashboard

Monitor branch protection effectiveness:

- PR review compliance rates
- Status check pass rates  
- Security scan findings
- Branch protection violations

### Automated Compliance Checks

GitHub Actions automatically verify:

```yaml
# .github/workflows/compliance.yml
- name: Check Branch Protection
  run: |
    gh api repos/${{ github.repository }}/branches/main/protection \
      | jq '.required_pull_request_reviews.required_approving_review_count >= 2'
```

### Regular Audits

Monthly audits should verify:

- Branch protection rules are active
- Code owner assignments are current
- Status checks are functioning
- Security scans are comprehensive

## Troubleshooting

### Common Issues

**Status checks not required:**
- Verify workflow names match protection rules exactly
- Check that workflows run on PR events
- Ensure required contexts are spelled correctly

**Code owner reviews not enforced:**
- Verify CODEOWNERS file syntax
- Check team membership and permissions
- Ensure code owners have write access

**Signed commits failing:**
- Verify GPG keys are configured
- Check commit signing configuration
- Ensure all contributors have signed commits enabled

### Debug Commands

```bash
# Check current branch protection
gh api repos/:owner/:repo/branches/main/protection

# List required status checks
gh api repos/:owner/:repo/branches/main/protection \
  | jq '.required_status_checks.contexts'

# Verify code owners
gh api repos/:owner/:repo/contents/.github/CODEOWNERS
```

## Best Practices

1. **Start with minimal protection** and gradually increase restrictions
2. **Test branch protection rules** on development branches first
3. **Document exceptions** and review them regularly
4. **Train team members** on branch protection workflows
5. **Monitor compliance** and adjust rules based on team needs
6. **Regular audits** of protection rules and effectiveness

## Related Documentation

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Code Owners Guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [AI Scientist Contributing Guidelines](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)