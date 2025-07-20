# AI Scientist v2 - Development Backlog

## WSJF Prioritization Framework

Each task is scored using Weighted Shortest Job First (WSJF):
**WSJF Score = (Business Value + Time Criticality + Risk Reduction) / Job Size**

- Business Value: 1-10 (impact on users/system functionality)
- Time Criticality: 1-10 (urgency of implementation)
- Risk Reduction: 1-10 (security/stability benefits)
- Job Size: 1-10 (implementation effort, 1=small, 10=large)

## High Priority (WSJF > 5.0)

### 1. Security: Replace os.popen with subprocess (WSJF: 8.0)
- **File**: `ai_scientist/perform_icbinb_writeup.py:1047`, `ai_scientist/perform_writeup.py:684`
- **Issue**: Using `os.popen()` for shell commands poses security risks
- **Impact**: High security vulnerability, code injection risk
- **Effort**: Low (simple subprocess replacement)
- **Business Value**: 9, Time Criticality: 8, Risk Reduction: 10, Job Size: 3
- **Test Plan**: Verify chktex command execution works with subprocess

### 2. Add Comprehensive Test Suite (WSJF: 6.7)
- **Issue**: No test coverage detected, critical for scientific software reliability
- **Impact**: Code quality, bug prevention, confidence in results
- **Effort**: Medium-High (setting up pytest, writing unit/integration tests)
- **Business Value**: 8, Time Criticality: 7, Risk Reduction: 8, Job Size: 7
- **Test Plan**: 
  - Unit tests for core components (LLM, VLM, agent_manager)
  - Integration tests for tree search pipeline
  - Mock external API calls (OpenAI, Anthropic, Semantic Scholar)

### 3. Add Input Validation and Sanitization (WSJF: 6.0)
- **Issue**: LLM-generated code execution without proper validation
- **Impact**: Security, system stability
- **Effort**: Medium (add validation layers)
- **Business Value**: 7, Time Criticality: 6, Risk Reduction: 9, Job Size: 5
- **Test Plan**: Test with malicious inputs, verify sandboxing

## Medium Priority (WSJF 3.0-5.0)

### 4. Enhanced Error Handling in Tree Search (WSJF: 4.5)
- **Issue**: Debug depth logic and error propagation could be improved
- **Impact**: Better experiment reliability and debugging
- **Effort**: Medium
- **Business Value**: 6, Time Criticality: 5, Risk Reduction: 6, Job Size: 5
- **Files**: `ai_scientist/treesearch/parallel_agent.py:1490-1993`

### 5. CI/CD Pipeline with GitHub Actions (WSJF: 4.0)
- **Issue**: No automated testing or deployment pipeline
- **Impact**: Development velocity, code quality
- **Effort**: Medium
- **Business Value**: 6, Time Criticality: 4, Risk Reduction: 6, Job Size: 4
- **Test Plan**: Automated testing, linting, security checks

### 6. Handle Nested ZIP Files (WSJF: 3.5)
- **File**: `ai_scientist/treesearch/utils/__init__.py:53`
- **Issue**: TODO comment indicates missing functionality
- **Impact**: Data extraction reliability
- **Effort**: Medium
- **Business Value**: 5, Time Criticality: 3, Risk Reduction: 4, Job Size: 4

### 7. Add Configuration Management (WSJF: 3.0)
- **Issue**: Hardcoded values, scattered configuration
- **Impact**: Deployment flexibility, maintainability
- **Effort**: Medium
- **Business Value**: 5, Time Criticality: 3, Risk Reduction: 4, Job Size: 4

## Low Priority (WSJF < 3.0)

### 8. ZIP File Content Validation (WSJF: 2.5)
- **File**: `ai_scientist/treesearch/utils/__init__.py:64`
- **Issue**: Missing content verification before file operations
- **Impact**: Data integrity
- **Effort**: Low-Medium
- **Business Value**: 3, Time Criticality: 2, Risk Reduction: 5, Job Size: 4

### 9. Documentation Improvements (WSJF: 2.0)
- **Issue**: Limited API documentation, missing architecture docs
- **Impact**: Developer onboarding, maintenance
- **Effort**: High
- **Business Value**: 4, Time Criticality: 2, Risk Reduction: 2, Job Size: 8

### 10. Performance Optimization (WSJF: 1.8)
- **Issue**: Tree search could be optimized for larger experiments
- **Impact**: Runtime efficiency
- **Effort**: High
- **Business Value**: 5, Time Criticality: 2, Risk Reduction: 2, Job Size: 9

## Architectural Debt

- **No Test Infrastructure**: Critical for scientific software reliability
- **Security Vulnerabilities**: Shell injection risks from os.popen usage
- **Hardcoded Dependencies**: Configuration scattered across files
- **Missing Error Recovery**: Limited graceful degradation in tree search
- **Code Quality Tools**: No linting, formatting, or pre-commit hooks

## Next Actions

1. **Immediate**: Fix security vulnerability (os.popen → subprocess)
2. **Short-term**: Implement basic test suite with pytest
3. **Medium-term**: Add CI/CD pipeline and enhanced error handling
4. **Long-term**: Performance optimization and comprehensive documentation

---
*Last updated: 2025-07-20*
*Next review: After completion of top 3 priorities*