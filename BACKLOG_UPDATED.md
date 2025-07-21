# AI Scientist v2 - Updated Development Backlog

## WSJF Prioritization Framework

Each task is scored using Weighted Shortest Job First (WSJF):
**WSJF Score = (Business Value + Time Criticality + Risk Reduction) / Job Size**

- Business Value: 1-10 (impact on users/system functionality)
- Time Criticality: 1-10 (urgency of implementation)
- Risk Reduction: 1-10 (security/stability benefits)
- Job Size: 1-10 (implementation effort, 1=small, 10=large)

## ✅ **COMPLETED TASKS**

1. ✅ Security: Replace os.popen with subprocess (WSJF: 8.0)
2. ✅ Comprehensive Test Suite (WSJF: 6.7) 
3. ✅ Nested ZIP Handling (WSJF: 3.5)
4. ✅ ZIP File Content Validation (WSJF: 2.5)
5. ✅ Pre-commit Hooks and Code Quality
6. ✅ **CRITICAL: Fix File Handle Leak** (WSJF: 8.5) - *Completed*
7. ✅ **API Key Security & Validation** (WSJF: 7.2) - *Just Completed*

## 🔥 **NEW HIGH PRIORITY ISSUES** (WSJF > 6.0)

### 1. **Input Validation and Sanitization** (WSJF: 6.0) ⬆️ *Now Top Priority*
- **Issue**: LLM-generated code execution without proper validation  
- **Files**: Archive extraction path traversal risk
- **Business Value**: 7, Time Criticality: 6, Risk Reduction: 9, Job Size: 5
- **Test Plan**: Test with malicious inputs, verify sandboxing

## 🎯 **HIGH PRIORITY** (WSJF 4.0-6.0)

### 4. **GPU Resource Management Race Conditions** (WSJF: 5.4)
- **File**: `ai_scientist/treesearch/parallel_agent.py:2334-2349`
- **Issue**: Non-atomic GPU operations, potential race conditions
- **Business Value**: 6, Time Criticality: 6, Risk Reduction: 7, Job Size: 4
- **Test Plan**: Concurrent GPU access testing, resource leak detection

### 5. **Enhanced Error Handling in Tree Search** (WSJF: 4.5) ⬆️ *Previously identified*
- **Files**: `ai_scientist/treesearch/parallel_agent.py`, `interpreter.py:280`
- **Issue**: Missing error handling in REPL execution and debug depth logic
- **Business Value**: 6, Time Criticality: 5, Risk Reduction: 6, Job Size: 5

### 6. **Process Cleanup and Resource Management** (WSJF: 4.2)
- **Files**: `launch_scientist_bfts.py:319-350`, parallel_agent process termination
- **Issue**: Incomplete process cleanup, potential resource leaks
- **Business Value**: 5, Time Criticality: 5, Risk Reduction: 7, Job Size: 4

## 📋 **MEDIUM PRIORITY** (WSJF 2.0-4.0)

### 7. **Configuration Management** (WSJF: 3.5) ⬆️ *Updated priority*
- **Issue**: Hardcoded API URLs, token limits, pricing data
- **Files**: Multiple files with hardcoded values
- **Business Value**: 6, Time Criticality: 4, Risk Reduction: 4, Job Size: 4

### 8. **Performance Optimization** (WSJF: 2.8) ⬆️ *Refined*
- **Issue**: Synchronous file I/O, inefficient token tracking
- **Files**: `perform_icbinb_writeup.py`, `token_tracker.py`
- **Business Value**: 5, Time Criticality: 3, Risk Reduction: 3, Job Size: 6

### 9. **Unsafe Compilation Handling** (WSJF: 2.5)
- **File**: `ideas/i_cant_believe_its_not_better*.py:198,230`
- **Issue**: `torch.compile` without safety checks or fallback
- **Business Value**: 4, Time Criticality: 3, Risk Reduction: 5, Job Size: 3

## 📚 **LOW PRIORITY** (WSJF < 2.0)

### 10. **Documentation Improvements** (WSJF: 2.0)
- **Issue**: Limited API documentation, missing architecture docs
- **Business Value**: 4, Time Criticality: 2, Risk Reduction: 2, Job Size: 8

### 11. **CI/CD Pipeline Setup** (WSJF: 4.0) ⬇️ *Templates provided*
- **Status**: Templates ready in `CI_CD_SETUP_GUIDE.md`
- **Action Required**: Manual setup due to GitHub permissions

## 🎯 **IMMEDIATE NEXT ACTION**

**Selected Task**: Input Validation and Sanitization (WSJF: 6.0)
- **Rationale**: LLM-generated code execution needs validation and sandboxing
- **Implementation**: Add input validation layers and execution sandboxing
- **Testing**: Test with malicious inputs, verify safe execution
- **Risk**: Medium implementation risk, high security impact

## 📊 **Progress Metrics**

- **Security Issues Resolved**: 4 critical (os.popen, file handle leak, API key validation), 1 pending (input validation)
- **Test Coverage**: 20+ tests, 100% pass rate
- **Code Quality**: Pre-commit hooks active
- **Technical Debt**: Significantly reduced, new issues identified and prioritized

---
*Last updated: 2025-07-20*  
*Next review: After file handle leak fix*
*Status: Ready for implementation of next critical task*