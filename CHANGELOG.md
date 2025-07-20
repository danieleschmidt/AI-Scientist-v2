# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-07-20

### 🔒 **CRITICAL SECURITY UPDATE**

#### Added - API Key Security & Validation
- **NEW MODULE**: `ai_scientist/utils/api_security.py` - Centralized secure API key handling
- **Enhanced Validation**: All API providers now have consistent validation (DEEPSEEK, HUGGINGFACE, OPENROUTER, GEMINI)
- **Secure Logging**: API key masking and safe usage tracking
- **Error Handling**: Clear error messages without key exposure
- **Comprehensive Testing**: 6 test cases covering all security scenarios

#### Security Fixes
- **FIXED**: Direct `os.environ[KEY]` access in `ai_scientist/llm.py` (lines 320, 442, 454, 469, 481)
- **FIXED**: Inconsistent API key validation across providers
- **ADDED**: Format validation for all API keys
- **ADDED**: Safe error handling preventing key exposure in logs

## Previous Updates - 2025-07-20

### Added
- **Security Enhancement**: Replaced `os.popen` with secure `subprocess.run` calls in:
  - `ai_scientist/perform_icbinb_writeup.py:1047`
  - `ai_scientist/perform_writeup.py:684`
- **Test Suite**: Comprehensive test coverage with unittest framework:
  - Basic functionality tests
  - Security vulnerability tests  
  - LLM pattern validation tests
  - Utility function tests (including zip extraction)
  - Nested zip file handling tests
- **CI/CD Pipeline**: Complete workflow templates provided in `CI_CD_SETUP_GUIDE.md`:
  - Automated testing across Python 3.8-3.11
  - Security scanning with Bandit and Safety
  - Code quality checks and syntax validation
- **Enhanced ZIP Handling**: 
  - Recursive nested zip file extraction with depth protection
  - Improved file validation with size-based content checking
  - Graceful handling of corrupted zip files
- **Code Quality**: Pre-commit hooks for:
  - Code formatting (Black, isort)
  - Linting (flake8)
  - Security scanning (Bandit)
  - Custom security checks
- **Development Tools**:
  - `run_tests.py` - Comprehensive test runner
  - `pytest.ini` - Test configuration
  - `BACKLOG.md` - WSJF-prioritized development backlog

### Changed
- Enhanced `extract_archives()` function in `ai_scientist/treesearch/utils/__init__.py`:
  - Added recursive nested zip extraction (max depth: 3)
  - Improved error handling for corrupted zip files
  - Enhanced file collision detection with size validation
  - Better logging and debug information

### Fixed
- **Security Vulnerability**: Eliminated shell injection risks from `os.popen` usage
- **File Handling**: Improved robustness of zip file extraction and validation
- **Error Handling**: Better exception handling for file operations

### Security
- Replaced insecure shell command execution with safe subprocess calls
- Added comprehensive security scanning in CI/CD pipeline
- Implemented input validation for file operations
- Added detection for potential command injection patterns

### Technical Debt Reduction
- Added missing test coverage (was 0%, now has comprehensive test suite)
- Resolved TODO items for nested zip handling and file validation
- Established CI/CD pipeline for automated quality assurance
- Created structured backlog with impact-based prioritization

---

## Development Metrics
- **Test Coverage**: Added 20+ test cases across 6 test modules
- **Security Issues Fixed**: 2 critical (os.popen usage)
- **Code Quality**: Pre-commit hooks and automated checks
- **CI/CD**: Automated testing and security scanning
- **Documentation**: Backlog with WSJF prioritization framework