# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-07-21

### 🛡️ **CRITICAL INPUT VALIDATION & CODE EXECUTION SECURITY**

#### Enhanced - LLM Code Execution Security
- **FIXED**: Direct `exec()` of LLM-generated code without validation (`interpreter.py:144`)
- **ADDED**: Comprehensive code safety validation before execution
- **IMPLEMENTED**: AST-based dangerous pattern detection (imports, builtins, patterns)
- **SECURITY**: Prevents code injection, system access, and malicious operations
- **COMPATIBILITY**: Maintains existing functionality while adding security layers

#### Enhanced - Archive Extraction Security  
- **REPLACED**: Unsafe `zip_ref.extractall()` with secure extraction (`utils/__init__.py:92`)
- **ADDED**: Path traversal protection and zip bomb detection
- **IMPLEMENTED**: File size limits and compression ratio validation
- **SECURITY**: Prevents directory traversal attacks and resource exhaustion

#### Enhanced - Configuration Security
- **FIXED**: Unsafe YAML loading with `FullLoader` → `safe_load` (`bfts_utils.py:60`)
- **ENHANCED**: JSON configuration validation with schema checking
- **ADDED**: XSS and injection pattern detection in configurations
- **SECURITY**: Prevents code injection via malicious YAML/JSON

#### Added - Comprehensive Security Module
- **NEW MODULE**: `ai_scientist/utils/input_validation.py` - Complete validation framework
- **FUNCTIONS**: Code safety, path validation, archive security, config validation
- **PATTERNS**: Dangerous import detection, builtin restriction, pattern matching
- **UTILITIES**: File sanitization, response validation, text content cleaning

### 🔒 **CRITICAL API KEY SECURITY ENHANCEMENT**

#### Enhanced - API Key Security & Validation
- **FIXED**: Direct `HF_TOKEN` access in idea files (`i_cant_believe_its_not_better*.py`)
- **ENHANCED**: Client creation with explicit API key validation for OpenAI & Anthropic
- **ADDED**: Secure API key handling in all model client creation paths
- **IMPROVED**: Consistent use of `get_api_key_secure()` across all providers
- **SECURITY**: Eliminated potential runtime failures from missing API keys
- **COMPREHENSIVE TESTING**: 12 test cases covering all security scenarios

#### Fixed - HuggingFace Token Security
- **REPLACED**: `os.environ["HF_TOKEN"]` with `get_api_key_secure("HF_TOKEN")`
- **ENHANCED**: Error handling prevents crashes on missing tokens
- **ADDED**: Proper import structure for security utilities

### 🛠️ **CRITICAL RESOURCE LEAK FIX**

#### Fixed - File Handle Leak
- **FIXED**: Critical file handle leak in `launch_scientist_bfts.py:165`
- **Enhanced**: `redirect_stdout_stderr_to_file()` context manager with proper exception handling
- **Added**: Graceful handling of file.close() failures
- **Risk Mitigation**: Prevents resource exhaustion under load
- **Comprehensive Testing**: 6 test cases covering all scenarios including exception paths

## [Previous] - 2025-07-20

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