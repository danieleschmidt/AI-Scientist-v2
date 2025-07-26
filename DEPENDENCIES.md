# Dependency Management

This document outlines the dependency management strategy for the AI Scientist project.

## Overview

The project uses Python with dependencies managed through `requirements.txt`. All dependencies are pinned with minimum version constraints to ensure security and stability.

## Dependency Categories

### Core AI/ML Stack
- **LLM APIs**: `anthropic`, `openai`, `backoff` for API interactions
- **ML Framework**: `torch`, `torchvision` for machine learning operations  
- **Data Processing**: `numpy`, `pandas`, `transformers`, `datasets`
- **Token Management**: `tiktoken` for token counting and management

### System Integration
- **Configuration**: `PyYAML`, `omegaconf`, `jsonschema` for configuration management
- **Process Management**: `psutil` for resource monitoring and process cleanup
- **File Processing**: `pypdf`, `PyMuPDF`, `pymupdf4llm` for PDF operations
- **Image Processing**: `Pillow` for image manipulation

### Development Tools
- **Code Quality**: `black` for code formatting
- **Data Structures**: `funcy`, `dataclasses-json` for enhanced data handling
- **Utilities**: `tqdm`, `rich`, `humanize`, `coolname` for user experience
- **Graph Processing**: `python-igraph` for graph operations

### External Services
- **HuggingFace**: `huggingface_hub` for model and dataset access
- **HTTP**: `requests` for HTTP requests
- **Interactive**: `IPython` for interactive computing support

## Security Considerations

### Version Pinning
All dependencies use minimum version constraints (`>=`) to:
- Ensure security patches are included
- Maintain compatibility with known working versions
- Allow automatic security updates

### Security Scanning
Regular dependency security scanning should be performed using:
```bash
# Install security scanning tools
pip install safety bandit

# Scan for known vulnerabilities
safety check -r requirements.txt

# Scan code for security issues
bandit -r ai_scientist/
```

### Dependency Audit Process
1. **Regular Reviews**: Dependencies should be reviewed quarterly
2. **Automated Testing**: All dependencies are tested via automated test suite
3. **Vulnerability Monitoring**: Use tools like `safety` to monitor for CVEs
4. **Minimal Dependencies**: Only include dependencies that are actively used

## Adding New Dependencies

When adding new dependencies:

1. **Check for Alternatives**: Prefer dependencies already in use
2. **Security Review**: Check security history and maintenance status
3. **Version Constraints**: Always specify minimum version with `>=`
4. **Testing**: Ensure new dependency works with existing test suite
5. **Documentation**: Update this file and relevant code documentation

### Example
```txt
# Good - specific minimum version
requests>=2.28.0

# Avoid - no version constraint
requests

# Avoid - overly restrictive
requests==2.28.0
```

## Dependency Testing

The project includes automated dependency testing via `tests/test_dependency_management.py`:

- **Import Coverage**: Verifies all imports have corresponding dependencies
- **Unused Dependencies**: Identifies and removes unused dependencies
- **Version Constraints**: Ensures all dependencies have proper version pins
- **Format Validation**: Validates requirements.txt format and prevents duplicates

Run dependency tests:
```bash
python -m unittest tests.test_dependency_management -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Usually indicates missing dependency in requirements.txt
2. **Version Conflicts**: May require updating minimum version constraints
3. **Security Alerts**: Update dependency to patched version

### Resolution Steps
1. Check if dependency is in requirements.txt
2. Verify version constraints allow security updates
3. Test functionality after dependency updates
4. Run full test suite to ensure compatibility

## Maintenance

### Monthly Tasks
- [ ] Run `safety check` for security vulnerabilities
- [ ] Review dependency update notifications
- [ ] Check for deprecated dependencies

### Quarterly Tasks
- [ ] Review all dependencies for necessity
- [ ] Update minimum version constraints
- [ ] Performance audit of dependency impact
- [ ] Documentation updates

---

*Last updated: 2025-07-26*  
*Managed by: Autonomous Backlog Management System*