# Security Policy

## Supported Versions

We provide security updates for the following versions of AI Scientist v2:

| Version | Supported          |
| ------- | ------------------ |
| main    | ✅ Latest development |
| Latest Release | ✅ Fully supported |
| Previous Release | ⚠️ Critical security fixes only |
| Older versions | ❌ Not supported |

## Security Features

AI Scientist v2 includes several security measures:

### 🔒 Code Execution Security
- **LLM Code Validation**: Comprehensive AST-based analysis of generated code
- **Sandboxed Execution**: Restricted environments for code execution
- **Path Traversal Protection**: Prevents access to files outside working directories
- **Resource Limits**: CPU, memory, and time constraints on code execution
- **Input Sanitization**: Validation and sanitization of all external inputs

### 🛡️ API Security
- **Secure Key Handling**: Environment-based API key management
- **Key Validation**: Format validation and exposure prevention
- **Error Message Sanitization**: Prevents key leakage in error messages

### 📁 File System Security
- **Secure Path Operations**: Protection against path traversal attacks
- **File Access Controls**: Restricted file operations to designated directories
- **Safe Archive Handling**: Secure extraction of ZIP files with validation

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Create a Public Issue
For security vulnerabilities, please **do not** create a public GitHub issue immediately.

### 2. Report Privately First
- **Email**: Contact the maintainers directly with details
- **GitHub Security Advisory**: Use GitHub's private vulnerability reporting
- **Include**: Full details, reproduction steps, and potential impact

### 3. Information to Include
- Description of the vulnerability
- Steps to reproduce
- Potential impact and attack scenarios  
- Affected versions/components
- Suggested fix (if you have one)
- Your contact information for follow-up

### 4. Response Timeline
- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: Varies by severity (hours to weeks)
- **Disclosure**: Coordinated with reporter

## Security Severity Levels

### Critical 🔴
- Remote code execution
- Full system compromise
- API key/secret exposure
- **Response Time**: Immediate (within hours)

### High 🟠  
- Privilege escalation
- Significant data exposure
- Authentication bypass
- **Response Time**: Within 24-48 hours

### Medium 🟡
- Limited access
- Minor data exposure  
- Denial of service
- **Response Time**: Within 1 week

### Low 🟢
- Information disclosure
- Non-exploitable issues
- **Response Time**: Next release cycle

## Security Best Practices

When using AI Scientist v2:

### For Developers
- Always run in containerized/sandboxed environments
- Never commit API keys or secrets to version control
- Regularly update dependencies
- Review generated code before execution
- Use strong authentication for any web interfaces

### For System Administrators  
- Keep the system updated
- Monitor resource usage
- Implement network-level controls
- Regular security scans
- Backup and disaster recovery plans

### For Researchers
- Be cautious with external data sources
- Validate all inputs to the system
- Use appropriate access controls
- Document security configurations

## Known Security Considerations

### LLM-Generated Code Execution
- **Risk**: LLMs can generate malicious or unsafe code
- **Mitigation**: Multi-layer validation and sandboxing
- **Recommendation**: Always run in isolated environments

### External API Dependencies
- **Risk**: Third-party API vulnerabilities  
- **Mitigation**: Input validation and error handling
- **Recommendation**: Use API keys with minimal necessary permissions

### File System Access
- **Risk**: Path traversal and file system manipulation
- **Mitigation**: Path validation and access controls
- **Recommendation**: Run with minimal file system permissions

## Security Updates

Security updates are released as soon as possible after vulnerability discovery:

1. **Critical**: Hotfix releases within hours
2. **High**: Patch releases within days  
3. **Medium/Low**: Regular release cycles

Updates are announced through:
- GitHub Security Advisories
- Release notes
- Security mailing list (if available)

## Security Audit History

| Date | Type | Scope | Findings | Status |
|------|------|-------|----------|---------|
| 2025-01 | Internal | Code execution security | Multiple improvements implemented | ✅ Resolved |
| 2025-01 | Internal | API key handling | Enhanced validation added | ✅ Resolved |

## Compliance and Standards

AI Scientist v2 follows these security standards:
- OWASP Top 10 considerations
- Secure coding practices  
- Dependency vulnerability management
- Regular security testing

## Contact

For security-related questions or concerns:
- **Security Issues**: Use private reporting channels
- **General Security Questions**: Create a discussion or issue
- **Security Policy Questions**: Contact maintainers directly

---

**Remember**: When in doubt about security, err on the side of caution and report privately first.