# Security Policy

## Supported Versions

We actively support security updates for the following versions of AI Scientist v2:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT** create a public GitHub issue

### 2. Send a report to our security team
- **Email**: security@terragonlabs.ai
- **Subject**: "Security Vulnerability in AI Scientist v2"

### 3. Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if known)
- Your contact information

### 4. Responsible Disclosure
- We will acknowledge receipt within 24 hours
- We will provide a detailed response within 72 hours
- We will work with you to understand and resolve the issue
- We will credit you in our security advisory (unless you prefer anonymity)

## Security Measures

### Code Execution Sandbox
- All LLM-generated code runs in isolated containers
- Resource limits (CPU, memory, network access)
- File system restrictions and monitoring
- Process monitoring and cleanup

### API Security
- API key validation and secure storage
- Rate limiting and throttling
- Input sanitization and validation
- Audit logging for all API calls

### Data Protection
- No sensitive data in logs or repositories
- Encrypted data in transit and at rest
- Secure credential management
- GDPR and data privacy compliance

### Infrastructure Security
- Container security scanning
- Dependency vulnerability monitoring
- Regular security assessments
- Automated security testing in CI/CD

## Security Best Practices

### For Users
1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive configuration
3. **Keep dependencies updated** using automated tools
4. **Monitor resource usage** during experiments
5. **Review generated code** before execution

### For Developers
1. **Follow secure coding practices**
2. **Use static analysis tools** (bandit, semgrep)
3. **Implement proper error handling**
4. **Validate all inputs** and sanitize outputs
5. **Use principle of least privilege**

## Incident Response

In case of a security incident:

1. **Immediate containment** of the threat
2. **Assessment** of impact and scope
3. **Communication** to affected users
4. **Remediation** and patching
5. **Post-incident review** and improvements

## Security Updates

Security updates are released as soon as possible after a vulnerability is confirmed:

- **Critical**: Within 24 hours
- **High**: Within 72 hours
- **Medium**: Within 1 week
- **Low**: Next regular release

## Compliance

AI Scientist v2 is designed to comply with:

- **SOC 2 Type II** security standards
- **ISO 27001** information security management
- **GDPR** data protection requirements
- **CCPA** consumer privacy regulations

## Security Audits

We conduct regular security audits:

- **Quarterly** internal security reviews
- **Annual** third-party security assessments
- **Continuous** automated vulnerability scanning
- **On-demand** security testing for major releases

## Contact Information

- **Security Team**: security@terragonlabs.ai
- **General Support**: support@terragonlabs.ai
- **Bug Reports**: https://github.com/SakanaAI/AI-Scientist-v2/issues

---

*This security policy is regularly reviewed and updated. Last updated: 2025-07-27*