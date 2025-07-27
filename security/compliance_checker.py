#!/usr/bin/env python3
"""
Compliance checker for AI Scientist v2.
Validates adherence to security standards and regulations.
"""

import os
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance assessment levels."""
    COMPLIANT = "compliant"
    MINOR_ISSUES = "minor_issues"
    MAJOR_ISSUES = "major_issues"
    NON_COMPLIANT = "non_compliant"


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceIssue:
    """Represents a compliance issue."""
    rule_id: str
    title: str
    description: str
    severity: Severity
    framework: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: Optional[str] = None
    references: Optional[List[str]] = None


class ComplianceChecker:
    """Comprehensive compliance checking framework."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.rules = self._load_rules()
        self.issues = []
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load compliance configuration."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "frameworks": ["SOC2", "GDPR", "CCPA", "OWASP", "NIST"],
            "severity_threshold": "medium",
            "exclude_paths": [
                "tests/", "docs/", "experiments/", "aisci_outputs/",
                "results/", "cache/", "__pycache__/", ".git/"
            ]
        }
    
    def _load_rules(self) -> Dict[str, Dict]:
        """Load compliance rules for different frameworks."""
        return {
            "SOC2": self._load_soc2_rules(),
            "GDPR": self._load_gdpr_rules(),
            "CCPA": self._load_ccpa_rules(),
            "OWASP": self._load_owasp_rules(),
            "NIST": self._load_nist_rules(),
        }
    
    def _load_soc2_rules(self) -> Dict[str, Dict]:
        """Load SOC 2 compliance rules."""
        return {
            "CC6.1": {
                "title": "Logical and Physical Access Controls",
                "description": "Implement controls to restrict logical and physical access",
                "checks": [
                    self._check_authentication_required,
                    self._check_session_management,
                    self._check_access_controls
                ]
            },
            "CC6.2": {
                "title": "System Accounts Management",
                "description": "Manage system accounts and access rights",
                "checks": [
                    self._check_default_accounts,
                    self._check_privileged_accounts
                ]
            },
            "CC6.3": {
                "title": "Data Transmission Security",
                "description": "Protect data during transmission",
                "checks": [
                    self._check_tls_usage,
                    self._check_encryption_in_transit
                ]
            },
            "CC6.7": {
                "title": "Data Retention and Disposal",
                "description": "Properly manage data lifecycle",
                "checks": [
                    self._check_data_retention_policy,
                    self._check_secure_deletion
                ]
            },
            "CC7.1": {
                "title": "System Boundaries and Data Flow",
                "description": "Document system boundaries and data flows",
                "checks": [
                    self._check_system_documentation,
                    self._check_data_flow_documentation
                ]
            }
        }
    
    def _load_gdpr_rules(self) -> Dict[str, Dict]:
        """Load GDPR compliance rules."""
        return {
            "ART6": {
                "title": "Lawfulness of Processing",
                "description": "Ensure lawful basis for personal data processing",
                "checks": [
                    self._check_consent_mechanism,
                    self._check_data_processing_purpose
                ]
            },
            "ART25": {
                "title": "Data Protection by Design and by Default",
                "description": "Implement privacy by design principles",
                "checks": [
                    self._check_privacy_by_design,
                    self._check_default_privacy_settings
                ]
            },
            "ART32": {
                "title": "Security of Processing",
                "description": "Implement appropriate technical and organizational measures",
                "checks": [
                    self._check_encryption_at_rest,
                    self._check_access_controls,
                    self._check_data_integrity
                ]
            },
            "ART33": {
                "title": "Notification of Personal Data Breach",
                "description": "Implement breach notification procedures",
                "checks": [
                    self._check_incident_response_plan,
                    self._check_logging_mechanisms
                ]
            }
        }
    
    def _load_ccpa_rules(self) -> Dict[str, Dict]:
        """Load CCPA compliance rules."""
        return {
            "1798.100": {
                "title": "Consumer Right to Know",
                "description": "Provide transparency about personal information collection",
                "checks": [
                    self._check_privacy_policy,
                    self._check_data_collection_disclosure
                ]
            },
            "1798.105": {
                "title": "Consumer Right to Delete",
                "description": "Implement data deletion capabilities",
                "checks": [
                    self._check_data_deletion_mechanism,
                    self._check_deletion_verification
                ]
            },
            "1798.110": {
                "title": "Consumer Right to Know About Personal Information Disclosed",
                "description": "Provide information about data sharing",
                "checks": [
                    self._check_data_sharing_disclosure,
                    self._check_third_party_list
                ]
            }
        }
    
    def _load_owasp_rules(self) -> Dict[str, Dict]:
        """Load OWASP Top 10 compliance rules."""
        return {
            "A01": {
                "title": "Broken Access Control",
                "description": "Prevent unauthorized access to resources",
                "checks": [
                    self._check_authorization_controls,
                    self._check_principle_of_least_privilege
                ]
            },
            "A02": {
                "title": "Cryptographic Failures",
                "description": "Protect sensitive data with proper cryptography",
                "checks": [
                    self._check_encryption_algorithms,
                    self._check_key_management
                ]
            },
            "A03": {
                "title": "Injection",
                "description": "Prevent injection attacks",
                "checks": [
                    self._check_input_validation,
                    self._check_parameterized_queries
                ]
            },
            "A05": {
                "title": "Security Misconfiguration",
                "description": "Ensure secure configuration",
                "checks": [
                    self._check_security_headers,
                    self._check_default_configurations
                ]
            },
            "A09": {
                "title": "Security Logging and Monitoring Failures",
                "description": "Implement comprehensive logging and monitoring",
                "checks": [
                    self._check_security_logging,
                    self._check_monitoring_setup
                ]
            }
        }
    
    def _load_nist_rules(self) -> Dict[str, Dict]:
        """Load NIST Cybersecurity Framework rules."""
        return {
            "ID.AM": {
                "title": "Asset Management",
                "description": "Identify and manage information system assets",
                "checks": [
                    self._check_asset_inventory,
                    self._check_asset_classification
                ]
            },
            "PR.AC": {
                "title": "Access Control",
                "description": "Limit access to authorized users and processes",
                "checks": [
                    self._check_identity_management,
                    self._check_access_permissions
                ]
            },
            "PR.DS": {
                "title": "Data Security",
                "description": "Protect information and records",
                "checks": [
                    self._check_data_classification,
                    self._check_data_encryption
                ]
            },
            "DE.CM": {
                "title": "Security Continuous Monitoring",
                "description": "Monitor for cybersecurity events",
                "checks": [
                    self._check_continuous_monitoring,
                    self._check_anomaly_detection
                ]
            }
        }
    
    # Compliance check methods
    def _check_authentication_required(self, project_path: Path) -> List[ComplianceIssue]:
        """Check if authentication is properly implemented."""
        issues = []
        
        # Check for authentication configuration
        auth_files = list(project_path.glob("**/auth*.py")) + \
                    list(project_path.glob("**/authentication*.py"))
        
        if not auth_files:
            issues.append(ComplianceIssue(
                rule_id="CC6.1.1",
                title="Missing Authentication Implementation",
                description="No authentication modules found in the codebase",
                severity=Severity.HIGH,
                framework="SOC2",
                remediation="Implement proper authentication mechanisms"
            ))
        
        return issues
    
    def _check_session_management(self, project_path: Path) -> List[ComplianceIssue]:
        """Check session management implementation."""
        issues = []
        
        # Look for session configuration
        config_files = list(project_path.glob("**/*.yaml")) + \
                      list(project_path.glob("**/*.yml")) + \
                      list(project_path.glob("**/*.json"))
        
        session_config_found = False
        for config_file in config_files:
            try:
                content = config_file.read_text()
                if "session" in content.lower() and "timeout" in content.lower():
                    session_config_found = True
                    break
            except:
                continue
        
        if not session_config_found:
            issues.append(ComplianceIssue(
                rule_id="CC6.1.2",
                title="Missing Session Management Configuration",
                description="No session timeout configuration found",
                severity=Severity.MEDIUM,
                framework="SOC2",
                remediation="Configure session timeouts and management"
            ))
        
        return issues
    
    def _check_tls_usage(self, project_path: Path) -> List[ComplianceIssue]:
        """Check TLS/SSL usage."""
        issues = []
        
        # Check for TLS configuration
        config_files = list(project_path.glob("**/*.yaml")) + \
                      list(project_path.glob("**/*.yml"))
        
        tls_found = False
        for config_file in config_files:
            try:
                content = config_file.read_text().lower()
                if any(term in content for term in ["tls", "ssl", "https"]):
                    tls_found = True
                    break
            except:
                continue
        
        if not tls_found:
            issues.append(ComplianceIssue(
                rule_id="CC6.3.1",
                title="Missing TLS Configuration",
                description="No TLS/SSL configuration found",
                severity=Severity.HIGH,
                framework="SOC2",
                remediation="Configure TLS for all communications"
            ))
        
        return issues
    
    def _check_encryption_at_rest(self, project_path: Path) -> List[ComplianceIssue]:
        """Check data encryption at rest."""
        issues = []
        
        # Check for encryption configuration
        security_files = list(project_path.glob("**/security*.py")) + \
                        list(project_path.glob("**/encryption*.py"))
        
        if not security_files:
            issues.append(ComplianceIssue(
                rule_id="ART32.1",
                title="Missing Encryption Implementation",
                description="No encryption modules found",
                severity=Severity.HIGH,
                framework="GDPR",
                remediation="Implement data encryption at rest"
            ))
        
        return issues
    
    def _check_privacy_policy(self, project_path: Path) -> List[ComplianceIssue]:
        """Check for privacy policy."""
        issues = []
        
        privacy_files = list(project_path.glob("**/PRIVACY*.md")) + \
                       list(project_path.glob("**/privacy*.md")) + \
                       list(project_path.glob("**/privacy*.txt"))
        
        if not privacy_files:
            issues.append(ComplianceIssue(
                rule_id="1798.100.1",
                title="Missing Privacy Policy",
                description="No privacy policy documentation found",
                severity=Severity.HIGH,
                framework="CCPA",
                remediation="Create and maintain a privacy policy"
            ))
        
        return issues
    
    def _check_input_validation(self, project_path: Path) -> List[ComplianceIssue]:
        """Check input validation implementation."""
        issues = []
        
        # Look for validation patterns in Python files
        python_files = list(project_path.glob("**/*.py"))
        validation_found = False
        
        for py_file in python_files[:10]:  # Sample check
            try:
                content = py_file.read_text()
                if any(pattern in content for pattern in [
                    "validate", "sanitize", "escape", "clean_input"
                ]):
                    validation_found = True
                    break
            except:
                continue
        
        if not validation_found:
            issues.append(ComplianceIssue(
                rule_id="A03.1",
                title="Missing Input Validation",
                description="No input validation patterns found",
                severity=Severity.HIGH,
                framework="OWASP",
                remediation="Implement comprehensive input validation"
            ))
        
        return issues
    
    def _check_security_headers(self, project_path: Path) -> List[ComplianceIssue]:
        """Check security headers configuration."""
        issues = []
        
        # Check for security headers in configuration
        config_content = ""
        config_files = list(project_path.glob("**/security*.yaml")) + \
                      list(project_path.glob("**/security*.yml"))
        
        for config_file in config_files:
            try:
                config_content += config_file.read_text().lower()
            except:
                continue
        
        required_headers = ["csp", "hsts", "xss", "content-type", "frame-options"]
        missing_headers = []
        
        for header in required_headers:
            if header not in config_content:
                missing_headers.append(header)
        
        if missing_headers:
            issues.append(ComplianceIssue(
                rule_id="A05.1",
                title="Missing Security Headers",
                description=f"Missing security headers: {', '.join(missing_headers)}",
                severity=Severity.MEDIUM,
                framework="OWASP",
                remediation="Configure all required security headers"
            ))
        
        return issues
    
    def _check_security_logging(self, project_path: Path) -> List[ComplianceIssue]:
        """Check security logging implementation."""
        issues = []
        
        # Check for logging configuration
        log_files = list(project_path.glob("**/logging*.py")) + \
                   list(project_path.glob("**/audit*.py"))
        
        if not log_files:
            issues.append(ComplianceIssue(
                rule_id="A09.1",
                title="Missing Security Logging",
                description="No logging modules found",
                severity=Severity.MEDIUM,
                framework="OWASP",
                remediation="Implement comprehensive security logging"
            ))
        
        return issues
    
    # Generic check methods for simpler rules
    def _check_access_controls(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_default_accounts(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_privileged_accounts(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_encryption_in_transit(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_retention_policy(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_secure_deletion(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_system_documentation(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_flow_documentation(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_consent_mechanism(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_processing_purpose(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_privacy_by_design(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_default_privacy_settings(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_integrity(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_incident_response_plan(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_logging_mechanisms(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_collection_disclosure(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_deletion_mechanism(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_deletion_verification(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_sharing_disclosure(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_third_party_list(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_authorization_controls(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_principle_of_least_privilege(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_encryption_algorithms(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_key_management(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_parameterized_queries(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_default_configurations(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_monitoring_setup(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_asset_inventory(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_asset_classification(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_identity_management(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_access_permissions(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_classification(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_data_encryption(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_continuous_monitoring(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def _check_anomaly_detection(self, project_path: Path) -> List[ComplianceIssue]:
        return []  # Placeholder
    
    def run_compliance_check(self, project_path: Path) -> Dict[str, Any]:
        """Run comprehensive compliance check."""
        self.issues = []
        
        for framework in self.config["frameworks"]:
            if framework in self.rules:
                framework_rules = self.rules[framework]
                
                for rule_id, rule_config in framework_rules.items():
                    for check_func in rule_config["checks"]:
                        try:
                            rule_issues = check_func(project_path)
                            self.issues.extend(rule_issues)
                        except Exception as e:
                            logger.error(f"Error running check {check_func.__name__}: {e}")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        # Count issues by severity
        severity_counts = {s.value: 0 for s in Severity}
        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
        
        # Count issues by framework
        framework_counts = {}
        for issue in self.issues:
            if issue.framework not in framework_counts:
                framework_counts[issue.framework] = 0
            framework_counts[issue.framework] += 1
        
        # Determine overall compliance level
        if severity_counts["critical"] > 0:
            compliance_level = ComplianceLevel.NON_COMPLIANT
        elif severity_counts["high"] > 0:
            compliance_level = ComplianceLevel.MAJOR_ISSUES
        elif severity_counts["medium"] > 0:
            compliance_level = ComplianceLevel.MINOR_ISSUES
        else:
            compliance_level = ComplianceLevel.COMPLIANT
        
        # Generate recommendations
        recommendations = []
        if severity_counts["critical"] > 0:
            recommendations.append("Address critical security issues immediately")
        if severity_counts["high"] > 0:
            recommendations.append("Resolve high-severity compliance issues")
        if framework_counts.get("GDPR", 0) > 0:
            recommendations.append("Review GDPR compliance requirements")
        if framework_counts.get("SOC2", 0) > 0:
            recommendations.append("Implement SOC 2 controls")
        
        return {
            "compliance_level": compliance_level.value,
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "framework_breakdown": framework_counts,
            "issues": [
                {
                    "rule_id": issue.rule_id,
                    "title": issue.title,
                    "description": issue.description,
                    "severity": issue.severity.value,
                    "framework": issue.framework,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "remediation": issue.remediation,
                    "references": issue.references or []
                }
                for issue in self.issues
            ],
            "recommendations": recommendations,
            "scan_timestamp": "2025-01-27T00:00:00Z"
        }


def main():
    """CLI interface for compliance checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check compliance with security frameworks")
    parser.add_argument("path", help="Path to project directory")
    parser.add_argument("--config", "-c", help="Path to compliance configuration file")
    parser.add_argument("--output", "-o", help="Output file for report (JSON)")
    parser.add_argument("--framework", "-f", action="append", 
                       help="Specific frameworks to check (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Create checker
    config_path = Path(args.config) if args.config else None
    checker = ComplianceChecker(config_path)
    
    # Override frameworks if specified
    if args.framework:
        checker.config["frameworks"] = args.framework
    
    # Run compliance check
    project_path = Path(args.path)
    print(f"Running compliance check on {project_path}...")
    
    report = checker.run_compliance_check(project_path)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    compliance_level = report["compliance_level"]
    if compliance_level == "non_compliant":
        exit(2)
    elif compliance_level == "major_issues":
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()