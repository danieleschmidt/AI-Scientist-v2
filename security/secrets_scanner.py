#!/usr/bin/env python3
"""
Secrets scanner for AI Scientist v2.
Detects and prevents exposure of sensitive information.
"""

import re
import os
import hashlib
import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets that can be detected."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    DATABASE_URL = "database_url"
    GENERIC_SECRET = "generic_secret"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"


@dataclass
class SecretMatch:
    """Represents a detected secret."""
    secret_type: SecretType
    value: str
    line_number: int
    column_start: int
    column_end: int
    context: str
    file_path: str
    confidence: float
    masked_value: str


class SecretsScanner:
    """Advanced secrets detection scanner."""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.whitelist = self._load_whitelist()
        self.entropy_threshold = 4.5
        
    def _load_patterns(self) -> Dict[SecretType, List[re.Pattern]]:
        """Load regex patterns for different secret types."""
        patterns = {
            SecretType.API_KEY: [
                # OpenAI API keys
                re.compile(r'\bsk-[a-zA-Z0-9]{48}\b', re.IGNORECASE),
                # Anthropic API keys
                re.compile(r'\bsk-ant-[a-zA-Z0-9-]{95}\b', re.IGNORECASE),
                # Generic API keys
                re.compile(r'\bapi[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{32,})', re.IGNORECASE),
                re.compile(r'\b[a-zA-Z0-9]{32,}\b'),  # Generic long strings
            ],
            
            SecretType.PASSWORD: [
                re.compile(r'\bpassword["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})', re.IGNORECASE),
                re.compile(r'\bpwd["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})', re.IGNORECASE),
                re.compile(r'\bpass["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})', re.IGNORECASE),
            ],
            
            SecretType.TOKEN: [
                # JWT tokens
                re.compile(r'\beyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b'),
                # GitHub tokens
                re.compile(r'\bghp_[a-zA-Z0-9]{36}\b'),
                re.compile(r'\bgho_[a-zA-Z0-9]{36}\b'),
                re.compile(r'\bghu_[a-zA-Z0-9]{36}\b'),
                re.compile(r'\bghs_[a-zA-Z0-9]{36}\b'),
                re.compile(r'\bghr_[a-zA-Z0-9]{36}\b'),
                # Generic tokens
                re.compile(r'\btoken["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', re.IGNORECASE),
            ],
            
            SecretType.PRIVATE_KEY: [
                re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.DOTALL),
                re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----', re.DOTALL),
            ],
            
            SecretType.DATABASE_URL: [
                re.compile(r'\b(?:postgres|mysql|mongodb|redis)://[^\s"\']+', re.IGNORECASE),
                re.compile(r'\bconnection[_-]?string["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
            ],
            
            SecretType.CREDIT_CARD: [
                # Credit card numbers (simplified)
                re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            ],
            
            SecretType.SSN: [
                # US Social Security Numbers
                re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b'),
                re.compile(r'\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b'),
            ],
            
            SecretType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ],
            
            SecretType.IP_ADDRESS: [
                re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
                re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),  # IPv6
            ],
        }
        
        # Compile all patterns
        compiled_patterns = {}
        for secret_type, pattern_list in patterns.items():
            compiled_patterns[secret_type] = pattern_list
            
        return compiled_patterns
    
    def _load_whitelist(self) -> Set[str]:
        """Load whitelist of known false positives."""
        return {
            # Common false positives
            'your_api_key_here',
            'sk-your-openai-api-key-here',
            'sk-ant-your-anthropic-api-key-here',
            'test-api-key-1234567890abcdef',
            'fake_password',
            'password123',
            'changeme',
            'secret',
            'token',
            'key',
            # Test values
            'test_secret',
            'mock_token',
            'dummy_key',
            'example_password',
            'placeholder',
            # Common patterns
            '12345678',
            'abcdefgh',
            '********',
            'XXXXXXXX',
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        
        # Count character frequencies
        frequencies = {}
        for char in text:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for count in frequencies.values():
            probability = count / text_len
            entropy -= probability * (probability.bit_length() - 1) if probability > 0 else 0
        
        return entropy
    
    def _mask_secret(self, secret: str) -> str:
        """Mask a secret value for logging/reporting."""
        if len(secret) <= 8:
            return '*' * len(secret)
        else:
            return secret[:2] + '*' * (len(secret) - 4) + secret[-2:]
    
    def _is_whitelisted(self, value: str) -> bool:
        """Check if a value is in the whitelist."""
        return value.lower() in self.whitelist or value in self.whitelist
    
    def _calculate_confidence(self, secret_type: SecretType, value: str, context: str) -> float:
        """Calculate confidence score for a potential secret."""
        confidence = 0.5  # Base confidence
        
        # Entropy-based scoring
        entropy = self._calculate_entropy(value)
        if entropy > self.entropy_threshold:
            confidence += 0.3
        
        # Length-based scoring
        if len(value) >= 32:  # Typical for strong secrets
            confidence += 0.2
        
        # Context-based scoring
        context_lower = context.lower()
        secret_keywords = ['key', 'token', 'secret', 'password', 'auth', 'api']
        for keyword in secret_keywords:
            if keyword in context_lower:
                confidence += 0.1
                break
        
        # Secret type specific adjustments
        if secret_type == SecretType.API_KEY:
            if value.startswith(('sk-', 'pk-', 'sk_')):
                confidence += 0.3
        elif secret_type == SecretType.PRIVATE_KEY:
            confidence = 0.9  # High confidence for PEM format
        
        return min(confidence, 1.0)
    
    def scan_text(self, text: str, file_path: str = "") -> List[SecretMatch]:
        """Scan text for secrets."""
        matches = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for secret_type, patterns in self.patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(line):
                        value = match.group(0)
                        
                        # Skip if whitelisted
                        if self._is_whitelisted(value):
                            continue
                        
                        # Calculate confidence
                        confidence = self._calculate_confidence(
                            secret_type, value, line
                        )
                        
                        # Skip low confidence matches for certain types
                        if secret_type in [SecretType.API_KEY, SecretType.TOKEN] and confidence < 0.7:
                            continue
                        
                        secret_match = SecretMatch(
                            secret_type=secret_type,
                            value=value,
                            line_number=line_num,
                            column_start=match.start(),
                            column_end=match.end(),
                            context=line.strip(),
                            file_path=file_path,
                            confidence=confidence,
                            masked_value=self._mask_secret(value)
                        )
                        
                        matches.append(secret_match)
        
        return matches
    
    def scan_file(self, file_path: Path) -> List[SecretMatch]:
        """Scan a single file for secrets."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.scan_text(content, str(file_path))
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def scan_directory(self, directory: Path, 
                      include_patterns: List[str] = None,
                      exclude_patterns: List[str] = None) -> List[SecretMatch]:
        """Scan a directory for secrets."""
        if include_patterns is None:
            include_patterns = ['*.py', '*.js', '*.json', '*.yaml', '*.yml', '*.env*', '*.conf', '*.config']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '*.pyc', '*.pyo', '*.pyd', '__pycache__/*',
                '.git/*', '.svn/*', '.hg/*',
                'node_modules/*', 'venv/*', '.venv/*',
                '*.log', '*.tmp', '*.cache',
                'experiments/*', 'aisci_outputs/*', 'results/*', 'cache/*'
            ]
        
        all_matches = []
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                Path(root, d).match(pattern) for pattern in exclude_patterns
            )]
            
            for file in files:
                file_path = Path(root, file)
                
                # Check if file matches include patterns
                if not any(file_path.match(pattern) for pattern in include_patterns):
                    continue
                
                # Check if file matches exclude patterns
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue
                
                matches = self.scan_file(file_path)
                all_matches.extend(matches)
        
        return all_matches
    
    def generate_report(self, matches: List[SecretMatch]) -> Dict:
        """Generate a comprehensive secrets scan report."""
        if not matches:
            return {
                "status": "clean",
                "total_secrets": 0,
                "secrets_by_type": {},
                "secrets_by_file": {},
                "high_confidence_secrets": 0,
                "recommendations": ["No secrets detected. Good job!"]
            }
        
        # Group by type
        by_type = {}
        for match in matches:
            secret_type = match.secret_type.value
            if secret_type not in by_type:
                by_type[secret_type] = []
            by_type[secret_type].append({
                "file": match.file_path,
                "line": match.line_number,
                "masked_value": match.masked_value,
                "confidence": match.confidence,
                "context": match.context[:100] + "..." if len(match.context) > 100 else match.context
            })
        
        # Group by file
        by_file = {}
        for match in matches:
            file_path = match.file_path
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append({
                "type": match.secret_type.value,
                "line": match.line_number,
                "masked_value": match.masked_value,
                "confidence": match.confidence
            })
        
        # Count high confidence secrets
        high_confidence = sum(1 for match in matches if match.confidence >= 0.8)
        
        # Generate recommendations
        recommendations = []
        if high_confidence > 0:
            recommendations.append(f"URGENT: {high_confidence} high-confidence secrets detected. Review immediately.")
        
        if any(match.secret_type == SecretType.API_KEY for match in matches):
            recommendations.append("API keys detected. Move to environment variables or secure storage.")
        
        if any(match.secret_type == SecretType.PRIVATE_KEY for match in matches):
            recommendations.append("Private keys detected. Remove from code and use secure key management.")
        
        if any(match.secret_type == SecretType.PASSWORD for match in matches):
            recommendations.append("Hardcoded passwords detected. Use configuration or environment variables.")
        
        recommendations.append("Review all detected secrets and ensure they are not real credentials.")
        recommendations.append("Add false positives to the whitelist to reduce noise.")
        recommendations.append("Consider using git-secrets or similar tools in your CI/CD pipeline.")
        
        return {
            "status": "secrets_found",
            "total_secrets": len(matches),
            "secrets_by_type": by_type,
            "secrets_by_file": by_file,
            "high_confidence_secrets": high_confidence,
            "recommendations": recommendations,
            "scan_timestamp": "2025-01-27T00:00:00Z"  # Would use actual timestamp
        }
    
    def scan_and_report(self, target_path: Path) -> Dict:
        """Scan target and generate report."""
        if target_path.is_file():
            matches = self.scan_file(target_path)
        elif target_path.is_dir():
            matches = self.scan_directory(target_path)
        else:
            return {"status": "error", "message": f"Target {target_path} does not exist"}
        
        return self.generate_report(matches)


def main():
    """CLI interface for secrets scanner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan for secrets in code")
    parser.add_argument("path", help="Path to file or directory to scan")
    parser.add_argument("--output", "-o", help="Output file for report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create scanner and run scan
    scanner = SecretsScanner()
    target_path = Path(args.path)
    
    print(f"Scanning {target_path} for secrets...")
    report = scanner.scan_and_report(target_path)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    if report["status"] == "secrets_found":
        high_confidence = report.get("high_confidence_secrets", 0)
        if high_confidence > 0:
            print(f"\n❌ CRITICAL: {high_confidence} high-confidence secrets found!")
            exit(2)
        else:
            print(f"\n⚠️  WARNING: {report['total_secrets']} potential secrets found.")
            exit(1)
    else:
        print("\n✅ No secrets detected.")
        exit(0)


if __name__ == "__main__":
    main()