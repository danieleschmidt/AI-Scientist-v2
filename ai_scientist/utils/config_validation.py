#!/usr/bin/env python3
"""
Configuration Validation Framework

Advanced configuration validation with schema validation, security checks,
and environment-specific validation rules for AI Scientist v2.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jsonschema import Draft7Validator, ValidationError
from rich.console import Console

console = Console()


class ConfigValidator:
    """Advanced configuration validator with security and compliance checks."""
    
    # Configuration schema definition
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            # API Configuration
            "API_DEEPSEEK_BASE_URL": {"type": "string", "format": "uri"},
            "API_HUGGINGFACE_BASE_URL": {"type": "string", "format": "uri"},
            "API_OPENROUTER_BASE_URL": {"type": "string", "format": "uri"},
            "API_GEMINI_BASE_URL": {"type": "string", "format": "uri"},
            "API_SEMANTIC_SCHOLAR_BASE_URL": {"type": "string", "format": "uri"},
            
            # Model Configuration
            "AVAILABLE_LLM_MODELS": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            },
            "AVAILABLE_VLM_MODELS": {
                "type": "array", 
                "items": {"type": "string"},
                "minItems": 1
            },
            
            # Default Models
            "DEFAULT_MODEL_PLOT_AGG": {"type": "string"},
            "DEFAULT_MODEL_WRITEUP": {"type": "string"},
            "DEFAULT_MODEL_CITATION": {"type": "string"},
            "DEFAULT_MODEL_REVIEW": {"type": "string"},
            
            # Token Limits
            "MAX_LLM_TOKENS": {"type": "integer", "minimum": 1, "maximum": 128000},
            "MAX_VLM_TOKENS": {"type": "integer", "minimum": 1, "maximum": 128000},
            "DEFAULT_CLAUDE_MAX_TOKENS": {"type": "integer", "minimum": 1, "maximum": 200000},
            
            # Timeout Configuration
            "LATEX_COMPILE_TIMEOUT": {"type": "integer", "minimum": 1, "maximum": 3600},
            "PDF_DETECTION_TIMEOUT": {"type": "integer", "minimum": 1, "maximum": 3600},
            "CHKTEX_TIMEOUT": {"type": "integer", "minimum": 1, "maximum": 3600},
            "INTERPRETER_TIMEOUT": {"type": "integer", "minimum": 1, "maximum": 7200},
            "EXEC_TIMEOUT": {"type": "integer", "minimum": 1, "maximum": 7200},
            
            # File Paths
            "LATEX_TEMPLATE_DIR_ICML": {"type": "string"},
            "LATEX_TEMPLATE_DIR_ICBINB": {"type": "string"},
            "FEWSHOT_EXAMPLES_DIR": {"type": "string"},
            
            # Limits and Settings
            "MAX_FIGURES_ALLOWED": {"type": "integer", "minimum": 1, "maximum": 50},
            "DEFAULT_WRITEUP_RETRIES": {"type": "integer", "minimum": 1, "maximum": 10},
            "DEFAULT_CITATION_ROUNDS": {"type": "integer", "minimum": 1, "maximum": 100},
            "DEFAULT_VLM_MAX_IMAGES": {"type": "integer", "minimum": 1, "maximum": 100},
            "BATCH_VLM_MAX_IMAGES": {"type": "integer", "minimum": 1, "maximum": 1000},
            "VLM_REVIEW_MAX_TOKENS": {"type": "integer", "minimum": 1, "maximum": 8192},
            
            # Temperature Values
            "TEMP_REPORT": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "TEMP_CODE": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "TEMP_FEEDBACK": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "TEMP_VLM_FEEDBACK": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "TEMP_REVIEW_DEFAULT": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            
            # Other Settings
            "DEFAULT_MODEL_SEED": {"type": "integer", "minimum": 0},
            "PAGE_LIMIT_NORMAL": {"type": "integer", "minimum": 1, "maximum": 100},
            "PAGE_LIMIT_ICBINB": {"type": "integer", "minimum": 1, "maximum": 100},
            "SEMANTIC_SCHOLAR_RATE_LIMIT_DELAY": {"type": "number", "minimum": 0.1, "maximum": 10.0},
            "NUM_WORKERS": {"type": "integer", "minimum": 1, "maximum": 32},
            "EVAL_NUM_SEEDS": {"type": "integer", "minimum": 1, "maximum": 10}
        },
        "required": [
            "AVAILABLE_LLM_MODELS",
            "AVAILABLE_VLM_MODELS", 
            "DEFAULT_MODEL_WRITEUP",
            "MAX_LLM_TOKENS",
            "INTERPRETER_TIMEOUT"
        ]
    }
    
    # Security patterns to check for
    SECURITY_PATTERNS = {
        "api_key_exposure": [
            r"(api_key|apikey|key)\s*[:=]\s*['\"][a-zA-Z0-9_-]{10,}['\"]",
            r"(secret|password|token)\s*[:=]\s*['\"][a-zA-Z0-9_-]{10,}['\"]"
        ],
        "hardcoded_urls": [
            r"https?://[a-zA-Z0-9.-]+\.(?:com|org|net|io)/[a-zA-Z0-9._-]*/?[a-zA-Z0-9._=-]*"
        ],
        "sql_injection_risk": [
            r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+",
            r"(UNION|OR|AND)\s+\d+\s*=\s*\d+"
        ]
    }
    
    def __init__(self):
        self.validator = Draft7Validator(self.CONFIG_SCHEMA)
        self.validation_errors = []
        self.security_warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        self.validation_errors = []
        self.security_warnings = []
        
        # Schema validation
        schema_valid = self._validate_schema(config)
        
        # Business logic validation
        logic_valid = self._validate_business_logic(config)
        
        # Security validation
        security_valid = self._validate_security(config)
        
        # Path validation
        path_valid = self._validate_paths(config)
        
        # Model validation
        model_valid = self._validate_models(config)
        
        # Environment validation
        env_valid = self._validate_environment_consistency(config)
        
        all_valid = all([
            schema_valid, logic_valid, security_valid, 
            path_valid, model_valid, env_valid
        ])
        
        if not all_valid:
            self._display_validation_results()
        
        return all_valid
    
    def _validate_schema(self, config: Dict[str, Any]) -> bool:
        """Validate against JSON schema."""
        try:
            self.validator.validate(config)
            return True
        except ValidationError as e:
            self.validation_errors.append(f"Schema validation failed: {e.message}")
            return False
    
    def _validate_business_logic(self, config: Dict[str, Any]) -> bool:
        """Validate business logic rules."""
        valid = True
        
        # Token limits consistency
        if config.get("MAX_LLM_TOKENS", 0) > config.get("DEFAULT_CLAUDE_MAX_TOKENS", 0):
            self.validation_errors.append(
                "MAX_LLM_TOKENS cannot exceed DEFAULT_CLAUDE_MAX_TOKENS"
            )
            valid = False
        
        # Timeout relationships
        interpreter_timeout = config.get("INTERPRETER_TIMEOUT", 0)
        exec_timeout = config.get("EXEC_TIMEOUT", 0)
        if interpreter_timeout > 0 and exec_timeout > 0:
            if exec_timeout > interpreter_timeout:
                self.validation_errors.append(
                    "EXEC_TIMEOUT should not exceed INTERPRETER_TIMEOUT"
                )
                valid = False
        
        # VLM image limits
        default_vlm_max = config.get("DEFAULT_VLM_MAX_IMAGES", 0)
        batch_vlm_max = config.get("BATCH_VLM_MAX_IMAGES", 0)
        if default_vlm_max > 0 and batch_vlm_max > 0:
            if default_vlm_max > batch_vlm_max:
                self.validation_errors.append(
                    "DEFAULT_VLM_MAX_IMAGES should not exceed BATCH_VLM_MAX_IMAGES"
                )
                valid = False
        
        # Page limits
        normal_pages = config.get("PAGE_LIMIT_NORMAL", 0)
        icbinb_pages = config.get("PAGE_LIMIT_ICBINB", 0)
        if normal_pages > 0 and icbinb_pages > 0:
            if icbinb_pages > normal_pages:
                self.validation_errors.append(
                    "PAGE_LIMIT_ICBINB should not exceed PAGE_LIMIT_NORMAL"
                )
                valid = False
        
        # Worker and seed relationships
        num_workers = config.get("NUM_WORKERS", 0)
        eval_seeds = config.get("EVAL_NUM_SEEDS", 0)
        if num_workers > 0 and eval_seeds > 0:
            if num_workers < 3 and eval_seeds != num_workers:
                self.validation_errors.append(
                    "EVAL_NUM_SEEDS should equal NUM_WORKERS when NUM_WORKERS < 3"
                )
                valid = False
            elif num_workers >= 3 and eval_seeds != 3:
                self.validation_errors.append(
                    "EVAL_NUM_SEEDS should be 3 when NUM_WORKERS >= 3"
                )
                valid = False
        
        return valid
    
    def _validate_security(self, config: Dict[str, Any]) -> bool:
        """Check for security vulnerabilities."""
        valid = True
        config_str = yaml.dump(config)
        
        for category, patterns in self.SECURITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, config_str, re.IGNORECASE)
                if matches:
                    self.security_warnings.append(
                        f"Potential {category.replace('_', ' ')} detected: {matches[0][:50]}..."
                    )
                    if category in ["api_key_exposure", "sql_injection_risk"]:
                        valid = False
        
        return valid
    
    def _validate_paths(self, config: Dict[str, Any]) -> bool:
        """Validate file and directory paths."""
        valid = True
        path_keys = [
            "LATEX_TEMPLATE_DIR_ICML",
            "LATEX_TEMPLATE_DIR_ICBINB", 
            "FEWSHOT_EXAMPLES_DIR"
        ]
        
        for key in path_keys:
            if key in config:
                path = Path(config[key])
                if not path.exists():
                    self.validation_errors.append(f"Path does not exist: {config[key]}")
                    valid = False
                elif not path.is_dir():
                    self.validation_errors.append(f"Path is not a directory: {config[key]}")
                    valid = False
        
        return valid
    
    def _validate_models(self, config: Dict[str, Any]) -> bool:
        """Validate model configurations."""
        valid = True
        
        llm_models = config.get("AVAILABLE_LLM_MODELS", [])
        vlm_models = config.get("AVAILABLE_VLM_MODELS", [])
        
        default_models = [
            config.get("DEFAULT_MODEL_PLOT_AGG"),
            config.get("DEFAULT_MODEL_WRITEUP"),
            config.get("DEFAULT_MODEL_CITATION"),
            config.get("DEFAULT_MODEL_REVIEW")
        ]
        
        # Check if default models are in available models
        for model in default_models:
            if model and model not in llm_models:
                self.validation_errors.append(
                    f"Default model '{model}' not in AVAILABLE_LLM_MODELS"
                )
                valid = False
        
        # Validate model name patterns
        model_pattern = r"^[a-zA-Z0-9._-]+$"
        for model_list, list_name in [(llm_models, "LLM"), (vlm_models, "VLM")]:
            for model in model_list:
                if not re.match(model_pattern, model):
                    self.validation_errors.append(
                        f"Invalid {list_name} model name format: {model}"
                    )
                    valid = False
        
        return valid
    
    def _validate_environment_consistency(self, config: Dict[str, Any]) -> bool:
        """Validate consistency with environment variables."""
        valid = True
        
        # Check for environment variable overrides
        env_prefix = "AI_SCIENTIST_"
        for key in config.keys():
            env_key = f"{env_prefix}{key}"
            if env_key in os.environ:
                env_value = os.environ[env_key]
                config_value = config[key]
                
                # Type consistency check
                if isinstance(config_value, bool):
                    if env_value.lower() not in ["true", "false", "1", "0"]:
                        self.validation_errors.append(
                            f"Environment variable {env_key} should be boolean-like"
                        )
                        valid = False
                elif isinstance(config_value, int):
                    try:
                        int(env_value)
                    except ValueError:
                        self.validation_errors.append(
                            f"Environment variable {env_key} should be integer"
                        )
                        valid = False
                elif isinstance(config_value, float):
                    try:
                        float(env_value)
                    except ValueError:
                        self.validation_errors.append(
                            f"Environment variable {env_key} should be numeric"
                        )
                        valid = False
        
        return valid
    
    def _display_validation_results(self):
        """Display validation errors and warnings."""
        if self.validation_errors:
            console.print("[red]❌ Configuration Validation Errors:[/red]")
            for error in self.validation_errors:
                console.print(f"  • {error}")
        
        if self.security_warnings:
            console.print("[yellow]⚠️  Security Warnings:[/yellow]")
            for warning in self.security_warnings:
                console.print(f"  • {warning}")
    
    def validate_bfts_config(self, bfts_config: Dict[str, Any]) -> bool:
        """Validate BFTS (Best-First Tree Search) configuration."""
        valid = True
        
        # Agent configuration validation
        agent_config = bfts_config.get("agent", {})
        
        # Validate worker counts
        num_workers = agent_config.get("num_workers", 0)
        if num_workers < 1 or num_workers > 32:
            self.validation_errors.append("num_workers must be between 1 and 32")
            valid = False
        
        # Validate stage iterations
        stages = agent_config.get("stages", {})
        for stage, iters in stages.items():
            if isinstance(iters, int) and (iters < 1 or iters > 100):
                self.validation_errors.append(f"{stage} iterations must be between 1 and 100")
                valid = False
        
        # Validate model configurations
        code_config = agent_config.get("code", {})
        feedback_config = agent_config.get("feedback", {})
        
        if "model" not in code_config:
            self.validation_errors.append("Missing code model configuration")
            valid = False
        
        if "model" not in feedback_config:
            self.validation_errors.append("Missing feedback model configuration")
            valid = False
        
        # Validate search parameters
        search_config = agent_config.get("search", {})
        
        max_debug_depth = search_config.get("max_debug_depth", 0)
        if max_debug_depth < 1 or max_debug_depth > 10:
            self.validation_errors.append("max_debug_depth must be between 1 and 10")
            valid = False
        
        debug_prob = search_config.get("debug_prob", 0)
        if debug_prob < 0.0 or debug_prob > 1.0:
            self.validation_errors.append("debug_prob must be between 0.0 and 1.0")
            valid = False
        
        return valid


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Main configuration validation function.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_config(config)


def validate_bfts_config(bfts_config: Dict[str, Any]) -> bool:
    """
    BFTS configuration validation function.
    
    Args:
        bfts_config: BFTS configuration dictionary to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_bfts_config(bfts_config)


def load_and_validate_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration file and perform validation.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict or None: Validated configuration or None if validation fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if validate_config(config):
            return config
        else:
            return None
            
    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        return None


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        config = load_and_validate_config(config_file)
        if config:
            console.print("[green]✅ Configuration validation passed[/green]")
        else:
            console.print("[red]❌ Configuration validation failed[/red]")
            sys.exit(1)
    else:
        console.print("Usage: python config_validation.py <config_file>")
        sys.exit(1)