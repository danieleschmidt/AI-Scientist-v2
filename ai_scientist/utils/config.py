"""
Centralized configuration management system for AI Scientist.

This module provides a unified interface for managing all configuration values
throughout the AI Scientist system, replacing hardcoded values with a flexible,
environment-aware configuration system.

Features:
- YAML/JSON configuration file support
- Environment variable overrides
- Configuration validation with schemas
- Default value fallbacks
- Type conversion and validation
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
import logging

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """Configuration schema definition for validation."""
    key: str
    type: type
    default: Any
    description: str
    required: bool = False
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


class ConfigurationManager:
    """
    Centralized configuration management system.
    
    Supports hierarchical configuration loading:
    1. Default values (hardcoded fallbacks)
    2. Configuration files (ai_scientist_config.yaml/json)
    3. Environment variables (AI_SCIENTIST_*)
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, searches for default locations.
        """
        self._config = {}
        self._schema = self._define_schema()
        self._load_defaults()
        
        if config_file:
            self.load_config_file(config_file)
        else:
            self._load_default_config_files()
        
        self._load_environment_overrides()
        self._validate_config()
    
    def _define_schema(self) -> Dict[str, ConfigSchema]:
        """Define the configuration schema with all supported configuration keys."""
        return {
            # API Configuration
            "API_DEEPSEEK_BASE_URL": ConfigSchema(
                "API_DEEPSEEK_BASE_URL", str, "https://api.deepseek.com",
                "Base URL for DeepSeek API"
            ),
            "API_HUGGINGFACE_BASE_URL": ConfigSchema(
                "API_HUGGINGFACE_BASE_URL", str, 
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                "Base URL for HuggingFace API"
            ),
            "API_OPENROUTER_BASE_URL": ConfigSchema(
                "API_OPENROUTER_BASE_URL", str, "https://openrouter.ai/api/v1",
                "Base URL for OpenRouter API"
            ),
            "API_GEMINI_BASE_URL": ConfigSchema(
                "API_GEMINI_BASE_URL", str, 
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "Base URL for Google Gemini API"
            ),
            "API_SEMANTIC_SCHOLAR_BASE_URL": ConfigSchema(
                "API_SEMANTIC_SCHOLAR_BASE_URL", str,
                "https://api.semanticscholar.org/graph/v1/paper/search",
                "Base URL for Semantic Scholar API"
            ),
            
            # Model Configuration
            "AVAILABLE_LLM_MODELS": ConfigSchema(
                "AVAILABLE_LLM_MODELS", list, [
                    "claude-3-5-sonnet-20240620", "gpt-4o-mini", "o1-2024-12-17",
                    "gpt-4o-2024-11-20", "o1-preview-2024-09-12", "o3-mini-2025-01-31"
                ],
                "List of available LLM models"
            ),
            "AVAILABLE_VLM_MODELS": ConfigSchema(
                "AVAILABLE_VLM_MODELS", list, [
                    "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20240620"
                ],
                "List of available VLM models"
            ),
            "DEFAULT_MODEL_PLOT_AGG": ConfigSchema(
                "DEFAULT_MODEL_PLOT_AGG", str, "o3-mini-2025-01-31",
                "Default model for plot aggregation"
            ),
            "DEFAULT_MODEL_WRITEUP": ConfigSchema(
                "DEFAULT_MODEL_WRITEUP", str, "o1-preview-2024-09-12",
                "Default model for writeup generation"
            ),
            "DEFAULT_MODEL_CITATION": ConfigSchema(
                "DEFAULT_MODEL_CITATION", str, "gpt-4o-2024-11-20",
                "Default model for citation generation"
            ),
            "DEFAULT_MODEL_REVIEW": ConfigSchema(
                "DEFAULT_MODEL_REVIEW", str, "gpt-4o-2024-11-20",
                "Default model for review tasks"
            ),
            
            # Token Limits
            "MAX_LLM_TOKENS": ConfigSchema(
                "MAX_LLM_TOKENS", int, 4096,
                "Maximum tokens for LLM calls", min_value=1, max_value=200000
            ),
            "MAX_VLM_TOKENS": ConfigSchema(
                "MAX_VLM_TOKENS", int, 4096,
                "Maximum tokens for VLM calls", min_value=1, max_value=200000
            ),
            "DEFAULT_CLAUDE_MAX_TOKENS": ConfigSchema(
                "DEFAULT_CLAUDE_MAX_TOKENS", int, 8192,
                "Default max tokens for Claude models", min_value=1, max_value=200000
            ),
            
            # Timeout Configuration
            "LATEX_COMPILE_TIMEOUT": ConfigSchema(
                "LATEX_COMPILE_TIMEOUT", int, 30,
                "Timeout for LaTeX compilation in seconds", min_value=5, max_value=600
            ),
            "PDF_DETECTION_TIMEOUT": ConfigSchema(
                "PDF_DETECTION_TIMEOUT", int, 30,
                "Timeout for PDF detection in seconds", min_value=5, max_value=300
            ),
            "CHKTEX_TIMEOUT": ConfigSchema(
                "CHKTEX_TIMEOUT", int, 60,
                "Timeout for Chktex execution in seconds", min_value=10, max_value=600
            ),
            "INTERPRETER_TIMEOUT": ConfigSchema(
                "INTERPRETER_TIMEOUT", int, 3600,
                "Default interpreter timeout in seconds", min_value=60, max_value=86400
            ),
            
            # File Paths
            "LATEX_TEMPLATE_DIR_ICML": ConfigSchema(
                "LATEX_TEMPLATE_DIR_ICML", str, "ai_scientist/blank_icml_latex",
                "Path to ICML LaTeX template directory"
            ),
            "LATEX_TEMPLATE_DIR_ICBINB": ConfigSchema(
                "LATEX_TEMPLATE_DIR_ICBINB", str, "ai_scientist/blank_icbinb_latex",
                "Path to ICBINB LaTeX template directory"
            ),
            "FEWSHOT_EXAMPLES_DIR": ConfigSchema(
                "FEWSHOT_EXAMPLES_DIR", str, "ai_scientist/fewshot_examples",
                "Path to fewshot examples directory"
            ),
            
            # Default Settings
            "MAX_FIGURES_ALLOWED": ConfigSchema(
                "MAX_FIGURES_ALLOWED", int, 12,
                "Maximum number of figures allowed", min_value=1, max_value=50
            ),
            "DEFAULT_WRITEUP_RETRIES": ConfigSchema(
                "DEFAULT_WRITEUP_RETRIES", int, 3,
                "Default number of writeup retries", min_value=1, max_value=10
            ),
            "DEFAULT_CITATION_ROUNDS": ConfigSchema(
                "DEFAULT_CITATION_ROUNDS", int, 20,
                "Default number of citation rounds", min_value=1, max_value=100
            ),
            "DEFAULT_VLM_MAX_IMAGES": ConfigSchema(
                "DEFAULT_VLM_MAX_IMAGES", int, 25,
                "Default max images for VLM", min_value=1, max_value=1000
            ),
            "BATCH_VLM_MAX_IMAGES": ConfigSchema(
                "BATCH_VLM_MAX_IMAGES", int, 200,
                "Max images for batch VLM processing", min_value=1, max_value=1000
            ),
            "VLM_REVIEW_MAX_TOKENS": ConfigSchema(
                "VLM_REVIEW_MAX_TOKENS", int, 1000,
                "Max tokens for VLM review", min_value=100, max_value=10000
            ),
            
            # Temperature Values
            "TEMP_REPORT": ConfigSchema(
                "TEMP_REPORT", float, 1.0,
                "Temperature for report generation", min_value=0.0, max_value=2.0
            ),
            "TEMP_CODE": ConfigSchema(
                "TEMP_CODE", float, 1.0,
                "Temperature for code generation", min_value=0.0, max_value=2.0
            ),
            "TEMP_FEEDBACK": ConfigSchema(
                "TEMP_FEEDBACK", float, 0.5,
                "Temperature for feedback generation", min_value=0.0, max_value=2.0
            ),
            "TEMP_VLM_FEEDBACK": ConfigSchema(
                "TEMP_VLM_FEEDBACK", float, 0.5,
                "Temperature for VLM feedback", min_value=0.0, max_value=2.0
            ),
            "TEMP_REVIEW_DEFAULT": ConfigSchema(
                "TEMP_REVIEW_DEFAULT", float, 0.75,
                "Default temperature for reviews", min_value=0.0, max_value=2.0
            ),
            
            # Other Settings
            "DEFAULT_MODEL_SEED": ConfigSchema(
                "DEFAULT_MODEL_SEED", int, 0,
                "Default seed for model reproducibility", min_value=0
            ),
            "PAGE_LIMIT_NORMAL": ConfigSchema(
                "PAGE_LIMIT_NORMAL", int, 8,
                "Page limit for normal writeups", min_value=1, max_value=50
            ),
            "PAGE_LIMIT_ICBINB": ConfigSchema(
                "PAGE_LIMIT_ICBINB", int, 4,
                "Page limit for ICBINB submissions", min_value=1, max_value=20
            ),
            "SEMANTIC_SCHOLAR_RATE_LIMIT_DELAY": ConfigSchema(
                "SEMANTIC_SCHOLAR_RATE_LIMIT_DELAY", float, 1.0,
                "Rate limiting delay for Semantic Scholar API", min_value=0.1, max_value=10.0
            ),
            
            # Execution Configuration
            "EXEC_TIMEOUT": ConfigSchema(
                "EXEC_TIMEOUT", int, 3600,
                "Execution timeout in seconds", min_value=60, max_value=86400
            ),
            "NUM_WORKERS": ConfigSchema(
                "NUM_WORKERS", int, 4,
                "Number of parallel workers", min_value=1, max_value=32
            ),
            "EVAL_NUM_SEEDS": ConfigSchema(
                "EVAL_NUM_SEEDS", int, 3,
                "Number of evaluation seeds", min_value=1, max_value=10
            ),
        }
    
    def _load_defaults(self):
        """Load default values from schema."""
        for key, schema in self._schema.items():
            self._config[key] = schema.default
    
    def _load_default_config_files(self):
        """Load configuration from default file locations."""
        # Look for config files in order of preference
        config_paths = [
            Path("ai_scientist_config.yaml"),
            Path("ai_scientist_config.yml"),
            Path("ai_scientist_config.json"),
            Path("config/ai_scientist.yaml"),
            Path("config/ai_scientist.yml"),
            Path("config/ai_scientist.json"),
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                logger.info(f"Loading configuration from {config_path}")
                self.load_config_file(config_path)
                break
        else:
            logger.info("No configuration file found, using defaults")
    
    def load_config_file(self, config_file: Union[str, Path]):
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not HAS_YAML:
                        logger.warning(f"PyYAML not available, skipping YAML config file: {config_path}")
                        return
                    file_config = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
                    return
            
            # Update configuration with values from file
            for key, value in file_config.items():
                if key in self._schema:
                    self._config[key] = value
                else:
                    logger.warning(f"Unknown configuration key in file: {key}")
                    
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {e}")
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        env_prefix = "AI_SCIENTIST_"
        
        for key in self._schema:
            env_key = f"{env_prefix}{key}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Convert environment string to appropriate type
                try:
                    schema_type = self._schema[key].type
                    if schema_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif schema_type == int:
                        converted_value = int(env_value)
                    elif schema_type == float:
                        converted_value = float(env_value)
                    elif schema_type == list:
                        # Assume comma-separated values for lists
                        converted_value = [item.strip() for item in env_value.split(',')]
                    else:
                        converted_value = env_value
                    
                    self._config[key] = converted_value
                    logger.info(f"Environment override: {key} = {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid environment value for {env_key}: {env_value} ({e})")
    
    def _validate_config(self):
        """Validate configuration values against schema."""
        for key, schema in self._schema.items():
            value = self._config.get(key)
            
            # Check required values
            if schema.required and value is None:
                raise ValueError(f"Required configuration key missing: {key}")
            
            # Check type
            if value is not None and not isinstance(value, schema.type):
                logger.warning(f"Configuration value {key} has wrong type: expected {schema.type.__name__}, got {type(value).__name__}")
            
            # Check choices
            if schema.choices and value not in schema.choices:
                logger.warning(f"Configuration value {key} not in allowed choices: {schema.choices}")
            
            # Check numeric ranges
            if isinstance(value, (int, float)):
                if schema.min_value is not None and value < schema.min_value:
                    logger.warning(f"Configuration value {key} below minimum: {value} < {schema.min_value}")
                if schema.max_value is not None and value > schema.max_value:
                    logger.warning(f"Configuration value {key} above maximum: {value} > {schema.max_value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        if key not in self._schema:
            logger.warning(f"Setting unknown configuration key: {key}")
        
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()
    
    def export_config(self, output_file: Union[str, Path], format: str = 'yaml'):
        """
        Export current configuration to file.
        
        Args:
            output_file: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == 'yaml':
                    if not HAS_YAML:
                        raise ValueError("PyYAML not available for YAML export")
                    yaml.dump(self._config, f, default_flow_style=False, sort_keys=True)
                elif format.lower() == 'json':
                    json.dump(self._config, f, indent=2, sort_keys=True)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")


# Global configuration instance
_config_manager = None

def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def init_config(config_file: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """
    Initialize the global configuration manager.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigurationManager(config_file)
    return _config_manager

# Convenience functions for common access patterns
def get_api_config(provider: str) -> str:
    """Get API configuration for a provider."""
    config = get_config()
    key = f"API_{provider.upper()}_BASE_URL"
    return config.get(key, "")

def get_model_config(model_type: str) -> Union[str, List[str]]:
    """Get model configuration."""
    config = get_config()
    if model_type.upper() == "LLM":
        return config.get("AVAILABLE_LLM_MODELS", [])
    elif model_type.upper() == "VLM":
        return config.get("AVAILABLE_VLM_MODELS", [])
    else:
        return config.get(f"DEFAULT_MODEL_{model_type.upper()}", "")

def get_timeout_config(operation: str) -> int:
    """Get timeout configuration for an operation."""
    config = get_config()
    key = f"{operation.upper()}_TIMEOUT"
    return config.get(key, 30)

def get_temp_config(operation: str) -> float:
    """Get temperature configuration for an operation."""
    config = get_config()
    key = f"TEMP_{operation.upper()}"
    return config.get(key, 1.0)