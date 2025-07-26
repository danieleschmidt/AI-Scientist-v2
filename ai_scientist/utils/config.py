"""
Centralized configuration management for AI Scientist.
Implements the requirements from backlog item: configuration-management (WSJF: 3.5)

Acceptance criteria:
- Create centralized config system
- Move all hardcoded values to config files  
- Add environment-specific overrides
- Implement config validation
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "models": {
        "gpt_models": [
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20", 
            "gpt-4o-mini-2024-07-18"
        ],
        "default_max_tokens": 4096,
        "anthropic_max_tokens": 8192,
        "vision_capable_models": [
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06", 
            "gpt-4o-2024-11-20",
            "gpt-4o-mini-2024-07-18"
        ]
    },
    "apis": {
        "semantic_scholar_base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
        "huggingface_base_url": "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
        "deepseek_base_url": "https://api.deepseek.com",
        "openrouter_base_url": "https://openrouter.ai/api/v1"
    },
    "pricing": {
        "gpt-4o-2024-11-20": {
            "prompt": 2.5 / 1000000,  # $2.5 per 1M tokens
            "completion": 10.0 / 1000000,  # $10.0 per 1M tokens
            "cached_prompt": 1.25 / 1000000  # $1.25 per 1M tokens
        },
        "gpt-4o-2024-08-06": {
            "prompt": 2.5 / 1000000,
            "completion": 10.0 / 1000000,
            "cached_prompt": 1.25 / 1000000
        },
        "gpt-4o-2024-05-13": {
            "prompt": 5.0 / 1000000,  # $5.0 per 1M tokens (no cached tokens)
            "completion": 15.0 / 1000000  # $15.0 per 1M tokens
        },
        "gpt-4o-mini-2024-07-18": {
            "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
            "completion": 0.6 / 1000000  # $0.6 per 1M tokens
        }
    },
    "timeouts": {
        "process_cleanup_interval": 0.1,
        "default_request_timeout": 30,
        "anthropic_function_call_timeout": 60
    },
    "cdn": {
        "p5js_url": "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js",
        "highlight_css_url": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css",
        "highlight_js_url": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js",
        "highlight_python_url": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"
    },
    "resource_management": {
        "max_memory_mb_per_process": 1024,
        "orphaned_process_keywords": ["python", "torch", "mp", "bfts", "experiment", "ai_scientist"],
        "gpu_cleanup_force": True
    }
}

# Environment variable mapping
ENV_VAR_MAPPING = {
    "AI_SCIENTIST_MAX_TOKENS": "models.default_max_tokens",
    "AI_SCIENTIST_ANTHROPIC_MAX_TOKENS": "models.anthropic_max_tokens",
    "AI_SCIENTIST_SEMANTIC_SCHOLAR_URL": "apis.semantic_scholar_base_url",
    "AI_SCIENTIST_HF_URL": "apis.huggingface_base_url",
    "AI_SCIENTIST_DEEPSEEK_URL": "apis.deepseek_base_url",
    "AI_SCIENTIST_OPENROUTER_URL": "apis.openrouter_base_url",
    "AI_SCIENTIST_REQUEST_TIMEOUT": "timeouts.default_request_timeout",
    "AI_SCIENTIST_CLEANUP_INTERVAL": "timeouts.process_cleanup_interval"
}

_config_cache = None
_config_file_path = None


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    # Try project root first
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.json"
    
    if config_path.exists():
        return config_path
    
    # Try user's home directory
    user_config = Path.home() / ".ai_scientist" / "config.json"
    if user_config.exists():
        return user_config
    
    # Return project root path (may not exist yet)
    return config_path


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from file with environment variable overrides.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_cache, _config_file_path
    
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    # Use cache if same file and already loaded
    if _config_cache is not None and _config_file_path == config_path:
        return _config_cache
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Deep merge configuration
            config = deep_merge(config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Config file {config_path} not found, using defaults")
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    # Cache the configuration
    _config_cache = config
    _config_file_path = config_path
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Args:
        config: Configuration dictionary to modify
        
    Returns:
        Configuration with environment overrides applied
    """
    for env_var, config_path in ENV_VAR_MAPPING.items():
        value = os.getenv(env_var)
        if value is not None:
            set_nested_value(config, config_path, value)
            logger.info(f"Applied environment override: {env_var} -> {config_path}")
    
    return config


def set_nested_value(config: Dict[str, Any], path: str, value: str) -> None:
    """
    Set a nested value in configuration using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        path: Dot-separated path to the value (e.g., "models.default_max_tokens")
        value: String value to set (will be converted to appropriate type)
    """
    keys = path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value with type conversion
    target_key = keys[-1]
    try:
        # Try to convert to appropriate type
        if value.lower() in ('true', 'false'):
            current[target_key] = value.lower() == 'true'
        elif value.isdigit():
            current[target_key] = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            current[target_key] = float(value)
        else:
            current[target_key] = value
    except Exception:
        # Fallback to string
        current[target_key] = value


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required keys.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_sections = ['models', 'apis', 'pricing', 'timeouts']
    
    # Check required top-level sections
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate models section
    models = config.get('models', {})
    if 'gpt_models' not in models or not isinstance(models['gpt_models'], list):
        logger.error("models.gpt_models must be a list")
        return False
    
    if 'default_max_tokens' not in models or not isinstance(models['default_max_tokens'], int):
        logger.error("models.default_max_tokens must be an integer")
        return False
    
    # Validate pricing section
    pricing = config.get('pricing', {})
    for model_name, model_pricing in pricing.items():
        if not isinstance(model_pricing, dict):
            logger.error(f"Pricing for {model_name} must be a dictionary")
            return False
        
        required_price_keys = ['prompt', 'completion']
        for key in required_price_keys:
            if key not in model_pricing:
                logger.error(f"Missing pricing key '{key}' for model {model_name}")
                return False
            
            if not isinstance(model_pricing[key], (int, float)):
                logger.error(f"Pricing key '{key}' for model {model_name} must be a number")
                return False
    
    logger.info("Configuration validation passed")
    return True


def save_config(config: Dict[str, Any], config_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Optional path to save configuration to
        
    Returns:
        True if saved successfully, False otherwise
    """
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate before saving
        if not validate_config(config):
            logger.error("Configuration validation failed, not saving")
            return False
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Clear cache to force reload
        global _config_cache, _config_file_path
        _config_cache = None
        _config_file_path = None
        
        return True
        
    except (IOError, OSError) as e:
        logger.error(f"Could not save configuration to {config_path}: {e}")
        return False


def get_model_list(model_type: str = 'gpt_models') -> list:
    """Get list of models from configuration."""
    config = load_config()
    return config.get('models', {}).get(model_type, [])


def get_api_url(api_name: str) -> str:
    """Get API URL from configuration."""
    config = load_config()
    return config.get('apis', {}).get(f"{api_name}_base_url", "")


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing information for a model."""
    config = load_config()
    return config.get('pricing', {}).get(model_name, {})


def get_timeout(timeout_type: str) -> Union[int, float]:
    """Get timeout value from configuration."""
    config = load_config()
    return config.get('timeouts', {}).get(timeout_type, 30)


# Initialize configuration on import
try:
    load_config()
except Exception as e:
    logger.warning(f"Could not initialize configuration: {e}")