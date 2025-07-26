#!/usr/bin/env python3
"""
Centralized configuration management system.
Implements the requirements from backlog item: configuration-management (WSJF: 3.5)

Acceptance criteria:
- Create centralized config system
- Move all hardcoded values to config files  
- Add environment-specific overrides
- Implement config validation
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    default_model: str = "gpt-4o-2024-11-20"


@dataclass 
class ModelConfig:
    """Configuration for specific models."""
    name: str
    max_tokens: int
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class AIScientistConfig:
    """Main configuration class for AI Scientist."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if not self.models:
            self.models = {
                'gpt4': ModelConfig(
                    name='gpt-4o-2024-11-20',
                    max_tokens=4096,
                    temperature=0.7
                ),
                'claude': ModelConfig(
                    name='claude-3-5-sonnet-20241022',
                    max_tokens=8192,
                    temperature=0.7
                ),
                'vlm': ModelConfig(
                    name='gpt-4o-2024-11-20',
                    max_tokens=4096,
                    temperature=0.7
                )
            }


class ConfigManager:
    """
    Centralized configuration manager for AI Scientist.
    
    Supports:
    - Loading from YAML/JSON config files
    - Environment variable overrides
    - Nested configuration access with dot notation
    - Configuration validation
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self._config = self._load_default_config()
            self._apply_environment_overrides()
    
    def _load_default_config(self) -> AIScientistConfig:
        """Load default configuration."""
        return AIScientistConfig()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_overrides = {
            'AI_SCIENTIST_MAX_TOKENS': ('llm.max_tokens', int),
            'AI_SCIENTIST_TEMPERATURE': ('llm.temperature', float),
            'AI_SCIENTIST_TIMEOUT': ('llm.timeout', int),
            'AI_SCIENTIST_DEFAULT_MODEL': ('llm.default_model', str),
        }
        
        for env_var, (config_path, type_func) in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    value = type_func(env_value)
                    self.set(config_path, value)
                    logger.info(f"Applied environment override {env_var}={value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment value for {env_var}: {env_value} ({e})")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'llm.max_tokens')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        parts = path.split('.')
        current = self._config
        
        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'llm.max_tokens')
            value: Value to set
        """
        parts = path.split('.')
        current = self._config
        
        # Navigate to parent of target attribute
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Create nested structure if needed
                setattr(current, part, {})
                current = getattr(current, part)
        
        # Set the target attribute
        final_part = parts[-1]
        if hasattr(current, final_part):
            setattr(current, final_part, value)
        else:
            current[final_part] = value
    
    def load_config(self, config_path: Union[str, Path]):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to YAML or JSON config file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Merge loaded config with existing config
            self._merge_config(data)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _merge_config(self, data: Dict[str, Any]):
        """Merge loaded config data with existing configuration."""
        if 'llm' in data:
            llm_data = data['llm']
            if 'max_tokens' in llm_data:
                self._config.llm.max_tokens = llm_data['max_tokens']
            if 'temperature' in llm_data:
                self._config.llm.temperature = llm_data['temperature']
            if 'timeout' in llm_data:
                self._config.llm.timeout = llm_data['timeout']
            if 'default_model' in llm_data:
                self._config.llm.default_model = llm_data['default_model']
        
        if 'models' in data:
            for model_name, model_data in data['models'].items():
                if model_name not in self._config.models:
                    self._config.models[model_name] = ModelConfig(
                        name=model_data.get('name', model_name),
                        max_tokens=model_data.get('max_tokens', 4096),
                        temperature=model_data.get('temperature', 0.7),
                        timeout=model_data.get('timeout', 60)
                    )
                else:
                    # Update existing model config
                    model_config = self._config.models[model_name]
                    for key, value in model_data.items():
                        if hasattr(model_config, key):
                            setattr(model_config, key, value)
    
    @staticmethod
    def validate_config(config_data: Dict[str, Any]):
        """
        Validate configuration data.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'llm' in config_data:
            llm_config = config_data['llm']
            
            if 'max_tokens' in llm_config:
                max_tokens = llm_config['max_tokens']
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    raise ValueError(f"Invalid max_tokens: {max_tokens} (must be positive integer)")
            
            if 'temperature' in llm_config:
                temperature = llm_config['temperature']
                if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
                    raise ValueError(f"Invalid temperature: {temperature} (must be 0-2)")
            
            if 'timeout' in llm_config:
                timeout = llm_config['timeout']
                if not isinstance(timeout, int) or timeout <= 0:
                    raise ValueError(f"Invalid timeout: {timeout} (must be positive integer)")
        
        if 'models' in config_data:
            models = config_data['models']
            if not isinstance(models, dict):
                raise ValueError("Models configuration must be a dictionary")
            
            for model_name, model_config in models.items():
                if not isinstance(model_config, dict):
                    raise ValueError(f"Model config for {model_name} must be a dictionary")
                
                if 'max_tokens' in model_config:
                    max_tokens = model_config['max_tokens']
                    if not isinstance(max_tokens, int) or max_tokens <= 0:
                        raise ValueError(f"Invalid max_tokens for {model_name}: {max_tokens}")


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(path: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration value.
    
    Args:
        path: Dot-separated path (e.g., 'llm.max_tokens')
        default: Default value if path not found
        
    Returns:
        Configuration value
    """
    return get_config_manager().get(path, default)