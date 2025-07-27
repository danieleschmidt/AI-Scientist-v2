"""
Unit tests for configuration management functionality.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import yaml

from ai_scientist.utils.config import ConfigManager, load_config, validate_config


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_init_with_default_config(self):
        """Test ConfigManager initialization with default configuration."""
        config_manager = ConfigManager()
        assert config_manager.config is not None
        assert "environment" in config_manager.config
        assert "models" in config_manager.config
    
    def test_init_with_custom_config_file(self, temp_dir):
        """Test ConfigManager initialization with custom config file."""
        config_file = temp_dir / "custom_config.yaml"
        config_data = {
            "environment": "test",
            "models": {"default": "gpt-4"},
            "debug": True
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(config_file=str(config_file))
        assert config_manager.config["environment"] == "test"
        assert config_manager.config["models"]["default"] == "gpt-4"
        assert config_manager.config["debug"] is True
    
    def test_get_existing_key(self):
        """Test getting an existing configuration key."""
        config_manager = ConfigManager()
        # Assuming default config has environment key
        result = config_manager.get("environment")
        assert result is not None
    
    def test_get_nested_key(self):
        """Test getting a nested configuration key."""
        config_manager = ConfigManager()
        config_manager.config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        result = config_manager.get("database.host")
        assert result == "localhost"
    
    def test_get_nonexistent_key_with_default(self):
        """Test getting a non-existent key with default value."""
        config_manager = ConfigManager()
        result = config_manager.get("nonexistent.key", default="default_value")
        assert result == "default_value"
    
    def test_get_nonexistent_key_without_default(self):
        """Test getting a non-existent key without default value."""
        config_manager = ConfigManager()
        result = config_manager.get("nonexistent.key")
        assert result is None
    
    def test_set_simple_key(self):
        """Test setting a simple configuration key."""
        config_manager = ConfigManager()
        config_manager.set("test_key", "test_value")
        assert config_manager.get("test_key") == "test_value"
    
    def test_set_nested_key(self):
        """Test setting a nested configuration key."""
        config_manager = ConfigManager()
        config_manager.set("nested.test.key", "nested_value")
        assert config_manager.get("nested.test.key") == "nested_value"
    
    def test_update_existing_config(self):
        """Test updating existing configuration with new values."""
        config_manager = ConfigManager()
        config_manager.config = {"existing": "value", "keep": "this"}
        
        new_config = {"existing": "updated", "new": "value"}
        config_manager.update(new_config)
        
        assert config_manager.get("existing") == "updated"
        assert config_manager.get("new") == "value"
        assert config_manager.get("keep") == "this"
    
    def test_save_config(self, temp_dir):
        """Test saving configuration to file."""
        config_file = temp_dir / "save_test.yaml"
        config_manager = ConfigManager()
        config_manager.config = {"test": "data", "save": True}
        
        config_manager.save(str(config_file))
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["test"] == "data"
        assert saved_config["save"] is True
    
    def test_reload_config(self, temp_dir):
        """Test reloading configuration from file."""
        config_file = temp_dir / "reload_test.yaml"
        initial_config = {"initial": "value"}
        updated_config = {"initial": "updated", "new": "value"}
        
        # Save initial config
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        config_manager = ConfigManager(config_file=str(config_file))
        assert config_manager.get("initial") == "value"
        
        # Update config file externally
        with open(config_file, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Reload and verify
        config_manager.reload()
        assert config_manager.get("initial") == "updated"
        assert config_manager.get("new") == "value"


class TestConfigFunctions:
    """Test cases for standalone config functions."""
    
    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "test_config.yaml"
        config_data = {
            "environment": "test",
            "debug": True,
            "api_keys": {
                "openai": "test-key",
                "anthropic": "test-key-2"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config["environment"] == "test"
        assert loaded_config["debug"] is True
        assert loaded_config["api_keys"]["openai"] == "test-key"
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_config_invalid_yaml(self, temp_dir):
        """Test loading configuration from invalid YAML file."""
        config_file = temp_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [[[")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))
    
    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        valid_config = {
            "environment": "development",
            "debug": True,
            "models": {
                "default": "gpt-4",
                "fallback": "gpt-3.5-turbo"
            },
            "api_keys": {
                "openai": "sk-test",
                "anthropic": "sk-ant-test"
            },
            "timeouts": {
                "default": 300,
                "long": 600
            }
        }
        
        # Should not raise any exception
        validate_config(valid_config)
    
    def test_validate_config_missing_required_fields(self):
        """Test validating configuration with missing required fields."""
        invalid_config = {
            "debug": True,
            # Missing required fields like environment, models, etc.
        }
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_types(self):
        """Test validating configuration with invalid field types."""
        invalid_config = {
            "environment": "development",
            "debug": "not_a_boolean",  # Should be boolean
            "models": "not_a_dict",    # Should be dict
            "timeouts": {
                "default": "not_an_int"  # Should be int
            }
        }
        
        with pytest.raises(ValueError, match="Invalid configuration type"):
            validate_config(invalid_config)


class TestEnvironmentVariableIntegration:
    """Test configuration integration with environment variables."""
    
    def test_config_with_env_var_override(self):
        """Test configuration override with environment variables."""
        with patch.dict(os.environ, {
            "AI_SCIENTIST_ENVIRONMENT": "production",
            "AI_SCIENTIST_DEBUG": "false",
            "AI_SCIENTIST_MAX_WORKERS": "8"
        }):
            config_manager = ConfigManager()
            # Config should be updated with environment variables
            assert config_manager.get("environment") == "production"
            assert config_manager.get("debug") is False
            assert config_manager.get("max_workers") == 8
    
    def test_config_env_var_type_conversion(self):
        """Test automatic type conversion for environment variables."""
        with patch.dict(os.environ, {
            "AI_SCIENTIST_DEBUG": "true",
            "AI_SCIENTIST_MAX_RETRIES": "5",
            "AI_SCIENTIST_TIMEOUT": "300.5",
            "AI_SCIENTIST_ENABLED_FEATURES": "feature1,feature2,feature3"
        }):
            config_manager = ConfigManager()
            assert isinstance(config_manager.get("debug"), bool)
            assert isinstance(config_manager.get("max_retries"), int)
            assert isinstance(config_manager.get("timeout"), float)
            assert isinstance(config_manager.get("enabled_features"), list)
    
    def test_config_env_var_precedence(self, temp_dir):
        """Test that environment variables take precedence over config file."""
        config_file = temp_dir / "precedence_test.yaml"
        config_data = {
            "environment": "development",
            "debug": False,
            "max_workers": 4
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch.dict(os.environ, {
            "AI_SCIENTIST_ENVIRONMENT": "production",
            "AI_SCIENTIST_DEBUG": "true"
            # max_workers not set in env, should come from file
        }):
            config_manager = ConfigManager(config_file=str(config_file))
            assert config_manager.get("environment") == "production"  # From env
            assert config_manager.get("debug") is True  # From env
            assert config_manager.get("max_workers") == 4  # From file


class TestConfigValidation:
    """Test configuration validation scenarios."""
    
    def test_api_key_validation(self):
        """Test API key format validation."""
        valid_configs = [
            {"api_keys": {"openai": "sk-1234567890abcdef"}},
            {"api_keys": {"anthropic": "sk-ant-1234567890abcdef"}},
            {"api_keys": {"gemini": "gemini-1234567890abcdef"}},
        ]
        
        for config in valid_configs:
            validate_config(config)  # Should not raise
    
    def test_timeout_validation(self):
        """Test timeout value validation."""
        # Valid timeouts
        valid_config = {
            "timeouts": {
                "default": 300,
                "long": 600,
                "short": 30
            }
        }
        validate_config(valid_config)
        
        # Invalid timeouts
        invalid_configs = [
            {"timeouts": {"default": -1}},     # Negative
            {"timeouts": {"default": 0}},      # Zero
            {"timeouts": {"default": "300"}},  # String instead of int
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                validate_config(config)
    
    def test_model_name_validation(self):
        """Test model name validation."""
        valid_models = [
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "gemini-pro"
        ]
        
        for model in valid_models:
            config = {"models": {"default": model}}
            validate_config(config)  # Should not raise
        
        # Invalid model names
        invalid_models = ["", "invalid-model", "gpt-5", "claude-4"]
        
        for model in invalid_models:
            config = {"models": {"default": model}}
            with pytest.raises(ValueError, match="Invalid model name"):
                validate_config(config)


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration management."""
    
    def test_full_config_lifecycle(self, temp_dir):
        """Test complete configuration lifecycle."""
        config_file = temp_dir / "lifecycle_test.yaml"
        
        # 1. Create initial config
        initial_config = {
            "environment": "test",
            "debug": True,
            "models": {"default": "gpt-4"}
        }
        
        config_manager = ConfigManager()
        config_manager.config = initial_config
        config_manager.save(str(config_file))
        
        # 2. Load config from file
        loaded_manager = ConfigManager(config_file=str(config_file))
        assert loaded_manager.get("environment") == "test"
        
        # 3. Update config
        loaded_manager.set("new_feature", True)
        loaded_manager.set("models.fallback", "gpt-3.5-turbo")
        
        # 4. Save updated config
        loaded_manager.save(str(config_file))
        
        # 5. Reload and verify
        final_manager = ConfigManager(config_file=str(config_file))
        assert final_manager.get("new_feature") is True
        assert final_manager.get("models.fallback") == "gpt-3.5-turbo"
        assert final_manager.get("environment") == "test"  # Original value preserved
    
    def test_config_with_secrets_handling(self, temp_dir):
        """Test configuration handling with secrets and sensitive data."""
        config_file = temp_dir / "secrets_test.yaml"
        
        # Config with placeholder secrets
        config_data = {
            "api_keys": {
                "openai": "${OPENAI_API_KEY}",
                "anthropic": "${ANTHROPIC_API_KEY}"
            },
            "database": {
                "password": "${DATABASE_PASSWORD}",
                "host": "localhost"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock environment with actual secrets
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-real-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-real-anthropic-key",
            "DATABASE_PASSWORD": "secret-db-password"
        }):
            config_manager = ConfigManager(config_file=str(config_file))
            
            # Secrets should be resolved from environment
            assert config_manager.get("api_keys.openai") == "sk-real-openai-key"
            assert config_manager.get("api_keys.anthropic") == "sk-ant-real-anthropic-key"
            assert config_manager.get("database.password") == "secret-db-password"
            assert config_manager.get("database.host") == "localhost"