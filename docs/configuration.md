# Configuration Management

The AI Scientist v2 uses a centralized configuration system to manage all configurable parameters.

## Overview

The configuration system provides:
- Centralized configuration management
- Environment variable overrides
- Configuration validation
- Support for YAML and JSON config files
- Dot notation access to nested values

## Usage

### Basic Usage

```python
from ai_scientist.utils.config_manager import get_config

# Get configuration values
max_tokens = get_config('llm.max_tokens', 4096)
temperature = get_config('llm.temperature', 0.7)
model_name = get_config('models.gpt4.name')
```

### Configuration Manager

```python
from ai_scientist.utils.config_manager import ConfigManager

config = ConfigManager()
value = config.get('llm.max_tokens')
config.set('llm.temperature', 0.8)
```

## Configuration Structure

The default configuration includes:

```yaml
llm:
  max_tokens: 4096
  temperature: 0.7
  timeout: 60
  default_model: "gpt-4o-2024-11-20"

models:
  gpt4:
    name: "gpt-4o-2024-11-20"
    max_tokens: 4096
    temperature: 0.7
  claude:
    name: "claude-3-5-sonnet-20241022"
    max_tokens: 8192
    temperature: 0.7
```

## Environment Variable Overrides

Environment variables can override configuration values:

- `AI_SCIENTIST_MAX_TOKENS`: Override `llm.max_tokens`
- `AI_SCIENTIST_TEMPERATURE`: Override `llm.temperature`
- `AI_SCIENTIST_TIMEOUT`: Override `llm.timeout`
- `AI_SCIENTIST_DEFAULT_MODEL`: Override `llm.default_model`

Example:
```bash
export AI_SCIENTIST_MAX_TOKENS=8192
export AI_SCIENTIST_TEMPERATURE=0.8
```

## Custom Configuration Files

Load custom configuration:

```python
config = ConfigManager()
config.load_config('my_config.yaml')
```

## Configuration Validation

The system validates configuration values:

```python
from ai_scientist.utils.config_manager import ConfigManager

# This will raise ValueError for invalid values
config_data = {"llm": {"max_tokens": -1}}
ConfigManager.validate_config(config_data)  # Raises ValueError
```

## Migration from Hardcoded Values

Previously hardcoded values have been moved to configuration:

- `MAX_NUM_TOKENS` in `vlm.py` and `llm.py` → `llm.max_tokens`
- Model-specific token limits → `models.*.max_tokens`
- Available VLM models → `vlm_models.available`

## Security

- Configuration validation prevents invalid values
- Environment variables are validated before use
- Configuration file parsing includes error handling
- Default values are provided for all settings