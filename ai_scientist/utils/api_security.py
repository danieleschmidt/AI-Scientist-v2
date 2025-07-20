"""
API Security utilities for safe handling of API keys and credentials
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_api_key_secure(key_name, required=True):
    """
    Securely retrieve an API key from environment variables with validation.
    
    Args:
        key_name (str): Name of the environment variable
        required (bool): Whether the key is required. If True, raises ValueError if missing.
    
    Returns:
        str or None: The API key value, or None if not required and not found
    
    Raises:
        ValueError: If required=True and key is missing, empty, or invalid
    """
    value = os.environ.get(key_name)
    
    if required and not value:
        raise ValueError(
            f"Required environment variable {key_name} is not set. "
            f"Please set this environment variable with your API key."
        )
    
    if value and not value.strip():
        raise ValueError(
            f"Environment variable {key_name} is empty or contains only whitespace. "
            f"Please set a valid API key value."
        )
    
    return value.strip() if value else None


def validate_api_key_format(key_value, key_name, min_length=8):
    """
    Validate API key format without exposing the actual key value.
    
    Args:
        key_value (str): The API key to validate
        key_name (str): Name of the key for error messages
        min_length (int): Minimum required length
    
    Returns:
        bool: True if valid
    
    Raises:
        ValueError: If the key format is invalid
    """
    if not key_value:
        raise ValueError(f"{key_name} is required")
    
    if len(key_value) < min_length:
        raise ValueError(
            f"{key_name} appears to be too short (minimum {min_length} characters)"
        )
    
    # Allow alphanumeric characters, hyphens, and underscores (common in API keys)
    if not key_value.replace('-', '').replace('_', '').isalnum():
        raise ValueError(
            f"{key_name} contains invalid characters. "
            f"API keys should only contain letters, numbers, hyphens, and underscores."
        )
    
    return True


def mask_api_key_for_logging(key_value, visible_chars=4):
    """
    Safely mask an API key for logging purposes.
    
    Args:
        key_value (str): The API key to mask
        visible_chars (int): Number of characters to show at the beginning
    
    Returns:
        str: Masked key suitable for logging
    """
    if not key_value:
        return "***"
    
    if len(key_value) <= visible_chars:
        return "***"
    
    return f"{key_value[:visible_chars]}..."


def log_api_key_usage(key_name, key_value=None):
    """
    Safely log API key usage without exposing the actual key.
    
    Args:
        key_name (str): Name of the API key
        key_value (str, optional): The key value (will be masked)
    """
    if key_value:
        masked_key = mask_api_key_for_logging(key_value)
        logger.info(f"Using {key_name}: {masked_key}")
    else:
        logger.warning(f"{key_name} not configured")


def get_required_api_keys(key_mapping):
    """
    Retrieve multiple required API keys with validation.
    
    Args:
        key_mapping (dict): Mapping of key_name -> environment_variable_name
    
    Returns:
        dict: Mapping of key_name -> api_key_value
    
    Raises:
        ValueError: If any required key is missing or invalid
    """
    result = {}
    missing_keys = []
    
    for key_name, env_var in key_mapping.items():
        try:
            result[key_name] = get_api_key_secure(env_var, required=True)
        except ValueError as e:
            missing_keys.append(env_var)
    
    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            f"Please set these environment variables before running the application."
        )
    
    return result


def create_secure_headers(api_key, key_name="API Key"):
    """
    Create secure headers for API requests with validation.
    
    Args:
        api_key (str): The API key to use
        key_name (str): Name for error messages
    
    Returns:
        dict: Headers with Authorization bearer token
    
    Raises:
        ValueError: If API key is invalid
    """
    validate_api_key_format(api_key, key_name)
    
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }