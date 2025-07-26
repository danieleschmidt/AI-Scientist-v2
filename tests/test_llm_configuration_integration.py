#!/usr/bin/env python3
"""
Test suite to verify LLM module integrates correctly with configuration system.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestLLMConfigurationIntegration(unittest.TestCase):
    """Test LLM module configuration integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any cached configuration
        import ai_scientist.utils.config
        ai_scientist.utils.config._config_manager = None
    
    def test_llm_uses_configuration_values(self):
        """Test that LLM module loads values from configuration system."""
        # Import after clearing config cache
        from ai_scientist.llm import MAX_NUM_TOKENS, AVAILABLE_LLMS
        from ai_scientist.utils.config import get_config
        
        config = get_config()
        
        # Test that MAX_NUM_TOKENS comes from config
        expected_tokens = config.get("MAX_LLM_TOKENS")
        self.assertEqual(MAX_NUM_TOKENS, expected_tokens)
        
        # Test that AVAILABLE_LLMS comes from config
        expected_models = config.get("AVAILABLE_LLM_MODELS")
        self.assertEqual(AVAILABLE_LLMS, expected_models)
        
        # Verify these are lists with expected content
        self.assertIsInstance(AVAILABLE_LLMS, list)
        self.assertGreater(len(AVAILABLE_LLMS), 0)
        self.assertIn("claude-3-5-sonnet-20240620", AVAILABLE_LLMS)
    
    def test_configuration_environment_override_affects_llm(self):
        """Test that environment variables affect LLM module values."""
        import os
        
        # Set environment override
        os.environ["AI_SCIENTIST_MAX_LLM_TOKENS"] = "8192"
        os.environ["AI_SCIENTIST_DEFAULT_MODEL_SEED"] = "42"
        
        try:
            # Clear config cache and reimport
            import ai_scientist.utils.config
            ai_scientist.utils.config._config_manager = None
            
            # Import LLM module fresh with new config
            import importlib
            if 'ai_scientist.llm' in sys.modules:
                importlib.reload(sys.modules['ai_scientist.llm'])
            
            from ai_scientist.llm import MAX_NUM_TOKENS
            from ai_scientist.utils.config import get_config
            
            config = get_config()
            
            # Verify environment override took effect
            self.assertEqual(config.get("MAX_LLM_TOKENS"), 8192)
            self.assertEqual(config.get("DEFAULT_MODEL_SEED"), 42)
            self.assertEqual(MAX_NUM_TOKENS, 8192)
            
        finally:
            # Clean up environment
            if "AI_SCIENTIST_MAX_LLM_TOKENS" in os.environ:
                del os.environ["AI_SCIENTIST_MAX_LLM_TOKENS"]
            if "AI_SCIENTIST_DEFAULT_MODEL_SEED" in os.environ:
                del os.environ["AI_SCIENTIST_DEFAULT_MODEL_SEED"]
    
    def test_api_url_configuration_usage(self):
        """Test that API URLs are loaded from configuration."""
        from ai_scientist.utils.config import get_config
        
        config = get_config()
        
        # Test that API URLs are configurable
        deepseek_url = config.get("API_DEEPSEEK_BASE_URL")
        self.assertEqual(deepseek_url, "https://api.deepseek.com")
        
        hf_url = config.get("API_HUGGINGFACE_BASE_URL")
        self.assertIn("huggingface", hf_url.lower())
        
        openrouter_url = config.get("API_OPENROUTER_BASE_URL")
        self.assertIn("openrouter", openrouter_url.lower())


if __name__ == '__main__':
    unittest.main()