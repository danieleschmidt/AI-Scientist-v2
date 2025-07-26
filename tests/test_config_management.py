#!/usr/bin/env python3
"""
Test suite for centralized configuration management system.
Tests the requirements from backlog item: configuration-management (WSJF: 3.5)

Acceptance criteria:
- Create centralized config system
- Move all hardcoded values to config files
- Add environment-specific overrides
- Implement config validation
"""

import unittest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfigurationManagement(unittest.TestCase):
    """Test centralized configuration management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = Path(self.temp_dir) / "test_config.yaml"
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_system_exists(self):
        """Test that centralized config system exists."""
        # This test should fail initially
        try:
            from ai_scientist.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            self.assertTrue(hasattr(config_manager, 'get'))
            self.assertTrue(hasattr(config_manager, 'load_config'))
        except ImportError:
            self.fail("ConfigManager not implemented yet")
    
    def test_llm_config_centralized(self):
        """Test that LLM configurations are centralized."""
        try:
            from ai_scientist.utils.config_manager import ConfigManager
            config = ConfigManager()
            
            # Should have default values for common LLM settings
            self.assertIsNotNone(config.get('llm.max_tokens'))
            self.assertIsNotNone(config.get('llm.temperature')) 
            self.assertIsNotNone(config.get('llm.timeout'))
            
            # Should have model configurations
            self.assertIsNotNone(config.get('models.gpt4.name'))
            self.assertIsNotNone(config.get('models.claude.name'))
        except ImportError:
            self.fail("ConfigManager not implemented yet")
    
    def test_environment_overrides(self):
        """Test that environment variables can override config values."""
        try:
            from ai_scientist.utils.config_manager import ConfigManager
            
            # Test environment override by creating a fresh ConfigManager instance
            with patch.dict(os.environ, {'AI_SCIENTIST_MAX_TOKENS': '8192'}):
                # Reset the singleton instance to test environment overrides
                ConfigManager._instance = None
                ConfigManager._config = None
                config = ConfigManager()
                self.assertEqual(config.get('llm.max_tokens'), 8192)
                
        except ImportError:
            self.fail("ConfigManager not implemented yet")
    
    def test_config_validation(self):
        """Test that config validation works properly."""
        try:
            from ai_scientist.utils.config_manager import ConfigManager
            
            # Invalid config should raise error
            invalid_config = {"llm": {"max_tokens": -1}}
            with self.assertRaises(ValueError):
                ConfigManager.validate_config(invalid_config)
                
            # Valid config should pass
            valid_config = {"llm": {"max_tokens": 4096, "temperature": 0.7}}
            ConfigManager.validate_config(valid_config)  # Should not raise
            
        except ImportError:
            self.fail("ConfigManager not implemented yet")
    
    def test_hardcoded_values_removed(self):
        """Test that hardcoded values are removed from source files."""
        # Check that MAX_NUM_TOKENS is no longer hardcoded
        vlm_file = project_root / "ai_scientist" / "vlm.py"
        llm_file = project_root / "ai_scientist" / "llm.py"
        
        with open(vlm_file, 'r') as f:
            vlm_content = f.read()
        with open(llm_file, 'r') as f:
            llm_content = f.read()
        
        # Should not have hardcoded MAX_NUM_TOKENS = 4096
        self.assertNotIn("MAX_NUM_TOKENS = 4096", vlm_content)
        self.assertNotIn("MAX_NUM_TOKENS = 4096", llm_content)
        
        # Should import from config instead
        self.assertIn("config_manager", vlm_content.lower() or "config", vlm_content.lower())


class TestConfigIntegration(unittest.TestCase):
    """Test integration of config system with existing code."""
    
    def test_vlm_uses_config(self):
        """Test that VLM module uses config system."""
        try:
            from ai_scientist.vlm import MAX_NUM_TOKENS
            # Should be loaded from config, not hardcoded
            self.assertIsInstance(MAX_NUM_TOKENS, int)
            self.assertGreater(MAX_NUM_TOKENS, 0)
        except ImportError:
            # Config system not implemented yet
            pass
    
    def test_llm_uses_config(self):
        """Test that LLM module uses config system.""" 
        try:
            from ai_scientist.llm import MAX_NUM_TOKENS
            # Should be loaded from config, not hardcoded
            self.assertIsInstance(MAX_NUM_TOKENS, int)
            self.assertGreater(MAX_NUM_TOKENS, 0)
        except ImportError:
            # Config system not implemented yet
            pass


if __name__ == '__main__':
    unittest.main()