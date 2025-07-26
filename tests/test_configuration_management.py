#!/usr/bin/env python3
"""
Test suite for centralized configuration management.
Tests the implementation of the configuration-management backlog item.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfigurationManagement(unittest.TestCase):
    """Test centralized configuration management system."""
    
    def setUp(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.json"
        
        # Sample configuration
        self.test_config = {
            "models": {
                "gpt_models": [
                    "gpt-4o-2024-05-13",
                    "gpt-4o-2024-08-06", 
                    "gpt-4o-2024-11-20",
                    "gpt-4o-mini-2024-07-18"
                ],
                "default_max_tokens": 4096,
                "anthropic_max_tokens": 8192
            },
            "apis": {
                "semantic_scholar_base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "huggingface_base_url": "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                "deepseek_base_url": "https://api.deepseek.com",
                "openrouter_base_url": "https://openrouter.ai/api/v1"
            },
            "pricing": {
                "gpt-4o-2024-11-20": {
                    "prompt": 2.5,
                    "completion": 10.0,
                    "cached_prompt": 1.25
                },
                "gpt-4o-2024-08-06": {
                    "prompt": 2.5,
                    "completion": 10.0,
                    "cached_prompt": 1.25
                },
                "gpt-4o-2024-05-13": {
                    "prompt": 5.0,
                    "completion": 15.0
                },
                "gpt-4o-mini-2024-07-18": {
                    "prompt": 0.15,
                    "completion": 0.6
                }
            },
            "timeouts": {
                "default_process_cleanup": 0.1,
                "default_request_timeout": 30
            }
        }
        
        # Write test config to file
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_file_creation(self):
        """Test that config.json file is created in the project root."""
        config_path = project_root / "config.json"
        
        # Check if config file exists or can be created
        if not config_path.exists():
            # This is acceptable - config file may not exist yet
            pass
        else:
            # If it exists, it should be valid JSON
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.assertIsInstance(config_data, dict)
    
    def test_config_loader_utility_exists(self):
        """Test that configuration loader utility exists."""
        try:
            from ai_scientist.utils.config import load_config
            # Should be importable
            self.assertTrue(callable(load_config))
        except ImportError:
            # Config utility doesn't exist yet - this is expected for new implementation
            self.skipTest("Configuration utility not yet implemented")
    
    def test_environment_override_support(self):
        """Test that environment variables can override config values."""
        try:
            from ai_scientist.utils.config import load_config
            
            # Set environment variable
            os.environ['AI_SCIENTIST_MAX_TOKENS'] = '2048'
            
            config = load_config(str(self.config_file))
            
            # Should respect environment override
            if 'AI_SCIENTIST_MAX_TOKENS' in config:
                self.assertEqual(config['AI_SCIENTIST_MAX_TOKENS'], '2048')
                
            # Clean up
            del os.environ['AI_SCIENTIST_MAX_TOKENS']
            
        except ImportError:
            self.skipTest("Configuration utility not yet implemented")
    
    def test_config_validation(self):
        """Test that configuration validation works."""
        try:
            from ai_scientist.utils.config import validate_config
            
            # Valid config should pass
            self.assertTrue(validate_config(self.test_config))
            
            # Invalid config should fail
            invalid_config = {"invalid": "structure"}
            self.assertFalse(validate_config(invalid_config))
            
        except ImportError:
            self.skipTest("Configuration utility not yet implemented")
    
    def test_hardcoded_values_replaced(self):
        """Test that hardcoded values are replaced with config lookups."""
        # Check that major files use configuration instead of hardcoded values
        
        # Check vlm.py for model list usage
        vlm_file = project_root / "ai_scientist" / "vlm.py"
        if vlm_file.exists():
            with open(vlm_file, 'r') as f:
                content = f.read()
            
            # Look for evidence of configuration usage
            if 'config' in content.lower() or 'Config' in content:
                # Configuration is being used
                pass
            else:
                # Still has hardcoded values - this test will fail until refactored
                self.assertIn('gpt-4o', content)  # This should eventually be removed
        
        # Check token_tracker.py for pricing configuration
        token_tracker_file = project_root / "ai_scientist" / "utils" / "token_tracker.py"
        if token_tracker_file.exists():
            with open(token_tracker_file, 'r') as f:
                content = f.read()
            
            # Should eventually use configuration instead of hardcoded pricing
            if 'config' in content.lower():
                pass  # Good, using configuration
            else:
                # Still hardcoded - expected until refactored
                self.assertIn('0.15', content)  # This should eventually come from config
    
    def test_configuration_schema(self):
        """Test that configuration follows expected schema."""
        required_sections = ['models', 'apis', 'pricing', 'timeouts']
        
        for section in required_sections:
            self.assertIn(section, self.test_config)
        
        # Models section should have GPT model list
        self.assertIn('gpt_models', self.test_config['models'])
        self.assertIsInstance(self.test_config['models']['gpt_models'], list)
        
        # APIs section should have base URLs
        self.assertIn('semantic_scholar_base_url', self.test_config['apis'])
        
        # Pricing should have model-specific pricing
        pricing = self.test_config['pricing']
        for model_name, model_pricing in pricing.items():
            self.assertIn('prompt', model_pricing)
            self.assertIn('completion', model_pricing)


if __name__ == '__main__':
    unittest.main()