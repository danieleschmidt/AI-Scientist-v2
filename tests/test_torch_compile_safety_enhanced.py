#!/usr/bin/env python3
"""
Enhanced test suite for torch.compile safety wrapper.
Tests the enhanced requirements from backlog item: unsafe-compilation (WSJF: 4.0)
"""

import unittest
import logging
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTorchCompileSafetyEnhanced(unittest.TestCase):
    """Test enhanced torch.compile safety functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_torch_compile_wrapper_exists(self):
        """Test that torch compile safety wrapper exists."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile
        self.assertTrue(callable(safe_torch_compile))

    def test_compilation_configuration_disable(self):
        """Test that compilation can be disabled via configuration."""
        from ai_scientist.utils.torch_compile_safety import CompilationConfig, load_compilation_config
        
        # Test default config
        config = CompilationConfig()
        self.assertTrue(config.enabled)
        
        # Test environment variable disable
        os.environ['TORCH_COMPILE_DISABLE'] = 'true'
        config = load_compilation_config()
        self.assertFalse(config.enabled)

    def test_is_compilation_disabled_function(self):
        """Test the global disable check function."""
        from ai_scientist.utils.torch_compile_safety import is_compilation_disabled
        
        # Should be False by default
        self.assertFalse(is_compilation_disabled())
        
        # Should be True when env var is set
        os.environ['TORCH_COMPILE_DISABLE'] = 'true'
        self.assertTrue(is_compilation_disabled())
        
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        self.assertTrue(is_compilation_disabled())

    def test_configuration_options(self):
        """Test various configuration options."""
        from ai_scientist.utils.torch_compile_safety import load_compilation_config
        
        # Test fallback disable
        os.environ['TORCH_COMPILE_NO_FALLBACK'] = 'true'
        config = load_compilation_config()
        self.assertFalse(config.fallback_on_error)
        
        # Test performance logging disable
        os.environ['TORCH_COMPILE_NO_PERF_LOG'] = 'true'
        config = load_compilation_config()
        self.assertFalse(config.log_performance)
        
        # Test max compilation time
        os.environ['TORCH_COMPILE_MAX_TIME'] = '60.0'
        config = load_compilation_config()
        self.assertEqual(config.max_compilation_time, 60.0)

    def test_safe_compile_with_mock_torch(self):
        """Test safe_torch_compile with mocked torch."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        # Mock a simple model
        mock_model = MagicMock()
        mock_model.__str__ = MagicMock(return_value="MockModel")
        
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.compile.return_value = mock_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            config = CompilationConfig(enabled=True)
            result = safe_torch_compile(mock_model, config)
            
            # Should call torch.compile
            mock_torch.compile.assert_called_once()
            self.assertEqual(result, mock_model)

    def test_safe_compile_disabled(self):
        """Test that safe_torch_compile respects disabled configuration."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        mock_model = MagicMock()
        config = CompilationConfig(enabled=False)
        
        # When compilation is disabled, should return original model
        result = safe_torch_compile(mock_model, config)
        self.assertEqual(result, mock_model)

    def test_safe_compile_fallback_on_error(self):
        """Test fallback behavior when compilation fails."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        mock_model = MagicMock()
        config = CompilationConfig(enabled=True, fallback_on_error=True)
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.compile.side_effect = RuntimeError("Compilation failed")
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            result = safe_torch_compile(mock_model, config)
            
            # Should return original model due to fallback
            self.assertEqual(result, mock_model)

    def test_safe_compile_no_fallback_raises(self):
        """Test that compilation errors are raised when fallback is disabled."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        mock_model = MagicMock()
        config = CompilationConfig(enabled=True, fallback_on_error=False)
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.compile.side_effect = RuntimeError("Compilation failed")
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with self.assertRaises(RuntimeError):
                safe_torch_compile(mock_model, config)

    def test_compilation_status_check(self):
        """Test compilation status checking functionality."""
        from ai_scientist.utils.torch_compile_safety import get_compilation_status
        
        mock_model = MagicMock()
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            status = get_compilation_status(mock_model)
            
            self.assertIsInstance(status, dict)
            self.assertIn('is_compiled', status)
            self.assertIn('backend', status)
            self.assertIn('mode', status)

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        from ai_scientist.utils.torch_compile_safety import monitor_compilation_performance
        
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.return_value = MagicMock()  # Mock model output
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            metrics = monitor_compilation_performance(mock_model)
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('compilation_status', metrics)


class TestLoggingEnhancements(unittest.TestCase):
    """Test enhanced logging for torch.compile operations."""
    
    def setUp(self):
        """Set up logging capture."""
        self.log_capture = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_capture.append(record.getMessage())
        
        logger = logging.getLogger('ai_scientist.utils.torch_compile_safety')
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)
        
    def tearDown(self):
        """Clean up logging."""
        logger = logging.getLogger('ai_scientist.utils.torch_compile_safety')
        logger.removeHandler(self.handler)
        
    def test_compilation_success_logging(self):
        """Test that successful compilation is logged."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        mock_model = MagicMock()
        mock_model.__str__ = MagicMock(return_value="MockModel")
        config = CompilationConfig(enabled=True, log_performance=True)
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.compile.return_value = mock_model
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            safe_torch_compile(mock_model, config)
            
            # Check that success was logged
            success_logs = [log for log in self.log_capture if 'completed successfully' in log]
            self.assertTrue(len(success_logs) > 0)
    
    def test_compilation_failure_logging_detail(self):
        """Test that detailed failure information is logged."""
        from ai_scientist.utils.torch_compile_safety import safe_torch_compile, CompilationConfig
        
        mock_model = MagicMock()
        config = CompilationConfig(enabled=True, fallback_on_error=True)
        
        # Mock the torch import inside the function
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.compile.side_effect = RuntimeError("Test compilation error")
            
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            safe_torch_compile(mock_model, config)
            
            # Check that detailed error was logged
            error_logs = [log for log in self.log_capture if 'torch.compile failed' in log]
            self.assertTrue(len(error_logs) > 0)
            
            # Check that fallback was logged
            fallback_logs = [log for log in self.log_capture if 'Falling back to eager mode' in log]
            self.assertTrue(len(fallback_logs) > 0)


if __name__ == '__main__':
    unittest.main()