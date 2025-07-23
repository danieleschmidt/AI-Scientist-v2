"""
Enhanced torch.compile safety wrapper.
Implements the requirements from backlog item: unsafe-compilation (WSJF: 4.0)

Acceptance criteria:
- Add try-catch around torch.compile calls
- Implement fallback to non-compiled version
- Add configuration option to disable compilation
- Log compilation failures appropriately
"""

import os
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompilationConfig:
    """Configuration for torch.compile operations."""
    enabled: bool = True
    fallback_on_error: bool = True
    log_performance: bool = True
    max_compilation_time: float = 30.0  # seconds
    compilation_backend: Optional[str] = None
    compilation_mode: str = "default"  # default, reduce-overhead, max-autotune


def load_compilation_config(config_path: Optional[str] = None) -> CompilationConfig:
    """
    Load compilation configuration from file or environment variables.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        CompilationConfig instance
    """
    config = CompilationConfig()
    
    # Check environment variables first
    if os.getenv('TORCH_COMPILE_DISABLE', '').lower() in ('true', '1', 'yes'):
        config.enabled = False
        logger.info("torch.compile disabled via TORCH_COMPILE_DISABLE environment variable")
    
    if os.getenv('TORCH_COMPILE_NO_FALLBACK', '').lower() in ('true', '1', 'yes'):
        config.fallback_on_error = False
        logger.info("torch.compile fallback disabled via TORCH_COMPILE_NO_FALLBACK")
    
    if os.getenv('TORCH_COMPILE_NO_PERF_LOG', '').lower() in ('true', '1', 'yes'):
        config.log_performance = False
    
    # Check for max compilation time
    max_time = os.getenv('TORCH_COMPILE_MAX_TIME')
    if max_time:
        try:
            config.max_compilation_time = float(max_time)
        except ValueError:
            logger.warning(f"Invalid TORCH_COMPILE_MAX_TIME value: {max_time}")
    
    # Backend and mode configuration
    backend = os.getenv('TORCH_COMPILE_BACKEND')
    if backend:
        config.compilation_backend = backend
        
    mode = os.getenv('TORCH_COMPILE_MODE')
    if mode:
        config.compilation_mode = mode
    
    # Load from config file if provided
    if config_path and Path(config_path).exists():
        try:
            import json
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Loaded torch.compile configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    return config


def is_compilation_disabled() -> bool:
    """Check if torch.compile is globally disabled."""
    return os.getenv('TORCH_COMPILE_DISABLE', '').lower() in ('true', '1', 'yes')


def safe_torch_compile(
    model, 
    config: Optional[CompilationConfig] = None,
    **compile_kwargs
) -> Any:
    """
    Safely compile a PyTorch model with enhanced error handling and configuration.
    
    Args:
        model: PyTorch model to compile
        config: Optional compilation configuration
        **compile_kwargs: Additional arguments to pass to torch.compile
        
    Returns:
        Compiled model or original model if compilation fails/disabled
    """
    if config is None:
        config = load_compilation_config()
    
    # Check if compilation is disabled
    if not config.enabled:
        logger.info("torch.compile is disabled via configuration")
        return model
    
    # Check CUDA availability (compilation usually most beneficial on CUDA)
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA not available, torch.compile may not provide significant benefits")
    except ImportError:
        logger.error("PyTorch not available, cannot compile model")
        return model
    
    # Prepare compilation arguments
    compile_args = {}
    
    if config.compilation_backend:
        compile_args['backend'] = config.compilation_backend
    if config.compilation_mode:
        compile_args['mode'] = config.compilation_mode
    
    # Override with user-provided kwargs
    compile_args.update(compile_kwargs)
    
    # Attempt compilation with timeout and error handling
    start_time = time.time()
    try:
        logger.info(f"Attempting to compile model with args: {compile_args}")
        
        # Compile the model
        compiled_model = torch.compile(model, **compile_args)
        
        compilation_time = time.time() - start_time
        
        if config.log_performance:
            logger.info(f"torch.compile completed successfully in {compilation_time:.2f}s")
            logger.info(f"Model compiled with backend: {compile_args.get('backend', 'default')}, "
                       f"mode: {compile_args.get('mode', 'default')}")
        
        # Verify compilation worked by doing a small test if possible
        try:
            # This is a basic smoke test - just accessing the model should work
            _ = str(compiled_model)
            logger.debug("Compiled model smoke test passed")
        except Exception as e:
            logger.warning(f"Compiled model smoke test failed: {e}")
            if config.fallback_on_error:
                logger.info("Falling back to eager mode due to smoke test failure")
                return model
            raise
        
        return compiled_model
        
    except Exception as e:
        compilation_time = time.time() - start_time
        
        # Enhanced error logging
        error_details = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'compilation_time': f"{compilation_time:.2f}s",
            'compile_args': compile_args,
            'cuda_available': torch.cuda.is_available() if 'torch' in locals() else False
        }
        
        logger.error(f"torch.compile failed: {error_details}")
        
        # Determine if we should fallback or re-raise
        if config.fallback_on_error:
            logger.info("Falling back to eager mode due to compilation failure")
            logger.info(f"Model will run without compilation optimizations")
            return model
        else:
            logger.error("Fallback disabled, re-raising compilation error")
            raise RuntimeError(f"torch.compile failed and fallback is disabled: {e}") from e


def get_compilation_status(model) -> Dict[str, Any]:
    """
    Get information about a model's compilation status.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        Dictionary with compilation status information
    """
    status = {
        'is_compiled': False,
        'backend': None,
        'mode': None,
        'compilation_time': None
    }
    
    try:
        import torch
        
        # Check if model has compilation attributes
        if hasattr(model, '_orig_mod'):
            status['is_compiled'] = True
            # Try to get more details if available
            if hasattr(model, '_compilation_info'):
                info = model._compilation_info
                status.update(info)
        
        # Alternative check for newer PyTorch versions
        elif hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'is_compiling'):
            try:
                # This is a more complex check that would require inspecting the model
                status['is_compiled'] = str(type(model)) != str(type(model).__bases__[0])
            except Exception:
                pass
                
    except Exception as e:
        logger.debug(f"Could not determine compilation status: {e}")
    
    return status


def monitor_compilation_performance(model, test_input=None) -> Dict[str, Any]:
    """
    Monitor and compare performance of compiled vs non-compiled model.
    
    Args:
        model: PyTorch model (compiled or not)
        test_input: Optional test input for performance testing
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        'compilation_status': get_compilation_status(model),
        'performance_test': None
    }
    
    if test_input is not None:
        try:
            import torch
            
            # Simple performance test
            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model(test_input)
                
                # Timing
                start_time = time.time()
                for _ in range(10):
                    _ = model(test_input)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                metrics['performance_test'] = {
                    'average_inference_time': f"{avg_time:.4f}s",
                    'throughput_fps': f"{1/avg_time:.2f}"
                }
                
        except Exception as e:
            logger.debug(f"Performance test failed: {e}")
            metrics['performance_test'] = {'error': str(e)}
    
    return metrics