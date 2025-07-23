#!/usr/bin/env python3
"""
Test suite to verify torch.compile safety checks are implemented.
"""

import unittest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTorchCompileSafety(unittest.TestCase):
    """Test that torch.compile safety checks are properly implemented."""
    
    def test_torch_compile_safety_in_idea_files(self):
        """Test that torch.compile safety checks are implemented in idea files."""
        idea_files = [
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py"
        ]
        
        for idea_file in idea_files:
            if idea_file.exists():
                with open(idea_file, 'r') as f:
                    content = f.read()
                
                # Check for enhanced safety pattern
                self.assertIn("safe_torch_compile", content, f"Missing safe_torch_compile import in {idea_file.name}")
                self.assertIn("from ai_scientist.utils.torch_compile_safety import safe_torch_compile", content, 
                            f"Missing safety import in {idea_file.name}")
                self.assertIn("model = safe_torch_compile(model)", content, 
                            f"Missing safe compilation call in {idea_file.name}")
    
    def test_cuda_availability_check(self):
        """Test that CUDA availability is checked before torch.compile."""
        idea_files = [
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py"
        ]
        
        for idea_file in idea_files:
            if idea_file.exists():
                with open(idea_file, 'r') as f:
                    content = f.read()
                
                # Check for CUDA availability check
                self.assertIn("torch.cuda.is_available()", content, 
                            f"Missing CUDA availability check in {idea_file.name}")
    
    def test_compilation_safety_pattern(self):
        """Test that the compilation safety pattern is complete."""
        idea_files = [
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py"
        ]
        
        for idea_file in idea_files:
            if idea_file.exists():
                with open(idea_file, 'r') as f:
                    content = f.read()
                
                # Verify the enhanced safety pattern
                self.assertIn("safe_torch_compile", content, f"Missing safe compilation in {idea_file.name}")
                self.assertIn("torch_compile_safety", content, f"Missing safety module import in {idea_file.name}")
                
                # Ensure old unsafe pattern is not used
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "torch.compile" in line and "safe_torch_compile" not in line and "import" not in line:
                        # Check if this is the old unsafe pattern
                        if "model = torch.compile(model)" in line:
                            self.fail(f"Found unsafe torch.compile pattern at line {i+1} in {idea_file.name}")


class TestTorchCompileSafetyImplementation(unittest.TestCase):
    """Test implementation details of torch.compile safety."""
    
    def test_error_handling_granularity(self):
        """Test that error handling is appropriately granular."""
        idea_files = [
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py"
        ]
        
        for idea_file in idea_files:
            if idea_file.exists():
                with open(idea_file, 'r') as f:
                    content = f.read()
                
                # Check that safe_torch_compile is used (which has proper error handling internally)
                self.assertIn("safe_torch_compile", content, 
                            f"Should use safe_torch_compile in {idea_file.name}")
                
                # Ensure we're not using the old pattern with manual error handling
                self.assertNotIn("except Exception as e:", content, 
                            f"Should not have manual exception handling when using safe_torch_compile in {idea_file.name}")
    
    def test_no_unsafe_torch_compile_calls(self):
        """Test that there are no unsafe torch.compile calls without error handling."""
        # Search for any torch.compile calls that might not have safety checks
        idea_files = [
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
            project_root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py"
        ]
        
        for idea_file in idea_files:
            if idea_file.exists():
                with open(idea_file, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if "torch.compile" in line and "model = torch.compile" in line:
                        # This line contains a torch.compile call
                        # Check that it's within a try block
                        
                        # Look backwards for try block
                        found_try = False
                        for j in range(max(0, i-10), i):
                            if "try:" in lines[j]:
                                found_try = True
                                break
                        
                        # Look forwards for except block
                        found_except = False
                        for j in range(i, min(len(lines), i+10)):
                            if "except Exception" in lines[j]:
                                found_except = True
                                break
                        
                        self.assertTrue(found_try, f"torch.compile call at line {i+1} in {idea_file.name} not in try block")
                        self.assertTrue(found_except, f"torch.compile call at line {i+1} in {idea_file.name} missing except block")


if __name__ == '__main__':
    unittest.main()