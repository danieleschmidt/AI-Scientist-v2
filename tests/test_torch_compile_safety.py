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
                
                # Check for safety pattern
                self.assertIn("try:", content, f"Missing try block in {idea_file.name}")
                self.assertIn("torch.compile", content, f"Missing torch.compile in {idea_file.name}")
                self.assertIn("except Exception as e:", content, f"Missing exception handling in {idea_file.name}")
                self.assertIn("torch.compile failed, falling back to eager mode", content, 
                            f"Missing fallback message in {idea_file.name}")
    
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
                
                # Find the torch.compile section
                lines = content.split('\\n')
                compile_section = []
                in_compile_section = False
                
                for line in lines:
                    if "torch.compile" in line or in_compile_section:
                        compile_section.append(line.strip())
                        in_compile_section = True
                        if line.strip().startswith("except") and "Exception" in line:
                            # Add a few more lines to capture the complete pattern
                            continue
                        elif in_compile_section and line.strip() and not line.strip().startswith(("try:", "except", "print", "model =", "#")):
                            break
                
                compile_text = '\\n'.join(compile_section)
                
                # Verify the complete safety pattern
                self.assertIn("try:", compile_text, f"Missing try in torch.compile section of {idea_file.name}")
                self.assertIn("model = torch.compile", compile_text, f"Missing compile call in {idea_file.name}")
                self.assertIn("except Exception", compile_text, f"Missing exception handling in {idea_file.name}")
                self.assertIn("falling back to eager mode", compile_text, f"Missing fallback message in {idea_file.name}")


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
                
                # Check that we're catching general Exception (broad enough to catch compilation issues)
                self.assertIn("except Exception as e:", content, 
                            f"Should catch general Exception in {idea_file.name}")
                
                # Check that error is properly logged
                self.assertIn("Error: {e}", content, 
                            f"Should log the actual error in {idea_file.name}")
    
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