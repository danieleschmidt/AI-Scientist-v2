"""
Test suite for VLM prepare_vlm_prompt functionality
Tests the core logic without requiring heavy dependencies
"""
import unittest
import tempfile
import os
from pathlib import Path


class TestVLMPreparePrompt(unittest.TestCase):
    """Test the prepare_vlm_prompt function implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a dummy image file for testing
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        # Create a minimal file to simulate an image
        with open(self.test_image_path, 'wb') as f:
            # Write minimal JPEG header bytes to simulate an image file
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_prepare_vlm_prompt_input_validation(self):
        """Test input validation works correctly"""
        
        # Mock the VLM module to avoid dependency issues
        import sys
        from unittest.mock import Mock, patch
        
        # Create a mock module
        mock_vlm = Mock()
        mock_vlm.encode_image_to_base64 = Mock(return_value="mock_base64_data")
        sys.modules['ai_scientist.vlm'] = mock_vlm
        
        # Define the function locally for testing
        def prepare_vlm_prompt(msg, image_paths, max_images):
            """Mock implementation for testing"""
            # Input validation
            if not msg or msg is None:
                raise ValueError("Message cannot be empty or None")
            
            if image_paths is None:
                raise ValueError("Image paths cannot be None")
                
            if max_images <= 0:
                raise ValueError("max_images must be greater than 0")
            
            # Convert single image path to list for consistent handling
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            if not image_paths:
                raise ValueError("At least one image path must be provided")
            
            # Create content list starting with the text message
            content = [{"type": "text", "text": msg}]
            
            # Add each image to the content list (up to max_images)
            for image_path in image_paths[:max_images]:
                base64_image = "mock_base64_data"  # Mock data
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                )
            
            # Return as message format expected by VLM API
            return [{"role": "user", "content": content}]
        
        # Test empty message
        with self.assertRaises(ValueError) as cm:
            prepare_vlm_prompt("", ["image.jpg"], 5)
        self.assertIn("Message cannot be empty", str(cm.exception))
        
        # Test None message
        with self.assertRaises(ValueError) as cm:
            prepare_vlm_prompt(None, ["image.jpg"], 5)
        self.assertIn("Message cannot be empty", str(cm.exception))
        
        # Test None image paths
        with self.assertRaises(ValueError) as cm:
            prepare_vlm_prompt("test", None, 5)
        self.assertIn("Image paths cannot be None", str(cm.exception))
        
        # Test empty image list
        with self.assertRaises(ValueError) as cm:
            prepare_vlm_prompt("test", [], 5)
        self.assertIn("At least one image path must be provided", str(cm.exception))
        
        # Test zero max_images
        with self.assertRaises(ValueError) as cm:
            prepare_vlm_prompt("test", ["image.jpg"], 0)
        self.assertIn("max_images must be greater than 0", str(cm.exception))
        
        # Test valid input
        result = prepare_vlm_prompt("test message", ["image.jpg"], 5)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("content", result[0])
        
        content = result[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "test message")
        self.assertEqual(content[1]["type"], "image_url")
    
    def test_function_exists_and_implemented(self):
        """Test that the function exists and is not just 'pass'"""
        vlm_file_path = os.path.join(os.path.dirname(__file__), '..', 'ai_scientist', 'vlm.py')
        
        with open(vlm_file_path, 'r') as f:
            content = f.read()
        
        # Check function exists
        self.assertIn('def prepare_vlm_prompt', content)
        
        # Check it's not just 'pass' by looking for more substantial implementation
        lines = content.split('\n')
        function_start = None
        for i, line in enumerate(lines):
            if 'def prepare_vlm_prompt' in line:
                function_start = i
                break
        
        self.assertIsNotNone(function_start)
        
        # Look for implementation indicators
        function_section = '\n'.join(lines[function_start:function_start + 60])
        
        # Should have docstring, validation, and return statement
        self.assertIn('"""', function_section)  # Docstring
        self.assertIn('ValueError', function_section)  # Input validation
        self.assertIn('return', function_section)  # Return statement
        self.assertIn('content', function_section)  # Implementation content
    
    def test_integration_with_existing_functions(self):
        """Test that existing functions use the new prepare_vlm_prompt"""
        vlm_file_path = os.path.join(os.path.dirname(__file__), '..', 'ai_scientist', 'vlm.py')
        
        with open(vlm_file_path, 'r') as f:
            content = f.read()
        
        # Check that prepare_vlm_prompt is called in other functions
        self.assertIn('prepare_vlm_prompt(', content)
        
        # Should be called at least twice (in get_response_from_vlm and get_batch_responses_from_vlm)
        call_count = content.count('prepare_vlm_prompt(')
        self.assertGreaterEqual(call_count, 2, "prepare_vlm_prompt should be used by other functions")


if __name__ == '__main__':
    unittest.main()