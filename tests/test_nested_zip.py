"""
Test suite for nested zip file handling
"""
import unittest
import tempfile
import zipfile
import os
from pathlib import Path
import sys

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_scientist.treesearch.utils import extract_archives


class TestNestedZipHandling(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_nested_zip_extraction(self):
        """Test extraction of zip files containing other zip files"""
        # Create inner zip file content
        inner_content = "This is content from the inner zip"
        
        # Create inner zip file
        inner_zip_path = self.test_path / "inner.zip"
        with zipfile.ZipFile(inner_zip_path, 'w') as inner_zip:
            inner_zip.writestr("inner_file.txt", inner_content)
        
        # Create outer zip file containing the inner zip
        outer_zip_path = self.test_path / "outer.zip"
        with zipfile.ZipFile(outer_zip_path, 'w') as outer_zip:
            outer_zip.write(inner_zip_path, "inner.zip")
            outer_zip.writestr("outer_file.txt", "This is content from the outer zip")
        
        # Remove the standalone inner zip
        inner_zip_path.unlink()
        
        # Extract with nested zip handling
        extract_archives(self.test_path)
        
        # Check that both levels were extracted
        outer_dir = self.test_path / "outer"
        self.assertTrue(outer_dir.exists())
        
        # Check outer level content
        outer_file = outer_dir / "outer_file.txt"
        self.assertTrue(outer_file.exists())
        self.assertEqual(outer_file.read_text(), "This is content from the outer zip")
        
        # Check inner level content (should be extracted from nested zip)
        inner_dir = outer_dir / "inner"
        inner_file = inner_dir / "inner_file.txt"
        self.assertTrue(inner_file.exists(), "Nested zip should be extracted")
        self.assertEqual(inner_file.read_text(), inner_content)
        
        # Original zips should be removed
        self.assertFalse(outer_zip_path.exists())
    
    def test_max_depth_protection(self):
        """Test that maximum depth prevents infinite recursion"""
        # Create a zip that would create infinite recursion
        zip1_path = self.test_path / "recursive1.zip"
        zip2_path = self.test_path / "recursive2.zip"
        
        # Create initial files
        with zipfile.ZipFile(zip1_path, 'w') as zip1:
            zip1.writestr("file1.txt", "content1")
        
        with zipfile.ZipFile(zip2_path, 'w') as zip2:
            zip2.writestr("file2.txt", "content2")
        
        # Extract with limited depth
        extract_archives(self.test_path, max_depth=1)
        
        # Should extract at least the first level
        dir1 = self.test_path / "recursive1"
        dir2 = self.test_path / "recursive2"
        
        self.assertTrue(dir1.exists())
        self.assertTrue(dir2.exists())
        
        file1 = dir1 / "file1.txt"
        file2 = dir2 / "file2.txt"
        
        self.assertTrue(file1.exists())
        self.assertTrue(file2.exists())
        self.assertEqual(file1.read_text(), "content1")
        self.assertEqual(file2.read_text(), "content2")
    
    def test_zip_validation_with_size_check(self):
        """Test improved zip validation with file size checking"""
        # Create a zip file
        zip_path = self.test_path / "test.zip"
        content = "Test content for validation"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", content)
        
        # Create a file with the same name but different content
        existing_file = self.test_path / "test.txt"
        existing_file.write_text("Different content")
        
        # Rename to collision target
        collision_target = self.test_path / "test"
        existing_file.rename(collision_target)
        
        # Extract - should detect size mismatch and keep zip
        extract_archives(self.test_path)
        
        # Zip should still exist due to size mismatch
        self.assertTrue(zip_path.exists())
        self.assertTrue(collision_target.exists())
    
    def test_corrupted_zip_handling(self):
        """Test handling of corrupted zip files"""
        # Create a file that looks like a zip but isn't
        fake_zip = self.test_path / "corrupted.zip"
        fake_zip.write_text("This is not a zip file")
        
        # Create a real zip for comparison
        real_zip = self.test_path / "real.zip"
        with zipfile.ZipFile(real_zip, 'w') as zf:
            zf.writestr("real.txt", "real content")
        
        # Extract - should handle corrupted zip gracefully
        extract_archives(self.test_path)
        
        # Real zip should be extracted
        real_dir = self.test_path / "real"
        self.assertTrue(real_dir.exists())
        
        real_file = real_dir / "real.txt"
        self.assertTrue(real_file.exists())
        self.assertEqual(real_file.read_text(), "real content")
        
        # Corrupted zip should still exist (not processed)
        self.assertTrue(fake_zip.exists())


if __name__ == '__main__':
    unittest.main()