"""
Test suite for path security and path traversal prevention
"""
import unittest
import tempfile
import zipfile
import os
from pathlib import Path
import sys

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPathSecurity(unittest.TestCase):
    """Test path validation and security measures"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.secure_dir = self.test_path / "secure"
        self.secure_dir.mkdir()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_path_validation_utility(self):
        """Test a utility function for secure path validation"""
        def validate_safe_path(path, base_dir):
            """
            Validate that a path is safe and within the allowed base directory
            """
            try:
                # Convert to Path objects
                path_obj = Path(path).resolve()
                base_obj = Path(base_dir).resolve()
                
                # Check if the resolved path is within the base directory
                try:
                    path_obj.relative_to(base_obj)
                    return True
                except ValueError:
                    # Path is outside the base directory
                    return False
            except Exception:
                # Invalid path
                return False
        
        # Test valid absolute paths
        test_file = self.secure_dir / "test.txt"
        test_file.write_text("test")
        self.assertTrue(validate_safe_path(test_file, self.secure_dir))
        
        # Test valid relative paths  
        subdir = self.secure_dir / "subdir"
        subdir.mkdir()
        subdir_file = subdir / "test.txt"
        subdir_file.write_text("test")
        self.assertTrue(validate_safe_path(subdir_file, self.secure_dir))
        
        # Test path traversal attempts
        self.assertFalse(validate_safe_path("../test.txt", self.secure_dir))
        self.assertFalse(validate_safe_path("../../etc/passwd", self.secure_dir))
        self.assertFalse(validate_safe_path("/etc/passwd", self.secure_dir))
        
        # Test complex traversal attempts
        self.assertFalse(validate_safe_path("safe/../../../etc/passwd", self.secure_dir))
        self.assertFalse(validate_safe_path("./safe/../../etc/passwd", self.secure_dir))
    
    def test_zip_extraction_path_traversal_detection(self):
        """Test detection of path traversal in zip files"""
        # Create a malicious zip file with path traversal
        malicious_zip = self.test_path / "malicious.zip"
        
        with zipfile.ZipFile(malicious_zip, 'w') as zf:
            # Normal file - should be safe
            zf.writestr("safe_file.txt", "safe content")
            
            # Path traversal attempts - should be detected and blocked
            zf.writestr("../escape.txt", "escaped content")
            zf.writestr("../../etc/passwd", "malicious content")
            zf.writestr("subdir/../../escape2.txt", "another escape")
        
        # Test our path validation function
        def is_safe_zip_path(zip_path, extract_dir):
            """Check if a zip entry path is safe for extraction"""
            try:
                # Normalize the path
                normalized_path = os.path.normpath(zip_path)
                
                # Check for path traversal attempts
                if normalized_path.startswith('..') or '/..' in normalized_path or '\\..\\' in normalized_path:
                    return False
                
                # Check if it would escape the extraction directory
                full_path = Path(extract_dir) / normalized_path
                try:
                    full_path.resolve().relative_to(Path(extract_dir).resolve())
                    return True
                except ValueError:
                    return False
            except Exception:
                return False
        
        # Test with the malicious zip
        with zipfile.ZipFile(malicious_zip, 'r') as zf:
            for member in zf.namelist():
                if member == "safe_file.txt":
                    self.assertTrue(is_safe_zip_path(member, self.secure_dir), 
                                  f"Safe file {member} should be allowed")
                else:
                    self.assertFalse(is_safe_zip_path(member, self.secure_dir),
                                   f"Malicious path {member} should be blocked")
    
    def test_file_move_security(self):
        """Test security of file move operations"""
        def secure_file_move(source, target_dir, allowed_base):
            """
            Securely move a file ensuring target is within allowed directory
            """
            import shutil
            
            try:
                # Validate paths
                source_path = Path(source).resolve()
                target_dir_path = Path(target_dir).resolve()
                base_path = Path(allowed_base).resolve()
                
                # Ensure target directory is within allowed base
                target_dir_path.relative_to(base_path)
                
                # Ensure source exists and is a file
                if not source_path.exists() or not source_path.is_file():
                    raise ValueError(f"Source {source} is not a valid file")
                
                # Create target path
                target_path = target_dir_path / source_path.name
                
                # Ensure target is also within allowed base (double-check)
                target_path.relative_to(base_path)
                
                # Perform the move
                shutil.move(str(source_path), str(target_path))
                return target_path
                
            except ValueError as e:
                raise SecurityError(f"Path security violation: {e}")
            except Exception as e:
                raise SecurityError(f"File operation failed: {e}")
        
        # Define a custom exception for testing
        class SecurityError(Exception):
            pass
        
        # Create test file
        test_file = self.test_path / "test.txt"
        test_file.write_text("test content")
        
        # Test valid move
        target_path = secure_file_move(test_file, self.secure_dir, self.test_path)
        self.assertTrue(target_path.exists())
        self.assertEqual(target_path.read_text(), "test content")
        
        # Create another test file for malicious move
        test_file2 = self.test_path / "test2.txt"
        test_file2.write_text("test content 2")
        
        # Test path traversal in target directory - should fail
        with self.assertRaises(SecurityError):
            secure_file_move(test_file2, "../outside", self.test_path)
    
    def test_relative_path_calculation_security(self):
        """Test security of relative path calculations"""
        def secure_relative_path(path, base):
            """
            Securely calculate relative path ensuring no traversal
            """
            try:
                path_obj = Path(path).resolve()
                base_obj = Path(base).resolve()
                
                # This will raise ValueError if path is outside base
                relative = path_obj.relative_to(base_obj)
                
                # Additional check: ensure no parent references in the relative path
                parts = relative.parts
                if '..' in parts or any(part.startswith('.') and len(part) > 1 for part in parts):
                    raise ValueError("Invalid relative path")
                
                return relative
            except ValueError as e:
                raise SecurityError(f"Path traversal detected: {e}")
        
        class SecurityError(Exception):
            pass
        
        # Test valid relative paths
        valid_path = self.secure_dir / "subdir" / "file.txt"
        relative = secure_relative_path(valid_path, self.secure_dir)
        self.assertEqual(str(relative), "subdir/file.txt")
        
        # Test path traversal attempts
        with self.assertRaises(SecurityError):
            secure_relative_path(self.test_path / "outside.txt", self.secure_dir)


class TestZipExtractionSecurity(unittest.TestCase):
    """Test security of zip extraction process"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.extract_dir = self.test_path / "extract"
        self.extract_dir.mkdir()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_malicious_zip_detection(self):
        """Test detection and prevention of malicious zip files"""
        # Create various types of malicious zip files
        malicious_zips = {
            "path_traversal.zip": ["../../../etc/passwd", "normal.txt"],
            "absolute_path.zip": ["/etc/passwd", "normal.txt"],
            "mixed_traversal.zip": ["safe/../../etc/passwd", "normal.txt"],
            "windows_traversal.zip": ["..\\..\\windows\\system32\\config\\sam", "normal.txt"]
        }
        
        for zip_name, file_list in malicious_zips.items():
            zip_path = self.test_path / zip_name
            
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for filename in file_list:
                    zf.writestr(filename, f"content of {filename}")
            
            # Test our security function
            dangerous_files = []
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    # Check for dangerous patterns
                    if (member.startswith('/') or 
                        '..' in member or 
                        member.startswith('\\') or
                        '\\..\\' in member):
                        dangerous_files.append(member)
            
            # Should detect at least one dangerous file in each zip
            self.assertGreater(len(dangerous_files), 0, 
                             f"Should detect malicious files in {zip_name}")
    
    def test_safe_extraction_patterns(self):
        """Test patterns for safe zip extraction"""
        def extract_zip_safely(zip_path, extract_to):
            """
            Safely extract zip file with path validation
            """
            import zipfile
            import os
            
            extract_to = Path(extract_to).resolve()
            extracted_files = []
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    # Validate the member path
                    if (member.startswith('/') or 
                        '..' in member or 
                        '\\..\\' in member or
                        member.startswith('\\')):
                        raise SecurityError(f"Dangerous path in zip: {member}")
                    
                    # Create safe target path
                    target_path = extract_to / member
                    
                    # Double-check that target is within extract directory
                    try:
                        target_path.resolve().relative_to(extract_to)
                    except ValueError:
                        raise SecurityError(f"Path traversal attempt: {member}")
                    
                    # Extract the file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    
                    extracted_files.append(target_path)
            
            return extracted_files
        
        class SecurityError(Exception):
            pass
        
        # Create a safe zip file
        safe_zip = self.test_path / "safe.zip"
        with zipfile.ZipFile(safe_zip, 'w') as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")
            zf.writestr("another/deep/file3.txt", "content3")
        
        # Test safe extraction
        extracted = extract_zip_safely(safe_zip, self.extract_dir)
        self.assertEqual(len(extracted), 3)
        
        # Verify files exist and have correct content
        for file_path in extracted:
            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.is_file())
        
        # Create a malicious zip file
        malicious_zip = self.test_path / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, 'w') as zf:
            zf.writestr("safe.txt", "safe content")
            zf.writestr("../malicious.txt", "malicious content")
        
        # Test malicious extraction - should fail
        with self.assertRaises(SecurityError):
            extract_zip_safely(malicious_zip, self.extract_dir)


if __name__ == '__main__':
    unittest.main()