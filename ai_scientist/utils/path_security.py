"""
Path security utilities for safe file operations and path traversal prevention
"""
import os
import shutil
from pathlib import Path
import zipfile
import logging

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Exception raised for path security violations"""
    pass


def validate_safe_path(path, base_dir, allow_relative=True):
    """
    Validate that a path is safe and within the allowed base directory.
    
    Args:
        path: The path to validate (str or Path)
        base_dir: The base directory that should contain the path (str or Path)
        allow_relative: Whether to allow relative paths (default: True)
    
    Returns:
        bool: True if path is safe, False otherwise
    
    Raises:
        PathSecurityError: If path contains dangerous patterns
    """
    try:
        # Convert to Path objects
        if isinstance(path, str) and not allow_relative and not os.path.isabs(path):
            # For relative paths when not allowed, try to resolve against base_dir
            path_obj = (Path(base_dir) / path).resolve()
        else:
            path_obj = Path(path).resolve()
        
        base_obj = Path(base_dir).resolve()
        
        # Check for obvious path traversal attempts before resolution
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/') or '\\..\\' in path_str:
            logger.warning(f"Detected path traversal attempt: {path}")
            return False
        
        # Check if the resolved path is within the base directory
        try:
            path_obj.relative_to(base_obj)
            return True
        except ValueError:
            # Path is outside the base directory
            logger.warning(f"Path {path} resolves outside base directory {base_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        return False


def secure_file_move(source, target_dir, allowed_base, overwrite=False):
    """
    Securely move a file ensuring target is within allowed directory.
    
    Args:
        source: Source file path (str or Path)
        target_dir: Target directory path (str or Path)
        allowed_base: Base directory that must contain target (str or Path)
        overwrite: Whether to overwrite existing files (default: False)
    
    Returns:
        Path: The final target path
    
    Raises:
        PathSecurityError: If operation would violate security constraints
        FileNotFoundError: If source doesn't exist
        FileExistsError: If target exists and overwrite=False
    """
    try:
        # Validate and resolve paths
        source_path = Path(source).resolve()
        target_dir_path = Path(target_dir).resolve()
        base_path = Path(allowed_base).resolve()
        
        # Ensure target directory is within allowed base
        try:
            target_dir_path.relative_to(base_path)
        except ValueError:
            raise PathSecurityError(
                f"Target directory {target_dir} is outside allowed base {allowed_base}"
            )
        
        # Ensure source exists and is a file
        if not source_path.exists():
            raise FileNotFoundError(f"Source file {source} not found")
        
        if not source_path.is_file():
            raise PathSecurityError(f"Source {source} is not a file")
        
        # Create target path
        target_path = target_dir_path / source_path.name
        
        # Ensure target is also within allowed base (double-check)
        try:
            target_path.resolve().relative_to(base_path)
        except ValueError:
            raise PathSecurityError(
                f"Target path {target_path} would be outside allowed base {allowed_base}"
            )
        
        # Check for overwrite
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Target file {target_path} already exists")
        
        # Create target directory if needed
        target_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Perform the move
        shutil.move(str(source_path), str(target_path))
        
        logger.debug(f"Securely moved {source} to {target_path}")
        return target_path
        
    except (PathSecurityError, FileNotFoundError, FileExistsError):
        raise
    except Exception as e:
        raise PathSecurityError(f"File move operation failed: {e}")


def validate_zip_member_path(member_path, extract_base):
    """
    Validate that a zip member path is safe for extraction.
    
    Args:
        member_path: Path of the member in the zip file
        extract_base: Base directory for extraction
    
    Returns:
        bool: True if safe to extract, False otherwise
    """
    try:
        # Check for obvious dangerous patterns
        if (member_path.startswith('/') or 
            member_path.startswith('\\') or
            '..' in member_path or 
            '\\..\\' in member_path):
            logger.warning(f"Dangerous zip member path detected: {member_path}")
            return False
        
        # Normalize and check the path
        normalized_path = os.path.normpath(member_path)
        if normalized_path.startswith('..') or '/..' in normalized_path:
            logger.warning(f"Normalized path still dangerous: {normalized_path}")
            return False
        
        # Check if it would escape the extraction directory
        full_path = Path(extract_base) / normalized_path
        try:
            full_path.resolve().relative_to(Path(extract_base).resolve())
            return True
        except ValueError:
            logger.warning(f"Zip member {member_path} would escape extraction directory")
            return False
            
    except Exception as e:
        logger.error(f"Error validating zip member path {member_path}: {e}")
        return False


def extract_zip_safely(zip_path, extract_to, max_files=1000, max_size=100*1024*1024):
    """
    Safely extract a zip file with path validation and limits.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        max_files: Maximum number of files to extract (default: 1000)
        max_size: Maximum total size to extract in bytes (default: 100MB)
    
    Returns:
        list: List of extracted file paths
    
    Raises:
        PathSecurityError: If extraction would violate security constraints
        zipfile.BadZipFile: If zip file is corrupted
    """
    extract_to = Path(extract_to).resolve()
    extracted_files = []
    total_size = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            
            # Check file count limit
            if len(members) > max_files:
                raise PathSecurityError(
                    f"Zip contains {len(members)} files, exceeding limit of {max_files}"
                )
            
            for member in members:
                # Validate the member path
                if not validate_zip_member_path(member, extract_to):
                    raise PathSecurityError(f"Dangerous path in zip: {member}")
                
                # Check size limits
                member_info = zf.getinfo(member)
                total_size += member_info.file_size
                if total_size > max_size:
                    raise PathSecurityError(
                        f"Zip extraction would exceed size limit of {max_size} bytes"
                    )
                
                # Skip directories
                if member.endswith('/') or member.endswith('\\'):
                    continue
                
                # Create safe target path
                target_path = extract_to / member
                
                # Double-check that target is within extract directory
                try:
                    target_path.resolve().relative_to(extract_to)
                except ValueError:
                    raise PathSecurityError(f"Path traversal attempt: {member}")
                
                # Extract the file safely
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(target_path, 'wb') as target:
                    # Extract in chunks to avoid memory issues
                    while True:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        target.write(chunk)
                
                extracted_files.append(target_path)
                logger.debug(f"Extracted {member} to {target_path}")
        
        logger.info(f"Safely extracted {len(extracted_files)} files from {zip_path}")
        return extracted_files
        
    except zipfile.BadZipFile:
        raise
    except PathSecurityError:
        raise
    except Exception as e:
        raise PathSecurityError(f"Zip extraction failed: {e}")


def secure_relative_path(path, base):
    """
    Securely calculate relative path ensuring no traversal.
    
    Args:
        path: The path to make relative
        base: The base directory
    
    Returns:
        Path: The relative path
    
    Raises:
        PathSecurityError: If path would escape base directory
    """
    try:
        path_obj = Path(path).resolve()
        base_obj = Path(base).resolve()
        
        # This will raise ValueError if path is outside base
        relative = path_obj.relative_to(base_obj)
        
        # Additional check: ensure no parent references in the relative path
        parts = relative.parts
        if '..' in parts:
            raise PathSecurityError("Relative path contains parent directory references")
        
        return relative
        
    except ValueError as e:
        raise PathSecurityError(f"Path {path} is outside base directory {base}: {e}")
    except Exception as e:
        raise PathSecurityError(f"Error calculating relative path: {e}")


def sanitize_filename(filename, max_length=255):
    """
    Sanitize a filename by removing dangerous characters.
    
    Args:
        filename: The filename to sanitize
        max_length: Maximum allowed length (default: 255)
    
    Returns:
        str: Sanitized filename
    """
    import re
    
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "sanitized_file"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized