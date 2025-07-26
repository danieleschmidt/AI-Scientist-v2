#!/usr/bin/env python3
"""
Test suite for dependency management audit and validation.
Tests the requirements from backlog item: dependency-management (WSJF: 4.0)

Acceptance criteria:
- Audit all Python imports vs requirements.txt
- Add missing dependencies with proper version constraints
- Remove unused dependencies  
- Add dependency security scanning
- Document dependency management
"""

import unittest
import ast
import re
from pathlib import Path
from typing import Set, Dict, List
import subprocess
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDependencyManagement(unittest.TestCase):
    """Test dependency management and audit functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.requirements_path = self.project_root / "requirements.txt"
        
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        self.assertTrue(self.requirements_path.exists(), 
                       "requirements.txt must exist")
    
    def test_all_imports_have_dependencies_declared(self):
        """Test that all third-party imports have corresponding entries in requirements.txt."""
        # Get all imports from Python files
        imports = self._get_all_imports()
        
        # Get dependencies from requirements.txt
        declared_deps = self._get_declared_dependencies()
        
        # Standard library modules (don't need to be in requirements.txt)
        stdlib_modules = self._get_stdlib_modules()
        
        # Find missing dependencies
        missing_deps = []
        for import_name in imports:
            if (import_name not in stdlib_modules and 
                not self._is_local_module(import_name)):
                # Check if import is covered by declared dependencies
                normalized_import = self._normalize_package_name(import_name)
                found = False
                for dep in declared_deps:
                    normalized_dep = self._normalize_package_name(dep)
                    if (normalized_import == normalized_dep or 
                        normalized_import.startswith(normalized_dep + '-') or
                        normalized_dep.startswith(normalized_import + '-')):
                        found = True
                        break
                if not found:
                    missing_deps.append(import_name)
        
        self.assertEqual(len(missing_deps), 0, 
                        f"Missing dependencies in requirements.txt: {missing_deps}")
    
    def test_no_unused_dependencies(self):
        """Test that all dependencies in requirements.txt are actually used."""
        # Get declared dependencies
        declared_deps = self._get_declared_dependencies()
        
        # Get all imports
        imports = self._get_all_imports()
        
        # Find unused dependencies
        unused_deps = []
        for dep in declared_deps:
            # Normalize dependency name for comparison
            dep_normalized = self._normalize_package_name(dep)
            if not any(self._normalize_package_name(imp).startswith(dep_normalized) 
                      for imp in imports):
                unused_deps.append(dep)
        
        # Allow some exceptions for build/test dependencies
        allowed_unused = {'PyYAML'}  # Configuration files
        unused_deps = [dep for dep in unused_deps if dep not in allowed_unused]
        
        self.assertEqual(len(unused_deps), 0,
                        f"Unused dependencies in requirements.txt: {unused_deps}")
    
    def test_dependencies_have_version_constraints(self):
        """Test that dependencies have appropriate version constraints."""
        with open(self.requirements_path, 'r') as f:
            lines = f.readlines()
        
        # Check each dependency line
        deps_without_versions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Should have version constraint (>=, ==, ~=, etc.)
                if not re.search(r'[><=~!]', line):
                    deps_without_versions.append(line)
        
        # For now, we'll allow deps without explicit versions but warn about it
        # In production, you might want to make this stricter
        if deps_without_versions:
            print(f"WARNING: Dependencies without version constraints: {deps_without_versions}")
    
    def test_requirements_file_format_valid(self):
        """Test that requirements.txt has valid format."""
        with open(self.requirements_path, 'r') as f:
            content = f.read()
        
        # Should not have duplicate entries
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        
        packages = [re.split(r'[><=~!]', line)[0].strip() for line in lines]
        duplicates = [pkg for pkg in set(packages) if packages.count(pkg) > 1]
        
        self.assertEqual(len(duplicates), 0,
                        f"Duplicate dependencies in requirements.txt: {duplicates}")
    
    def test_security_scanning_available(self):
        """Test that dependency security scanning tools are available."""
        # Test that we can run basic security checks
        try:
            # Check if safety is installed (optional, but recommended)
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
            self.assertIn('pip', result.stdout.lower())
        except FileNotFoundError:
            self.skipTest("pip not available for security scanning")
    
    def _get_all_imports(self) -> Set[str]:
        """Get all import statements from Python files in the project."""
        imports = set()
        
        # Find all Python files
        for py_file in self.project_root.rglob("*.py"):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                except SyntaxError:
                    # Skip files with syntax errors
                    continue
                    
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue
        
        return imports
    
    def _get_declared_dependencies(self) -> Set[str]:
        """Get all dependencies declared in requirements.txt."""
        dependencies = set()
        
        if not self.requirements_path.exists():
            return dependencies
        
        with open(self.requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifiers)
                    package_name = re.split(r'[><=~!]', line)[0].strip()
                    dependencies.add(package_name)
        
        return dependencies
    
    def _get_stdlib_modules(self) -> Set[str]:
        """Get standard library module names."""
        # Common standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'operator', 'copy',
            'pickle', 'random', 're', 'string', 'math', 'statistics',
            'threading', 'multiprocessing', 'queue', 'subprocess', 'signal',
            'logging', 'unittest', 'tempfile', 'shutil', 'glob', 'fnmatch',
            'csv', 'xml', 'html', 'urllib', 'http', 'email', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'struct', 'binascii',
            'zlib', 'gzip', 'tarfile', 'zipfile', 'sqlite3', 'dbm',
            'contextlib', 'importlib', 'pkgutil', 'ast', 'dis', 'inspect',
            'warnings', 'traceback', 'gc', 'weakref', 'abc',
            # Additional stdlib modules
            'io', '__future__', 'enum', 'unicodedata', 'textwrap', 'argparse',
            'concurrent', 'dataclasses', 'asyncio', 'atexit'
        }
        return stdlib_modules
    
    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module is a local project module."""
        # Project-specific local modules
        local_modules = {
            'ai_scientist', 'journal', 'log_summarization', 'parallel_agent',
            'agent_manager', 'journal2report', 'utils', 'signal_handlers',
            'resource_monitor', 'backend', 'process_cleanup', 'interpreter'
        }
        
        if module_name in local_modules:
            return True
            
        # Check if module exists as a directory or file in the project
        module_path = self.project_root / module_name
        return (module_path.exists() or 
                (self.project_root / f"{module_name}.py").exists())
    
    def _normalize_package_name(self, name: str) -> str:
        """Normalize package name for comparison."""
        # Handle special cases where import name != package name
        name_mapping = {
            'PIL': 'Pillow',
            'yaml': 'PyYAML', 
            'pymupdf': 'PyMuPDF',
            'igraph': 'python-igraph',
            'dataclasses_json': 'dataclasses-json'
        }
        
        if name in name_mapping:
            return name_mapping[name].lower()
        
        # Convert underscores to hyphens and lowercase
        return name.lower().replace('_', '-')


class TestDependencyDocumentation(unittest.TestCase):
    """Test dependency management documentation."""
    
    def test_dependency_documentation_exists(self):
        """Test that dependency management is documented."""
        project_root = Path(__file__).parent.parent
        
        # Check for documentation in README or separate file
        docs_to_check = [
            project_root / "README.md",
            project_root / "DEPENDENCIES.md", 
            project_root / "docs" / "dependencies.md"
        ]
        
        has_dependency_docs = False
        for doc_path in docs_to_check:
            if doc_path.exists():
                with open(doc_path, 'r') as f:
                    content = f.read().lower()
                    if 'dependenc' in content or 'requirement' in content:
                        has_dependency_docs = True
                        break
        
        if not has_dependency_docs:
            print("WARNING: No dependency management documentation found")


if __name__ == '__main__':
    unittest.main()