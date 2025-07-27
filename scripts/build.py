#!/usr/bin/env python3
"""
Build script for AI Scientist v2.
Handles packaging, distribution, and build automation.
"""

import argparse
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import json
import time

# Build configuration
BUILD_CONFIG = {
    "name": "ai-scientist-v2",
    "version": "2.0.0",
    "python_version": "3.11",
    "build_dir": "build",
    "dist_dir": "dist",
    "temp_dir": ".build_temp",
    "docs_dir": "docs",
    "assets_dir": "assets",
}

class BuildError(Exception):
    """Custom exception for build errors."""
    pass

class Builder:
    """Main builder class for AI Scientist v2."""
    
    def __init__(self, config: Dict = None):
        self.config = config or BUILD_CONFIG
        self.root_dir = Path.cwd()
        self.build_dir = self.root_dir / self.config["build_dir"]
        self.dist_dir = self.root_dir / self.config["dist_dir"]
        self.temp_dir = self.root_dir / self.config["temp_dir"]
        
        # Build metadata
        self.build_info = {
            "version": self.config["version"],
            "build_time": time.time(),
            "python_version": sys.version,
            "platform": sys.platform,
            "commit_hash": self._get_git_commit(),
            "build_number": os.environ.get("BUILD_NUMBER", "dev"),
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _run_command(self, command: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """Run a command and handle errors."""
        cwd = cwd or self.root_dir
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Exit code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise BuildError(f"Command failed: {' '.join(command)}")
    
    def clean(self):
        """Clean build artifacts."""
        print("🧹 Cleaning build artifacts...")
        
        # Remove build directories
        for dir_path in [self.build_dir, self.dist_dir, self.temp_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  Removed {dir_path}")
        
        # Remove Python cache files
        for pattern in ["**/__pycache__", "**/*.pyc", "**/*.pyo", "**/*.egg-info"]:
            for path in self.root_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        
        print("✅ Clean completed")
    
    def setup_directories(self):
        """Create necessary build directories."""
        print("📁 Setting up build directories...")
        
        for dir_path in [self.build_dir, self.dist_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
            print(f"  Created {dir_path}")
    
    def check_dependencies(self):
        """Check build dependencies."""
        print("🔍 Checking build dependencies...")
        
        required_tools = ["python", "pip", "git"]
        optional_tools = ["docker", "latex", "pandoc"]
        
        for tool in required_tools:
            try:
                self._run_command([tool, "--version"])
                print(f"  ✅ {tool} available")
            except BuildError:
                raise BuildError(f"Required tool not found: {tool}")
        
        for tool in optional_tools:
            try:
                self._run_command([tool, "--version"])
                print(f"  ✅ {tool} available")
            except BuildError:
                print(f"  ⚠️  {tool} not available (optional)")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("📦 Installing dependencies...")
        
        # Upgrade pip
        self._run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install build dependencies
        build_deps = [
            "build",
            "wheel",
            "setuptools>=65.0",
            "twine",
            "check-manifest",
        ]
        
        self._run_command([sys.executable, "-m", "pip", "install"] + build_deps)
        
        # Install project dependencies
        if (self.root_dir / "requirements.txt").exists():
            self._run_command([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt"
            ])
        
        print("✅ Dependencies installed")
    
    def run_tests(self):
        """Run test suite."""
        print("🧪 Running tests...")
        
        # Run pytest with coverage
        test_command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=ai_scientist",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--junit-xml=test-results.xml",
            "-v"
        ]
        
        try:
            self._run_command(test_command)
            print("✅ All tests passed")
        except BuildError:
            print("❌ Tests failed")
            raise
    
    def run_linting(self):
        """Run code quality checks."""
        print("🔍 Running code quality checks...")
        
        # Run flake8
        try:
            self._run_command([
                sys.executable, "-m", "flake8", 
                "ai_scientist/", "tests/",
                "--max-line-length=88",
                "--extend-ignore=E203,W503"
            ])
            print("  ✅ flake8 passed")
        except BuildError:
            print("  ❌ flake8 failed")
            raise
        
        # Run mypy
        try:
            self._run_command([
                sys.executable, "-m", "mypy",
                "ai_scientist/",
                "--ignore-missing-imports"
            ])
            print("  ✅ mypy passed")
        except BuildError:
            print("  ❌ mypy failed")
            raise
        
        # Run bandit security check
        try:
            self._run_command([
                sys.executable, "-m", "bandit",
                "-r", "ai_scientist/",
                "-f", "json",
                "-o", str(self.build_dir / "bandit-report.json")
            ])
            print("  ✅ bandit passed")
        except BuildError:
            print("  ❌ bandit failed")
            raise
        
        print("✅ Code quality checks passed")
    
    def build_package(self):
        """Build Python package."""
        print("📦 Building Python package...")
        
        # Generate build metadata
        build_info_file = self.root_dir / "ai_scientist" / "_build_info.py"
        with open(build_info_file, 'w') as f:
            f.write(f'"""Build information for AI Scientist v2."""\n\n')
            f.write(f'BUILD_INFO = {json.dumps(self.build_info, indent=2)}\n')
        
        # Build source distribution and wheel
        self._run_command([sys.executable, "-m", "build"])
        
        # Check package
        self._run_command([sys.executable, "-m", "twine", "check", "dist/*"])
        
        print("✅ Package built successfully")
    
    def build_documentation(self):
        """Build documentation."""
        print("📚 Building documentation...")
        
        docs_source = self.root_dir / "docs"
        docs_build = self.build_dir / "docs"
        
        if not docs_source.exists():
            print("  ⚠️  No docs directory found, skipping documentation build")
            return
        
        # Create docs build directory
        docs_build.mkdir(exist_ok=True)
        
        # Build with Sphinx if available
        try:
            self._run_command([
                "sphinx-build",
                "-b", "html",
                str(docs_source),
                str(docs_build)
            ])
            print("  ✅ Sphinx documentation built")
        except BuildError:
            print("  ⚠️  Sphinx not available, copying docs as-is")
            shutil.copytree(docs_source, docs_build, dirs_exist_ok=True)
        
        print("✅ Documentation built")
    
    def build_docker_image(self, tag: str = None):
        """Build Docker image."""
        print("🐳 Building Docker image...")
        
        tag = tag or f"{self.config['name']}:latest"
        
        try:
            # Build production image
            self._run_command([
                "docker", "build",
                "-t", tag,
                "--target", "production",
                "."
            ])
            
            # Build development image
            dev_tag = tag.replace(":latest", ":dev")
            self._run_command([
                "docker", "build", 
                "-t", dev_tag,
                "--target", "development",
                "."
            ])
            
            print(f"✅ Docker images built: {tag}, {dev_tag}")
            
        except BuildError:
            print("❌ Docker build failed")
            raise
    
    def run_security_scans(self):
        """Run security scans."""
        print("🔒 Running security scans...")
        
        # Safety check for dependencies
        try:
            self._run_command([
                sys.executable, "-m", "safety", "check",
                "--json",
                "--output", str(self.build_dir / "safety-report.json")
            ])
            print("  ✅ Safety check passed")
        except BuildError:
            print("  ⚠️  Safety check failed or not available")
        
        # Semgrep static analysis
        try:
            self._run_command([
                "semgrep", "--config=auto",
                "ai_scientist/",
                "--json",
                "--output", str(self.build_dir / "semgrep-report.json")
            ])
            print("  ✅ Semgrep scan completed")
        except BuildError:
            print("  ⚠️  Semgrep not available")
        
        print("✅ Security scans completed")
    
    def create_release_archive(self):
        """Create release archive."""
        print("📦 Creating release archive...")
        
        # Create release directory
        release_dir = self.build_dir / "release"
        release_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "README.md",
            "LICENSE", 
            "CHANGELOG.md",
            "requirements.txt",
            "pyproject.toml"
        ]
        
        for file_name in essential_files:
            src_file = self.root_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, release_dir / file_name)
        
        # Copy source code
        src_dir = release_dir / "ai_scientist"
        shutil.copytree(
            self.root_dir / "ai_scientist",
            src_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        )
        
        # Copy built packages
        if self.dist_dir.exists():
            dist_release_dir = release_dir / "dist"
            shutil.copytree(self.dist_dir, dist_release_dir)
        
        # Create archive
        archive_name = f"{self.config['name']}-{self.config['version']}"
        archive_path = self.dist_dir / f"{archive_name}.tar.gz"
        
        shutil.make_archive(
            str(self.dist_dir / archive_name),
            'gztar',
            str(release_dir)
        )
        
        print(f"✅ Release archive created: {archive_path}")
    
    def generate_build_report(self):
        """Generate build report."""
        print("📊 Generating build report...")
        
        report = {
            "build_info": self.build_info,
            "config": self.config,
            "artifacts": [],
            "test_results": {},
            "security_results": {},
        }
        
        # Collect artifacts
        if self.dist_dir.exists():
            for artifact in self.dist_dir.glob("*"):
                report["artifacts"].append({
                    "name": artifact.name,
                    "size": artifact.stat().st_size,
                    "type": artifact.suffix
                })
        
        # Include test results if available
        test_results_file = self.root_dir / "test-results.xml"
        if test_results_file.exists():
            report["test_results"]["junit_xml"] = str(test_results_file)
        
        # Include security scan results
        for scan_file in ["bandit-report.json", "safety-report.json", "semgrep-report.json"]:
            scan_path = self.build_dir / scan_file
            if scan_path.exists():
                report["security_results"][scan_file] = str(scan_path)
        
        # Save report
        report_file = self.build_dir / "build-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Build report saved: {report_file}")
        return report


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build AI Scientist v2")
    
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--no-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--no-lint", action="store_true", help="Skip linting checks")
    parser.add_argument("--no-docs", action="store_true", help="Skip documentation build")
    parser.add_argument("--docker", action="store_true", help="Build Docker images")
    parser.add_argument("--docker-tag", default="ai-scientist:latest", help="Docker image tag")
    parser.add_argument("--security", action="store_true", help="Run security scans")
    parser.add_argument("--release", action="store_true", help="Create release archive")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    builder = Builder()
    
    try:
        if args.clean:
            builder.clean()
            return
        
        print(f"🚀 Building AI Scientist v2 v{BUILD_CONFIG['version']}")
        
        # Setup
        builder.setup_directories()
        builder.check_dependencies()
        builder.install_dependencies()
        
        # Quality checks
        if not args.no_lint:
            builder.run_linting()
        
        if not args.no_tests:
            builder.run_tests()
        
        # Build
        builder.build_package()
        
        if not args.no_docs:
            builder.build_documentation()
        
        if args.docker:
            builder.build_docker_image(args.docker_tag)
        
        if args.security:
            builder.run_security_scans()
        
        if args.release:
            builder.create_release_archive()
        
        # Generate report
        report = builder.generate_build_report()
        
        print("\n🎉 Build completed successfully!")
        print(f"   Version: {report['build_info']['version']}")
        print(f"   Artifacts: {len(report['artifacts'])}")
        print(f"   Build time: {time.time() - report['build_info']['build_time']:.1f}s")
        
    except BuildError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()