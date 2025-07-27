#!/usr/bin/env python3
"""
Release management script for AI Scientist v2.
Handles version bumping, changelog generation, and release automation.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import semantic_version
import requests


class ReleaseError(Exception):
    """Custom exception for release errors."""
    pass


class ReleaseManager:
    """Manages releases for AI Scientist v2."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.pyproject_path = self.repo_path / "pyproject.toml"
        self.changelog_path = self.repo_path / "CHANGELOG.md"
        self.version_pattern = re.compile(r'version = "([^"]+)"')
        
    def _run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and handle errors."""
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Exit code: {e.returncode}")
            if capture_output:
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
            raise ReleaseError(f"Command failed: {' '.join(command)}")
    
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        if not self.pyproject_path.exists():
            raise ReleaseError("pyproject.toml not found")
        
        content = self.pyproject_path.read_text()
        match = self.version_pattern.search(content)
        
        if not match:
            raise ReleaseError("Version not found in pyproject.toml")
        
        return match.group(1)
    
    def set_version(self, new_version: str) -> None:
        """Set version in pyproject.toml."""
        if not self.pyproject_path.exists():
            raise ReleaseError("pyproject.toml not found")
        
        content = self.pyproject_path.read_text()
        new_content = self.version_pattern.sub(
            f'version = "{new_version}"',
            content
        )
        
        self.pyproject_path.write_text(new_content)
        print(f"Updated version to {new_version} in pyproject.toml")
    
    def bump_version(self, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        current_version = self.get_current_version()
        sem_version = semantic_version.Version(current_version)
        
        if bump_type == "major":
            new_version = sem_version.next_major()
        elif bump_type == "minor":
            new_version = sem_version.next_minor()
        elif bump_type == "patch":
            new_version = sem_version.next_patch()
        else:
            raise ReleaseError(f"Invalid bump type: {bump_type}")
        
        new_version_str = str(new_version)
        self.set_version(new_version_str)
        
        return new_version_str
    
    def get_git_commits_since_tag(self, tag: str = None) -> List[Dict]:
        """Get git commits since a specific tag or last tag."""
        if tag is None:
            # Get last tag
            try:
                result = self._run_command(["git", "describe", "--tags", "--abbrev=0"])
                tag = result.stdout.strip()
            except ReleaseError:
                # No previous tags, get all commits
                tag = ""
        
        # Get commits since tag
        if tag:
            commit_range = f"{tag}..HEAD"
        else:
            commit_range = "HEAD"
        
        result = self._run_command([
            "git", "log", commit_range,
            "--pretty=format:%H|%s|%an|%ad",
            "--date=short"
        ])
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('|', 3)
                commits.append({
                    "hash": parts[0],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": parts[3]
                })
        
        return commits
    
    def categorize_commits(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize commits by type based on conventional commits."""
        categories = {
            "feat": {"title": "🚀 Features", "commits": []},
            "fix": {"title": "🐛 Bug Fixes", "commits": []},
            "docs": {"title": "📚 Documentation", "commits": []},
            "style": {"title": "💎 Style", "commits": []},
            "refactor": {"title": "♻️ Code Refactoring", "commits": []},
            "perf": {"title": "⚡ Performance Improvements", "commits": []},
            "test": {"title": "🧪 Tests", "commits": []},
            "build": {"title": "🏗️ Build System", "commits": []},
            "ci": {"title": "👷 Continuous Integration", "commits": []},
            "chore": {"title": "🔧 Chores", "commits": []},
            "other": {"title": "📝 Other Changes", "commits": []}
        }
        
        conventional_pattern = re.compile(r'^(\w+)(\(.+\))?: (.+)$')
        
        for commit in commits:
            subject = commit["subject"]
            match = conventional_pattern.match(subject)
            
            if match:
                commit_type = match.group(1).lower()
                scope = match.group(2)
                description = match.group(3)
                
                commit["type"] = commit_type
                commit["scope"] = scope
                commit["description"] = description
                
                if commit_type in categories:
                    categories[commit_type]["commits"].append(commit)
                else:
                    categories["other"]["commits"].append(commit)
            else:
                commit["type"] = "other"
                commit["description"] = subject
                categories["other"]["commits"].append(commit)
        
        return categories
    
    def generate_changelog_entry(self, version: str, commits: List[Dict]) -> str:
        """Generate changelog entry for a version."""
        categories = self.categorize_commits(commits)
        
        changelog_lines = [
            f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ]
        
        # Add breaking changes first if any
        breaking_changes = []
        for commit in commits:
            if "BREAKING CHANGE" in commit["subject"] or commit["subject"].startswith("!"):
                breaking_changes.append(commit)
        
        if breaking_changes:
            changelog_lines.extend([
                "### ⚠️ BREAKING CHANGES",
                ""
            ])
            for commit in breaking_changes:
                changelog_lines.append(f"- {commit['description']} ({commit['hash'][:8]})")
            changelog_lines.append("")
        
        # Add categorized changes
        for category_key, category_info in categories.items():
            if category_info["commits"] and category_key != "other":
                changelog_lines.extend([
                    f"### {category_info['title']}",
                    ""
                ])
                
                for commit in category_info["commits"]:
                    scope_str = f"{commit['scope']}: " if commit.get("scope") else ""
                    changelog_lines.append(
                        f"- {scope_str}{commit['description']} ({commit['hash'][:8]})"
                    )
                
                changelog_lines.append("")
        
        # Add other changes if any
        if categories["other"]["commits"]:
            changelog_lines.extend([
                "### 📝 Other Changes",
                ""
            ])
            for commit in categories["other"]["commits"]:
                changelog_lines.append(f"- {commit['description']} ({commit['hash'][:8]})")
            changelog_lines.append("")
        
        return '\n'.join(changelog_lines)
    
    def update_changelog(self, version: str, commits: List[Dict] = None) -> None:
        """Update CHANGELOG.md with new version."""
        if commits is None:
            commits = self.get_git_commits_since_tag()
        
        new_entry = self.generate_changelog_entry(version, commits)
        
        if self.changelog_path.exists():
            current_content = self.changelog_path.read_text()
            
            # Find insertion point (after header)
            lines = current_content.split('\n')
            header_end = 0
            
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    header_end = i
                    break
            
            # Insert new entry
            new_lines = lines[:header_end] + new_entry.split('\n') + [''] + lines[header_end:]
            new_content = '\n'.join(new_lines)
        else:
            # Create new changelog
            header = f"""# Changelog

All notable changes to AI Scientist v2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
            new_content = header + new_entry
        
        self.changelog_path.write_text(new_content)
        print(f"Updated changelog for version {version}")
    
    def create_git_tag(self, version: str, message: str = None) -> None:
        """Create and push git tag."""
        tag_name = f"v{version}"
        tag_message = message or f"Release version {version}"
        
        # Create annotated tag
        self._run_command([
            "git", "tag", "-a", tag_name, 
            "-m", tag_message
        ])
        
        print(f"Created git tag: {tag_name}")
    
    def push_changes(self, push_tags: bool = True) -> None:
        """Push changes and tags to remote."""
        # Push commits
        self._run_command(["git", "push"])
        
        if push_tags:
            # Push tags
            self._run_command(["git", "push", "--tags"])
        
        print("Pushed changes to remote repository")
    
    def create_github_release(self, version: str, token: str, draft: bool = False) -> Dict:
        """Create GitHub release."""
        tag_name = f"v{version}"
        
        # Get release notes from changelog
        release_notes = self._extract_release_notes(version)
        
        # GitHub API request
        url = f"https://api.github.com/repos/SakanaAI/AI-Scientist-v2/releases"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "tag_name": tag_name,
            "target_commitish": "main",
            "name": f"AI Scientist v2 {version}",
            "body": release_notes,
            "draft": draft,
            "prerelease": False
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 201:
            raise ReleaseError(f"Failed to create GitHub release: {response.text}")
        
        release_info = response.json()
        print(f"Created GitHub release: {release_info['html_url']}")
        
        return release_info
    
    def _extract_release_notes(self, version: str) -> str:
        """Extract release notes for a specific version from changelog."""
        if not self.changelog_path.exists():
            return f"Release version {version}"
        
        content = self.changelog_path.read_text()
        lines = content.split('\n')
        
        # Find version section
        version_pattern = re.compile(f'## \\[{re.escape(version)}\\]')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if version_pattern.match(line):
                start_idx = i + 1
            elif start_idx is not None and line.startswith('## '):
                end_idx = i
                break
        
        if start_idx is None:
            return f"Release version {version}"
        
        if end_idx is None:
            end_idx = len(lines)
        
        release_notes = '\n'.join(lines[start_idx:end_idx]).strip()
        return release_notes or f"Release version {version}"
    
    def build_and_upload_package(self, test_pypi: bool = False) -> None:
        """Build and upload package to PyPI."""
        # Build package
        self._run_command([sys.executable, "-m", "build"])
        
        # Upload to PyPI
        repository = "testpypi" if test_pypi else "pypi"
        
        self._run_command([
            sys.executable, "-m", "twine", "upload",
            "--repository", repository,
            "dist/*"
        ])
        
        pypi_name = "Test PyPI" if test_pypi else "PyPI"
        print(f"Package uploaded to {pypi_name}")
    
    def run_pre_release_checks(self) -> bool:
        """Run pre-release checks."""
        print("🔍 Running pre-release checks...")
        
        checks_passed = True
        
        # Check git status
        try:
            result = self._run_command(["git", "status", "--porcelain"])
            if result.stdout.strip():
                print("❌ Working directory is not clean")
                checks_passed = False
            else:
                print("✅ Working directory is clean")
        except ReleaseError:
            print("❌ Git status check failed")
            checks_passed = False
        
        # Check if on main branch
        try:
            result = self._run_command(["git", "branch", "--show-current"])
            current_branch = result.stdout.strip()
            if current_branch != "main":
                print(f"⚠️  Currently on branch '{current_branch}', not 'main'")
            else:
                print("✅ On main branch")
        except ReleaseError:
            print("❌ Branch check failed")
        
        # Run tests
        try:
            self._run_command([sys.executable, "-m", "pytest", "tests/", "-x"])
            print("✅ Tests passed")
        except ReleaseError:
            print("❌ Tests failed")
            checks_passed = False
        
        # Check package can be built
        try:
            self._run_command([sys.executable, "-m", "build", "--sdist", "--wheel"])
            print("✅ Package builds successfully")
        except ReleaseError:
            print("❌ Package build failed")
            checks_passed = False
        
        return checks_passed


def main():
    """Main release script entry point."""
    parser = argparse.ArgumentParser(description="Release AI Scientist v2")
    
    parser.add_argument("bump_type", choices=["major", "minor", "patch"], 
                       help="Version bump type")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--no-checks", action="store_true",
                       help="Skip pre-release checks")
    parser.add_argument("--no-git", action="store_true",
                       help="Skip git operations")
    parser.add_argument("--no-github", action="store_true",
                       help="Skip GitHub release creation")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Upload to Test PyPI instead of PyPI")
    parser.add_argument("--draft", action="store_true",
                       help="Create GitHub release as draft")
    
    args = parser.parse_args()
    
    release_manager = ReleaseManager()
    
    try:
        current_version = release_manager.get_current_version()
        print(f"Current version: {current_version}")
        
        # Calculate new version
        if args.dry_run:
            sem_version = semantic_version.Version(current_version)
            if args.bump_type == "major":
                new_version = str(sem_version.next_major())
            elif args.bump_type == "minor":
                new_version = str(sem_version.next_minor())
            else:
                new_version = str(sem_version.next_patch())
            
            print(f"Would bump version to: {new_version}")
            
            # Get commits for changelog preview
            commits = release_manager.get_git_commits_since_tag()
            print(f"Found {len(commits)} commits since last release")
            
            if commits:
                changelog_entry = release_manager.generate_changelog_entry(new_version, commits)
                print("\nChangelog entry would be:")
                print(changelog_entry)
            
            return
        
        # Run pre-release checks
        if not args.no_checks:
            if not release_manager.run_pre_release_checks():
                print("❌ Pre-release checks failed")
                sys.exit(1)
        
        # Bump version
        new_version = release_manager.bump_version(args.bump_type)
        print(f"Bumped version to: {new_version}")
        
        # Update changelog
        commits = release_manager.get_git_commits_since_tag()
        release_manager.update_changelog(new_version, commits)
        
        # Git operations
        if not args.no_git:
            # Stage changes
            release_manager._run_command(["git", "add", "pyproject.toml", "CHANGELOG.md"])
            
            # Commit changes
            release_manager._run_command([
                "git", "commit", "-m", f"chore(release): bump version to {new_version}"
            ])
            
            # Create tag
            release_manager.create_git_tag(new_version)
            
            # Push changes
            release_manager.push_changes()
        
        # Build and upload package
        release_manager.build_and_upload_package(test_pypi=args.test_pypi)
        
        # Create GitHub release
        if not args.no_github and args.github_token:
            release_manager.create_github_release(
                new_version, 
                args.github_token,
                draft=args.draft
            )
        
        print(f"\n🎉 Successfully released version {new_version}!")
        
    except ReleaseError as e:
        print(f"\n❌ Release failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Release interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()