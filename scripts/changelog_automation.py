#!/usr/bin/env python3
"""
Advanced CHANGELOG automation for AI Scientist v2

Generates semantic changelogs from conventional commits with AI-enhanced
categorization and impact analysis.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class CommitInfo:
    """Structured commit information."""
    hash: str
    type: str
    scope: Optional[str]
    description: str
    body: str
    breaking: bool
    author: str
    date: str
    pr_number: Optional[str] = None


class ChangelogGenerator:
    """Advanced changelog generator with semantic analysis."""
    
    # Conventional commit types with emojis and descriptions
    COMMIT_TYPES = {
        'feat': {'emoji': '✨', 'title': 'Features', 'description': 'New features and capabilities'},
        'fix': {'emoji': '🐛', 'title': 'Bug Fixes', 'description': 'Bug fixes and corrections'},
        'perf': {'emoji': '⚡', 'title': 'Performance', 'description': 'Performance improvements'},
        'refactor': {'emoji': '♻️', 'title': 'Refactoring', 'description': 'Code refactoring without functional changes'},
        'docs': {'emoji': '📝', 'title': 'Documentation', 'description': 'Documentation updates'},
        'test': {'emoji': '✅', 'title': 'Testing', 'description': 'Test additions and improvements'},
        'ci': {'emoji': '👷', 'title': 'CI/CD', 'description': 'CI/CD pipeline changes'},
        'build': {'emoji': '📦', 'title': 'Build', 'description': 'Build system and dependencies'},
        'chore': {'emoji': '🔧', 'title': 'Maintenance', 'description': 'Maintenance and housekeeping'},
        'security': {'emoji': '🔒', 'title': 'Security', 'description': 'Security improvements and fixes'},
        'style': {'emoji': '💄', 'title': 'Style', 'description': 'Code style and formatting'},
        'revert': {'emoji': '⏪', 'title': 'Reverts', 'description': 'Reverted changes'},
    }
    
    # High-impact scopes for AI Scientist
    HIGH_IMPACT_SCOPES = {
        'ai_scientist', 'treesearch', 'llm', 'experiments', 'security', 'api'
    }
    
    def __init__(self, repo_path: Path = Path.cwd()):
        self.repo_path = repo_path
        self.changelog_path = repo_path / 'CHANGELOG.md'
        
    def parse_commit(self, commit_line: str) -> Optional[CommitInfo]:
        """Parse a commit line into structured information."""
        # Extract hash, author, date, and message
        parts = commit_line.split('|', 3)
        if len(parts) < 4:
            return None
            
        hash_val, author, date, message = parts
        
        # Parse conventional commit format
        conventional_pattern = r'^(?P<type>\\w+)(?:\\((?P<scope>[^)]+)\\))?(?P<breaking>!)?:\\s*(?P<description>.+?)(?:\\n\\n(?P<body>.*))?$'
        match = re.match(conventional_pattern, message.strip(), re.DOTALL)
        
        if not match:
            # Try to infer type from message
            type_val = self._infer_commit_type(message)
            return CommitInfo(
                hash=hash_val.strip(),
                type=type_val,
                scope=None,
                description=message.strip(),
                body='',
                breaking=False,
                author=author.strip(),
                date=date.strip()
            )
        
        # Extract PR number from description or body
        pr_pattern = r'#(\\d+)'
        pr_match = re.search(pr_pattern, message)
        pr_number = pr_match.group(1) if pr_match else None
        
        return CommitInfo(
            hash=match.group('hash') or hash_val.strip(),
            type=match.group('type'),
            scope=match.group('scope'),
            description=match.group('description'),
            body=match.group('body') or '',
            breaking=bool(match.group('breaking')),
            author=author.strip(),
            date=date.strip(),
            pr_number=pr_number
        )
    
    def _infer_commit_type(self, message: str) -> str:
        """Infer commit type from message content."""
        message_lower = message.lower()
        
        # Define inference patterns
        patterns = {
            'fix': ['fix', 'bug', 'error', 'issue', 'patch'],
            'feat': ['add', 'new', 'feature', 'implement', 'create'],
            'docs': ['doc', 'readme', 'comment', 'documentation'],
            'test': ['test', 'spec', 'coverage'],
            'refactor': ['refactor', 'restructure', 'reorganize'],
            'perf': ['perf', 'optimize', 'speed', 'performance'],
            'security': ['security', 'vulnerability', 'secure', 'auth'],
            'ci': ['ci', 'workflow', 'action', 'pipeline'],
            'build': ['build', 'deps', 'dependency', 'requirements'],
        }
        
        for commit_type, keywords in patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return commit_type
                
        return 'chore'
    
    def get_commits_since_tag(self, since_tag: Optional[str] = None) -> List[str]:
        """Get commits since the last tag or all commits."""
        try:
            if since_tag:
                cmd = ['git', 'log', f'{since_tag}..HEAD', '--pretty=format:%H|%an|%ad|%s%n%b', '--date=short']
            else:
                # Get commits since last tag
                try:
                    last_tag = subprocess.check_output(
                        ['git', 'describe', '--tags', '--abbrev=0'], 
                        cwd=self.repo_path, 
                        text=True
                    ).strip()
                    cmd = ['git', 'log', f'{last_tag}..HEAD', '--pretty=format:%H|%an|%ad|%s%n%b', '--date=short']
                except subprocess.CalledProcessError:
                    # No tags found, get all commits
                    cmd = ['git', 'log', '--pretty=format:%H|%an|%ad|%s%n%b', '--date=short']
            
            result = subprocess.check_output(cmd, cwd=self.repo_path, text=True)
            return [line for line in result.split('\\n') if line.strip()]
        except subprocess.CalledProcessError as e:
            print(f"Error getting commits: {e}")
            return []
    
    def categorize_commits(self, commits: List[CommitInfo]) -> Dict[str, List[CommitInfo]]:
        """Categorize commits by type with impact analysis."""
        categories = {}
        
        for commit in commits:
            commit_type = commit.type
            if commit_type not in categories:
                categories[commit_type] = []
            categories[commit_type].append(commit)
        
        # Sort categories by importance
        sorted_categories = {}
        type_order = ['security', 'feat', 'fix', 'perf', 'refactor', 'docs', 'test', 'ci', 'build', 'chore', 'style']
        
        for commit_type in type_order:
            if commit_type in categories:
                sorted_categories[commit_type] = sorted(
                    categories[commit_type], 
                    key=lambda c: (c.breaking, c.scope in self.HIGH_IMPACT_SCOPES, c.scope),
                    reverse=True
                )
        
        # Add any remaining categories
        for commit_type, commits_list in categories.items():
            if commit_type not in sorted_categories:
                sorted_categories[commit_type] = commits_list
                
        return sorted_categories
    
    def generate_changelog_section(self, version: str, date: str, categories: Dict[str, List[CommitInfo]]) -> str:
        """Generate a changelog section for a version."""
        changelog = f"## [{version}] - {date}\\n\\n"
        
        # Add breaking changes first if any
        breaking_changes = []
        for commits_list in categories.values():
            breaking_changes.extend([c for c in commits_list if c.breaking])
        
        if breaking_changes:
            changelog += "### ⚠️ BREAKING CHANGES\\n\\n"
            for commit in breaking_changes:
                scope_text = f"**{commit.scope}**: " if commit.scope else ""
                changelog += f"- {scope_text}{commit.description}\\n"
            changelog += "\\n"
        
        # Add other categories
        for commit_type, commits_list in categories.items():
            if not commits_list:
                continue
                
            type_info = self.COMMIT_TYPES.get(commit_type, {
                'emoji': '🔀', 'title': commit_type.title(), 'description': f'{commit_type} changes'
            })
            
            changelog += f"### {type_info['emoji']} {type_info['title']}\\n\\n"
            
            for commit in commits_list:
                if commit.breaking:
                    continue  # Already handled in breaking changes
                    
                scope_text = f"**{commit.scope}**: " if commit.scope else ""
                pr_text = f" ([#{commit.pr_number}](https://github.com/SakanaAI/AI-Scientist-v2/pull/{commit.pr_number}))" if commit.pr_number else ""
                changelog += f"- {scope_text}{commit.description}{pr_text}\\n"
            
            changelog += "\\n"
        
        return changelog
    
    def update_changelog(self, version: str, new_content: str) -> None:
        """Update the CHANGELOG.md file with new content."""
        if not self.changelog_path.exists():
            # Create new changelog
            header = """# Changelog

All notable changes to AI Scientist v2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
            content = header + new_content
        else:
            # Insert new content after header
            with open(self.changelog_path, 'r') as f:
                existing_content = f.read()
            
            # Find the insertion point (after the header)
            header_end = existing_content.find('## [')
            if header_end == -1:
                # No existing versions, append to end
                content = existing_content + "\\n" + new_content
            else:
                # Insert before first version
                content = existing_content[:header_end] + new_content + existing_content[header_end:]
        
        with open(self.changelog_path, 'w') as f:
            f.write(content)
    
    def generate_release_notes(self, version: str, categories: Dict[str, List[CommitInfo]]) -> str:
        """Generate GitHub release notes."""
        notes = f"# AI Scientist v{version}\\n\\n"
        
        # Add summary
        total_commits = sum(len(commits) for commits in categories.values())
        notes += f"This release includes {total_commits} changes with improvements across multiple areas.\\n\\n"
        
        # Add highlights
        highlights = []
        for commit_type in ['security', 'feat', 'perf']:
            if commit_type in categories and categories[commit_type]:
                type_info = self.COMMIT_TYPES[commit_type]
                count = len(categories[commit_type])
                highlights.append(f"- {type_info['emoji']} {count} {type_info['title'].lower()}")
        
        if highlights:
            notes += "## 🌟 Highlights\\n\\n"
            notes += "\\n".join(highlights) + "\\n\\n"
        
        # Add detailed sections (same as changelog but with GitHub formatting)
        return notes + self.generate_changelog_section(version, datetime.now().strftime("%Y-%m-%d"), categories)


def main():
    """Main entry point for changelog generation."""
    parser = argparse.ArgumentParser(description="Generate automated changelog for AI Scientist v2")
    parser.add_argument("--version", required=True, help="Version number for the changelog")
    parser.add_argument("--since", help="Generate changelog since this tag/commit")
    parser.add_argument("--output", choices=['changelog', 'release-notes', 'both'], default='both',
                       help="Type of output to generate")
    parser.add_argument("--dry-run", action='store_true', help="Print output without writing files")
    
    args = parser.parse_args()
    
    generator = ChangelogGenerator()
    
    # Get commits
    commit_lines = generator.get_commits_since_tag(args.since)
    if not commit_lines:
        print("No commits found.")
        return
    
    # Parse commits
    commits = []
    for line in commit_lines:
        if '|' in line:  # Skip body lines
            commit = generator.parse_commit(line)
            if commit:
                commits.append(commit)
    
    if not commits:
        print("No valid commits found.")
        return
    
    print(f"Found {len(commits)} commits to process.")
    
    # Categorize commits
    categories = generator.categorize_commits(commits)
    
    # Generate outputs
    date = datetime.now().strftime("%Y-%m-%d")
    
    if args.output in ['changelog', 'both']:
        changelog_content = generator.generate_changelog_section(args.version, date, categories)
        
        if args.dry_run:
            print("=== CHANGELOG CONTENT ===")
            print(changelog_content)
        else:
            generator.update_changelog(args.version, changelog_content)
            print(f"Updated CHANGELOG.md with version {args.version}")
    
    if args.output in ['release-notes', 'both']:
        release_notes = generator.generate_release_notes(args.version, categories)
        
        if args.dry_run:
            print("=== RELEASE NOTES ===")
            print(release_notes)
        else:
            with open(f"release-notes-v{args.version}.md", 'w') as f:
                f.write(release_notes)
            print(f"Generated release-notes-v{args.version}.md")


if __name__ == "__main__":
    main()