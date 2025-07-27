import os
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Repository:
    name: str
    full_name: str
    description: Optional[str]
    homepage: Optional[str] 
    topics: List[str]
    stargazers_count: int
    pushed_at: str
    archived: bool
    fork: bool
    template: bool
    owner: str


class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'repo-hygiene-bot/1.0'
        }
        self.base_url = 'https://api.github.com'
        
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        
        kwargs = {
            'headers': self.headers,
            'timeout': 30
        }
        
        if params:
            kwargs['params'] = params
            
        if data:
            kwargs['json'] = data
            
        response = requests.request(method, url, **kwargs)
        
        if response.status_code == 404:
            return {}
        elif not response.ok:
            print(f"GitHub API error {response.status_code}: {response.text}")
            response.raise_for_status()
            
        return response.json() if response.content else {}
    
    def get_user_repositories(self, per_page: int = 100) -> List[Repository]:
        repos = []
        page = 1
        
        while True:
            params = {
                'per_page': per_page,
                'page': page,
                'affiliation': 'owner',
                'sort': 'updated',
                'direction': 'desc'
            }
            
            data = self._request('GET', '/user/repos', params=params)
            
            if not data:
                break
                
            for repo_data in data:
                if repo_data.get('archived') or repo_data.get('fork') or repo_data.get('template'):
                    continue
                    
                repos.append(Repository(
                    name=repo_data['name'],
                    full_name=repo_data['full_name'],
                    description=repo_data.get('description'),
                    homepage=repo_data.get('homepage'),
                    topics=repo_data.get('topics', []),
                    stargazers_count=repo_data['stargazers_count'],
                    pushed_at=repo_data['pushed_at'],
                    archived=repo_data['archived'],
                    fork=repo_data['fork'],
                    template=repo_data.get('template', False),
                    owner=repo_data['owner']['login']
                ))
            
            if len(data) < per_page:
                break
                
            page += 1
            
        return repos
    
    def update_repository(self, owner: str, repo: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('PATCH', f'/repos/{owner}/{repo}', data=data)
    
    def set_repository_topics(self, owner: str, repo: str, topics: List[str]) -> Dict[str, Any]:
        data = {'names': topics}
        return self._request('PUT', f'/repos/{owner}/{repo}/topics', data=data)
    
    def get_repository_content(self, owner: str, repo: str, path: str) -> Optional[Dict[str, Any]]:
        try:
            return self._request('GET', f'/repos/{owner}/{repo}/contents/{path}')
        except requests.exceptions.HTTPError:
            return None
    
    def create_or_update_file(self, owner: str, repo: str, path: str, content: str, message: str, sha: Optional[str] = None) -> Dict[str, Any]:
        import base64
        
        data = {
            'message': message,
            'content': base64.b64encode(content.encode()).decode(),
            'branch': 'main'
        }
        
        if sha:
            data['sha'] = sha
            
        return self._request('PUT', f'/repos/{owner}/{repo}/contents/{path}', data=data)
    
    def create_pull_request(self, owner: str, repo: str, title: str, body: str, head: str, base: str = 'main') -> Dict[str, Any]:
        data = {
            'title': title,
            'body': body,
            'head': head,
            'base': base
        }
        return self._request('POST', f'/repos/{owner}/{repo}/pulls', data=data)
    
    def create_branch(self, owner: str, repo: str, branch_name: str, from_branch: str = 'main') -> Dict[str, Any]:
        main_branch = self._request('GET', f'/repos/{owner}/{repo}/git/ref/heads/{from_branch}')
        sha = main_branch['object']['sha']
        
        data = {
            'ref': f'refs/heads/{branch_name}',
            'sha': sha
        }
        return self._request('POST', f'/repos/{owner}/{repo}/git/refs', data=data)
    
    def pin_repositories(self, repo_names: List[str]) -> Dict[str, Any]:
        data = {'repository_ids': repo_names}
        return self._request('PUT', '/user/pinned_repositories', data=data)
    
    def get_workflows(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        data = self._request('GET', f'/repos/{owner}/{repo}/actions/workflows')
        return data.get('workflows', [])
    
    def is_stale_repository(self, pushed_at: str, days_threshold: int = 400) -> bool:
        last_push = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
        threshold_date = datetime.now().replace(tzinfo=last_push.tzinfo) - timedelta(days=days_threshold)
        return last_push < threshold_date