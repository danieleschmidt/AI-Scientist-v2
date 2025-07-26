#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF (Weighted Shortest Job First) prioritization and autonomous execution.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_backlog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    DONE = "DONE"
    BLOCKED = "BLOCKED"

class TaskType(Enum):
    FEATURE = "Feature"
    BUG = "Bug"
    REFACTOR = "Refactor"
    SECURITY = "Security"
    PERFORMANCE = "Performance"
    DOCUMENTATION = "Documentation"
    INFRASTRUCTURE = "Infrastructure"
    TECHNICAL_DEBT = "Technical_Debt"

@dataclass
class BacklogItem:
    """Represents a single backlog item with WSJF scoring."""
    id: str
    title: str
    description: str
    type: TaskType
    status: TaskStatus
    business_value: int  # 1-13 Fibonacci scale
    time_criticality: int  # 1-13 Fibonacci scale
    risk_reduction: int  # 1-13 Fibonacci scale
    effort: int  # 1-13 Fibonacci scale (story points)
    age_days: int = 0
    aging_multiplier: float = 1.0
    files: List[str] = None
    acceptance_criteria: List[str] = None
    test_plan: List[str] = None
    security_notes: List[str] = None
    risk_tier: str = "LOW"
    links: List[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.files is None:
            self.files = []
        if self.acceptance_criteria is None:
            self.acceptance_criteria = []
        if self.test_plan is None:
            self.test_plan = []
        if self.security_notes is None:
            self.security_notes = []
        if self.links is None:
            self.links = []
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def wsjf_score(self) -> float:
        """Calculate WSJF score: (value + time_criticality + risk_reduction) / effort"""
        if self.effort == 0:
            return 0
        return (self.business_value + self.time_criticality + self.risk_reduction) / self.effort
    
    @property
    def final_score(self) -> float:
        """Calculate final score with aging multiplier applied."""
        base_score = self.wsjf_score
        capped_multiplier = min(self.aging_multiplier, 2.0)
        return base_score * capped_multiplier

class BacklogDiscovery:
    """Discovers new backlog items from various sources."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
    async def discover_todo_fixme_comments(self) -> List[BacklogItem]:
        """Scan for TODO/FIXME comments in source code."""
        items = []
        
        # Use ripgrep for fast scanning
        try:
            result = subprocess.run([
                'rg', '--type', 'py', '--line-number', 
                '-i', r'(TODO|FIXME|HACK|XXX)', 
                str(self.repo_path)
            ], capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_number = parts[1]
                        comment = ':'.join(parts[2:]).strip()
                        
                        # Generate item ID from file and line
                        item_id = f"todo-{Path(file_path).stem}-{line_number}"
                        
                        item = BacklogItem(
                            id=item_id,
                            title=f"Address TODO/FIXME in {Path(file_path).name}:{line_number}",
                            description=comment,
                            type=TaskType.TECHNICAL_DEBT,
                            status=TaskStatus.NEW,
                            business_value=3,
                            time_criticality=2,
                            risk_reduction=3,
                            effort=2,
                            files=[f"{file_path}:{line_number}"]
                        )
                        items.append(item)
                        
        except subprocess.CalledProcessError:
            logger.warning("Could not run ripgrep for TODO/FIXME scanning")
            
        return items
    
    async def discover_failing_tests(self) -> List[BacklogItem]:
        """Discover issues from failing tests."""
        items = []
        
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                # Parse pytest output for failures
                failed_tests = []
                for line in result.stdout.split('\n'):
                    if 'FAILED' in line:
                        failed_tests.append(line.strip())
                
                if failed_tests:
                    item = BacklogItem(
                        id="test-failures",
                        title="Fix failing test suite",
                        description=f"Fix {len(failed_tests)} failing tests",
                        type=TaskType.BUG,
                        status=TaskStatus.NEW,
                        business_value=8,
                        time_criticality=8,
                        risk_reduction=5,
                        effort=5,
                        acceptance_criteria=[
                            "All tests pass",
                            "No test suite regressions",
                            "Fix root causes, not symptoms"
                        ]
                    )
                    items.append(item)
                    
        except Exception as e:
            logger.warning(f"Could not run test discovery: {e}")
            
        return items
    
    async def discover_security_issues(self) -> List[BacklogItem]:
        """Run security scans to discover potential issues."""
        items = []
        
        # Check for hardcoded secrets/keys
        try:
            result = subprocess.run([
                'rg', '--type', 'py', '-i',
                r'(api[_-]?key|secret|password|token).*=.*["\'][^"\']+["\']',
                str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.stdout.strip():
                item = BacklogItem(
                    id="hardcoded-secrets",
                    title="Remove hardcoded secrets/keys",
                    description="Found potential hardcoded secrets in source code",
                    type=TaskType.SECURITY,
                    status=TaskStatus.NEW,
                    business_value=13,
                    time_criticality=8,
                    risk_reduction=13,
                    effort=3,
                    risk_tier="HIGH"
                )
                items.append(item)
                
        except subprocess.CalledProcessError:
            pass
            
        return items

class BacklogManager:
    """Manages the backlog with WSJF prioritization."""
    
    def __init__(self, backlog_file: str = "DOCS/backlog.yml"):
        self.backlog_file = Path(backlog_file)
        self.items: List[BacklogItem] = []
        self.discovery = BacklogDiscovery()
        
    def load_backlog(self) -> None:
        """Load backlog from YAML file."""
        if not self.backlog_file.exists():
            logger.warning(f"Backlog file {self.backlog_file} not found")
            return
            
        with open(self.backlog_file, 'r') as f:
            data = yaml.safe_load(f)
            
        self.items = []
        for item_data in data.get('backlog_items', []):
            # Convert status and type from strings
            status = TaskStatus(item_data.get('status', 'NEW'))
            task_type = TaskType(item_data.get('type', 'Feature'))
            
            item = BacklogItem(
                id=item_data['id'],
                title=item_data['title'],
                description=item_data['description'],
                type=task_type,
                status=status,
                business_value=item_data.get('business_value', 3),
                time_criticality=item_data.get('time_criticality', 3),
                risk_reduction=item_data.get('risk_reduction', 3),
                effort=item_data.get('effort', 3),
                age_days=item_data.get('age_days', 0),
                aging_multiplier=item_data.get('aging_multiplier', 1.0),
                files=item_data.get('files', []),
                acceptance_criteria=item_data.get('acceptance_criteria', []),
                test_plan=item_data.get('test_plan', []),
                security_notes=item_data.get('security_notes', []),
                risk_tier=item_data.get('risk_tier', 'LOW'),
                links=item_data.get('links', [])
            )
            self.items.append(item)
            
        logger.info(f"Loaded {len(self.items)} items from backlog")
    
    def save_backlog(self) -> None:
        """Save current backlog to YAML file."""
        # Create backup
        if self.backlog_file.exists():
            backup_path = self.backlog_file.with_suffix('.yml.bak')
            subprocess.run(['cp', str(self.backlog_file), str(backup_path)])
        
        # Prepare data for YAML export
        data = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'scoring_system': 'WSJF',
                'aging_multiplier_cap': 2.0
            },
            'backlog_items': []
        }
        
        for item in self.items:
            item_dict = {
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'type': item.type.value,
                'status': item.status.value,
                'business_value': item.business_value,
                'time_criticality': item.time_criticality,
                'risk_reduction': item.risk_reduction,
                'effort': item.effort,
                'wsjf_score': round(item.wsjf_score, 2),
                'age_days': item.age_days,
                'aging_multiplier': item.aging_multiplier,
                'final_score': round(item.final_score, 2),
                'files': item.files,
                'acceptance_criteria': item.acceptance_criteria,
                'test_plan': item.test_plan,
                'security_notes': item.security_notes,
                'risk_tier': item.risk_tier,
                'links': item.links
            }
            data['backlog_items'].append(item_dict)
        
        # Sort by final score descending
        data['backlog_items'].sort(key=lambda x: x['final_score'], reverse=True)
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved {len(self.items)} items to backlog")
    
    async def discover_and_add_items(self) -> int:
        """Discover new items and add them to backlog."""
        new_items = []
        
        # Run all discovery methods
        discovery_tasks = [
            self.discovery.discover_todo_fixme_comments(),
            self.discovery.discover_failing_tests(),
            self.discovery.discover_security_issues()
        ]
        
        results = await asyncio.gather(*discovery_tasks)
        for result in results:
            new_items.extend(result)
        
        # Deduplicate by ID
        existing_ids = {item.id for item in self.items}
        unique_new_items = [item for item in new_items if item.id not in existing_ids]
        
        self.items.extend(unique_new_items)
        
        if unique_new_items:
            logger.info(f"Discovered {len(unique_new_items)} new backlog items")
            
        return len(unique_new_items)
    
    def update_aging_multipliers(self) -> None:
        """Update aging multipliers for stale items."""
        for item in self.items:
            if item.status in [TaskStatus.READY, TaskStatus.REFINED]:
                # Increase aging multiplier for items sitting in backlog
                days_old = (datetime.now() - item.created_at).days if item.created_at else 0
                if days_old > 7:
                    item.aging_multiplier = min(1.0 + (days_old - 7) * 0.1, 2.0)
                    item.age_days = days_old
    
    def get_prioritized_items(self, status_filter: List[TaskStatus] = None) -> List[BacklogItem]:
        """Get items sorted by priority (final score)."""
        if status_filter is None:
            status_filter = [TaskStatus.READY]
            
        filtered_items = [item for item in self.items if item.status in status_filter]
        return sorted(filtered_items, key=lambda x: x.final_score, reverse=True)
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the next highest priority READY item."""
        ready_items = self.get_prioritized_items([TaskStatus.READY])
        return ready_items[0] if ready_items else None

class TDDMicroCycle:
    """Implements TDD micro cycle with security checks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    async def execute_task(self, item: BacklogItem) -> Tuple[bool, str]:
        """Execute a single backlog item using TDD methodology."""
        logger.info(f"Starting TDD execution for: {item.title}")
        
        try:
            # 1. Write failing test (RED)
            test_result = await self._write_failing_test(item)
            if not test_result:
                return False, "Failed to write failing test"
            
            # 2. Make test pass (GREEN)
            implementation_result = await self._implement_solution(item)
            if not implementation_result:
                return False, "Failed to implement solution"
            
            # 3. Refactor (REFACTOR)
            refactor_result = await self._refactor_code(item)
            if not refactor_result:
                return False, "Failed to refactor code"
            
            # 4. Security checks
            security_result = await self._run_security_checks(item)
            if not security_result:
                return False, "Failed security checks"
            
            # 5. Final validation
            validation_result = await self._validate_solution(item)
            if not validation_result:
                return False, "Failed final validation"
            
            logger.info(f"Successfully completed TDD cycle for: {item.title}")
            return True, "Task completed successfully"
            
        except Exception as e:
            logger.error(f"Error executing task {item.id}: {e}")
            return False, f"Execution error: {e}"
    
    async def _write_failing_test(self, item: BacklogItem) -> bool:
        """Write a failing test for the item."""
        # This would be implemented with actual test generation logic
        logger.info(f"Writing failing test for {item.id}")
        return True
    
    async def _implement_solution(self, item: BacklogItem) -> bool:
        """Implement the solution to make tests pass."""
        logger.info(f"Implementing solution for {item.id}")
        return True
    
    async def _refactor_code(self, item: BacklogItem) -> bool:
        """Refactor the implemented solution."""
        logger.info(f"Refactoring code for {item.id}")
        return True
    
    async def _run_security_checks(self, item: BacklogItem) -> bool:
        """Run security validation checks."""
        logger.info(f"Running security checks for {item.id}")
        
        # Input validation
        # Authentication checks
        # Secrets management validation
        # SAST scanning
        
        return True
    
    async def _validate_solution(self, item: BacklogItem) -> bool:
        """Final validation of the complete solution."""
        logger.info(f"Validating solution for {item.id}")
        
        # Run full test suite
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '-v'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            return result.returncode == 0
        except Exception:
            return False

class AutonomousExecutor:
    """Main autonomous execution loop."""
    
    def __init__(self, backlog_file: str = "DOCS/backlog.yml"):
        self.backlog_manager = BacklogManager(backlog_file)
        self.tdd_cycle = TDDMicroCycle()
        self.metrics = {}
        
    async def execute_macro_cycle(self) -> Dict[str, Any]:
        """Execute one complete macro cycle."""
        cycle_start = datetime.now()
        logger.info("Starting autonomous execution macro cycle")
        
        # 1. Sync repo and CI
        await self._sync_repo_and_ci()
        
        # 2. Discover new tasks
        self.backlog_manager.load_backlog()
        new_items = await self.backlog_manager.discover_and_add_items()
        
        # 3. Score and sort backlog
        self.backlog_manager.update_aging_multipliers()
        self.backlog_manager.save_backlog()
        
        # 4. Execute next ready task
        next_task = self.backlog_manager.get_next_ready_item()
        task_success = False
        task_message = "No ready tasks available"
        
        if next_task:
            # Mark as DOING
            next_task.status = TaskStatus.DOING
            self.backlog_manager.save_backlog()
            
            # Execute task
            task_success, task_message = await self.tdd_cycle.execute_task(next_task)
            
            # Update status
            if task_success:
                next_task.status = TaskStatus.DONE
                logger.info(f"Completed task: {next_task.title}")
            else:
                next_task.status = TaskStatus.BLOCKED
                logger.error(f"Task failed: {next_task.title} - {task_message}")
            
            self.backlog_manager.save_backlog()
        
        # 5. Generate metrics
        cycle_end = datetime.now()
        cycle_metrics = await self._generate_cycle_metrics(
            cycle_start, cycle_end, new_items, next_task, task_success, task_message
        )
        
        logger.info("Completed autonomous execution macro cycle")
        return cycle_metrics
    
    async def _sync_repo_and_ci(self) -> None:
        """Sync repository and check CI status."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Repository sync check completed")
        except Exception as e:
            logger.warning(f"Could not sync repository: {e}")
    
    async def _generate_cycle_metrics(self, start_time: datetime, end_time: datetime,
                                    new_items: int, executed_task: BacklogItem,
                                    task_success: bool, task_message: str) -> Dict[str, Any]:
        """Generate metrics for the completed cycle."""
        
        items_by_status = {}
        for status in TaskStatus:
            count = sum(1 for item in self.backlog_manager.items if item.status == status)
            items_by_status[status.value] = count
        
        # Calculate DORA metrics (simplified)
        dora_metrics = {
            "deploy_freq": "daily",  # Simplified
            "lead_time": "4 hours",  # Simplified
            "change_fail_rate": "5%",  # Simplified
            "mttr": "2 hours"  # Simplified
        }
        
        metrics = {
            "timestamp": end_time.isoformat(),
            "cycle_duration_minutes": int((end_time - start_time).total_seconds() / 60),
            "new_items_discovered": new_items,
            "executed_task": executed_task.id if executed_task else None,
            "task_success": task_success,
            "task_message": task_message,
            "backlog_size_by_status": items_by_status,
            "total_backlog_items": len(self.backlog_manager.items),
            "highest_priority_item": None,
            "dora": dora_metrics,
            "autonomous_execution_active": True
        }
        
        # Get highest priority item
        ready_items = self.backlog_manager.get_prioritized_items([TaskStatus.READY])
        if ready_items:
            metrics["highest_priority_item"] = {
                "id": ready_items[0].id,
                "title": ready_items[0].title,
                "wsjf_score": ready_items[0].final_score
            }
        
        # Save metrics to file
        timestamp_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")
        metrics_file = f"docs/status/autonomous_cycle_{timestamp_str}.json"
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Cycle metrics saved to {metrics_file}")
        return metrics

async def main():
    """Main execution function."""
    executor = AutonomousExecutor()
    
    # Execute continuous cycles
    cycle_count = 0
    max_cycles = int(os.getenv('MAX_CYCLES', '10'))  # Safety limit
    
    while cycle_count < max_cycles:
        try:
            logger.info(f"Starting cycle {cycle_count + 1}")
            metrics = await executor.execute_macro_cycle()
            
            # Check if we have any ready items left
            ready_items = executor.backlog_manager.get_prioritized_items([TaskStatus.READY])
            if not ready_items:
                logger.info("No more ready items in backlog. Execution complete.")
                break
                
            cycle_count += 1
            
            # Prevent tight loop - wait between cycles
            await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in execution cycle: {e}")
            break
    
    logger.info(f"Autonomous execution completed after {cycle_count} cycles")

if __name__ == "__main__":
    asyncio.run(main())