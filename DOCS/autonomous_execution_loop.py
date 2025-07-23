#!/usr/bin/env python3
"""
Autonomous Backlog Execution Loop
Continuously processes all actionable items in the backlog using WSJF prioritization.
"""

import json
import time
import yaml
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    MERGED = "MERGED"
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
    business_value: int
    time_criticality: int
    risk_reduction: int
    effort: int
    age_days: int = 0
    aging_multiplier: float = 1.0
    files: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    test_plan: List[str] = field(default_factory=list)
    security_notes: List[str] = field(default_factory=list)
    
    @property
    def wsjf_score(self) -> float:
        """Calculate WSJF score: (Business Value + Time Criticality + Risk Reduction) / Effort"""
        cost_of_delay = self.business_value + self.time_criticality + self.risk_reduction
        return cost_of_delay / max(self.effort, 1)  # Avoid division by zero
    
    @property
    def final_score(self) -> float:
        """Calculate final score with aging multiplier."""
        return self.wsjf_score * min(self.aging_multiplier, 2.0)  # Cap aging multiplier
    
    def update_aging(self) -> None:
        """Update aging multiplier based on how long item has been in backlog."""
        if self.age_days > 7:
            self.aging_multiplier = 1 + (self.age_days - 7) * 0.1
        else:
            self.aging_multiplier = 1.0

class BacklogManager:
    """Manages the backlog and implements continuous execution loop."""
    
    def __init__(self, backlog_file: Path = Path("DOCS/backlog.yml")):
        self.backlog_file = backlog_file
        self.backlog: List[BacklogItem] = []
        self.completed_items: List[BacklogItem] = []
        self.status_dir = Path("DOCS/status")
        self.status_dir.mkdir(exist_ok=True)
        
    def load_backlog(self) -> None:
        """Load backlog from YAML file."""
        if not self.backlog_file.exists():
            logger.warning(f"Backlog file {self.backlog_file} not found")
            return
            
        with open(self.backlog_file, 'r') as f:
            data = yaml.safe_load(f)
            
        self.backlog = []
        for item_data in data.get('backlog_items', []):
            item = BacklogItem(
                id=item_data['id'],
                title=item_data['title'],
                description=item_data['description'],
                type=TaskType(item_data['type']),
                status=TaskStatus(item_data['status']),
                business_value=item_data['business_value'],
                time_criticality=item_data['time_criticality'],
                risk_reduction=item_data['risk_reduction'],
                effort=item_data['effort'],
                age_days=item_data.get('age_days', 0),
                aging_multiplier=item_data.get('aging_multiplier', 1.0),
                files=item_data.get('files', []),
                acceptance_criteria=item_data.get('acceptance_criteria', []),
                test_plan=item_data.get('test_plan', []),
                security_notes=item_data.get('security_notes', [])
            )
            self.backlog.append(item)
            
        logger.info(f"Loaded {len(self.backlog)} items from backlog")
    
    def save_backlog(self) -> None:
        """Save current backlog state to YAML file."""
        data = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'next_review': 'After completion of current high-priority items',
                'scoring_system': 'WSJF',
                'aging_multiplier_cap': 2.0
            },
            'backlog_items': []
        }
        
        # Add current backlog items
        for item in self.backlog:
            item_data = {
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
                'aging_multiplier': round(item.aging_multiplier, 2),
                'final_score': round(item.final_score, 2),
                'files': item.files,
                'acceptance_criteria': item.acceptance_criteria,
                'test_plan': item.test_plan,
                'security_notes': item.security_notes
            }
            data['backlog_items'].append(item_data)
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved {len(self.backlog)} items to backlog")
    
    def discover_new_tasks(self) -> List[BacklogItem]:
        """Discover new tasks from various sources."""
        new_tasks = []
        
        # Scan for TODO/FIXME comments
        try:
            result = subprocess.run(
                ['grep', '-r', '-n', '-i', 'TODO\\|FIXME\\|XXX\\|HACK', 'ai_scientist/'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\\n'):
                    if line.strip() and 'Binary file' not in line:
                        # Parse TODO/FIXME comments into tasks
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_num = parts[1]
                            comment = parts[2].strip()
                            
                            # Create task from TODO comment
                            task_id = f"todo-{hash(line)}"
                            if not any(item.id == task_id for item in self.backlog):
                                new_task = BacklogItem(
                                    id=task_id,
                                    title=f"Address TODO in {Path(file_path).name}",
                                    description=f"TODO comment: {comment}",
                                    type=TaskType.TECHNICAL_DEBT,
                                    status=TaskStatus.NEW,
                                    business_value=3,
                                    time_criticality=2,
                                    risk_reduction=2,
                                    effort=2,
                                    files=[f"{file_path}:{line_num}"]
                                )
                                new_tasks.append(new_task)
        except Exception as e:
            logger.error(f"Error discovering TODO comments: {e}")
        
        # Check for failing tests
        try:
            result = subprocess.run(['python3', 'run_tests.py'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode != 0 and 'FAILED' in result.stderr:
                # Create task for failing tests
                task_id = "failing-tests"
                if not any(item.id == task_id for item in self.backlog):
                    new_task = BacklogItem(
                        id=task_id,
                        title="Fix failing test cases",
                        description="Some test cases are failing and need attention",
                        type=TaskType.BUG,
                        status=TaskStatus.NEW,
                        business_value=6,
                        time_criticality=5,
                        risk_reduction=7,
                        effort=4,
                        files=["tests/"]
                    )
                    new_tasks.append(new_task)
        except Exception as e:
            logger.error(f"Error checking test status: {e}")
        
        if new_tasks:
            logger.info(f"Discovered {len(new_tasks)} new tasks")
            
        return new_tasks
    
    def score_and_rank(self) -> None:
        """Update scores and rank backlog items by final score."""
        for item in self.backlog:
            item.update_aging()
        
        # Sort by final score (descending)
        self.backlog.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info("Backlog scored and ranked")
    
    def get_next_actionable_item(self) -> Optional[BacklogItem]:
        """Get the next actionable item from the backlog."""
        for item in self.backlog:
            if item.status == TaskStatus.READY and item.status != TaskStatus.BLOCKED:
                return item
        return None
    
    def implement_tdd_cycle(self, item: BacklogItem) -> bool:
        """
        Implement TDD cycle for a backlog item.
        Returns True if successful, False if failed or blocked.
        """
        logger.info(f"Starting TDD cycle for: {item.title}")
        
        # Mark as in progress
        item.status = TaskStatus.DOING
        self.save_backlog()
        
        try:
            # 1. Write failing test (Red)
            logger.info("Step 1: Writing failing test")
            if not self._write_failing_test(item):
                logger.warning("Failed to write failing test")
                return False
            
            # 2. Implement minimal code (Green)
            logger.info("Step 2: Implementing minimal code")
            if not self._implement_minimal_code(item):
                logger.warning("Failed to implement minimal code")
                return False
            
            # 3. Refactor (Refactor)
            logger.info("Step 3: Refactoring")
            if not self._refactor_code(item):
                logger.warning("Failed to refactor code")
                return False
            
            # 4. Security and compliance checks
            logger.info("Step 4: Security and compliance checks")
            if not self._security_checks(item):
                logger.warning("Security checks failed")
                return False
            
            # 5. Run full CI pipeline
            logger.info("Step 5: Running full CI pipeline")
            if not self._run_ci_pipeline():
                logger.warning("CI pipeline failed")
                return False
            
            # Mark as ready for PR
            item.status = TaskStatus.PR
            logger.info(f"Completed TDD cycle for: {item.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error in TDD cycle for {item.title}: {e}")
            item.status = TaskStatus.BLOCKED
            return False
    
    def _write_failing_test(self, item: BacklogItem) -> bool:
        """Write a failing test for the backlog item."""
        # This would contain actual test writing logic
        # For now, return True as placeholder
        return True
    
    def _implement_minimal_code(self, item: BacklogItem) -> bool:
        """Implement minimal code to make the test pass."""
        # This would contain actual implementation logic
        # For now, return True as placeholder
        return True
    
    def _refactor_code(self, item: BacklogItem) -> bool:
        """Refactor the implementation for better design."""
        # This would contain actual refactoring logic
        # For now, return True as placeholder
        return True
    
    def _security_checks(self, item: BacklogItem) -> bool:
        """Run security and compliance checks."""
        # Run security linting tools
        try:
            result = subprocess.run(['bandit', '-r', 'ai_scientist/'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Security checks passed")
                return True
            else:
                logger.warning("Security checks found issues")
                return False
        except Exception as e:
            logger.error(f"Error running security checks: {e}")
            return False
    
    def _run_ci_pipeline(self) -> bool:
        """Run the full CI pipeline."""
        try:
            # Run tests
            result = subprocess.run(['python3', 'run_tests.py'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Tests failed")
                return False
            
            logger.info("CI pipeline passed")
            return True
        except Exception as e:
            logger.error(f"Error running CI pipeline: {e}")
            return False
    
    def generate_status_report(self) -> Dict:
        """Generate current status report."""
        ready_items = [item for item in self.backlog if item.status == TaskStatus.READY]
        blocked_items = [item for item in self.backlog if item.status == TaskStatus.BLOCKED]
        in_progress_items = [item for item in self.backlog if item.status == TaskStatus.DOING]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'backlog_size_by_status': {
                'total': len(self.backlog),
                'ready': len(ready_items),
                'blocked': len(blocked_items),
                'in_progress': len(in_progress_items),
                'completed': len(self.completed_items)
            },
            'top_priority_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'final_score': round(item.final_score, 2),
                    'status': item.status.value
                }
                for item in self.backlog[:5]
            ],
            'blocked_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'reason': 'Implementation blocked'
                }
                for item in blocked_items
            ]
        }
        
        # Save status report
        status_file = self.status_dir / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(status_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def continuous_execution_loop(self, max_iterations: Optional[int] = None) -> None:
        """
        Main continuous execution loop.
        Processes all actionable items in the backlog until none remain.
        """
        logger.info("Starting continuous execution loop")
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")
            
            # 1. Sync & Refresh
            logger.info("Step 1: Sync & Refresh")
            self.load_backlog()
            new_tasks = self.discover_new_tasks()
            self.backlog.extend(new_tasks)
            self.score_and_rank()
            
            # 2. Select Next Feasible Item
            logger.info("Step 2: Select Next Feasible Item")
            next_item = self.get_next_actionable_item()
            
            if not next_item:
                logger.info("No actionable items found. Checking for blocked items...")
                blocked_items = [item for item in self.backlog if item.status == TaskStatus.BLOCKED]
                if not blocked_items:
                    logger.info("Backlog is empty or all items are completed. Loop complete.")
                    break
                else:
                    logger.info(f"All remaining {len(blocked_items)} items are blocked. Waiting...")
                    time.sleep(60)  # Wait before checking again
                    continue
            
            logger.info(f"Selected item: {next_item.title} (Score: {next_item.final_score:.2f})")
            
            # 3. Execute Per-Item Micro-Cycle
            logger.info("Step 3: Execute TDD Implementation")
            success = self.implement_tdd_cycle(next_item)
            
            if success:
                logger.info(f"Successfully implemented: {next_item.title}")
                # Move to completed
                self.backlog.remove(next_item)
                next_item.status = TaskStatus.DONE
                self.completed_items.append(next_item)
            else:
                logger.warning(f"Implementation failed for: {next_item.title}")
                # Item remains in backlog with updated status
            
            # 4. Update and Report
            logger.info("Step 4: Update and Report")
            self.save_backlog()
            report = self.generate_status_report()
            
            logger.info(f"Status: {report['backlog_size_by_status']}")
            
            # Small delay between iterations
            time.sleep(5)
        
        logger.info("Continuous execution loop completed")

def main():
    """Main entry point for autonomous execution."""
    manager = BacklogManager()
    
    try:
        manager.continuous_execution_loop(max_iterations=10)  # Limit for safety
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"Execution loop failed: {e}")
        raise

if __name__ == "__main__":
    main()