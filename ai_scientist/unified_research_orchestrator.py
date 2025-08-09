#!/usr/bin/env python3
"""
Unified Research Orchestrator - Generation 1: MAKE IT WORK

Coordinates the entire scientific research workflow from ideation to publication.
Provides unified interface for hypothesis generation, experiment execution, and paper writing.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


@dataclass
class ResearchTask:
    """Represents a single research task in the workflow."""
    task_id: str
    task_type: str  # 'ideation', 'experiment', 'analysis', 'writing'
    parameters: Dict[str, Any]
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass 
class ResearchWorkflow:
    """Defines a complete research workflow configuration."""
    workflow_id: str
    name: str
    description: str
    tasks: List[ResearchTask]
    dependencies: Dict[str, List[str]]  # task_id -> list of prerequisite task_ids
    metadata: Dict[str, Any]


class UnifiedResearchOrchestrator:
    """
    Generation 1: MAKE IT WORK
    Unified orchestrator for the complete scientific research pipeline.
    """
    
    def __init__(self, workspace_dir: str = "orchestrator_workspace"):
        self.console = Console()
        self.logger = self._setup_logging()
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Core workflow tracking
        self.active_workflows: Dict[str, ResearchWorkflow] = {}
        self.completed_workflows: List[str] = []
        self.failed_workflows: List[str] = []
        
        # Execution management
        self.max_concurrent_tasks = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        
        # Success metrics
        self.success_metrics = {
            'workflows_completed': 0,
            'tasks_completed': 0,
            'average_execution_time': 0.0,
            'success_rate': 0.0
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the orchestrator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def create_standard_workflow(
        self, 
        research_topic: str,
        models_config: Optional[Dict[str, str]] = None
    ) -> ResearchWorkflow:
        """Create a standard end-to-end research workflow."""
        
        if models_config is None:
            models_config = {
                'ideation_model': 'gpt-4o-2024-11-20',
                'experiment_model': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                'analysis_model': 'gpt-4o-2024-11-20',
                'writing_model': 'o1-preview-2024-09-12'
            }
            
        workflow_id = f"research_{int(time.time())}"
        
        # Define research tasks
        tasks = [
            ResearchTask(
                task_id="ideation",
                task_type="ideation", 
                parameters={
                    'topic': research_topic,
                    'model': models_config['ideation_model'],
                    'max_num_generations': 20,
                    'num_reflections': 5,
                    'output_file': f"{workflow_id}_ideas.json"
                }
            ),
            ResearchTask(
                task_id="experiment_setup",
                task_type="experiment_setup",
                parameters={
                    'ideas_file': f"{workflow_id}_ideas.json", 
                    'model': models_config['experiment_model'],
                    'workspace': f"experiments/{workflow_id}"
                }
            ),
            ResearchTask(
                task_id="experiment_execution",
                task_type="experiment",
                parameters={
                    'experiment_config': 'bfts_config.yaml',
                    'model': models_config['experiment_model'],
                    'max_steps': 21,
                    'num_workers': 3
                }
            ),
            ResearchTask(
                task_id="result_analysis", 
                task_type="analysis",
                parameters={
                    'experiment_dir': f"experiments/{workflow_id}",
                    'model': models_config['analysis_model'],
                    'metrics_to_analyze': ['accuracy', 'performance', 'novelty']
                }
            ),
            ResearchTask(
                task_id="paper_writing",
                task_type="writing",
                parameters={
                    'results_dir': f"experiments/{workflow_id}",
                    'model_writeup': models_config['writing_model'],
                    'model_citation': 'gpt-4o-2024-11-20',
                    'template': 'icml2025',
                    'num_cite_rounds': 20
                }
            ),
            ResearchTask(
                task_id="quality_review",
                task_type="review",
                parameters={
                    'paper_path': f"experiments/{workflow_id}/paper.pdf",
                    'model': 'gpt-4o-2024-11-20',
                    'review_criteria': ['novelty', 'methodology', 'clarity', 'reproducibility']
                }
            )
        ]
        
        # Define task dependencies
        dependencies = {
            'experiment_setup': ['ideation'],
            'experiment_execution': ['experiment_setup'], 
            'result_analysis': ['experiment_execution'],
            'paper_writing': ['result_analysis'],
            'quality_review': ['paper_writing']
        }
        
        workflow = ResearchWorkflow(
            workflow_id=workflow_id,
            name=f"Research: {research_topic}",
            description=f"End-to-end research workflow for topic: {research_topic}",
            tasks=tasks,
            dependencies=dependencies,
            metadata={
                'created_at': datetime.now().isoformat(),
                'topic': research_topic,
                'models': models_config
            }
        )
        
        return workflow
        
    def submit_workflow(self, workflow: ResearchWorkflow) -> str:
        """Submit a workflow for execution."""
        self.active_workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Submitted workflow: {workflow.name} (ID: {workflow.workflow_id})")
        return workflow.workflow_id
        
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a research workflow asynchronously."""
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.active_workflows[workflow_id]
        start_time = time.time()
        
        self.console.print(f"[bold green]🚀 Starting workflow: {workflow.name}[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            
            workflow_task = progress.add_task(
                description="Overall Progress",
                total=len(workflow.tasks)
            )
            
            try:
                # Execute tasks based on dependencies
                completed_tasks = set()
                
                while len(completed_tasks) < len(workflow.tasks):
                    # Find ready tasks (dependencies satisfied)
                    ready_tasks = []
                    for task in workflow.tasks:
                        if (task.task_id not in completed_tasks and 
                            task.status != 'running' and 
                            self._are_dependencies_satisfied(task.task_id, workflow.dependencies, completed_tasks)):
                            ready_tasks.append(task)
                    
                    if not ready_tasks:
                        # Check if we have running tasks
                        running_tasks = [t for t in workflow.tasks if t.status == 'running']
                        if not running_tasks:
                            raise RuntimeError("No ready tasks and no running tasks - possible circular dependency")
                        # Wait a bit for running tasks to complete
                        await asyncio.sleep(1)
                        continue
                    
                    # Execute ready tasks concurrently 
                    futures = []
                    for task in ready_tasks[:self.max_concurrent_tasks]:
                        task.status = 'running'
                        task.start_time = time.time()
                        future = asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self._execute_single_task,
                            task
                        )
                        futures.append((task.task_id, future))
                    
                    # Wait for at least one task to complete
                    if futures:
                        done, _ = await asyncio.wait(
                            [f for _, f in futures],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Process completed tasks
                        for task_id, future in futures:
                            if future in done:
                                try:
                                    result = await future
                                    task = next(t for t in workflow.tasks if t.task_id == task_id)
                                    task.results = result
                                    task.status = 'completed'
                                    task.end_time = time.time()
                                    completed_tasks.add(task_id)
                                    
                                    progress.update(workflow_task, advance=1)
                                    self.console.print(f"✅ Completed: {task.task_type} ({task_id})")
                                    
                                except Exception as e:
                                    task = next(t for t in workflow.tasks if t.task_id == task_id)
                                    task.status = 'failed'
                                    task.error = str(e)
                                    task.end_time = time.time()
                                    
                                    self.logger.error(f"Task {task_id} failed: {e}")
                                    self.console.print(f"❌ Failed: {task.task_type} ({task_id}): {e}")
                                    
                                    # For now, continue with other tasks (fail-soft approach)
                                    completed_tasks.add(task_id)
                                    progress.update(workflow_task, advance=1)
                
                # Calculate results
                end_time = time.time()
                execution_time = end_time - start_time
                
                success_count = sum(1 for task in workflow.tasks if task.status == 'completed')
                total_tasks = len(workflow.tasks)
                success_rate = success_count / total_tasks
                
                # Update metrics
                self.success_metrics['workflows_completed'] += 1
                self.success_metrics['tasks_completed'] += success_count
                self.success_metrics['success_rate'] = success_rate
                
                if success_rate > 0.7:  # Consider successful if >70% tasks completed
                    self.completed_workflows.append(workflow_id)
                    status = 'completed'
                    self.console.print(f"[bold green]✅ Workflow completed successfully![/bold green]")
                else:
                    self.failed_workflows.append(workflow_id)
                    status = 'failed'
                    self.console.print(f"[bold red]❌ Workflow failed (success rate: {success_rate:.1%})[/bold red]")
                
                # Save workflow results
                self._save_workflow_results(workflow_id, workflow)
                
                return {
                    'workflow_id': workflow_id,
                    'status': status,
                    'execution_time': execution_time,
                    'success_rate': success_rate,
                    'completed_tasks': success_count,
                    'total_tasks': total_tasks,
                    'results_saved_to': str(self.workspace_dir / f"{workflow_id}_results.json")
                }
                
            except Exception as e:
                self.logger.error(f"Workflow execution failed: {e}")
                self.failed_workflows.append(workflow_id)
                raise
        
    def _are_dependencies_satisfied(self, task_id: str, dependencies: Dict[str, List[str]], completed: set) -> bool:
        """Check if task dependencies are satisfied."""
        if task_id not in dependencies:
            return True
        return all(dep in completed for dep in dependencies[task_id])
        
    def _execute_single_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute a single research task."""
        
        self.logger.info(f"Executing task: {task.task_type} ({task.task_id})")
        
        # Task-specific execution logic
        if task.task_type == "ideation":
            return self._execute_ideation_task(task)
        elif task.task_type == "experiment_setup":
            return self._execute_experiment_setup_task(task)  
        elif task.task_type == "experiment":
            return self._execute_experiment_task(task)
        elif task.task_type == "analysis":
            return self._execute_analysis_task(task)
        elif task.task_type == "writing":
            return self._execute_writing_task(task)
        elif task.task_type == "review":
            return self._execute_review_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _execute_ideation_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute ideation task - generate research ideas."""
        params = task.parameters
        
        # Create topic file for ideation
        topic_file = self.workspace_dir / f"{task.task_id}_topic.md"
        topic_content = f"""# Research Topic: {params['topic']}

## Title
{params['topic']}

## Keywords  
artificial intelligence, machine learning, research automation, scientific discovery

## TL;DR
Explore novel approaches and improvements in {params['topic']} using automated research methodologies.

## Abstract
This research focuses on advancing the field of {params['topic']} through systematic exploration of novel algorithms, methodologies, and applications. The goal is to identify breakthrough opportunities and validate them through rigorous experimentation.
"""
        
        with open(topic_file, 'w') as f:
            f.write(topic_content)
        
        # Simulate ideation process (in real implementation, would call actual ideation module)
        ideas = {
            'ideas': [
                {
                    'title': f"Novel Approach to {params['topic']}",
                    'hypothesis': f"Advanced methods can significantly improve {params['topic']} performance",
                    'novelty_score': 0.85,
                    'feasibility_score': 0.78,
                    'expected_impact': 0.82
                }
            ],
            'generated_at': datetime.now().isoformat(),
            'model_used': params['model'],
            'topic_file': str(topic_file)
        }
        
        # Save ideas to output file
        output_file = self.workspace_dir / params['output_file']
        with open(output_file, 'w') as f:
            json.dump(ideas, f, indent=2)
            
        return {
            'ideas_generated': len(ideas['ideas']),
            'output_file': str(output_file),
            'topic_file': str(topic_file)
        }
    
    def _execute_experiment_setup_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute experiment setup task."""
        params = task.parameters
        
        # Create experiment workspace
        exp_workspace = Path(params['workspace'])
        exp_workspace.mkdir(parents=True, exist_ok=True)
        
        # Load ideas and prepare experiment configuration
        ideas_file = self.workspace_dir / params['ideas_file']
        if ideas_file.exists():
            with open(ideas_file, 'r') as f:
                ideas_data = json.load(f)
        else:
            ideas_data = {'ideas': [{'title': 'Default Experiment'}]}
        
        # Create experiment config
        exp_config = {
            'experiment_id': task.task_id,
            'ideas': ideas_data['ideas'],
            'workspace': str(exp_workspace),
            'model': params['model'],
            'setup_completed_at': datetime.now().isoformat()
        }
        
        config_file = exp_workspace / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(exp_config, f, indent=2)
            
        return {
            'workspace_created': str(exp_workspace),
            'config_file': str(config_file),
            'ideas_loaded': len(ideas_data['ideas'])
        }
    
    def _execute_experiment_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute core experiment task."""
        params = task.parameters
        
        # Simulate experiment execution
        time.sleep(2)  # Simulate processing time
        
        # Generate mock results
        results = {
            'experiment_completed': True,
            'metrics': {
                'accuracy': 0.87,
                'performance_improvement': 0.15,
                'novelty_validated': True
            },
            'model_used': params['model'],
            'steps_completed': params.get('max_steps', 21),
            'workers_used': params.get('num_workers', 3),
            'execution_time': 120.5
        }
        
        return results
    
    def _execute_analysis_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute result analysis task."""
        params = task.parameters
        
        # Simulate analysis
        time.sleep(1)
        
        analysis_results = {
            'analysis_completed': True,
            'key_findings': [
                'Significant performance improvement observed',
                'Novel approach validated experimentally',
                'Results exceed baseline by 15%'
            ],
            'statistical_significance': 0.002,  # p-value
            'reproducibility_score': 0.92,
            'metrics_analyzed': params.get('metrics_to_analyze', [])
        }
        
        return analysis_results
    
    def _execute_writing_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute paper writing task."""
        params = task.parameters
        
        # Simulate paper writing
        time.sleep(3)
        
        writing_results = {
            'paper_generated': True,
            'template_used': params.get('template', 'icml2025'),
            'model_writeup': params['model_writeup'],
            'citations_added': params.get('num_cite_rounds', 20),
            'sections_completed': ['abstract', 'introduction', 'methodology', 'results', 'conclusion'],
            'word_count': 6500,
            'paper_quality_score': 0.89
        }
        
        return writing_results
    
    def _execute_review_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute quality review task.""" 
        params = task.parameters
        
        # Simulate review process
        time.sleep(1)
        
        review_results = {
            'review_completed': True,
            'overall_score': 8.2,
            'criteria_scores': {
                'novelty': 8.5,
                'methodology': 8.0, 
                'clarity': 8.3,
                'reproducibility': 7.8
            },
            'recommendations': [
                'Strong contribution to the field',
                'Methodology is sound and well-executed', 
                'Results are clearly presented',
                'Minor improvements needed in reproducibility section'
            ],
            'publication_ready': True
        }
        
        return review_results
        
    def _save_workflow_results(self, workflow_id: str, workflow: ResearchWorkflow):
        """Save complete workflow results to file."""
        results_file = self.workspace_dir / f"{workflow_id}_results.json"
        
        workflow_results = {
            'workflow': asdict(workflow),
            'completion_time': datetime.now().isoformat(),
            'success_metrics': self.success_metrics.copy()
        }
        
        with open(results_file, 'w') as f:
            json.dump(workflow_results, f, indent=2, default=str)
            
        self.logger.info(f"Workflow results saved to: {results_file}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return {'error': f'Workflow {workflow_id} not found'}
            
        workflow = self.active_workflows[workflow_id]
        
        completed_tasks = sum(1 for task in workflow.tasks if task.status == 'completed')
        running_tasks = sum(1 for task in workflow.tasks if task.status == 'running')
        failed_tasks = sum(1 for task in workflow.tasks if task.status == 'failed')
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'total_tasks': len(workflow.tasks),
            'completed_tasks': completed_tasks,
            'running_tasks': running_tasks, 
            'failed_tasks': failed_tasks,
            'progress': completed_tasks / len(workflow.tasks),
            'status': 'completed' if workflow_id in self.completed_workflows 
                     else 'failed' if workflow_id in self.failed_workflows 
                     else 'running'
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows and their statuses."""
        workflows = []
        
        for workflow_id, workflow in self.active_workflows.items():
            status_info = self.get_workflow_status(workflow_id)
            workflows.append(status_info)
            
        return workflows
    
    def get_success_metrics(self) -> Dict[str, Any]:
        """Get overall system success metrics."""
        return self.success_metrics.copy()


# Demonstration and testing functions
def demo_research_orchestrator():
    """Demonstrate the research orchestrator functionality."""
    
    console = Console()
    console.print("[bold blue]🧪 AI Scientist Research Orchestrator - Generation 1 Demo[/bold blue]")
    
    # Initialize orchestrator
    orchestrator = UnifiedResearchOrchestrator()
    
    # Create a research workflow
    workflow = orchestrator.create_standard_workflow(
        research_topic="Quantum-Inspired Optimization for Neural Networks"
    )
    
    # Submit workflow
    workflow_id = orchestrator.submit_workflow(workflow)
    
    console.print(f"[green]✅ Created workflow: {workflow_id}[/green]")
    console.print(f"[cyan]📋 Tasks in workflow: {len(workflow.tasks)}[/cyan]")
    
    return orchestrator, workflow_id


async def main():
    """Main entry point for the research orchestrator."""
    
    orchestrator, workflow_id = demo_research_orchestrator()
    
    # Execute the workflow
    console = Console()
    console.print("\n[bold yellow]🚀 Executing research workflow...[/bold yellow]")
    
    try:
        results = await orchestrator.execute_workflow(workflow_id)
        
        console.print(f"\n[bold green]🎉 Workflow Results:[/bold green]")
        console.print(f"• Status: {results['status']}")
        console.print(f"• Execution Time: {results['execution_time']:.1f}s")
        console.print(f"• Success Rate: {results['success_rate']:.1%}")
        console.print(f"• Completed Tasks: {results['completed_tasks']}/{results['total_tasks']}")
        console.print(f"• Results Saved: {results['results_saved_to']}")
        
        # Show final metrics
        metrics = orchestrator.get_success_metrics()
        console.print(f"\n[bold cyan]📊 System Metrics:[/bold cyan]")
        console.print(f"• Workflows Completed: {metrics['workflows_completed']}")
        console.print(f"• Tasks Completed: {metrics['tasks_completed']}")
        console.print(f"• Success Rate: {metrics['success_rate']:.1%}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Workflow execution failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())