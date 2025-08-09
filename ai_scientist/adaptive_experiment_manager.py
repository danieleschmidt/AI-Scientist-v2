#!/usr/bin/env python3
"""
Adaptive Experiment Manager - Generation 1: MAKE IT WORK

Real-time experiment optimization with intelligent resource allocation and adaptive strategies.
Monitors experiment performance and automatically adjusts parameters for optimal results.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    INITIALIZING = "initializing" 
    RUNNING = "running"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ExperimentMetrics:
    """Real-time experiment metrics."""
    accuracy: float = 0.0
    loss: float = float('inf')
    convergence_rate: float = 0.0
    resource_utilization: float = 0.0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cost_efficiency: float = 0.0
    novelty_score: float = 0.0
    
    # Historical tracking
    history: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_metric(self, metric_name: str, value: float):
        """Update a metric and maintain history."""
        setattr(self, metric_name, value)
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)
        
    def get_trend(self, metric_name: str, window: int = 5) -> str:
        """Analyze trend for a specific metric."""
        if metric_name not in self.history or len(self.history[metric_name]) < 2:
            return "stable"
            
        recent_values = self.history[metric_name][-window:]
        if len(recent_values) < 2:
            return "stable"
            
        # Calculate trend
        slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving" if metric_name in ['accuracy', 'convergence_rate', 'cost_efficiency', 'novelty_score'] else "degrading"
        else:
            return "degrading" if metric_name in ['accuracy', 'convergence_rate', 'cost_efficiency', 'novelty_score'] else "improving"


@dataclass
class ExperimentConfiguration:
    """Adaptive experiment configuration."""
    experiment_id: str
    base_config: Dict[str, Any]
    adaptive_params: Dict[str, Any] = field(default_factory=dict)
    optimization_targets: List[str] = field(default_factory=lambda: ['accuracy', 'cost_efficiency'])
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptation settings
    adaptation_enabled: bool = True
    adaptation_frequency: int = 10  # seconds
    early_stopping_patience: int = 5
    performance_threshold: float = 0.95


@dataclass 
class ActiveExperiment:
    """Represents an active experiment being managed."""
    experiment_id: str
    config: ExperimentConfiguration
    status: ExperimentStatus = ExperimentStatus.PENDING
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    adaptation_count: int = 0
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class AdaptiveExperimentManager:
    """
    Generation 1: MAKE IT WORK
    Manages multiple experiments with real-time optimization and resource allocation.
    """
    
    def __init__(self, max_concurrent_experiments: int = 4):
        self.console = Console()
        self.logger = self._setup_logging()
        
        # Experiment tracking
        self.active_experiments: Dict[str, ActiveExperiment] = {}
        self.completed_experiments: Dict[str, ActiveExperiment] = {}
        self.failed_experiments: Dict[str, ActiveExperiment] = {}
        
        # Resource management
        self.max_concurrent_experiments = max_concurrent_experiments
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_experiments * 2)
        self.resource_allocator = ResourceAllocator()
        
        # Optimization engine
        self.optimizer = ExperimentOptimizer()
        
        # Global metrics
        self.system_metrics = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'average_execution_time': 0.0,
            'resource_efficiency': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Control flags
        self.monitoring_active = False
        self.optimization_active = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment manager."""
        logger = logging.getLogger(f"{__name__}.AdaptiveExperimentManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_experiment(
        self, 
        experiment_id: str,
        base_config: Dict[str, Any],
        optimization_targets: Optional[List[str]] = None
    ) -> str:
        """Create a new adaptive experiment configuration."""
        
        if optimization_targets is None:
            optimization_targets = ['accuracy', 'cost_efficiency']
            
        config = ExperimentConfiguration(
            experiment_id=experiment_id,
            base_config=base_config,
            optimization_targets=optimization_targets,
            adaptive_params={
                'learning_rate': base_config.get('learning_rate', 0.001),
                'batch_size': base_config.get('batch_size', 32),
                'max_iterations': base_config.get('max_iterations', 1000),
                'early_stopping': True
            }
        )
        
        experiment = ActiveExperiment(
            experiment_id=experiment_id,
            config=config
        )
        
        self.active_experiments[experiment_id] = experiment
        self.system_metrics['total_experiments'] += 1
        
        self.logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment with adaptive monitoring."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments[experiment_id]
        
        # Check resource availability
        if len([e for e in self.active_experiments.values() if e.status == ExperimentStatus.RUNNING]) >= self.max_concurrent_experiments:
            self.logger.warning(f"Maximum concurrent experiments reached. Queuing {experiment_id}")
            experiment.status = ExperimentStatus.PENDING
            return False
            
        # Allocate resources
        resources = self.resource_allocator.allocate_resources(experiment.config)
        if not resources:
            self.logger.error(f"Failed to allocate resources for {experiment_id}")
            experiment.status = ExperimentStatus.FAILED
            return False
            
        # Start experiment execution
        experiment.status = ExperimentStatus.INITIALIZING
        experiment.start_time = datetime.now()
        
        self.console.print(f"[bold green]🚀 Starting experiment: {experiment_id}[/bold green]")
        
        # Launch experiment monitoring task
        asyncio.create_task(self._monitor_experiment(experiment_id))
        
        # Launch experiment execution task  
        asyncio.create_task(self._execute_experiment(experiment_id, resources))
        
        return True
    
    async def _execute_experiment(self, experiment_id: str, resources: Dict[str, Any]):
        """Execute experiment with real-time monitoring and adaptation."""
        
        experiment = self.active_experiments[experiment_id]
        config = experiment.config
        
        try:
            experiment.status = ExperimentStatus.RUNNING
            
            # Simulate experiment execution with periodic metrics updates
            total_iterations = config.adaptive_params['max_iterations']
            batch_size = config.adaptive_params['batch_size']
            learning_rate = config.adaptive_params['learning_rate']
            
            best_accuracy = 0.0
            stagnation_count = 0
            
            for iteration in range(total_iterations):
                # Simulate training iteration
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Generate realistic metrics
                progress = iteration / total_iterations
                base_accuracy = min(0.95, 0.5 + 0.4 * progress + 0.1 * (1 - progress) * (iteration % 10) / 10)
                noise = 0.02 * (1 - progress) * (1 if iteration % 2 == 0 else -1)
                current_accuracy = max(0.0, min(1.0, base_accuracy + noise))
                
                # Update metrics
                experiment.metrics.update_metric('accuracy', current_accuracy)
                experiment.metrics.update_metric('loss', max(0.01, 2.0 * (1 - current_accuracy)))
                experiment.metrics.update_metric('convergence_rate', (current_accuracy - best_accuracy) if current_accuracy > best_accuracy else 0)
                experiment.metrics.update_metric('resource_utilization', 0.7 + 0.3 * progress)
                experiment.metrics.update_metric('execution_time', (datetime.now() - experiment.start_time).total_seconds())
                experiment.metrics.update_metric('cost_efficiency', current_accuracy / (progress + 0.1))
                experiment.metrics.update_metric('novelty_score', 0.8 + 0.2 * progress)
                
                experiment.last_update = datetime.now()
                
                # Check for improvement
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Adaptive optimization every N iterations
                if iteration % 50 == 0 and config.adaptation_enabled:
                    await self._adapt_experiment_parameters(experiment_id)
                
                # Early stopping check
                if (config.early_stopping_patience > 0 and 
                    stagnation_count >= config.early_stopping_patience and
                    current_accuracy >= config.performance_threshold):
                    
                    self.logger.info(f"Early stopping triggered for {experiment_id} at iteration {iteration}")
                    break
                
                # Check if experiment should be paused or stopped
                if experiment.status != ExperimentStatus.RUNNING:
                    break
            
            # Mark as completed
            experiment.status = ExperimentStatus.COMPLETED
            self.completed_experiments[experiment_id] = experiment
            del self.active_experiments[experiment_id]
            
            # Update system metrics
            self.system_metrics['successful_experiments'] += 1
            self._update_system_metrics()
            
            self.console.print(f"[bold green]✅ Experiment completed: {experiment_id}[/bold green]")
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            self.failed_experiments[experiment_id] = experiment
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            self.system_metrics['failed_experiments'] += 1
            self._update_system_metrics()
            
        finally:
            # Release resources
            self.resource_allocator.release_resources(experiment_id)
    
    async def _monitor_experiment(self, experiment_id: str):
        """Monitor experiment progress and health."""
        
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return
            
        self.logger.info(f"Starting monitoring for experiment: {experiment_id}")
        
        while (experiment_id in self.active_experiments and 
               experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.OPTIMIZING]):
            
            await asyncio.sleep(5)  # Monitor every 5 seconds
            
            experiment = self.active_experiments.get(experiment_id)
            if not experiment:
                break
                
            # Check for anomalies
            self._detect_experiment_anomalies(experiment)
            
            # Check for optimization opportunities
            if experiment.config.adaptation_enabled:
                await self._check_optimization_opportunities(experiment_id)
    
    async def _adapt_experiment_parameters(self, experiment_id: str):
        """Adaptively adjust experiment parameters based on current performance."""
        
        experiment = self.active_experiments.get(experiment_id)
        if not experiment or not experiment.config.adaptation_enabled:
            return
            
        experiment.status = ExperimentStatus.OPTIMIZING
        
        # Get optimization recommendations
        recommendations = self.optimizer.get_parameter_recommendations(experiment)
        
        if recommendations:
            self.logger.info(f"Adapting parameters for {experiment_id}: {recommendations}")
            
            # Apply adaptations
            for param, new_value in recommendations.items():
                if param in experiment.config.adaptive_params:
                    old_value = experiment.config.adaptive_params[param]
                    experiment.config.adaptive_params[param] = new_value
                    
                    # Log adaptation
                    experiment.optimization_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'parameter': param,
                        'old_value': old_value,
                        'new_value': new_value,
                        'accuracy_before': experiment.metrics.accuracy,
                        'reason': f'Optimization based on {experiment.config.optimization_targets}'
                    })
            
            experiment.adaptation_count += 1
            
        experiment.status = ExperimentStatus.RUNNING
    
    async def _check_optimization_opportunities(self, experiment_id: str):
        """Check if experiment could benefit from parameter optimization."""
        
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return
            
        # Check accuracy trend
        accuracy_trend = experiment.metrics.get_trend('accuracy', window=10)
        
        # If accuracy is stagnating, consider optimization
        if (accuracy_trend == "stable" and 
            experiment.metrics.accuracy < experiment.config.performance_threshold and
            len(experiment.metrics.history.get('accuracy', [])) > 20):
            
            await self._adapt_experiment_parameters(experiment_id)
    
    def _detect_experiment_anomalies(self, experiment: ActiveExperiment):
        """Detect potential issues in experiment execution."""
        
        # Check for memory issues
        if experiment.metrics.memory_usage > 0.9:
            self.logger.warning(f"High memory usage detected in {experiment.experiment_id}")
        
        # Check for performance degradation
        loss_trend = experiment.metrics.get_trend('loss')
        if loss_trend == "degrading":
            self.logger.warning(f"Loss increasing in {experiment.experiment_id}")
        
        # Check for resource underutilization
        if experiment.metrics.resource_utilization < 0.3:
            self.logger.info(f"Low resource utilization in {experiment.experiment_id} - optimization opportunity")
    
    def _update_system_metrics(self):
        """Update global system performance metrics."""
        
        total_experiments = self.system_metrics['total_experiments']
        if total_experiments > 0:
            success_rate = self.system_metrics['successful_experiments'] / total_experiments
            
            # Calculate average execution time from completed experiments
            if self.completed_experiments:
                execution_times = []
                for exp in self.completed_experiments.values():
                    if exp.start_time:
                        execution_times.append(exp.metrics.execution_time)
                
                if execution_times:
                    self.system_metrics['average_execution_time'] = statistics.mean(execution_times)
            
            # Calculate adaptation success rate
            total_adaptations = sum(exp.adaptation_count for exp in self.completed_experiments.values())
            if total_adaptations > 0:
                # Simplified success rate based on completed experiments
                self.system_metrics['adaptation_success_rate'] = success_rate
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed status of an experiment."""
        
        # Check all experiment collections
        experiment = (self.active_experiments.get(experiment_id) or 
                     self.completed_experiments.get(experiment_id) or
                     self.failed_experiments.get(experiment_id))
        
        if not experiment:
            return {'error': f'Experiment {experiment_id} not found'}
        
        status_info = {
            'experiment_id': experiment_id,
            'status': experiment.status.value,
            'current_metrics': {
                'accuracy': experiment.metrics.accuracy,
                'loss': experiment.metrics.loss,
                'convergence_rate': experiment.metrics.convergence_rate,
                'resource_utilization': experiment.metrics.resource_utilization,
                'execution_time': experiment.metrics.execution_time,
                'cost_efficiency': experiment.metrics.cost_efficiency,
                'novelty_score': experiment.metrics.novelty_score
            },
            'adaptation_count': experiment.adaptation_count,
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'last_update': experiment.last_update.isoformat() if experiment.last_update else None
        }
        
        # Add trends for active experiments
        if experiment_id in self.active_experiments:
            status_info['trends'] = {
                'accuracy': experiment.metrics.get_trend('accuracy'),
                'loss': experiment.metrics.get_trend('loss'), 
                'resource_utilization': experiment.metrics.get_trend('resource_utilization')
            }
        
        return status_info
    
    def list_experiments(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all experiments grouped by status."""
        
        result = {
            'active': [],
            'completed': [],
            'failed': []
        }
        
        for exp_id in self.active_experiments:
            result['active'].append(self.get_experiment_status(exp_id))
            
        for exp_id in self.completed_experiments:
            result['completed'].append(self.get_experiment_status(exp_id))
            
        for exp_id in self.failed_experiments:
            result['failed'].append(self.get_experiment_status(exp_id))
        
        return result
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        return self.system_metrics.copy()
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment."""
        
        if experiment_id not in self.active_experiments:
            return False
            
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.FAILED  # Will cause execution loop to exit
        
        self.logger.info(f"Stopping experiment: {experiment_id}")
        return True
    
    async def pause_experiment(self, experiment_id: str) -> bool:
        """Pause a running experiment."""
        
        if experiment_id not in self.active_experiments:
            return False
            
        experiment = self.active_experiments[experiment_id]
        if experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.PAUSED
            self.logger.info(f"Pausing experiment: {experiment_id}")
            return True
            
        return False
    
    async def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        
        if experiment_id not in self.active_experiments:
            return False
            
        experiment = self.active_experiments[experiment_id]
        if experiment.status == ExperimentStatus.PAUSED:
            experiment.status = ExperimentStatus.RUNNING
            self.logger.info(f"Resuming experiment: {experiment_id}")
            return True
            
        return False
    
    def create_monitoring_dashboard(self) -> Table:
        """Create a rich table showing current experiment status."""
        
        table = Table(title="🧪 Adaptive Experiment Manager - Live Dashboard")
        
        table.add_column("Experiment ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Accuracy", style="green")
        table.add_column("Loss", style="red")
        table.add_column("Resource Util", style="blue")
        table.add_column("Adaptations", style="yellow")
        table.add_column("Runtime", style="white")
        
        for exp_id, experiment in self.active_experiments.items():
            runtime = ""
            if experiment.start_time:
                runtime = str(datetime.now() - experiment.start_time).split('.')[0]
            
            table.add_row(
                exp_id[:12],
                experiment.status.value,
                f"{experiment.metrics.accuracy:.3f}",
                f"{experiment.metrics.loss:.3f}",
                f"{experiment.metrics.resource_utilization:.1%}",
                str(experiment.adaptation_count),
                runtime
            )
        
        return table


class ResourceAllocator:
    """Manages resource allocation for experiments."""
    
    def __init__(self):
        self.allocated_resources: Dict[str, Dict[str, Any]] = {}
        self.total_resources = {
            'cpu_cores': 8,
            'memory_gb': 32,
            'gpu_memory_gb': 12,
            'storage_gb': 100
        }
        self.available_resources = self.total_resources.copy()
    
    def allocate_resources(self, config: ExperimentConfiguration) -> Optional[Dict[str, Any]]:
        """Allocate resources for an experiment."""
        
        # Calculate resource requirements based on experiment configuration
        required = self._calculate_resource_requirements(config)
        
        # Check availability
        if self._can_allocate(required):
            # Allocate resources
            for resource, amount in required.items():
                self.available_resources[resource] -= amount
                
            self.allocated_resources[config.experiment_id] = required
            return required
        
        return None
    
    def release_resources(self, experiment_id: str):
        """Release resources allocated to an experiment."""
        
        if experiment_id in self.allocated_resources:
            allocated = self.allocated_resources[experiment_id]
            
            for resource, amount in allocated.items():
                self.available_resources[resource] += amount
                
            del self.allocated_resources[experiment_id]
    
    def _calculate_resource_requirements(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """Calculate resource requirements for an experiment."""
        
        base_config = config.base_config
        
        # Estimate requirements based on model size, data size, etc.
        requirements = {
            'cpu_cores': 2,  # Default CPU allocation
            'memory_gb': 4,  # Default memory allocation
            'gpu_memory_gb': 2,  # Default GPU memory
            'storage_gb': 5   # Default storage
        }
        
        # Adjust based on configuration
        if 'model_size' in base_config:
            model_size = base_config['model_size']
            if model_size == 'large':
                requirements['memory_gb'] = 8
                requirements['gpu_memory_gb'] = 6
            elif model_size == 'xlarge':
                requirements['memory_gb'] = 16
                requirements['gpu_memory_gb'] = 10
        
        return requirements
    
    def _can_allocate(self, required: Dict[str, Any]) -> bool:
        """Check if required resources can be allocated."""
        
        for resource, amount in required.items():
            if self.available_resources.get(resource, 0) < amount:
                return False
        return True


class ExperimentOptimizer:
    """Provides intelligent parameter optimization recommendations."""
    
    def get_parameter_recommendations(self, experiment: ActiveExperiment) -> Dict[str, Any]:
        """Generate parameter optimization recommendations."""
        
        recommendations = {}
        config = experiment.config
        metrics = experiment.metrics
        
        # Learning rate optimization
        if 'learning_rate' in config.adaptive_params:
            accuracy_trend = metrics.get_trend('accuracy')
            loss_trend = metrics.get_trend('loss')
            
            current_lr = config.adaptive_params['learning_rate']
            
            if accuracy_trend == "stable" and loss_trend == "stable":
                # Increase learning rate slightly to escape plateau
                recommendations['learning_rate'] = min(0.1, current_lr * 1.2)
            elif loss_trend == "degrading":
                # Decrease learning rate to stabilize
                recommendations['learning_rate'] = max(0.0001, current_lr * 0.8)
        
        # Batch size optimization
        if 'batch_size' in config.adaptive_params:
            resource_util = metrics.resource_utilization
            
            current_batch_size = config.adaptive_params['batch_size']
            
            if resource_util < 0.5 and current_batch_size < 128:
                # Increase batch size to better utilize resources
                recommendations['batch_size'] = min(128, current_batch_size * 2)
            elif resource_util > 0.9 and current_batch_size > 8:
                # Decrease batch size to reduce memory pressure
                recommendations['batch_size'] = max(8, current_batch_size // 2)
        
        return recommendations


# Demo and testing functions
async def demo_adaptive_experiment_manager():
    """Demonstrate the adaptive experiment manager."""
    
    console = Console()
    console.print("[bold blue]🧪 Adaptive Experiment Manager - Generation 1 Demo[/bold blue]")
    
    # Initialize manager
    manager = AdaptiveExperimentManager(max_concurrent_experiments=3)
    
    # Create multiple experiments
    experiments = []
    for i in range(5):
        exp_id = f"quantum_nn_experiment_{i}"
        
        base_config = {
            'model_type': 'quantum_neural_network',
            'model_size': 'medium' if i < 3 else 'large',
            'learning_rate': 0.001 + i * 0.0005,
            'batch_size': 32 + i * 16,
            'max_iterations': 200 + i * 50,
            'dataset': f'quantum_dataset_{i}'
        }
        
        exp_id = manager.create_experiment(
            exp_id, 
            base_config,
            optimization_targets=['accuracy', 'cost_efficiency', 'novelty_score']
        )
        experiments.append(exp_id)
        
        console.print(f"[green]✅ Created experiment: {exp_id}[/green]")
    
    # Start experiments
    console.print(f"\n[bold yellow]🚀 Starting {len(experiments)} experiments...[/bold yellow]")
    
    start_tasks = []
    for exp_id in experiments:
        task = asyncio.create_task(manager.start_experiment(exp_id))
        start_tasks.append(task)
    
    # Start monitoring dashboard
    dashboard_task = asyncio.create_task(run_monitoring_dashboard(manager, duration=30))
    
    # Wait for experiments to complete or timeout
    await asyncio.gather(*start_tasks)
    await dashboard_task
    
    # Show final results
    console.print(f"\n[bold green]🎉 Experiment Session Complete![/bold green]")
    
    experiment_list = manager.list_experiments()
    console.print(f"• Active: {len(experiment_list['active'])}")
    console.print(f"• Completed: {len(experiment_list['completed'])}")
    console.print(f"• Failed: {len(experiment_list['failed'])}")
    
    system_metrics = manager.get_system_metrics()
    console.print(f"\n[bold cyan]📊 System Performance:[/bold cyan]")
    console.print(f"• Success Rate: {system_metrics['successful_experiments']}/{system_metrics['total_experiments']}")
    console.print(f"• Avg Execution Time: {system_metrics['average_execution_time']:.1f}s")
    console.print(f"• Adaptation Success Rate: {system_metrics['adaptation_success_rate']:.1%}")
    
    return manager


async def run_monitoring_dashboard(manager: AdaptiveExperimentManager, duration: int = 60):
    """Run live monitoring dashboard for specified duration."""
    
    with Live(manager.create_monitoring_dashboard(), refresh_per_second=2) as live:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            await asyncio.sleep(0.5)
            
            # Update dashboard
            table = manager.create_monitoring_dashboard()
            live.update(table)
            
            # Exit if no active experiments
            if not manager.active_experiments:
                break


async def main():
    """Main entry point for the adaptive experiment manager."""
    
    try:
        manager = await demo_adaptive_experiment_manager()
        
        # Additional status check
        console = Console()
        console.print(f"\n[bold blue]📋 Final Experiment Status:[/bold blue]")
        
        experiments = manager.list_experiments()
        for status, exp_list in experiments.items():
            if exp_list:
                console.print(f"\n[bold]{status.upper()}:[/bold]")
                for exp in exp_list:
                    console.print(f"  • {exp['experiment_id']}: {exp['current_metrics']['accuracy']:.3f} accuracy")
    
    except Exception as e:
        console = Console()
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())