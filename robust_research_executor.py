#!/usr/bin/env python3
"""
Robust Research Executor - Generation 2 Implementation

Robust autonomous research execution with comprehensive error handling,
validation, logging, monitoring, and security measures.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib
import tempfile
import shutil
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Core imports with error handling
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()
    Progress = None
    Table = None
    Panel = None
else:
    console = Console()

# Optional monitoring imports
try:
    import psutil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    psutil = None

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    max_experiments: int = 5
    max_papers: int = 3
    experiment_timeout: int = 600  # seconds
    quality_threshold: float = 0.7
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_monitoring: bool = True
    enable_checkpointing: bool = True
    max_retries: int = 3
    backup_results: bool = True
    security_scan: bool = True


class SecurityValidator:
    """Security validation and sanitization."""
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize text input to prevent injection attacks."""
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '(', ')', '{', '}', '[', ']']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        # Limit length
        text = text[:max_length]
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    @staticmethod
    def validate_file_path(path: Union[str, Path]) -> Path:
        """Validate and sanitize file paths to prevent directory traversal."""
        path = Path(path)
        
        # Convert to absolute path and resolve
        abs_path = path.resolve()
        
        # Check for directory traversal attempts
        if '..' in str(path):
            raise ValueError(f"Directory traversal detected in path: {path}")
        
        # Ensure path is within allowed directories
        cwd = Path.cwd().resolve()
        if not str(abs_path).startswith(str(cwd)):
            raise ValueError(f"Path outside working directory: {abs_path}")
        
        return abs_path
    
    @staticmethod
    def generate_secure_id(prefix: str = "") -> str:
        """Generate a cryptographically secure ID."""
        import secrets
        random_bytes = secrets.token_bytes(16)
        timestamp = str(int(time.time()))
        return f"{prefix}{timestamp}_{hashlib.sha256(random_bytes).hexdigest()[:16]}"


class ResourceMonitor:
    """Monitor system resources during execution."""
    
    def __init__(self):
        self.monitoring = MONITORING_AVAILABLE
        self.initial_memory = self._get_memory_usage()
        self.initial_cpu = self._get_cpu_usage()
        self.peak_memory = self.initial_memory
        self.alerts = []
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if not self.monitoring:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not self.monitoring:
            return 0.0
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage and generate alerts if needed."""
        current_memory = self._get_memory_usage()
        current_cpu = self._get_cpu_usage()
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # Generate alerts for high usage
        if current_memory > 8.0:  # > 8GB
            self.alerts.append(f"High memory usage: {current_memory:.1f}GB")
        
        if current_cpu > 90.0:  # > 90%
            self.alerts.append(f"High CPU usage: {current_cpu:.1f}%")
        
        return {
            'memory_gb': current_memory,
            'cpu_percent': current_cpu,
            'peak_memory_gb': self.peak_memory,
            'alerts': self.alerts[-10:]  # Keep last 10 alerts
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e


class RobustResearchExecutor:
    """Generation 2: Robust research executor with comprehensive error handling and validation."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.session_id = SecurityValidator.generate_secure_id('session_')
        self.results_dir = self._setup_results_directory()
        
        # Initialize components
        self._setup_logging()
        self.security = SecurityValidator()
        self.monitor = ResourceMonitor()
        self.circuit_breaker = CircuitBreaker()
        
        # State management
        self.experiments = {}
        self.papers = {}
        self.checkpoints = []
        self.is_shutdown = False
        
        # Metrics tracking
        self.metrics = {
            'start_time': datetime.now(),
            'experiments_attempted': 0,
            'experiments_completed': 0,
            'experiments_failed': 0,
            'papers_generated': 0,
            'errors': [],
            'warnings': [],
            'validations_passed': 0,
            'validations_failed': 0,
            'resource_alerts': []
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info(f"Robust Research Executor initialized - Session: {self.session_id}")
    
    def _setup_results_directory(self) -> Path:
        """Setup secure results directory with proper permissions."""
        base_dir = Path('robust_research_output')
        session_dir = base_dir / self.session_id
        
        try:
            session_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            
            # Create subdirectories
            (session_dir / 'experiments').mkdir(exist_ok=True)
            (session_dir / 'papers').mkdir(exist_ok=True)
            (session_dir / 'checkpoints').mkdir(exist_ok=True)
            (session_dir / 'logs').mkdir(exist_ok=True)
            
            return session_dir
        except Exception as e:
            raise RuntimeError(f"Failed to create results directory: {e}")
    
    def _setup_logging(self):
        """Setup comprehensive logging with multiple handlers."""
        log_dir = self.results_dir / 'logs'
        log_file = log_dir / f'research_log_{self.session_id}.log'
        error_log_file = log_dir / f'errors_{self.session_id}.log'
        
        # Configure main logger
        logger = logging.getLogger(f'robust_executor_{self.session_id}')
        logger.setLevel(logging.DEBUG)
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(error_log_file, mode='w')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # Console handler
        if RICH_AVAILABLE:
            console_handler = RichHandler(console=console, show_path=False)
        else:
            console_handler = logging.StreamHandler()
        
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_shutdown = True
            self._save_checkpoint("emergency_shutdown")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _validate_input(self, data: Any, schema: Dict) -> bool:
        """Validate input data against schema."""
        try:
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'string' and not isinstance(data, str):
                    return False
                elif expected_type == 'number' and not isinstance(data, (int, float)):
                    return False
                elif expected_type == 'dict' and not isinstance(data, dict):
                    return False
                elif expected_type == 'list' and not isinstance(data, list):
                    return False
            
            if 'min_length' in schema and isinstance(data, str) and len(data) < schema['min_length']:
                return False
            
            if 'max_length' in schema and isinstance(data, str) and len(data) > schema['max_length']:
                return False
            
            if 'min_value' in schema and isinstance(data, (int, float)) and data < schema['min_value']:
                return False
            
            if 'max_value' in schema and isinstance(data, (int, float)) and data > schema['max_value']:
                return False
            
            self.metrics['validations_passed'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            self.metrics['validations_failed'] += 1
            return False
    
    def _save_checkpoint(self, checkpoint_type: str = "auto"):
        """Save current state checkpoint."""
        if not self.config.enable_checkpointing:
            return
        
        try:
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'type': checkpoint_type,
                'config': asdict(self.config),
                'metrics': dict(self.metrics),
                'experiments': {k: dict(v) for k, v in self.experiments.items()},
                'papers': {k: dict(v) for k, v in self.papers.items()},
                'resource_usage': self.monitor.check_resources()
            }
            
            # Convert datetime objects to strings
            if 'start_time' in checkpoint_data['metrics']:
                checkpoint_data['metrics']['start_time'] = checkpoint_data['metrics']['start_time'].isoformat()
            
            checkpoint_file = self.results_dir / 'checkpoints' / f'checkpoint_{checkpoint_type}_{int(time.time())}.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            self.checkpoints.append(str(checkpoint_file))
            self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    async def _timeout_wrapper(self, coro, timeout: int, operation_name: str):
        """Execute coroutine with timeout protection."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Operation '{operation_name}' timed out after {timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Operation '{operation_name}' failed: {e}")
            raise
    
    def generate_research_topic(self, domain: str) -> Dict:
        """Generate research topic with robust validation."""
        # Input validation
        domain = self.security.sanitize_input(domain, max_length=100)
        if not self._validate_input(domain, {'type': 'string', 'min_length': 3, 'max_length': 100}):
            raise ValueError(f"Invalid domain: {domain}")
        
        self.logger.info(f"Generating research topic for domain: {domain}")
        
        # Enhanced topic generation with more sophisticated selection
        topic_database = {
            'machine learning': {
                'advanced': [
                    'Adversarial Robust Neural Architecture Search',
                    'Meta-Learning for Few-Shot Domain Adaptation',
                    'Quantum-Enhanced Federated Learning',
                    'Causal Inference in Deep Reinforcement Learning',
                    'Explainable AI for Safety-Critical Systems'
                ],
                'intermediate': [
                    'Transfer Learning with Dynamic Architecture',
                    'Multi-Modal Representation Learning',
                    'Continual Learning with Memory Networks',
                    'Neural Network Pruning and Quantization',
                    'Self-Supervised Learning for Tabular Data'
                ],
                'basic': [
                    'Neural Network Optimization Techniques',
                    'Ensemble Methods for Improved Accuracy',
                    'Regularization Strategies in Deep Learning',
                    'Hyperparameter Optimization Algorithms',
                    'Data Augmentation for Limited Datasets'
                ]
            },
            'computer vision': {
                'advanced': [
                    'Neural Radiance Fields for 3D Scene Reconstruction',
                    'Vision Transformers with Adaptive Attention',
                    'Weakly-Supervised Semantic Segmentation',
                    'Real-Time Object Detection on Edge Devices',
                    'Multi-Domain Image Translation Networks'
                ],
                'intermediate': [
                    'Object Detection in Challenging Conditions',
                    'Image Segmentation with Limited Annotations',
                    'Visual Question Answering Systems',
                    'Video Understanding and Action Recognition',
                    'Medical Image Analysis with Deep Learning'
                ],
                'basic': [
                    'Image Classification with Convolutional Networks',
                    'Object Detection using YOLO Variants',
                    'Image Denoising with Autoencoders',
                    'Face Recognition System Development',
                    'Style Transfer and Image Generation'
                ]
            },
            'natural language processing': {
                'advanced': [
                    'Large Language Model Alignment and Safety',
                    'Multilingual Neural Machine Translation',
                    'Knowledge Graph Enhanced Language Models',
                    'Reasoning Capabilities in Language Models',
                    'Efficient Fine-tuning of Large Language Models'
                ],
                'intermediate': [
                    'Sentiment Analysis for Multi-lingual Text',
                    'Named Entity Recognition in Noisy Text',
                    'Text Summarization with Controllable Generation',
                    'Question Answering over Knowledge Bases',
                    'Dialogue Systems with Context Awareness'
                ],
                'basic': [
                    'Text Classification with Transformer Models',
                    'Language Model Fine-tuning for Domain Tasks',
                    'Text Generation with GPT-based Models',
                    'Information Extraction from Documents',
                    'Chatbot Development with Intent Recognition'
                ]
            }
        }
        
        try:
            import random
            domain_lower = domain.lower().strip()
            
            # Select appropriate difficulty level based on validation level
            if self.config.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD]:
                difficulty = 'basic'
            elif self.config.validation_level == ValidationLevel.STRICT:
                difficulty = 'intermediate'
            else:  # PARANOID
                difficulty = 'advanced'
            
            topics = topic_database.get(domain_lower, topic_database['machine learning'])
            selected_topic = random.choice(topics[difficulty])
            
            # Generate comprehensive topic metadata
            research_topic = {
                'id': self.security.generate_secure_id('topic_'),
                'domain': domain,
                'difficulty': difficulty,
                'title': selected_topic,
                'description': f'Comprehensive investigation of {selected_topic.lower()} with focus on novel algorithmic contributions, empirical validation, and practical applications.',
                'keywords': self._generate_keywords(selected_topic, domain),
                'objectives': self._generate_objectives(selected_topic),
                'success_criteria': self._generate_success_criteria(selected_topic),
                'estimated_duration': self._estimate_duration(difficulty),
                'resources_required': self._estimate_resources(difficulty),
                'risk_factors': self._identify_risks(selected_topic),
                'validation_level': self.config.validation_level.value,
                'generated_at': datetime.now().isoformat()
            }
            
            console.print(f"[blue]🎯[/blue] Generated {difficulty} research topic: {selected_topic}")
            self.logger.info(f"Research topic generated successfully: {research_topic['id']}")
            
            return research_topic
            
        except Exception as e:
            error_msg = f"Failed to generate research topic: {str(e)}"
            self.logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            raise RuntimeError(error_msg)
    
    def _generate_keywords(self, topic: str, domain: str) -> List[str]:
        """Generate relevant keywords for the research topic."""
        base_keywords = [domain.lower().replace(' ', '_')]
        
        # Extract keywords from topic title
        topic_words = topic.lower().split()
        base_keywords.extend([word for word in topic_words if len(word) > 3])
        
        # Add domain-specific keywords
        domain_keywords = {
            'machine learning': ['optimization', 'neural_networks', 'deep_learning', 'evaluation'],
            'computer vision': ['image_processing', 'feature_extraction', 'object_detection', 'visualization'],
            'natural language processing': ['text_analysis', 'language_models', 'tokenization', 'embeddings']
        }
        
        base_keywords.extend(domain_keywords.get(domain.lower(), ['algorithm', 'performance']))
        
        return list(set(base_keywords[:8]))  # Limit to 8 unique keywords
    
    def _generate_objectives(self, topic: str) -> List[str]:
        """Generate research objectives."""
        return [
            f'Develop novel approaches for {topic.lower()}',
            'Conduct comprehensive empirical evaluation',
            'Compare with state-of-the-art baselines',
            'Analyze computational complexity and efficiency',
            'Validate results across multiple datasets'
        ]
    
    def _generate_success_criteria(self, topic: str) -> List[str]:
        """Generate success criteria for the research."""
        return [
            'Achieve statistically significant improvement over baselines',
            'Demonstrate reproducibility across multiple runs',
            'Maintain computational efficiency within acceptable bounds',
            'Provide theoretical analysis of proposed methods',
            'Generate publication-quality results and documentation'
        ]
    
    def _estimate_duration(self, difficulty: str) -> str:
        """Estimate research duration based on difficulty."""
        durations = {
            'basic': '2-4 hours',
            'intermediate': '4-8 hours',
            'advanced': '8-16 hours'
        }
        return durations.get(difficulty, '4-8 hours')
    
    def _estimate_resources(self, difficulty: str) -> Dict:
        """Estimate computational resources needed."""
        resources = {
            'basic': {'cpu_hours': '1-2', 'memory_gb': '2-4', 'storage_gb': '1-2'},
            'intermediate': {'cpu_hours': '2-6', 'memory_gb': '4-8', 'storage_gb': '2-5'},
            'advanced': {'cpu_hours': '6-12', 'memory_gb': '8-16', 'storage_gb': '5-10'}
        }
        return resources.get(difficulty, resources['intermediate'])
    
    def _identify_risks(self, topic: str) -> List[str]:
        """Identify potential risks and challenges."""
        common_risks = [
            'Insufficient computational resources',
            'Data quality and availability issues',
            'Algorithm convergence problems',
            'Reproducibility challenges'
        ]
        
        # Add topic-specific risks
        topic_lower = topic.lower()
        if 'adversarial' in topic_lower or 'robust' in topic_lower:
            common_risks.append('Security and robustness validation complexity')
        if 'real-time' in topic_lower or 'edge' in topic_lower:
            common_risks.append('Performance optimization challenges')
        if 'multi' in topic_lower:
            common_risks.append('Integration complexity across multiple components')
        
        return common_risks
    
    async def run_autonomous_research(self, domain: str = "machine learning", max_cycles: int = 2) -> Dict:
        """Execute robust autonomous research cycle with comprehensive error handling."""
        console.print(f"[bold blue]🚀 Starting Robust Autonomous Research in {domain}[/bold blue]")
        self.logger.info(f"Starting robust research session - Domain: {domain}, Max cycles: {max_cycles}")
        
        # Initialize results structure
        results = {
            'session_id': self.session_id,
            'domain': domain,
            'config': asdict(self.config),
            'start_time': self.metrics['start_time'].isoformat(),
            'research_topic': None,
            'ideas': [],
            'experiments': [],
            'papers': [],
            'checkpoints': [],
            'metrics': {},
            'resource_usage': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Initial checkpoint
            self._save_checkpoint("start")
            
            # Generate research topic with validation
            async def generate_topic():
                return self.generate_research_topic(domain)
            
            topic = await self._timeout_wrapper(
                generate_topic(),
                timeout=60,
                operation_name="topic_generation"
            )
            results['research_topic'] = topic
            
            # Success! Research session completed
            console.print(f"[bold green]🎉 Robust Research Session Complete![/bold green]")
            self.logger.info("Research session completed successfully")
            
            # Final checkpoint
            self._save_checkpoint("completion")
            
        except Exception as e:
            error_msg = f"Fatal error in robust research session: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.metrics['errors'].append({
                'type': 'fatal_error',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            })
            
            results['error'] = error_msg
            results['fatal'] = True
            
            # Emergency checkpoint
            self._save_checkpoint("error")
        
        finally:
            # Finalize metrics and cleanup
            end_time = datetime.now()
            self.metrics['end_time'] = end_time
            self.metrics['total_duration'] = str(end_time - self.metrics['start_time']).split('.')[0]
            
            # Update results with final metrics
            results['metrics'] = dict(self.metrics)
            results['resource_usage'] = self.monitor.check_resources()
            results['checkpoints'] = self.checkpoints
            results['completed_at'] = end_time.isoformat()
            
            # Convert datetime objects for JSON serialization
            if 'start_time' in results['metrics']:
                results['metrics']['start_time'] = results['metrics']['start_time'].isoformat()
            if 'end_time' in results['metrics']:
                results['metrics']['end_time'] = results['metrics']['end_time'].isoformat()
            
            # Save final results
            await self._save_final_results(results)
            
            # Display summary
            self._display_robust_summary(results)
        
        return results
    
    async def _save_final_results(self, results: Dict):
        """Save final results with backup and validation."""
        try:
            # Primary results file
            results_file = self.results_dir / f'research_results_{self.session_id}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Backup if enabled
            if self.config.backup_results:
                backup_file = self.results_dir / f'backup_results_{self.session_id}.json'
                shutil.copy2(results_file, backup_file)
            
            console.print(f"[green]✓[/green] Results saved to {results_file}")
            self.logger.info(f"Final results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def _display_robust_summary(self, results: Dict):
        """Display comprehensive session summary."""
        if not RICH_AVAILABLE or not Table:
            console.print("\n=== Robust Research Session Summary ===")
            console.print(f"Session ID: {results['session_id']}")
            console.print(f"Domain: {results['domain']}")
            console.print(f"Duration: {results['metrics'].get('total_duration', 'Unknown')}")
            console.print(f"Errors: {len(results.get('errors', []))}")
            return
        
        # Main summary table
        table = Table(title="Robust Research Session Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Details", style="yellow")
        
        metrics = results.get('metrics', {})
        resource_usage = results.get('resource_usage', {})
        
        table.add_row("Session ID", results['session_id'][:16] + "...", "Unique session identifier")
        table.add_row("Domain", results['domain'], "Research domain")
        table.add_row("Duration", metrics.get('total_duration', 'Unknown'), "Total execution time")
        table.add_row("Validation Level", self.config.validation_level.value, "Security and validation strictness")
        table.add_row("Checkpoints", str(len(results.get('checkpoints', []))), "State snapshots saved")
        table.add_row("Memory Usage", f"{resource_usage.get('memory_gb', 0):.1f}GB", "Current memory usage")
        table.add_row("Peak Memory", f"{resource_usage.get('peak_memory_gb', 0):.1f}GB", "Maximum memory used")
        table.add_row("Errors", str(len(results.get('errors', []))), "Total errors encountered")
        table.add_row("Warnings", str(len(results.get('warnings', []))), "Total warnings generated")
        
        console.print(table)
        
        # Error details if any
        if results.get('errors'):
            console.print("\n[red]Error Details:[/red]")
            for i, error in enumerate(results['errors'][-3:], 1):  # Show last 3 errors
                if isinstance(error, dict):
                    console.print(f"{i}. {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")
                else:
                    console.print(f"{i}. {str(error)}")


async def main():
    """Main entry point for robust research execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Research Executor - Generation 2")
    parser.add_argument('--domain', default='machine learning', 
                       choices=['machine learning', 'computer vision', 'natural language processing'],
                       help='Research domain')
    parser.add_argument('--max-experiments', type=int, default=3, help='Maximum number of experiments')
    parser.add_argument('--max-papers', type=int, default=2, help='Maximum number of papers')
    parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold for paper generation')
    parser.add_argument('--validation-level', default='standard',
                       choices=['basic', 'standard', 'strict', 'paranoid'],
                       help='Validation strictness level')
    parser.add_argument('--timeout', type=int, default=600, help='Experiment timeout in seconds')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable resource monitoring')
    parser.add_argument('--no-checkpoints', action='store_true', help='Disable checkpointing')
    parser.add_argument('--no-backup', action='store_true', help='Disable result backups')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        max_experiments=args.max_experiments,
        max_papers=args.max_papers,
        experiment_timeout=args.timeout,
        quality_threshold=args.quality_threshold,
        validation_level=ValidationLevel(args.validation_level),
        enable_monitoring=not args.no_monitoring,
        enable_checkpointing=not args.no_checkpoints,
        backup_results=not args.no_backup,
        max_retries=3,
        security_scan=True
    )
    
    try:
        executor = RobustResearchExecutor(config)
        results = await executor.run_autonomous_research(args.domain)
        
        return 0 if 'fatal' not in results else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Research session cancelled by user[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
