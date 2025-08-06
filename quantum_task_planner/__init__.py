"""
Quantum-Inspired Task Planner v1.0

A quantum-inspired task planning system that leverages quantum computing principles
for efficient task scheduling and resource optimization.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.ai"

from .core.planner import QuantumTaskPlanner
from .core.quantum_optimizer import QuantumOptimizer
from .core.task_scheduler import TaskScheduler
from .utils.metrics import PlannerMetrics

__all__ = [
    "QuantumTaskPlanner",
    "QuantumOptimizer", 
    "TaskScheduler",
    "PlannerMetrics",
]