"""
Comprehensive Validation System for Quantum Task Planner

Input validation, constraint checking, and data integrity verification
for all quantum task planning components.
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from .error_handling import ValidationError, ResourceConstraintError, DependencyError
from ..core.planner import Task, TaskPriority
from ..utils.quantum_math import QubitState

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Represents a single validation rule."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info_messages: List[str]
    field_errors: Dict[str, List[str]]


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str):
        """
        Initialize validator.
        
        Args:
            name: Validator name for logging
        """
        self.name = name
        self.rules: List[ValidationRule] = []
        self._setup_rules()
    
    @abstractmethod
    def _setup_rules(self) -> None:
        """Set up validation rules (implemented by subclasses)."""
        pass
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)
        logger.debug(f"Added validation rule '{rule.name}' to {self.name}")
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against all rules.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with detailed results
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        for rule in self.rules:
            try:
                if not rule.validator(data):
                    if rule.severity == "error":
                        result.errors.append(f"{rule.name}: {rule.error_message}")
                        result.is_valid = False
                    elif rule.severity == "warning":
                        result.warnings.append(f"{rule.name}: {rule.error_message}")
                    else:
                        result.info_messages.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                result.errors.append(f"{rule.name}: Validation error - {str(e)}")
                result.is_valid = False
                logger.error(f"Validation rule '{rule.name}' failed with exception: {e}")
        
        logger.debug(f"{self.name} validation: {'PASS' if result.is_valid else 'FAIL'}")
        return result


class TaskValidator(BaseValidator):
    """Validates Task objects and task-related data."""
    
    def __init__(self):
        super().__init__("TaskValidator")
    
    def _setup_rules(self) -> None:
        """Set up task validation rules."""
        
        # Task ID validation
        self.add_rule(ValidationRule(
            name="task_id_format",
            validator=lambda task: self._validate_task_id(task.id if hasattr(task, 'id') else None),
            error_message="Task ID must be non-empty string with valid characters"
        ))
        
        # Task name validation
        self.add_rule(ValidationRule(
            name="task_name_length",
            validator=lambda task: self._validate_task_name(task.name if hasattr(task, 'name') else None),
            error_message="Task name must be between 1 and 200 characters"
        ))
        
        # Duration validation
        self.add_rule(ValidationRule(
            name="task_duration_positive",
            validator=lambda task: self._validate_duration(task.duration if hasattr(task, 'duration') else None),
            error_message="Task duration must be positive number"
        ))
        
        # Priority validation
        self.add_rule(ValidationRule(
            name="task_priority_valid",
            validator=lambda task: self._validate_priority(task.priority if hasattr(task, 'priority') else None),
            error_message="Task priority must be valid TaskPriority enum value"
        ))
        
        # Dependencies validation
        self.add_rule(ValidationRule(
            name="dependencies_format",
            validator=lambda task: self._validate_dependencies(task.dependencies if hasattr(task, 'dependencies') else None),
            error_message="Dependencies must be list of valid task IDs"
        ))
        
        # Resources validation
        self.add_rule(ValidationRule(
            name="resources_format",
            validator=lambda task: self._validate_resources(task.resources if hasattr(task, 'resources') else None),
            error_message="Resources must be dictionary with non-negative values"
        ))
        
        # Quantum weight validation
        self.add_rule(ValidationRule(
            name="quantum_weight_range",
            validator=lambda task: self._validate_quantum_weight(task.quantum_weight if hasattr(task, 'quantum_weight') else 1.0),
            error_message="Quantum weight must be between 0.0 and 1.0",
            severity="warning"
        ))
        
        # Deadline validation
        self.add_rule(ValidationRule(
            name="deadline_consistency",
            validator=lambda task: self._validate_deadline_consistency(task),
            error_message="Deadline must be greater than task duration",
            severity="warning"
        ))
    
    def _validate_task_id(self, task_id: Any) -> bool:
        """Validate task ID format."""
        if not isinstance(task_id, str):
            return False
        if len(task_id) == 0 or len(task_id) > 100:
            return False
        # Allow alphanumeric, underscore, hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            return False
        return True
    
    def _validate_task_name(self, name: Any) -> bool:
        """Validate task name."""
        if not isinstance(name, str):
            return False
        return 1 <= len(name) <= 200
    
    def _validate_duration(self, duration: Any) -> bool:
        """Validate task duration."""
        if not isinstance(duration, (int, float)):
            return False
        return duration > 0
    
    def _validate_priority(self, priority: Any) -> bool:
        """Validate task priority."""
        if not isinstance(priority, TaskPriority):
            return False
        return priority in TaskPriority
    
    def _validate_dependencies(self, dependencies: Any) -> bool:
        """Validate task dependencies."""
        if not isinstance(dependencies, list):
            return False
        
        for dep in dependencies:
            if not self._validate_task_id(dep):
                return False
        
        # Check for duplicates
        if len(dependencies) != len(set(dependencies)):
            return False
        
        return True
    
    def _validate_resources(self, resources: Any) -> bool:
        """Validate task resources."""
        if not isinstance(resources, dict):
            return False
        
        for resource_name, amount in resources.items():
            if not isinstance(resource_name, str) or len(resource_name) == 0:
                return False
            if not isinstance(amount, (int, float)) or amount < 0:
                return False
        
        return True
    
    def _validate_quantum_weight(self, weight: Any) -> bool:
        """Validate quantum weight."""
        if not isinstance(weight, (int, float)):
            return False
        return 0.0 <= weight <= 1.0
    
    def _validate_deadline_consistency(self, task: Any) -> bool:
        """Validate deadline consistency with duration."""
        if not hasattr(task, 'deadline') or task.deadline is None:
            return True  # No deadline is okay
        
        if not hasattr(task, 'duration'):
            return True  # Can't validate without duration
        
        return task.deadline > task.duration
    
    def validate_task_list(self, tasks: List[Task]) -> ValidationResult:
        """
        Validate a list of tasks including cross-task constraints.
        
        Args:
            tasks: List of tasks to validate
            
        Returns:
            ValidationResult for the entire task list
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        # Validate individual tasks
        for i, task in enumerate(tasks):
            task_result = self.validate(task)
            if not task_result.is_valid:
                result.is_valid = False
                result.field_errors[f"task_{i}"] = task_result.errors
            result.errors.extend([f"Task {i}: {err}" for err in task_result.errors])
            result.warnings.extend([f"Task {i}: {warn}" for warn in task_result.warnings])
        
        # Cross-task validation
        task_ids = [task.id for task in tasks]
        
        # Check for duplicate task IDs
        if len(task_ids) != len(set(task_ids)):
            result.errors.append("Duplicate task IDs found")
            result.is_valid = False
        
        # Check dependency references
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    result.errors.append(f"Task {task.id} references non-existent dependency: {dep_id}")
                    result.is_valid = False
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(tasks)
        if circular_deps:
            result.errors.append(f"Circular dependencies detected: {circular_deps}")
            result.is_valid = False
        
        logger.info(f"Task list validation: {len(tasks)} tasks, {'PASS' if result.is_valid else 'FAIL'}")
        return result
    
    def _detect_circular_dependencies(self, tasks: List[Task]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        task_deps = {task.id: set(task.dependencies) for task in tasks}
        circular_deps = []
        
        def dfs(task_id: str, path: List[str], visited: set) -> None:
            if task_id in path:
                # Found cycle
                cycle_start = path.index(task_id)
                circular_deps.append(path[cycle_start:] + [task_id])
                return
            
            if task_id in visited:
                return
            
            visited.add(task_id)
            path.append(task_id)
            
            if task_id in task_deps:
                for dep in task_deps[task_id]:
                    dfs(dep, path.copy(), visited)
        
        for task in tasks:
            dfs(task.id, [], set())
        
        return circular_deps


class ResourceValidator(BaseValidator):
    """Validates resource constraints and allocations."""
    
    def __init__(self):
        super().__init__("ResourceValidator")
    
    def _setup_rules(self) -> None:
        """Set up resource validation rules."""
        
        # Resource capacity validation
        self.add_rule(ValidationRule(
            name="resource_capacity_positive",
            validator=lambda data: self._validate_capacity(data),
            error_message="Resource capacity must be positive number"
        ))
        
        # Resource name validation
        self.add_rule(ValidationRule(
            name="resource_name_format",
            validator=lambda data: self._validate_resource_name(data),
            error_message="Resource name must be non-empty alphanumeric string"
        ))
        
        # Resource utilization validation
        self.add_rule(ValidationRule(
            name="resource_utilization_bounds",
            validator=lambda data: self._validate_utilization_bounds(data),
            error_message="Resource utilization must be between 0% and 100%",
            severity="warning"
        ))
    
    def _validate_capacity(self, data: Any) -> bool:
        """Validate resource capacity."""
        if hasattr(data, 'capacity'):
            return isinstance(data.capacity, (int, float)) and data.capacity > 0
        return True  # No capacity field is okay
    
    def _validate_resource_name(self, data: Any) -> bool:
        """Validate resource name format."""
        if hasattr(data, 'name'):
            name = data.name
            if not isinstance(name, str) or len(name) == 0:
                return False
            return re.match(r'^[a-zA-Z0-9_-]+$', name) is not None
        return True
    
    def _validate_utilization_bounds(self, data: Any) -> bool:
        """Validate resource utilization is within bounds."""
        if hasattr(data, 'utilization_ratio'):
            ratio = data.utilization_ratio
            return 0.0 <= ratio <= 1.0
        return True
    
    def validate_resource_allocation(self, 
                                   tasks: List[Task], 
                                   resource_limits: Dict[str, float]) -> ValidationResult:
        """
        Validate resource allocation for task list.
        
        Args:
            tasks: List of tasks requiring resources
            resource_limits: Available resource capacities
            
        Returns:
            ValidationResult for resource allocation
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        # Calculate total resource requirements
        total_requirements = {}
        for task in tasks:
            for resource, amount in task.resources.items():
                total_requirements[resource] = total_requirements.get(resource, 0) + amount
        
        # Check against limits
        for resource, required in total_requirements.items():
            if resource in resource_limits:
                available = resource_limits[resource]
                if required > available:
                    result.errors.append(
                        f"Resource '{resource}' over-allocated: {required} required, {available} available"
                    )
                    result.is_valid = False
                elif required > available * 0.9:  # 90% utilization warning
                    result.warnings.append(
                        f"High resource utilization for '{resource}': {required/available:.1%}"
                    )
            else:
                result.warnings.append(f"Resource '{resource}' not found in limits")
        
        # Check for resource conflicts
        conflicting_tasks = self._find_resource_conflicts(tasks)
        if conflicting_tasks:
            result.warnings.extend([
                f"Tasks {task_pair} may conflict on resources" 
                for task_pair in conflicting_tasks
            ])
        
        logger.info(f"Resource allocation validation: {'PASS' if result.is_valid else 'FAIL'}")
        return result
    
    def _find_resource_conflicts(self, tasks: List[Task]) -> List[Tuple[str, str]]:
        """Find tasks that might conflict on resource usage."""
        conflicts = []
        
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                # Check if tasks share resource requirements
                shared_resources = set(task1.resources.keys()) & set(task2.resources.keys())
                if shared_resources:
                    # Check if combined usage might be problematic
                    for resource in shared_resources:
                        combined = task1.resources[resource] + task2.resources[resource]
                        # This is a simplified heuristic - in practice you'd check actual limits
                        if combined > 10.0:  # Arbitrary threshold
                            conflicts.append((task1.id, task2.id))
                            break
        
        return conflicts


class QuantumValidator(BaseValidator):
    """Validates quantum-specific states and operations."""
    
    def __init__(self):
        super().__init__("QuantumValidator")
    
    def _setup_rules(self) -> None:
        """Set up quantum validation rules."""
        
        # Quantum state normalization
        self.add_rule(ValidationRule(
            name="quantum_state_normalized",
            validator=lambda state: self._validate_normalization(state),
            error_message="Quantum state must be normalized (|α|² + |β|² = 1)"
        ))
        
        # Amplitude validation
        self.add_rule(ValidationRule(
            name="amplitude_finite",
            validator=lambda state: self._validate_amplitudes_finite(state),
            error_message="Quantum amplitudes must be finite complex numbers"
        ))
        
        # Probability validation
        self.add_rule(ValidationRule(
            name="probability_bounds",
            validator=lambda state: self._validate_probabilities(state),
            error_message="Quantum probabilities must be between 0 and 1"
        ))
        
        # Coherence validation
        self.add_rule(ValidationRule(
            name="coherence_reasonable",
            validator=lambda data: self._validate_coherence_bounds(data),
            error_message="Quantum coherence should be between 0 and 1",
            severity="warning"
        ))
    
    def _validate_normalization(self, state: Any) -> bool:
        """Validate quantum state normalization."""
        if not isinstance(state, QubitState):
            return True  # Not a quantum state
        
        norm_squared = abs(state.amplitude_0)**2 + abs(state.amplitude_1)**2
        return abs(norm_squared - 1.0) < 1e-10
    
    def _validate_amplitudes_finite(self, state: Any) -> bool:
        """Validate quantum amplitudes are finite."""
        if not isinstance(state, QubitState):
            return True
        
        return (np.isfinite(state.amplitude_0) and 
                np.isfinite(state.amplitude_1))
    
    def _validate_probabilities(self, state: Any) -> bool:
        """Validate quantum measurement probabilities."""
        if not isinstance(state, QubitState):
            return True
        
        prob_0 = state.probability_0
        prob_1 = state.probability_1
        
        return (0.0 <= prob_0 <= 1.0 and 
                0.0 <= prob_1 <= 1.0 and
                abs(prob_0 + prob_1 - 1.0) < 1e-10)
    
    def _validate_coherence_bounds(self, data: Any) -> bool:
        """Validate quantum coherence is within reasonable bounds."""
        if hasattr(data, 'coherence'):
            coherence = data.coherence
            return 0.0 <= coherence <= 1.0
        elif isinstance(data, dict) and 'coherence' in data:
            coherence = data['coherence']
            return 0.0 <= coherence <= 1.0
        return True
    
    def validate_quantum_circuit(self, parameters: np.ndarray) -> ValidationResult:
        """
        Validate quantum circuit parameters.
        
        Args:
            parameters: Circuit parameters (angles, etc.)
            
        Returns:
            ValidationResult for quantum circuit
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        # Check parameter array
        if not isinstance(parameters, np.ndarray):
            result.errors.append("Circuit parameters must be numpy array")
            result.is_valid = False
            return result
        
        # Check for finite values
        if not np.all(np.isfinite(parameters)):
            result.errors.append("Circuit parameters must be finite")
            result.is_valid = False
        
        # Check parameter bounds (angles should be reasonable)
        if np.any(np.abs(parameters) > 10 * np.pi):
            result.warnings.append("Some circuit parameters are very large (>10π)")
        
        # Check for potential numerical instability
        if np.any(np.abs(parameters) < 1e-15):
            result.warnings.append("Some circuit parameters are very small and may cause numerical issues")
        
        logger.debug(f"Quantum circuit validation: {'PASS' if result.is_valid else 'FAIL'}")
        return result
    
    def validate_optimization_result(self, result: Any) -> ValidationResult:
        """
        Validate quantum optimization result.
        
        Args:
            result: Optimization result object
            
        Returns:
            ValidationResult for optimization result
        """
        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        # Check required attributes
        required_attrs = ['solution', 'objective_value', 'iterations']
        for attr in required_attrs:
            if not hasattr(result, attr):
                validation_result.errors.append(f"Missing required attribute: {attr}")
                validation_result.is_valid = False
        
        if hasattr(result, 'solution'):
            if not isinstance(result.solution, np.ndarray):
                validation_result.errors.append("Solution must be numpy array")
                validation_result.is_valid = False
            elif not np.all(np.isfinite(result.solution)):
                validation_result.errors.append("Solution contains non-finite values")
                validation_result.is_valid = False
        
        if hasattr(result, 'objective_value'):
            if not np.isfinite(result.objective_value):
                validation_result.errors.append("Objective value must be finite")
                validation_result.is_valid = False
        
        if hasattr(result, 'iterations') and hasattr(result, 'convergence_time'):
            if result.iterations == 0 and result.convergence_time > 0:
                validation_result.warnings.append("Zero iterations with non-zero convergence time")
        
        return validation_result


class CompositeValidator:
    """
    Composite validator that combines multiple validators.
    
    Useful for validating complex objects that need multiple validation types.
    """
    
    def __init__(self, name: str):
        """
        Initialize composite validator.
        
        Args:
            name: Name for logging
        """
        self.name = name
        self.validators: List[BaseValidator] = []
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the composite."""
        self.validators.append(validator)
        logger.debug(f"Added {validator.name} to {self.name}")
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data using all validators.
        
        Args:
            data: Data to validate
            
        Returns:
            Combined ValidationResult
        """
        combined_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info_messages=[],
            field_errors={}
        )
        
        for validator in self.validators:
            result = validator.validate(data)
            
            # Combine results
            if not result.is_valid:
                combined_result.is_valid = False
            
            combined_result.errors.extend([f"{validator.name}: {err}" for err in result.errors])
            combined_result.warnings.extend([f"{validator.name}: {warn}" for warn in result.warnings])
            combined_result.info_messages.extend([f"{validator.name}: {info}" for info in result.info_messages])
            
            # Merge field errors
            for field, field_errors in result.field_errors.items():
                field_key = f"{validator.name}_{field}"
                combined_result.field_errors[field_key] = field_errors
        
        logger.info(f"{self.name} composite validation: {'PASS' if combined_result.is_valid else 'FAIL'}")
        return combined_result


def create_full_validator() -> CompositeValidator:
    """
    Create a composite validator with all validation types.
    
    Returns:
        CompositeValidator configured for full quantum task planner validation
    """
    composite = CompositeValidator("FullQuantumPlannerValidator")
    
    composite.add_validator(TaskValidator())
    composite.add_validator(ResourceValidator())
    composite.add_validator(QuantumValidator())
    
    return composite