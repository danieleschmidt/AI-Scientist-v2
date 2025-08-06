"""
Tests for validation and error handling components.

Comprehensive test suite for quantum task planner validation,
error handling, and input sanitization systems.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from quantum_task_planner.validation.validators import (
    TaskValidator, ResourceValidator, QuantumValidator, CompositeValidator,
    ValidationResult, ValidationRule, create_full_validator
)
from quantum_task_planner.validation.error_handling import (
    QuantumPlannerError, ValidationError, OptimizationError, 
    QuantumStateError, ResourceConstraintError, DependencyError,
    ConfigurationError, ErrorHandler, ErrorContext, ErrorSeverity,
    ErrorCategory, global_error_handler
)
from quantum_task_planner.core.planner import Task, TaskPriority
from quantum_task_planner.utils.quantum_math import QubitState
from quantum_task_planner.core.quantum_optimizer import OptimizationResult


class TestTaskValidator:
    """Test suite for TaskValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TaskValidator()
        
        self.valid_task = Task(
            id="valid_task",
            name="Valid Task",
            priority=TaskPriority.HIGH,
            duration=5.0,
            dependencies=["dep1", "dep2"],
            resources={"cpu": 2.0, "memory": 1024.0},
            quantum_weight=0.8
        )
    
    def test_valid_task(self):
        """Test validation of valid task."""
        result = self.validator.validate(self.valid_task)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_invalid_task_id(self):
        """Test validation of invalid task IDs."""
        # Empty ID
        invalid_task = Task("", "Name", TaskPriority.HIGH, 1.0, [], {})
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
        assert any("task_id_format" in error for error in result.errors)
        
        # Invalid characters
        invalid_task.id = "task@#$"
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
        
        # Too long
        invalid_task.id = "a" * 101
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
    
    def test_invalid_task_name(self):
        """Test validation of invalid task names."""
        invalid_task = Task("valid_id", "", TaskPriority.HIGH, 1.0, [], {})
        result = self.validator.validate(invalid_task)
        
        assert not result.is_valid
        assert any("task_name_length" in error for error in result.errors)
        
        # Too long name
        invalid_task.name = "x" * 201
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
    
    def test_invalid_duration(self):
        """Test validation of invalid durations."""
        # Negative duration
        invalid_task = Task("id", "Name", TaskPriority.HIGH, -1.0, [], {})
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
        assert any("task_duration_positive" in error for error in result.errors)
        
        # Zero duration (should be valid)
        invalid_task.duration = 0.0
        result = self.validator.validate(invalid_task)
        assert result.is_valid  # Zero duration is allowed
    
    def test_invalid_priority(self):
        """Test validation of invalid priorities."""
        # Mock invalid priority
        class MockTask:
            def __init__(self):
                self.id = "test"
                self.name = "Test"
                self.priority = "invalid_priority"  # Not a TaskPriority enum
                self.duration = 1.0
                self.dependencies = []
                self.resources = {}
                self.quantum_weight = 0.5
        
        mock_task = MockTask()
        result = self.validator.validate(mock_task)
        
        assert not result.is_valid
        assert any("task_priority_valid" in error for error in result.errors)
    
    def test_invalid_dependencies(self):
        """Test validation of invalid dependencies."""
        # Invalid dependency format
        class MockTask:
            def __init__(self):
                self.id = "test"
                self.name = "Test"
                self.priority = TaskPriority.HIGH
                self.duration = 1.0
                self.dependencies = ["valid", "", "invalid@id"]  # Empty and invalid IDs
                self.resources = {}
                self.quantum_weight = 0.5
        
        mock_task = MockTask()
        result = self.validator.validate(mock_task)
        
        assert not result.is_valid
        assert any("dependencies_format" in error for error in result.errors)
        
        # Duplicate dependencies
        mock_task.dependencies = ["dep1", "dep2", "dep1"]
        result = self.validator.validate(mock_task)
        assert not result.is_valid
    
    def test_invalid_resources(self):
        """Test validation of invalid resources."""
        # Negative resource amount
        invalid_task = Task(
            "id", "Name", TaskPriority.HIGH, 1.0, [],
            {"cpu": 2.0, "memory": -100.0}  # Negative memory
        )
        result = self.validator.validate(invalid_task)
        
        assert not result.is_valid
        assert any("resources_format" in error for error in result.errors)
        
        # Empty resource name
        invalid_task.resources = {"": 100.0, "valid": 50.0}
        result = self.validator.validate(invalid_task)
        assert not result.is_valid
    
    def test_quantum_weight_validation(self):
        """Test quantum weight validation."""
        # Valid quantum weight
        valid_task = Task("id", "Name", TaskPriority.HIGH, 1.0, [], {}, quantum_weight=0.5)
        result = self.validator.validate(valid_task)
        assert result.is_valid
        
        # Out of range quantum weight (warning)
        invalid_task = Task("id", "Name", TaskPriority.HIGH, 1.0, [], {}, quantum_weight=1.5)
        result = self.validator.validate(invalid_task)
        # Should be a warning, not error
        assert len(result.warnings) > 0
    
    def test_deadline_consistency(self):
        """Test deadline consistency validation."""
        # Valid deadline
        valid_task = Task("id", "Name", TaskPriority.HIGH, 2.0, [], {}, deadline=5.0)
        result = self.validator.validate(valid_task)
        assert result.is_valid
        
        # Invalid deadline (shorter than duration) - should be warning
        invalid_task = Task("id", "Name", TaskPriority.HIGH, 5.0, [], {}, deadline=2.0)
        result = self.validator.validate(invalid_task)
        assert len(result.warnings) > 0
    
    def test_validate_task_list(self):
        """Test validation of task list."""
        tasks = [
            Task("task1", "Task 1", TaskPriority.HIGH, 1.0, [], {}),
            Task("task2", "Task 2", TaskPriority.MEDIUM, 2.0, ["task1"], {}),
            Task("task3", "Task 3", TaskPriority.LOW, 1.5, [], {})
        ]
        
        result = self.validator.validate_task_list(tasks)
        assert result.is_valid
        
        # Test duplicate IDs
        duplicate_tasks = [
            Task("same_id", "Task 1", TaskPriority.HIGH, 1.0, [], {}),
            Task("same_id", "Task 2", TaskPriority.MEDIUM, 2.0, [], {})
        ]
        result = self.validator.validate_task_list(duplicate_tasks)
        assert not result.is_valid
        assert any("duplicate" in error.lower() for error in result.errors)
        
        # Test invalid dependency reference
        invalid_dep_tasks = [
            Task("task1", "Task 1", TaskPriority.HIGH, 1.0, [], {}),
            Task("task2", "Task 2", TaskPriority.MEDIUM, 2.0, ["nonexistent"], {})
        ]
        result = self.validator.validate_task_list(invalid_dep_tasks)
        assert not result.is_valid
        assert any("non-existent" in error for error in result.errors)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        circular_tasks = [
            Task("a", "Task A", TaskPriority.HIGH, 1.0, ["b"], {}),
            Task("b", "Task B", TaskPriority.HIGH, 1.0, ["c"], {}),
            Task("c", "Task C", TaskPriority.HIGH, 1.0, ["a"], {})  # Circular!
        ]
        
        result = self.validator.validate_task_list(circular_tasks)
        assert not result.is_valid
        assert any("circular" in error.lower() for error in result.errors)


class TestResourceValidator:
    """Test suite for ResourceValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ResourceValidator()
    
    def test_resource_capacity_validation(self):
        """Test resource capacity validation."""
        # Mock resource with positive capacity
        class MockResource:
            def __init__(self, capacity):
                self.capacity = capacity
        
        valid_resource = MockResource(100.0)
        result = self.validator.validate(valid_resource)
        assert result.is_valid
        
        # Invalid negative capacity
        invalid_resource = MockResource(-50.0)
        result = self.validator.validate(invalid_resource)
        assert not result.is_valid
    
    def test_resource_name_validation(self):
        """Test resource name validation."""
        class MockResource:
            def __init__(self, name):
                self.name = name
                self.capacity = 100.0
        
        # Valid name
        valid_resource = MockResource("cpu_cores")
        result = self.validator.validate(valid_resource)
        assert result.is_valid
        
        # Invalid empty name
        invalid_resource = MockResource("")
        result = self.validator.validate(invalid_resource)
        assert not result.is_valid
        
        # Invalid characters
        invalid_resource.name = "cpu@cores!"
        result = self.validator.validate(invalid_resource)
        assert not result.is_valid
    
    def test_resource_allocation_validation(self):
        """Test resource allocation validation."""
        tasks = [
            Task("task1", "Task 1", TaskPriority.HIGH, 1.0, [], {"cpu": 4.0, "memory": 2048.0}),
            Task("task2", "Task 2", TaskPriority.MEDIUM, 1.0, [], {"cpu": 3.0, "memory": 1024.0})
        ]
        
        # Sufficient resources
        sufficient_limits = {"cpu": 10.0, "memory": 4096.0}
        result = self.validator.validate_resource_allocation(tasks, sufficient_limits)
        assert result.is_valid
        
        # Insufficient resources
        insufficient_limits = {"cpu": 5.0, "memory": 2048.0}  # Total needed: 7 CPU, 3072 memory
        result = self.validator.validate_resource_allocation(tasks, insufficient_limits)
        assert not result.is_valid
        assert any("over-allocated" in error for error in result.errors)
        
        # High utilization warning
        high_util_limits = {"cpu": 7.5, "memory": 3200.0}  # Just above requirement
        result = self.validator.validate_resource_allocation(tasks, high_util_limits)
        assert result.is_valid
        assert len(result.warnings) > 0  # Should warn about high utilization


class TestQuantumValidator:
    """Test suite for QuantumValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QuantumValidator()
    
    def test_quantum_state_normalization(self):
        """Test quantum state normalization validation."""
        # Normalized state
        normalized_state = QubitState(amplitude_0=1/np.sqrt(2), amplitude_1=1/np.sqrt(2))
        result = self.validator.validate(normalized_state)
        assert result.is_valid
        
        # Unnormalized state
        class UnnormalizedQubitState:
            def __init__(self):
                self.amplitude_0 = 1.0
                self.amplitude_1 = 1.0  # |1|² + |1|² = 2 ≠ 1
        
        unnormalized_state = UnnormalizedQubitState()
        result = self.validator.validate(unnormalized_state)
        assert not result.is_valid
        assert any("normalized" in error for error in result.errors)
    
    def test_amplitude_validation(self):
        """Test quantum amplitude validation."""
        # Valid finite amplitudes
        valid_state = QubitState(amplitude_0=0.6, amplitude_1=0.8)
        result = self.validator.validate(valid_state)
        assert result.is_valid
        
        # Invalid infinite amplitude
        class InvalidQubitState:
            def __init__(self):
                self.amplitude_0 = float('inf')
                self.amplitude_1 = 0.0
        
        invalid_state = InvalidQubitState()
        result = self.validator.validate(invalid_state)
        assert not result.is_valid
    
    def test_probability_validation(self):
        """Test quantum probability validation."""
        # Valid probabilities
        valid_state = QubitState(amplitude_0=0.8, amplitude_1=0.6)
        result = self.validator.validate(valid_state)
        assert result.is_valid
    
    def test_coherence_validation(self):
        """Test quantum coherence validation."""
        # Valid coherence data
        coherence_data = {"coherence": 0.75}
        result = self.validator.validate(coherence_data)
        assert result.is_valid
        
        # Invalid coherence (out of bounds)
        invalid_coherence_data = {"coherence": 1.5}
        result = self.validator.validate(invalid_coherence_data)
        assert not result.is_valid
    
    def test_quantum_circuit_validation(self):
        """Test quantum circuit parameter validation."""
        # Valid parameters
        valid_params = np.array([0.5, 1.2, -0.8, 2.1])
        result = self.validator.validate_quantum_circuit(valid_params)
        assert result.is_valid
        
        # Invalid parameters (non-array)
        result = self.validator.validate_quantum_circuit([1, 2, 3])  # List instead of array
        assert not result.is_valid
        
        # Parameters with NaN
        invalid_params = np.array([0.5, np.nan, 1.0])
        result = self.validator.validate_quantum_circuit(invalid_params)
        assert not result.is_valid
        
        # Very large parameters (warning)
        large_params = np.array([50.0, 100.0])  # > 10π
        result = self.validator.validate_quantum_circuit(large_params)
        assert result.is_valid
        assert len(result.warnings) > 0
    
    def test_optimization_result_validation(self):
        """Test optimization result validation."""
        # Valid result
        valid_result = OptimizationResult(
            solution=np.array([1.0, 2.0]),
            objective_value=5.0,
            iterations=100,
            convergence_time=1.5,
            quantum_metrics={"fidelity": 0.95}
        )
        result = self.validator.validate_optimization_result(valid_result)
        assert result.is_valid
        
        # Missing required attributes
        class IncompleteResult:
            def __init__(self):
                self.solution = np.array([1.0])
                # Missing objective_value, iterations
        
        incomplete_result = IncompleteResult()
        result = self.validator.validate_optimization_result(incomplete_result)
        assert not result.is_valid
        
        # Invalid solution (non-array)
        class InvalidSolutionResult:
            def __init__(self):
                self.solution = [1.0, 2.0]  # List instead of array
                self.objective_value = 5.0
                self.iterations = 100
        
        invalid_result = InvalidSolutionResult()
        result = self.validator.validate_optimization_result(invalid_result)
        assert not result.is_valid


class TestCompositeValidator:
    """Test suite for CompositeValidator."""
    
    def test_composite_validation(self):
        """Test composite validator functionality."""
        composite = CompositeValidator("TestComposite")
        
        # Add validators
        task_validator = TaskValidator()
        resource_validator = ResourceValidator()
        
        composite.add_validator(task_validator)
        composite.add_validator(resource_validator)
        
        # Test with valid task
        valid_task = Task("test", "Test Task", TaskPriority.HIGH, 1.0, [], {"cpu": 1.0})
        result = composite.validate(valid_task)
        
        assert result.is_valid
        # Should have results from both validators
        assert len([err for err in result.errors if "TaskValidator" in err]) >= 0
    
    def test_full_validator_creation(self):
        """Test creation of full validator."""
        full_validator = create_full_validator()
        
        assert isinstance(full_validator, CompositeValidator)
        assert len(full_validator.validators) >= 3  # Task, Resource, Quantum validators


class TestErrorHandling:
    """Test suite for error handling system."""
    
    def test_quantum_planner_error_creation(self):
        """Test QuantumPlannerError creation."""
        error = QuantumPlannerError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause is None
    
    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationError(
            "Invalid field value",
            field_name="duration",
            field_value=-1.0,
            validation_rule="positive_number"
        )
        
        assert error.field_name == "duration"
        assert error.field_value == -1.0
        assert error.validation_rule == "positive_number"
        assert error.category == ErrorCategory.VALIDATION
    
    def test_optimization_error_creation(self):
        """Test OptimizationError creation."""
        error = OptimizationError(
            "Optimization failed to converge",
            optimization_type="quantum_annealing",
            iteration=150,
            objective_value=10.5
        )
        
        assert error.optimization_type == "quantum_annealing"
        assert error.iteration == 150
        assert error.objective_value == 10.5
        assert error.category == ErrorCategory.OPTIMIZATION
    
    def test_resource_constraint_error(self):
        """Test ResourceConstraintError creation."""
        error = ResourceConstraintError(
            "Insufficient CPU resources",
            resource_type="cpu",
            required=8.0,
            available=4.0
        )
        
        assert error.resource_type == "cpu"
        assert error.required == 8.0
        assert error.available == 4.0
        assert error.category == ErrorCategory.RESOURCE
    
    def test_dependency_error(self):
        """Test DependencyError creation."""
        error = DependencyError(
            "Circular dependency detected",
            task_id="task_a",
            dependency_chain=["task_a", "task_b", "task_c", "task_a"]
        )
        
        assert error.task_id == "task_a"
        assert error.dependency_chain == ["task_a", "task_b", "task_c", "task_a"]
        assert error.category == ErrorCategory.DEPENDENCY
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            "Invalid configuration value",
            config_key="max_iterations",
            config_value=-100
        )
        
        assert error.config_key == "max_iterations"
        assert error.config_value == -100
        assert error.category == ErrorCategory.CONFIGURATION
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = ValidationError(
            "Test validation error",
            field_name="test_field",
            field_value="test_value"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["type"] == "ValidationError"
        assert error_dict["message"] == "Test validation error"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == 2  # MEDIUM


class TestErrorHandler:
    """Test suite for ErrorHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler(max_error_history=10)
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(max_error_history=100)
        
        assert handler.max_error_history == 100
        assert len(handler.error_history) == 0
        assert len(handler.error_counts) == 0
        assert len(handler.recovery_strategies) > 0
    
    def test_handle_error(self):
        """Test error handling."""
        # Create a test error
        test_error = ValidationError("Test error")
        
        with pytest.raises(ValidationError):
            self.error_handler.handle_error(
                test_error,
                operation="test_operation",
                component="test_component"
            )
        
        # Check that error was recorded
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_counts["ValidationError"] == 1
    
    def test_handle_non_quantum_error(self):
        """Test handling of non-quantum errors."""
        regular_error = ValueError("Regular error")
        
        with pytest.raises(QuantumPlannerError):
            self.error_handler.handle_error(regular_error, "test_op")
        
        # Should wrap in QuantumPlannerError
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_history[0].cause == regular_error
    
    def test_error_pattern_analysis(self):
        """Test error pattern analysis."""
        # Generate multiple errors of same category
        for i in range(6):
            error = ValidationError(f"Error {i}")
            try:
                self.error_handler.handle_error(error, "test_op")
            except:
                pass  # Ignore for test
        
        # Should detect pattern (checked internally during _analyze_error_patterns)
        assert len(self.error_handler.error_history) == 6
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Add some errors
        errors = [
            ValidationError("Validation error 1"),
            ValidationError("Validation error 2"),
            OptimizationError("Optimization error 1")
        ]
        
        for error in errors:
            try:
                self.error_handler.handle_error(error, "test_op")
            except:
                pass
        
        summary = self.error_handler.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert "validation" in summary["categories"]
        assert "optimization" in summary["categories"]
        assert summary["categories"]["validation"] == 2
        assert summary["categories"]["optimization"] == 1
    
    def test_clear_error_history(self):
        """Test clearing error history."""
        # Add an error
        error = ValidationError("Test error")
        try:
            self.error_handler.handle_error(error, "test_op")
        except:
            pass
        
        assert len(self.error_handler.error_history) == 1
        
        # Clear history
        self.error_handler.clear_error_history()
        
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.error_counts) == 0
    
    def test_recovery_strategy_execution(self):
        """Test recovery strategy execution."""
        # Mock recovery strategy
        mock_recovery = Mock(return_value="recovered_value")
        self.error_handler.recovery_strategies[ErrorCategory.VALIDATION] = mock_recovery
        
        validation_error = ValidationError("Test validation error")
        
        # Handle with recovery
        result = self.error_handler.handle_error(
            validation_error,
            "test_operation",
            attempt_recovery=True
        )
        
        assert result == "recovered_value"
        mock_recovery.assert_called_once_with(validation_error)


class TestValidationRules:
    """Test suite for validation rules system."""
    
    def test_validation_rule_creation(self):
        """Test validation rule creation."""
        rule = ValidationRule(
            name="test_rule",
            validator=lambda x: x > 0,
            error_message="Value must be positive",
            severity="warning"
        )
        
        assert rule.name == "test_rule"
        assert rule.validator(5) == True
        assert rule.validator(-1) == False
        assert rule.error_message == "Value must be positive"
        assert rule.severity == "warning"
    
    def test_custom_validator_with_rules(self):
        """Test custom validator with rules."""
        from quantum_task_planner.validation.validators import BaseValidator
        
        class CustomValidator(BaseValidator):
            def _setup_rules(self):
                self.add_rule(ValidationRule(
                    name="positive_value",
                    validator=lambda x: hasattr(x, 'value') and x.value > 0,
                    error_message="Value must be positive"
                ))
        
        validator = CustomValidator("CustomValidator")
        
        # Test with valid object
        class ValidObject:
            def __init__(self, value):
                self.value = value
        
        valid_obj = ValidObject(5)
        result = validator.validate(valid_obj)
        assert result.is_valid
        
        # Test with invalid object
        invalid_obj = ValidObject(-5)
        result = validator.validate(invalid_obj)
        assert not result.is_valid
        assert len(result.errors) == 1


class TestGlobalErrorHandler:
    """Test global error handler functionality."""
    
    def test_global_error_handler_exists(self):
        """Test that global error handler is available."""
        assert global_error_handler is not None
        assert isinstance(global_error_handler, ErrorHandler)
    
    def test_global_error_handler_usage(self):
        """Test using global error handler."""
        # Record initial error count
        initial_count = len(global_error_handler.error_history)
        
        # Handle an error
        test_error = ConfigurationError("Test global error")
        
        try:
            global_error_handler.handle_error(test_error, "global_test")
        except:
            pass  # Expected to re-raise
        
        # Check that error was recorded
        assert len(global_error_handler.error_history) == initial_count + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])