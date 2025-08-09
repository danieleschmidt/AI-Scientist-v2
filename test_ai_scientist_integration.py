#!/usr/bin/env python3
"""
Integration test for AI Scientist v2 autonomous implementation
Tests core functionality without external dependencies
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_unified_research_orchestrator():
    """Test unified research orchestrator without rich dependency."""
    
    print("🧪 Testing Unified Research Orchestrator...")
    
    try:
        # Test basic imports and class instantiation
        from ai_scientist.unified_research_orchestrator import (
            ResearchTask, ResearchWorkflow, UnifiedResearchOrchestrator
        )
        
        # Create orchestrator with minimal dependencies
        orchestrator = UnifiedResearchOrchestrator()
        
        # Create a simple workflow
        workflow = orchestrator.create_standard_workflow("Test Research Topic")
        
        assert workflow.workflow_id is not None
        assert workflow.name == "Research: Test Research Topic"
        assert len(workflow.tasks) > 0
        
        print("  ✅ Orchestrator creation and workflow generation successful")
        
        # Test task creation
        task = ResearchTask(
            task_id="test_task",
            task_type="test",
            parameters={"param": "value"}
        )
        
        assert task.status == 'pending'
        assert task.task_id == "test_task"
        
        print("  ✅ Task creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Orchestrator test failed: {e}")
        return False


def test_adaptive_experiment_manager():
    """Test adaptive experiment manager."""
    
    print("🧪 Testing Adaptive Experiment Manager...")
    
    try:
        from ai_scientist.adaptive_experiment_manager import (
            AdaptiveExperimentManager, ExperimentConfiguration, ExperimentStatus
        )
        
        # Create manager
        manager = AdaptiveExperimentManager(max_concurrent_experiments=2)
        
        # Create experiment
        experiment_id = manager.create_experiment(
            experiment_id="test_exp_001",
            base_config={"model": "test_model", "epochs": 10},
            optimization_targets=['accuracy']
        )
        
        assert experiment_id == "test_exp_001"
        assert experiment_id in manager.active_experiments
        
        print("  ✅ Experiment creation successful")
        
        # Test metrics
        experiment = manager.active_experiments[experiment_id]
        experiment.metrics.update_metric('accuracy', 0.85)
        
        assert experiment.metrics.accuracy == 0.85
        
        print("  ✅ Metrics tracking successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Experiment manager test failed: {e}")
        return False


def test_intelligent_hypothesis_generator():
    """Test intelligent hypothesis generator."""
    
    print("🧪 Testing Intelligent Hypothesis Generator...")
    
    try:
        from ai_scientist.intelligent_hypothesis_generator import (
            IntelligentHypothesisGenerator, NoveltyLevel, FeasibilityLevel
        )
        
        # Create generator
        generator = IntelligentHypothesisGenerator()
        
        # Test domain knowledge initialization
        assert 'machine_learning' in generator.domain_knowledge
        assert 'improvement' in generator.research_patterns
        
        print("  ✅ Domain knowledge initialization successful")
        
        # Create a simple hypothesis generation (without async execution)
        from ai_scientist.intelligent_hypothesis_generator import ResearchGap, HypothesisCandidate
        
        gap = ResearchGap(
            gap_id="test_gap",
            domain="machine_learning",
            description="Test research gap",
            severity=0.8,
            opportunity_score=0.7
        )
        
        hypothesis = HypothesisCandidate(
            hypothesis_id="test_hyp",
            title="Test Hypothesis",
            statement="This is a test hypothesis",
            domain="machine_learning",
            research_question="How can we test this?",
            novelty_level=NoveltyLevel.SIGNIFICANT,
            novelty_score=0.7,
            feasibility_level=FeasibilityLevel.HIGH,
            feasibility_score=0.8,
            impact_score=0.6,
            methodology="Test methodology"
        )
        
        assert hypothesis.hypothesis_id == "test_hyp"
        assert hypothesis.novelty_level == NoveltyLevel.SIGNIFICANT
        
        print("  ✅ Hypothesis structure creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hypothesis generator test failed: {e}")
        return False


def test_robust_error_handling():
    """Test robust error handling system."""
    
    print("🧪 Testing Robust Error Handling...")
    
    try:
        from ai_scientist.robust_error_handling import (
            RobustErrorHandler, ErrorSeverity, ErrorCategory
        )
        
        # Create error handler
        handler = RobustErrorHandler()
        
        # Test error classification
        test_exception = ValueError("Test error message")
        error_context = handler.create_error_context(
            exception=test_exception,
            operation="test_operation",
            component="test_component"
        )
        
        assert error_context.error_id is not None
        assert error_context.severity in ErrorSeverity
        assert error_context.category in ErrorCategory
        
        print("  ✅ Error context creation successful")
        
        # Test recovery strategies
        assert len(handler.recovery_strategies) > 0
        
        print("  ✅ Recovery strategies initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def test_advanced_monitoring():
    """Test advanced monitoring system."""
    
    print("🧪 Testing Advanced Monitoring...")
    
    try:
        from ai_scientist.advanced_monitoring import (
            AdvancedMonitoringSystem, MetricType, HealthStatus
        )
        
        # Create monitoring system
        monitoring = AdvancedMonitoringSystem()
        
        # Test metric registration
        metric = monitoring.register_metric(
            name="test_metric",
            metric_type=MetricType.PERFORMANCE,
            description="Test metric",
            unit="count"
        )
        
        assert metric.name == "test_metric"
        assert "test_metric" in monitoring.metrics
        
        print("  ✅ Metric registration successful")
        
        # Test metric recording
        monitoring.record_metric("test_metric", 42)
        
        current_value = monitoring.metrics["test_metric"].get_current_value()
        assert current_value == 42
        
        print("  ✅ Metric recording successful")
        
        # Test health checks
        assert len(monitoring.health_checks) > 0
        
        print("  ✅ Health checks initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring test failed: {e}")
        return False


def test_security_framework():
    """Test security framework."""
    
    print("🧪 Testing Security Framework...")
    
    try:
        from ai_scientist.security_framework import (
            SecurityFramework, ThreatLevel, SecurityEvent
        )
        
        # Create security framework
        security = SecurityFramework()
        
        # Test input sanitization
        malicious_input = "'; DROP TABLE users; --"
        sanitized = security.sanitize_input(malicious_input, "test_field")
        
        assert sanitized != malicious_input
        assert "[FILTERED]" in sanitized or "&" in sanitized  # HTML encoded or filtered
        
        print("  ✅ Input sanitization successful")
        
        # Test file path validation
        safe_path = "data/test.txt"
        unsafe_path = "../../../etc/passwd"
        
        assert security.validate_file_path(safe_path, "data") == True
        assert security.validate_file_path(unsafe_path, "data") == False
        
        print("  ✅ Path validation successful")
        
        # Test API key generation
        api_key = security.generate_api_key("test_user", {"read", "write"})
        
        assert "." in api_key  # Should have key_id.key_secret format
        
        key_data = security.validate_api_key(api_key)
        assert key_data is not None
        assert key_data['user_id'] == "test_user"
        
        print("  ✅ API key management successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security framework test failed: {e}")
        return False


def test_distributed_computing():
    """Test distributed computing engine."""
    
    print("🧪 Testing Distributed Computing...")
    
    try:
        from ai_scientist.distributed_computing_engine import (
            DistributedComputingEngine, ComputeNode, NodeStatus, TaskPriority
        )
        
        # Create distributed engine
        engine = DistributedComputingEngine(cluster_name="test_cluster")
        
        # Test node management
        node_id = engine.add_node(
            hostname="test-node",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0
        )
        
        assert node_id in engine.nodes
        assert engine.nodes[node_id].hostname == "test-node"
        
        print("  ✅ Node management successful")
        
        # Test task submission (without execution)
        def dummy_function(x):
            return x * 2
        
        task_id = engine.submit_task(
            dummy_function,
            5,
            priority=TaskPriority.NORMAL,
            cpu_cores=1
        )
        
        assert task_id in engine.tasks
        assert engine.tasks[task_id].function_name == "dummy_function"
        
        print("  ✅ Task submission successful")
        
        # Test cluster metrics
        engine._update_cluster_metrics()
        
        assert engine.cluster_metrics.total_nodes >= 1
        assert engine.cluster_metrics.total_cpu_cores > 0
        
        print("  ✅ Cluster metrics successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Distributed computing test failed: {e}")
        return False


def test_system_integration():
    """Test integration between systems."""
    
    print("🧪 Testing System Integration...")
    
    try:
        # Test that all systems can be imported together
        from ai_scientist.unified_research_orchestrator import UnifiedResearchOrchestrator
        from ai_scientist.adaptive_experiment_manager import AdaptiveExperimentManager
        from ai_scientist.intelligent_hypothesis_generator import IntelligentHypothesisGenerator
        from ai_scientist.robust_error_handling import RobustErrorHandler
        from ai_scientist.advanced_monitoring import AdvancedMonitoringSystem
        from ai_scientist.security_framework import SecurityFramework
        from ai_scientist.distributed_computing_engine import DistributedComputingEngine
        
        # Create instances of all systems
        orchestrator = UnifiedResearchOrchestrator()
        experiment_manager = AdaptiveExperimentManager()
        hypothesis_generator = IntelligentHypothesisGenerator()
        error_handler = RobustErrorHandler()
        monitoring = AdvancedMonitoringSystem()
        security = SecurityFramework()
        distributed_engine = DistributedComputingEngine()
        
        # Verify they all have basic functionality
        assert hasattr(orchestrator, 'create_standard_workflow')
        assert hasattr(experiment_manager, 'create_experiment')
        assert hasattr(hypothesis_generator, 'domain_knowledge')
        assert hasattr(error_handler, 'recovery_strategies')
        assert hasattr(monitoring, 'metrics')
        assert hasattr(security, 'security_config')
        assert hasattr(distributed_engine, 'nodes')
        
        print("  ✅ All systems integrated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    
    print("🚀 Starting AI Scientist v2 - Autonomous Implementation Integration Tests\n")
    
    tests = [
        test_unified_research_orchestrator,
        test_adaptive_experiment_manager,
        test_intelligent_hypothesis_generator,
        test_robust_error_handling,
        test_advanced_monitoring,
        test_security_framework,
        test_distributed_computing,
        test_system_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests PASSED! System is ready for deployment.")
        
        # Generate system summary
        print("\n📋 System Summary:")
        print("  ✅ Generation 1: MAKE IT WORK - Core functionality implemented")
        print("  ✅ Generation 2: MAKE IT ROBUST - Error handling, monitoring, security")
        print("  ✅ Generation 3: MAKE IT SCALE - Distributed computing, load balancing")
        print("  🔍 Quality Gates: Integration tests passed")
        
        return True
    else:
        print(f"❌ {total - passed} tests failed. Review implementation before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)