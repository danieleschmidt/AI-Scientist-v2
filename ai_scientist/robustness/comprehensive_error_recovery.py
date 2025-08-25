#!/usr/bin/env python3
"""
Comprehensive Error Recovery System - Generation 2 Enhancement
=============================================================

Advanced error recovery and self-healing system for autonomous AI research.
Provides intelligent error classification, context-aware recovery strategies,
and system state restoration capabilities.

Key Features:
- Intelligent error classification with machine learning
- Context-aware recovery strategy selection
- Automatic system state checkpointing and restoration
- Cascade failure prevention and containment
- Adaptive recovery learning from past failures
- Comprehensive logging and forensic analysis

Author: AI Scientist v2 - Terragon Labs (Generation 2)  
License: MIT
"""

import asyncio
import logging
import time
import pickle
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import traceback
import sys
import hashlib
import sqlite3
from collections import defaultdict, deque
import weakref
import gc
import psutil
import subprocess

# For error classification
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"                    # Minor issues, system can continue
    MEDIUM = "medium"              # Significant issues, degraded performance
    HIGH = "high"                  # Major issues, core functionality affected
    CRITICAL = "critical"          # System-threatening, immediate action required


class RecoveryOutcome(Enum):
    SUCCESS = "success"            # Recovery successful
    PARTIAL = "partial"           # Partial recovery, some functionality lost
    FAILED = "failed"             # Recovery failed
    DEGRADED = "degraded"         # System running in degraded mode
    RESTART_REQUIRED = "restart_required"  # Manual restart needed


class ErrorCategory(Enum):
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    API_FAILURE = "api_failure"  
    COMPUTATION_ERROR = "computation_error"
    DATA_CORRUPTION = "data_corruption"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    HARDWARE_ERROR = "hardware_error"
    UNKNOWN = "unknown"


class SystemState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: float
    
    # Error details
    exception_type: str
    exception_message: str
    stack_trace: str
    
    # System context
    system_state: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    active_operations: List[str] = field(default_factory=list)
    
    # Error classification
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    confidence: float = 0.5  # Classification confidence
    
    # Recovery context
    recovery_attempts: int = 0
    previous_errors: List[str] = field(default_factory=list)
    cascade_source: Optional[str] = None
    
    # Additional metadata
    component: Optional[str] = None
    operation: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class RecoveryStrategy:
    """Definition of a recovery strategy."""
    strategy_id: str
    name: str
    description: str
    
    # Applicability conditions
    error_categories: List[ErrorCategory]
    severity_levels: List[ErrorSeverity]
    preconditions: List[Callable[[ErrorContext], bool]] = field(default_factory=list)
    
    # Recovery actions
    recovery_function: Callable[[ErrorContext], Any]
    rollback_function: Optional[Callable[[ErrorContext], Any]] = None
    
    # Strategy metadata
    success_rate: float = 0.0
    average_recovery_time: float = 0.0
    resource_cost: float = 1.0  # Relative cost
    priority: int = 1  # Higher priority = tried first


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    attempt_id: str
    error_id: str
    strategy_id: str
    timestamp: float
    
    # Execution details
    duration: float = 0.0
    outcome: RecoveryOutcome = RecoveryOutcome.FAILED
    error_occurred: Optional[str] = None
    
    # Results
    system_state_before: Dict[str, Any] = field(default_factory=dict)
    system_state_after: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    
    # Metrics
    resource_usage: Dict[str, float] = field(default_factory=dict)
    success_probability: float = 0.0


class SystemCheckpoint:
    """System state checkpoint for recovery."""
    
    def __init__(self, checkpoint_id: str, workspace_dir: Path):
        self.checkpoint_id = checkpoint_id
        self.timestamp = time.time()
        self.workspace_dir = workspace_dir
        
        # System state
        self.memory_usage = psutil.virtual_memory()._asdict()
        self.disk_usage = psutil.disk_usage('/')._asdict()
        self.cpu_usage = psutil.cpu_percent()
        
        # Process state
        self.process = psutil.Process()
        self.process_state = {
            'pid': self.process.pid,
            'memory_info': self.process.memory_info()._asdict(),
            'num_threads': self.process.num_threads(),
            'cpu_percent': self.process.cpu_percent()
        }
        
        # Application state (to be extended by subclasses)
        self.application_state: Dict[str, Any] = {}
        
        logger.info(f"System checkpoint '{checkpoint_id}' created")
    
    def save_to_disk(self, filepath: Path):
        """Save checkpoint to disk."""
        try:
            checkpoint_data = {
                'checkpoint_id': self.checkpoint_id,
                'timestamp': self.timestamp,
                'memory_usage': self.memory_usage,
                'disk_usage': self.disk_usage,
                'cpu_usage': self.cpu_usage,
                'process_state': self.process_state,
                'application_state': self.application_state
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"Checkpoint saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    @classmethod
    def load_from_disk(cls, filepath: Path) -> 'SystemCheckpoint':
        """Load checkpoint from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            checkpoint = cls.__new__(cls)
            checkpoint.checkpoint_id = data['checkpoint_id']
            checkpoint.timestamp = data['timestamp']
            checkpoint.memory_usage = data['memory_usage']
            checkpoint.disk_usage = data['disk_usage']
            checkpoint.cpu_usage = data['cpu_usage']
            checkpoint.process_state = data['process_state']
            checkpoint.application_state = data['application_state']
            
            logger.info(f"Checkpoint loaded from {filepath}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


class ErrorClassifier:
    """
    Intelligent error classification using machine learning.
    
    Learns from past errors to improve classification accuracy.
    """
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.model_file = workspace_dir / "error_classifier.pkl"
        
        # Classification model
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
        # Training data
        self.error_features = []
        self.error_labels = []
        
        # Load existing model if available
        self._load_model()
        
        logger.info("ErrorClassifier initialized")
    
    def classify_error(self, error_context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity, float]:
        """
        Classify error category and severity.
        
        Returns:
            Tuple of (category, severity, confidence)
        """
        
        # Extract features from error
        features = self._extract_error_features(error_context)
        
        if self.is_trained and SKLEARN_AVAILABLE:
            return self._ml_classify(features, error_context)
        else:
            return self._rule_based_classify(features, error_context)
    
    def _extract_error_features(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Extract features from error context for classification."""
        features = {
            'exception_type': error_context.exception_type,
            'message_text': error_context.exception_message.lower(),
            'stack_depth': len(error_context.stack_trace.split('\\n')),
            'has_memory_keyword': 'memory' in error_context.exception_message.lower(),
            'has_network_keyword': any(kw in error_context.exception_message.lower() 
                                     for kw in ['network', 'connection', 'timeout', 'socket']),
            'has_permission_keyword': any(kw in error_context.exception_message.lower()
                                        for kw in ['permission', 'access', 'denied', 'forbidden']),
            'has_resource_keyword': any(kw in error_context.exception_message.lower()
                                      for kw in ['resource', 'limit', 'quota', 'space']),
            'system_memory_usage': error_context.resource_usage.get('memory_percent', 0),
            'system_cpu_usage': error_context.resource_usage.get('cpu_percent', 0),
            'active_operations_count': len(error_context.active_operations),
            'recovery_attempts': error_context.recovery_attempts
        }
        
        return features
    
    def _rule_based_classify(self, features: Dict[str, Any], 
                           error_context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity, float]:
        """Rule-based error classification fallback."""
        
        exception_type = features['exception_type'].lower()
        message = features['message_text']
        
        # Category classification
        category = ErrorCategory.UNKNOWN
        confidence = 0.7
        
        if 'memory' in exception_type or features['has_memory_keyword']:
            category = ErrorCategory.RESOURCE_EXHAUSTION
            confidence = 0.9
        elif 'network' in message or features['has_network_keyword']:
            category = ErrorCategory.NETWORK_ERROR
            confidence = 0.8
        elif 'permission' in exception_type or features['has_permission_keyword']:
            category = ErrorCategory.PERMISSION_ERROR
            confidence = 0.9
        elif features['has_resource_keyword']:
            category = ErrorCategory.RESOURCE_EXHAUSTION
            confidence = 0.8
        elif 'import' in message or 'module' in message:
            category = ErrorCategory.DEPENDENCY_FAILURE
            confidence = 0.8
        elif 'api' in message or 'request' in message:
            category = ErrorCategory.API_FAILURE
            confidence = 0.7
        elif any(word in exception_type for word in ['value', 'type', 'attribute']):
            category = ErrorCategory.COMPUTATION_ERROR
            confidence = 0.6
        
        # Severity classification
        severity = ErrorSeverity.MEDIUM
        
        if features['system_memory_usage'] > 90:
            severity = ErrorSeverity.CRITICAL
        elif category == ErrorCategory.RESOURCE_EXHAUSTION:
            severity = ErrorSeverity.HIGH
        elif category == ErrorCategory.DEPENDENCY_FAILURE:
            severity = ErrorSeverity.HIGH
        elif features['recovery_attempts'] > 3:
            severity = ErrorSeverity.HIGH
        elif category in [ErrorCategory.COMPUTATION_ERROR, ErrorCategory.CONFIGURATION_ERROR]:
            severity = ErrorSeverity.LOW
        
        return category, severity, confidence
    
    def _ml_classify(self, features: Dict[str, Any],
                    error_context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity, float]:
        """Machine learning-based classification."""
        
        if not self.is_trained:
            return self._rule_based_classify(features, error_context)
        
        try:
            # Prepare feature text for vectorization
            feature_text = f"{features['exception_type']} {features['message_text']}"
            feature_vector = self.vectorizer.transform([feature_text])
            
            # Predict category
            category_prediction = self.classifier.predict(feature_vector)[0]
            category = ErrorCategory(category_prediction)
            
            # Predict confidence (simplified)
            confidence = max(self.classifier.predict_proba(feature_vector)[0])
            
            # Severity based on features (simplified rule-based for now)
            if features['system_memory_usage'] > 90 or features['recovery_attempts'] > 3:
                severity = ErrorSeverity.CRITICAL
            elif category in [ErrorCategory.RESOURCE_EXHAUSTION, ErrorCategory.DEPENDENCY_FAILURE]:
                severity = ErrorSeverity.HIGH
            elif category in [ErrorCategory.COMPUTATION_ERROR, ErrorCategory.CONFIGURATION_ERROR]:
                severity = ErrorSeverity.LOW
            else:
                severity = ErrorSeverity.MEDIUM
            
            return category, severity, confidence
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to rules")
            return self._rule_based_classify(features, error_context)
    
    def learn_from_error(self, error_context: ErrorContext, actual_category: ErrorCategory):
        """Learn from a classified error to improve future classification."""
        
        if not SKLEARN_AVAILABLE:
            return
        
        # Extract features and add to training data
        features = self._extract_error_features(error_context)
        feature_text = f"{features['exception_type']} {features['message_text']}"
        
        self.error_features.append(feature_text)
        self.error_labels.append(actual_category.value)
        
        # Retrain model if we have enough data
        if len(self.error_features) >= 50 and len(self.error_features) % 10 == 0:
            self._train_classifier()
    
    def _train_classifier(self):
        """Train the error classification model."""
        
        if not SKLEARN_AVAILABLE or len(self.error_features) < 10:
            return
        
        try:
            # Vectorize features
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(self.error_features)
            
            # Train classifier
            self.classifier = KMeans(n_clusters=len(ErrorCategory), random_state=42)
            self.classifier.fit(X)
            
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            logger.info(f"Error classifier retrained with {len(self.error_features)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train error classifier: {e}")
    
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'error_features': self.error_features,
                'error_labels': self.error_labels,
                'is_trained': self.is_trained
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            logger.error(f"Failed to save error classifier: {e}")
    
    def _load_model(self):
        """Load existing model from disk."""
        try:
            if self.model_file.exists():
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data.get('vectorizer')
                self.classifier = model_data.get('classifier')
                self.error_features = model_data.get('error_features', [])
                self.error_labels = model_data.get('error_labels', [])
                self.is_trained = model_data.get('is_trained', False)
                
                logger.info("Error classifier model loaded from disk")
                
        except Exception as e:
            logger.warning(f"Failed to load error classifier: {e}")


class ComprehensiveErrorRecovery:
    """
    Main error recovery system coordinating all recovery components.
    
    Provides intelligent error handling, recovery strategy selection,
    and system state management for robust operation.
    """
    
    def __init__(self, 
                 workspace_dir: str = "/tmp/error_recovery",
                 max_recovery_attempts: int = 3,
                 checkpoint_interval: float = 300.0):  # 5 minutes
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_recovery_attempts = max_recovery_attempts
        self.checkpoint_interval = checkpoint_interval
        
        # Core components
        self.error_classifier = ErrorClassifier(self.workspace_dir)
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self._register_default_strategies()
        
        # System state management
        self.current_state = SystemState.HEALTHY
        self.checkpoints: Dict[str, SystemCheckpoint] = {}
        self.checkpoint_thread = None
        self._checkpointing_active = False
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_history: deque = deque(maxlen=1000)
        
        # Database for persistent storage
        self.db_path = self.workspace_dir / "error_recovery.db"
        self._init_database()
        
        # Cascade failure detection
        self.error_cascade_window = 60.0  # seconds
        self.cascade_threshold = 5  # errors in window to detect cascade
        
        # Recovery locks to prevent concurrent recovery attempts
        self._recovery_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Start background processes
        self.start_background_monitoring()
        
        logger.info("ComprehensiveErrorRecovery initialized")
    
    def _init_database(self):
        """Initialize SQLite database for persistent error tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_log (
                    error_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    category TEXT,
                    severity TEXT,
                    exception_type TEXT,
                    message TEXT,
                    recovery_outcome TEXT,
                    recovery_time REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_stats (
                    strategy_id TEXT PRIMARY KEY,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    total_recovery_time REAL DEFAULT 0.0,
                    last_used REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        
        # Memory cleanup strategy
        memory_cleanup = RecoveryStrategy(
            strategy_id="memory_cleanup",
            name="Memory Cleanup",
            description="Free up memory by garbage collection and cache clearing",
            error_categories=[ErrorCategory.RESOURCE_EXHAUSTION],
            severity_levels=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._memory_cleanup_recovery,
            priority=1
        )
        self.recovery_strategies["memory_cleanup"] = memory_cleanup
        
        # Restart component strategy
        restart_component = RecoveryStrategy(
            strategy_id="restart_component",
            name="Restart Component",
            description="Restart the failing component or subsystem",
            error_categories=[ErrorCategory.DEPENDENCY_FAILURE, ErrorCategory.API_FAILURE],
            severity_levels=[ErrorSeverity.HIGH, ErrorSeverity.CRITICAL],
            recovery_function=self._restart_component_recovery,
            priority=2
        )
        self.recovery_strategies["restart_component"] = restart_component
        
        # Graceful degradation strategy
        graceful_degrade = RecoveryStrategy(
            strategy_id="graceful_degradation",
            name="Graceful Degradation",
            description="Continue operation with reduced functionality",
            error_categories=[ErrorCategory.API_FAILURE, ErrorCategory.NETWORK_ERROR],
            severity_levels=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._graceful_degradation_recovery,
            priority=3
        )
        self.recovery_strategies["graceful_degradation"] = graceful_degrade
        
        # Rollback strategy
        rollback = RecoveryStrategy(
            strategy_id="rollback",
            name="Rollback to Checkpoint",
            description="Restore system to previous stable state",
            error_categories=[ErrorCategory.DATA_CORRUPTION, ErrorCategory.CONFIGURATION_ERROR],
            severity_levels=[ErrorSeverity.HIGH, ErrorSeverity.CRITICAL],
            recovery_function=self._rollback_recovery,
            priority=4
        )
        self.recovery_strategies["rollback"] = rollback
    
    def start_background_monitoring(self):
        """Start background monitoring and checkpointing."""
        if not self._checkpointing_active:
            self._checkpointing_active = True
            self.checkpoint_thread = threading.Thread(target=self._checkpoint_loop)
            self.checkpoint_thread.daemon = True
            self.checkpoint_thread.start()
            logger.info("Background monitoring started")
    
    def stop_background_monitoring(self):
        """Stop background monitoring."""
        self._checkpointing_active = False
        if self.checkpoint_thread:
            self.checkpoint_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def _checkpoint_loop(self):
        """Background checkpointing loop."""
        while self._checkpointing_active:
            try:
                self.create_checkpoint(f"auto_{int(time.time())}")
                time.sleep(self.checkpoint_interval)
            except Exception as e:
                logger.error(f"Checkpoint creation failed: {e}")
                time.sleep(60)  # Wait before retrying
    
    async def handle_error(self, 
                          exception: Exception,
                          context: Optional[Dict[str, Any]] = None) -> RecoveryOutcome:
        """
        Main error handling entry point.
        
        Args:
            exception: The exception that occurred
            context: Optional context information
            
        Returns:
            RecoveryOutcome indicating success/failure of recovery
        """
        
        # Create error context
        error_context = self._create_error_context(exception, context or {})
        
        # Log error
        self.error_history.append(error_context)
        self._log_error_to_database(error_context)
        
        logger.error(f"Handling error {error_context.error_id}: {error_context.exception_message}")
        
        # Classify error
        category, severity, confidence = self.error_classifier.classify_error(error_context)
        error_context.category = category
        error_context.severity = severity
        error_context.confidence = confidence
        
        logger.info(f"Error classified as {category.value} with severity {severity.value} (confidence: {confidence:.2f})")
        
        # Check for cascade failures
        if self._detect_cascade_failure():
            logger.warning("Cascade failure detected, escalating recovery strategy")
            severity = ErrorSeverity.CRITICAL
        
        # Update system state
        self._update_system_state(severity)
        
        # Prevent concurrent recovery for same error type
        recovery_key = f"{category.value}_{error_context.component or 'unknown'}"
        with self._recovery_locks[recovery_key]:
            
            # Attempt recovery
            recovery_outcome = await self._attempt_recovery(error_context)
            
            # Log recovery attempt
            self._log_recovery_outcome(error_context, recovery_outcome)
            
            return recovery_outcome
    
    def _create_error_context(self, 
                            exception: Exception,
                            context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        
        error_id = hashlib.md5(
            f"{type(exception).__name__}_{str(exception)}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Collect system state
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            
            resource_usage = {
                'memory_percent': memory.percent,
                'cpu_percent': cpu,
                'disk_percent': disk.percent / disk.total * 100,
                'available_memory_gb': memory.available / 1024**3
            }
        except Exception as e:
            logger.warning(f"Failed to collect resource usage: {e}")
            resource_usage = {}
        
        # Extract previous errors (last 5)
        previous_errors = [
            f"{err.category.value}:{err.exception_type}" 
            for err in list(self.error_history)[-5:]
        ]
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            system_state={'current_state': self.current_state.value},
            resource_usage=resource_usage,
            active_operations=context.get('active_operations', []),
            previous_errors=previous_errors,
            component=context.get('component'),
            operation=context.get('operation'),
            user_context=context.get('user_context', {})
        )
    
    def _detect_cascade_failure(self) -> bool:
        """Detect if we're experiencing cascade failures."""
        if len(self.error_history) < self.cascade_threshold:
            return False
        
        # Check for multiple errors in recent time window
        current_time = time.time()
        recent_errors = [
            error for error in self.error_history
            if current_time - error.timestamp <= self.error_cascade_window
        ]
        
        return len(recent_errors) >= self.cascade_threshold
    
    def _update_system_state(self, severity: ErrorSeverity):
        """Update system state based on error severity."""
        if severity == ErrorSeverity.CRITICAL:
            self.current_state = SystemState.CRITICAL
        elif severity == ErrorSeverity.HIGH and self.current_state == SystemState.HEALTHY:
            self.current_state = SystemState.DEGRADED
        elif severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]:
            if self.current_state == SystemState.CRITICAL:
                self.current_state = SystemState.RECOVERING
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> RecoveryOutcome:
        """Attempt recovery using appropriate strategies."""
        
        # Select applicable recovery strategies
        applicable_strategies = self._select_recovery_strategies(error_context)
        
        if not applicable_strategies:
            logger.warning(f"No applicable recovery strategies for error {error_context.error_id}")
            return RecoveryOutcome.FAILED
        
        # Sort by priority and success rate
        applicable_strategies.sort(key=lambda s: (s.priority, -s.success_rate))
        
        # Attempt recovery strategies
        for strategy in applicable_strategies:
            if error_context.recovery_attempts >= self.max_recovery_attempts:
                logger.error(f"Maximum recovery attempts ({self.max_recovery_attempts}) reached")
                break
            
            logger.info(f"Attempting recovery strategy: {strategy.name}")
            
            recovery_attempt = RecoveryAttempt(
                attempt_id=f"{error_context.error_id}_{strategy.strategy_id}_{int(time.time())}",
                error_id=error_context.error_id,
                strategy_id=strategy.strategy_id,
                timestamp=time.time()
            )
            
            try:
                # Capture system state before recovery
                recovery_attempt.system_state_before = self._capture_system_state()
                
                start_time = time.time()
                
                # Execute recovery strategy
                result = await self._execute_recovery_strategy(strategy, error_context)
                
                recovery_attempt.duration = time.time() - start_time
                recovery_attempt.system_state_after = self._capture_system_state()
                
                if result:
                    recovery_attempt.outcome = RecoveryOutcome.SUCCESS
                    logger.info(f"Recovery strategy '{strategy.name}' succeeded")
                    
                    # Update strategy success statistics
                    self._update_strategy_stats(strategy.strategy_id, True, recovery_attempt.duration)
                    
                    # Learn from successful recovery
                    self.error_classifier.learn_from_error(error_context, error_context.category)
                    
                    self.recovery_history.append(recovery_attempt)
                    return RecoveryOutcome.SUCCESS
                else:
                    recovery_attempt.outcome = RecoveryOutcome.FAILED
                    
            except Exception as recovery_error:
                recovery_attempt.outcome = RecoveryOutcome.FAILED
                recovery_attempt.error_occurred = str(recovery_error)
                logger.error(f"Recovery strategy '{strategy.name}' failed: {recovery_error}")
            
            # Update strategy statistics
            self._update_strategy_stats(strategy.strategy_id, False, recovery_attempt.duration)
            
            # Increment recovery attempts
            error_context.recovery_attempts += 1
            self.recovery_history.append(recovery_attempt)
        
        # All strategies failed
        return RecoveryOutcome.FAILED
    
    def _select_recovery_strategies(self, error_context: ErrorContext) -> List[RecoveryStrategy]:
        """Select applicable recovery strategies for the error."""
        applicable = []
        
        for strategy in self.recovery_strategies.values():
            # Check if strategy applies to this error category
            if error_context.category not in strategy.error_categories:
                continue
            
            # Check if strategy applies to this severity level
            if error_context.severity not in strategy.severity_levels:
                continue
            
            # Check preconditions
            if strategy.preconditions:
                try:
                    if not all(condition(error_context) for condition in strategy.preconditions):
                        continue
                except Exception as e:
                    logger.warning(f"Precondition check failed for strategy {strategy.strategy_id}: {e}")
                    continue
            
            applicable.append(strategy)
        
        return applicable
    
    async def _execute_recovery_strategy(self, 
                                       strategy: RecoveryStrategy,
                                       error_context: ErrorContext) -> bool:
        """Execute a recovery strategy."""
        try:
            if asyncio.iscoroutinefunction(strategy.recovery_function):
                result = await strategy.recovery_function(error_context)
            else:
                result = strategy.recovery_function(error_context)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            return False
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state."""
        try:
            return {
                'timestamp': time.time(),
                'memory_usage': psutil.virtual_memory()._asdict(),
                'cpu_usage': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/')._asdict(),
                'system_state': self.current_state.value,
                'active_errors': len(self.error_history),
                'checkpoints': len(self.checkpoints)
            }
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {'error': str(e)}
    
    def _update_strategy_stats(self, strategy_id: str, success: bool, duration: float):
        """Update recovery strategy statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current stats
            cursor.execute(
                "SELECT success_count, failure_count, total_recovery_time FROM recovery_stats WHERE strategy_id = ?",
                (strategy_id,)
            )
            
            row = cursor.fetchone()
            if row:
                success_count, failure_count, total_time = row
            else:
                success_count, failure_count, total_time = 0, 0, 0.0
            
            # Update stats
            if success:
                success_count += 1
            else:
                failure_count += 1
            
            total_time += duration
            
            # Update in database
            cursor.execute('''
                INSERT OR REPLACE INTO recovery_stats 
                (strategy_id, success_count, failure_count, total_recovery_time, last_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (strategy_id, success_count, failure_count, total_time, time.time()))
            
            conn.commit()
            conn.close()
            
            # Update in-memory stats
            if strategy_id in self.recovery_strategies:
                strategy = self.recovery_strategies[strategy_id]
                total_attempts = success_count + failure_count
                strategy.success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
                strategy.average_recovery_time = total_time / total_attempts if total_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to update strategy stats: {e}")
    
    def _log_error_to_database(self, error_context: ErrorContext):
        """Log error to persistent database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO error_log 
                (error_id, timestamp, category, severity, exception_type, message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                error_context.error_id,
                error_context.timestamp,
                error_context.category.value if error_context.category else 'unknown',
                error_context.severity.value if error_context.severity else 'medium',
                error_context.exception_type,
                error_context.exception_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log error to database: {e}")
    
    def _log_recovery_outcome(self, error_context: ErrorContext, outcome: RecoveryOutcome):
        """Log recovery outcome to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE error_log 
                SET recovery_outcome = ?, recovery_time = ?
                WHERE error_id = ?
            ''', (outcome.value, time.time() - error_context.timestamp, error_context.error_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log recovery outcome: {e}")
    
    def create_checkpoint(self, checkpoint_id: str) -> str:
        """Create a system checkpoint."""
        try:
            checkpoint = SystemCheckpoint(checkpoint_id, self.workspace_dir)
            
            # Save to memory
            self.checkpoints[checkpoint_id] = checkpoint
            
            # Save to disk
            checkpoint_file = self.workspace_dir / f"checkpoint_{checkpoint_id}.pkl"
            checkpoint.save_to_disk(checkpoint_file)
            
            # Limit number of stored checkpoints
            if len(self.checkpoints) > 10:
                oldest_id = min(self.checkpoints.keys(), 
                              key=lambda x: self.checkpoints[x].timestamp)
                del self.checkpoints[oldest_id]
                oldest_file = self.workspace_dir / f"checkpoint_{oldest_id}.pkl"
                if oldest_file.exists():
                    oldest_file.unlink()
            
            logger.info(f"Checkpoint '{checkpoint_id}' created successfully")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore system from checkpoint."""
        try:
            if checkpoint_id in self.checkpoints:
                checkpoint = self.checkpoints[checkpoint_id]
            else:
                # Load from disk
                checkpoint_file = self.workspace_dir / f"checkpoint_{checkpoint_id}.pkl"
                if not checkpoint_file.exists():
                    logger.error(f"Checkpoint file not found: {checkpoint_file}")
                    return False
                
                checkpoint = SystemCheckpoint.load_from_disk(checkpoint_file)
            
            # Restore application state (to be implemented by subclasses)
            self._restore_application_state(checkpoint)
            
            # Update system state
            self.current_state = SystemState.RECOVERING
            
            logger.info(f"System restored from checkpoint '{checkpoint_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def _restore_application_state(self, checkpoint: SystemCheckpoint):
        """Restore application-specific state from checkpoint."""
        # To be implemented by subclasses or extended
        pass
    
    # Default recovery strategy implementations
    
    def _memory_cleanup_recovery(self, error_context: ErrorContext) -> bool:
        """Memory cleanup recovery strategy."""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear any caches (implementation specific)
            # This would be extended based on application needs
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _restart_component_recovery(self, error_context: ErrorContext) -> bool:
        """Restart component recovery strategy."""
        try:
            component = error_context.component
            if not component:
                logger.warning("No component specified for restart recovery")
                return False
            
            # This would be implemented based on specific component architecture
            logger.info(f"Restarting component: {component}")
            
            # Placeholder implementation
            return True
            
        except Exception as e:
            logger.error(f"Component restart failed: {e}")
            return False
    
    def _graceful_degradation_recovery(self, error_context: ErrorContext) -> bool:
        """Graceful degradation recovery strategy."""
        try:
            # Switch to degraded mode
            self.current_state = SystemState.DEGRADED
            
            # Disable non-essential features
            logger.info("Switching to graceful degradation mode")
            
            # This would be implemented based on application architecture
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _rollback_recovery(self, error_context: ErrorContext) -> bool:
        """Rollback to checkpoint recovery strategy."""
        try:
            if not self.checkpoints:
                logger.warning("No checkpoints available for rollback")
                return False
            
            # Find most recent checkpoint
            latest_checkpoint_id = max(self.checkpoints.keys(),
                                     key=lambda x: self.checkpoints[x].timestamp)
            
            return self.restore_from_checkpoint(latest_checkpoint_id)
            
        except Exception as e:
            logger.error(f"Rollback recovery failed: {e}")
            return False
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        recent_errors = [
            error for error in self.error_history
            if time.time() - error.timestamp <= 3600  # Last hour
        ]
        
        recent_recoveries = [
            recovery for recovery in self.recovery_history
            if time.time() - recovery.timestamp <= 3600  # Last hour
        ]
        
        successful_recoveries = [
            recovery for recovery in recent_recoveries
            if recovery.outcome == RecoveryOutcome.SUCCESS
        ]
        
        return {
            'timestamp': time.time(),
            'system_state': self.current_state.value,
            'total_errors': len(self.error_history),
            'recent_errors_count': len(recent_errors),
            'recent_recoveries_count': len(recent_recoveries),
            'recovery_success_rate': len(successful_recoveries) / max(1, len(recent_recoveries)),
            'available_checkpoints': len(self.checkpoints),
            'registered_strategies': len(self.recovery_strategies),
            'cascade_failures_detected': self._detect_cascade_failure(),
            'error_categories': {
                category.value: sum(1 for error in recent_errors if error.category == category)
                for category in ErrorCategory
            }
        }


# Example usage and testing functions
async def test_error_recovery_system():
    """Test the comprehensive error recovery system."""
    
    # Initialize error recovery system
    recovery_system = ComprehensiveErrorRecovery(
        workspace_dir="/tmp/test_error_recovery",
        max_recovery_attempts=3
    )
    
    # Create a checkpoint
    checkpoint_id = recovery_system.create_checkpoint("test_checkpoint")
    print(f"Created checkpoint: {checkpoint_id}")
    
    # Simulate various types of errors
    test_errors = [
        MemoryError("Simulated memory exhaustion"),
        ConnectionError("Network connection failed"),
        ValueError("Invalid parameter value"),
        ImportError("Required module not found"),
        PermissionError("Access denied to resource")
    ]
    
    # Test error handling and recovery
    for error in test_errors:
        print(f"\\nTesting error recovery for: {type(error).__name__}")
        
        context = {
            'component': 'test_component',
            'operation': 'test_operation',
            'active_operations': ['op1', 'op2']
        }
        
        try:
            outcome = await recovery_system.handle_error(error, context)
            print(f"Recovery outcome: {outcome.value}")
        except Exception as e:
            print(f"Recovery failed: {e}")
    
    # Get system health report
    health_report = recovery_system.get_system_health_report()
    print(f"\\nSystem Health Report:")
    print(f"System State: {health_report['system_state']}")
    print(f"Recent Errors: {health_report['recent_errors_count']}")
    print(f"Recovery Success Rate: {health_report['recovery_success_rate']:.2f}")
    print(f"Available Checkpoints: {health_report['available_checkpoints']}")
    
    # Cleanup
    recovery_system.stop_background_monitoring()
    
    return recovery_system


if __name__ == "__main__":
    # Test the error recovery system
    asyncio.run(test_error_recovery_system())