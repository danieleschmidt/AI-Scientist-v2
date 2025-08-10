#!/usr/bin/env python3
"""
Predictive Resource Management for Autonomous Experimentation
============================================================

Intelligent experiment orchestration system using time-series forecasting and 
reinforcement learning to anticipate experiment resource needs and automatically 
scale infrastructure.

Research Hypothesis: A predictive orchestration system using time-series forecasting 
and reinforcement learning will reduce costs by 45% while maintaining research 
quality and eliminating resource bottlenecks.

Key Innovation: Time-series transformer models for resource demand forecasting with 
reinforcement learning for dynamic resource allocation decisions.

Author: AI Scientist v2 Autonomous System
License: MIT
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingAction(Enum):
    """Possible scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    MIGRATE = "migrate"


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    timestamp: float
    cpu_usage: float       # 0-1
    gpu_usage: float       # 0-1
    memory_usage: float    # 0-1
    storage_usage: float   # 0-1
    network_usage: float   # 0-1
    active_experiments: int
    queue_length: int
    cost_per_hour: float


@dataclass
class ExperimentDemand:
    """Predicted experiment resource demand."""
    experiment_id: str
    predicted_cpu: float
    predicted_gpu: float
    predicted_memory: float
    predicted_duration: float
    priority: float
    complexity_score: float
    estimated_cost: float


@dataclass
class ScalingDecision:
    """Resource scaling decision."""
    action: ScalingAction
    resource_type: ResourceType
    scale_factor: float
    confidence: float
    expected_cost_impact: float
    rationale: str
    timestamp: float = field(default_factory=time.time)


class TimeSeriesForecaster:
    """Time-series forecasting for resource demand prediction."""
    
    def __init__(self, sequence_length: int = 50, feature_dim: int = 7):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.history_buffer = deque(maxlen=1000)
        self.model_weights = self._initialize_transformer_weights()
        self.prediction_horizon = 6  # Predict 6 time steps ahead
        
    def _initialize_transformer_weights(self) -> Dict[str, np.ndarray]:
        """Initialize simplified transformer model weights."""
        # Simplified transformer-like architecture
        d_model = 64
        n_heads = 4
        
        weights = {
            'input_projection': np.random.normal(0, 0.1, (self.feature_dim, d_model)),
            'positional_encoding': self._generate_positional_encoding(self.sequence_length, d_model),
            'attention_weights': {
                f'head_{i}': {
                    'query': np.random.normal(0, 0.1, (d_model, d_model // n_heads)),
                    'key': np.random.normal(0, 0.1, (d_model, d_model // n_heads)),
                    'value': np.random.normal(0, 0.1, (d_model, d_model // n_heads))
                } for i in range(n_heads)
            },
            'feedforward': {
                'fc1': np.random.normal(0, 0.1, (d_model, d_model * 4)),
                'fc2': np.random.normal(0, 0.1, (d_model * 4, d_model))
            },
            'output_projection': np.random.normal(0, 0.1, (d_model, self.feature_dim * self.prediction_horizon))
        }
        return weights
    
    def _generate_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Generate positional encodings for transformer."""
        pos_encoding = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def add_observation(self, usage: ResourceUsage) -> None:
        """Add new resource usage observation."""
        feature_vector = np.array([
            usage.cpu_usage,
            usage.gpu_usage,
            usage.memory_usage,
            usage.storage_usage,
            usage.network_usage,
            usage.active_experiments / 10.0,  # Normalize
            usage.queue_length / 50.0  # Normalize
        ])
        
        self.history_buffer.append({
            'timestamp': usage.timestamp,
            'features': feature_vector,
            'cost': usage.cost_per_hour
        })
    
    def predict_resource_demand(self, horizon_minutes: int = 30) -> List[ResourceUsage]:
        """Predict future resource demand using transformer model."""
        if len(self.history_buffer) < self.sequence_length:
            # Not enough history, return simple heuristic prediction
            return self._heuristic_prediction(horizon_minutes)
        
        # Prepare input sequence
        recent_history = list(self.history_buffer)[-self.sequence_length:]
        input_sequence = np.array([obs['features'] for obs in recent_history])
        
        # Forward pass through simplified transformer
        predictions = self._transformer_forward(input_sequence)
        
        # Convert predictions to ResourceUsage objects
        predicted_usages = []
        current_time = time.time()
        
        for i, pred_features in enumerate(predictions):
            pred_time = current_time + (i + 1) * (horizon_minutes / len(predictions)) * 60
            
            predicted_usage = ResourceUsage(
                timestamp=pred_time,
                cpu_usage=np.clip(pred_features[0], 0, 1),
                gpu_usage=np.clip(pred_features[1], 0, 1),
                memory_usage=np.clip(pred_features[2], 0, 1),
                storage_usage=np.clip(pred_features[3], 0, 1),
                network_usage=np.clip(pred_features[4], 0, 1),
                active_experiments=max(0, int(pred_features[5] * 10)),
                queue_length=max(0, int(pred_features[6] * 50)),
                cost_per_hour=self._estimate_cost(pred_features)
            )
            predicted_usages.append(predicted_usage)
        
        return predicted_usages
    
    def _transformer_forward(self, input_sequence: np.ndarray) -> np.ndarray:
        """Simplified transformer forward pass."""
        # Input projection and positional encoding
        x = np.dot(input_sequence, self.model_weights['input_projection'])
        x = x + self.model_weights['positional_encoding']
        
        # Multi-head attention (simplified)
        attention_outputs = []
        for head_name, head_weights in self.model_weights['attention_weights'].items():
            q = np.dot(x, head_weights['query'])
            k = np.dot(x, head_weights['key'])
            v = np.dot(x, head_weights['value'])
            
            # Attention scores
            scores = np.dot(q, k.T) / np.sqrt(k.shape[-1])
            attention_weights = self._softmax(scores, axis=-1)
            attention_output = np.dot(attention_weights, v)
            attention_outputs.append(attention_output)
        
        # Concatenate heads
        attention_concat = np.concatenate(attention_outputs, axis=-1)
        
        # Feedforward network
        ff_output = self._relu(np.dot(attention_concat, self.model_weights['feedforward']['fc1']))
        ff_output = np.dot(ff_output, self.model_weights['feedforward']['fc2'])
        
        # Output projection to prediction horizon
        output = np.dot(ff_output[-1], self.model_weights['output_projection'])  # Use last timestep
        output = output.reshape(self.prediction_horizon, self.feature_dim)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _estimate_cost(self, features: np.ndarray) -> float:
        """Estimate cost based on predicted resource usage."""
        cpu_cost = features[0] * 0.10  # $0.10/hour per CPU
        gpu_cost = features[1] * 2.50  # $2.50/hour per GPU
        memory_cost = features[2] * 0.05  # $0.05/hour per GB
        storage_cost = features[3] * 0.01  # $0.01/hour per GB
        
        return cpu_cost + gpu_cost + memory_cost + storage_cost
    
    def _heuristic_prediction(self, horizon_minutes: int) -> List[ResourceUsage]:
        """Simple heuristic prediction when insufficient history."""
        if not self.history_buffer:
            # Default conservative prediction
            base_usage = ResourceUsage(
                timestamp=time.time() + horizon_minutes * 60,
                cpu_usage=0.5,
                gpu_usage=0.3,
                memory_usage=0.4,
                storage_usage=0.2,
                network_usage=0.1,
                active_experiments=3,
                queue_length=5,
                cost_per_hour=5.0
            )
            return [base_usage]
        
        # Use recent average with slight trend
        recent_obs = list(self.history_buffer)[-5:] if len(self.history_buffer) >= 5 else list(self.history_buffer)
        avg_features = np.mean([obs['features'] for obs in recent_obs], axis=0)
        
        # Apply slight upward trend for conservative estimation
        trend_factor = 1.1
        
        predicted_usage = ResourceUsage(
            timestamp=time.time() + horizon_minutes * 60,
            cpu_usage=min(avg_features[0] * trend_factor, 1.0),
            gpu_usage=min(avg_features[1] * trend_factor, 1.0),
            memory_usage=min(avg_features[2] * trend_factor, 1.0),
            storage_usage=min(avg_features[3] * trend_factor, 1.0),
            network_usage=min(avg_features[4] * trend_factor, 1.0),
            active_experiments=int(avg_features[5] * 10 * trend_factor),
            queue_length=int(avg_features[6] * 50 * trend_factor),
            cost_per_hour=self._estimate_cost(avg_features * trend_factor)
        )
        
        return [predicted_usage]


class ReinforcementLearningScaler:
    """Reinforcement learning agent for dynamic resource allocation."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # State-action value table
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.state_discretization_bins = 10
        
    def get_state_key(self, usage: ResourceUsage, demand_forecast: List[ResourceUsage]) -> str:
        """Convert continuous state to discrete state key."""
        # Current resource utilization
        current_utilization = (usage.cpu_usage + usage.gpu_usage + usage.memory_usage) / 3
        util_bin = int(current_utilization * self.state_discretization_bins)
        
        # Queue pressure
        queue_pressure = min(usage.queue_length / 20.0, 1.0)  # Normalize to [0,1]
        queue_bin = int(queue_pressure * self.state_discretization_bins)
        
        # Future demand trend
        if len(demand_forecast) > 1:
            future_util = np.mean([(d.cpu_usage + d.gpu_usage + d.memory_usage) / 3 
                                  for d in demand_forecast])
            trend = future_util - current_utilization
            trend_bin = int((trend + 1) * self.state_discretization_bins / 2)  # Map [-1,1] to [0,10]
        else:
            trend_bin = 5  # Neutral
        
        # Cost pressure (current cost vs budget)
        cost_bin = min(int(usage.cost_per_hour / 10.0 * self.state_discretization_bins), 
                      self.state_discretization_bins - 1)
        
        return f"{util_bin}_{queue_bin}_{trend_bin}_{cost_bin}"
    
    def select_action(self, state_key: str, available_actions: List[ScalingAction]) -> ScalingAction:
        """Select action using epsilon-greedy policy."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            # Select action with highest Q-value
            action_values = self.q_table[state_key]
            best_action = max(action_values, key=action_values.get)
            return best_action
    
    def update_q_value(self, state_key: str, action: ScalingAction, 
                      reward: float, next_state_key: str) -> None:
        """Update Q-value using Q-learning update rule."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ScalingAction}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in ScalingAction}
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        
        updated_q = current_q + self.learning_rate * (reward + 0.95 * max_next_q - current_q)
        self.q_table[state_key][action] = updated_q
        
        # Store experience
        self.action_history.append({
            'state': state_key,
            'action': action,
            'reward': reward,
            'next_state': next_state_key,
            'timestamp': time.time()
        })
        self.reward_history.append(reward)
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.999)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        if not self.reward_history:
            return {'message': 'No learning history available'}
        
        recent_rewards = list(self.reward_history)[-100:]  # Last 100 rewards
        
        return {
            'total_actions': len(self.action_history),
            'average_reward': np.mean(self.reward_history),
            'recent_average_reward': np.mean(recent_rewards),
            'exploration_rate': self.epsilon,
            'q_table_size': len(self.q_table),
            'learning_trend': self._compute_learning_trend()
        }
    
    def _compute_learning_trend(self) -> float:
        """Compute learning improvement trend."""
        if len(self.reward_history) < 20:
            return 0.0
        
        # Compare first and last quartiles
        first_quartile = list(self.reward_history)[:len(self.reward_history)//4]
        last_quartile = list(self.reward_history)[-len(self.reward_history)//4:]
        
        improvement = np.mean(last_quartile) - np.mean(first_quartile)
        return improvement


class PredictiveResourceManager:
    """Main orchestrator for predictive resource management."""
    
    def __init__(self):
        self.forecaster = TimeSeriesForecaster()
        self.rl_scaler = ReinforcementLearningScaler()
        self.current_resources = {
            ResourceType.CPU: 1.0,      # Current allocation factor
            ResourceType.GPU: 1.0,
            ResourceType.MEMORY: 1.0,
            ResourceType.STORAGE: 1.0,
            ResourceType.NETWORK: 1.0
        }
        self.cost_budget = 1000.0  # Budget per hour
        self.performance_threshold = 0.9  # Minimum performance target
        self.monitoring_active = False
        self.monitoring_thread = None
        self.scaling_decisions = deque(maxlen=1000)
        
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous resource monitoring and prediction."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started predictive resource monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped predictive resource monitoring")
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Continuous monitoring and scaling loop."""
        while self.monitoring_active:
            try:
                # Get current resource usage
                current_usage = self._get_current_usage()
                
                # Add observation to forecaster
                self.forecaster.add_observation(current_usage)
                
                # Generate demand forecast
                demand_forecast = self.forecaster.predict_resource_demand(horizon_minutes=30)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(current_usage, demand_forecast)
                
                # Execute scaling if needed
                if scaling_decision.action != ScalingAction.MAINTAIN:
                    self._execute_scaling(scaling_decision)
                
                # Update RL agent based on outcome
                self._update_rl_agent(current_usage, scaling_decision, demand_forecast)
                
                # Store decision
                self.scaling_decisions.append(scaling_decision)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _get_current_usage(self) -> ResourceUsage:
        """Get current resource usage (mock implementation)."""
        # Mock current usage with realistic patterns
        base_time = time.time() % (24 * 3600)  # Time of day
        daily_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * base_time / (24 * 3600))  # Daily usage pattern
        
        # Add some noise and randomness
        noise = np.random.normal(0, 0.1)
        
        return ResourceUsage(
            timestamp=time.time(),
            cpu_usage=np.clip(daily_pattern + noise, 0.1, 0.95),
            gpu_usage=np.clip(daily_pattern * 0.8 + noise, 0.05, 0.90),
            memory_usage=np.clip(daily_pattern * 1.1 + noise, 0.2, 0.85),
            storage_usage=np.clip(0.3 + noise * 0.1, 0.1, 0.8),
            network_usage=np.clip(daily_pattern * 0.6 + noise, 0.05, 0.7),
            active_experiments=max(1, int(daily_pattern * 10 + np.random.poisson(3))),
            queue_length=max(0, int(np.random.poisson(daily_pattern * 8))),
            cost_per_hour=self._calculate_current_cost()
        )
    
    def _calculate_current_cost(self) -> float:
        """Calculate current hourly cost based on resource allocation."""
        cost = 0.0
        cost += self.current_resources[ResourceType.CPU] * 2.0      # $2/hour per CPU unit
        cost += self.current_resources[ResourceType.GPU] * 15.0     # $15/hour per GPU unit
        cost += self.current_resources[ResourceType.MEMORY] * 0.5   # $0.5/hour per memory unit
        cost += self.current_resources[ResourceType.STORAGE] * 0.1  # $0.1/hour per storage unit
        cost += self.current_resources[ResourceType.NETWORK] * 0.05 # $0.05/hour per network unit
        
        return cost
    
    def _make_scaling_decision(self, current_usage: ResourceUsage, 
                             demand_forecast: List[ResourceUsage]) -> ScalingDecision:
        """Make intelligent scaling decision using RL agent."""
        # Get state representation
        state_key = self.rl_scaler.get_state_key(current_usage, demand_forecast)
        
        # Determine available actions based on current state
        available_actions = self._get_available_actions(current_usage)
        
        # Select action using RL agent
        selected_action = self.rl_scaler.select_action(state_key, available_actions)
        
        # Determine resource type to scale (heuristic selection)
        resource_to_scale = self._select_resource_to_scale(current_usage, demand_forecast)
        
        # Calculate scale factor
        scale_factor = self._calculate_scale_factor(selected_action, current_usage, demand_forecast)
        
        # Estimate confidence and cost impact
        confidence = self._estimate_confidence(current_usage, demand_forecast)
        cost_impact = self._estimate_cost_impact(selected_action, resource_to_scale, scale_factor)
        
        # Generate rationale
        rationale = self._generate_rationale(selected_action, current_usage, demand_forecast)
        
        return ScalingDecision(
            action=selected_action,
            resource_type=resource_to_scale,
            scale_factor=scale_factor,
            confidence=confidence,
            expected_cost_impact=cost_impact,
            rationale=rationale
        )
    
    def _get_available_actions(self, usage: ResourceUsage) -> List[ScalingAction]:
        """Determine available scaling actions based on current state."""
        actions = [ScalingAction.MAINTAIN]
        
        # Check if scaling up is feasible (budget and utilization constraints)
        if usage.cost_per_hour < self.cost_budget * 0.8:  # Within 80% of budget
            actions.append(ScalingAction.SCALE_UP)
        
        # Check if scaling down is feasible (not if resources are already minimal)
        current_total_allocation = sum(self.current_resources.values())
        if current_total_allocation > 2.0:  # More than minimal allocation
            actions.append(ScalingAction.SCALE_DOWN)
        
        return actions
    
    def _select_resource_to_scale(self, usage: ResourceUsage, 
                                forecast: List[ResourceUsage]) -> ResourceType:
        """Select which resource type to scale based on usage patterns."""
        # Calculate resource pressure scores
        cpu_pressure = usage.cpu_usage
        gpu_pressure = usage.gpu_usage
        memory_pressure = usage.memory_usage
        
        # Consider forecast trends
        if forecast:
            avg_future_cpu = np.mean([f.cpu_usage for f in forecast])
            avg_future_gpu = np.mean([f.gpu_usage for f in forecast])
            avg_future_memory = np.mean([f.memory_usage for f in forecast])
            
            cpu_pressure = 0.7 * cpu_pressure + 0.3 * avg_future_cpu
            gpu_pressure = 0.7 * gpu_pressure + 0.3 * avg_future_gpu
            memory_pressure = 0.7 * memory_pressure + 0.3 * avg_future_memory
        
        # Select resource with highest pressure
        pressures = {
            ResourceType.CPU: cpu_pressure,
            ResourceType.GPU: gpu_pressure,
            ResourceType.MEMORY: memory_pressure,
            ResourceType.STORAGE: usage.storage_usage,
            ResourceType.NETWORK: usage.network_usage
        }
        
        return max(pressures, key=pressures.get)
    
    def _calculate_scale_factor(self, action: ScalingAction, usage: ResourceUsage, 
                              forecast: List[ResourceUsage]) -> float:
        """Calculate appropriate scale factor for the action."""
        if action == ScalingAction.MAINTAIN:
            return 1.0
        elif action == ScalingAction.SCALE_UP:
            # Scale up based on demand forecast
            if forecast:
                max_future_demand = max(
                    max([f.cpu_usage, f.gpu_usage, f.memory_usage]) for f in forecast
                )
                scale_factor = min(max_future_demand / 0.7, 2.0)  # Cap at 2x
            else:
                scale_factor = 1.3  # Conservative scale-up
        elif action == ScalingAction.SCALE_DOWN:
            # Scale down based on current utilization
            current_max_usage = max([usage.cpu_usage, usage.gpu_usage, usage.memory_usage])
            if current_max_usage < 0.5:
                scale_factor = 0.8  # Scale down to 80%
            else:
                scale_factor = 0.9  # Conservative scale-down
        else:
            scale_factor = 1.0
        
        return max(0.1, min(scale_factor, 3.0))  # Bound scale factor
    
    def _estimate_confidence(self, usage: ResourceUsage, forecast: List[ResourceUsage]) -> float:
        """Estimate confidence in scaling decision."""
        base_confidence = 0.7
        
        # Higher confidence with more historical data
        history_bonus = min(len(self.forecaster.history_buffer) / 100.0, 0.2)
        
        # Lower confidence with high variability in forecast
        if len(forecast) > 1:
            forecast_variability = np.std([f.cpu_usage for f in forecast])
            variability_penalty = min(forecast_variability * 0.5, 0.3)
        else:
            variability_penalty = 0.1
        
        confidence = base_confidence + history_bonus - variability_penalty
        return np.clip(confidence, 0.1, 0.95)
    
    def _estimate_cost_impact(self, action: ScalingAction, resource_type: ResourceType, 
                            scale_factor: float) -> float:
        """Estimate cost impact of scaling decision."""
        if action == ScalingAction.MAINTAIN:
            return 0.0
        
        # Base cost per resource type per hour
        resource_costs = {
            ResourceType.CPU: 2.0,
            ResourceType.GPU: 15.0,
            ResourceType.MEMORY: 0.5,
            ResourceType.STORAGE: 0.1,
            ResourceType.NETWORK: 0.05
        }
        
        base_cost = resource_costs[resource_type] * self.current_resources[resource_type]
        new_cost = resource_costs[resource_type] * self.current_resources[resource_type] * scale_factor
        
        return new_cost - base_cost
    
    def _generate_rationale(self, action: ScalingAction, usage: ResourceUsage, 
                          forecast: List[ResourceUsage]) -> str:
        """Generate human-readable rationale for scaling decision."""
        if action == ScalingAction.MAINTAIN:
            return "Current resource levels are adequate for predicted demand"
        elif action == ScalingAction.SCALE_UP:
            if forecast and max(f.cpu_usage for f in forecast) > 0.8:
                return f"High demand predicted (CPU: {max(f.cpu_usage for f in forecast):.2f}), scaling up preemptively"
            elif usage.queue_length > 10:
                return f"Queue backlog detected ({usage.queue_length} experiments), increasing capacity"
            else:
                return "Proactive scaling to handle increased workload"
        elif action == ScalingAction.SCALE_DOWN:
            current_max = max(usage.cpu_usage, usage.gpu_usage, usage.memory_usage)
            return f"Low utilization detected ({current_max:.2f}), optimizing costs"
        else:
            return "Scaling decision based on current system state"
    
    def _execute_scaling(self, decision: ScalingDecision) -> None:
        """Execute the scaling decision."""
        if decision.action == ScalingAction.MAINTAIN:
            return
        
        old_allocation = self.current_resources[decision.resource_type]
        new_allocation = old_allocation * decision.scale_factor
        
        # Apply bounds checking
        new_allocation = max(0.1, min(new_allocation, 5.0))
        
        self.current_resources[decision.resource_type] = new_allocation
        
        logger.info(f"Executed {decision.action.value} for {decision.resource_type.value}: "
                   f"{old_allocation:.2f} → {new_allocation:.2f} "
                   f"(confidence: {decision.confidence:.2f}, cost impact: ${decision.expected_cost_impact:.2f})")
    
    def _update_rl_agent(self, usage: ResourceUsage, decision: ScalingDecision, 
                        forecast: List[ResourceUsage]) -> None:
        """Update RL agent based on scaling outcome."""
        # Calculate reward based on multiple factors
        reward = self._calculate_scaling_reward(usage, decision, forecast)
        
        # Get current and next state
        current_state = self.rl_scaler.get_state_key(usage, forecast)
        
        # Mock next state (would be actual next observation in real system)
        next_usage = self._simulate_next_usage(usage, decision)
        next_state = self.rl_scaler.get_state_key(next_usage, forecast)
        
        # Update Q-value
        self.rl_scaler.update_q_value(current_state, decision.action, reward, next_state)
    
    def _calculate_scaling_reward(self, usage: ResourceUsage, decision: ScalingDecision, 
                                forecast: List[ResourceUsage]) -> float:
        """Calculate reward for scaling decision."""
        reward = 0.0
        
        # Performance reward (negative if performance degrades)
        current_performance = 1.0 - max(usage.cpu_usage, usage.gpu_usage, usage.memory_usage)
        if current_performance > self.performance_threshold:
            reward += 10.0
        else:
            reward -= 20.0 * (self.performance_threshold - current_performance)
        
        # Cost efficiency reward
        cost_efficiency = 1.0 - (usage.cost_per_hour / self.cost_budget)
        reward += 5.0 * cost_efficiency
        
        # Queue management reward
        if usage.queue_length == 0:
            reward += 5.0
        else:
            reward -= 2.0 * usage.queue_length
        
        # Scaling appropriateness reward
        if decision.action == ScalingAction.SCALE_UP and usage.cpu_usage > 0.8:
            reward += 5.0  # Good preemptive scaling
        elif decision.action == ScalingAction.SCALE_DOWN and usage.cpu_usage < 0.3:
            reward += 3.0  # Good cost optimization
        elif decision.action == ScalingAction.MAINTAIN and 0.3 <= usage.cpu_usage <= 0.8:
            reward += 2.0  # Good steady-state management
        
        return reward
    
    def _simulate_next_usage(self, current_usage: ResourceUsage, 
                           decision: ScalingDecision) -> ResourceUsage:
        """Simulate next usage state after scaling decision (mock)."""
        # Simple simulation of scaling effects
        usage_factor = 1.0
        if decision.action == ScalingAction.SCALE_UP:
            usage_factor = 0.8  # Resources become less utilized
        elif decision.action == ScalingAction.SCALE_DOWN:
            usage_factor = 1.2  # Resources become more utilized
        
        return ResourceUsage(
            timestamp=time.time() + 60,  # 1 minute later
            cpu_usage=min(current_usage.cpu_usage * usage_factor, 1.0),
            gpu_usage=min(current_usage.gpu_usage * usage_factor, 1.0),
            memory_usage=min(current_usage.memory_usage * usage_factor, 1.0),
            storage_usage=current_usage.storage_usage,
            network_usage=current_usage.network_usage,
            active_experiments=current_usage.active_experiments,
            queue_length=max(0, current_usage.queue_length - 1),  # Assume queue progresses
            cost_per_hour=self._calculate_current_cost()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance metrics."""
        current_usage = self._get_current_usage()
        
        # Get recent scaling decisions
        recent_decisions = list(self.scaling_decisions)[-10:] if self.scaling_decisions else []
        
        # Forecast next 30 minutes
        forecast = self.forecaster.predict_resource_demand(30)
        
        # Get RL agent performance
        rl_metrics = self.rl_scaler.get_performance_metrics()
        
        return {
            'monitoring_status': {
                'active': self.monitoring_active,
                'current_cost_per_hour': current_usage.cost_per_hour,
                'cost_budget': self.cost_budget,
                'budget_utilization': current_usage.cost_per_hour / self.cost_budget
            },
            'current_resources': {
                resource.value: allocation 
                for resource, allocation in self.current_resources.items()
            },
            'current_usage': {
                'cpu': current_usage.cpu_usage,
                'gpu': current_usage.gpu_usage,
                'memory': current_usage.memory_usage,
                'storage': current_usage.storage_usage,
                'network': current_usage.network_usage,
                'active_experiments': current_usage.active_experiments,
                'queue_length': current_usage.queue_length
            },
            'demand_forecast': [
                {
                    'time_offset_minutes': i * 6,  # Every 6 minutes
                    'predicted_cpu': f.cpu_usage,
                    'predicted_gpu': f.gpu_usage,
                    'predicted_cost': f.cost_per_hour
                } for i, f in enumerate(forecast)
            ],
            'recent_scaling_decisions': [
                {
                    'action': d.action.value,
                    'resource': d.resource_type.value,
                    'scale_factor': d.scale_factor,
                    'confidence': d.confidence,
                    'cost_impact': d.expected_cost_impact,
                    'rationale': d.rationale
                } for d in recent_decisions
            ],
            'rl_agent_performance': rl_metrics,
            'cost_optimization_metrics': self._compute_cost_metrics()
        }
    
    def _compute_cost_metrics(self) -> Dict[str, float]:
        """Compute cost optimization metrics."""
        if not self.scaling_decisions:
            return {'message': 'No scaling history available'}
        
        # Calculate cost savings from scaling decisions
        total_cost_impact = sum(d.expected_cost_impact for d in self.scaling_decisions)
        num_scale_downs = sum(1 for d in self.scaling_decisions if d.action == ScalingAction.SCALE_DOWN)
        num_scale_ups = sum(1 for d in self.scaling_decisions if d.action == ScalingAction.SCALE_UP)
        
        # Estimate cost savings percentage
        baseline_cost = self.cost_budget * 0.8  # Assume 80% baseline utilization
        actual_cost = baseline_cost + total_cost_impact
        cost_savings_pct = max(0, (baseline_cost - actual_cost) / baseline_cost * 100)
        
        return {
            'total_cost_impact': total_cost_impact,
            'estimated_cost_savings_percent': cost_savings_pct,
            'scale_up_decisions': num_scale_ups,
            'scale_down_decisions': num_scale_downs,
            'average_decision_confidence': np.mean([d.confidence for d in self.scaling_decisions])
        }


# Example usage and testing framework
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Predictive Resource Management Demo ===\n")
    
    # Initialize predictive resource manager
    manager = PredictiveResourceManager()
    
    # Simulate resource usage for demonstration
    print("Simulating resource usage and scaling decisions...\n")
    
    # Add some historical data
    for i in range(20):
        usage = manager._get_current_usage()
        manager.forecaster.add_observation(usage)
        time.sleep(0.1)  # Small delay to show progression
    
    # Get demand forecast
    forecast = manager.forecaster.predict_resource_demand(horizon_minutes=30)
    print(f"Generated demand forecast for next 30 minutes: {len(forecast)} data points")
    
    # Test scaling decisions
    current_usage = manager._get_current_usage()
    print(f"\nCurrent resource usage:")
    print(f"  CPU: {current_usage.cpu_usage:.2f}")
    print(f"  GPU: {current_usage.gpu_usage:.2f}")
    print(f"  Memory: {current_usage.memory_usage:.2f}")
    print(f"  Queue length: {current_usage.queue_length}")
    print(f"  Cost/hour: ${current_usage.cost_per_hour:.2f}")
    
    # Make scaling decision
    decision = manager._make_scaling_decision(current_usage, forecast)
    print(f"\nScaling decision:")
    print(f"  Action: {decision.action.value}")
    print(f"  Resource: {decision.resource_type.value}")
    print(f"  Scale factor: {decision.scale_factor:.2f}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Cost impact: ${decision.expected_cost_impact:.2f}")
    print(f"  Rationale: {decision.rationale}")
    
    # Start monitoring for a brief period
    print("\nStarting monitoring for 10 seconds...")
    manager.start_monitoring(interval_seconds=2)
    time.sleep(10)
    manager.stop_monitoring()
    
    # Get final system status
    status = manager.get_system_status()
    print(f"\n=== Final System Status ===")
    print(f"Budget utilization: {status['monitoring_status']['budget_utilization']:.1%}")
    print(f"Recent scaling decisions: {len(status['recent_scaling_decisions'])}")
    
    if 'estimated_cost_savings_percent' in status['cost_optimization_metrics']:
        print(f"Estimated cost savings: {status['cost_optimization_metrics']['estimated_cost_savings_percent']:.1f}%")
    
    print(f"RL agent learning trend: {status['rl_agent_performance'].get('learning_trend', 0):.3f}")
    print("\nPredictive resource management demo completed!")