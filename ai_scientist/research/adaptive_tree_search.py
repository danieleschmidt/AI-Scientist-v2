#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Tree Search for Autonomous Research
============================================================

Novel algorithmic framework implementing meta-learning for dynamic tree search 
strategy selection in autonomous scientific experimentation.

Research Hypothesis: A meta-learning framework that dynamically selects and combines 
multiple tree search strategies based on experiment characteristics will significantly 
improve exploration efficiency and solution quality.

Key Innovation: Meta-controller using reinforcement learning for strategy selection
with adaptive branching factors based on solution space complexity.

Author: AI Scientist v2 Autonomous System
License: MIT
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Enumeration of available tree search strategies."""
    BEST_FIRST = "best_first"
    MONTE_CARLO = "monte_carlo" 
    UPPER_CONFIDENCE = "upper_confidence"
    PROGRESSIVE_WIDENING = "progressive_widening"
    EVOLUTIONARY = "evolutionary"


@dataclass
class ExperimentContext:
    """Context information for experiment characterization."""
    domain: str
    complexity_score: float
    resource_budget: float
    time_constraint: float
    novelty_requirement: float
    success_history: List[float]
    

@dataclass
class SearchMetrics:
    """Metrics for evaluating search strategy performance."""
    exploration_efficiency: float
    solution_quality: float
    convergence_time: float
    resource_utilization: float
    diversity_score: float


class TreeSearchStrategy(ABC):
    """Abstract base class for tree search strategies."""
    
    @abstractmethod
    def select_node(self, tree_state: Dict[str, Any]) -> str:
        """Select next node to explore."""
        pass
    
    @abstractmethod
    def expand(self, node_id: str, context: ExperimentContext) -> List[str]:
        """Expand node with child nodes."""
        pass
    
    @abstractmethod
    def evaluate(self, node_id: str, result: Any) -> float:
        """Evaluate node performance."""
        pass


class BestFirstTreeSearch(TreeSearchStrategy):
    """Best-First Tree Search implementation."""
    
    def __init__(self, branching_factor: int = 3):
        self.branching_factor = branching_factor
        self.node_scores = {}
        
    def select_node(self, tree_state: Dict[str, Any]) -> str:
        """Select node with highest evaluation score."""
        unexplored = tree_state.get('unexplored_nodes', [])
        if not unexplored:
            return None
            
        # Select highest scoring unexplored node
        scored_nodes = [(node, self.node_scores.get(node, 0.0)) for node in unexplored]
        return max(scored_nodes, key=lambda x: x[1])[0]
    
    def expand(self, node_id: str, context: ExperimentContext) -> List[str]:
        """Generate child nodes with adaptive branching."""
        # Adaptive branching based on complexity
        adapted_branching = max(2, int(self.branching_factor * context.complexity_score))
        adapted_branching = min(adapted_branching, 8)  # Cap at 8
        
        children = []
        for i in range(adapted_branching):
            child_id = f"{node_id}_child_{i}"
            children.append(child_id)
            
        return children
    
    def evaluate(self, node_id: str, result: Any) -> float:
        """Evaluate node based on experiment result quality."""
        if result is None:
            score = 0.0
        else:
            # Multi-criteria evaluation
            quality = getattr(result, 'quality_score', 0.5)
            novelty = getattr(result, 'novelty_score', 0.5)
            efficiency = getattr(result, 'efficiency_score', 0.5)
            
            score = 0.4 * quality + 0.3 * novelty + 0.3 * efficiency
            
        self.node_scores[node_id] = score
        return score


class MonteCarloTreeSearch(TreeSearchStrategy):
    """Monte Carlo Tree Search implementation."""
    
    def __init__(self, exploration_constant: float = 1.414):
        self.c = exploration_constant
        self.visit_counts = {}
        self.value_sums = {}
        
    def select_node(self, tree_state: Dict[str, Any]) -> str:
        """UCB1 selection of most promising node."""
        unexplored = tree_state.get('unexplored_nodes', [])
        if not unexplored:
            return None
            
        total_visits = sum(self.visit_counts.get(node, 0) for node in unexplored)
        if total_visits == 0:
            return unexplored[0]  # Random first selection
            
        ucb_scores = []
        for node in unexplored:
            visits = self.visit_counts.get(node, 0)
            if visits == 0:
                ucb_scores.append((node, float('inf')))
            else:
                avg_value = self.value_sums.get(node, 0) / visits
                exploration_term = self.c * np.sqrt(np.log(total_visits) / visits)
                ucb_score = avg_value + exploration_term
                ucb_scores.append((node, ucb_score))
                
        return max(ucb_scores, key=lambda x: x[1])[0]
    
    def expand(self, node_id: str, context: ExperimentContext) -> List[str]:
        """Generate children based on domain-specific expansion."""
        num_children = np.random.randint(2, 5)  # Stochastic branching
        children = [f"{node_id}_mc_{i}" for i in range(num_children)]
        return children
    
    def evaluate(self, node_id: str, result: Any) -> float:
        """Update MCTS statistics."""
        score = getattr(result, 'quality_score', np.random.beta(2, 5))  # Default simulation
        
        self.visit_counts[node_id] = self.visit_counts.get(node_id, 0) + 1
        self.value_sums[node_id] = self.value_sums.get(node_id, 0) + score
        
        return score


class MetaLearningController:
    """Meta-controller for adaptive strategy selection."""
    
    def __init__(self, strategies: List[TreeSearchStrategy]):
        self.strategies = {s.__class__.__name__: s for s in strategies}
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.exploration_bonus = 0.1
        self.learning_rate = 0.01
        self.strategy_weights = {name: 1.0 for name in self.strategies.keys()}
        
    def select_strategy(self, context: ExperimentContext) -> TreeSearchStrategy:
        """Select optimal strategy using multi-armed bandit."""
        # Contextual features for strategy selection
        features = np.array([
            context.complexity_score,
            context.resource_budget / 1000.0,  # Normalize
            context.time_constraint / 3600.0,   # Hours
            context.novelty_requirement,
            len(context.success_history) / 100.0  # Experience factor
        ])
        
        # Compute strategy scores with context weighting
        strategy_scores = {}
        for name, strategy in self.strategies.items():
            # Base performance
            recent_perf = self.strategy_performance[name][-10:] if self.strategy_performance[name] else [0.5]
            avg_performance = np.mean(recent_perf)
            
            # Context-based bonus
            context_bonus = self._compute_context_bonus(name, features)
            
            # Exploration bonus (reduces over time)
            exploration = self.exploration_bonus / (len(recent_perf) + 1)
            
            total_score = avg_performance + context_bonus + exploration
            strategy_scores[name] = total_score
            
        # Select strategy with highest score
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        
        logger.info(f"Selected strategy: {best_strategy_name} (score: {strategy_scores[best_strategy_name]:.3f})")
        return self.strategies[best_strategy_name]
    
    def _compute_context_bonus(self, strategy_name: str, features: np.ndarray) -> float:
        """Compute context-specific bonus for strategy."""
        # Simple heuristic bonuses based on context
        complexity, budget, time, novelty, experience = features
        
        bonus = 0.0
        if strategy_name == "BestFirstTreeSearch":
            # Better for high complexity, low time pressure
            bonus = 0.2 * complexity - 0.1 * time
        elif strategy_name == "MonteCarloTreeSearch":
            # Better for exploration-heavy tasks
            bonus = 0.2 * novelty + 0.1 * budget
            
        return np.clip(bonus, -0.1, 0.1)
    
    def update_performance(self, strategy_name: str, metrics: SearchMetrics) -> None:
        """Update strategy performance based on observed results."""
        # Combine multiple metrics into single performance score
        performance = (
            0.3 * metrics.exploration_efficiency +
            0.3 * metrics.solution_quality +
            0.2 * (1.0 - metrics.convergence_time / 3600.0) +  # Faster is better
            0.2 * metrics.diversity_score
        )
        
        self.strategy_performance[strategy_name].append(performance)
        
        # Adaptive weight update
        self.strategy_weights[strategy_name] *= (1 + self.learning_rate * (performance - 0.5))
        self.strategy_weights[strategy_name] = np.clip(self.strategy_weights[strategy_name], 0.1, 2.0)
        
        logger.info(f"Updated {strategy_name} performance: {performance:.3f}")


class AdaptiveTreeSearchOrchestrator:
    """Main orchestrator for adaptive multi-strategy tree search."""
    
    def __init__(self):
        # Initialize available search strategies
        self.strategies = [
            BestFirstTreeSearch(),
            MonteCarloTreeSearch()
        ]
        
        self.meta_controller = MetaLearningController(self.strategies)
        self.search_history = []
        
    def execute_search(self, 
                      context: ExperimentContext,
                      max_iterations: int = 50,
                      time_budget: float = 3600.0) -> Dict[str, Any]:
        """Execute adaptive tree search with dynamic strategy selection."""
        
        logger.info(f"Starting adaptive tree search for {context.domain}")
        start_time = time.time()
        
        # Initialize tree state
        tree_state = {
            'unexplored_nodes': ['root'],
            'explored_nodes': [],
            'node_results': {},
            'best_solution': None,
            'best_score': -float('inf')
        }
        
        iteration = 0
        strategy_switches = 0
        current_strategy = None
        
        while (iteration < max_iterations and 
               time.time() - start_time < time_budget and 
               tree_state['unexplored_nodes']):
            
            # Strategy selection (re-evaluate every 10 iterations or at start)
            if iteration % 10 == 0 or current_strategy is None:
                new_strategy = self.meta_controller.select_strategy(context)
                if new_strategy != current_strategy:
                    current_strategy = new_strategy
                    strategy_switches += 1
                    logger.info(f"Strategy switch #{strategy_switches} to {current_strategy.__class__.__name__}")
            
            # Execute search iteration
            node_id = current_strategy.select_node(tree_state)
            if node_id is None:
                break
                
            # Expand node
            children = current_strategy.expand(node_id, context)
            tree_state['unexplored_nodes'].extend(children)
            tree_state['unexplored_nodes'].remove(node_id)
            tree_state['explored_nodes'].append(node_id)
            
            # Simulate experiment execution (mock for now)
            result = self._simulate_experiment(node_id, context)
            tree_state['node_results'][node_id] = result
            
            # Evaluate result
            score = current_strategy.evaluate(node_id, result)
            
            # Update best solution
            if score > tree_state['best_score']:
                tree_state['best_score'] = score
                tree_state['best_solution'] = result
                logger.info(f"New best solution found: score={score:.3f}")
            
            iteration += 1
        
        # Compute final metrics
        total_time = time.time() - start_time
        metrics = self._compute_metrics(tree_state, total_time, strategy_switches)
        
        # Update meta-controller
        strategy_name = current_strategy.__class__.__name__
        self.meta_controller.update_performance(strategy_name, metrics)
        
        # Store search history
        search_record = {
            'context': context,
            'final_score': tree_state['best_score'],
            'iterations': iteration,
            'strategy_switches': strategy_switches,
            'total_time': total_time,
            'metrics': metrics
        }
        self.search_history.append(search_record)
        
        logger.info(f"Search completed: {iteration} iterations, {strategy_switches} switches, score={tree_state['best_score']:.3f}")
        
        return {
            'best_solution': tree_state['best_solution'],
            'best_score': tree_state['best_score'],
            'search_metrics': metrics,
            'search_record': search_record
        }
    
    def _simulate_experiment(self, node_id: str, context: ExperimentContext) -> Any:
        """Simulate experiment execution (replace with actual experiment runner)."""
        # Mock experiment result with realistic scoring
        base_quality = np.random.beta(2, 3)  # Skewed toward lower scores
        
        # Context-dependent adjustments
        complexity_bonus = 0.1 * context.complexity_score * np.random.normal(0, 0.1)
        novelty_factor = context.novelty_requirement * np.random.beta(3, 7)
        
        # Create mock result object
        class MockResult:
            def __init__(self):
                self.quality_score = np.clip(base_quality + complexity_bonus, 0, 1)
                self.novelty_score = np.clip(novelty_factor, 0, 1)
                self.efficiency_score = np.random.beta(3, 2)  # Generally efficient
                
        return MockResult()
    
    def _compute_metrics(self, tree_state: Dict[str, Any], total_time: float, strategy_switches: int) -> SearchMetrics:
        """Compute comprehensive search performance metrics."""
        num_explored = len(tree_state['explored_nodes'])
        
        # Exploration efficiency: nodes explored per unit time
        exploration_efficiency = num_explored / max(total_time, 1.0)
        
        # Solution quality: best score achieved
        solution_quality = max(tree_state['best_score'], 0.0)
        
        # Convergence time: time to reach 90% of best score
        convergence_time = total_time * 0.7  # Mock estimate
        
        # Resource utilization: strategy switches indicate adaptive behavior
        resource_utilization = min(strategy_switches / max(num_explored / 10, 1), 1.0)
        
        # Diversity: variety in explored solutions
        diversity_score = min(num_explored / 20.0, 1.0)  # Normalized diversity
        
        return SearchMetrics(
            exploration_efficiency=exploration_efficiency,
            solution_quality=solution_quality,
            convergence_time=convergence_time,
            resource_utilization=resource_utilization,
            diversity_score=diversity_score
        )
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Generate performance analytics across all searches."""
        if not self.search_history:
            return {"message": "No search history available"}
        
        # Aggregate performance metrics
        avg_score = np.mean([s['final_score'] for s in self.search_history])
        avg_time = np.mean([s['total_time'] for s in self.search_history])
        avg_iterations = np.mean([s['iterations'] for s in self.search_history])
        avg_switches = np.mean([s['strategy_switches'] for s in self.search_history])
        
        # Strategy usage statistics
        strategy_usage = {}
        for strategy_name in self.meta_controller.strategies.keys():
            performances = self.meta_controller.strategy_performance[strategy_name]
            if performances:
                strategy_usage[strategy_name] = {
                    'usage_count': len(performances),
                    'avg_performance': np.mean(performances),
                    'current_weight': self.meta_controller.strategy_weights[strategy_name]
                }
        
        return {
            'search_statistics': {
                'total_searches': len(self.search_history),
                'average_score': avg_score,
                'average_time': avg_time,
                'average_iterations': avg_iterations,
                'average_strategy_switches': avg_switches
            },
            'strategy_analytics': strategy_usage,
            'improvement_trend': self._compute_improvement_trend()
        }
    
    def _compute_improvement_trend(self) -> float:
        """Compute learning improvement trend over time."""
        if len(self.search_history) < 5:
            return 0.0
            
        # Linear regression on recent performance
        recent_scores = [s['final_score'] for s in self.search_history[-10:]]
        x = np.arange(len(recent_scores))
        trend = np.polyfit(x, recent_scores, 1)[0]  # Slope of trend line
        
        return trend


# Example usage and testing framework
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test context
    test_context = ExperimentContext(
        domain="computer_vision",
        complexity_score=0.7,
        resource_budget=500.0,
        time_constraint=1800.0,  # 30 minutes
        novelty_requirement=0.8,
        success_history=[0.6, 0.7, 0.5, 0.8]
    )
    
    # Initialize orchestrator
    orchestrator = AdaptiveTreeSearchOrchestrator()
    
    # Execute multiple searches to demonstrate learning
    print("=== Adaptive Tree Search Demonstration ===\n")
    
    for i in range(5):
        print(f"Search {i+1}:")
        result = orchestrator.execute_search(test_context, max_iterations=30)
        print(f"  Best score: {result['best_score']:.3f}")
        print(f"  Exploration efficiency: {result['search_metrics'].exploration_efficiency:.2f}")
        print()
    
    # Display analytics
    analytics = orchestrator.get_performance_analytics()
    print("=== Performance Analytics ===")
    print(f"Total searches: {analytics['search_statistics']['total_searches']}")
    print(f"Average score: {analytics['search_statistics']['average_score']:.3f}")
    print(f"Improvement trend: {analytics['improvement_trend']:.4f}")
    print("\nStrategy usage:")
    for strategy, stats in analytics['strategy_analytics'].items():
        print(f"  {strategy}: {stats['usage_count']} uses, {stats['avg_performance']:.3f} avg performance")