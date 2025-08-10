#!/usr/bin/env python3
"""
Multi-Objective Autonomous Experimentation Framework
=====================================================

Novel multi-objective optimization framework that simultaneously optimizes research 
quality, computational cost, time efficiency, and novelty using adaptive Pareto 
frontier exploration.

Research Hypothesis: Multi-objective optimization with adaptive Pareto frontier 
exploration will achieve superior resource utilization and research outcomes 
compared to single-objective approaches.

Key Innovation: Dynamic Multi-Objective Evolutionary Algorithm (MOEA) with 
adaptive operators and real-time Pareto frontier updating with preference learning.

Author: AI Scientist v2 Autonomous System
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MultiObjective:
    """Represents multiple optimization objectives."""
    research_quality: float      # 0-1, higher better
    computational_cost: float    # 0-1, lower better (normalized)
    time_efficiency: float       # 0-1, higher better (1/time_taken)
    novelty_score: float        # 0-1, higher better
    resource_utilization: float # 0-1, higher better
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization."""
        return np.array([
            self.research_quality,
            1.0 - self.computational_cost,  # Convert to maximization
            self.time_efficiency,
            self.novelty_score,
            self.resource_utilization
        ])
    
    def weighted_sum(self, weights: np.ndarray) -> float:
        """Compute weighted sum with given weights."""
        return np.dot(self.to_array(), weights)


@dataclass
class ExperimentSolution:
    """Represents a solution in the multi-objective space."""
    solution_id: str
    parameters: Dict[str, Any]
    objectives: MultiObjective
    predicted_cost: float
    predicted_time: float
    experiment_metadata: Dict[str, Any]
    
    def dominates(self, other: 'ExperimentSolution') -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        self_obj = self.objectives.to_array()
        other_obj = other.objectives.to_array()
        
        return np.all(self_obj >= other_obj) and np.any(self_obj > other_obj)


class PreferenceLearner:
    """Learns user preferences for multi-objective trade-offs."""
    
    def __init__(self):
        self.preference_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])  # Default weights
        self.preference_history = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
    def update_preferences(self, chosen_solution: ExperimentSolution, 
                          alternatives: List[ExperimentSolution], 
                          user_feedback: float = 1.0) -> None:
        """Update preference weights based on user choices."""
        if not alternatives:
            return
            
        # Compute relative preference indicators
        chosen_obj = chosen_solution.objectives.to_array()
        
        for alt in alternatives:
            alt_obj = alt.objectives.to_array()
            # Compute preference direction
            preference_direction = chosen_obj - alt_obj
            
            # Update weights based on preference direction and feedback strength
            weight_update = self.learning_rate * user_feedback * preference_direction
            self.preference_weights += weight_update
            
        # Normalize and constrain weights
        self.preference_weights = np.clip(self.preference_weights, 0.01, 1.0)
        self.preference_weights /= np.sum(self.preference_weights)
        
        # Store preference history
        self.preference_history.append({
            'timestamp': time.time(),
            'weights': self.preference_weights.copy(),
            'confidence': user_feedback
        })
        
        logger.info(f"Updated preferences: {self.preference_weights}")
    
    def get_preference_weights(self) -> np.ndarray:
        """Get current preference weights."""
        return self.preference_weights.copy()
    
    def get_confidence(self) -> float:
        """Get confidence in current preferences."""
        if not self.preference_history:
            return 0.0
            
        # Confidence based on consistency of recent preferences
        recent_weights = [h['weights'] for h in self.preference_history[-10:]]
        if len(recent_weights) < 2:
            return 0.5
            
        # Compute variance in recent weights
        weight_variance = np.var(recent_weights, axis=0)
        avg_variance = np.mean(weight_variance)
        
        # Higher variance = lower confidence
        confidence = max(0.0, 1.0 - 5.0 * avg_variance)
        return confidence


class ParetoFrontier:
    """Manages and updates Pareto optimal solutions."""
    
    def __init__(self):
        self.pareto_solutions: List[ExperimentSolution] = []
        self.archive_size_limit = 100
        
    def add_solution(self, solution: ExperimentSolution) -> bool:
        """Add solution to Pareto frontier if non-dominated."""
        # Check if solution is dominated by existing solutions
        for existing in self.pareto_solutions:
            if existing.dominates(solution):
                return False  # Solution is dominated
        
        # Remove solutions dominated by new solution
        self.pareto_solutions = [
            sol for sol in self.pareto_solutions 
            if not solution.dominates(sol)
        ]
        
        # Add new solution
        self.pareto_solutions.append(solution)
        
        # Manage archive size
        if len(self.pareto_solutions) > self.archive_size_limit:
            self._prune_archive()
            
        return True
    
    def _prune_archive(self) -> None:
        """Prune archive to maintain diversity and size limit."""
        if len(self.pareto_solutions) <= self.archive_size_limit:
            return
            
        # Compute crowding distances for diversity preservation
        crowding_distances = self._compute_crowding_distances()
        
        # Keep solutions with highest crowding distances
        indexed_solutions = list(enumerate(self.pareto_solutions))
        indexed_solutions.sort(key=lambda x: crowding_distances[x[0]], reverse=True)
        
        # Keep top solutions
        self.pareto_solutions = [
            sol for idx, sol in indexed_solutions[:self.archive_size_limit]
        ]
        
        logger.info(f"Pruned Pareto archive to {len(self.pareto_solutions)} solutions")
    
    def _compute_crowding_distances(self) -> List[float]:
        """Compute crowding distances for diversity measure."""
        n_solutions = len(self.pareto_solutions)
        if n_solutions <= 2:
            return [float('inf')] * n_solutions
            
        crowding_distances = [0.0] * n_solutions
        n_objectives = 5  # Number of objectives
        
        # For each objective
        for obj_idx in range(n_objectives):
            # Sort solutions by objective value
            obj_values = [(i, self.pareto_solutions[i].objectives.to_array()[obj_idx]) 
                         for i in range(n_solutions)]
            obj_values.sort(key=lambda x: x[1])
            
            # Set boundary solutions to infinite distance
            crowding_distances[obj_values[0][0]] = float('inf')
            crowding_distances[obj_values[-1][0]] = float('inf')
            
            # Compute distances for interior solutions
            obj_range = obj_values[-1][1] - obj_values[0][1]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (obj_values[i+1][1] - obj_values[i-1][1]) / obj_range
                    crowding_distances[obj_values[i][0]] += distance
                    
        return crowding_distances
    
    def get_pareto_solutions(self) -> List[ExperimentSolution]:
        """Get all Pareto optimal solutions."""
        return self.pareto_solutions.copy()
    
    def get_best_compromise(self, weights: np.ndarray) -> Optional[ExperimentSolution]:
        """Get best compromise solution given preference weights."""
        if not self.pareto_solutions:
            return None
            
        best_solution = None
        best_score = -float('inf')
        
        for solution in self.pareto_solutions:
            score = solution.objectives.weighted_sum(weights)
            if score > best_score:
                best_score = score
                best_solution = solution
                
        return best_solution


class MultiObjectiveEvolutionaryAlgorithm:
    """Dynamic Multi-Objective Evolutionary Algorithm (MOEA) implementation."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.8
        self.generation = 0
        self.population: List[ExperimentSolution] = []
        self.pareto_frontier = ParetoFrontier()
        
    def initialize_population(self, search_space: Dict[str, Any]) -> None:
        """Initialize population with random solutions."""
        self.population = []
        
        for i in range(self.population_size):
            # Generate random parameters within search space
            parameters = {}
            for param_name, param_range in search_space.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], float):
                        parameters[param_name] = np.random.uniform(param_range[0], param_range[1])
                    else:
                        parameters[param_name] = np.random.randint(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    parameters[param_name] = np.random.choice(param_range)
            
            # Create mock objectives (replace with actual evaluation)
            objectives = self._evaluate_solution(parameters)
            
            solution = ExperimentSolution(
                solution_id=f"gen0_ind{i}",
                parameters=parameters,
                objectives=objectives,
                predicted_cost=np.random.uniform(10, 1000),
                predicted_time=np.random.uniform(60, 3600),
                experiment_metadata={'generation': 0, 'individual': i}
            )
            
            self.population.append(solution)
            self.pareto_frontier.add_solution(solution)
    
    def evolve_generation(self, preference_learner: PreferenceLearner) -> None:
        """Evolve one generation using NSGA-II inspired approach."""
        self.generation += 1
        
        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(self.population), 2):
            parent1 = self.population[i]
            parent2 = self.population[min(i + 1, len(self.population) - 1)]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        # Mutation
        for individual in offspring:
            if np.random.random() < self.mutation_rate:
                self._mutate(individual)
        
        # Combine parent and offspring populations
        combined_population = self.population + offspring
        
        # Environmental selection using preference-guided NSGA-II
        self.population = self._environmental_selection(
            combined_population, preference_learner.get_preference_weights()
        )
        
        # Update Pareto frontier
        for solution in self.population:
            self.pareto_frontier.add_solution(solution)
            
        logger.info(f"Generation {self.generation}: Population size = {len(self.population)}, "
                   f"Pareto solutions = {len(self.pareto_frontier.get_pareto_solutions())}")
    
    def _evaluate_solution(self, parameters: Dict[str, Any]) -> MultiObjective:
        """Evaluate solution objectives (mock implementation)."""
        # Mock evaluation - replace with actual experiment execution
        quality = np.random.beta(3, 2)  # Slightly biased toward higher quality
        cost = np.random.gamma(2, 0.3)  # Cost distribution
        time_eff = np.random.beta(2, 3)  # Time efficiency
        novelty = np.random.uniform(0, 1)
        resource_util = np.random.beta(4, 2)
        
        return MultiObjective(
            research_quality=quality,
            computational_cost=min(cost, 1.0),
            time_efficiency=time_eff,
            novelty_score=novelty,
            resource_utilization=resource_util
        )
    
    def _crossover(self, parent1: ExperimentSolution, 
                   parent2: ExperimentSolution) -> Tuple[ExperimentSolution, ExperimentSolution]:
        """Simulated Binary Crossover (SBX) for real parameters."""
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            p1_val = parent1.parameters[param_name]
            p2_val = parent2.parameters[param_name]
            
            if isinstance(p1_val, (int, float)):
                # SBX crossover for numerical parameters
                eta_c = 20  # Crossover distribution index
                rand = np.random.random()
                
                if rand <= 0.5:
                    beta = (2 * rand) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
                
                c1_val = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
                c2_val = 0.5 * ((1 - beta) * p1_val + (1 + beta) * p2_val)
                
                child1_params[param_name] = c1_val
                child2_params[param_name] = c2_val
            else:
                # Uniform crossover for categorical parameters
                if np.random.random() < 0.5:
                    child1_params[param_name] = p1_val
                    child2_params[param_name] = p2_val
                else:
                    child1_params[param_name] = p2_val
                    child2_params[param_name] = p1_val
        
        # Evaluate offspring
        child1_obj = self._evaluate_solution(child1_params)
        child2_obj = self._evaluate_solution(child2_params)
        
        child1 = ExperimentSolution(
            solution_id=f"gen{self.generation}_cross1",
            parameters=child1_params,
            objectives=child1_obj,
            predicted_cost=np.random.uniform(10, 1000),
            predicted_time=np.random.uniform(60, 3600),
            experiment_metadata={'generation': self.generation, 'operator': 'crossover'}
        )
        
        child2 = ExperimentSolution(
            solution_id=f"gen{self.generation}_cross2",
            parameters=child2_params,
            objectives=child2_obj,
            predicted_cost=np.random.uniform(10, 1000),
            predicted_time=np.random.uniform(60, 3600),
            experiment_metadata={'generation': self.generation, 'operator': 'crossover'}
        )
        
        return child1, child2
    
    def _mutate(self, individual: ExperimentSolution) -> None:
        """Polynomial mutation for numerical parameters."""
        eta_m = 20  # Mutation distribution index
        
        for param_name, param_value in individual.parameters.items():
            if isinstance(param_value, (int, float)) and np.random.random() < 0.1:
                rand = np.random.random()
                
                if rand < 0.5:
                    delta = (2 * rand) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - rand)) ** (1 / (eta_m + 1))
                
                # Apply mutation with bounds checking
                mutated_value = param_value + 0.1 * delta * abs(param_value)
                individual.parameters[param_name] = mutated_value
        
        # Re-evaluate objectives after mutation
        individual.objectives = self._evaluate_solution(individual.parameters)
    
    def _environmental_selection(self, population: List[ExperimentSolution], 
                               preference_weights: np.ndarray) -> List[ExperimentSolution]:
        """Select next generation using preference-guided NSGA-II."""
        # Non-dominated sorting
        fronts = self._non_dominated_sort(population)
        
        # Select individuals for next generation
        next_generation = []
        front_idx = 0
        
        while len(next_generation) + len(fronts[front_idx]) <= self.population_size:
            next_generation.extend(fronts[front_idx])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        # If we need to select partial front
        if len(next_generation) < self.population_size and front_idx < len(fronts):
            remaining_slots = self.population_size - len(next_generation)
            partial_front = fronts[front_idx]
            
            # Use preference-guided crowding distance
            crowding_distances = self._compute_preference_crowding_distance(
                partial_front, preference_weights
            )
            
            # Sort by crowding distance and select best
            indexed_front = list(enumerate(partial_front))
            indexed_front.sort(key=lambda x: crowding_distances[x[0]], reverse=True)
            
            selected = [sol for idx, sol in indexed_front[:remaining_slots]]
            next_generation.extend(selected)
        
        return next_generation
    
    def _non_dominated_sort(self, population: List[ExperimentSolution]) -> List[List[ExperimentSolution]]:
        """Perform non-dominated sorting."""
        fronts = []
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        
        # Compute domination relationships
        for i, sol1 in enumerate(population):
            for j, sol2 in enumerate(population):
                if i != j:
                    if sol1.dominates(sol2):
                        dominated_solutions[i].append(j)
                    elif sol2.dominates(sol1):
                        domination_count[i] += 1
        
        # Find first front
        first_front = []
        for i in range(len(population)):
            if domination_count[i] == 0:
                first_front.append(population[i])
        
        fronts.append(first_front)
        
        # Find subsequent fronts
        current_front_indices = [i for i in range(len(population)) if domination_count[i] == 0]
        
        while current_front_indices:
            next_front_indices = []
            
            for i in current_front_indices:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front_indices.append(j)
            
            if next_front_indices:
                next_front = [population[i] for i in next_front_indices]
                fronts.append(next_front)
                current_front_indices = next_front_indices
            else:
                break
        
        return fronts
    
    def _compute_preference_crowding_distance(self, solutions: List[ExperimentSolution], 
                                            weights: np.ndarray) -> List[float]:
        """Compute preference-guided crowding distance."""
        n_solutions = len(solutions)
        if n_solutions <= 2:
            return [float('inf')] * n_solutions
        
        crowding_distances = [0.0] * n_solutions
        
        # For each objective, weight by preference
        for obj_idx in range(len(weights)):
            weight = weights[obj_idx]
            
            # Sort solutions by weighted objective value
            obj_values = [(i, solutions[i].objectives.to_array()[obj_idx] * weight) 
                         for i in range(n_solutions)]
            obj_values.sort(key=lambda x: x[1])
            
            # Set boundary solutions to infinite distance
            crowding_distances[obj_values[0][0]] = float('inf')
            crowding_distances[obj_values[-1][0]] = float('inf')
            
            # Compute weighted distances for interior solutions
            obj_range = obj_values[-1][1] - obj_values[0][1]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (obj_values[i+1][1] - obj_values[i-1][1]) / obj_range
                    crowding_distances[obj_values[i][0]] += weight * distance
        
        return crowding_distances


class MultiObjectiveOrchestrator:
    """Main orchestrator for multi-objective autonomous experimentation."""
    
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.moea = MultiObjectiveEvolutionaryAlgorithm()
        self.preference_learner = PreferenceLearner()
        self.experiment_history = []
        self.cost_budget = 10000.0  # Total computational budget
        self.time_budget = 7200.0   # Total time budget in seconds
        
    def optimize_experiments(self, max_generations: int = 20, 
                           budget_constraint: float = None) -> Dict[str, Any]:
        """Execute multi-objective optimization of experiments."""
        start_time = time.time()
        budget_used = 0.0
        
        if budget_constraint:
            self.cost_budget = budget_constraint
        
        logger.info(f"Starting multi-objective optimization with {max_generations} generations")
        logger.info(f"Budget constraints: Cost={self.cost_budget}, Time={self.time_budget}s")
        
        # Initialize population
        self.moea.initialize_population(self.search_space)
        initial_pareto_size = len(self.moea.pareto_frontier.get_pareto_solutions())
        
        # Evolution loop
        for generation in range(max_generations):
            # Check time constraint
            if time.time() - start_time > self.time_budget:
                logger.info(f"Time budget exceeded, stopping at generation {generation}")
                break
            
            # Evolve generation
            self.moea.evolve_generation(self.preference_learner)
            
            # Estimate budget usage (mock calculation)
            generation_cost = np.sum([sol.predicted_cost for sol in self.moea.population]) / 10
            budget_used += generation_cost
            
            # Check budget constraint
            if budget_used > self.cost_budget:
                logger.info(f"Cost budget exceeded, stopping at generation {generation}")
                break
            
            # Log progress
            pareto_size = len(self.moea.pareto_frontier.get_pareto_solutions())
            logger.info(f"Generation {generation + 1}: Pareto solutions = {pareto_size}, "
                       f"Budget used = {budget_used:.1f}")
        
        # Get final results
        final_pareto_solutions = self.moea.pareto_frontier.get_pareto_solutions()
        best_compromise = self.moea.pareto_frontier.get_best_compromise(
            self.preference_learner.get_preference_weights()
        )
        
        # Compute optimization metrics
        total_time = time.time() - start_time
        final_pareto_size = len(final_pareto_solutions)
        
        optimization_results = {
            'final_pareto_solutions': final_pareto_solutions,
            'best_compromise_solution': best_compromise,
            'optimization_metrics': {
                'total_time': total_time,
                'generations_completed': self.moea.generation,
                'budget_used': budget_used,
                'pareto_size_improvement': final_pareto_size - initial_pareto_size,
                'preference_confidence': self.preference_learner.get_confidence()
            },
            'pareto_frontier_analysis': self._analyze_pareto_frontier(final_pareto_solutions)
        }
        
        logger.info(f"Optimization completed: {final_pareto_size} Pareto solutions found")
        return optimization_results
    
    def _analyze_pareto_frontier(self, solutions: List[ExperimentSolution]) -> Dict[str, Any]:
        """Analyze Pareto frontier characteristics."""
        if not solutions:
            return {'message': 'No solutions in Pareto frontier'}
        
        # Extract objective values
        objectives_matrix = np.array([sol.objectives.to_array() for sol in solutions])
        
        # Compute statistics
        obj_means = np.mean(objectives_matrix, axis=0)
        obj_stds = np.std(objectives_matrix, axis=0)
        obj_ranges = np.ptp(objectives_matrix, axis=0)  # Range (max - min)
        
        # Compute hypervolume (simplified 2D approximation)
        hypervolume = self._compute_hypervolume_2d(objectives_matrix)
        
        # Compute diversity metrics
        pairwise_distances = []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                dist = np.linalg.norm(objectives_matrix[i] - objectives_matrix[j])
                pairwise_distances.append(dist)
        
        diversity = np.mean(pairwise_distances) if pairwise_distances else 0.0
        
        objective_names = ['Quality', 'Cost_Efficiency', 'Time_Efficiency', 'Novelty', 'Resource_Util']
        
        return {
            'pareto_size': len(solutions),
            'objective_statistics': {
                objective_names[i]: {
                    'mean': obj_means[i],
                    'std': obj_stds[i],
                    'range': obj_ranges[i]
                } for i in range(len(objective_names))
            },
            'hypervolume': hypervolume,
            'diversity_score': diversity,
            'trade_off_analysis': self._analyze_trade_offs(objectives_matrix)
        }
    
    def _compute_hypervolume_2d(self, objectives: np.ndarray) -> float:
        """Compute simplified 2D hypervolume for first two objectives."""
        if len(objectives) < 2:
            return 0.0
        
        # Use first two objectives for 2D hypervolume
        points_2d = objectives[:, :2]
        
        # Sort points by first objective
        sorted_points = points_2d[points_2d[:, 0].argsort()]
        
        # Compute hypervolume with reference point at origin
        hypervolume = 0.0
        for i, point in enumerate(sorted_points):
            if i == 0:
                width = point[0]
            else:
                width = point[0] - sorted_points[i-1][0]
            
            height = point[1]
            hypervolume += width * height
        
        return hypervolume
    
    def _analyze_trade_offs(self, objectives: np.ndarray) -> Dict[str, float]:
        """Analyze trade-offs between objectives."""
        if len(objectives) < 2:
            return {}
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(objectives.T)
        
        objective_names = ['Quality', 'Cost_Eff', 'Time_Eff', 'Novelty', 'Resource_Util']
        
        # Identify significant trade-offs (strong negative correlations)
        trade_offs = {}
        for i in range(len(objective_names)):
            for j in range(i + 1, len(objective_names)):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > 0.5:  # Significant correlation threshold
                    trade_off_name = f"{objective_names[i]}_vs_{objective_names[j]}"
                    trade_offs[trade_off_name] = correlation
        
        return trade_offs
    
    def update_preferences(self, chosen_solution_id: str, 
                          alternatives: List[str], 
                          user_feedback: float = 1.0) -> None:
        """Update user preferences based on solution choices."""
        # Find solutions by ID
        all_solutions = self.moea.pareto_frontier.get_pareto_solutions()
        solution_dict = {sol.solution_id: sol for sol in all_solutions}
        
        chosen_solution = solution_dict.get(chosen_solution_id)
        alternative_solutions = [solution_dict[alt_id] for alt_id in alternatives if alt_id in solution_dict]
        
        if chosen_solution and alternative_solutions:
            self.preference_learner.update_preferences(chosen_solution, alternative_solutions, user_feedback)
            logger.info(f"Updated preferences based on choice: {chosen_solution_id}")


# Example usage and testing framework
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define search space for experiments
    search_space = {
        'learning_rate': (0.0001, 0.1),
        'batch_size': [16, 32, 64, 128],
        'model_architecture': ['resnet', 'vgg', 'densenet'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'regularization': (0.0, 0.1),
        'epochs': (10, 100)
    }
    
    print("=== Multi-Objective Autonomous Experimentation Demo ===\n")
    
    # Initialize orchestrator
    orchestrator = MultiObjectiveOrchestrator(search_space)
    
    # Execute optimization
    results = orchestrator.optimize_experiments(max_generations=15, budget_constraint=5000.0)
    
    print(f"Optimization completed!")
    print(f"Total time: {results['optimization_metrics']['total_time']:.1f}s")
    print(f"Generations: {results['optimization_metrics']['generations_completed']}")
    print(f"Budget used: ${results['optimization_metrics']['budget_used']:.1f}")
    print(f"Pareto solutions found: {len(results['final_pareto_solutions'])}")
    
    # Display best compromise solution
    if results['best_compromise_solution']:
        best_sol = results['best_compromise_solution']
        print(f"\nBest compromise solution:")
        print(f"  Quality: {best_sol.objectives.research_quality:.3f}")
        print(f"  Cost efficiency: {1-best_sol.objectives.computational_cost:.3f}")
        print(f"  Time efficiency: {best_sol.objectives.time_efficiency:.3f}")
        print(f"  Novelty: {best_sol.objectives.novelty_score:.3f}")
        print(f"  Resource utilization: {best_sol.objectives.resource_utilization:.3f}")
    
    # Display Pareto frontier analysis
    pf_analysis = results['pareto_frontier_analysis']
    print(f"\nPareto Frontier Analysis:")
    print(f"  Hypervolume: {pf_analysis['hypervolume']:.3f}")
    print(f"  Diversity score: {pf_analysis['diversity_score']:.3f}")
    print(f"  Trade-offs identified: {len(pf_analysis['trade_off_analysis'])}")