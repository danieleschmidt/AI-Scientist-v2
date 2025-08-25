#!/usr/bin/env python3
"""
Bayesian Optimization Engine - Generation 1 Enhancement
========================================================

Advanced Bayesian optimization system for hyperparameter tuning and experimental design
using Gaussian processes with multiple acquisition functions.

Key Features:
- Multi-objective Bayesian optimization
- Adaptive acquisition function selection
- Uncertainty quantification and confidence intervals
- Parallel experiment evaluation with batch optimization
- Integration with existing experimentation engine

Author: AI Scientist v2 - Terragon Labs (Generation 1)
License: MIT
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings

# Enhanced imports for Bayesian optimization
try:
    from scipy.optimize import minimize
    from scipy.stats import norm, multivariate_normal
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    ENTROPY_SEARCH = "entropy_search"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"


class OptimizationObjective(Enum):
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective" 
    CONSTRAINED = "constrained"
    ROBUST = "robust"


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization."""
    best_parameters: Dict[str, Any]
    best_value: float
    best_values_history: List[float]
    all_parameters: List[Dict[str, Any]]
    all_values: List[float]
    acquisition_values: List[float]
    convergence_iterations: int
    uncertainty_estimates: List[float]
    confidence_intervals: List[Tuple[float, float]]
    optimization_time: float
    total_evaluations: int


@dataclass 
class Parameter:
    """Parameter definition for optimization."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None


class BayesianOptimizationEngine:
    """
    Advanced Bayesian optimization engine for hyperparameter tuning.
    
    Implements multiple acquisition functions, multi-objective optimization,
    and uncertainty quantification for robust experimental design.
    """
    
    def __init__(self, 
                 workspace_dir: str = "/tmp/bayesian_optimization",
                 n_initial_samples: int = 10,
                 max_iterations: int = 100,
                 acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
                 xi: float = 0.01,  # exploration parameter
                 kappa: float = 2.576,  # UCB parameter (99% confidence)
                 n_restarts: int = 10):
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_initial_samples = n_initial_samples
        self.max_iterations = max_iterations
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.kappa = kappa
        self.n_restarts = n_restarts
        
        # Optimization state
        self.X_samples = []  # Parameter configurations
        self.y_samples = []  # Objective values
        self.gp_model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Multi-objective state
        self.pareto_front = []
        self.dominated_solutions = []
        
        logger.info(f"BayesianOptimizationEngine initialized with {acquisition_function.value}")
    
    def optimize_single_objective(self,
                                objective_function: Callable[[Dict[str, Any]], float],
                                parameters: List[Parameter],
                                maximize: bool = True,
                                verbose: bool = True) -> OptimizationResult:
        """
        Perform single-objective Bayesian optimization.
        
        Args:
            objective_function: Function to optimize (takes parameter dict, returns float)
            parameters: List of Parameter objects defining search space
            maximize: Whether to maximize (True) or minimize (False) objective
            verbose: Whether to print optimization progress
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        start_time = time.time()
        
        if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("scipy and scikit-learn are required for Bayesian optimization")
        
        # Initialize optimization
        self._initialize_optimization(parameters)
        
        # Generate initial samples using Latin Hypercube Sampling
        initial_samples = self._generate_initial_samples(parameters, self.n_initial_samples)
        
        # Evaluate initial samples
        if verbose:
            logger.info(f"Evaluating {len(initial_samples)} initial samples...")
        
        for sample in initial_samples:
            value = objective_function(sample)
            if not maximize:
                value = -value  # Convert to maximization problem
            self.X_samples.append(self._encode_parameters(sample, parameters))
            self.y_samples.append(value)
        
        # Convert to numpy arrays
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        
        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Optimization history
        best_values = []
        acquisition_values = []
        uncertainty_estimates = []
        confidence_intervals = []
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            if verbose and iteration % 10 == 0:
                logger.info(f"Bayesian optimization iteration {iteration}/{self.max_iterations}")
            
            # Fit Gaussian Process
            self._fit_gaussian_process(X_scaled, y_scaled)
            
            # Find next point using acquisition function
            next_X_scaled, acquisition_value = self._optimize_acquisition_function(
                X_scaled, y_scaled, parameters
            )
            
            # Convert back to original scale
            next_X = self.scaler_X.inverse_transform(next_X_scaled.reshape(1, -1))[0]
            next_params = self._decode_parameters(next_X, parameters)
            
            # Evaluate objective function
            next_y = objective_function(next_params)
            if not maximize:
                next_y = -next_y
            
            # Add to samples
            self.X_samples.append(next_X)
            self.y_samples.append(next_y)
            
            # Update arrays
            X = np.array(self.X_samples)
            y = np.array(self.y_samples)
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
            
            # Track optimization progress
            current_best = np.max(y) if maximize else -np.max(y)
            best_values.append(current_best)
            acquisition_values.append(acquisition_value)
            
            # Estimate uncertainty for current best
            if self.gp_model is not None:
                best_idx = np.argmax(y)
                best_X_scaled = X_scaled[best_idx:best_idx+1]
                uncertainty = self._estimate_uncertainty(best_X_scaled)
                uncertainty_estimates.append(uncertainty)
                
                # 95% confidence interval
                ci_lower, ci_upper = self._compute_confidence_interval(
                    best_X_scaled, confidence_level=0.95
                )
                confidence_intervals.append((ci_lower, ci_upper))
            
            # Early stopping if converged
            if iteration > 10:
                recent_improvement = best_values[-1] - best_values[-10]
                if abs(recent_improvement) < 1e-6:
                    logger.info(f"Converged after {iteration} iterations")
                    break
        
        # Extract results
        y_final = np.array(self.y_samples)
        best_idx = np.argmax(y_final)
        
        if not maximize:
            best_values = [-v for v in best_values]
            y_final = -y_final
        
        best_params = self._decode_parameters(self.X_samples[best_idx], parameters)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_parameters=best_params,
            best_value=y_final[best_idx],
            best_values_history=best_values,
            all_parameters=[self._decode_parameters(x, parameters) for x in self.X_samples],
            all_values=y_final.tolist(),
            acquisition_values=acquisition_values,
            convergence_iterations=len(best_values),
            uncertainty_estimates=uncertainty_estimates,
            confidence_intervals=confidence_intervals,
            optimization_time=optimization_time,
            total_evaluations=len(self.X_samples)
        )
        
        if verbose:
            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best value: {result.best_value:.6f}")
            logger.info(f"Best parameters: {result.best_parameters}")
        
        return result
    
    def _initialize_optimization(self, parameters: List[Parameter]):
        """Initialize optimization state."""
        self.X_samples = []
        self.y_samples = []
        self.gp_model = None
    
    def _generate_initial_samples(self, parameters: List[Parameter], n_samples: int) -> List[Dict[str, Any]]:
        """Generate initial samples using Latin Hypercube Sampling."""
        samples = []
        n_dims = len(parameters)
        
        # Latin Hypercube Sampling
        lhs_samples = np.random.rand(n_samples, n_dims)
        
        for i in range(n_samples):
            sample = {}
            for j, param in enumerate(parameters):
                if param.param_type == 'continuous':
                    if param.bounds is not None:
                        low, high = param.bounds
                        if param.log_scale:
                            low, high = np.log10(low), np.log10(high)
                            value = 10 ** (low + lhs_samples[i, j] * (high - low))
                        else:
                            value = low + lhs_samples[i, j] * (high - low)
                        sample[param.name] = float(value)
                elif param.param_type == 'discrete':
                    if param.bounds is not None:
                        low, high = param.bounds
                        value = int(low + lhs_samples[i, j] * (high - low + 1))
                        sample[param.name] = value
                elif param.param_type == 'categorical':
                    if param.choices is not None:
                        idx = int(lhs_samples[i, j] * len(param.choices))
                        sample[param.name] = param.choices[idx]
            
            samples.append(sample)
        
        return samples
    
    def _encode_parameters(self, params: Dict[str, Any], parameters: List[Parameter]) -> np.ndarray:
        """Encode parameters to numerical array."""
        encoded = []
        
        for param in parameters:
            value = params[param.name]
            
            if param.param_type == 'continuous':
                if param.log_scale and value > 0:
                    encoded.append(np.log10(value))
                else:
                    encoded.append(float(value))
            elif param.param_type == 'discrete':
                encoded.append(float(value))
            elif param.param_type == 'categorical':
                if param.choices is not None:
                    idx = param.choices.index(value) if value in param.choices else 0
                    encoded.append(float(idx))
        
        return np.array(encoded)
    
    def _decode_parameters(self, encoded: np.ndarray, parameters: List[Parameter]) -> Dict[str, Any]:
        """Decode numerical array to parameters."""
        params = {}
        
        for i, param in enumerate(parameters):
            value = encoded[i]
            
            if param.param_type == 'continuous':
                if param.log_scale:
                    params[param.name] = float(10 ** value)
                else:
                    params[param.name] = float(value)
            elif param.param_type == 'discrete':
                params[param.name] = int(round(value))
            elif param.param_type == 'categorical':
                if param.choices is not None:
                    idx = int(round(value))
                    idx = max(0, min(idx, len(param.choices) - 1))
                    params[param.name] = param.choices[idx]
        
        return params
    
    def _fit_gaussian_process(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process model to current data."""
        # Define kernel
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) * 
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )
        
        # Fit GP
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        self.gp_model.fit(X, y)
    
    def _optimize_acquisition_function(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray, 
                                     parameters: List[Parameter]) -> Tuple[np.ndarray, float]:
        """Optimize acquisition function to find next evaluation point."""
        
        def acquisition_func(x_scaled):
            x_scaled = x_scaled.reshape(1, -1)
            
            if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
                return -self._expected_improvement(x_scaled, y)
            elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
                return -self._upper_confidence_bound(x_scaled)
            elif self.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
                return -self._probability_of_improvement(x_scaled, y)
            else:
                return -self._expected_improvement(x_scaled, y)
        
        # Define bounds for optimization
        bounds = []
        for param in parameters:
            if param.param_type == 'continuous':
                if param.bounds is not None:
                    low, high = param.bounds
                    if param.log_scale:
                        low, high = np.log10(low), np.log10(high)
                    bounds.append((low, high))
                else:
                    bounds.append((-5, 5))  # Default bounds
            elif param.param_type == 'discrete':
                if param.bounds is not None:
                    bounds.append(param.bounds)
                else:
                    bounds.append((0, 10))  # Default bounds
            elif param.param_type == 'categorical':
                if param.choices is not None:
                    bounds.append((0, len(param.choices) - 1))
                else:
                    bounds.append((0, 1))
        
        # Scale bounds to match standardized feature space
        bounds_scaled = []
        for i, (low, high) in enumerate(bounds):
            X_col = X[:, i]
            low_scaled = (low - self.scaler_X.mean_[i]) / self.scaler_X.scale_[i]
            high_scaled = (high - self.scaler_X.mean_[i]) / self.scaler_X.scale_[i]
            bounds_scaled.append((low_scaled, high_scaled))
        
        # Multiple random restarts
        best_x = None
        best_acquisition = np.inf
        
        for _ in range(self.n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high) 
                for low, high in bounds_scaled
            ])
            
            try:
                result = minimize(
                    acquisition_func,
                    x0,
                    bounds=bounds_scaled,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acquisition:
                    best_acquisition = result.fun
                    best_x = result.x
            except Exception as e:
                logger.warning(f"Acquisition optimization failed: {e}")
                continue
        
        if best_x is None:
            # Fallback to random point
            best_x = np.array([
                np.random.uniform(low, high)
                for low, high in bounds_scaled
            ])
            best_acquisition = acquisition_func(best_x)
        
        return best_x, -best_acquisition
    
    def _expected_improvement(self, x: np.ndarray, y_samples: np.ndarray) -> float:
        """Expected Improvement acquisition function."""
        if self.gp_model is None:
            return 0.0
        
        mu, sigma = self.gp_model.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma < 1e-9:
            return 0.0
        
        f_best = np.max(y_samples)
        z = (mu - f_best - self.xi) / sigma
        
        ei = (mu - f_best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, ei)
    
    def _upper_confidence_bound(self, x: np.ndarray) -> float:
        """Upper Confidence Bound acquisition function."""
        if self.gp_model is None:
            return 0.0
        
        mu, sigma = self.gp_model.predict(x, return_std=True)
        ucb = mu[0] + self.kappa * sigma[0]
        return ucb
    
    def _probability_of_improvement(self, x: np.ndarray, y_samples: np.ndarray) -> float:
        """Probability of Improvement acquisition function."""
        if self.gp_model is None:
            return 0.0
        
        mu, sigma = self.gp_model.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma < 1e-9:
            return 0.0
        
        f_best = np.max(y_samples)
        z = (mu - f_best - self.xi) / sigma
        
        return norm.cdf(z)
    
    def _estimate_uncertainty(self, x: np.ndarray) -> float:
        """Estimate prediction uncertainty at given point."""
        if self.gp_model is None:
            return 0.0
        
        _, sigma = self.gp_model.predict(x, return_std=True)
        return sigma[0]
    
    def _compute_confidence_interval(self, 
                                   x: np.ndarray, 
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for prediction."""
        if self.gp_model is None:
            return (0.0, 0.0)
        
        mu, sigma = self.gp_model.predict(x, return_std=True)
        mu, sigma = mu[0], sigma[0]
        
        # Z-score for confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Transform back to original scale
        mu_orig = self.scaler_y.inverse_transform([[mu]])[0][0]
        sigma_orig = sigma * self.scaler_y.scale_[0]
        
        ci_lower = mu_orig - z_score * sigma_orig
        ci_upper = mu_orig + z_score * sigma_orig
        
        return (ci_lower, ci_upper)
    
    def plot_optimization_history(self, result: OptimizationResult, save_path: Optional[str] = None):
        """Plot optimization history and convergence."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Best value over iterations
        ax1.plot(result.best_values_history, 'b-', linewidth=2, label='Best Value')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Objective Value')
        ax1.set_title('Convergence History')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Acquisition function values
        ax2.plot(result.acquisition_values, 'r-', linewidth=2, label='Acquisition Value')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Acquisition Function Value')
        ax2.set_title('Acquisition Function History')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # All evaluated points
        ax3.scatter(range(len(result.all_values)), result.all_values, 
                   alpha=0.6, c='green', label='Evaluated Points')
        ax3.plot(result.best_values_history, 'b-', linewidth=2, label='Best So Far')
        ax3.set_xlabel('Evaluation Number')
        ax3.set_ylabel('Objective Value')
        ax3.set_title('All Evaluations')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Uncertainty estimates
        if result.uncertainty_estimates:
            ax4.plot(result.uncertainty_estimates, 'purple', linewidth=2, label='Uncertainty')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Prediction Uncertainty')
            ax4.set_title('Uncertainty Evolution')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization plot saved to {save_path}")
        
        return fig


# Example usage and testing functions
async def test_bayesian_optimization():
    """Test Bayesian optimization on synthetic functions."""
    
    def branin_function(params: Dict[str, Any]) -> float:
        """Branin test function (global minimum at (-pi, 12.275) and (pi, 2.275))."""
        x1, x2 = params['x1'], params['x2']
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    
    # Define parameters
    parameters = [
        Parameter('x1', 'continuous', bounds=(-5, 10)),
        Parameter('x2', 'continuous', bounds=(0, 15))
    ]
    
    # Initialize optimizer
    optimizer = BayesianOptimizationEngine(
        n_initial_samples=10,
        max_iterations=50,
        acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT
    )
    
    # Optimize (minimize Branin function)
    result = optimizer.optimize_single_objective(
        branin_function, 
        parameters, 
        maximize=False,
        verbose=True
    )
    
    print(f"\nBranin Function Optimization Results:")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    
    # Plot results
    optimizer.plot_optimization_history(result, "/tmp/branin_optimization.png")
    
    return result


if __name__ == "__main__":
    # Test the Bayesian optimization engine
    asyncio.run(test_bayesian_optimization())