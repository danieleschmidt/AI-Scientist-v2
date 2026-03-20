"""ExperimentSimulator: evaluate a hypothesis by running toy experiments."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats
from scipy.optimize import curve_fit

from .hypothesis_tree import HypothesisNode


@dataclass
class ExperimentResult:
    hypothesis_id: str
    p_value: float
    effect_size: float
    r_squared: float
    supported: bool
    description: str


class ExperimentSimulator:
    """Simulate scientific experiments to evaluate hypotheses.

    Uses two experiment types:
    1. Curve fitting — fit a model to synthetic data, compute R²
    2. Statistical test — t-test between two synthetic conditions
    """

    def __init__(self, noise_level: float = 0.1, seed: Optional[int] = 42):
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def _generate_data(self, node: HypothesisNode, n: int = 30):
        """Generate synthetic data whose quality depends on node confidence."""
        x = np.linspace(0, 2 * np.pi, n)
        # Higher confidence → cleaner signal
        noise = self.noise_level / (node.confidence + 0.1)
        y = np.sin(x) + self.rng.normal(0, noise, n)
        return x, y

    def _curve_fit_experiment(self, node: HypothesisNode) -> dict:
        """Fit y = A*sin(x + phi) to data, compute R²."""
        x, y = self._generate_data(node)

        def model(x, A, phi):
            return A * np.sin(x + phi)

        try:
            popt, _ = curve_fit(model, x, y, p0=[1.0, 0.0], maxfev=1000)
            y_pred = model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        except Exception:
            r2 = 0.0

        return {"r2": float(r2)}

    def _statistical_test(self, node: HypothesisNode) -> dict:
        """Two-sample t-test between treatment and control groups."""
        n = 20
        effect = node.confidence  # true effect proportional to confidence
        control = self.rng.normal(0, 1, n)
        treatment = self.rng.normal(effect, 1, n)
        t_stat, p_val = stats.ttest_ind(treatment, control)
        effect_size = abs(np.mean(treatment) - np.mean(control)) / (
            np.sqrt((np.var(treatment) + np.var(control)) / 2) + 1e-9
        )
        return {"p_value": float(p_val), "effect_size": float(effect_size)}

    def evaluate(self, node: HypothesisNode) -> ExperimentResult:
        """Run both experiments and return aggregated result."""
        fit_result = self._curve_fit_experiment(node)
        stat_result = self._statistical_test(node)

        r2 = fit_result["r2"]
        p_val = stat_result["p_value"]
        effect = stat_result["effect_size"]
        supported = r2 > 0.7 and p_val < 0.05

        description = (
            f"Curve fit R²={r2:.3f}, t-test p={p_val:.3f}, effect={effect:.3f}. "
            f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}."
        )

        result = ExperimentResult(
            hypothesis_id=node.node_id,
            p_value=p_val,
            effect_size=effect,
            r_squared=r2,
            supported=supported,
            description=description,
        )
        # Update node
        node.experiment_result = {
            "p_value": p_val, "effect_size": effect, "r_squared": r2
        }
        node.add_evidence(description, confidence_delta=(0.1 if supported else -0.1))
        return result
