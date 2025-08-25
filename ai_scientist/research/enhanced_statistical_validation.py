#!/usr/bin/env python3
"""
Enhanced Statistical Validation Suite - Generation 1 Enhancement
===============================================================

Advanced statistical validation methods for AI research including Bayesian hypothesis testing,
multiple testing correction, effect size estimation, and uncertainty quantification.

Key Features:
- Bayesian hypothesis testing with ROPE (Region of Practical Equivalence)
- Multiple testing correction (FDR, Bonferroni, Holm-Bonferroni)
- Effect size estimation and confidence intervals
- Meta-analysis capabilities for combining results
- Bootstrap confidence intervals and permutation tests
- Reproducibility assessment and uncertainty quantification

Author: AI Scientist v2 - Terragon Labs (Generation 1)
License: MIT
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import warnings

# Statistical computing
try:
    from scipy import stats
    from scipy.stats import (
        ttest_ind, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
        chi2_contingency, pearsonr, spearmanr, kendalltau,
        jarque_bera, shapiro, levene, bartlett
    )
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Advanced statistical methods
try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.meta_analysis import combine_effects
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


class HypothesisTest(Enum):
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    CHI_SQUARE = "chi_square"
    MCNEMAR = "mcnemar"
    PERMUTATION = "permutation"


class EffectSizeMetric(Enum):
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    R_SQUARED = "r_squared"


class MultipleTestingCorrection(Enum):
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    SIDAK = "sidak"
    BENJAMINI_HOCHBERG = "fdr_bh"
    BENJAMINI_YEKUTIELI = "fdr_by"
    FALSE_DISCOVERY_RATE = "fdr_tsbh"


@dataclass
class StatisticalTest:
    """Configuration for a statistical test."""
    test_name: str
    test_type: HypothesisTest
    data_groups: List[np.ndarray]
    alpha: float = 0.05
    alternative: str = "two-sided"  # 'two-sided', 'less', 'greater'
    effect_size_metrics: List[EffectSizeMetric] = field(default_factory=list)
    bootstrap_samples: int = 1000
    permutation_samples: int = 10000


@dataclass
class TestResult:
    """Results from a statistical test."""
    test_name: str
    test_type: HypothesisTest
    
    # Test statistics
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    
    # Effect sizes
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    effect_size_confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Confidence intervals
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # Bayesian analysis
    bayes_factor: Optional[float] = None
    rope_percentage: Optional[float] = None
    posterior_probability: Optional[float] = None
    
    # Power analysis
    statistical_power: Optional[float] = None
    minimum_detectable_effect: Optional[float] = None
    
    # Bootstrap results
    bootstrap_confidence_interval: Optional[Tuple[float, float]] = None
    bootstrap_bias: Optional[float] = None
    
    # Meta information
    sample_sizes: List[int] = field(default_factory=list)
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    computation_time: float = 0.0


@dataclass
class MultipleTestingResults:
    """Results from multiple testing correction."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected_hypotheses: List[bool]
    correction_method: MultipleTestingCorrection
    alpha: float
    number_of_discoveries: int
    false_discovery_rate: float


@dataclass
class BayesianTestResult:
    """Results from Bayesian hypothesis testing."""
    test_name: str
    bayes_factor: float
    posterior_probability_h1: float
    posterior_probability_h0: float
    rope_lower: float
    rope_upper: float
    rope_percentage: float
    evidence_strength: str  # 'decisive', 'strong', 'moderate', 'weak', 'inconclusive'
    posterior_samples: np.ndarray
    credible_interval: Tuple[float, float]


class EnhancedStatisticalValidator:
    """
    Enhanced statistical validation suite with modern statistical methods.
    
    Provides comprehensive statistical analysis including Bayesian methods,
    multiple testing correction, effect size estimation, and uncertainty
    quantification for robust AI research validation.
    """
    
    def __init__(self, 
                 workspace_dir: str = "/tmp/statistical_validation",
                 default_alpha: float = 0.05,
                 default_confidence_level: float = 0.95,
                 bootstrap_samples: int = 10000,
                 permutation_samples: int = 10000):
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_alpha = default_alpha
        self.default_confidence_level = default_confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.permutation_samples = permutation_samples
        
        logger.info(f"EnhancedStatisticalValidator initialized")
    
    def run_comprehensive_test(self, 
                             test_config: StatisticalTest,
                             include_bayesian: bool = True,
                             include_bootstrap: bool = True) -> TestResult:
        """
        Run comprehensive statistical test with multiple validation methods.
        
        Args:
            test_config: Statistical test configuration
            include_bayesian: Whether to include Bayesian analysis
            include_bootstrap: Whether to include bootstrap analysis
            
        Returns:
            TestResult with comprehensive statistical analysis
        """
        start_time = time.time()
        
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for statistical testing")
        
        logger.info(f"Running comprehensive test: {test_config.test_name}")
        
        # Initialize result
        result = TestResult(
            test_name=test_config.test_name,
            test_type=test_config.test_type,
            statistic=0.0,
            p_value=1.0,
            sample_sizes=[len(group) for group in test_config.data_groups]
        )
        
        try:
            # Run primary statistical test
            result = self._run_primary_test(test_config, result)
            
            # Calculate effect sizes
            if test_config.effect_size_metrics:
                result = self._calculate_effect_sizes(test_config, result)
            
            # Bayesian analysis
            if include_bayesian:
                result = self._run_bayesian_analysis(test_config, result)
            
            # Bootstrap analysis
            if include_bootstrap:
                result = self._run_bootstrap_analysis(test_config, result)
            
            # Power analysis
            result = self._run_power_analysis(test_config, result)
            
            # Check assumptions
            result = self._check_test_assumptions(test_config, result)
            
            result.computation_time = time.time() - start_time
            
            logger.info(f"Test completed: {test_config.test_name} "
                       f"(p={result.p_value:.6f}, power={result.statistical_power:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in statistical test {test_config.test_name}: {e}")
            result.warnings.append(f"Test failed: {str(e)}")
            result.computation_time = time.time() - start_time
            return result
    
    def _run_primary_test(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Run the primary statistical test."""
        
        if config.test_type == HypothesisTest.T_TEST:
            if len(config.data_groups) != 2:
                raise ValueError("T-test requires exactly 2 groups")
            
            statistic, p_value = ttest_ind(
                config.data_groups[0], 
                config.data_groups[1],
                alternative=config.alternative
            )
            
            result.statistic = statistic
            result.p_value = p_value
            result.degrees_of_freedom = len(config.data_groups[0]) + len(config.data_groups[1]) - 2
        
        elif config.test_type == HypothesisTest.MANN_WHITNEY:
            if len(config.data_groups) != 2:
                raise ValueError("Mann-Whitney U test requires exactly 2 groups")
            
            statistic, p_value = mannwhitneyu(
                config.data_groups[0],
                config.data_groups[1],
                alternative=config.alternative
            )
            
            result.statistic = statistic
            result.p_value = p_value
        
        elif config.test_type == HypothesisTest.WILCOXON:
            if len(config.data_groups) != 2:
                raise ValueError("Wilcoxon test requires exactly 2 groups")
            
            statistic, p_value = wilcoxon(
                config.data_groups[0],
                config.data_groups[1],
                alternative=config.alternative
            )
            
            result.statistic = statistic
            result.p_value = p_value
        
        elif config.test_type == HypothesisTest.KRUSKAL_WALLIS:
            statistic, p_value = kruskal(*config.data_groups)
            result.statistic = statistic
            result.p_value = p_value
            result.degrees_of_freedom = len(config.data_groups) - 1
        
        elif config.test_type == HypothesisTest.FRIEDMAN:
            statistic, p_value = friedmanchisquare(*config.data_groups)
            result.statistic = statistic
            result.p_value = p_value
            result.degrees_of_freedom = len(config.data_groups) - 1
        
        elif config.test_type == HypothesisTest.PERMUTATION:
            result = self._run_permutation_test(config, result)
        
        else:
            raise ValueError(f"Unsupported test type: {config.test_type}")
        
        return result
    
    def _run_permutation_test(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Run permutation test for non-parametric hypothesis testing."""
        
        if len(config.data_groups) != 2:
            raise ValueError("Permutation test implemented for 2 groups only")
        
        group1, group2 = config.data_groups
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Observed test statistic (difference in means)
        observed_diff = np.mean(group1) - np.mean(group2)
        
        # Permutation test
        permuted_diffs = []
        for _ in range(config.permutation_samples or self.permutation_samples):
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            permuted_diffs.append(perm_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        
        # Calculate p-value based on alternative hypothesis
        if config.alternative == "two-sided":
            p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        elif config.alternative == "greater":
            p_value = np.mean(permuted_diffs >= observed_diff)
        else:  # less
            p_value = np.mean(permuted_diffs <= observed_diff)
        
        result.statistic = observed_diff
        result.p_value = p_value
        
        return result
    
    def _calculate_effect_sizes(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Calculate effect sizes and their confidence intervals."""
        
        for effect_size_metric in config.effect_size_metrics:
            try:
                if effect_size_metric == EffectSizeMetric.COHENS_D:
                    effect_size, ci = self._cohens_d(config.data_groups)
                    result.effect_sizes["cohens_d"] = effect_size
                    result.effect_size_confidence_intervals["cohens_d"] = ci
                
                elif effect_size_metric == EffectSizeMetric.HEDGES_G:
                    effect_size, ci = self._hedges_g(config.data_groups)
                    result.effect_sizes["hedges_g"] = effect_size
                    result.effect_size_confidence_intervals["hedges_g"] = ci
                
                elif effect_size_metric == EffectSizeMetric.CLIFF_DELTA:
                    effect_size, ci = self._cliff_delta(config.data_groups)
                    result.effect_sizes["cliff_delta"] = effect_size
                    result.effect_size_confidence_intervals["cliff_delta"] = ci
                
                # Add more effect size metrics as needed
                
            except Exception as e:
                result.warnings.append(f"Failed to calculate {effect_size_metric.value}: {e}")
        
        return result
    
    def _cohens_d(self, data_groups: List[np.ndarray]) -> Tuple[float, Tuple[float, float]]:
        """Calculate Cohen's d effect size with confidence interval."""
        if len(data_groups) != 2:
            raise ValueError("Cohen's d requires exactly 2 groups")
        
        group1, group2 = data_groups
        n1, n2 = len(group1), len(group2)
        
        # Cohen's d calculation
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Confidence interval using bias correction
        # Hedges and Olkin (1985) formula
        j = 1 - (3 / (4 * (n1 + n2) - 9))  # bias correction factor
        cohens_d_corrected = j * cohens_d
        
        # Standard error
        se = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d_corrected**2 / (2 * (n1 + n2)))
        
        # 95% confidence interval
        z_score = stats.norm.ppf((1 + self.default_confidence_level) / 2)
        ci_lower = cohens_d_corrected - z_score * se
        ci_upper = cohens_d_corrected + z_score * se
        
        return cohens_d, (ci_lower, ci_upper)
    
    def _hedges_g(self, data_groups: List[np.ndarray]) -> Tuple[float, Tuple[float, float]]:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d, _ = self._cohens_d(data_groups)
        
        n1, n2 = len(data_groups[0]), len(data_groups[1])
        df = n1 + n2 - 2
        
        # Hedges' correction factor
        j = 1 - (3 / (4 * df - 1))
        hedges_g = j * cohens_d
        
        # Standard error for Hedges' g
        se = np.sqrt((n1 + n2) / (n1 * n2) + hedges_g**2 / (2 * (n1 + n2 - 2)))
        
        # 95% confidence interval
        z_score = stats.norm.ppf((1 + self.default_confidence_level) / 2)
        ci_lower = hedges_g - z_score * se
        ci_upper = hedges_g + z_score * se
        
        return hedges_g, (ci_lower, ci_upper)
    
    def _cliff_delta(self, data_groups: List[np.ndarray]) -> Tuple[float, Tuple[float, float]]:
        """Calculate Cliff's delta (non-parametric effect size)."""
        if len(data_groups) != 2:
            raise ValueError("Cliff's delta requires exactly 2 groups")
        
        group1, group2 = data_groups
        n1, n2 = len(group1), len(group2)
        
        # Calculate Cliff's delta
        dominance_matrix = group1[:, np.newaxis] > group2[np.newaxis, :]
        ties_matrix = group1[:, np.newaxis] == group2[np.newaxis, :]
        
        dominance_count = np.sum(dominance_matrix)
        ties_count = np.sum(ties_matrix)
        
        cliff_delta = (dominance_count - (n1 * n2 - dominance_count - ties_count)) / (n1 * n2)
        
        # Bootstrap confidence interval
        bootstrap_deltas = []
        for _ in range(1000):
            boot_group1 = np.random.choice(group1, size=n1, replace=True)
            boot_group2 = np.random.choice(group2, size=n2, replace=True)
            
            boot_dom_matrix = boot_group1[:, np.newaxis] > boot_group2[np.newaxis, :]
            boot_ties_matrix = boot_group1[:, np.newaxis] == boot_group2[np.newaxis, :]
            
            boot_dom_count = np.sum(boot_dom_matrix)
            boot_ties_count = np.sum(boot_ties_matrix)
            
            boot_delta = (boot_dom_count - (n1 * n2 - boot_dom_count - boot_ties_count)) / (n1 * n2)
            bootstrap_deltas.append(boot_delta)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_deltas, (1 - self.default_confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_deltas, (1 + self.default_confidence_level) / 2 * 100)
        
        return cliff_delta, (ci_lower, ci_upper)
    
    def _run_bayesian_analysis(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Run Bayesian hypothesis testing."""
        
        if len(config.data_groups) != 2:
            result.warnings.append("Bayesian analysis implemented for 2-group comparisons only")
            return result
        
        try:
            # Bayesian t-test approximation using BIC approximation
            group1, group2 = config.data_groups
            n1, n2 = len(group1), len(group2)
            
            # Calculate t-statistic and degrees of freedom
            t_stat, _ = ttest_ind(group1, group2)
            df = n1 + n2 - 2
            
            # BIC approximation for Bayes Factor
            # BF10 = exp((BIC_null - BIC_alternative) / 2)
            bic_null = n1 * np.log(np.var(group1, ddof=1)) + n2 * np.log(np.var(group2, ddof=1))
            
            # For alternative, use pooled variance
            pooled_var = ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / df
            bic_alt = (n1 + n2) * np.log(pooled_var)
            
            # Rough approximation - would be more sophisticated in practice
            log_bf10 = (bic_null - bic_alt) / 2
            bf10 = np.exp(log_bf10)
            
            result.bayes_factor = bf10
            
            # Posterior probability (assuming equal priors)
            result.posterior_probability = bf10 / (1 + bf10)
            
            # ROPE analysis (Region of Practical Equivalence)
            # Define ROPE as [-0.1, 0.1] in standardized effect size
            rope_lower, rope_upper = -0.1, 0.1
            
            # Estimate percentage of posterior distribution within ROPE
            # This is a simplified approximation
            mean_diff = np.mean(group1) - np.mean(group2)
            se_diff = np.sqrt(np.var(group1, ddof=1)/n1 + np.var(group2, ddof=1)/n2)
            
            # Standardize
            standardized_diff = mean_diff / np.sqrt(pooled_var)
            standardized_se = se_diff / np.sqrt(pooled_var)
            
            # Percentage in ROPE (approximation using normal distribution)
            rope_prob = (stats.norm.cdf(rope_upper, standardized_diff, standardized_se) - 
                        stats.norm.cdf(rope_lower, standardized_diff, standardized_se))
            
            result.rope_percentage = rope_prob
            
        except Exception as e:
            result.warnings.append(f"Bayesian analysis failed: {e}")
        
        return result
    
    def _run_bootstrap_analysis(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Run bootstrap analysis for confidence intervals."""
        
        if len(config.data_groups) != 2:
            result.warnings.append("Bootstrap analysis implemented for 2-group comparisons only")
            return result
        
        try:
            group1, group2 = config.data_groups
            
            # Bootstrap resampling for mean difference
            bootstrap_diffs = []
            original_diff = np.mean(group1) - np.mean(group2)
            
            for _ in range(self.bootstrap_samples):
                boot_group1 = np.random.choice(group1, size=len(group1), replace=True)
                boot_group2 = np.random.choice(group2, size=len(group2), replace=True)
                boot_diff = np.mean(boot_group1) - np.mean(boot_group2)
                bootstrap_diffs.append(boot_diff)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # Bootstrap confidence interval
            alpha = 1 - self.default_confidence_level
            ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
            
            result.bootstrap_confidence_interval = (ci_lower, ci_upper)
            
            # Bootstrap bias estimation
            result.bootstrap_bias = np.mean(bootstrap_diffs) - original_diff
            
        except Exception as e:
            result.warnings.append(f"Bootstrap analysis failed: {e}")
        
        return result
    
    def _run_power_analysis(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Run power analysis."""
        
        if not STATSMODELS_AVAILABLE or len(config.data_groups) != 2:
            return result
        
        try:
            group1, group2 = config.data_groups
            n1, n2 = len(group1), len(group2)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(group1) - np.mean(group2)
            pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1))/(n1+n2-2))
            effect_size = abs(mean_diff) / pooled_std
            
            # Calculate power
            power = ttest_power(effect_size, n1, config.alpha, alternative=config.alternative)
            result.statistical_power = power
            
            # Minimum detectable effect (for 80% power)
            if power < 0.8:
                min_effect = ttest_power(None, n1, config.alpha, 0.8, alternative=config.alternative)
                result.minimum_detectable_effect = min_effect
            
        except Exception as e:
            result.warnings.append(f"Power analysis failed: {e}")
        
        return result
    
    def _check_test_assumptions(self, config: StatisticalTest, result: TestResult) -> TestResult:
        """Check statistical test assumptions."""
        
        assumptions = {}
        
        try:
            if config.test_type in [HypothesisTest.T_TEST]:
                # Check normality
                for i, group in enumerate(config.data_groups):
                    if len(group) >= 3:  # Minimum for Shapiro-Wilk
                        _, p_value = shapiro(group)
                        assumptions[f"normality_group_{i+1}"] = p_value > 0.05
                
                # Check equal variances
                if len(config.data_groups) == 2:
                    _, p_value = levene(*config.data_groups)
                    assumptions["equal_variances"] = p_value > 0.05
            
            result.assumptions_met = assumptions
            
        except Exception as e:
            result.warnings.append(f"Assumption checking failed: {e}")
        
        return result
    
    def correct_multiple_testing(self, 
                                p_values: List[float],
                                method: MultipleTestingCorrection = MultipleTestingCorrection.BENJAMINI_HOCHBERG,
                                alpha: float = 0.05) -> MultipleTestingResults:
        """
        Apply multiple testing correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method to use
            alpha: Family-wise error rate or false discovery rate
            
        Returns:
            MultipleTestingResults with corrected p-values
        """
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available, using simple Bonferroni correction")
            corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]
            rejected = [p < alpha for p in corrected_p_values]
        else:
            rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method.value
            )
        
        return MultipleTestingResults(
            original_p_values=p_values,
            corrected_p_values=list(corrected_p_values),
            rejected_hypotheses=list(rejected),
            correction_method=method,
            alpha=alpha,
            number_of_discoveries=sum(rejected),
            false_discovery_rate=sum(rejected) / max(1, len(p_values))
        )
    
    def run_meta_analysis(self, 
                         effect_sizes: List[float],
                         standard_errors: List[float],
                         study_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run meta-analysis to combine effect sizes across studies.
        
        Args:
            effect_sizes: List of effect sizes from different studies
            standard_errors: List of standard errors for each effect size
            study_labels: Optional labels for studies
            
        Returns:
            Meta-analysis results dictionary
        """
        
        if not STATSMODELS_AVAILABLE:
            # Simple fixed-effects meta-analysis
            weights = [1/se**2 for se in standard_errors]
            total_weight = sum(weights)
            pooled_effect = sum(es * w for es, w in zip(effect_sizes, weights)) / total_weight
            pooled_se = np.sqrt(1 / total_weight)
            
            return {
                'pooled_effect_size': pooled_effect,
                'pooled_standard_error': pooled_se,
                'confidence_interval': (
                    pooled_effect - 1.96 * pooled_se,
                    pooled_effect + 1.96 * pooled_se
                ),
                'heterogeneity_statistic': None,
                'p_value': 2 * (1 - stats.norm.cdf(abs(pooled_effect / pooled_se)))
            }
        
        # Use statsmodels for more sophisticated meta-analysis
        try:
            # Fixed effects meta-analysis
            fe_result = combine_effects(effect_sizes, standard_errors, method_re="fixed")
            
            return {
                'pooled_effect_size': fe_result.effect,
                'pooled_standard_error': fe_result.se,
                'confidence_interval': fe_result.conf_int(),
                'heterogeneity_statistic': fe_result.het_statistic,
                'p_value': fe_result.pvalue,
                'method': 'fixed_effects'
            }
            
        except Exception as e:
            logger.error(f"Meta-analysis failed: {e}")
            return {'error': str(e)}
    
    def plot_test_results(self, 
                         results: List[TestResult],
                         save_path: Optional[str] = None) -> None:
        """Plot statistical test results."""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # P-values histogram
        p_values = [r.p_value for r in results if r.p_value is not None]
        if p_values:
            axes[0, 0].hist(p_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[0, 0].set_xlabel('P-values')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of P-values')
            axes[0, 0].legend()
        
        # Effect sizes
        effect_sizes = []
        test_names = []
        for r in results:
            if 'cohens_d' in r.effect_sizes:
                effect_sizes.append(r.effect_sizes['cohens_d'])
                test_names.append(r.test_name[:10] + '...' if len(r.test_name) > 10 else r.test_name)
        
        if effect_sizes:
            axes[0, 1].barh(range(len(effect_sizes)), effect_sizes)
            axes[0, 1].set_yticks(range(len(effect_sizes)))
            axes[0, 1].set_yticklabels(test_names)
            axes[0, 1].set_xlabel("Cohen's d")
            axes[0, 1].set_title('Effect Sizes')
        
        # Statistical power
        powers = [r.statistical_power for r in results if r.statistical_power is not None]
        if powers:
            axes[1, 0].hist(powers, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0.8, color='red', linestyle='--', label='Power = 0.8')
            axes[1, 0].set_xlabel('Statistical Power')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Statistical Power')
            axes[1, 0].legend()
        
        # P-value vs Effect Size scatter
        scatter_p_values = []
        scatter_effects = []
        for r in results:
            if r.p_value is not None and 'cohens_d' in r.effect_sizes:
                scatter_p_values.append(r.p_value)
                scatter_effects.append(abs(r.effect_sizes['cohens_d']))
        
        if scatter_p_values:
            axes[1, 1].scatter(scatter_effects, scatter_p_values, alpha=0.6)
            axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[1, 1].set_xlabel('|Effect Size| (Cohen\'s d)')
            axes[1, 1].set_ylabel('P-value')
            axes[1, 1].set_title('P-value vs Effect Size')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Test results plot saved to {save_path}")
        
        plt.show()


# Example usage and testing functions
def test_enhanced_validation():
    """Test enhanced statistical validation suite."""
    
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 30)  # Control group
    group2 = np.random.normal(108, 16, 28)  # Treatment group (small effect)
    group3 = np.random.normal(85, 12, 25)   # Additional group
    
    # Initialize validator
    validator = EnhancedStatisticalValidator()
    
    # Test configuration
    test_config = StatisticalTest(
        test_name="Treatment vs Control",
        test_type=HypothesisTest.T_TEST,
        data_groups=[group1, group2],
        alpha=0.05,
        alternative="two-sided",
        effect_size_metrics=[
            EffectSizeMetric.COHENS_D,
            EffectSizeMetric.HEDGES_G,
            EffectSizeMetric.CLIFF_DELTA
        ]
    )
    
    # Run comprehensive test
    result = validator.run_comprehensive_test(test_config)
    
    print(f"\nEnhanced Statistical Validation Results:")
    print(f"Test: {result.test_name}")
    print(f"P-value: {result.p_value:.6f}")
    print(f"Effect sizes: {result.effect_sizes}")
    print(f"Statistical power: {result.statistical_power:.3f}")
    print(f"Bayes factor: {result.bayes_factor:.3f}")
    print(f"Bootstrap CI: {result.bootstrap_confidence_interval}")
    
    # Test multiple testing correction
    p_values = [0.001, 0.02, 0.045, 0.08, 0.12, 0.3]
    correction_result = validator.correct_multiple_testing(
        p_values, MultipleTestingCorrection.BENJAMINI_HOCHBERG
    )
    
    print(f"\nMultiple Testing Correction:")
    print(f"Original p-values: {p_values}")
    print(f"Corrected p-values: {correction_result.corrected_p_values}")
    print(f"Rejected hypotheses: {correction_result.rejected_hypotheses}")
    print(f"Number of discoveries: {correction_result.number_of_discoveries}")
    
    return result


if __name__ == "__main__":
    # Test the enhanced statistical validation suite
    test_enhanced_validation()