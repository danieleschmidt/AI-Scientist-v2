"""Tests for ExperimentSimulator."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_scientist.hypothesis_tree import HypothesisTree
from ai_scientist.experiment_simulator import ExperimentSimulator, ExperimentResult


def make_node(confidence=0.8):
    tree = HypothesisTree("Test hypothesis")
    tree.root.confidence = confidence
    return tree.root


def test_evaluate_returns_result():
    sim = ExperimentSimulator(seed=42)
    node = make_node()
    result = sim.evaluate(node)
    assert isinstance(result, ExperimentResult)


def test_result_has_p_value():
    sim = ExperimentSimulator(seed=42)
    result = sim.evaluate(make_node())
    assert 0.0 <= result.p_value <= 1.0


def test_result_has_r_squared():
    sim = ExperimentSimulator(seed=42)
    result = sim.evaluate(make_node())
    assert isinstance(result.r_squared, float)


def test_result_has_effect_size():
    sim = ExperimentSimulator(seed=42)
    result = sim.evaluate(make_node())
    assert result.effect_size >= 0


def test_node_evidence_updated():
    sim = ExperimentSimulator(seed=42)
    node = make_node()
    sim.evaluate(node)
    assert len(node.evidence) > 0


def test_node_experiment_result_set():
    sim = ExperimentSimulator(seed=42)
    node = make_node()
    sim.evaluate(node)
    assert node.experiment_result is not None
