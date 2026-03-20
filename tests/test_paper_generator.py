"""Tests for PaperDraftGenerator."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_scientist.hypothesis_tree import HypothesisTree
from ai_scientist.tree_search import TreeSearchAgent
from ai_scientist.experiment_simulator import ExperimentSimulator
from ai_scientist.paper_generator import PaperDraftGenerator, PaperDraft


def make_evaluated_tree():
    tree = HypothesisTree("Oscillations drive learning in neural circuits")
    agent = TreeSearchAgent(tree, strategy="bfs", max_nodes=6, max_depth=2)
    agent.expand()
    sim = ExperimentSimulator(seed=42)
    for leaf in tree.leaves():
        sim.evaluate(leaf)
    return tree


def test_generate_returns_draft():
    tree = make_evaluated_tree()
    gen = PaperDraftGenerator(tree)
    draft = gen.generate()
    assert isinstance(draft, PaperDraft)


def test_draft_has_title():
    tree = make_evaluated_tree()
    gen = PaperDraftGenerator(tree)
    draft = gen.generate()
    assert len(draft.title) > 0


def test_draft_has_abstract():
    tree = make_evaluated_tree()
    gen = PaperDraftGenerator(tree)
    draft = gen.generate()
    assert len(draft.abstract) > 50


def test_draft_has_methods():
    tree = make_evaluated_tree()
    gen = PaperDraftGenerator(tree)
    draft = gen.generate()
    assert "Methods" in draft.methods


def test_draft_confidence_range():
    tree = make_evaluated_tree()
    gen = PaperDraftGenerator(tree)
    draft = gen.generate()
    assert 0.0 <= draft.confidence <= 1.0


def test_empty_tree_raises():
    tree = HypothesisTree("Root")
    gen = PaperDraftGenerator(tree)
    # Single root leaf should still work (not raise)
    draft = gen.generate()
    assert draft is not None
