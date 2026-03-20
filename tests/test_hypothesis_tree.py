"""Tests for HypothesisTree and HypothesisNode."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_scientist.hypothesis_tree import HypothesisTree, HypothesisNode


def make_tree():
    return HypothesisTree("Root hypothesis")


def test_tree_has_root():
    tree = make_tree()
    assert tree.root.hypothesis == "Root hypothesis"


def test_tree_initial_size():
    tree = make_tree()
    assert tree.size() == 1


def test_add_child():
    tree = make_tree()
    child = tree.add_child(tree.root_id, "Child hypothesis")
    assert tree.size() == 2
    assert child.hypothesis == "Child hypothesis"


def test_child_depth():
    tree = make_tree()
    child = tree.add_child(tree.root_id, "Child")
    assert child.depth == 1


def test_grandchild_depth():
    tree = make_tree()
    child = tree.add_child(tree.root_id, "Child")
    grandchild = tree.add_child(child.node_id, "Grandchild")
    assert grandchild.depth == 2


def test_leaves_root_only():
    tree = make_tree()
    assert len(tree.leaves()) == 1


def test_leaves_after_add():
    tree = make_tree()
    tree.add_child(tree.root_id, "Child 1")
    tree.add_child(tree.root_id, "Child 2")
    assert len(tree.leaves()) == 2


def test_best_path_returns_list():
    tree = make_tree()
    path = tree.best_path()
    assert isinstance(path, list)
    assert len(path) >= 1


def test_best_path_highest_confidence():
    tree = make_tree()
    c1 = tree.add_child(tree.root_id, "Low conf", confidence=0.2)
    c2 = tree.add_child(tree.root_id, "High conf", confidence=0.9)
    path = tree.best_path()
    assert path[-1].node_id == c2.node_id


def test_add_evidence():
    tree = make_tree()
    tree.root.add_evidence("Evidence 1", 0.1)
    assert len(tree.root.evidence) == 1
    assert tree.root.confidence > 0.5


def test_max_depth():
    tree = make_tree()
    child = tree.add_child(tree.root_id, "Child")
    tree.add_child(child.node_id, "Grandchild")
    assert tree.max_depth() == 2
