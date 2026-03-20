"""Tests for TreeSearchAgent."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_scientist.hypothesis_tree import HypothesisTree
from ai_scientist.tree_search import TreeSearchAgent


def make_agent(strategy="bfs", max_nodes=10, max_depth=3):
    tree = HypothesisTree("Root")
    agent = TreeSearchAgent(tree, strategy=strategy, max_nodes=max_nodes, max_depth=max_depth)
    return tree, agent


def test_bfs_adds_nodes():
    tree, agent = make_agent("bfs", max_nodes=5)
    added = agent.expand()
    assert added > 0
    assert tree.size() > 1


def test_dfs_adds_nodes():
    tree, agent = make_agent("dfs", max_nodes=5)
    added = agent.expand()
    assert added > 0


def test_mcts_adds_nodes():
    tree, agent = make_agent("mcts", max_nodes=5)
    added = agent.expand()
    assert added > 0


def test_max_nodes_respected():
    tree, agent = make_agent("bfs", max_nodes=7)
    agent.expand()
    assert tree.size() <= 7


def test_max_depth_respected():
    tree, agent = make_agent("bfs", max_nodes=100, max_depth=2)
    agent.expand()
    assert tree.max_depth() <= 2


def test_unknown_strategy_raises():
    tree = HypothesisTree("Root")
    agent = TreeSearchAgent(tree, strategy="unknown")
    with pytest.raises(ValueError):
        agent.expand()
