"""TreeSearchAgent: BFS/DFS/MCTS-inspired expansion of hypothesis tree."""
from __future__ import annotations
from collections import deque
from typing import Callable, Optional
import random
import math

from .hypothesis_tree import HypothesisTree, HypothesisNode


HypothesisGenerator = Callable[[HypothesisNode], list[str]]


def _default_generator(node: HypothesisNode) -> list[str]:
    """Generate sub-hypotheses by appending mechanistic variants."""
    templates = [
        f"{node.hypothesis} — mediated by mechanism A",
        f"{node.hypothesis} — dependent on variable X",
        f"{node.hypothesis} — with boundary condition B",
    ]
    return templates


class TreeSearchAgent:
    """Expand a HypothesisTree using BFS, DFS, or MCTS strategies."""

    def __init__(
        self,
        tree: HypothesisTree,
        generator: Optional[HypothesisGenerator] = None,
        strategy: str = "bfs",
        max_nodes: int = 20,
        max_depth: int = 4,
    ):
        self.tree = tree
        self.generator = generator or _default_generator
        self.strategy = strategy
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def expand(self) -> int:
        """Expand the tree. Returns number of nodes added."""
        if self.strategy == "bfs":
            return self._bfs()
        elif self.strategy == "dfs":
            return self._dfs()
        elif self.strategy == "mcts":
            return self._mcts()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _bfs(self) -> int:
        added = 0
        queue: deque[str] = deque([self.tree.root_id])
        while queue and self.tree.size() < self.max_nodes:
            node_id = queue.popleft()
            node = self.tree.get(node_id)
            if node.depth >= self.max_depth:
                continue
            children_hyps = self.generator(node)
            for hyp in children_hyps:
                if self.tree.size() >= self.max_nodes:
                    break
                child = self.tree.add_child(node_id, hyp)
                queue.append(child.node_id)
                added += 1
        return added

    def _dfs(self) -> int:
        added = 0
        stack: list[str] = [self.tree.root_id]
        while stack and self.tree.size() < self.max_nodes:
            node_id = stack.pop()
            node = self.tree.get(node_id)
            if node.depth >= self.max_depth:
                continue
            children_hyps = self.generator(node)
            for hyp in reversed(children_hyps):
                if self.tree.size() >= self.max_nodes:
                    break
                child = self.tree.add_child(node_id, hyp)
                stack.append(child.node_id)
                added += 1
        return added

    def _uct_score(self, node: HypothesisNode, parent_visits: int, c: float = 1.41) -> float:
        """Upper Confidence Bound for Trees score."""
        visits = max(len(node.evidence), 1)
        exploitation = node.confidence
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / visits)
        return exploitation + exploration

    def _mcts(self) -> int:
        added = 0
        iterations = min(self.max_nodes - self.tree.size(), 10)
        for _ in range(iterations):
            if self.tree.size() >= self.max_nodes:
                break
            # Selection: pick best leaf by UCT
            leaves = self.tree.leaves()
            expandable = [n for n in leaves if n.depth < self.max_depth]
            if not expandable:
                break
            n_total = self.tree.size()
            selected = max(expandable, key=lambda n: self._uct_score(n, n_total))
            # Expansion: add one child
            children_hyps = self.generator(selected)
            if not children_hyps:
                continue
            hyp = children_hyps[0]
            confidence = random.uniform(0.3, 0.9)
            self.tree.add_child(selected.node_id, hyp, confidence=confidence)
            added += 1
        return added
