"""HypothesisTree: tree of research hypotheses with evidence and confidence."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class HypothesisNode:
    hypothesis: str
    confidence: float = 0.5           # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    children: list[str] = field(default_factory=list)  # child node_ids
    depth: int = 0
    experiment_result: Optional[dict] = None

    def add_evidence(self, evidence: str, confidence_delta: float = 0.0) -> None:
        """Add evidence and optionally update confidence."""
        self.evidence.append(evidence)
        self.confidence = max(0.0, min(1.0, self.confidence + confidence_delta))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"HypothesisNode(id={self.node_id}, conf={self.confidence:.2f}, h='{self.hypothesis[:40]}')"


class HypothesisTree:
    """A tree of research hypotheses for agentic scientific exploration."""

    def __init__(self, root_hypothesis: str):
        root = HypothesisNode(hypothesis=root_hypothesis, depth=0)
        self._nodes: dict[str, HypothesisNode] = {root.node_id: root}
        self.root_id = root.node_id

    @property
    def root(self) -> HypothesisNode:
        return self._nodes[self.root_id]

    def get(self, node_id: str) -> HypothesisNode:
        return self._nodes[node_id]

    def add_child(self, parent_id: str, hypothesis: str, confidence: float = 0.5) -> HypothesisNode:
        """Add a child hypothesis to a parent node."""
        parent = self._nodes[parent_id]
        child = HypothesisNode(
            hypothesis=hypothesis,
            confidence=confidence,
            parent_id=parent_id,
            depth=parent.depth + 1,
        )
        self._nodes[child.node_id] = child
        parent.children.append(child.node_id)
        return child

    def leaves(self) -> list[HypothesisNode]:
        """Return all leaf nodes."""
        return [n for n in self._nodes.values() if n.is_leaf()]

    def best_path(self) -> list[HypothesisNode]:
        """Return the path from root to the highest-confidence leaf."""
        if not self._nodes:
            return []
        best_leaf = max(self.leaves(), key=lambda n: n.confidence)
        return self._path_to(best_leaf.node_id)

    def _path_to(self, node_id: str) -> list[HypothesisNode]:
        """Return list of nodes from root to given node."""
        path = []
        current_id: Optional[str] = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def all_nodes(self) -> list[HypothesisNode]:
        return list(self._nodes.values())

    def size(self) -> int:
        return len(self._nodes)

    def max_depth(self) -> int:
        return max(n.depth for n in self._nodes.values())
