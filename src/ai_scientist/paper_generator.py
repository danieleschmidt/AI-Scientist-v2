"""PaperDraftGenerator: generate abstract + methods from best hypothesis path."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .hypothesis_tree import HypothesisNode, HypothesisTree


@dataclass
class PaperDraft:
    title: str
    abstract: str
    methods: str
    hypothesis_path: list[str]
    confidence: float


class PaperDraftGenerator:
    """Generate a structured research paper draft from the best hypothesis path."""

    def __init__(self, tree: HypothesisTree):
        self.tree = tree

    def generate(self, dataset_description: str = "synthetic dataset") -> PaperDraft:
        """Generate a draft from the best-confidence path in the tree."""
        path = self.tree.best_path()
        if not path:
            raise ValueError("Tree is empty")

        leaf = path[-1]
        title = self._generate_title(leaf)
        abstract = self._generate_abstract(path, dataset_description)
        methods = self._generate_methods(path)
        confidence = leaf.confidence

        return PaperDraft(
            title=title,
            abstract=abstract,
            methods=methods,
            hypothesis_path=[n.hypothesis for n in path],
            confidence=confidence,
        )

    def _generate_title(self, leaf: HypothesisNode) -> str:
        h = leaf.hypothesis
        # Capitalize and turn into title-case paper title
        return f"Evidence for: {h[:80]}"

    def _generate_abstract(self, path: list[HypothesisNode], dataset: str) -> str:
        root = path[0]
        leaf = path[-1]
        depth = len(path)
        conf = leaf.confidence

        supported_count = sum(
            1 for n in path if n.experiment_result and n.experiment_result.get("r_squared", 0) > 0.7
        )

        abstract = (
            f"We investigate the hypothesis: \"{root.hypothesis}\". "
            f"Using a {dataset}, we conducted {depth} rounds of hypothesis refinement "
            f"via agentic tree search, evaluating {depth} intermediate sub-hypotheses. "
            f"Our experiments provide {supported_count} confirmatory observations. "
            f"The refined hypothesis \"{leaf.hypothesis}\" achieves a confidence score of {conf:.2f}. "
            f"We demonstrate that agentic hypothesis tree search can efficiently navigate "
            f"the hypothesis space to converge on supported scientific claims."
        )
        return abstract

    def _generate_methods(self, path: list[HypothesisNode]) -> str:
        lines = [
            "## Methods",
            "",
            "### Hypothesis Tree Construction",
            f"We initialize a HypothesisTree with the root hypothesis and expand it using "
            f"a TreeSearchAgent (BFS strategy, max_depth={len(path)}).",
            "",
            "### Experiment Simulation",
            "Each hypothesis node is evaluated with two complementary tests:",
            "1. **Curve fitting**: synthetic sinusoidal data is generated; "
            "   an A·sin(x+φ) model is fit and R² computed.",
            "2. **Statistical test**: a two-sample Welch's t-test compares treatment "
            "   and control groups; effect size (Cohen's d) is reported.",
            "",
            "### Hypothesis Path",
        ]
        for i, node in enumerate(path):
            result = node.experiment_result
            if result:
                lines.append(
                    f"  {i+1}. {node.hypothesis[:60]} "
                    f"(R²={result.get('r_squared',0):.2f}, p={result.get('p_value',1):.3f})"
                )
            else:
                lines.append(f"  {i+1}. {node.hypothesis[:60]}")
        return "\n".join(lines)
