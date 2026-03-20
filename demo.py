#!/usr/bin/env python3
"""Demo: automated scientific discovery on a synthetic dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_scientist.hypothesis_tree import HypothesisTree
from ai_scientist.tree_search import TreeSearchAgent
from ai_scientist.experiment_simulator import ExperimentSimulator
from ai_scientist.paper_generator import PaperDraftGenerator


def main():
    print("AI Scientist v2 — Automated Discovery Demo")
    print("=" * 50)

    # 1. Initialize hypothesis tree
    root = "Oscillatory dynamics govern information transmission in biological networks"
    tree = HypothesisTree(root)

    # 2. Expand tree via BFS
    agent = TreeSearchAgent(tree, strategy="bfs", max_nodes=12, max_depth=3)
    added = agent.expand()
    print(f"Tree expanded: {tree.size()} nodes ({added} added)")

    # 3. Evaluate each leaf with experiments
    simulator = ExperimentSimulator(noise_level=0.15)
    for leaf in tree.leaves():
        result = simulator.evaluate(leaf)
        print(f"  [{leaf.node_id}] conf={leaf.confidence:.2f}  {result.description[:60]}")

    # 4. Generate paper draft
    generator = PaperDraftGenerator(tree)
    draft = generator.generate(dataset_description="synthetic oscillatory dataset")

    print(f"\nTitle: {draft.title}")
    print(f"\nAbstract:\n{draft.abstract}")
    print(f"\n{draft.methods}")
    print(f"\nFinal confidence: {draft.confidence:.2f}")


if __name__ == "__main__":
    main()
