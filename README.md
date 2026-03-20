# AI-Scientist-v2

An agentic framework for automated scientific discovery via hypothesis tree search. Implements workshop-level research automation: generate hypotheses, design experiments, simulate results, and draft papers.

## Architecture

- **HypothesisTree** — tree structure where each node is a (hypothesis, evidence, confidence) tuple
- **TreeSearchAgent** — BFS/DFS/MCTS-inspired expansion of the hypothesis tree
- **ExperimentSimulator** — evaluate a hypothesis by running a toy experiment (curve fitting, statistical test)
- **PaperDraftGenerator** — generate a structured abstract + methods section from the best hypothesis path

## Requirements

- Python 3.9+
- numpy, scipy

## Quick Demo

```bash
python demo.py
```

## Testing

```bash
pip install pytest numpy scipy
pytest tests/ -v
```
