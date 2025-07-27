# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the AI Scientist v2 project.

## About ADRs

Architecture Decision Records (ADRs) are lightweight documents that capture important architectural decisions made during the project development. Each ADR describes:

- The decision that was made
- The context that led to the decision
- The alternatives that were considered
- The consequences of the decision

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXXX: Title

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
Description of the issue or situation that prompted this decision.

## Decision
The decision that was made and rationale behind it.

## Alternatives Considered
Other options that were evaluated.

## Consequences
Positive and negative impacts of this decision.
```

## Index of ADRs

- [ADR-0001: Use Agentic Tree Search for Experiment Management](001-agentic-tree-search.md)
- [ADR-0002: Multi-Model LLM Integration Strategy](002-multi-model-llm.md)
- [ADR-0003: Security-First Development Approach](003-security-first.md)
- [ADR-0004: Python-Based Implementation](004-python-implementation.md)

## Creating New ADRs

1. Copy the template: `cp template.md adr-XXXX-title.md`
2. Fill in the sections
3. Update this index
4. Submit as part of your PR