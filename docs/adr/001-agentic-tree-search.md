# ADR-0001: Use Agentic Tree Search for Experiment Management

## Status
Accepted

## Context
The AI Scientist v2 requires a systematic approach to explore experimental hypotheses and manage the complexity of autonomous scientific research. Traditional linear experimentation approaches are limited in their ability to:

- Explore multiple experimental paths simultaneously
- Recover from failed experiments intelligently
- Maintain state across complex experimental workflows
- Scale experimentation across diverse research domains

## Decision
Implement an agentic tree search system based on best-first tree search (BFTS) principles for managing experiment execution and exploration.

## Alternatives Considered

### Linear Sequential Execution
- **Pros**: Simple to implement, predictable execution flow
- **Cons**: Cannot explore multiple paths, poor failure recovery, limited scalability

### Multi-Agent Parallel Execution  
- **Pros**: High parallelism, independent agent operation
- **Cons**: No shared state, difficult coordination, no systematic exploration

### Graph-Based Workflow Management
- **Pros**: Clear dependencies, good for known workflows
- **Cons**: Requires predefined structure, limited adaptability

## Decision Rationale
The agentic tree search approach provides:

1. **Systematic Exploration**: Tree structure allows methodical exploration of research paths
2. **Intelligent Recovery**: Debug mechanisms enable recovery from failed experiments
3. **Parallel Efficiency**: Multiple workers can explore different branches simultaneously
4. **State Management**: Tree nodes maintain experiment state and context
5. **Adaptability**: Dynamic branching based on experiment outcomes

## Implementation Details

### Core Components
- `agent_manager.py`: Coordinates multiple worker agents
- `parallel_agent.py`: Individual experiment execution units
- Tree visualization for experiment tracking
- Configurable search parameters via `bfts_config.yaml`

### Key Parameters
- `num_workers`: Number of parallel exploration paths (typically 3)
- `steps`: Maximum nodes to explore (typically 21)
- `max_debug_depth`: Error recovery attempts (typically 3)
- `debug_prob`: Probability of debugging failed nodes

## Consequences

### Positive
- Enables sophisticated experiment management
- Provides clear visualization of research exploration
- Allows for intelligent failure recovery
- Scales well with available computational resources
- Maintains research context across experiments

### Negative  
- Increased complexity compared to linear approaches
- Requires careful tuning of search parameters
- Higher memory usage for maintaining tree state
- Learning curve for understanding tree search mechanics

### Technical Debt
- Tree state management complexity
- Need for robust cleanup mechanisms
- Potential for resource leaks in long-running searches

## Related Decisions
- ADR-0002: Multi-Model LLM Integration Strategy
- ADR-0003: Security-First Development Approach