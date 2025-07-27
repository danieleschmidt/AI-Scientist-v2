# AI Scientist v2 - System Architecture

## Overview

The AI Scientist v2 is a generalized end-to-end agentic system for autonomous scientific discovery that combines Large Language Models (LLMs) with progressive agentic tree search to generate, execute, and write scientific research papers.

## Problem Statement

Traditional scientific research requires significant human expertise and time investment. The AI Scientist v2 aims to automate the entire scientific discovery pipeline, from hypothesis generation to paper writing, enabling rapid exploration of research domains with minimal human intervention.

## Success Criteria

- **Autonomous Research**: Complete end-to-end research pipeline without human templates
- **Cross-Domain Generalization**: Works across multiple Machine Learning domains
- **Quality Output**: Generates workshop-level papers accepted through peer review
- **Reproducibility**: Consistent results across different research topics
- **Safety**: Controlled execution environment preventing malicious code execution

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Scientist v2                         │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Layer                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   CLI Interface │ │  Configuration  │ │   Monitoring    │   │
│  │                 │ │    Management   │ │   Dashboard     │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Experiment      │ │  Agent Manager  │ │  Resource       │   │
│  │ Launcher        │ │                 │ │  Monitor        │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Core Engine Layer                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Ideation      │ │  Tree Search    │ │  Paper Writing  │   │
│  │   Engine        │ │    Engine       │ │    Engine       │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Service Layer                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   LLM Gateway   │ │  Code Execution │ │  Literature     │   │
│  │   (Multi-Model) │ │    Sandbox      │ │  Search         │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   GPU Resource  │ │  Storage        │ │  Security       │   │
│  │   Management    │ │  Management     │ │  Framework      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ideation Phase**: Topic description → Research ideas → JSON structured output
2. **Tree Search Phase**: Ideas → Experiment execution → Result analysis → Best path selection
3. **Writing Phase**: Results → Literature search → Paper generation → Review & refinement

### Key Design Patterns

- **Agent-based Architecture**: Specialized agents for different research phases
- **Tree Search Strategy**: Best-first tree search (BFTS) for experiment exploration
- **Multi-model Support**: OpenAI, Anthropic, Gemini model integration
- **Sandbox Execution**: Isolated environment for safe code execution
- **Progressive Refinement**: Iterative improvement through reflection and debugging

## Component Details

### Ideation Engine (`ai_scientist/perform_ideation_temp_free.py`)
- Generates research hypotheses from topic descriptions
- Validates novelty through Semantic Scholar integration
- Produces structured research ideas in JSON format

### Tree Search Engine (`ai_scientist/treesearch/`)
- Best-first tree search implementation
- Parallel agent execution with configurable workers
- Debugging and recovery mechanisms
- Resource monitoring and cleanup

### Paper Writing Engine (`ai_scientist/perform_writeup.py`)
- Automated manuscript generation
- Literature citation integration
- LaTeX formatting and compilation
- Multi-round review and refinement

### LLM Integration (`ai_scientist/llm.py`, `ai_scientist/backend/`)
- Multi-provider LLM support (OpenAI, Anthropic, AWS Bedrock)
- Token tracking and cost management
- Rate limiting and error handling
- Model-specific optimization

### Security Framework (`ai_scientist/utils/`)
- API key security and validation
- Path traversal protection
- Input sanitization
- Process isolation and cleanup

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model integration
- **CUDA**: GPU acceleration

### External Services
- **OpenAI API**: GPT model access
- **Anthropic API**: Claude model access
- **AWS Bedrock**: Cloud-based LLM access
- **Semantic Scholar API**: Literature search

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **Git**: Version control
- **Docker**: Containerization (planned)

## Security Considerations

### Execution Safety
- Sandboxed code execution environment
- Resource monitoring and limits
- Process cleanup and isolation
- Input validation and sanitization

### API Security
- Secure credential management
- Rate limiting and throttling
- Error handling and logging
- Access control and permissions

### Data Protection
- No sensitive data in repositories
- Secure configuration management
- Audit logging and monitoring
- Compliance with data protection regulations

## Scalability & Performance

### Resource Management
- GPU memory optimization
- Parallel processing capabilities
- Efficient model loading and caching
- Dynamic resource allocation

### Monitoring & Observability
- Performance metrics tracking
- Resource utilization monitoring
- Error tracking and alerting
- Experiment result analytics

## Future Enhancements

### Planned Features
- Multi-domain research support
- Enhanced collaboration tools
- Real-time experiment monitoring
- Advanced result visualization

### Technical Improvements
- Kubernetes deployment support
- Enhanced security frameworks
- Performance optimization
- Cost reduction strategies

## Compliance & Ethics

### Research Ethics
- Reproducible research practices
- Open science principles
- Ethical AI development
- Responsible disclosure

### Technical Standards
- Code quality standards
- Security best practices
- Documentation requirements
- Testing and validation protocols