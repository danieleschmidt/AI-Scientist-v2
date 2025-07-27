# AI Scientist v2 - System Architecture

## Overview

The AI Scientist v2 is an autonomous scientific research system that generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts through agentic tree search.

## Problem Statement

Scientific research traditionally requires significant human involvement in hypothesis generation, experimentation, and manuscript writing. The AI Scientist v2 aims to automate these processes while maintaining research quality and scientific rigor.

## Success Criteria

- Generate valid research ideas autonomously
- Execute experiments with minimal human intervention
- Produce peer-reviewable scientific papers
- Maintain security and safety throughout the research process
- Scale across multiple ML domains

## System Components

### Core Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ideation      │    │  Experimentation │    │  Writing        │
│   Engine        │───▶│   Engine         │───▶│   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Semantic Scholar│    │ Tree Search     │    │ Citation        │
│ Integration     │    │ Manager         │    │ Manager         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input**: Research topic markdown file
2. **Ideation**: Generate research ideas via LLM + novelty checking
3. **Experimentation**: Agentic tree search with parallel workers
4. **Analysis**: Results aggregation and plotting
5. **Writing**: Manuscript generation with citations
6. **Output**: Complete scientific paper (PDF)

### Key Modules

#### AI Scientist Core (`ai_scientist/`)
- `perform_ideation_temp_free.py`: Research idea generation
- `perform_writeup.py`: Manuscript writing
- `perform_llm_review.py`: Automated peer review
- `treesearch/`: Agentic tree search implementation

#### Tree Search Engine (`ai_scientist/treesearch/`)
- `agent_manager.py`: Parallel experiment coordination
- `parallel_agent.py`: Individual experiment execution
- `interpreter.py`: Code execution and safety
- `resource_monitor.py`: System resource management

#### Security & Safety (`ai_scientist/utils/`)
- `api_security.py`: API key protection
- `input_validation.py`: Input sanitization
- `path_security.py`: File system protection
- `process_cleanup_enhanced.py`: Resource cleanup

#### Tools Integration (`ai_scientist/tools/`)
- `semantic_scholar.py`: Literature search and novelty checking
- `base_tool.py`: Tool interface abstraction

## Security Architecture

### Defense-in-Depth Strategy

1. **Input Validation**: All user inputs are sanitized
2. **Process Isolation**: Experiments run in controlled environments
3. **Resource Monitoring**: System resource usage tracking
4. **API Security**: Secure handling of API credentials
5. **File System Protection**: Path traversal prevention

### Safety Measures

- GPU memory management and cleanup
- Process timeout enforcement
- Automatic resource cleanup on errors
- Secure temporary file handling

## Configuration Management

### Primary Configs
- `ai_scientist_config.yaml`: Main system configuration
- `bfts_config.yaml`: Tree search parameters
- Environment variables for API keys

### Tree Search Configuration
- `num_workers`: Parallel exploration paths
- `steps`: Maximum nodes to explore
- `max_debug_depth`: Error recovery attempts
- `debug_prob`: Debug attempt probability

## Technology Stack

### Core Technologies
- Python 3.11+
- PyTorch with CUDA support
- Transformers library for ML models
- Rich library for enhanced CLI output

### LLM Integration
- OpenAI GPT models
- Anthropic Claude models (via AWS Bedrock)
- Google Gemini models

### Infrastructure
- Docker for containerization
- Git for version control
- LaTeX for paper generation
- PDF processing tools

## Scalability Considerations

### Horizontal Scaling
- Parallel worker processes for experiments
- Concurrent tree search exploration
- Independent ideation pipelines

### Resource Management
- GPU memory optimization
- Process cleanup automation
- Timeout-based resource recovery

## Quality Assurance

### Testing Strategy
- Unit tests for core components
- Integration tests for full pipelines
- Security testing for input validation
- Performance testing for resource usage

### Code Quality
- Black for code formatting
- Type hints for better maintainability
- Comprehensive logging and monitoring

## Future Enhancements

### Planned Features
- Multi-domain research capability expansion
- Enhanced security scanning
- Improved citation accuracy
- Real-time collaboration features

### Architecture Evolution
- Microservices decomposition
- Cloud-native deployment options
- Enhanced monitoring and observability
- API-first design principles