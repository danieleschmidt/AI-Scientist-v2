# ADR-001: Multi-Model LLM Strategy

## Status
Accepted

## Context
The AI Scientist v2 system requires robust Large Language Model (LLM) integration to support various research tasks including ideation, experimentation, writing, and review. Different models excel at different tasks, and single-provider dependency poses availability and cost risks.

## Decision
Implement a multi-provider LLM strategy supporting OpenAI, Anthropic, and AWS Bedrock models with intelligent model selection based on task requirements.

## Rationale

### Benefits
- **Risk Mitigation**: Reduces dependency on single provider
- **Cost Optimization**: Use most cost-effective model for each task
- **Performance Optimization**: Leverage model strengths for specific tasks
- **Availability**: Fallback options if primary provider is unavailable
- **Future-Proofing**: Easy integration of new models and providers

### Task-Specific Model Selection
- **Ideation**: GPT-4 or Claude-3.5-Sonnet for creative reasoning
- **Code Generation**: Claude-3.5-Sonnet for accurate implementation
- **Writing**: GPT-4 or o1-preview for structured academic writing
- **Review**: GPT-4o for comprehensive analysis and feedback
- **Citation**: GPT-4o for accurate literature search and formatting

## Implementation Details

### Model Configuration
```yaml
models:
  ideation:
    primary: "claude-3-5-sonnet-20241022"
    fallback: "gpt-4o-2024-11-20"
  experimentation:
    primary: "claude-3-5-sonnet-20241022"
    fallback: "gpt-4o-2024-11-20"
  writing:
    primary: "o1-preview-2024-09-12"
    fallback: "gpt-4o-2024-11-20"
  review:
    primary: "gpt-4o-2024-11-20"
    fallback: "claude-3-5-sonnet-20241022"
  citation:
    primary: "gpt-4o-2024-11-20"
    fallback: "claude-3-5-sonnet-20241022"
```

### Provider Integration
1. **OpenAI**: Direct API integration with retry logic
2. **Anthropic**: Native Anthropic SDK with AWS Bedrock fallback
3. **AWS Bedrock**: Claude models via Amazon infrastructure
4. **Gemini**: Google AI Studio integration (future)

### Failover Strategy
- Automatic failover to backup provider on errors
- Provider health monitoring and circuit breaker pattern
- Cost tracking and budget management per provider
- Performance metrics collection for optimization

## Consequences

### Positive
- Improved system reliability and availability
- Cost optimization through intelligent model selection
- Better performance through task-specific model usage
- Reduced vendor lock-in and negotiation leverage

### Negative
- Increased complexity in configuration and management
- Additional API keys and credentials to manage
- Potential inconsistencies in model outputs
- Higher development and maintenance overhead

### Mitigating Actions
- Comprehensive configuration management system
- Standardized model interface abstraction
- Extensive testing across all providers
- Clear documentation and monitoring dashboards

## Alternatives Considered

### Single Provider Strategy
- **Pros**: Simplicity, consistency, easier management
- **Cons**: Vendor lock-in, availability risk, limited optimization
- **Rejected**: Too risky for production system

### Open Source Models Only
- **Pros**: No API costs, full control, privacy
- **Cons**: Resource intensive, lower quality, maintenance overhead
- **Rejected**: Current open source models insufficient for research quality

### Cloud-Agnostic Model Serving
- **Pros**: Maximum flexibility, cost control
- **Cons**: Significant infrastructure complexity, higher latency
- **Deferred**: Consider for future enterprise deployment

## References
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Model Performance Benchmarks](internal_benchmarks.md)

## Related ADRs
- ADR-002: API Security and Key Management
- ADR-003: Cost Management and Budget Controls
- ADR-004: Model Performance Monitoring

---
**Date**: 2025-07-27  
**Author**: Terry (Terragon Labs)  
**Reviewers**: System Architecture Team