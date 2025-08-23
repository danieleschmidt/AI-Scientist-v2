# Unified Research Pipeline for AI Scientist v2

A clean, simple implementation that demonstrates the core value of the AI Scientist system. This pipeline can automatically go from research topic to complete scientific paper.

## Features

✅ **Research Idea Generation**: Automatically generate novel research ideas from a topic  
✅ **Experiment Execution**: Run experiments using the tree search system  
✅ **Scientific Paper Generation**: Generate complete LaTeX papers with citations  
✅ **Progress Tracking**: Clear status reporting and error handling  
✅ **Flexible Configuration**: Skip steps, use custom models, adjust parameters  
✅ **Production Ready**: Proper logging, results saving, session management  

## Quick Start

```bash
# Generate ideas and run complete pipeline
python unified_research_pipeline.py --topic "transformer attention mechanisms" --output-dir ./research_output

# Use existing ideas file
python unified_research_pipeline.py --idea-file ideas.json --output-dir ./research_output --skip-ideation

# Skip experiments and just generate ideas + paper
python unified_research_pipeline.py --topic "reinforcement learning" --output-dir ./output --skip-experiments
```

## How It Works

The pipeline integrates three existing AI Scientist components:

1. **Ideation System** (`perform_ideation_temp_free.py`)
   - Generates research ideas automatically
   - Uses semantic scholar for literature search
   - Produces structured JSON with hypotheses, experiments, etc.

2. **Tree Search System** (`treesearch/`)
   - Executes experiments using breadth-first tree search
   - Manages agent workflows and resource allocation
   - Generates experiment logs and summaries

3. **Writeup System** (`perform_writeup.py`)
   - Creates scientific papers from experiment results
   - Handles LaTeX compilation and citation management
   - Produces publication-ready PDFs

## Output Structure

```
output_dir/
└── session_YYYYMMDD_HHMMSS/
    ├── pipeline_results.json          # Complete session results
    ├── ideation/
    │   ├── ideas.json                 # Generated research ideas
    │   └── workshop_description.md    # Research topic description
    ├── experiments/
    │   ├── research_idea.md           # Selected idea for experiments
    │   ├── bfts_config.yaml          # Experiment configuration
    │   └── logs/                      # Experiment execution logs
    └── writeup/
        ├── paper_YYYYMMDD_HHMMSS.pdf # Generated research paper
        ├── latex/                     # LaTeX source files
        └── figures/                   # Generated plots and figures
```

## CLI Reference

### Required Arguments

- `--output-dir DIR`: Directory to store all research outputs
- `--topic TOPIC` OR `--idea-file FILE`: Research topic or existing ideas file

### Model Configuration

- `--model MODEL`: Model for ideation/experiments (default: gpt-4o-2024-11-20)
- `--big-model MODEL`: Model for paper writeup (default: o1-2024-12-17)

### Pipeline Control

- `--max-ideas N`: Maximum ideas to generate (default: 3)
- `--num-reflections N`: Reflection rounds per idea (default: 5)
- `--page-limit N`: Page limit for paper (default: 8)
- `--skip-ideation`: Skip idea generation (requires --idea-file)
- `--skip-experiments`: Skip experiment execution
- `--skip-writeup`: Skip paper generation

### Other Options

- `--experiment-config FILE`: Custom experiment configuration
- `--verbose`: Enable verbose logging

## Available Models

- gpt-4o-2024-11-20
- o1-2024-12-17  
- gpt-4o-mini
- claude-3-5-sonnet-20240620
- o1-preview-2024-09-12
- o3-mini-2025-01-31

## Demo Mode

When dependencies are not available, the pipeline runs in demo mode:
- No external API calls are made
- Sample outputs are generated for demonstration
- Shows complete pipeline structure and workflow
- Useful for testing and development

## Example Use Cases

### 1. Explore New Research Direction
```bash
python unified_research_pipeline.py \
  --topic "federated learning privacy preservation" \
  --output-dir ./fl_research \
  --max-ideas 5 \
  --verbose
```

### 2. Continue from Generated Ideas  
```bash
# First, generate ideas only
python unified_research_pipeline.py \
  --topic "graph neural networks" \
  --output-dir ./gnn_research \
  --skip-experiments \
  --skip-writeup

# Then continue with selected experiments
python unified_research_pipeline.py \
  --idea-file ./gnn_research/session_*/ideation/ideas.json \
  --output-dir ./gnn_experiments \
  --skip-ideation
```

### 3. Paper Generation Only
```bash
python unified_research_pipeline.py \
  --idea-file ./existing_ideas.json \
  --output-dir ./paper_only \
  --skip-ideation \
  --skip-experiments \
  --page-limit 6
```

## Integration with Existing Components

The pipeline is designed to work with your existing AI Scientist setup:

- **Config Files**: Uses existing `ai_scientist_config.yaml` for model settings
- **Templates**: Leverages existing LaTeX templates in `blank_icml_latex/`
- **Examples**: Can reference existing few-shot examples in `fewshot_examples/`
- **Tools**: Integrates semantic scholar and other research tools

## Error Handling

The pipeline includes robust error handling:
- Each step can fail independently without breaking the pipeline
- Detailed error logs with tracebacks
- Automatic session state saving
- Resume capability for long-running processes

## Next Steps

This is a foundation that can be extended with:
- Multi-idea experimentation
- Advanced experiment selection strategies  
- Real-time progress monitoring
- Distributed execution across multiple machines
- Integration with external evaluation systems

## Requirements

When running with full AI Scientist integration (non-demo mode):
- Python 3.8+
- All dependencies from `requirements.txt`
- API keys for chosen LLM providers
- LaTeX distribution for paper compilation