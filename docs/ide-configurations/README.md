# 💻 IDE Configuration Templates
## AI Scientist v2 - Development Environment Setup

This directory contains IDE configuration templates optimized for AI Scientist v2 development.

## 🎯 VS Code Configuration

### Quick Setup
1. Copy `vscode-settings.json` to `.vscode/settings.json` in your workspace root
2. Install recommended extensions when prompted
3. Reload VS Code to apply settings

### Features Included
- **Python Development**: Complete toolchain integration (Black, Flake8, MyPy, isort)
- **Testing**: PyTest integration with debugging support
- **AI/ML Focus**: Optimized for machine learning workflows
- **Performance**: Smart file exclusions for large AI datasets
- **Security**: Workspace trust and security settings

### Recommended Extensions
The configuration works optimally with these extensions:
- `ms-python.python` - Python language support
- `ms-python.black-formatter` - Code formatting
- `ms-python.flake8` - Linting
- `ms-python.mypy-type-checker` - Type checking
- `GitHub.copilot` - AI-powered code completion
- `ms-toolsai.jupyter` - Jupyter notebook support

### Custom Settings Highlights

#### Python Integration
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.testing.pytestEnabled": true
}
```

#### AI/ML Optimizations
```json
{
  "files.exclude": {
    "experiments/": true,
    "aisci_outputs/": true,
    "results/": true,
    "cache/": true,
    "huggingface/": true
  }
}
```

## 🚀 Other IDE Support

### PyCharm Configuration
For PyCharm users, recommended settings:
- **Interpreter**: Use project virtual environment
- **Code Style**: Import Black configuration
- **Inspections**: Enable MyPy and Flake8 plugins
- **Test Runner**: Configure PyTest as default

### Vim/Neovim Setup
For Vim users, consider these plugins:
- `python-mode` - Python development mode
- `ale` - Asynchronous linting engine
- `vim-test` - Test runner integration
- `coc.nvim` - LSP support with Python language server

## 🔧 Customization

### Project-Specific Adjustments
Modify these settings based on your project needs:

```json
{
  "python.defaultInterpreterPath": "./your-venv/bin/python",
  "python.testing.pytestArgs": [
    "your-test-directory"
  ]
}
```

### Performance Tuning
For large repositories, additional exclusions:

```json
{
  "files.watcherExclude": {
    "**/large-dataset/**": true,
    "**/model-checkpoints/**": true
  }
}
```

## 🐛 Debugging Configuration

The VS Code setup includes debugging profiles for:
- **Python Files**: Debug current file
- **PyTest**: Debug test suites  
- **AI Scientist**: Launch main application
- **Docker**: Attach to containerized development

Access via Run and Debug panel (Ctrl+Shift+D).

## 📊 Code Quality Integration

### Automatic Formatting
- **On Save**: Automatic Black formatting
- **Import Sorting**: isort integration
- **Linting**: Real-time Flake8 feedback

### Type Checking
- **MyPy Integration**: Real-time type checking
- **Error Highlighting**: Inline type error display
- **Quick Fixes**: Suggested type annotations

## 🔍 Troubleshooting

### Common Issues

#### Python Interpreter Not Found
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

#### Linting Not Working
1. Ensure Flake8 is installed: `pip install flake8`
2. Check Python interpreter path in VS Code
3. Reload window: Ctrl+Shift+P → "Developer: Reload Window"

#### Slow Performance
1. Check file exclusions are properly set
2. Disable unnecessary extensions
3. Increase VS Code memory limit if needed

## 🔄 Updates and Maintenance

### Keeping Configuration Current
1. Regularly update extension recommendations
2. Review and update file exclusions
3. Monitor VS Code settings documentation for new features

### Team Synchronization
1. Commit `.vscode/settings.json` to version control
2. Document any team-specific customizations
3. Use workspace settings for project-specific configurations

---

*These IDE configurations are optimized for AI Scientist v2 development. Adjust settings based on your specific workflow and preferences.*