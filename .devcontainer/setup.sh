#!/bin/bash
set -e

echo "🚀 Setting up AI Scientist v2 development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install additional system dependencies for LaTeX and PDF processing
echo "📄 Installing LaTeX and PDF tools..."
sudo apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    latexmk \
    poppler-utils \
    imagemagick

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "📚 Installing project in development mode..."
pip install -e ".[dev,security,docs,monitoring]"

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p experiments aisci_outputs results cache logs
mkdir -p docs/adr docs/guides docs/runbooks
mkdir -p .github/workflows .github/ISSUE_TEMPLATE .github/PULL_REQUEST_TEMPLATE

# Set up git hooks
echo "🔧 Setting up git hooks..."
if [ ! -f .git/hooks/pre-push ]; then
    ln -sf ../../scripts/git_hooks/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
fi

if [ ! -f .git/hooks/prepare-commit-msg ]; then
    ln -sf ../../scripts/git_hooks/prepare-commit-msg .git/hooks/prepare-commit-msg
    chmod +x .git/hooks/prepare-commit-msg
fi

# Initialize configuration files if they don't exist
echo "⚙️ Initializing configuration..."
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || echo "# AI Scientist v2 Environment Variables" > .env
fi

# Set up monitoring
echo "📊 Setting up monitoring..."
if [ -f monitoring/prometheus.yml ]; then
    echo "Prometheus configuration found"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Run 'make test' to verify the setup"
echo "3. Start developing! 🚀"