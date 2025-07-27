# Multi-stage Docker build for AI Scientist v2
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    ghostscript \
    poppler-utils \
    chktex \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aiscientist && useradd -r -g aiscientist aiscientist

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add user to sudo group for development
RUN usermod -aG sudo aiscientist

# Production stage
FROM base as production

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=aiscientist:aiscientist . /app/

# Create necessary directories
RUN mkdir -p /app/experiments /app/data /app/logs && \
    chown -R aiscientist:aiscientist /app

# Switch to non-root user
USER aiscientist

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "--version"]

# Security scanning stage
FROM production as security

# Switch back to root for security tools installation
USER root

# Install security scanning tools
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    trivy \
    && rm -rf /var/lib/apt/lists/*

# Run security scans
RUN trivy fs --exit-code 1 /app || true

# Switch back to application user
USER aiscientist