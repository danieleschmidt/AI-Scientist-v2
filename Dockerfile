# AI Scientist v2 - Multi-stage Docker build
# Secure, optimized container for autonomous scientific research

# Stage 1: Base system with CUDA support
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    poppler-utils \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    chktex \
    latexmk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 scientist
USER scientist
WORKDIR /home/scientist

# Stage 2: Python dependencies
FROM base AS python-deps

# Copy requirements first for better caching
COPY --chown=scientist:scientist requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN python3.11 -m pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    python3.11 -m pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Development environment (for development)
FROM python-deps AS development

# Install development dependencies
RUN python3.11 -m pip install --user --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=scientist:scientist . /home/scientist/ai-scientist-v2/
WORKDIR /home/scientist/ai-scientist-v2

# Install package in development mode
RUN python3.11 -m pip install --user -e .

# Expose ports for development services
EXPOSE 8888 6006 8080

# Development entrypoint
CMD ["bash"]

# Stage 4: Production environment
FROM python-deps AS production

# Copy only necessary source files
COPY --chown=scientist:scientist ai_scientist/ /home/scientist/ai-scientist-v2/ai_scientist/
COPY --chown=scientist:scientist *.py /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist *.yaml /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist *.yml /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist setup.py /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist pyproject.toml /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist README.md /home/scientist/ai-scientist-v2/
COPY --chown=scientist:scientist LICENSE /home/scientist/ai-scientist-v2/

WORKDIR /home/scientist/ai-scientist-v2

# Install package
RUN python3.11 -m pip install --user .

# Create directories for data and outputs
RUN mkdir -p \
    /home/scientist/data \
    /home/scientist/experiments \
    /home/scientist/results \
    /home/scientist/cache \
    /home/scientist/logs

# Set up PATH to include user-installed packages
ENV PATH="/home/scientist/.local/bin:${PATH}"
ENV PYTHONPATH="/home/scientist/ai-scientist-v2:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import ai_scientist; print('AI Scientist v2 is healthy')" || exit 1

# Security: Run as non-root user
USER scientist

# Production entrypoint
ENTRYPOINT ["python3.11", "launch_scientist_bfts.py"]

# Stage 5: Testing environment
FROM development AS testing

# Copy test files
COPY --chown=scientist:scientist tests/ /home/scientist/ai-scientist-v2/tests/

# Run tests
RUN python3.11 -m pytest tests/ --tb=short

# Testing entrypoint
CMD ["python3.11", "-m", "pytest", "tests/", "-v"]

# Default target is production
FROM production AS final