# Multi-stage production Dockerfile for AI Scientist v2
FROM python:3.11-slim as base

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Create non-root user for security
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build stage
FROM base as builder

WORKDIR /build

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage  
FROM base as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ai_scientist/ ./ai_scientist/
COPY launch_scientist_bfts.py ./
COPY bfts_config.yaml ./
COPY simple_research_executor.py ./
COPY robust_research_executor.py ./
COPY scalable_research_executor.py ./

# Create necessary directories
RUN mkdir -p /app/data /app/experiments /app/logs /app/cache

# Set proper permissions
RUN chown -R aiuser:aiuser /app
USER aiuser

# Add local packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import ai_scientist; print('OK')" || exit 1

# Default command
CMD ["python", "scalable_research_executor.py", "--domain", "machine learning"]
