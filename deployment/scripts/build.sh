#!/bin/bash
# AI Scientist v2 Docker Image Build Script
# Builds optimized Docker images for different environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
REGISTRY="${REGISTRY:-your-registry.com}"
IMAGE_NAME="${IMAGE_NAME:-ai-scientist}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TARGET="${TARGET:-production}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH="${PUSH:-false}"
CACHE="${CACHE:-true}"
BUILD_ARGS="${BUILD_ARGS:-}"
MULTI_ARCH="${MULTI_ARCH:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "[$(date -Iseconds)] [${BLUE}INFO${NC}] $*" >&2
}

log_warn() {
    echo -e "[$(date -Iseconds)] [${YELLOW}WARN${NC}] $*" >&2
}

log_error() {
    echo -e "[$(date -Iseconds)] [${RED}ERROR${NC}] $*" >&2
}

log_success() {
    echo -e "[$(date -Iseconds)] [${GREEN}SUCCESS${NC}] $*" >&2
}

# Help function
show_help() {
    cat << EOF
AI Scientist v2 Docker Build Script

Usage: $0 [OPTIONS]

Options:
    -r, --registry REGISTRY    Docker registry (default: your-registry.com)
    -n, --name NAME           Image name (default: ai-scientist)
    -t, --tag TAG             Image tag (default: latest)
    --target TARGET           Build target (default: production)
    --platform PLATFORM      Target platform (default: linux/amd64)
    -p, --push                Push image to registry after build
    --no-cache               Build without cache
    --build-args ARGS         Additional build arguments
    --multi-arch             Build multi-architecture image
    -h, --help                Show this help message

Build targets:
    base                      Base image with system dependencies
    dependencies              Base + Python dependencies
    security                  Dependencies + security scanning
    production               Production-ready image (default)
    development              Development image with dev tools
    testing                  Testing image with test tools

Examples:
    # Build production image
    $0 --tag v2.0.1 --push

    # Build development image
    $0 --target development --tag dev-latest

    # Build multi-architecture image
    $0 --multi-arch --platform linux/amd64,linux/arm64 --tag multi-latest

    # Build with custom build args
    $0 --build-args "PYTHON_VERSION=3.11" --tag custom-python

Environment Variables:
    DOCKER_REGISTRY_USERNAME   Docker registry username
    DOCKER_REGISTRY_PASSWORD   Docker registry password
    DOCKER_BUILDKIT           Enable BuildKit (default: 1)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        -p|--push)
            PUSH="true"
            shift
            ;;
        --no-cache)
            CACHE="false"
            shift
            ;;
        --build-args)
            BUILD_ARGS="$2"
            shift 2
            ;;
        --multi-arch)
            MULTI_ARCH="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
validate_environment() {
    log_info "Validating build environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    
    # Check if buildx is available for multi-arch builds
    if [ "$MULTI_ARCH" = "true" ]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker Buildx is required for multi-architecture builds"
            exit 1
        fi
        
        # Create builder instance if needed
        if ! docker buildx inspect multiarch &> /dev/null; then
            log_info "Creating multi-arch builder..."
            docker buildx create --name multiarch --use --bootstrap
        else
            docker buildx use multiarch
        fi
    fi
    
    # Check registry credentials if pushing
    if [ "$PUSH" = "true" ]; then
        if [ -n "${DOCKER_REGISTRY_USERNAME:-}" ] && [ -n "${DOCKER_REGISTRY_PASSWORD:-}" ]; then
            log_info "Logging into Docker registry..."
            echo "$DOCKER_REGISTRY_PASSWORD" | docker login "$REGISTRY" -u "$DOCKER_REGISTRY_USERNAME" --password-stdin
        else
            log_warn "Registry credentials not found, push may fail"
        fi
    fi
    
    log_success "Environment validation completed"
}

# Pre-build checks
pre_build_checks() {
    log_info "Running pre-build checks..."
    
    # Check Dockerfile exists
    local dockerfile="$PROJECT_ROOT/deployment/docker/Dockerfile.production"
    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        exit 1
    fi
    
    # Check required files
    local required_files=(
        "requirements.txt"
        "deployment/docker/requirements-prod.txt"
        "deployment/scripts/entrypoint.sh"
        "deployment/scripts/health_check.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_ROOT/$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Validate requirements.txt
    if ! python -c "
import pkg_resources
try:
    with open('$PROJECT_ROOT/requirements.txt') as f:
        requirements = f.read()
    pkg_resources.parse_requirements(requirements)
    print('Requirements file is valid')
except Exception as e:
    print(f'Requirements file is invalid: {e}')
    exit(1)
"; then
        log_error "Invalid requirements.txt file"
        exit 1
    fi
    
    # Check disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=5000000  # 5GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space. Required: 5GB, Available: $((available_space/1024/1024))GB"
        exit 1
    fi
    
    log_success "Pre-build checks completed"
}

# Generate build context
generate_build_context() {
    log_info "Generating build context..."
    
    # Create temporary build context
    local build_context
    build_context=$(mktemp -d)
    
    # Copy necessary files
    cp -r "$PROJECT_ROOT/ai_scientist" "$build_context/"
    cp -r "$PROJECT_ROOT/deployment" "$build_context/"
    cp "$PROJECT_ROOT/requirements.txt" "$build_context/"
    cp "$PROJECT_ROOT/pyproject.toml" "$build_context/" 2>/dev/null || true
    cp "$PROJECT_ROOT/setup.py" "$build_context/" 2>/dev/null || true
    
    # Create .dockerignore
    cat > "$build_context/.dockerignore" << EOF
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache
**/.coverage
**/htmlcov
**/.git
**/.gitignore
**/README.md
**/docs
**/tests
**/*.log
**/logs
**/cache
**/tmp
**/temp
**/.env
**/.venv
**/venv
**/node_modules
**/.DS_Store
**/Thumbs.db
EOF
    
    echo "$build_context"
}

# Build image
build_image() {
    log_info "Building Docker image..."
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    local build_context
    build_context=$(generate_build_context)
    
    # Prepare build arguments
    local docker_build_args=()
    
    # Add cache options
    if [ "$CACHE" = "false" ]; then
        docker_build_args+=(--no-cache)
    fi
    
    # Add target
    docker_build_args+=(--target "$TARGET")
    
    # Add platform
    if [ "$MULTI_ARCH" = "true" ]; then
        docker_build_args+=(--platform "$PLATFORM")
    else
        docker_build_args+=(--platform "$PLATFORM")
    fi
    
    # Add custom build args
    if [ -n "$BUILD_ARGS" ]; then
        IFS=' ' read -ra ARGS <<< "$BUILD_ARGS"
        for arg in "${ARGS[@]}"; do
            docker_build_args+=(--build-arg "$arg")
        done
    fi
    
    # Add standard build args
    docker_build_args+=(--build-arg "BUILDKIT_INLINE_CACHE=1")
    docker_build_args+=(--build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
    docker_build_args+=(--build-arg "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')")
    docker_build_args+=(--build-arg "VERSION=${IMAGE_TAG}")
    
    # Add tags
    docker_build_args+=(--tag "$full_image_name")
    
    # Add additional tags
    if [ "$IMAGE_TAG" != "latest" ]; then
        docker_build_args+=(--tag "${REGISTRY}/${IMAGE_NAME}:latest")
    fi
    
    # Add environment-specific tag
    if [ "$TARGET" != "production" ]; then
        docker_build_args+=(--tag "${REGISTRY}/${IMAGE_NAME}:${TARGET}-${IMAGE_TAG}")
    fi
    
    # Set dockerfile path
    docker_build_args+=(--file "$build_context/deployment/docker/Dockerfile.production")
    
    # Build command
    local build_cmd="docker"
    if [ "$MULTI_ARCH" = "true" ]; then
        build_cmd="docker buildx"
        if [ "$PUSH" = "true" ]; then
            docker_build_args+=(--push)
        fi
    fi
    
    log_info "Build command: $build_cmd build ${docker_build_args[*]} $build_context"
    
    # Execute build
    if ! $build_cmd build "${docker_build_args[@]}" "$build_context"; then
        log_error "Docker build failed"
        cleanup_build_context "$build_context"
        exit 1
    fi
    
    cleanup_build_context "$build_context"
    log_success "Docker image built successfully: $full_image_name"
}

# Security scan
security_scan() {
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Running security scan..."
    
    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy security scan..."
        
        if ! trivy image --format json --output "/tmp/trivy-report.json" "$full_image_name"; then
            log_warn "Trivy scan completed with warnings"
        fi
        
        # Show summary
        trivy image --format table "$full_image_name" | head -20
        
        log_info "Full security report saved to /tmp/trivy-report.json"
    else
        log_warn "Trivy not found, skipping security scan"
    fi
    
    # Check for common vulnerabilities
    log_info "Checking for common security issues..."
    
    # Check if image runs as root
    local user_check
    user_check=$(docker run --rm "$full_image_name" whoami 2>/dev/null || echo "unknown")
    
    if [ "$user_check" = "root" ]; then
        log_warn "Image runs as root user"
    else
        log_success "Image runs as non-root user: $user_check"
    fi
    
    log_success "Security scan completed"
}

# Test image
test_image() {
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Testing Docker image..."
    
    # Basic functionality test
    log_info "Testing basic functionality..."
    
    if ! docker run --rm "$full_image_name" python -c "import ai_scientist; print('AI Scientist imported successfully')"; then
        log_error "Basic functionality test failed"
        exit 1
    fi
    
    # Health check test
    log_info "Testing health check..."
    
    if ! docker run --rm "$full_image_name" python /app/scripts/health_check.py --check env; then
        log_error "Health check test failed"
        exit 1
    fi
    
    # Entrypoint test
    log_info "Testing entrypoint..."
    
    if ! timeout 30 docker run --rm "$full_image_name" --help &> /dev/null; then
        log_error "Entrypoint test failed"
        exit 1
    fi
    
    # Size check
    local image_size
    image_size=$(docker images --format "table {{.Size}}" "$full_image_name" | tail -1)
    log_info "Image size: $image_size"
    
    log_success "Image tests completed"
}

# Push image
push_image() {
    if [ "$PUSH" != "true" ]; then
        return 0
    fi
    
    if [ "$MULTI_ARCH" = "true" ]; then
        log_info "Multi-arch image was pushed during build"
        return 0
    fi
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Pushing Docker image to registry..."
    
    if ! docker push "$full_image_name"; then
        log_error "Failed to push image"
        exit 1
    fi
    
    # Push additional tags
    if [ "$IMAGE_TAG" != "latest" ]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:latest" || log_warn "Failed to push latest tag"
    fi
    
    if [ "$TARGET" != "production" ]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:${TARGET}-${IMAGE_TAG}" || log_warn "Failed to push target-specific tag"
    fi
    
    log_success "Image pushed successfully"
}

# Cleanup function
cleanup_build_context() {
    local build_context="$1"
    if [ -n "$build_context" ] && [ -d "$build_context" ]; then
        rm -rf "$build_context"
    fi
}

# Generate image metadata
generate_metadata() {
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Generating image metadata..."
    
    local metadata_file="$PROJECT_ROOT/image-metadata.json"
    
    cat > "$metadata_file" << EOF
{
    "image": "$full_image_name",
    "tag": "$IMAGE_TAG",
    "target": "$TARGET",
    "platform": "$PLATFORM",
    "build_date": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "builder": "$(whoami)@$(hostname)",
    "docker_version": "$(docker version --format '{{.Server.Version}}')",
    "registry": "$REGISTRY",
    "multi_arch": $MULTI_ARCH,
    "pushed": $PUSH
}
EOF
    
    log_info "Metadata saved to $metadata_file"
}

# Main function
main() {
    log_info "Starting AI Scientist v2 Docker build"
    log_info "Registry: $REGISTRY"
    log_info "Image: $IMAGE_NAME:$IMAGE_TAG"
    log_info "Target: $TARGET"
    log_info "Platform: $PLATFORM"
    log_info "Push: $PUSH"
    log_info "Multi-arch: $MULTI_ARCH"
    
    validate_environment
    pre_build_checks
    build_image
    
    if [ "$TARGET" = "production" ] || [ "$TARGET" = "security" ]; then
        security_scan
    fi
    
    test_image
    push_image
    generate_metadata
    
    log_success "AI Scientist v2 Docker build completed successfully!"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "Built image: $full_image_name"
    
    if [ "$PUSH" = "true" ]; then
        log_info "Image is available in registry: $REGISTRY"
    fi
}

# Run main function
main "$@"