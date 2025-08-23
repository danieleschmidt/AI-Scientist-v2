#!/bin/bash
# AI Scientist v2 Production Deployment Script
# Comprehensive deployment automation with safety checks and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"
KUBE_DIR="$DEPLOYMENT_DIR/kubernetes"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-ai-scientist}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"
DRY_RUN="${DRY_RUN:-false}"
ROLLBACK="${ROLLBACK:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"
TIMEOUT="${TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    local level="$1"
    shift
    echo -e "[$(date -Iseconds)] [${level}] $*" >&2
}

log_info() {
    log "${BLUE}INFO${NC}" "$@"
}

log_warn() {
    log "${YELLOW}WARN${NC}" "$@"
}

log_error() {
    log "${RED}ERROR${NC}" "$@"
}

log_success() {
    log "${GREEN}SUCCESS${NC}" "$@"
}

log_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        log "${PURPLE}DEBUG${NC}" "$@"
    fi
}

# Help function
show_help() {
    cat << EOF
AI Scientist v2 Production Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Deployment environment (default: production)
    -n, --namespace NS          Kubernetes namespace (default: ai-scientist)
    -t, --tag TAG              Docker image tag (default: latest)
    -r, --registry REGISTRY     Docker registry (default: your-registry.com)
    -d, --dry-run              Perform dry run without actual deployment
    -b, --rollback             Rollback to previous deployment
    -s, --skip-tests           Skip pre-deployment tests
    -f, --force                Force deployment even with warnings
    --timeout SECONDS          Deployment timeout in seconds (default: 600)
    -h, --help                 Show this help message

Examples:
    # Standard production deployment
    $0 --environment production --tag v2.0.1

    # Staging deployment with dry run
    $0 --environment staging --tag latest --dry-run

    # Rollback production deployment
    $0 --environment production --rollback

    # Force deployment with custom timeout
    $0 --environment production --tag v2.0.2 --force --timeout 1200

Environment Variables:
    KUBECONFIG                 Path to kubectl config file
    DOCKER_REGISTRY_USERNAME   Docker registry username
    DOCKER_REGISTRY_PASSWORD   Docker registry password
    DEPLOYMENT_KEY             Deployment authentication key
    SLACK_WEBHOOK_URL          Slack webhook for notifications
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -b|--rollback)
            ROLLBACK="true"
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY="true"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
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
    log_info "Validating deployment environment..."
    
    # Check required tools
    local required_tools=("kubectl" "docker" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists or can be created
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [ "$DRY_RUN" = "false" ]; then
            log_warn "Namespace $NAMESPACE does not exist, creating..."
            kubectl create namespace "$NAMESPACE" || {
                log_error "Failed to create namespace $NAMESPACE"
                exit 1
            }
        else
            log_info "Would create namespace $NAMESPACE (dry-run mode)"
        fi
    fi
    
    # Check environment-specific configuration exists
    local config_file="$DEPLOYMENT_DIR/configs/${ENVIRONMENT}.yaml"
    if [ ! -f "$config_file" ]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Validate registry credentials
    if [ -n "${DOCKER_REGISTRY_USERNAME:-}" ] && [ -n "${DOCKER_REGISTRY_PASSWORD:-}" ]; then
        log_info "Docker registry credentials found"
    else
        log_warn "Docker registry credentials not found, deployment may fail"
        if [ "$FORCE_DEPLOY" != "true" ]; then
            log_error "Use --force to proceed without registry credentials"
            exit 1
        fi
    fi
    
    log_success "Environment validation completed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if image exists
    local full_image="${REGISTRY}/ai-scientist:${IMAGE_TAG}"
    log_info "Checking image availability: $full_image"
    
    if ! docker pull "$full_image" &> /dev/null; then
        log_warn "Cannot pull image $full_image"
        if [ "$FORCE_DEPLOY" != "true" ]; then
            log_error "Use --force to proceed with potentially unavailable image"
            exit 1
        fi
    else
        log_success "Image $full_image is available"
    fi
    
    # Run security scan
    if command -v trivy &> /dev/null; then
        log_info "Running security scan on image..."
        if ! trivy image --exit-code 1 "$full_image"; then
            log_warn "Security vulnerabilities found in image"
            if [ "$FORCE_DEPLOY" != "true" ]; then
                log_error "Use --force to proceed despite security vulnerabilities"
                exit 1
            fi
        fi
    fi
    
    # Check resource availability
    log_info "Checking cluster resources..."
    local available_nodes
    available_nodes=$(kubectl get nodes --no-headers | grep -c "Ready" || echo "0")
    
    if [ "$available_nodes" -lt 3 ]; then
        log_warn "Only $available_nodes nodes available (recommended: 3+)"
        if [ "$FORCE_DEPLOY" != "true" ]; then
            log_error "Use --force to proceed with limited nodes"
            exit 1
        fi
    fi
    
    # Check GPU availability for production
    if [ "$ENVIRONMENT" = "production" ]; then
        local gpu_nodes
        gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers | wc -l || echo "0")
        
        if [ "$gpu_nodes" -eq 0 ]; then
            log_warn "No GPU nodes found for production deployment"
            if [ "$FORCE_DEPLOY" != "true" ]; then
                log_error "Use --force to proceed without GPU nodes"
                exit 1
            fi
        else
            log_success "Found $gpu_nodes GPU nodes"
        fi
    fi
    
    log_success "Pre-deployment checks completed"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_info "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running deployment tests..."
    
    # Validate Kubernetes manifests
    log_info "Validating Kubernetes manifests..."
    find "$KUBE_DIR" -name "*.yaml" -exec kubectl apply --dry-run=client -f {} \; &> /dev/null || {
        log_error "Kubernetes manifest validation failed"
        exit 1
    }
    
    # Run configuration tests
    if [ -f "$PROJECT_ROOT/tests/test_config.py" ]; then
        log_info "Running configuration tests..."
        cd "$PROJECT_ROOT"
        python -m pytest tests/test_config.py -v || {
            log_error "Configuration tests failed"
            exit 1
        }
    fi
    
    # Test health check endpoints
    log_info "Testing health check scripts..."
    python "$DEPLOYMENT_DIR/scripts/health_check.py" --check env || {
        log_error "Health check script validation failed"
        exit 1
    }
    
    log_success "All tests passed"
}

# Create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Check if secrets template exists
    local secrets_file="$KUBE_DIR/secrets.yaml"
    if [ ! -f "$secrets_file" ]; then
        log_error "Secrets template not found: $secrets_file"
        exit 1
    fi
    
    # Validate required environment variables
    local required_secrets=(
        "ANTHROPIC_API_KEY"
        "OPENAI_API_KEY"
        "SECRET_KEY"
        "JWT_SECRET_KEY"
    )
    
    local missing_secrets=()
    for secret in "${required_secrets[@]}"; do
        if [ -z "${!secret:-}" ]; then
            missing_secrets+=("$secret")
        fi
    done
    
    if [ ${#missing_secrets[@]} -gt 0 ]; then
        log_error "Missing required secrets: ${missing_secrets[*]}"
        log_error "Please set these environment variables before deploying"
        exit 1
    fi
    
    # Apply secrets (skip template sections)
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f "$secrets_file" -n "$NAMESPACE" || {
            log_error "Failed to create secrets"
            exit 1
        }
    else
        log_info "Would create secrets (dry-run mode)"
    fi
    
    log_success "Secrets created successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying AI Scientist v2 to $ENVIRONMENT..."
    
    # Set image tag in manifests
    local temp_dir
    temp_dir=$(mktemp -d)
    cp -r "$KUBE_DIR"/* "$temp_dir/"
    
    # Update image tags
    find "$temp_dir" -name "*.yaml" -exec sed -i "s/ai-scientist:latest/ai-scientist:${IMAGE_TAG}/g" {} \;
    find "$temp_dir" -name "*.yaml" -exec sed -i "s/your-registry.com/${REGISTRY}/g" {} \;
    
    # Apply configurations in order
    local apply_order=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "rbac.yaml"
        "pvc.yaml"
        "service.yaml"
        "deployment.yaml"
        "ingress.yaml"
        "hpa.yaml"
        "monitoring.yaml"
    )
    
    for file in "${apply_order[@]}"; do
        local file_path="$temp_dir/$file"
        if [ -f "$file_path" ]; then
            log_info "Applying $file..."
            if [ "$DRY_RUN" = "false" ]; then
                kubectl apply -f "$file_path" -n "$NAMESPACE" || {
                    log_error "Failed to apply $file"
                    cleanup_temp_files "$temp_dir"
                    exit 1
                }
            else
                log_info "Would apply $file (dry-run mode)"
            fi
        else
            log_warn "File not found, skipping: $file"
        fi
    done
    
    cleanup_temp_files "$temp_dir"
    log_success "Application deployment completed"
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    local deployments=("ai-scientist" "ai-scientist-worker" "ai-scientist-redis")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for deployment $deployment..."
        if [ "$DRY_RUN" = "false" ]; then
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s" || {
                log_error "Deployment $deployment failed to become ready"
                return 1
            }
        else
            log_info "Would wait for deployment $deployment (dry-run mode)"
        fi
    done
    
    log_success "All deployments are ready"
}

# Health checks
post_deployment_health_checks() {
    log_info "Running post-deployment health checks..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping health checks (dry-run mode)"
        return 0
    fi
    
    # Wait for pods to be ready
    sleep 30
    
    # Check pod status
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running -o name | wc -l)
    
    if [ "$unhealthy_pods" -gt 0 ]; then
        log_error "$unhealthy_pods pods are not running"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        return 1
    fi
    
    # Test application endpoints
    log_info "Testing application endpoints..."
    
    # Port forward for testing
    kubectl port-forward -n "$NAMESPACE" service/ai-scientist-internal 18000:8000 &
    local port_forward_pid=$!
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:18000/health &> /dev/null; then
        log_success "Health endpoint is responding"
    else
        log_error "Health endpoint is not responding"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    # Test readiness endpoint
    if curl -f http://localhost:18000/ready &> /dev/null; then
        log_success "Readiness endpoint is responding"
    else
        log_error "Readiness endpoint is not responding"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    # Clean up port forward
    kill $port_forward_pid 2>/dev/null || true
    
    log_success "Post-deployment health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would rollback deployment (dry-run mode)"
        return 0
    fi
    
    # Get previous revision
    local previous_revision
    previous_revision=$(kubectl rollout history deployment/ai-scientist -n "$NAMESPACE" | grep -E '[0-9]+' | tail -2 | head -1 | awk '{print $1}')
    
    if [ -z "$previous_revision" ]; then
        log_error "No previous revision found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to revision $previous_revision..."
    
    kubectl rollout undo deployment/ai-scientist -n "$NAMESPACE" --to-revision="$previous_revision" || {
        log_error "Rollback failed"
        exit 1
    }
    
    # Wait for rollback to complete
    kubectl rollout status deployment/ai-scientist -n "$NAMESPACE" --timeout="${TIMEOUT}s" || {
        log_error "Rollback did not complete successfully"
        exit 1
    }
    
    log_success "Rollback completed successfully"
}

# Cleanup function
cleanup_temp_files() {
    local temp_dir="$1"
    if [ -n "$temp_dir" ] && [ -d "$temp_dir" ]; then
        rm -rf "$temp_dir"
    fi
}

# Notification function
send_notification() {
    local status="$1"
    local message="$2"
    
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        local color="good"
        if [ "$status" = "error" ]; then
            color="danger"
        elif [ "$status" = "warning" ]; then
            color="warning"
        fi
        
        local payload
        payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "AI Scientist v2 Deployment - $ENVIRONMENT",
            "text": "$message",
            "fields": [
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Image Tag",
                    "value": "$IMAGE_TAG",
                    "short": true
                },
                {
                    "title": "Namespace",
                    "value": "$NAMESPACE",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date -Iseconds)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -X POST -H 'Content-type: application/json' \
            --data "$payload" \
            "$SLACK_WEBHOOK_URL" &> /dev/null || true
    fi
}

# Cleanup on exit
cleanup_on_exit() {
    local exit_code=$?
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    if [ $exit_code -eq 0 ]; then
        send_notification "success" "Deployment completed successfully"
    else
        send_notification "error" "Deployment failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

trap cleanup_on_exit EXIT

# Main deployment function
main() {
    log_info "Starting AI Scientist v2 deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Registry: $REGISTRY"
    log_info "Dry Run: $DRY_RUN"
    log_info "Rollback: $ROLLBACK"
    
    if [ "$ROLLBACK" = "true" ]; then
        validate_environment
        rollback_deployment
        post_deployment_health_checks
    else
        validate_environment
        pre_deployment_checks
        run_tests
        create_secrets
        deploy_application
        wait_for_deployment
        post_deployment_health_checks
    fi
    
    log_success "AI Scientist v2 deployment completed successfully!"
    log_info "Access the application at: https://${ENVIRONMENT}.ai-scientist.yourdomain.com"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "This was a dry run - no actual changes were made"
    fi
}

# Run main function
main "$@"