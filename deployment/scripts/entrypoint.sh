#!/bin/bash
# AI Scientist v2 Production Entrypoint
# Handles initialization, health checks, and graceful shutdown

set -euo pipefail

# Configuration
export PYTHONPATH="/app:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Logging configuration
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FORMAT="${LOG_FORMAT:-json}"

# Application settings
APP_MODULE="${APP_MODULE:-ai_scientist.cli_enterprise}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-300}"

# Health check configuration
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"

# Directories
DATA_DIR="${DATA_DIR:-/app/data}"
LOGS_DIR="${LOGS_DIR:-/app/logs}"
CACHE_DIR="${CACHE_DIR:-/app/cache}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Cleanup function for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    if [ -n "${HEALTH_PID:-}" ] && kill -0 "$HEALTH_PID" 2>/dev/null; then
        log_info "Stopping health check process..."
        kill "$HEALTH_PID" 2>/dev/null || true
    fi
    
    # Additional cleanup
    if [ -f "/tmp/app.pid" ]; then
        PID=$(cat /tmp/app.pid)
        if kill -0 "$PID" 2>/dev/null; then
            log_info "Stopping main application (PID: $PID)..."
            kill -TERM "$PID" 2>/dev/null || true
            sleep 5
            if kill -0 "$PID" 2>/dev/null; then
                log_warn "Force killing application..."
                kill -KILL "$PID" 2>/dev/null || true
            fi
        fi
        rm -f /tmp/app.pid
    fi
    
    log_info "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check required directories
    for dir in "$DATA_DIR" "$LOGS_DIR" "$CACHE_DIR"; do
        if [ ! -d "$dir" ]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
        if [ ! -w "$dir" ]; then
            log_error "Directory not writable: $dir"
            exit 1
        fi
    done
    
    # Check Python environment
    if ! python -c "import ai_scientist" 2>/dev/null; then
        log_error "AI Scientist module not found"
        exit 1
    fi
    
    # Check required environment variables
    required_vars=("ANTHROPIC_API_KEY" "OPENAI_API_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_warn "Required environment variable not set: $var"
        fi
    done
    
    log_success "Environment validation completed"
}

# Initialize application
initialize_app() {
    log_info "Initializing AI Scientist v2..."
    
    # Run database migrations if needed
    if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
        log_info "Running database migrations..."
        python -c "
import sys
sys.path.append('/app')
from ai_scientist.utils.database import run_migrations
run_migrations()
" || log_warn "Migration step failed or not applicable"
    fi
    
    # Initialize cache
    log_info "Initializing cache..."
    python -c "
import sys
sys.path.append('/app')
from ai_scientist.utils.cache import initialize_cache
initialize_cache()
" || log_warn "Cache initialization failed"
    
    # Create default configuration if not exists
    if [ ! -f "$CONFIG_DIR/production.yaml" ]; then
        log_info "Creating default production configuration..."
        python -c "
import sys
sys.path.append('/app')
from ai_scientist.utils.config import create_default_config
create_default_config('$CONFIG_DIR/production.yaml')
" || log_warn "Configuration creation failed"
    fi
    
    log_success "Application initialization completed"
}

# Health check daemon
start_health_daemon() {
    log_info "Starting health check daemon..."
    
    while true; do
        sleep "$HEALTH_CHECK_INTERVAL"
        
        if ! python /app/scripts/health_check.py --quiet; then
            log_error "Health check failed"
            # Don't exit immediately, allow for recovery
            sleep 10
        fi
    done &
    
    HEALTH_PID=$!
    log_info "Health check daemon started (PID: $HEALTH_PID)"
}

# Wait for dependencies
wait_for_dependencies() {
    log_info "Waiting for dependencies..."
    
    # Wait for Redis if configured
    if [ -n "${REDIS_URL:-}" ]; then
        log_info "Waiting for Redis..."
        python -c "
import redis
import time
import os
import sys

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        r = redis.from_url(redis_url)
        r.ping()
        print('Redis connection successful')
        break
    except Exception as e:
        print(f'Redis connection attempt {attempt + 1}/{max_attempts} failed: {e}')
        time.sleep(2)
        attempt += 1
        
if attempt >= max_attempts:
    print('Redis connection failed after all attempts')
    sys.exit(1)
"
    fi
    
    # Wait for PostgreSQL if configured
    if [ -n "${DATABASE_URL:-}" ]; then
        log_info "Waiting for PostgreSQL..."
        python -c "
import psycopg2
import time
import os
import sys
from urllib.parse import urlparse

db_url = os.environ.get('DATABASE_URL')
if not db_url:
    sys.exit(0)

parsed = urlparse(db_url)
max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path[1:]
        )
        conn.close()
        print('PostgreSQL connection successful')
        break
    except Exception as e:
        print(f'PostgreSQL connection attempt {attempt + 1}/{max_attempts} failed: {e}')
        time.sleep(2)
        attempt += 1
        
if attempt >= max_attempts:
    print('PostgreSQL connection failed after all attempts')
    sys.exit(1)
" || log_warn "PostgreSQL not configured or connection failed"
    fi
    
    log_success "Dependencies check completed"
}

# Main application startup
start_application() {
    log_info "Starting AI Scientist v2 application..."
    log_info "Configuration: APP_MODULE=$APP_MODULE, PORT=$PORT, WORKERS=$WORKERS"
    
    # Choose startup method based on environment
    case "${DEPLOYMENT_MODE:-production}" in
        "development"|"dev")
            log_info "Starting in development mode..."
            exec python -m "$APP_MODULE" --host 0.0.0.0 --port "$PORT" --reload
            ;;
        "testing"|"test")
            log_info "Starting in testing mode..."
            exec python -m "$APP_MODULE" --host 0.0.0.0 --port "$PORT" --testing
            ;;
        "production"|"prod"|*)
            log_info "Starting in production mode with Gunicorn..."
            exec gunicorn \
                --bind "0.0.0.0:$PORT" \
                --workers "$WORKERS" \
                --worker-class uvicorn.workers.UvicornWorker \
                --worker-connections 1000 \
                --max-requests 1000 \
                --max-requests-jitter 100 \
                --timeout "$TIMEOUT" \
                --keepalive 5 \
                --preload \
                --access-logfile "-" \
                --error-logfile "-" \
                --log-level "$LOG_LEVEL" \
                --capture-output \
                --enable-stdio-inheritance \
                --pid /tmp/app.pid \
                "$APP_MODULE:app"
            ;;
    esac
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check disk space
    disk_usage=$(df /app | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_error "Disk usage is critical: ${disk_usage}%"
        exit 1
    elif [ "$disk_usage" -gt 80 ]; then
        log_warn "Disk usage is high: ${disk_usage}%"
    fi
    
    # Check memory
    if [ -f /proc/meminfo ]; then
        mem_available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        mem_usage=$((100 - (mem_available * 100 / mem_total)))
        
        if [ "$mem_usage" -gt 90 ]; then
            log_error "Memory usage is critical: ${mem_usage}%"
            exit 1
        elif [ "$mem_usage" -gt 80 ]; then
            log_warn "Memory usage is high: ${mem_usage}%"
        fi
    fi
    
    # Check Python module integrity
    python -c "
import sys
sys.path.append('/app')
try:
    from ai_scientist import __version__
    print(f'AI Scientist v2 version: {__version__}')
except Exception as e:
    print(f'Module integrity check failed: {e}')
    sys.exit(1)
"
    
    log_success "Pre-flight checks completed"
}

# Security hardening
apply_security_hardening() {
    log_info "Applying security hardening..."
    
    # Set secure umask
    umask 027
    
    # Clear sensitive environment variables from history
    unset HISTFILE
    export HISTSIZE=0
    
    # Set resource limits if running as root (container init)
    if [ "$EUID" -eq 0 ]; then
        log_warn "Running as root - applying additional security measures"
        
        # Drop capabilities if possible
        # Note: This would typically be handled by the container runtime
        
        # Set file permissions
        find /app -type f -executable -exec chmod 755 {} \;
        find /app -type f ! -executable -exec chmod 644 {} \;
        chmod 600 /app/config/*.yaml 2>/dev/null || true
    fi
    
    log_success "Security hardening applied"
}

# Main execution
main() {
    log_info "Starting AI Scientist v2 Production Container"
    log_info "================================================"
    
    # Apply security measures
    apply_security_hardening
    
    # Run pre-flight checks
    preflight_checks
    
    # Validate environment
    validate_environment
    
    # Wait for external dependencies
    wait_for_dependencies
    
    # Initialize application
    initialize_app
    
    # Start health check daemon in background
    if [ "${ENABLE_HEALTH_DAEMON:-true}" = "true" ]; then
        start_health_daemon
    fi
    
    # Start the main application
    log_info "Starting application with command: $*"
    
    if [ $# -eq 0 ]; then
        start_application
    else
        # Execute custom command
        log_info "Executing custom command: $*"
        exec "$@"
    fi
}

# Handle special cases
case "${1:-}" in
    "bash"|"sh"|"/bin/bash"|"/bin/sh")
        log_info "Starting interactive shell"
        exec "$@"
        ;;
    "python")
        log_info "Starting Python directly"
        exec "$@"
        ;;
    "health-check")
        exec python /app/scripts/health_check.py
        ;;
    "--help"|"-h")
        echo "AI Scientist v2 Production Container Entrypoint"
        echo ""
        echo "Usage: $0 [command] [args...]"
        echo ""
        echo "Special commands:"
        echo "  health-check    Run health check and exit"
        echo "  bash/sh         Start interactive shell"
        echo "  python          Start Python interpreter"
        echo ""
        echo "Environment variables:"
        echo "  LOG_LEVEL           Logging level (default: INFO)"
        echo "  DEPLOYMENT_MODE     Deployment mode (default: production)"
        echo "  PORT               Server port (default: 8000)"
        echo "  WORKERS            Worker processes (default: 4)"
        echo "  TIMEOUT            Request timeout (default: 300)"
        echo "  ENABLE_HEALTH_DAEMON Enable health daemon (default: true)"
        exit 0
        ;;
    *)
        # Run main application
        main "$@"
        ;;
esac