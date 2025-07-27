# AI Scientist v2 - Development Automation

.PHONY: help install test lint format clean build docker run dev setup docs security

# Default target
help: ## Show this help message
	@echo "AI Scientist v2 - Development Commands"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
setup: ## Set up development environment
	@echo "Setting up AI Scientist v2 development environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	pre-commit install
	@echo "✅ Development environment ready!"

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

# Code Quality
lint: ## Run linting checks
	@echo "Running linting checks..."
	flake8 ai_scientist/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy ai_scientist/ --ignore-missing-imports
	black --check ai_scientist/ tests/
	isort --check-only ai_scientist/ tests/

format: ## Format code with black and isort
	@echo "Formatting code..."
	black ai_scientist/ tests/
	isort ai_scientist/ tests/
	@echo "✅ Code formatted!"

# Testing
test: ## Run all tests
	@echo "Running tests..."
	pytest tests/ -v --cov=ai_scientist --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	pytest tests/ -x -v

test-integration: ## Run integration tests only
	pytest tests/test_*integration*.py -v

test-security: ## Run security tests
	pytest tests/test_*security*.py -v

test-performance: ## Run performance tests
	pytest tests/test_*performance*.py -v

# Development
dev: ## Start development environment
	docker-compose up -d
	@echo "🚀 Development environment started!"
	@echo "   - AI Scientist: http://localhost:8000"
	@echo "   - Jupyter Lab: http://localhost:8888"
	@echo "   - Grafana: http://localhost:3000"

dev-stop: ## Stop development environment
	docker-compose down
	@echo "🛑 Development environment stopped!"

dev-logs: ## Show development logs
	docker-compose logs -f ai-scientist

# Building
build: ## Build application
	@echo "Building AI Scientist v2..."
	python -m build
	@echo "✅ Build complete!"

docker-build: ## Build Docker image
	docker build -t ai-scientist:latest .

docker-build-dev: ## Build development Docker image
	docker build -t ai-scientist:dev --target development .

# Docker Operations
docker: ## Build and run Docker container
	docker-compose up --build

docker-prod: ## Run production Docker setup
	docker-compose -f docker-compose.prod.yml up -d

# Cleaning
clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

clean-data: ## Clean experimental data and caches
	@echo "Cleaning data and caches..."
	rm -rf experiments/
	rm -rf aisci_outputs/
	rm -rf results/
	rm -rf cache/
	rm -rf huggingface/
	@echo "✅ Data cleaned!"

clean-all: clean clean-data ## Clean everything

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html/
	@echo "✅ Documentation generated in docs/_build/html/"

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

# Security
security: ## Run security checks
	@echo "Running security checks..."
	bandit -r ai_scientist/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	semgrep --config=auto ai_scientist/ --json --output=semgrep-report.json
	@echo "✅ Security checks complete!"

security-scan: ## Run container security scan
	trivy fs . --format json --output trivy-report.json

# Database Operations
db-setup: ## Set up database
	@echo "Setting up database..."
	python scripts/setup_database.py
	@echo "✅ Database ready!"

db-migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database
	@echo "⚠️  Resetting database..."
	docker-compose exec postgres psql -U ai_scientist -d ai_scientist -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	make db-migrate
	@echo "✅ Database reset!"

# Monitoring
monitor: ## Start monitoring stack
	docker-compose up -d prometheus grafana
	@echo "📊 Monitoring stack started!"
	@echo "   - Prometheus: http://localhost:9090"
	@echo "   - Grafana: http://localhost:3000"

# Deployment
deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.staging.yml up -d
	@echo "✅ Deployed to staging!"

deploy-prod: ## Deploy to production
	@echo "🚨 Deploying to production..."
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Deployed to production!"

# Git Hooks
hooks-install: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-update: ## Update git hooks
	pre-commit autoupdate

# Performance
benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

profile: ## Profile application performance
	python -m cProfile -o profile.stats scripts/profile_app.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Utilities
check-env: ## Check environment configuration
	@echo "Checking environment configuration..."
	python scripts/check_environment.py

generate-config: ## Generate configuration files
	python scripts/generate_config.py

check-deps: ## Check for dependency updates
	pip list --outdated

update-deps: ## Update dependencies
	pip-review --local --interactive

# Research Operations
ideate: ## Generate research ideas
	python ai_scientist/perform_ideation_temp_free.py \
		--workshop-file ai_scientist/ideas/example.md \
		--model gpt-4o-2024-05-13 \
		--max-num-generations 5

experiment: ## Run full experiment pipeline
	python launch_scientist_bfts.py \
		--load_ideas ai_scientist/ideas/example.json \
		--model_writeup gpt-4o-2024-11-20 \
		--model_review gpt-4o-2024-11-20

# CI/CD
ci-setup: ## Set up CI/CD environment
	@echo "Setting up CI/CD..."
	mkdir -p .github/workflows
	cp templates/ci.yml .github/workflows/
	@echo "✅ CI/CD setup complete!"

release: ## Create a new release
	@echo "Creating new release..."
	python scripts/create_release.py
	@echo "✅ Release created!"

# Maintenance
health-check: ## Run system health check
	python scripts/health_check.py

backup: ## Create system backup
	python scripts/backup.py

restore: ## Restore from backup
	@read -p "Enter backup file path: " backup_file; \
	python scripts/restore.py "$$backup_file"

# Version Info
version: ## Show version information
	@python -c "import ai_scientist; print(f'AI Scientist v{ai_scientist.__version__}')"
	@echo "Python: $$(python --version)"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git: $$(git --version)"