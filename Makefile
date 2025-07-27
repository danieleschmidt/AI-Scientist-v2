# AI Scientist v2 - Makefile

.PHONY: help install install-dev test lint format type-check security-check clean build docs docker run-ideation run-experiment

# Default target
help:
	@echo "AI Scientist v2 - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-hooks      Setup git pre-commit hooks"
	@echo ""
	@echo "Development Commands:"
	@echo "  test             Run all tests with coverage"
	@echo "  test-fast        Run tests without coverage (faster)"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security-check   Run security scanning"
	@echo "  quality-check    Run all quality checks (lint + type + security)"
	@echo ""
	@echo "Build Commands:"
	@echo "  clean            Clean build artifacts and cache files"
	@echo "  build            Build distribution packages"
	@echo "  docs             Generate documentation"
	@echo ""
	@echo "Container Commands:"
	@echo "  docker-build     Build Docker container"
	@echo "  docker-run       Run container interactively"
	@echo "  docker-test      Run tests in container"
	@echo ""
	@echo "AI Scientist Commands:"
	@echo "  run-ideation     Generate research ideas (requires TOPIC variable)"
	@echo "  run-experiment   Run full AI Scientist pipeline (requires IDEAS variable)"
	@echo ""
	@echo "Examples:"
	@echo "  make run-ideation TOPIC=my_research_topic.md"
	@echo "  make run-experiment IDEAS=my_research_topic.json"

# Setup Commands
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

setup-hooks:
	pre-commit install
	@echo "Git hooks installed successfully"

# Development Commands
test:
	pytest tests/ --cov=ai_scientist --cov-report=html --cov-report=term-missing --cov-fail-under=80

test-fast:
	pytest tests/ -x -v

lint:
	flake8 ai_scientist/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	black --check ai_scientist/ tests/
	isort --check-only ai_scientist/ tests/

format:
	black ai_scientist/ tests/
	isort ai_scientist/ tests/
	@echo "Code formatting completed"

type-check:
	mypy ai_scientist/ --ignore-missing-imports

security-check:
	bandit -r ai_scientist/ -f json -o security-report.json
	safety check --json --output safety-report.json
	python scripts/security_scan.py

quality-check: lint type-check security-check
	@echo "All quality checks completed"

# Build Commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup completed"

build: clean
	python setup.py sdist bdist_wheel
	@echo "Build completed - check dist/ directory"

docs:
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation generated in docs/_build/html"

# Container Commands
docker-build:
	docker build -t ai-scientist-v2:latest .

docker-run:
	docker run -it --gpus all -v $(PWD):/workspace ai-scientist-v2:latest bash

docker-test:
	docker run --gpus all -v $(PWD):/workspace ai-scientist-v2:latest make test

# AI Scientist Commands
run-ideation:
	@if [ -z "$(TOPIC)" ]; then \
		echo "Error: TOPIC variable required. Usage: make run-ideation TOPIC=my_research_topic.md"; \
		exit 1; \
	fi
	python ai_scientist/perform_ideation_temp_free.py \
		--workshop-file "ai_scientist/ideas/$(TOPIC)" \
		--model gpt-4o-2024-05-13 \
		--max-num-generations 20 \
		--num-reflections 5

run-experiment:
	@if [ -z "$(IDEAS)" ]; then \
		echo "Error: IDEAS variable required. Usage: make run-experiment IDEAS=my_research_topic.json"; \
		exit 1; \
	fi
	python launch_scientist_bfts.py \
		--load_ideas "ai_scientist/ideas/$(IDEAS)" \
		--load_code \
		--add_dataset_ref \
		--model_writeup o1-preview-2024-09-12 \
		--model_citation gpt-4o-2024-11-20 \
		--model_review gpt-4o-2024-11-20 \
		--model_agg_plots o3-mini-2025-01-31 \
		--num_cite_rounds 20

# Performance and monitoring
profile:
	python -m cProfile -o profile.stats launch_scientist_bfts.py --help
	@echo "Profile saved to profile.stats"

monitor:
	python metrics_reporter.py --watch

# Database and cache management
reset-cache:
	rm -rf cache/
	mkdir -p cache/
	@echo "Cache reset completed"

# Git helpers
check-branch:
	@git status --porcelain
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Warning: Working directory is not clean"; \
	else \
		echo "Working directory is clean"; \
	fi

# CI/CD helpers
ci-setup: install-dev setup-hooks

ci-test: quality-check test

ci-build: ci-test build

# Release helpers
version-bump:
	@echo "Current version: $$(python setup.py --version)"
	@echo "Update version in setup.py manually, then run 'make build'"

release-check: ci-build
	twine check dist/*
	@echo "Release check completed"