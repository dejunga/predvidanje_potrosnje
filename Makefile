# Makefile for Consumption Forecast App

.PHONY: help build run dev stop clean logs shell test install setup-dev lint format test-cov security-check quality-check

# Default target
help:
	@echo "Available commands:"
	@echo "  Docker commands:"
	@echo "    build     - Build the Docker image"
	@echo "    run       - Run the production container"
	@echo "    dev       - Run development container with hot reloading"
	@echo "    stop      - Stop all containers"
	@echo "    clean     - Remove containers and images"
	@echo "    logs      - Show container logs"
	@echo "    shell     - Open shell in running container"
	@echo "    test      - Run tests in container"
	@echo ""
	@echo "  Development commands:"
	@echo "    install       - Install dependencies"
	@echo "    setup-dev     - Set up development environment"
	@echo "    test-local    - Run tests locally"
	@echo "    test-cov      - Run tests with coverage"
	@echo "    lint          - Run linting checks"
	@echo "    format        - Format code with black and isort"
	@echo "    format-check  - Check code formatting"
	@echo "    security-check - Run security checks"
	@echo "    quality-check - Run all quality checks"
	@echo "    run-dev       - Run development server"
	@echo "    run-prod      - Run production server"

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t consumption-forecast .

# Run production container
run:
	@echo "Starting production container..."
	docker-compose up -d consumption-forecast
	@echo "App running at http://localhost:8501"

# Run development container with hot reloading
dev:
	@echo "Starting development container with hot reloading..."
	docker-compose --profile dev up -d consumption-forecast-dev
	@echo "Development app running at http://localhost:8502"
	@echo "Changes to your code will automatically reload the app!"

# Stop all containers
stop:
	@echo "Stopping all containers..."
	docker-compose down

# Clean up containers and images
clean:
	@echo "Cleaning up containers and images..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Show logs
logs:
	docker-compose logs -f

# Open shell in running container
shell:
	docker-compose exec consumption-forecast bash

# Run tests in container
test:
	@echo "Running tests in container..."
	docker-compose run --rm consumption-forecast python -m pytest

# Quick start - build and run
quick-start: build run

# Development start - build and run dev
dev-start: build dev

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================

# Installation
install:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt
	pip install pytest-cov flake8 black isort safety bandit
	@echo "Development environment set up successfully!"

# Testing
test-local:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-integration:
	pytest tests/test_integration.py -v

# Code quality
lint:
	flake8 src/ tests/ app*.py
	@echo "Linting completed!"

format:
	black src/ tests/ app*.py
	isort src/ tests/ app*.py
	@echo "Code formatted successfully!"

format-check:
	black --check src/ tests/ app*.py
	isort --check-only src/ tests/ app*.py

# Security
security-check:
	safety check
	bandit -r src/ -f json

# Development server
run-dev:
	streamlit run app_enhanced.py --server.reload=true --server.runOnSave=true

run-prod:
	streamlit run app_enhanced.py --server.headless=true --server.port=8501

# CI/CD simulation
ci-local:
	@echo "Running local CI checks..."
	make format-check
	make lint
	make test-cov
	make security-check
	@echo "Local CI checks completed!"

# All quality checks
quality-check: format-check lint test-local security-check
	@echo "All quality checks passed!"

# Clean local build artifacts
clean-local:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned build artifacts!"