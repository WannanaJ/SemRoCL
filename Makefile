# Makefile for SemRoCL Project
# =============================

.PHONY: help install install-dev test test-cov lint format docs clean

# Default target
help:
	@echo "SemRoCL Project Makefile"
	@echo "========================"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install project dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests with pytest"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run code quality checks"
	@echo "  format       - Format code with black and isort"
	@echo "  docs         - Build documentation"
	@echo "  docs-serve   - Build and serve documentation locally"
	@echo "  clean        - Clean build artifacts and cache"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

# Code Quality
lint:
	@echo "Running Flake8..."
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
	@echo ""
	@echo "Running Pylint..."
	pylint src/ --exit-zero --max-line-length=120
	@echo ""
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports --no-strict-optional

format:
	@echo "Formatting code with black..."
	black --line-length 120 src/ tests/
	@echo ""
	@echo "Sorting imports with isort..."
	isort --profile black src/ tests/

check-format:
	@echo "Checking code format..."
	black --check --line-length 120 src/ tests/
	isort --check-only --profile black src/ tests/

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && $(MAKE) html

docs-serve:
	@echo "Building and serving documentation..."
	cd docs && $(MAKE) html
	@echo "Starting web server on http://localhost:8000"
	python -m http.server 8000 --directory docs/build/html

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	@echo "Clean complete!"

# Training shortcuts
train-stage1:
	cd src && python train_stage1_moco.py --config ../configs/train_stage1.yaml

train-stage2:
	cd src && python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced_optimized.yaml

# Docker (optional)
docker-build:
	docker build -t semrocl:latest .

docker-run:
	docker run --gpus all -it --rm -v $(PWD):/workspace semrocl:latest
