.PHONY: help lint format
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check --extend-exclude 'veeksha/lm_eval' veeksha

lint/isort: ## check style with isort
	isort --check-only --profile black --extend-skip veeksha/lm_eval --extend-skip veeksha/_version.py veeksha

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check --exclude 'veeksha/lm_eval/*,veeksha/_version.py' veeksha

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**,./veeksha/lm_eval/**,./veeksha.egg-info/**,./test_output/**,./benchmark_results/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black --extend-exclude 'veeksha/lm_eval' veeksha

format/isort: ## format code with isort
	isort --profile black --extend-skip veeksha/lm_eval veeksha

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports --exclude 'veeksha/lm_eval/*,veeksha/_version.py' veeksha

format: format/isort format/autoflake format/black ## format code

# Test targets
test: test/unit test/functional test/gpu ## Run all tests

test/functional: ## Run functional tests
	@echo "Running functional tests..."
	python -m pytest -s tests/functional -v -m "functional and not gpu" --tb=short \
			--junitxml=test_output/pytest-functional-nogpu-results.xml \
			--cov=veeksha --cov-append --cov-report=

test/gpu: ## Run GPU tests
	@echo "Running GPU tests..."
	python -m pytest -s tests/functional -v -m "gpu" --tb=short \
			--junitxml=test_output/pytest-functional-gpu-results.xml \
			--cov=veeksha --cov-append --cov-report=

# optional: keep unit generating initial data but skip reports, or regenerate at end
test/unit: ## Run unit tests
	@echo "Running unit tests..."
	python -m pytest -s tests -v -m "unit" --tb=short \
			--junitxml=test_output/pytest-unit-results.xml \
			--cov=veeksha --cov-report=
	
test/integration: ## Run integration tests
	@echo "Integration tests not yet implemented..."

test/all: ## Run all tests including GPU
	@echo "Running all tests..."
	python -m pytest -s tests -v --tb=short  \
		--junitxml=test_output/pytest-all-results.xml \

# Emit final coverage reports into mounted test_output directory
coverage/report:
	coverage xml -o test_output/python_coverage.xml
	coverage html -d test_output/python_coverage_html

# Rerun failed tests
test/failed-only: ## Rerun only failed tests
	@echo "Rerunning failed tests..."
	python -m pytest -s tests --lf -v --tb=short
