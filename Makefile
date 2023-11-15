PYTHON := python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
COVERAGE := $(PYTHON) -m coverage
BLACK := black
RUFF := ruff
PYLINT := pylint
TOX := $(PYTHON) -m tox
SETUP := $(PYTHON) setup.py

.PHONY: lint lint-notebooks test coverage tox install install-dev pypi clean help

lint:
	@echo "Running linting..."
	@$(BLACK) denseclus tests setup.py
	@$(RUFF) denseclus tests setup.py --fix --preview
	@$(PYLINT) denseclus --disable=R0902,W0222,W0221,C0103,W0632

lint-notebooks:
	@echo "Linting notebooks..."
	@nbqa black notebooks/*.ipynb
	@nbqa isort notebooks/*.ipynb


test:
	@echo "Running tests..."
	@$(PYTEST) -ra

coverage:
	@echo "Running coverage..."
	@$(COVERAGE) erase
	@$(COVERAGE) run --include=denseclus/* -m pytest -ra
	@$(COVERAGE) report -m

tox:
	@echo "Running tox..."
	@$(TOX)

install:
	@echo "Installing..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e .

install-dev: install
	@echo "Installing dev dependencies..."
	@$(PIP) install -r requirements-dev.txt

install-test: install
	@echo "Installing test dependencies..."
	@$(PIP) install -e ".[test]"
	@$(PIP) install -e ".[all]"


pypi:
	@echo "Uploading to PyPi..."
	@$(SETUP) sdist
	@$(SETUP) bdist_wheel --universal
	@twine upload dist/*

clean:
	@echo "Cleaning..."
	@rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache
	@rm -rf .mypy_cache .ruff_cache .coverage build .tox
	coverage erase

help:
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
