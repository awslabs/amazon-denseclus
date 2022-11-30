.DEFAULT_GOAL := help
.PHONY: coverage deps help lint publish push test tox

lint:
	black denseclus tests setup.py --check
	flake8 denseclus tests setup.py --max-line-length=90

test:
	python -m pytest -ra

coverage:  ## Run tests with coverage
	python -m coverage erase
	python -m coverage run --include=denseclus/* -m pytest -ra
	python -m coverage report -m

tox: tox
	python -m tox

install:
	python -m pip install --upgrade pip
	python -m pip install black coverage flake8 mypy pytest tox tox-gh-actions
	python -m pip install -e .

install-dev: install
	python -m pip install -e ".[dev]"
	pre-commit install

install-test: install
	python -m pip install -e ".[test]"
	python -m pip install -e ".[all]"

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

help: ## Show help message
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
