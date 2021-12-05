#* Variables
SHELL := /usr/bin/env bash
PYTHON := python3.9
LINTING_DIRS := detection

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl --silent --show-error --location https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl --silent --show-error --location https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run isort --settings-path pyproject.toml $(LINTING_DIRS)
	poetry run black --config pyproject.toml $(LINTING_DIRS)
	poetry run flake8 $(LINTING_DIRS)

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	poetry run pytest

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml $(LINTING_DIRS)
	poetry run black --diff --check --config pyproject.toml $(LINTING_DIRS)
	poetry run darglint --verbosity 2 $(LINTING_DIRS)
	poetry run flake8 $(LINTING_DIRS)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep --extended-regexp "(__pycache__|\.pyc|\.pyo$$)" | xargs rm --recursive --force

.PHONY: clean-all
clean-all: pycache-remove
