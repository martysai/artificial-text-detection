#* Variables
SHELL := /usr/bin/env bash
PYTHON := python3.8
LINTING_DIRS := detection

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl --silent --show-error --location https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl --silent --show-error --location https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

# suggest using virtualenvwrapper
# https://virtualenvwrapper.readthedocs.io/en/latest/
.PHONY: poetry-shell
	poetry shell
	export POETRY_VIRTUALENVS_PATH=$WORKON_HOME
	workon $(poetry env info -p | sed 's:.*/::')

#* Installation
.PHONY: install
install:
	poetry lock --no-interaction && poetry export --without-hashes > requirements.txt
	poetry install --no-interaction
	rm requirements.txt


.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

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

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml $(LINTING_DIRS)

.PHONY: lint
lint: test check-codestyle mypy

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep --extended-regexp "(__pycache__|\.pyc|\.pyo$$)" | xargs rm --recursive --force

.PHONY: clean-all
clean-all: pycache-remove
