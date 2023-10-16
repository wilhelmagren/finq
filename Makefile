.PHONY: clean
clean:
	poetry run pyclean finq tests

.PHONY: install
install:
	poetry install --with dev --no-root

.PHONY: test
test:
	poetry run pytest tests

.PHONY: build
build:
	poetry build --format wheel

.PHONY: format
format:
	poetry run black finq tests

.PHONY: lint
lint:
	poetry run ruff check finq tests --fix

.PHONY: static-check
static-check:
	poetry run mypy finq

.PHONY: clean-test
clean-test: clean test
