.PHONY: clear-pycache
clear-pycache:
	./scripts/cc.sh

.PHONY: pip-install
pip-install:
	python3 -m pip install -r requirements.txt

.PHONY: test
test:
	python3 -m unittest

.PHONY: build
build:
	python3 -m pip install pip --upgrade
	python3 -m pip install build --upgrade
	python3 -m build

.PHONY: format
format:
	python3 -m black finq

.PHONY: lint
lint:
	ruff check --output-format=github --target-version=py310 . --fix

.PHONY: requp
requp:
	python3 -m pipreqs.pipreqs . --force

.PHONY: clean-test
clean-test: clear-pycache unittest

