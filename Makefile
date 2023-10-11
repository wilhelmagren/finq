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

.PHONY: clean-test
clean-test: clear-pycache unittest

