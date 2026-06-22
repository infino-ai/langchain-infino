.PHONY: install test unit integration lint format type build smoke

install:
	pip install -e ".[test,lint]"

# Default test target is the unit suite (no engine / network).
test: unit

unit:
	pytest tests/unit -q

integration:
	pytest tests/integration -q

lint:
	ruff check .

format:
	ruff format .

type:
	mypy langchain_infino

# Build the sdist + wheel into dist/.
build:
	python -m pip install --quiet build
	python -m build

# End-to-end check of the built wheel in a throwaway venv: installs it
# (pulling infino + langchain-core from PyPI) and runs the smoke test.
smoke: build
	rm -rf .smoke-venv
	python -m venv .smoke-venv
	.smoke-venv/bin/pip install --quiet "$$(ls dist/*.whl)" pytest pytest-asyncio
	.smoke-venv/bin/pytest tests/smoke -q
