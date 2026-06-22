.PHONY: install test unit integration lint format type

install:
	pip install -e ".[test,lint]"

# Default test target is the unit suite (no engine / network).
test: unit

unit:
	pytest tests/unit_tests -q

integration:
	pytest tests/integration_tests -q

lint:
	ruff check .

format:
	ruff format .

type:
	mypy langchain_infino
