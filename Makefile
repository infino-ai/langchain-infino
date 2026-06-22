.PHONY: install test unit integration lint format type

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
