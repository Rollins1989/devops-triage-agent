.PHONY: install test demo stress clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

demo:
	python -m main --scenario crashloop

stress:
	python -m main --scenario crashloop --flake-rate 0.9 --confirm-actions

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
