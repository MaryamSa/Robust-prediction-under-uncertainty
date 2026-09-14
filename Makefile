.PHONY: install analysis test lint clean

install:
	python -m pip install -r requirements-dev.txt

analysis:
	PYTHONPATH=src python -m kepler_uncertainty.run

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

clean:
	rm -rf artifacts data/raw .pytest_cache .ruff_cache
