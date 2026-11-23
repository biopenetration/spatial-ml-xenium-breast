.PHONY: install test demo clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

demo:
	spatialml demo --out demo_output --cells 6000

clean:
	rm -rf demo_output .pytest_cache build *.egg-info
