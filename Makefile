# Some simple testing tasks (sorry, UNIX only).


flake:
	flake8 torch_optimizer tests examples setup.py

test: flake
	pytest -sv

vtest:
	pytest -sv -vv

checkrst:
	python setup.py check --restructuredtext

pyroma:
	pyroma -d .

bandit:
	bandit -r ./torch_optimizer

mypy:
	mypy torch_optimizer --ignore-missing-imports

cov cover coverage: flake checkrst pyroma bandit
	pytest -sv -vv --cov=torch_optimizer --cov-report=term --cov-report=html ./tests
	@echo "open file://`pwd`/htmlcov/index.html"

clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf cover
	rm -rf dist
	rm -rf docs/_build

doc:
	make -C docs html
	@echo "open file://`pwd`/docs/_build/html/index.html"

black:
	black -S -l 79 setup.py torch_optimizer/ tests/ examples/

.PHONY: all flake test vtest cov clean doc
