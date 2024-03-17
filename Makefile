.PHONY: test clean docs

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0

test: clean
	pip install -q -r requirements.txt
	pip install -q -r requirements/test.txt

	# use this to run tests
	python -m coverage run --source thunder -m pytest thunder tests -v --flake8
	python -m coverage report

docs: clean
	pip install --quiet -r docs/requirements.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated
	rm -rf ./docs/source/api
	rm -rf _ckpt_*
