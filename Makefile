.PHONY: test clean docs

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0

test: clean
	pip install -q -r requirements.txt -r requirements/test.txt

	# use this to run tests
	python -m coverage run --source thunder -m pytest thunder tests -v
	python -m coverage report

docs: clean
	pip install -e . --quiet -r requirements/docs.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
	cd docs ; python -m sphinx -b html -W --keep-going source build

clean:
	# clean all temp runs
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated
	rm -rf ./docs/source/api
	rm -rf _ckpt_*
