.PHONY: setup test clean docs get-sphinx-theme
.ONESHELL:
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0
INSTALL := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip install" || echo "pip install")

test: clean
	pip install -q -r requirements.txt -r requirements/test.txt

	# use this to run tests
	python -m coverage run --source thunder -m pytest thunder tests -v
	python -m coverage report

get-sphinx-theme:
	pip install -q awscli
	mkdir -p dist/
	aws s3 sync --no-sign-request s3://sphinx-packages/ dist/
	pip install lai-sphinx-theme -f dist/

docs: clean get-sphinx-theme
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

# install all requirements for development
# install pre-commit hooks
# install editable package
setup:
	echo "Using $(INSTALL)"
	$(INSTALL) -r requirements.txt \
			-r requirements/devel.txt \
			-r requirements/test.txt
	pre-commit install
	$(INSTALL) -e .
