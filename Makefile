#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python$(PYVERSION) -m coverage
endif

RUN=python$(PYVERSION) -m coverage run -a --source src

test:
	$(COVERAGE) erase
	$(RUN) src/FSITM.py
