#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python$(PYVERSION) -m coverage
endif

RUN=python$(PYVERSION) -m coverage run -a --source src

test:
	$(COVERAGE) erase
	$(RUN) src/FSITM.py
	$(RUN) src/FSITM.py https://www.floridamemory.com/fpc/prints/pr76815.jpg https://www.floridamemory.com/fpc/prints/pr76815.jpg
	$(RUN) src/FSITM.py --revised https://www.floridamemory.com/fpc/prints/pr76815.jpg https://www.floridamemory.com/fpc/prints/pr76815.jpg
