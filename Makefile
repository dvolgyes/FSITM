#!/usr/bin/make

ifndef COVERAGE
COVERAGE=python3 -m coverage
endif

RUN=python3 -m coverage run -a --source src

test:
	$(COVERAGE) erase
	$(RUN) src/FSITM.py
	$(RUN) src/FSITM.py testdata/test.png testdata/test_ldr.png --alpha 0.5
	$(RUN) src/FSITM.py --revised testdata/test.png testdata/test_ldr.png  --alpha 0.5
