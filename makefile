# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2023 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

PIP = python -m pip
PYTHON = python
PYTHONVERSION = python`python -c 'import platform; print(platform.python_version())'`
VERSION = `python -c 'import lsqfit; print(lsqfit.__version__)'`

DOCFILES :=  $(shell ls doc/source/conf.py doc/source/*.{rst,out,png} 2>/dev/null)
SRCFILES := $(shell ls setup.py src/lsqfit/*.{py,pyx})
CYTHONFILES := src/lsqfit/_gsl.c src/lsqfit/_utilities.c

install-user : 
	rm -rf src/lsqfit/*.c 
	python make_version.py src/lsqfit/_version.py
	$(PIP) install . --user --no-cache-dir

install install-sys : 
	rm -rf src/lsqfit/*.c 
	python make_version.py src/lsqfit/_version.py
	$(PIP) install . --no-cache-dir

update:
	make uninstall install

uninstall :			# mostly works (may leave some empty directories)
	- $(PIP) uninstall lsqfit
	
.PHONY : doc

doc-html doc:
	make doc/html/index.html

doc/html/index.html : $(SRCFILES) $(DOCFILES) setup.cfg
	sphinx-build -b html doc/source doc/html

clear-doc:
	rm -rf doc/html

doc-zip doc.zip:
	cd doc/html; zip -r doc *; mv doc.zip ../..

# doc-all: doc-html # doc-pdf

sdist:	$(CYTHONFILES)		# source distribution
	$(PYTHON) setup.py sdist
	# $(PYTHON) -m build --sdist

.PHONY: tests

tests test-all:
	@echo 'N.B. Some tests involve random numbers and so fail occasionally'
	@echo '     (less than 1 in 100 times) due to multi-sigma fluctuations.'
	@echo '     Run again if any test fails.'
	@echo ''
	$(PYTHON) -m unittest discover

time:
	$(MAKE) -C examples PYTHON=$(PYTHON) time

run run-examples:
	$(MAKE) -C examples PYTHON=$(PYTHON) run

upload-twine: 
	twine upload dist/lsqfit-$(VERSION)*

upload-git: 
	echo  "version $(VERSION)"
	make doc-html # doc-pdf
	git diff --exit-code
	git diff --cached --exit-code
	git push origin main

tag-git:
	echo  "version $(VERSION)"
	git tag -a v$(VERSION) -m "version $(VERSION)"
	git push origin v$(VERSION)

test-download:
	-$(PIP) uninstall lsqfit
	$(PIP) install lsqfit --no-cache-dir

test-readme:
	python setup.py --long-description | rst2html.py > README.html

clean:
	rm -f -r build
	rm -rf __pycache__
	rm -f *.so *.tmp *.pyc *.prof .coverage doc.zip
	rm -f -r dist
	$(MAKE) -C doc/source clean
	$(MAKE) -C tests clean
	$(MAKE) -C examples clean


