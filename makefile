# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2012 G. Peter Lepage. 
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

PYTHON = python

install : 
	$(PYTHON) setup.py install --user --record installed-files.$(PYTHON)

install-sys : 		
	$(PYTHON) setup.py install --record installed-files.$(PYTHON)

uninstall :			# not sure this works --- be careful
	cat installed-files.$(PYTHON) | xargs rm -rf

install-gdev :
	$(PYTHON) gdev-setup.py install --user --record installed-files.gdev

doc-html:
	rm -rf doc/html; cd doc/source; make html; mv _build/html ..

doc-pdf:
	cd doc/source; make latex; cd _build/latex;  make all-pdf
	mv doc/source/_build/latex/lsqfit.pdf doc/lsqfit.pdf

doc-all: doc-html doc-pdf

sdist:			# source distribution
	$(PYTHON) setup.py sdist

lsqfit.tz:		# everything
	$(MAKE) clean
	tar --exclude '\.svn' --exclude 'old' --exclude 'tmp' -z -c -v -C .. -f lsqfit.tz lsqfit

.PHONY: tests

tests:
	$(MAKE) -C tests PYTHON=$(PYTHON) tests

run-examples:
	$(MAKE) -C examples PYTHON=$(PYTHON) run

clean:
	rm -f -r build __pycache__
	rm -f *.so *.tmp *.pyc *.prof .coverage
	rm -f gvar.c lsqfit_util.c
	rm -f lsqfit.tz
	rm -f -r dist
	rm -rf src/lsqfit/*.c
	rm -rf src/gvar/*.c
	rm -rf src/lsqfit/*.pyc
	rm -rf src/gvar/*.pyc
	$(MAKE) -C doc/source clean
	$(MAKE) -C tests clean
	$(MAKE) -C examples clean


