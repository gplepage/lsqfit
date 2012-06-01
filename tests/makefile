# Copyright (c) 2012 G. Peter Lepage. 
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

tests:
	#================================================================ gvar
	- $(PYTHON) -m unittest test-gvar
	#========================================================== bufferdict
	- $(PYTHON) -m unittest test-bufferdict
	#============================================================= dataset
	- $(PYTHON) -m unittest test-dataset
	#============================================================== lsqfit
	- $(PYTHON) -m unittest test-lsqfit

coverage:
	nosetests -s --with-coverage --cover-erase\
	   --cover-package=lsqfit test-bufferdict.py test-lsqfit.py

clean:
	rm -rf *.pyc __pycache__ .coverage *.p




