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

run:
	$(PYTHON) simple.py > simple.tmp
	- diff -w simple.out simple.tmp
	$(PYTHON) y-vs-x.py > y-vs-x.tmp
	- diff -w y-vs-x.out y-vs-x.tmp
	$(PYTHON) x-err.py > x-err.tmp
	- diff -w x-err.out x-err.tmp
	$(PYTHON) p-corr.py > p-corr.tmp
	- diff -w p-corr.out p-corr.tmp
	$(PYTHON) err-budget.py > err-budget.tmp
	- diff -w err-budget.out err-budget.tmp

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) y-vs-x.py > y-vs-x.out
	$(PYTHON) x-err.py > x-err.out
	$(PYTHON) p-corr.py > p-corr.out
	$(PYTHON) err-budget.py > err-budget.out

clean:
	rm -f *.pyc *.tmp *.p *.prof

