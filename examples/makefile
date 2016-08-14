# Copyright (c) 2012-2016 G. Peter Lepage.
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
	- diff -w -I svdcut simple.out simple.tmp
	$(PYTHON) y-vs-x.py > y-vs-x.tmp
	- diff -w -I svdcut y-vs-x.out y-vs-x.tmp
	$(PYTHON) x-err.py > x-err.tmp
	- diff -w -I svdcut x-err.out x-err.tmp
	$(PYTHON) p-corr.py > p-corr.tmp
	- diff -w -I svdcut p-corr.out p-corr.tmp
	$(PYTHON) err-budget.py > err-budget.tmp
	- diff -w -I svdcut err-budget.out err-budget.tmp
	$(PYTHON) pendulum.py > pendulum.tmp
	- diff -w -I svdcut pendulum.out pendulum.tmp
	$(PYTHON) bayes.py > bayes.tmp
	- diff -w -B -I svdcut bayes.out bayes.tmp

time:
	time $(MAKE) _time

_time:
	$(PYTHON) simple.py > /dev/null
	$(PYTHON) y-vs-x.py > /dev/null
	$(PYTHON) x-err.py > /dev/null
	$(PYTHON) p-corr.py > /dev/null
	$(PYTHON) err-budget.py > /dev/null
	$(PYTHON) pendulum.py > /dev/null
	# $(PYTHON) bayes.py > /dev/null

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) y-vs-x.py > y-vs-x.out
	$(PYTHON) x-err.py > x-err.out
	$(PYTHON) p-corr.py > p-corr.out
	$(PYTHON) err-budget.py > err-budget.out
	$(PYTHON) pendulum.py > pendulum.out
	$(PYTHON) bayes.py > bayes.out

clean:
	rm -f *.pyc *.tmp *.p *.prof

