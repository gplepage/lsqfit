# Copyright (c) 2012-2019 G. Peter Lepage.
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
PLOT = True

run:
	$(PYTHON) simple.py > simple.tmp
	- diff -w -I svdcut -I time -I fitter simple.out simple.tmp
	$(PYTHON) multifitter.py $(PLOT) > multifitter.tmp
	- diff -w -I svdcut -I time -I fitter multifitter.out multifitter.tmp
	$(PYTHON) y-vs-x.py $(PLOT) > y-vs-x.tmp
	- diff -w -I svdcut -I time -I fitter y-vs-x.out y-vs-x.tmp
	$(PYTHON) x-err.py > x-err.tmp
	- diff -w -I svdcut -I time -I fitter x-err.out x-err.tmp
	$(PYTHON) p-corr.py False > p-corr.tmp
	- diff -w -I svdcut -I time -I fitter p-corr.out p-corr.tmp
	$(PYTHON) y-noerr.py > y-noerr.tmp
	- diff -w -I svdcut -I time -I fitter y-noerr.out y-noerr.tmp
	$(PYTHON) empbayes.py > empbayes.tmp
	- diff -w -I svdcut -I time -I fitter empbayes.out empbayes.tmp
	$(PYTHON) pendulum.py > pendulum.tmp
	- diff -w -I svdcut -I time -I fitter pendulum.out pendulum.tmp
	$(PYTHON) spline.py False > spline.tmp
	- diff -w -I svdcut -I time -I fitter spline.out spline.tmp
	$(PYTHON) nist.py > nist.tmp
	- diff -w -I svdcut -I time -I fitter nist.out nist.tmp
	$(PYTHON) uncorrelated.py > uncorrelated.tmp
	- diff -w -I svdcut -I time -I fitter uncorrelated.out uncorrelated.tmp
	$(PYTHON) bayes.py False > bayes.tmp
	- diff -w -B -I svdcut -I time -I fitter bayes.out bayes.tmp

time:
	time $(MAKE) _time

_time:
	$(PYTHON) simple.py > /dev/null
	$(PYTHON) multifitter.py False > /dev/null
	$(PYTHON) y-vs-x.py False > /dev/null
	$(PYTHON) x-err.py > /dev/null
	$(PYTHON) p-corr.py False > /dev/null
	$(PYTHON) y-noerr.py > /dev/null
	$(PYTHON) empbayes.py > /dev/null
	$(PYTHON) pendulum.py > /dev/null
	$(PYTHON) nist.py > /dev/null
	$(PYTHON) uncorrelated.py > /dev/null
	$(PYTHON) bayes.py False > /dev/null

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) multifitter.py False > multifitter.out
	$(PYTHON) y-vs-x.py False > y-vs-x.out
	$(PYTHON) x-err.py > x-err.out
	$(PYTHON) p-corr.py False > p-corr.out
	$(PYTHON) y-noerr.py > y-noerr.out
	$(PYTHON) empbayes.py > empbayes.out
	$(PYTHON) pendulum.py > pendulum.out
	$(PYTHON) spline.py > spline.out
	$(PYTHON) nist.py > nist.out
	$(PYTHON) uncorrelated.py > uncorrelated.out
	$(PYTHON) bayes.py False > bayes.out

clean:
	rm -f *.pyc *.tmp *.p *.prof *.1.0 *.1.1

