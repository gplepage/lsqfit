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
	- diff -w -I svdcut -I fitter simple.out simple.tmp
	$(PYTHON) y-vs-x.py > y-vs-x.tmp
	- diff -w -I svdcut -I fitter y-vs-x.out y-vs-x.tmp
	$(PYTHON) x-err.py > x-err.tmp
	- diff -w -I svdcut -I fitter x-err.out x-err.tmp
	$(PYTHON) p-corr.py > p-corr.tmp
	- diff -w -I svdcut -I fitter p-corr.out p-corr.tmp
	$(PYTHON) y-noerr.py > y-noerr.tmp
	- diff -w -I svdcut -I fitter y-noerr.out y-noerr.tmp
	$(PYTHON) empbayes.py > empbayes.tmp
	- diff -w -I svdcut -I fitter empbayes.out empbayes.tmp
	$(PYTHON) pendulum.py > pendulum.tmp
	- diff -w -I svdcut -I fitter pendulum.out pendulum.tmp
	$(PYTHON) nist.py > nist.tmp
	- diff -w -I svdcut -I fitter nist.out nist.tmp
	$(PYTHON) uncorrelated.py > uncorrelated.tmp
	- diff -w -I svdcut -I fitter uncorrelated.out uncorrelated.tmp
	$(PYTHON) bayes.py > bayes.tmp
	- diff -w -B -I svdcut -I fitter bayes.out bayes.tmp
	# mpi tests
	- mpirun -np 2 -output-filename mpi-bayes $(PYTHON) bayes.py
	- diff -w -I svdcut -I fitter mpi-bayes.1.0 mpi-bayes.1.1
	- diff -w -I svdcut -I fitter mpi-bayes.1.0 bayes.out

time:
	time $(MAKE) _time

_time:
	$(PYTHON) simple.py > /dev/null
	$(PYTHON) y-vs-x.py > /dev/null
	$(PYTHON) x-err.py > /dev/null
	$(PYTHON) p-corr.py > /dev/null
	$(PYTHON) y-noerr.py > /dev/null
	$(PYTHON) empbayes.py > /dev/null
	$(PYTHON) pendulum.py > /dev/null
	$(PYTHON) nist.py > /dev/null
	$(PYTHON) uncorrelated.py > /dev/null
	$(PYTHON) bayes.py > /dev/null

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) y-vs-x.py > y-vs-x.out
	$(PYTHON) x-err.py > x-err.out
	$(PYTHON) p-corr.py > p-corr.out
	$(PYTHON) y-noerr.py > y-noerr.out
	$(PYTHON) empbayes.py > empbayes.out
	$(PYTHON) pendulum.py > pendulum.out
	$(PYTHON) nist.py > nist.out
	$(PYTHON) uncorrelated.py > uncorrelated.out
	$(PYTHON) bayes.py > bayes.out
	mpirun -np 2 -output-filename mpi-bayes $(PYTHON) bayes.py
	cp mpi-bayes.1.0 mpi-bayes.out

clean:
	rm -f *.pyc *.tmp *.p *.prof *.1.0 *.1.1

