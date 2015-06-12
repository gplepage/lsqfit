""" Basic linear algebra for GVars. """

# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2015 G. Peter Lepage. 
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

import numpy 
import gvar 

def det(a):
    """ Determinant of matrix ``a``. 

    Args:
        a: Two-dimensional, square matrix/array of numbers 
            and/or :class:`gvar.GVar`\s.

    Returns:
        Deterimant of the matrix.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('Bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = inv(amean)
    return numpy.linalg.det(amean) * (1 + numpy.matrix.trace(da.dot(ainv)))

def slogdet(a):
    """ Sign and logarithm of determinant of matrix ``a``. 

    Args:
        a: Two-dimensional, square matrix/array of numbers 
            and/or :class:`gvar.GVar`\s.

    Returns:
        Tuple ``(s, logdet)`` where the determinant of matrix ``a`` is
            ``s * exp(logdet)``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('Bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = inv(amean)
    s, ldet = numpy.linalg.slogdet(amean)
    ldet += numpy.matrix.trace(da.dot(ainv))
    return s, ldet

def eigvalsh(a):
    """ Eigenvalues of Hermitian matrix ``a``.

    Args:
        a: Two-dimensional, square matrix/array of numbers 
            and/or :class:`gvar.GVar`\s.

    Returns: 
        Array of eigenvalues of matrix ``a``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.

    This function has the following attribute:

        eigvalsh.vec: Array containing eigenvectors of ``a`` ignoring
            uncertainties (that is, eigenvectors of ``gvar.mean(a)``).
            The eigenvector corresponding to the ``i``-th 
            eigenvalue is ``eigvalsh.vec[:, i]``.


    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('Bad matrix shape: ' + str(a.shape))
    da = a - amean 
    val, vec = numpy.linalg.eigh(amean)
    eigvalsh.vec = vec
    return val + [vec[:, i].dot(da.dot(vec[:, i])) for i in range(vec.shape[1])]

def inv(a):
    """ Inverse of matrix ``a``. 

    Args:
        a: Two-dimensional, square matrix/array of numbers 
            and/or :class:`gvar.GVar`\s.

    Returns:
        The inverse of matrix ``a``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('Bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = numpy.linalg.inv(amean)
    return ainv - ainv.dot(da.dot(ainv))

def solve(a, b):
    """ Find ``x`` such that ``a.dot(x) = b`` for matrix ``a``. 

    Args:
        a: Two-dimensional, square matrix/array of numbers 
            and/or :class:`gvar.GVar`\s.
        b: One-dimensional vector/array of numbers and/or 
            :class:`gvar.GVar`\s, or an array of such vectors.
            Requires ``b.shape[0] == a.shape[1]``.

    Returns:
        The solution ``x`` of ``a.dot(x) = b``, which is equivalent
        to ``inv(a).dot(b)``.

    Raises:
        ValueError: If ``a`` is not square and two-dimensional.
        ValueError: If shape of ``b`` does not match that of ``a``
            (that is ``b.shape[0] != a.shape[1]``).
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('Bad matrix shape: ' + str(a.shape))
    bmean = gvar.mean(b)
    if bmean.shape[0] != a.shape[1]:
        raise ValueError(
            'Mismatch between shapes of a and b: {} {}'.format(a.shape, b.shape)
            )
    xmean = numpy.linalg.solve(amean, bmean)
    ainv = inv(a)
    return xmean + ainv.dot(b-bmean - (a-amean).dot(xmean))
