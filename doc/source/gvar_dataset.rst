:mod:`gvar.dataset` - Random Data Sets
==================================================

.. |GVar| replace:: :class:`gvar.GVar`

.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. |Dataset| replace:: :class:`gvar.dataset.Dataset`

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. module:: gvar.dataset
   :synopsis: Tools for analyzing collections of random samples.

Introduction 
------------------
:mod:`gvar.dataset` contains a several tools for collecting and analyzing
random samples from arbitrary distributions. The random samples are
represented by lists of numbers or arrays, where each number/array is a new
sample from the underlying distribution. For example, six samples from a
one-dimensional gaussian distribution, 1±1, might look like ::
    
    >>> random_numbers = [1.739, 2.682, 2.493, -0.460, 0.603, 0.800]
    
while six samples from a two-dimensional distribution, [1±1, 2±1],
might be ::
    
    >>> random_arrays = [[ 0.494, 2.734], [ 0.172, 1.400], [ 1.571, 1.304], 
    ...                  [ 1.532, 1.510], [ 0.669, 0.873], [ 1.242, 2.188]]
    
Samples from more complicated multidimensional distributions are represented
by dictionaries whose values are lists of numbers or arrays: for example, ::

   >>> random_dict = dict(n=random_numbers, a=random_arrays)

where list elements ``random_dict['n'][i]`` and ``random_dict['a'][i]`` are
part of the same multidimensional sample for every ``i`` --- that is, the
lists for different keys in the dictionary are synchronized one with the
other.
    
With large samples, we typically want to estimate the mean value of the 
underlying distribution. This is done using :func:`gvar.dataset.avg_data`:
for example, ::
    
    >>> print(avg_data(random_numbers))
    1.31(45)
    
indicates that ``1.31(45)`` is our best guess, based only upon the samples in
``random_numbers``, for the mean of the distribution from which those samples
were drawn. Similarly ::
    
    >>> print(avg_data(random_arrays))
    [0.95(22) 1.67(25)]
      
indicates that the means for the two-dimensional distribution behind
``random_arrays`` are ``[0.95(22), 1.67(25)]``. :func:`avg_data` can also
be applied to a dictionary whose values are lists of numbers/arrays: for
example, ::
    
    >>> print(avg_data(random_dict))
    {'a': array([0.95(22), 1.67(25)], dtype=object),'n': 1.31(45)}
    
Class |Dataset| can be used to assemble dictionaries containing
random samples. For example, imagine that the random samples above were
originally written into a file, as they were generated::
    
    # file: datafile
    n 1.739
    a [ 0.494, 2.734]
    n 2.682
    a [ 0.172, 1.400]
    n 2.493
    a [ 1.571, 1.304]
    n -0.460
    a [ 1.532, 1.510]
    n 0.603
    a [ 0.669, 0.873]
    n 0.800
    a [ 1.242, 2.188]
    
Here each line is a different random sample, either from the one-dimensional
distribution (labeled ``n``) or from the two-dimensional distribution (labeled
``a``). Assuming the file is called ``datafile``, this data can be read into
a dictionary, essentially identical to the ``data`` dictionary above, using::
    
    >>> data = Dataset("datafile")
    >>> print(data['a'])
    [array([ 0.494, 2.734]), array([ 0.172, 1.400]), array([ 1.571, 1.304]) ... ]
    >>> print(avg_data(data['n']))
    1.31(45)
    
The brackets and commas can be omitted in the input file for one-dimensional
arrays: for example, ``datafile`` (above) could equivalently be written ::
    
    # file: datafile
    n 1.739
    a 0.494 2.734
    n 2.682
    a 0.172 1.400
    ...
   
Other data formats may also be easy to use. For example, a data file written 
using ``yaml`` would look like ::
    
    # file: datafile
    ---
    n: 1.739
    a: [ 0.494, 2.734]
    ---
    n: 2.682
    a: [ 0.172, 1.400]
    .
    .
    .
    
and could be read into a |Dataset| using::
    
    import yaml
    
    data = Dataset()
    with open("datafile", "r") as dfile:
        for d in yaml.load_all(dfile.read()):   # iterate over yaml records  
            data.append(d)                      # d is a dictionary
    
Finally note that data can be binned, into bins of size ``binsize``, using
:func:`gvar.dataset.bin_data`. For example,
``gvar.dataset.bin_data(data, binsize=3)`` replaces every three samples in
``data`` by the average of those samples. This creates a dataset that is
``1/3`` the size of the original but has the same mean. Binning is useful
for making large datasets more manageable, and also for removing
sample-to-sample correlations. Over-binning, however, erases statistical
information.
    
Class |Dataset| can also be used to build a dataset sample by
sample in code: for example, ::
    
    >>> a = Dataset()
    >>> a.append(n=1.739, a=[ 0.494, 2.734])
    >>> a.append(n=2.682, a=[ 0.172, 1.400])
    ...
    
creates the same dataset as above.
 

Functions
----------   
The functions defined in the module are:

.. autofunction:: gvar.dataset.avg_data(data, spread=False, median=False, bstrap=False, noerror=False, warn=True)

.. autofunction:: gvar.dataset.autocorr(data)

.. autofunction:: gvar.dataset.bin_data(data, binsize=2)

.. autofunction:: gvar.dataset.bootstrap_iter(data, n=None)


Classes
---------
:class:`gvar.dataset.Dataset` is used to assemble random samples from
multidimensional distributions:

.. autoclass:: gvar.dataset.Dataset

   The main attributes and methods are:
   
   .. autoattribute:: samplesize
      
   .. automethod:: append(*args, **kargs)
   
   .. automethod:: extend(*args, **kargs)
   
   .. automethod:: grep(rexp)
   
   .. automethod:: slice(sl)
   
   .. automethod:: arrayzip(template)
   
   .. automethod:: trim()
   
   .. automethod:: toarray()
    

