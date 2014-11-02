# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-14 G. Peter Lepage. 
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

import collections
import numpy
import copy
import pickle
import json
try:
    # python 2
    from StringIO import StringIO as _StringIO
    _BytesIO = _StringIO
except ImportError:
    # python 3
    from io import BytesIO as _BytesIO
    from io import StringIO as _StringIO
import gvar as _gvar

BUFFERDICTDATA = collections.namedtuple('BUFFERDICTDATA',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==None implies a scalar. """
    
class BufferDict(collections.OrderedDict): 
    """ Ordered dictionary whose data are packed into a 1-d buffer (numpy.array).
        
    A |BufferDict| object is an ordered dictionary whose values must
    either be scalars or arrays (like :mod:`numpy` arrays, with arbitrary
    shapes). The scalars and arrays are assembled into different parts of a
    single one-dimensional buffer. The various scalars and arrays are
    retrieved using keys: *e.g.*,
        
        >>> a = BufferDict()
        >>> a['scalar'] = 0.0
        >>> a['vector'] = [1.,2.]
        >>> a['tensor'] = [[3.,4.],[5.,6.]]
        >>> print(a.flatten())              # print a's buffer
        [ 0.  1.  2.  3.  4.  5.  6.]
        >>> for k in a:                     # iterate over keys in a
        ...     print(k,a[k])
        scalar 0.0
        vector [ 1.  2.]
        tensor [[ 3.  4.]
         [ 5.  6.]]
        >>> a['vector'] = a['vector']*10    # change the 'vector' part of a
        >>> print(a.flatten())
        [  0.  10.  20.   3.   4.   5.   6.]
        
    The first four lines here could have been collapsed to one statement::
        
        a = BufferDict(scalar=0.0,vector=[1.,2.],tensor=[[3.,4.],[5.,6.]])
        
    or ::
        
        a = BufferDict([('scalar',0.0),('vector',[1.,2.]),
                        ('tensor',[[3.,4.],[5.,6.]])])
        
    where in the second case the order of the keys is preserved in ``a``
    (since ``BufferDict`` is an ordered dictionary).
        
    The keys and associated shapes in a |BufferDict| can be transferred to a
    different buffer, creating a new |BufferDict|: *e.g.*, using ``a`` from
    above,
        
        >>> buf = numpy.array([0.,10.,20.,30.,40.,50.,60.])
        >>> b = BufferDict(a, buf=buf)          # clone a but with new buffer
        >>> print(b['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        >>> b['scalar'] += 1
        >>> print(buf)
        [  1.  10.  20.  30.  40.  50.  60.]
        
    Note how ``b`` references ``buf`` and can modify it. One can also
    replace the buffer in the original |BufferDict| using, for example,
    ``a.buf = buf``:
        
        >>> a.buf = buf
        >>> print(a['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        >>> a['tensor'] *= 10.
        >>> print(buf)
        [  1.  10.  20.  300.  400.  500.  600.]
        
    ``a.buf`` is the numpy array used for ``a``'s buffer. It can be used to
    access and change the buffer directly. In ``a.buf = buf``, the new
    buffer ``buf`` must be a :mod:`numpy` array of the correct shape. The
    buffer can also be accessed through iterator ``a.flat`` (in analogy
    with :mod:`numpy` arrays), and through ``a.flatten()`` which returns a
    copy of the buffer.

    When creating a |BufferDict| from a dictionary (or another |BufferDict|),
    the keys included and their order can be specified using a list of keys: 
    for example, ::

        >>> d = dict(a=0.0,b=[1.,2.],c=[[3.,4.],[5.,6.]],d=None)
        >>> print(d)
        {'a': 0.0, 'c': [[3.0, 4.0], [5.0, 6.0]], 'b': [1.0, 2.0], 'd': None}
        >>> a = BufferDict(d, keys=['d', 'b', 'a'])
        >>> for k in a:
        ...     print(k, a[k])
        d None
        b [1.0 2.0]
        a 0.0
         
    A |BufferDict| functions like a dictionary except: a) items cannot be
    deleted once inserted; b) all values must be either scalars or arrays
    of scalars, where the scalars can be any noniterable type that works
    with :mod:`numpy` arrays; and c) any new value assigned to an existing 
    key must have the same size and shape as the original value.
        
    Note that |BufferDict|\s can be pickled and unpickled even when they 
    store |GVar|\s (which themselves cannot be pickled separately).
    """
    def __init__(self, *args, **kargs):
        super(BufferDict, self).__init__()
        self.shape = None
        if len(args)==0:
            # kargs are dictionary entries 
            self._buf = numpy.array([],int)
            for k in sorted(kargs):
                self[k] = kargs[k]
        elif len(args) == 1 and 'keys' in kargs and len(kargs) == 1:
            self._buf = numpy.array([],int)
            try:
                for k in kargs['keys']:
                    self[k] = args[0][k] 
            except KeyError:
                raise KeyError('Dictionary does not contain key in keys: ' + str(k))               
        else:
            if len(args)==2 and len(kargs)==0:
                bd,buf = args
            elif len(args)==1 and len(kargs)==0:
                bd = args[0]
                buf = None
            elif len(args)==1 and 'buf' in kargs and len(kargs)==1:
                bd = args[0]
                buf = kargs['buf']
            else:
                raise ValueError("Bad arguments for BufferDict.")
            if isinstance(bd, BufferDict):
                # make copy of BufferDict bd, possibly with new buffer 
                # copy keys, slices and shapes
                for k in bd:
                    super(BufferDict, self).__setitem__(
                        k, super(BufferDict, bd).__getitem__(k)
                        )
                # copy buffer or use new one
                self._buf = (numpy.array(bd._buf) if buf is None 
                             else numpy.asarray(buf))
                if bd.size != self.size:
                    raise ValueError("buf is wrong size --- %s not %s"
                                     % (self.size, bd.size))
                if self._buf.ndim != 1:
                    raise ValueError("buf must be 1-d, not shape = %s"
                                     % (self._buf.shape,))
            elif buf is None:
                self._buf = numpy.array([],int)
                # add initial data  
                if hasattr(bd,"keys"):
                    # bd a dictionary 
                    for k in sorted(bd):
                        self[k] = bd[k]
                else:
                    # bd an array of tuples 
                    if not all([(isinstance(bdi,tuple) 
                               and len(bdi)==2) for bdi in bd]):
                        raise ValueError(
                            "BufferDict argument must be dict or list of 2-tuples.")
                    for ki,vi in bd:
                        self[ki] = vi
            else:
                raise ValueError(
                    "bd must be a BufferDict in BufferDict(bd,buf), not %s"
                                    % str(type(bd)))

    def __getstate__(self):
        """ Capture state for pickling when elements are GVars. """
        state = {}
        buf = self._buf
        if len(self._buf) > 0 and isinstance(self._buf[0], _gvar.GVar):
            state['buf'] = ( _gvar.mean(buf),  _gvar.evalcov(buf))
        else:
            state['buf'] = numpy.asarray(buf)
        layout = collections.OrderedDict()
        od = super(BufferDict, self)
        for k in self:
            layout[k] = (od.__getitem__(k).slice, od.__getitem__(k).shape)
        state['layout'] = layout
        return state
    
    def __setstate__(self, state):
        """ Restore state when unpickling when elements are GVars. """
        layout = state['layout']
        buf = state['buf']
        if isinstance(buf, tuple):
            buf = _gvar.gvar(*buf)
        for k in layout:
            super(BufferDict, self).__setitem__(
                k, 
                BUFFERDICTDATA(slice=layout[k][0], shape=layout[k][1])
                )
        self._buf = buf

    def __reduce_ex__(self, dummy):
        return (BufferDict, (), self.__getstate__())
    
    def add(self,k,v):
        """ Augment buffer with data ``v``, indexed by key ``k``.
            
        ``v`` is either a scalar or a :mod:`numpy` array (or a list or
        other data type that can be changed into a numpy.array).
        If ``v`` is a :mod:`numpy` array, it can have any shape.
            
        Same as ``self[k] = v`` except when ``k`` is already used in
        ``self``, in which case a ``ValueError`` is raised.
        """
        if k in self:
            raise ValueError("Key %s already used." % str(k))
        else:
            self[k] = v
    
    def __getitem__(self,k):
        """ Return piece of buffer corresponding to key ``k``. """
        if not super(BufferDict, self).__contains__(k):
            raise KeyError("undefined key: %s" % str(k))
        if isinstance(self._buf, list):
            self._buf = numpy.array(self._buf)
        d = super(BufferDict, self).__getitem__(k)
        ans = self._buf[d.slice]
        return ans if d.shape is None else ans.reshape(d.shape)
    
    def __setitem__(self,k,v):
        """ Set piece of buffer corresponding to ``k`` to value ``v``. 
            
        The shape of ``v`` must equal that of ``self[k]``. If key ``k`` 
        is not in ``self``, use ``self.add(k,v)`` to add it.
        """
        if k not in self:
            v = numpy.asarray(v)
            if v.shape==():
                # add single piece of data 
                super(BufferDict, self).__setitem__(k, BUFFERDICTDATA(slice=len(self._buf),shape=None))
                self._buf = numpy.append(self._buf,v)
            else:
                # add array 
                n = numpy.size(v)
                i = len(self._buf)
                super(BufferDict, self).__setitem__(k, BUFFERDICTDATA(slice=slice(i,i+n),shape=tuple(v.shape)))
                self._buf = numpy.append(self._buf,v)
        else:
            d = super(BufferDict, self).__getitem__(k)
            if d.shape is None:
                try:
                    self._buf[d.slice] = v
                except ValueError:
                    raise ValueError("*** Not a scalar? Shape=%s" 
                                     % str(numpy.shape(v)))
            else:
                v = numpy.asarray(v)
                try:
                    self._buf[d.slice] = v.flat
                except ValueError:
                    raise ValueError("*** Shape mismatch? %s not %s" % 
                                     (str(v.shape),str(d.shape)))
    
    def __delitem__(self,k):
        raise NotImplementedError("Cannot delete items from BufferDict.")
                
    def __str__(self):
        ans = "{"
        for k in self:
            ans += "%s: %s," % (repr(k), repr(self[k]))
        if ans[-1] == ',':
            ans = ans[:-1]
            ans += "}"
        return ans
    
    def __repr__(self):
        cn = self.__class__.__name__
        return cn+"("+repr([k for k in self.items()])+")"
    
    def _getflat(self):
        return self._buf.flat
    
    def _setflat(self,buf):
        """ Assigns buffer with buf if same size. """
        self._buf.flat = buf
    
    flat = property(_getflat,_setflat,doc='Buffer array iterator.')
    def flatten(self):
        """ Copy of buffer array. """
        return numpy.array(self._buf)
    
    def _getdtype(self):
        return self._buf.dtype

    dtype = property(_getdtype, doc='Data type of buffer array elements.')

    def _getbuf(self):  
        return self._buf
    
    def _setbuf(self,buf):
        """ Replace buffer with ``buf``. 
            
        ``buf`` must be a 1-dimensional :mod:`numpy` array of the same size
        as ``self._buf``.
        """
        if isinstance(buf,numpy.ndarray) and buf.shape == self._buf.shape:
            self._buf = buf
        else:
            raise ValueError(
                "New buffer wrong type or shape ---\n    %s,%s   not   %s,%s"
                % (type(buf), numpy.shape(buf), 
                type(self._buf), self._buf.shape))
    
    buf = property(_getbuf,_setbuf,doc='The buffer array (not a copy).')
    def _getsize(self):
        """ Length of buffer. """
        return len(self._buf)
    
    size = property(_getsize,doc='Size of buffer array.')
    def slice(self,k):
        """ Return slice/index in ``self.flat`` corresponding to key ``k``."""
        return super(BufferDict, self).__getitem__(k).slice
    
    def isscalar(self,k):
        """ Return ``True`` if ``self[k]`` is scalar else ``False``."""
        return super(BufferDict, self).__getitem__(k).shape is None
    
    def dump(self, fobj, use_json=False):
        """ Serialize |BufferDict| in file object ``fobj``.
                    
        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obviously). :mod:`json` does not handle 
        non-string valued keys very well. This attempts a workaround, but
        it will only work in simpler cases. Serialization only works when
        :mod:`pickle` (or :mod:`json`) knows how to serialize the data type
        stored in the |BufferDict|'s buffer (or for |GVar|\s).
        """
        if not use_json:
            pickle.dump(self, fobj)
        else:
            if isinstance(self._buf[0], _gvar.GVar):
                tmp = _gvar.mean(self)
                cov = _gvar.evalcov(self._buf)
            else:
                tmp = self
                cov = None
            d = {}
            keys = []
            for k in tmp:
                jk = 's:' + k if str(k) == k else 'e:'+str(k)
                keys.append(jk)
                d[jk] = tmp[k] if self.isscalar(k) else tmp[k].tolist()
            d['keys'] = keys
            if cov is not None:
                d['cov'] = cov.tolist()
            json.dump(d, fobj)
    
    def dumps(self, use_json=False):
        """ Serialize |BufferDict| into string.
                    
        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obviously). :mod:`json` does not handle
        non-string valued keys very well. This attempts a workaround, but
        it will only work in simpler cases (e.g., integers, tuples of
        integers, etc.). Serialization only works when :mod:`pickle` (or
        :mod:`json`) knows how to serialize the data type stored in the
        |BufferDict|'s buffer (or for |GVar|\s).
        """
        f = _StringIO() if use_json else _BytesIO()
        self.dump(f, use_json=use_json)
        return f.getvalue()
    
    @staticmethod
    def load(fobj, use_json=False):
        """ Load serialized |BufferDict| from file object ``fobj``.
                    
        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obvioulsy).
        """
        if not use_json:
            return pickle.load(fobj)
        else:
            d = json.load(fobj)
            ans = BufferDict()
            for jk in d['keys']:
                k = str(jk[2:]) if jk[0] == 's' else eval(jk[2:])
                ans[k] = d[jk]
            if 'cov' in d:
                ans.buf = _gvar.gvar(ans._buf,d['cov'])
            return ans
    
    @staticmethod
    def loads(s, use_json=False):
        """ Load serialized |BufferDict| from file object ``fobj``.
                    
        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obvioulsy).
        """
        f = _StringIO(s) if use_json else _BytesIO(s)
        return BufferDict.load(f, use_json=use_json)


def asbufferdict(g, keylist=None):
    """ Convert ``g`` to a BufferDict, keeping only ``g[k]`` for ``k in keylist``.

    ``asbufferdict(g)`` will return ``g`` if it is already a 
    :class:`gvar.BufferDict`; otherwise it will convert the dictionary-like 
    object into a :class:`gvar.BufferDict`. If ``keylist`` is not ``None``, 
    only objects ``g[k]`` for which ``k in keylist`` are kept.
    """
    if isinstance(g, BufferDict) and keylist is None:
        return g 
    if keylist is None:
        return BufferDict(g)
    ans = BufferDict()
    for k in keylist:
        ans[k] = g[k]
    return ans

