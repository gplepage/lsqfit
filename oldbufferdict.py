""" Tools for reading pickled data using old gvar.BufferDict.

Version 6.0 of lsqfit changed the pickle format for gvar.BufferDict to deal
with changes in version 1.9 of numpy. The code in this module can read
pickled files that use the  old BufferDict. Objects of type BufferDict in the
file are  converted to type OldBufferDict, which can then be converted to the
new BufferDict or otherwise handled.

For example, the following reads into y an old  pickle file containing a
dictionary. The contents are transcribed to a new dictionary (new_y), where
any value y[k] that is an  OldBufferDict is converted to a gvar.BufferDict::

    from oldbufferdict import load_pickled_file, OldBufferDict
    import gvar as gv 

    new_y = {}
    with open(pickle_file_name, 'rb') as ifile:
        y = load_pickled_file(ifile)
        for k in y:
            if isinstance(y[k], OldBufferDict):
                new_y[k] = y[k].to_bufferdict()
            else:
                new_y[k] = y[k]
        print new_y

This code is a bit of a kludge. It ought to help deal with legacy files, but 
it is not guaranteed to work with future releases of lsqfit.
"""

import collections
import numpy
import pickle
import gvar as _gvar

BUFFERDICTDATA = collections.namedtuple('BUFFERDICTDATA',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==None implies a scalar. """
    
class OldBufferDict(object): #collections.MutableMapping):
    """ Stripped down version of old gvar.BufferDict for unpickling legacy files.

    This class is used by :func:`load_pickled_file`, which reads old 
    pickle archives and converts objects of type ``BufferDict`` to 
    ``OldBufferDict``. It assumes that the ``BufferDict``\s were created
    using gvar.BufferDict from versions earlier than version 6 of gvar/lsqfit.
    ``OldBufferDict`` has no use other than this. Its contents can be 
    obtained using method :func:`OldBufferDict.items`, or one-by-one using
    :func:`OldBufferDict.__call__`. It can also be converted to a modern
    gvar.BufferDict using method :func:`OldBufferDict.to_bufferdict`.
    """
    def __init__(self, *args, **kargs):
        super(BufferDict, self).__init__()
    
    ## the __getstate__ used to create old pickle files:
    # def __getstate__(self):
    #     """ Capture state for pickling when elements are GVars. """
    #     if len(self._buf)<1:
    #         return self.__dict__.copy()
    #     odict = self.__dict__.copy()
    #     if isinstance(self._buf[0],_gvar.GVar):
    #         buf = odict['_buf']
    #         del odict['_buf']
    #         odict['_buf.mean'] = _gvar.mean(buf)
    #         odict['_buf.cov'] = _gvar.evalcov(buf)
    #     data = odict['_data']
    #     del odict['_data']
    #     odict['_data.tuple'] = {}
    #     for k in data:
    #         odict['_data.tuple'][k] = (data[k].slice,data[k].shape)
    #     return odict
    
    def __setstate__(self,odict):
        """ Restore state when unpickling when elements are GVars. """
        if '_buf.mean' in odict:
            buf = _gvar.gvar(odict['_buf.mean'],odict['_buf.cov'])
            del odict['_buf.mean']
            del odict['_buf.cov']
            odict['_buf'] = buf
        if '_data.tuple' in odict:
            data = odict['_data.tuple']
            del odict['_data.tuple']
            odict['_data'] = {}
            for k in data:
                odict['_data'][k] = BUFFERDICTDATA(slice=data[k][0],
                                                    shape=data[k][1])
        self.__dict__.update(odict)
    
    def to_bufferdict(self):
        " Convert to new gvar.BufferDict. "
        return _gvar.BufferDict(self.items())

    def items(self):
        " Return list of (key, value) tuples. "
        ans = []
        for k in self._data:
            ans.append((k, self(k)))
        return ans

    def __call__(self,k):
        """ Return piece of buffer corresponding to key ``k``. """
        if k not in self._data:
            raise KeyError("undefined key: %s" % str(k))
        if isinstance(self._buf,list):
            self._buf = numpy.array(self._buf)
        d = self._data[k]
        ans = self._buf[d.slice]
        return ans if d.shape is None else ans.reshape(d.shape)

def load_pickled_file(fobj):
    " Load file pickled with old gvar.BufferDict, converting to OldBufferDict. "
    ans = ""
    lastline = ""
    for line in fobj:
        if 'BufferDict' in line:
            lastline = lastline.replace('gvar','oldbufferdict')
            line = line.replace('BufferDict', 'OldBufferDict')
        if lastline != "":
            ans += lastline
        lastline = line
    ans += lastline
    return pickle.loads(ans)
