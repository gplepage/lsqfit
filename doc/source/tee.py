import sys

SYSSTREAMS = [sys.stdout, sys.stderr]

class tee :
    """ redirect same output to two streams 
        
    Typical usage is
    
        sys.stdout = tee(sys.stdout, open("outputfile","w"))
        
    which causes all output to be directed both to standard output and
    to a file named "outputfile."
    """
    def __init__(self, _fd1, _fd2) :
        self.fd1 = _fd1
        self.fd2 = _fd2
    ##
    def __del__(self) :
        if self.fd1 not in SYSSTREAMS :
            self.fd1.close()
        if self.fd2 not in SYSSTREAMS :
            self.fd2.close()
    ##
    def write(self, text) :
        self.fd1.write(text)
        self.fd2.write(text)
    ##
    def flush(self) :
        self.fd1.flush()
        self.fd2.flush()
    ##
##
