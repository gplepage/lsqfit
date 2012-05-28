from lsqfit import *
import gdev as gd
import numpy as np
import pickle
import os


llist = """
190, 226, 231, 265, 330, 343, 403-404, 414, 427, 450, 463, 472, 480, 492, 506,
516, 555, 611, 684, 708-709, 720, 733, 754, 761-762, 781, 829, 878, 886,
894-896, 984, 1041-1044, 1060-1070
"""
nllist = []
for x in llist.split(','):
    if '-' in x:
        a,b = x.split('-')
        for i in range(eval(a),eval(b)+1):
            nllist.append(i)
    else:
        nllist.append(eval(x))
            
llist = nllist

# llist = "["+llist+"]"
# llist = eval(','.join(llist.strip().split('-')))

lineno = 0
fmt = "%10s"
for line in open("lsqfit.py","r"):
    lineno += 1
    outline = line[:-1]
    if lineno in llist:
        outline = (fmt%str(lineno))+outline
    else:
        outline = (fmt%" ")+outline
    print outline
