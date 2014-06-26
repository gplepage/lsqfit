"""
Alias gvar to the old name, gdev. For use with legacy codes only.
"""
# Created by G. Peter Lepage (Cornell University) on 2012-05-27.
# Copyright (c) 2008-2012 G. Peter Lepage. 
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

from gvar import *
import gvar as _gvar

gdev = gvar
GDev = GVar
GDevFactory = GVarFactory
switch_gdev = switch_gvar
restore_gdev = restore_gvar
gdev_factory = gvar_factory
valder_var = valder

def asgdev(g):
    return g 

def rebuild(g, corr=0.0, gdev=gdev):
    return _gvar.rebuild(g, corr=corr, gvar=gdev)
##
