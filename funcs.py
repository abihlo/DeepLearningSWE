"""
This script contains all desired functions
"""

from firedrake import *

def initial_conditions(x,t,num=1):
    if num==1:
        u = 0.05 * exp(-400*(x-0.4)**2) + 1
        h = 0.1 * exp(-400*(x-0.4)**2) + 1
    else:
        raise NotImplementedError('num=%s is not an implemented initial condition' % num)

    return u, h

def bottom(x,num=1):
    if num==1:
        beta = 1 - 0.04*exp(-100*(x-0.6)**2)
    else:
        raise NotImplementedError('num=%s is not an implemented bottom' % num)

    return beta
