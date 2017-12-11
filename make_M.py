import numpy as np
import pdb

def m_to_M(m, columns):
    out = np.zeros(columns)
    m *= (columns-1)
    int_m = int(m)
    p_m = m-int_m
    out[int_m] = 1-p_m
    if p_m != 0:
        out[int_m+1] = p_m
    return out

def m_to_M2(m, boundaries):
    diffs = (m - boundaries)
    first = np.where(diffs >= 0)[0][-1]
    second = np.where(diffs <= 0)[0][0]
    delta = boundaries[second] - boundaries[first]
    delta += delta == 0
    M = np.zeros(boundaries.shape)
    M[second] = diffs[first] / delta
    M[first] = 1 - diffs[first] / delta
    return M

def m_to_M_coord(m, boundaries):
    m = np.max([m, boundaries[0]])
    m = np.min([m, boundaries[-1]])
    diffs = (m - boundaries)
    first = np.where(diffs >= 0)[0][-1]
    second = np.where(diffs <= 0)[0][0]
    delta = boundaries[second] - boundaries[first]
    delta += delta == 0
    secondval = diffs[first] / delta
    firstval = 1 - diffs[first] / delta
    return first, firstval, second, secondval

#a = np.random.rand(20)
#a.sort()
#b = np.argsort(a)
#numsteps = 5
#sample_points = np.array(np.linspace(0, len(b)-1, numsteps), dtype=int)
#my_b = a[b[sample_points]]

#T = [t_to_R2(ii, my_b) for ii in a]
#T = np.array(T)
#print T

#t=np.array(xrange(21),dtype=float)/5/4
#R=np.array([t_to_R(ii,5) for ii in t])
#print R