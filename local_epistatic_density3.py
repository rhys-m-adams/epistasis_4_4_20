#!/usr/bin/env python
import pdb
import pandas
from helper import *
from make_energy_funs import A_wt, E_wt, A_heatmap1, A_heatmap3, E_heatmap1, E_heatmap3, sub_heatmap2, get_f1, pos1, pos3, A1, A3, num_muts1, num_muts3, KD1, KD3, A_wt, E_wt, E1, E3, logKD2PWM
import sys
import numpy as np
import os

max_energy = logKD2PWM(-5) - A_wt
min_energy = logKD2PWM(-9.5) - A_wt
num_indices = 41
max_expression = 0.5
min_expression = -1

def energy_ind_fun2(x, bound_energy, A_heatmap, E_heatmap, A_heatmap2, E_heatmap2):
    PWM1_terms = np.array([[A_heatmap[residue, int(ind)] for residue, ind in enumerate(x)]]).T
    PWM2_terms = np.array([[A_heatmap2[ii, jj, x[ii],x[jj]] for ii in range(len(x))] for jj in range(len(x))])
    return_energy1 = PWM1_terms.sum()
    return_energy = return_energy1 + np.nansum(PWM2_terms)/2.

    energy = np.min([max_energy, return_energy])
    energy = np.max([min_energy, energy])

    energy1 = np.min([max_energy, return_energy1])
    energy1 = np.max([min_energy, energy1])

    epistatic_change1 = A_heatmap - PWM1_terms

    epistatic_change = [A_heatmap2[ii, :, x[ii],:] - np.array([PWM2_terms[ii]]).T for ii in range(len(x))]
    epistatic_change.append(epistatic_change1)
    epistatic_change = np.sum(np.array(epistatic_change), axis=0)

    outside_bound1 = np.array([np.max([np.sum((epistatic_change1 + energy1)>=be)/float(19*len(x)), energy1 >= be]) for be in bound_energy])
    outside_bound1[outside_bound1>1] = 1

    outside_bound = np.array([np.max([np.sum((epistatic_change + energy)>=be)/float(19*len(x)), energy >= be]) for be in bound_energy])
    outside_bound[outside_bound>1] = 1

    return np.hstack(([return_energy<bound_energy, return_energy1<bound_energy, outside_bound,  outside_bound1]))

def k_subsets_i(n, k):
    '''
    Yield each subset of size k from the set of intergers 0 .. n - 1
    n -- an integer > 0
    k -- an integer > 0
    '''
    # Validate args
    if n < 0:
        raise ValueError('n must be > 0, got n=%d' % n)
    if k < 0:
        raise ValueError('k must be > 0, got k=%d' % k)
    # check base cases
    if k == 0 or n < k:
        yield set()
    elif n == k:
        yield set(range(n))

    else:
        # Use recursive formula based on binomial coeffecients:
        # choose(n, k) = choose(n - 1, k - 1) + choose(n - 1, k)
        for s in k_subsets_i(n - 1, k - 1):
            s.add(n - 1)
            yield s
        for s in k_subsets_i(n - 1, k):
            yield s

def k_subsets(s, k):
    '''
    Yield all subsets of size k from set (or list) s
    s -- a set or list (any iterable will suffice)
    k -- an integer > 0
    '''
    s = list(s)
    n = len(s)
    for k_set in k_subsets_i(n, k):
        yield set([s[i] for i in k_set])

def plot_local(num_muts, AA_ind, name, num_J, seed_ID, boundary, energy_fun):
    num_bounds = boundary.shape[0]
    KDs = []
    Es = []
    kd_exit_pairs = []
    kn = 101
    en = 102
    total_count = 0.
    viable_count = 0.
    viable_count1 = 0.
    area = 0.
    area1 = 0.
    numeric_cutoff = int(1e6+0.5)
    for ii in range(1, num_muts+1):
        num_subsets = np.math.factorial(10)/np.math.factorial(10-ii)/np.math.factorial(ii)
        predicted_total = 19**ii * num_subsets
        if predicted_total < numeric_cutoff:
            ranges = (',range(1, 20)'*ii)[1:]
            grid_tup = eval('np.meshgrid('+ranges+')')
            flat_grid = np.array([gt.flatten() for gt in grid_tup]).T
            scaling = 1

        num_sampled = 0
        for subset in k_subsets(list(range(len(AA_ind))), ii):

            if predicted_total >= numeric_cutoff:
                fg_size = numeric_cutoff / num_subsets+1
                flat_grid = np.random.randint(1,20,(fg_size,ii))
                scaling = float(19 ** ii)/fg_size

            x = np.zeros((flat_grid.shape[0], len(AA_ind)), dtype = int)
            subset = list(subset)
            x[:, subset] = flat_grid
            temp = np.array([energy_fun(xi) for xi in x])
            total_count += temp.shape[0] * scaling
            viable_temp = temp[:,:num_bounds]
            viable_count += np.sum(viable_temp, axis = 0) * scaling
            area += np.sum(viable_temp * temp[:, (2*num_bounds):(3*num_bounds)],axis=0) * scaling

            viable_temp = temp[:,(num_bounds):(2*num_bounds)]
            viable_count1 += np.sum(viable_temp, axis = 0) * scaling
            area1 += np.sum(viable_temp * temp[:, (3*num_bounds):],axis=0) * scaling

            num_sampled += x.shape[0]

        for bii in range(num_bounds):
            fraction_viable = np.log10(float(viable_count[bii])) - np.log10(float(total_count))
            fraction_viable1 = np.log10(float(viable_count1[bii])) - np.log10(float(total_count))
            exit_percent = float(area[bii]/viable_count[bii])
            exit_percent1 = float(area1[bii]/viable_count1[bii])
            out_text = '%i,%i,%f,%f,%i,%i,%i,%i,%i,%f\n'%(num_J, ii, fraction_viable, exit_percent, seed_id, viable_count[bii], total_count, num_sampled, predicted_total, boundary[bii])
            out_text = out_text + '%i,%i,%f,%f,%i,%i,%i,%i,%i,%f\n'%(0, ii, fraction_viable1, exit_percent1, seed_id, viable_count1[bii], total_count, num_sampled, predicted_total, boundary[bii])
            filename = '%s.csv'%(name)
            out_header = 'num_J,num_muts,log10_fraction_viable,exit_percent,seed,num_viable,total_count,num_seq_sumpled,predicted_total,boundary\n'
            if not os.path.isfile(filename) :
                with open(filename, "a") as myfile:
                    myfile.write(out_header)
                    myfile.write(out_text)

            else:
                with open(filename, "a") as myfile:
                    myfile.write(out_text)
        myfile.close()

seed_id = int(float(sys.argv[2]))
np.random.seed(seed_id)

num_muts = 10
num_J = int(np.sum(num_muts3==2) - 1)
out_name = sys.argv[1]

f1A = get_f1(A3, num_muts3, KD3, A_wt, limit=[logKD2PWM(-9.5), logKD2PWM(-5)])[0]
f1E = get_f1(A3, num_muts3, E3, A_wt, limit=[-1, 0.5])[0]

A_heatmap3_2, E_heatmap3_2 = sub_heatmap2(pos3, num_muts3, KD3 - f1A, E3 - f1E, seed_id)
A_heatmap3_2[~np.isfinite(A_heatmap3_2)]=0
boundary = np.array([-6,-6.5,-7,-7.5,-8,-8.5,-9, -9.5])
bound_energy = logKD2PWM(boundary) - A_wt
plot_local(num_muts, list(range(10)), out_name, num_J, seed_id, boundary, lambda x:energy_ind_fun2(x, bound_energy, A_heatmap3, E_heatmap3, A_heatmap3_2, E_heatmap3_2))
