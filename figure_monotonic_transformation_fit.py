#!/opt/hpc/bin/python2.7
# -*- coding: utf-8 -*-
from cvxopt import matrix, solvers, sparse
from make_M import *
from scipy.sparse import coo_matrix, dia_matrix, vstack, linalg, csc_matrix, hstack, diags, eye
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import numpy as np
import pdb
import time
import random
import itertools
import pandas
from scipy.stats import pearsonr
from scipy import stats
from labeler import Labeler
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from data_preparation_transformed import get_f1, get_data, get_data_ind
from get_fit_PWM_transformation import get_transformations, get_spline_transformations
from figure_1 import plot_epistasis
solvers.options['show_progress'] = False

def make_residual_matrix(m, my_b, y_inds=None, y_shape=None):
    if y_inds is None:
        y_inds = range(len(m))
    if y_shape is None:
        y_shape = len(m)
    coords = [m_to_M_coord(ii, my_b) for ii in m]
    ycoords = [ii for ii in y_inds] * 2
    xcoords = [coords[ii][0] for ii in range(len(coords))] + [coords[ii][2] for ii in range(len(coords))]
    vals = [coords[ii][1] for ii in range(len(coords))] + [coords[ii][3] for ii in range(len(coords))]
    M = coo_matrix((vals, (ycoords, xcoords)), shape=(y_shape, len(my_b)))
    return M

def get_data_splits(num_points, k=10):
    indices = range(num_points)
    random.shuffle(indices)
    split_at = np.array(np.linspace(0, num_points, k + 1), dtype=int)
    data_set = []
    for ii in range(k):
        data_set.append(indices[split_at[ii]:split_at[ii + 1]])
    return data_set

def A2C_matrix(A, num_muts, vals, my_b):
    # calculate the C matrix such that C.dot(x) = f_1
    # get the subset of A which only has 1 mutant
    just_1_mut = num_muts==1
    A1 = A[just_1_mut].todense()

    #find the columns where the mutant occurs. This is used to rearrange the columns so that f1_vals[f1_ind[ii]] corresponds to A[:,ii] when calculating f1
    f1_ind = np.where(A1)[1]
    f1_vals = vals[just_1_mut]
    #calculate the m matrices for single mutants, and then rearrange them so that A.dot(f1_m).dot(x) = f1
    #quick test - C[just_1_mut].dot(my_b) = f1_vals, C.dot(my_b) = f1
    f1_m = make_residual_matrix(f1_vals, my_b, y_inds=f1_ind, y_shape=A.shape[1])
    C = A.dot(coo_matrix(f1_m/f1_m.sum(axis=1)))
    return C

def make_matrices(M, my_b, C, wt_val, alpha, upper_constrain, lower_constrain):
    grid_size = my_b.shape[0]
    #minimize ||f-f_1||^2
    #We want to find the geometry of the problem so instead we find:
    #minimize ||Mx-Cx||^2 w.r.t. x
    #Remove Mx-d corresponding to zero or single mutants for efficiency...
    #the error is defined to be 0 for these data points
    #M = csc_matrix(make_residual_matrix(vals, my_b))
    Obj = M - C
    upper_constrain = np.where(upper_constrain)[0]
    lower_constrain = np.where(lower_constrain)[0]
    num_contraints = upper_constrain.shape[0] + lower_constrain.shape[0]
    boundary = csc_matrix((np.ones(num_contraints), (upper_constrain.tolist()+lower_constrain.tolist(), range(num_contraints))),
        shape=(Obj.shape[0], num_contraints))
    upper_Obj = hstack((Obj, boundary))

    #lower_Obj = diags([-alpha,2*alpha,-alpha], offsets=[0,1,2], shape=(my_b.shape[0]-2,upper_Obj.shape[1]))
    #bound_Obj = csc_matrix(vstack((upper_Obj, lower_Obj)))
    bound_Obj = upper_Obj
    q = matrix(np.zeros((bound_Obj.shape[1], 1)))

    #subject to the absolute constraint (A*x = b)
    #f(M_wt) = 0
    #fmax - fmin = 1
    M_wt = np.array(make_residual_matrix([wt_val], my_b).todense()).flatten()
    M_constraint = m_to_M2(my_b[-1], my_b) - m_to_M2(my_b[0], my_b)
    A1 = np.zeros((2, bound_Obj.shape[1]))
    A1[:, :Obj.shape[1]] = np.array([M_wt, M_constraint])

    A2 = np.array(C[upper_constrain].todense())
    A2[:,-1] -= 1
    A2 = np.hstack((A2, -np.array(boundary[upper_constrain.tolist()].todense())))

    A3 = np.array(C[lower_constrain].todense())
    A3[:,0]  += 1
    A3 = np.hstack((A3, -np.array(boundary[lower_constrain.tolist()].todense())))

    A = np.vstack((A1, A2, A3))
    A = matrix(A)

    b = np.array([[0, -1]]).T
    #b = np.array([[0, -1]]).T
    b = np.vstack((b, np.zeros((A2.shape[0]+A3.shape[0], 1))))
    b = matrix(b)

    #With inequality Δx < 0,
    # G*x <= h
    offsets = np.array([0, 1])
    derivative = np.array([[-1., 1.]]).repeat(grid_size, axis=0)
    G1 = dia_matrix((derivative.T, offsets), shape=(grid_size - 1, bound_Obj.shape[1])).todense()

    G = G1
    h = np.zeros(G.shape[0])

    #change to cvxopt format
    G = matrix(G)
    h = matrix(h)

    P = (bound_Obj.T).dot(bound_Obj)

    def spline_penalty(grid_points, alpha):
        n = grid_points.shape[0]
        delta_b = np.diff(grid_points)
    	W = np.diag(np.ones(n-3) * delta_b[:-2]/6,k=-1) + np.diag(np.ones(n-3) * delta_b[2:]/6,k=1) + np.diag(np.ones(n-2) * (delta_b[:-1]+delta_b[1:]) / 3,k=0)
    	delta = np.array(diags([1./delta_b[:-1],-1./delta_b[:-1] - 1./delta_b[1:],1./delta_b[1:]], offsets=[0,1,2], shape=(n-2,n)).todense())
    	K = delta.T.dot(np.linalg.lstsq(W, delta)[0])
    	A = np.linalg.inv(np.eye(K.shape[0]) - alpha * K)
    	return K, 1 - np.mean(A.diagonal())

    K, coeff = spline_penalty(my_b, alpha)
    Obj = np.array(Obj.todense())
    S = Obj.dot(np.linalg.lstsq(Obj.T.dot(Obj) + alpha * K, Obj.T)[0])
    P = P.todense()

    P[:K.shape[0],:K.shape[1]] += alpha * K
    P = matrix(P)
    return P, q, G, h, A, b, M, upper_Obj

def fit_energy(M, my_b, C, wt_val, a):
    grid_size = my_b.shape[0]
    boundary_exceeded = True
    exceeded_upper = np.zeros(C.shape[0])==1
    exceeded_lower = np.zeros(C.shape[0])==1
    count = 0

    while boundary_exceeded:
        P, q, G, h, A, b, M, Obj = make_matrices(M, my_b, C, wt_val, a, exceeded_upper, exceeded_lower)
        if count>0:
            ret = solvers.qp(P, q, G = G, h = h, A=A, b=b, init_vals=matrix(fit_x))
        else:
            ret = solvers.qp(P, q, G = G, h = h, A=A, b=b, init_vals=my_b)
        fit_x = np.array(ret['x'])
        raw_f1 =  C.dot(fit_x[:grid_size])
        raw_f1 = raw_f1.flatten()
        f = M.dot(fit_x[:grid_size])
        temp_f1 = f - Obj.dot(fit_x)
        temp_f1 = temp_f1.flatten()
        f_max = np.max(fit_x[:grid_size ])
        f_min = np.min(fit_x[:grid_size ])
        new_exceeded_upper = (exceeded_upper | ((temp_f1) >= f_max)) & ((raw_f1) >= f_max).flatten()
        new_exceeded_lower = (exceeded_lower | ((temp_f1) <= f_min)) & ((raw_f1) <= f_min).flatten()
        count += 1
        print 'upper: %i / %i, lower: %i / %i disagreed/out of bounds'%((new_exceeded_upper != exceeded_upper).sum(), new_exceeded_upper.sum(),(new_exceeded_lower != exceeded_lower).sum(), new_exceeded_lower.sum())
        if count >10:
            boundary_exceeded = False
        if ((new_exceeded_upper != exceeded_upper).sum()==0) and ((new_exceeded_lower != exceeded_lower).sum()==0):
            boundary_exceeded = False
        else:
            exceeded_upper = new_exceeded_upper
            exceeded_lower = new_exceeded_lower

    return fit_x[:grid_size]


def scan_fits(A, vals, lims, alphas, grid_size=100):
    energies = np.nan * np.ones(vals.shape) # will be filled out at the end
    num_muts = np.array(A.sum(axis=1)).flatten()
    wt_val = np.mean(vals[num_muts == 0])

    f1, x = get_f1(A, num_muts, vals, wt_val, lims)

    #remove nan data for fits
    OK_data = np.isfinite(vals)
    A = A[OK_data]
    vals = vals[OK_data]
    num_muts = num_muts[OK_data]
    f1 = f1[OK_data]
    #calculate number of mutants away from wt and wt_val

    usethis = (num_muts>=2) #& (f1>lims[0]) & (f1<lims[1])

    # find the step sizes so that there are approximately an equal number of data points in each step
    my_b = np.linspace(lims[0], lims[1], grid_size)

    # calculate the C matrix such that C.dot(x) = f_1
    vals = vals - wt_val
    my_b = my_b - wt_val
    lims = np.array(lims) - wt_val
    wt_val = 0
    C = A2C_matrix(A, num_muts, vals, my_b)
    Asub = A[usethis]
    Csub = C[usethis]
    vals_sub = vals[usethis]

    unq, unq_ind, inv = np.unique(np.array(Asub.todense()), axis=0, return_index=True, return_inverse=True)

    Csub = csc_matrix([Csub[np.where(inv==ii)[0]].mean(axis=0).tolist()[0] for ii in set(inv)])
    M = csc_matrix(make_residual_matrix(vals_sub, my_b))
    M = csc_matrix([M[np.where(inv==ii)[0]].mean(axis=0).tolist()[0] for ii in set(inv)])
    k = 10
    data_set = get_data_splits(M.shape[0], k=k)

    objective = []
    for a in alphas:
        t0 = time.time()
        g = 0
        for ii in range(len(data_set)):
            print 'rotate set'
            test_set = data_set.pop(0)
            training_set = list(itertools.chain(*data_set))
            fit_x = fit_energy(M[training_set], my_b, Csub[training_set], wt_val, a)

            #M = csc_matrix(make_residual_matrix(vals_sub, my_b))

            f = M.dot(fit_x).flatten()
            f_max = np.max(fit_x)
            f_min = np.min(fit_x)

            raw_f1 =  Csub.dot(fit_x[:grid_size])
            f1 = raw_f1[test_set].flatten()
            f1[f1>f_max] = f_max
            f1[f1<f_min] = f_min
            g += np.nansum((f1-f[test_set])**2)/np.nansum((f[test_set] - np.nanmean(f[test_set]))**2)/k
            data_set.append(test_set)

        objective.append(g)
        if min(objective) == g:
            best_x = fit_x
        print 'alpha: %f  SSE: %f  time: %f '%(a, g, time.time() - t0)

    objective = np.array(objective).flatten()
    best_alpha = alphas[np.argsort(objective)[0]]
    fit_x = fit_energy(M, my_b, Csub, wt_val, best_alpha)

    M = csc_matrix(make_residual_matrix(vals, my_b))
    energies[OK_data] = -M.dot(fit_x).flatten()
    energies = energies / (np.nanmax(energies) - np.nanmin(energies))

    return energies, alphas, objective


def plot_transformation(x, y, lims, ax, xname):
    ind = np.argsort(x)
    x = np.array(x[ind])
    y = np.array(y[ind])

    usethis = np.isfinite(x) & np.isfinite(y)
    x = x[usethis]
    y = y[usethis]

    A = np.vstack((10**x, np.ones(x.shape))).T
    myf = np.linalg.lstsq(A, y)
    fx =  myf[0][0] * 10**x + myf[0][1]
    #ax.plot(10**x, fx, label = r'$E\propto 10^F$,$R^2$: %.2f'%(pearsonr(fx, y)[0]**2), lw=2, zorder=11,c=[0.3,1,0.3])


    ##############################
    A = np.vstack((x, np.ones(x.shape))).T
    myf = np.linalg.lstsq(A, y)
    fx =  myf[0][0] * x + myf[0][1]
    ax.plot(10**x, fx, label = r'$R^2$: %.2f'%(pearsonr(fx, y)[0]**2), lw=2, zorder=11,c=[1,0.3,0.3])

    ##############################
    ind = np.argsort(x)
    ax.semilogx(10**x[ind], y[ind], lw=2, c=[0,0,0], label='Ideal transform', zorder=10)


    ax.legend(loc='best', scatterpoints=1, numpoints=1, borderaxespad=0., handlelength=1, handletextpad=0.5, frameon=False)
    ax.set_ylabel(r'$E$')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticks([1e-9, 1e-7, 1e-5])
    ax.set_xlim([10**lims[0], 10**lims[1]])
    ax.set_xlabel(xname)


def plot_scan(alphas, objective, ax, full_height=True):
    ax.semilogx(alphas, 1-objective)
    ax.axhline(1-objective[-1], c=[0.3,0.3,0.3])
    best_ind = np.argsort(objective)[0]
    ax.scatter(alphas[best_ind], 1-objective[best_ind],s=100, c=[1,0,0])
    ax.set_xlim([np.min(alphas), np.max(alphas)])
    if full_height:
        ax.set_ylim([0,1])

    print 'Monotonic fit, R^2=%f'%( 1-objective[best_ind])
    print 'Fully smoothed monotonic fit(i.e. straight line), R^2=%f'%(1-objective[-1])


def monotonic_fit(A, x, xlims, alphas, name):
    usethis = np.isfinite(x)
    sub_x = x[usethis]
    sub_x[sub_x<xlims[0]] = xlims[0]
    sub_x[sub_x>xlims[1]] = xlims[1]
    x[usethis] = sub_x
    try:
        fit = pandas.read_csv(name + '.csv', header=0, index_col=0)
        scan = pandas.read_csv(name+'_scan.csv', header=0, index_col=0)

    except:
        energies, alphas, objective = scan_fits(A, x, xlims, alphas)
        fit = pandas.DataFrame({'x':x,'y':energies})
        fit.to_csv(name + '.csv')
        scan = pandas.DataFrame({'alphas':alphas,'objective':objective})
        scan.to_csv(name + '_scan.csv')

    x = np.array(fit['x']).flatten()
    y = np.array(fit['y']).flatten()
    alphas = np.array(scan['alphas']).flatten()
    objective = np.array(scan['objective']).flatten()
    return x, y, alphas, objective

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42

    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data_ind()
    KD = np.array(med_rep['KD'])
    random.seed(0)

    labeler = Labeler(xpad=0.07,ypad=0.02,fontsize=14)
    plt.ion()
    plt.close('all')
    figsize=(3.5,7)
    rows = 3
    cols = 1
    fig, axes = plt.subplots(rows,cols,figsize=figsize)
    plt.subplots_adjust(
        bottom = 0.07,
        top = 0.94,
        left = 0.2,
        right = 0.75,
        hspace = 0.6,
        wspace = 0.4)

    num_points = 10
    alphas = np.logspace(2,6, num_points)
    x, y, alphas, objective = monotonic_fit(A, KD, KD_lims, alphas, name='CDR_KD_spline')
    ax = axes[0]
    plot_scan(alphas, objective, ax)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'cross validated $R^2$')
    labeler.label_subplot(ax,'A')

    ax = axes[1]
    plot_transformation(x, y, KD_lims, ax, xname=r'$K_D$ [M]')
    labeler.label_subplot(ax,'B')

    ax = axes[2]
    labeler.label_subplot(ax,'C')
    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(get_spline_transformations()[0])
    num_muts = np.array(med_rep['CDR1_muts']) + np.array(med_rep['CDR3_muts'])
    KD = np.array(med_rep['KD'])
    #plot_epistasis(A, num_muts, KD, r'log $K_D$', KD_lims, '', ax, max_freq=1, logscale=True, make_cbar=True)

    wt_val = KD[num_muts==0]
    f1, x = get_f1(A, num_muts, KD, wt_val, limit=KD_lims)
    plot_epistasis(KD, f1, num_muts, KD_lims, ax, plot_ytick=True, max_freq=1, logscale=False, make_cbar=True, custom_axis= [-0.2,0.1,0.4,0.7])

    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$E_{\rm{PWM}}$')

    plt.savefig('figure_monotonic_transformation_fit.pdf')
    plt.close('all')
