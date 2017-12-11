#!/opt/hpc/bin/python2.7
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import numpy as np
import pdb
import time
import random
import pandas
from scipy.stats import linregress
from data_preparation_transformed import get_f1, get_data , get_data_ind
from matplotlib.ticker import MaxNLocator
from labeler import Labeler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from figure_monotonic_transformation_fit import scan_fits, plot_scan

med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data_ind()
mpl.rcParams['font.size'] = 10
mpl.font_manager.FontProperties(family = 'Helvetica')
mpl.rcParams['pdf.fonttype'] = 42

num_muts = np.array(med_rep['CDR1_muts']) + np.array(med_rep['CDR3_muts'])
random.seed(0)
np.random.seed(0)

def get_params(name, x_name, A, x, xerr, xlims, alphas):
    try:
        fit = pandas.read_csv(name + '.csv', header=0, index_col=0)
        x = np.array(fit['x']).flatten()
        y = np.array(fit['y']).flatten()
        scan = pandas.read_csv(name+'_scan.csv', header=0, index_col=0)
        alphas = np.array(scan['alphas']).flatten()
        objective = np.array(scan['objective']).flatten()
    except:
        energies, alpha, objective = scan_fits(A, x, xlims, alphas)
        fit = pandas.DataFrame({'x':x,'y':energies})
        fit.to_csv(name + '.csv')
        scan = pandas.DataFrame({'alphas':alphas,'objective':objective})
        scan.to_csv(name + '_scan.csv')
        x = np.array(fit['x']).flatten()
        y = np.array(fit['y']).flatten()

        alphas = np.array(scan['alphas']).flatten()
        objective = np.array(scan['objective']).flatten()

    fy = np.array(x)
    params = 0
    return x, y, fy, alphas, objective, params

def plot_inset(x, y1, y2, l1, l2, ax, zoom_in=False):
    actually_fit = (x>x.min()) & (x<x.max())
    slope, intercept, r_value, p_value, std_err = linregress(y1[actually_fit],y2[actually_fit])
    y1 = y1 * slope+intercept

    y1, ind = np.unique(y1, return_index=True)
    #ind = np.argsort(y)
    #y1 = y1[ind]
    x = x[ind]
    y2 = y2[ind]
    delta = np.max([1,x.shape[0]/500])
    x = x[::delta]
    y1 = y1[::delta]
    y2 = y2[::delta]
    ax.plot(x,y1, lw=2, c=[0,0,1], label=l1)
    ax.plot(x,y2, lw=2, c=[1,0,0], label=l2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(loc='lower right', numpoints=1, borderaxespad=0., handlelength=1, handletextpad=0.5, frameon=False)

    if zoom_in:
        axins = inset_axes(ax, width="40%", height="40%", loc=4)  # zoom = 6
        axins.plot(x,y1,lw=4, c=[0,0,1])
        axins.plot(x,y2,lw=4, c=[1,0,0])
        # sub region of the original image
        xmin = x[np.where(y1 > -0.2)[0][0]]
        xmax = x[np.where(y1 <  0.1)[0][-1]]
        x1, x2, y1, y2 = xmin,xmax, -0.2, 0.1
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        # fix the number of ticks on the inset axes
        #axins.yaxis.get_major_locator().set_params(nbins=7)
        #axins.xaxis.get_major_locator().set_params(nbins=7)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")


figsize=(7.3*0.7,7)
rows = 4
cols = 2
fig, axes = plt.subplots(rows,cols,figsize=figsize)
plt.subplots_adjust(
    bottom = 0.08,
    top = 0.95,
    left = 0.15,
    right = 0.96,
    hspace = 0.8,
    wspace = 0.6)

# Make a labler to add labels to subplots
labeler = Labeler(xpad=0.04,ypad=0.0,fontsize=14)
num_points = 36
alphas = np.logspace(-3, 3, num_points)

x = np.random.randn(A.shape[1])
E = A.dot(x)
epsilon = np.random.randn(E.shape[0]) * np.std(E)/2.
E += epsilon
lims = np.array([np.sort(E)[200], np.sort(E)[-200]])
rel_E = E/(lims[1]-lims[0])

def make_f(transform):
    f_lims = np.sort(transform(lims))
    f = transform(E)
    f[f<f_lims[0]] = f_lims[0]
    f[f>f_lims[1]] = f_lims[1]
    return f, f_lims


ax = axes[0,0]
f, f_lims = make_f(np.exp)
_, y, E_fit, alphas, objective, paramsE = get_params('test_exp',r'exponential', A, f, np.ones(f.shape), f_lims, alphas)
plot_scan(alphas, objective, ax, full_height=False)
ax.set_ylabel(r'$R^2$')
labeler.label_subplot(ax,'A')

ax = axes[0,1]
plot_inset(f, rel_E, y, r'$F= e^E$', 'Fit', ax)
#ax.set_xlabel('F')
ax.set_ylabel('E')

ax = axes[1,0]
f, f_lims = make_f(lambda x:x)
_, y, E_fit, alphas, objective, paramsE = get_params('test_lin',r'linear', A, f, np.ones(f.shape), f_lims, alphas)
plot_scan(alphas, objective, ax, full_height=False)
ax.set_ylabel(r'$R^2$')
labeler.label_subplot(ax,'B')
ax = axes[1,1]
plot_inset(f, rel_E, y, r'$F= E$', 'Fit', ax)
#ax.set_xlabel('F')
ax.set_ylabel('E')

ax = axes[2,0]
f, f_lims = make_f(lambda x:np.sin(2*x) + 2*x)
_, y, E_fit, alphas, objective, paramsE = get_params('test_sin',r'sin', A, f, np.ones(f.shape), f_lims, alphas)
plot_scan(alphas, objective, ax, full_height=False)
ax.set_ylabel(r'$R^2$')
labeler.label_subplot(ax,'C')
ax = axes[2,1]
plot_inset(f, rel_E, y, r'$F= 2E + $sin$ (2E)$', 'Fit', ax, zoom_in=False)
#ax.set_xlabel('F')
ax.set_ylabel('E')
ax.set_ylim([-1.2,0.9])

ax = axes[3,0]
f, f_lims = make_f(lambda x:1./(1+np.exp(-1*x)))
_, y, E_fit, alphas, objective, paramsE = get_params('test_logistic',r'logistic', A, f, np.ones(f.shape), f_lims, alphas)
plot_scan(alphas, objective, ax, full_height=False)
ax.set_ylabel(r'$R^2$')
ax.set_xlabel(r'$\alpha$')

labeler.label_subplot(ax,'D')
ax = axes[3,1]
plot_inset(f, rel_E, y, r'$F= (1+e^{-E})^{-1}$', 'Fit', ax)
ax.set_xlabel('F')
ax.set_ylabel('E')

plt.savefig('simulated_energy_fit.pdf')
plt.close()
