#!/opt/hpc/bin/python2.7
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pdb
import pandas
from scipy.stats import pearsonr
from labeler import Labeler
from matplotlib.ticker import MaxNLocator
from data_preparation_transformed import get_f1, get_data, get_data_ind
from get_fit_PWM_transformation import get_transformations, get_spline_transformations
from figure_1 import plot_epistasis
from monotonic_fit import monotonic_fit

def plot_transformation(x, y, lims, ax, xname):
    ind = np.argsort(x)
    x = np.array(x[ind])
    y = np.array(y[ind])

    usethis = np.isfinite(x) & np.isfinite(y)
    x = x[usethis]
    y = y[usethis]

    A = np.vstack((10**x, np.ones(x.shape))).T
    myf = np.linalg.lstsq(A, y, rcond=-1)
    fx =  myf[0][0] * 10**x + myf[0][1]

    ##############################
    A = np.vstack((x, np.ones(x.shape))).T
    myf = np.linalg.lstsq(A, y, rcond=-1)
    fx =  myf[0][0] * x + myf[0][1]
    ax.plot(10**x, fx, label = r'$R^2$: %.2f'%(pearsonr(fx, y)[0]**2), lw=2, zorder=11,c=[1,0.3,0.3])

    ##############################
    ind = np.argsort(x)
    ax.semilogx(10**x[ind], y[ind], lw=2, c=[0,0,0], label='Ideal', zorder=10)

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

    print('Monotonic fit, R^2=%f'%( 1-objective[best_ind]))
    print('Fully smoothed monotonic fit(i.e. straight line), R^2=%f'%(1-objective[-1]))


if __name__ == '__main__':
    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data_ind()
    
    KD = np.array(med_rep['KD'])
    num_muts = np.array((A!=0).sum(axis=1)).flatten()
    num_points = 50
    alphas = np.logspace(-2,6, num_points)
    x, y, alphas, objective = monotonic_fit(A, KD, KD_lims, alphas, name='CDR_KD_spline', already_fit=True, random_seed=0)
    print('The correlation between optimized and linear Energy is %.3f'%pearsonr(x[num_muts==1], y[num_muts==1])[0]**2)
    labeler = Labeler(xpad=0.07,ypad=0.02,fontsize=14)
    plt.ion()
    plt.close('all')
    figsize=(7.3,7)
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows,cols,figsize=figsize)
    plt.subplots_adjust(
        bottom = 0.07,
        top = 0.94,
        left = 0.1,
        right = 0.88,
        hspace = 0.6,
        wspace = 0.1)
    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42
    for ii, rep_ind in enumerate([None, 0,1,2]):
        

        med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data_ind(replicate_use=rep_ind)
        KD = np.array(med_rep['KD'])

        num_points = 50
        alphas = np.logspace(-2,6, num_points)
        if ii==0:
            x, y, alphas, objective = monotonic_fit(A, KD, KD_lims, alphas, name='CDR_KD_spline', already_fit=True, random_seed=0)
        else:
            x, y, alphas, objective = monotonic_fit(A, KD, KD_lims, alphas, name='CDR_KD_spline_%i'%ii, already_fit=True, random_seed=0)
        ax = axes[0, ii]
        plot_scan(alphas, objective, ax)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'cross validated $R^2$')
        if ii==0:
            labeler.label_subplot(ax,'A')
            ax.set_title('Replicate average')
        else:
            ax.set_title('Replicate %i'%(ii))
            ax.set_yticks([])
            ax.set_ylabel('')
            
        ax = axes[1, ii]
        

        plot_transformation(x, y, KD_lims, ax, xname=r'$K_d$ [M]')
        if ii==0:
            labeler.label_subplot(ax,'B')
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        
        
        ax = axes[2, ii]
        if ii==0:
            labeler.label_subplot(ax,'C')
        
        PWM2logKD = get_spline_transformations()[0]
        med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(PWM2logKD, replicate_use=rep_ind)
        num_muts = np.array(med_rep['CDR1_muts']) + np.array(med_rep['CDR3_muts'])
        KD = np.array(med_rep['KD'])

        wt_val = KD[num_muts==0]
        f1, x = get_f1(A, num_muts, KD, wt_val, limit=KD_lims)
        plot_epistasis(KD, f1, num_muts, KD_lims, ax, plot_ytick=(ii==3), max_freq=4, min_freq=2, logscale=False, make_cbar=(ii==3), custom_axis= [-0.2,0.1,0.4,0.7])

        ax.set_xlabel(r'$E$')
        if ii!=0:
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel(r'$E_{\rm{PWM}}$')

    plt.savefig('figure_monotonic_transformation_fit.pdf')
    plt.close('all')
