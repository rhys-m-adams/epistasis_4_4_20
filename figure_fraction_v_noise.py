#!/usr/bin/env python
import pylab
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import pandas
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from helper import *
from scipy.stats import norm, levene, kstest, mannwhitneyu, skew
from labeler import Labeler
from data_preparation_transformed import get_data, get_f1, get_null, cdr1_list, cdr3_list, wt_aa
from get_fit_PWM_transformation import get_transformations, get_spline_transformations
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as PathEffects
from scipy.stats import linregress
#from scipy.interpolate import interp1d
logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=logKD2PWM)


pos_cutoff = 2

logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.font_manager.FontProperties(family = 'Helvetica')
mpl.rcParams['pdf.fonttype'] = 42

usethis = np.array(med_rep['CDR1_muts'] == 0) & np.array(med_rep['CDR3_muts'] == 0)


if __name__ == '__main__':
    
    figsize=(3.5,3.5)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    x = np.array(med_rep['fraction']).flatten()
    y = np.array(med_rep['KD_std']).flatten()
    usethis = np.isfinite(x) & np.isfinite(y) & ~np.array(med_rep['KD_exclude']).flatten() & (y!=0)
    
    myf = linregress(np.log10(x[usethis]), np.log10(y[usethis]))
    usethis = np.isfinite(x) & np.isfinite(y) & ~np.array(med_rep['KD_exclude']).flatten()
    print 'average standard deviation: %f'%(np.mean(y[usethis]))
    freq, xpos, ypos = np.histogram2d(x[usethis],y[usethis], bins=(np.logspace(-7,0,16),np.linspace(0,4.5,20)))
    cax = ax.pcolor(np.logspace(-7,0,15), np.linspace(0,4.5,19),freq.T+1e-1, norm = LogNorm(), vmin=0.5, vmax=1e3, cmap='bone_r')
    x = np.logspace(-7,0,1000)
    plt.colorbar(cax, label='# of sequences')    
    ax.plot(x, 10**(myf.intercept + myf.slope * np.log10(x)), label='fit')
    ax.set_xscale('log')
    ax.set_xlabel('percent of population')
    ax.set_ylabel(r'standard error of the mean (log$_{10}K_d$)')
    ax.set_ylim([0,4.5])
    plt.legend()
    plt.savefig('figure_fraction_std.pdf')
    plt.close()
    print myf

