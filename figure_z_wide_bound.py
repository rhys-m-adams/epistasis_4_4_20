#!/usr/bin/env python
import pylab
import pdb
import matplotlib.pyplot as plt
from helper import *
from scipy.stats import kstest
from data_preparation_transformed import get_data, get_f1, get_null
from get_fit_PWM_transformation import get_transformations
from figure_2 import plot_epistasis_Z

if __name__ == '__main__':
    logKD2PWM, PWM2logKD = get_transformations()
    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=logKD2PWM, prefix='data_wide_bounds/', KD_lims=[-11, 0])#, exclude_boundary=False)

    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42

    usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
    usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
    A1 = A[usethis1]
    A3 = A[usethis3]
    
    usethis1 = med_rep.index[usethis1]
    usethis3 = med_rep.index[usethis3]
    KD1 = np.array((med_rep['KD'].loc[usethis1]))
    KD3 = np.array((med_rep['KD'].loc[usethis3]))
    KD1_std = np.array((med_rep['KD_std'].loc[usethis1]))
    KD3_std = np.array((med_rep['KD_std'].loc[usethis3]))
    num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
    num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
    
    xK, yK, xE, yE, Z, ZE = get_null(transform = logKD2PWM, exclude_boundary=False, prefix='data_wide_bounds/', KD_lims=[-11, 0])
    
    x = np.sort((Z - np.mean(Z)) / np.std(Z))
    num_muts = np.array(med_rep[['CDR1_muts','CDR3_muts']]).sum(axis=1).flatten()
    KD = np.array(med_rep['KD']).flatten()
    KD_std = np.array(med_rep['KD_std']).flatten()
    
    plot_epistasis_Z(A, num_muts, KD, KD_std, Z, '', 'All', KD_lims, None)
    
    print('kolmogorov smirnov test of normality for log KD null distribution: %f'%(kstest(x,'norm')[1]))
    
    plt.ion()
    plt.close('all')
    figsize=(7.3*0.7,2)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15,0.21,0.75, 0.7])
    plot_epistasis_Z(A1, num_muts1, KD1, KD1_std, Z, r'', '1H', KD_lims, ax, make_ytick=True, plot_null=True)
    plot_epistasis_Z(A3, num_muts3, KD3, KD3_std, Z, r'Z', '3H', KD_lims, ax, make_ytick=True)
    ax.set_yscale('symlog',linthreshy=1e-2, linscaley=0.2)
    plt.savefig('figure_Z_wide_bound.pdf')
    plt.close()
