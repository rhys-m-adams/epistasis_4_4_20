#!/usr/bin/env python
import matplotlib as mpl
import pandas
import pdb
import matplotlib.pyplot as plt
from helper import *
from scipy.stats import kstest
from labeler import Labeler
from data_preparation_transformed import get_data, get_f1, get_null, wt_aa
from get_fit_PWM_transformation import get_spline_transformations
from matplotlib import gridspec
from figure_2 import plot_Z_epistasis_by_pos, plot_epistasis_Z, plot_KD_sign_epistasis, calculate_Z_epistasis_by_pos

logKD2PWM, PWM2logKD = get_spline_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.font_manager.FontProperties(family = 'Helvetica')
mpl.rcParams['pdf.fonttype'] = 42

usethis = np.array(med_rep['CDR1_muts'] == 0) & np.array(med_rep['CDR3_muts'] == 0)
wt_val = np.array(med_rep['KD'])[usethis]


if __name__ == '__main__':
    usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
    usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
    A1 = A[usethis1]
    A3 = A[usethis3]
    AA1= [AA[ii] for ii in usethis1]
    AA3= [AA[ii] for ii in usethis3]

    pos1 = pos[usethis1]
    pos3 = pos[usethis3]
    pos1 = pos1[:,:10]
    pos3 = pos3[:,10:]
    usethis1 = med_rep.index[usethis1]
    usethis3 = med_rep.index[usethis3]
    KD1 = np.array((med_rep['KD'].loc[usethis1]))
    KD3 = np.array((med_rep['KD'].loc[usethis3]))
    KD1_std = np.array((med_rep['KD_std'].loc[usethis1]))
    KD3_std = np.array((med_rep['KD_std'].loc[usethis3]))
    num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
    num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
    KD_use1 = ~np.array(med_rep['KD_exclude'].loc[usethis1])
    KD_use3 = ~np.array(med_rep['KD_exclude'].loc[usethis3])

    xK, yK, xE, yE, Z, ZE = get_null(transform = logKD2PWM)
    #Z = np.hstack((Z,-Z))
    x = np.sort((Z-np.mean(Z))/np.std(Z))
    num_muts = np.array(med_rep[['CDR1_muts','CDR3_muts']]).sum(axis=1).flatten()
    KD = np.array(med_rep['KD']).flatten()
    KD_std = np.array(med_rep['KD_std']).flatten()
    KD_use = ~np.array(med_rep['KD_exclude'])

    plot_epistasis_Z(A[KD_use], num_muts[KD_use], KD[KD_use], KD_std[KD_use], Z, r'', 'All', KD_lims, None)

    opt1 = [2, 3]
    opt3 = [1, 2, 6, 8]

    Z_by_pos1 = calculate_Z_epistasis_by_pos(A1, num_muts1, KD1, KD1_std, pos1, KD_use1, KD_lims)
    Z_by_pos3 = calculate_Z_epistasis_by_pos(A3, num_muts3, KD3, KD3_std, pos3, KD_use3, KD_lims)


    print('kolmogorov smirnov test of normality for log KD null distribution: %f'%(kstest(x,'norm')[1]))

    plt.ion()
    plt.close('all')

    figsize=(7.3*0.7,4.8)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(figsize=figsize)
    gs = gridspec.GridSpec(16, 41)
    plt.subplots_adjust(
        bottom = 0.09,
        top = 0.95,
        left = 0.11,
        right = 0.99,
        hspace = 4,
        wspace = 2)

    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=0.04,ypad=0.0,fontsize=14)

    ax = plt.subplot(gs[0:4,0:22])
    plot_epistasis_Z(A1[KD_use1], num_muts1[KD_use1], KD1[KD_use1], KD1_std[KD_use1], Z, r'', '1H', KD_lims, ax, make_ytick=True, plot_null=True)
    labeler.label_subplot(ax,'A')
    #ax.set_xticks([])
    #ax.set_yscale('symlog',linthreshy=3e-2)
    #ax = axes[0]
    plot_epistasis_Z(A3[KD_use3], num_muts3[KD_use3], KD3[KD_use3], KD3_std[KD_use3], Z, r'Z', '3H', KD_lims, ax, make_ytick=True)
    ax.set_yscale('symlog',linthreshy=1e-2, linscaley=0.2)
    #labeler.label_subplot(ax,'B')

    ax = plt.subplot(gs[5:11,0:8])
    plot_Z_epistasis_by_pos(Z_by_pos1, 20, 28, 3, ax, curr_title = '1H', opt=opt1, make_ylabel=True)
    labeler.label_subplot(ax,'B')

    ax = plt.subplot(gs[5:11,12:20])
    plot_Z_epistasis_by_pos(Z_by_pos3, 20, 100, 3, ax, curr_title = '3H', opt=opt3, make_colorbar=True)

    ax = plt.subplot(gs[0:15,26:])
    labeler = Labeler(xpad=-0.01,ypad=-0.0, fontsize=14)
    labeler.label_subplot(ax,'C')
    CDR1_del = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'ALL', ax=ax, make_colorbar=False, epistasis='beneficial', y_offset=4, PWM2logKD=PWM2logKD)
    CDR3_del = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', ax=ax, make_colorbar=True, epistasis='beneficial', PWM2logKD=PWM2logKD)
    CDR1_pos = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'ALL', make_colorbar=False, epistasis='deleterious', y_offset=4, PWM2logKD=PWM2logKD)
    CDR3_pos = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=True, epistasis='deleterious', PWM2logKD=PWM2logKD)
    print('beneficial epistasis, 1H:%i, 3H:%i'%(CDR1_del.shape[0],CDR3_del.shape[0]))
    print('deleterious epistasis, 1H:%i, 3H:%i'%(CDR1_pos.shape[0],CDR3_pos.shape[0]))
    print('total: %i'%(CDR1_del.shape[0]+CDR3_del.shape[0]+CDR1_pos.shape[0]+CDR3_pos.shape[0]))

    plt.savefig('figure_2_optimized.pdf')
    plt.close()
