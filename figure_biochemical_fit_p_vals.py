from figure_1 import plot_epistasis
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from data_preparation_transformed import get_data, get_f1, cdr1_list, cdr3_list
from get_fit_PWM_transformation import get_transformations, get_spline_transformations
import numpy as np
import pandas
from labeler import Labeler
from figure_3 import epi_p1,epi_p3, KD_average_epi_1, KD_average_epi_3, KD_epi_1, KD_epi_3, penalties1, Rsquare1, penalties3, Rsquare3
import pdb
logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]
A2_1 = A2[usethis1]
A2_3 = A2[usethis3]
pos1 = pos[usethis1]
pos3 = pos[usethis3]
pos1 = pos1[:,:10]
pos3 = pos3[:,10:]
AA1 = [AA[ii][:10] for ii in usethis1]
AA3 = [AA[ii][10:] for ii in usethis3]
usethis1 = med_rep.index[usethis1]
usethis3 = med_rep.index[usethis3]
KD1 = np.array((med_rep['KD'].loc[usethis1]))
KD3 = np.array((med_rep['KD'].loc[usethis3]))
E1 = np.array((med_rep['E'].loc[usethis1]))
E3 = np.array((med_rep['E'].loc[usethis3]))
num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
KD_use1 = ~np.array(med_rep['KD_exclude'].loc[usethis1])
KD_use3 = ~np.array(med_rep['KD_exclude'].loc[usethis3])

if __name__=='__main__':
    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42

    plt.ion()
    plt.close('all')
    figsize=(7.3*0.7,7)
    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows,cols,figsize=figsize)
    plt.subplots_adjust(
        bottom = 0.07,
        top = 0.95,
        left = 0.18,
        right = 0.89,
        hspace = 0.6,
        wspace = 0.6)

    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=0.09,ypad=0.01,fontsize=14)

    ax = axes[0,0]
    usethis = penalties1<1e-2

    ax.semilogx(penalties1[usethis], Rsquare1[usethis])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title('1H')
    ax.set_xticks([1e-6,1e-4,1e-2])
    ax.set_yticks(np.arange(np.round(Rsquare1[usethis].min(),2),np.round(Rsquare1[usethis].max(),2),0.01))
    ax.set_xlabel(r'$\lambda$ (penalty)')
    ax.set_ylabel(r'$R^2$')
    labeler.label_subplot(ax,'A')

    labeler = Labeler(xpad=0.09,ypad=0.0,fontsize=14)
    print '1H base R^2: %f, biochemical R^2: %f'%(Rsquare1[0], np.max(Rsquare1))


    ax = axes[0,1]
    usethis = penalties3<1e-2
    ax.semilogx(penalties3[usethis], Rsquare3[usethis])
    ax.set_xticks([1e-6,1e-4,1e-2])
    ax.set_yticks(np.arange(np.round(Rsquare3[usethis].min(),2),np.round(Rsquare3[usethis].max(),2),0.01))

    ax.set_xlabel(r'$\lambda$ (penalty)')
    ax.set_title('3H')
    print '3H base R^2: %f, biochemical R^2: %f'%(Rsquare3[0], np.max(Rsquare3))

    f = (np.array(KD1)).flatten()
    ind_zero = np.where(np.array(num_muts1==0))[0]
    wt_val = KD1[ind_zero]
    f1, x = get_f1(A1, num_muts1, KD1, wt_val, limit =KD_lims)
    f2 = f1.flatten() + KD_epi_1.flatten()
    f2[f2<KD_lims[0]] = KD_lims[0]
    f2[f2>KD_lims[1]] = KD_lims[1]
    usethis = (KD_epi_1.flatten()!=0) & np.isfinite(f2) & np.isfinite(KD1.flatten())
    finite_vals= np.isfinite(f2) & np.isfinite(KD1.flatten()) & (num_muts1>=2)
    ax = axes[1,0]
    labeler.label_subplot(ax,'B')
    print '# CDR1H muts affected: %i'%(usethis.sum())
    plot_epistasis(KD1[usethis], f1[usethis], num_muts1[usethis], KD_lims, ax, make_cbar=False, plot_ytick=False, max_freq=1)
    ax.set_ylabel(r'PWM [M]')
    
    ax = axes[2,0]
    labeler.label_subplot(ax,'C')
    plot_epistasis(KD1[usethis], f2[usethis], num_muts1[usethis], KD_lims, ax, make_cbar=False, plot_ytick=False, max_freq=1)
    ax.set_xlabel(r'$K_d$ [M]')
    ax.set_ylabel(r'pairwise [M]')


    f = (np.array(KD3)).flatten()
    ind_zero = np.where(np.array(num_muts3==0))[0]
    wt_val = KD3[ind_zero]
    f1, x = get_f1(A3, num_muts3, KD3, wt_val, limit =KD_lims)
    f2 = f1.flatten() + KD_epi_3.flatten()
    f2[f2<KD_lims[0]] = KD_lims[0]
    f2[f2>KD_lims[1]] = KD_lims[1]
    usethis = (KD_epi_3.flatten()!=0) & np.isfinite(f2) & np.isfinite(KD3.flatten())
    finite_vals= np.isfinite(f2) & np.isfinite(KD3.flatten()) & (num_muts3>=2)
    print '# CDR3H muts affected: %i'%((usethis).sum())
    ax = axes[1,1]
    plot_epistasis(KD3[usethis], f1[usethis], num_muts3[usethis], KD_lims, ax, make_cbar=True, plot_ytick=False, max_freq=1)
    
    ax = axes[2,1]
    plot_epistasis(KD3[usethis], f2[usethis], num_muts3[usethis], KD_lims, ax, make_cbar=True, plot_ytick=False, max_freq=1)
    ax.set_xlabel(r'$K_d$ [M]')

    labeler = Labeler(xpad=0.08,ypad=-0.0, fontsize=14)

    ax = axes[3,0]
    labeler.label_subplot(ax,'D')

    usethis = KD_average_epi_1.flatten()!=0
    p1 = epi_p1.flatten()[usethis]
    p1[p1<1e-30] = 1e-30
    ax.plot(range(p1.shape[0]), np.log10(np.sort(p1)))
    #ax.axhline(np.log10(np.sort(p1))[-1],c=[0.3,0.3,0.3])

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ticks = [-5,-4,-3,-2,-1,0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([r'$10^{%i}$'%power for power in ticks])
    ax.set_ylim([-5,0])
    ax.set_xlim([0,p1.shape[0]-1])
    ax.set_xlabel('parameter')
    ax.set_ylabel('posterior')
    #ax.set_title('1H')
    ax.set_aspect((p1.shape[0]-1)/5.)

    ax = axes[3,1]
    usethis = KD_average_epi_3.flatten()!=0
    p3 = epi_p3.flatten()[usethis]
    p3[p3<1e-30] = 1e-30

    ax.plot(range(p3.shape[0]), np.log10(np.sort(p3)))
    #ax.axhline(np.log10(np.sort(p3))[-1],c=[0.3,0.3,0.3])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_yticks([])
    ax.set_yticklabels([r'$10^{%i}$'%power for power in ticks])
    ax.set_ylim([-5,0])
    ax.set_xlim([0,p3.shape[0]-1])
    ax.set_aspect((p3.shape[0]-1)/5.)
    ax.set_xlabel('parameter')
    #ax.set_title('3H')

    plt.savefig('figure_biochemical_fit_pvals.pdf')
    plt.close()
