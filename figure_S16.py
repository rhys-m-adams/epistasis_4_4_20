#!/usr/bin/env python
import pylab
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import pandas
from helper import *
from scipy.stats import norm
from labeler import Labeler
from sklearn.linear_model import lasso_path, Lasso
import os
from matplotlib import gridspec
from data_preparation_transformed import get_data, get_f1, cdr1_list, cdr3_list
from get_fit_PWM_transformation import get_transformations, get_spline_transformations
import itertools
import errno
import os
from model_plot import fit_ave_epistasis_by_pos, plot_connections, summary_plot, make_linear_epistasis_model, get_stats

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

logKD2PWM, PWM2logKD = get_spline_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
mpl.font_manager.FontProperties(family = 'Helvetica')

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

print('# of triple mutants, CDR1: %i, CDR3:%i'%(np.sum(num_muts1==3),np.sum(num_muts3==3)))
print('# of triple mutants within boundary, CDR1: %i, CDR3:%i'%(np.sum(KD_use1[num_muts1==3]),np.sum(KD_use3[num_muts3==3])))
mkdir('./biochemical_fit_optimized')

if __name__ == '__main__':
    try:
        KD_epi_1 = np.array(pandas.read_csv('./biochemical_fit_optimized/KD_epi_1.csv',index_col=0))
        KD_average_epi_1= np.array(pandas.read_csv('./biochemical_fit_optimized/KD_average_epi_1.csv',index_col=0))
        Rsquare1= np.array(pandas.read_csv('./biochemical_fit_optimized/Rsquare1.csv',index_col=0))
        penalties1= np.array(pandas.read_csv('./biochemical_fit_optimized/penalties1.csv',index_col=0))
        epi_mu1= np.array(pandas.read_csv('./biochemical_fit_optimized/epi_mu1.csv',index_col=0))
        epi_sigma1= np.array(pandas.read_csv('./biochemical_fit_optimized/epi_sigma1.csv',index_col=0))
        epi_p1 = np.array(pandas.read_csv('./biochemical_fit_optimized/epi_p1.csv',index_col=0))

    except:
        KD_epi_1, KD_average_epi_1, Rsquare1, penalties1, epi_mu1, epi_sigma1, epi_p1 = fit_ave_epistasis_by_pos(A1, num_muts1, KD1, AA1, cdr1_list, KD_lims)
        pandas.DataFrame(KD_epi_1).to_csv('./biochemical_fit_optimized/KD_epi_1.csv')
        pandas.DataFrame(KD_average_epi_1).to_csv('./biochemical_fit_optimized/KD_average_epi_1.csv')
        pandas.DataFrame(Rsquare1).to_csv('./biochemical_fit_optimized/Rsquare1.csv')
        pandas.DataFrame(penalties1).to_csv('./biochemical_fit_optimized/penalties1.csv')
        pandas.DataFrame(epi_mu1).to_csv('./biochemical_fit_optimized/epi_mu1.csv')
        pandas.DataFrame(epi_sigma1).to_csv('./biochemical_fit_optimized/epi_sigma1.csv')
        pandas.DataFrame(epi_p1).to_csv('./biochemical_fit_optimized/epi_p1.csv')

    try:
        KD_epi_3 = np.array(pandas.read_csv('./biochemical_fit_optimized/KD_epi_3.csv',index_col=0))
        KD_average_epi_3= np.array(pandas.read_csv('./biochemical_fit_optimized/KD_average_epi_3.csv',index_col=0))
        Rsquare3= np.array(pandas.read_csv('./biochemical_fit_optimized/Rsquare3.csv',index_col=0))
        penalties3= np.array(pandas.read_csv('./biochemical_fit_optimized/penalties3.csv',index_col=0))
        epi_mu3= np.array(pandas.read_csv('./biochemical_fit_optimized/epi_mu3.csv',index_col=0))
        epi_sigma3= np.array(pandas.read_csv('./biochemical_fit_optimized/epi_sigma3.csv',index_col=0))
        epi_p3 = np.array(pandas.read_csv('./biochemical_fit_optimized/epi_p3.csv',index_col=0))

    except:
        KD_epi_3, KD_average_epi_3, Rsquare3, penalties3, epi_mu3, epi_sigma3, epi_p3 = fit_ave_epistasis_by_pos(A3, num_muts3, KD3, AA3, cdr3_list, KD_lims)
        pandas.DataFrame(KD_epi_3).to_csv('./biochemical_fit_optimized/KD_epi_3.csv')
        pandas.DataFrame(KD_average_epi_3).to_csv('./biochemical_fit_optimized/KD_average_epi_3.csv')
        pandas.DataFrame(Rsquare3).to_csv('./biochemical_fit_optimized/Rsquare3.csv')
        pandas.DataFrame(penalties3).to_csv('./biochemical_fit_optimized/penalties3.csv')
        pandas.DataFrame(epi_mu3).to_csv('./biochemical_fit_optimized/epi_mu3.csv')
        pandas.DataFrame(epi_sigma3).to_csv('./biochemical_fit_optimized/epi_sigma3.csv')
        pandas.DataFrame(epi_p3).to_csv('./biochemical_fit_optimized/epi_p3.csv')

    print('# of found epistatic terms from biochemical model, CDR1H: %i, CDR3H: %i'%(np.sum(KD_average_epi_1!=0),np.sum(KD_average_epi_3!=0)))
    A_opt1 = make_linear_epistasis_model(['TFGHYWMNWV'], cdr1_list)
    A_opt3 = make_linear_epistasis_model(['GASYGMEYLG'], cdr3_list)
    epi_contributions1 = (A_opt1 * (KD_average_epi_1.flatten())).flatten()
    epi_contributions3 = (A_opt3 * (KD_average_epi_3.flatten())).flatten()
    usethis1 = np.where(epi_contributions1)[0]
    usethis3 = np.where(epi_contributions3)[0]
    print('Epistatic contribution to OPT CDR1H domain: '+ str(epi_contributions1[usethis1]))
    print('Epistatic contribution to OPT CDR3H domain: '+ str(epi_contributions3[usethis3]))
    print('Epistatic contribution to OPT CDR1H domain: '+ str(np.sum(epi_contributions1[usethis1])))
    print('Epistatic contribution to OPT CDR3H domain: '+ str(np.sum(epi_contributions3[usethis3])))

    print('sum |epistatic contribution| CDR1: %f'%(np.sum(np.abs(KD_average_epi_1))))
    print('sum |epistatic contribution| CDR3: %f'%(np.sum(np.abs(KD_average_epi_3))))
    plt.ion()
    plt.close('all')
    figsize=(7.3*0.7,4)

    fig, axes = plt.subplots(figsize=figsize)
    num_x = 20
    gs = gridspec.GridSpec(11, num_x)
    plt.subplots_adjust(
        bottom = 0.05,
        top = 0.93,
        left = 0.01,
        right = 0.95,
        hspace = 0.4,
        wspace = 1.5)


    def get_sign_model(x, sign):
        out = np.array(x)
        out[(out*sign)>0] = 0
        return out


    ax = plt.subplot(gs[0:5,0:int(num_x/2)])
    CDR1_pos_connections = plot_connections(get_sign_model(KD_average_epi_1,1) * (epi_p1<(5e-2)), cdr1_list, 28,0., ax,0)
    ax.axis('off')
    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=-0.03,ypad=0.01,fontsize=14)
    labeler.label_subplot(ax,'A')
    ax.set_title('beneficial')
    CDR3_pos_connections = plot_connections(get_sign_model(KD_average_epi_3,1) * (epi_p3<(5e-2)), cdr3_list, 100,0., ax,2.5)
    ax.set_xlim([-1.3,3.8])
    ax.axis('off')

    ax = plt.subplot(gs[6:,0:int(num_x/2)])
    CDR1_neg_connections = plot_connections(get_sign_model(KD_average_epi_1,-1) * (epi_p1<(5e-2)), cdr1_list, 28,0., ax,0)
    ax.axis('off')
    labeler.label_subplot(ax,'B')
    ax.set_title('deleterious')

    CDR3_neg_connections = plot_connections(get_sign_model(KD_average_epi_3,-1) * (epi_p3<(5e-2)), cdr3_list, 100,0., ax,2.5, label_on=True)
    ax.set_xlim([-1.3,3.8])
    ax.axis('off')

    #ax = plt.subplot(gs[6,:])
    #plot_connections(KD_average_epi_3, cdr3_list, 100,0.0, ax, 0, visible=False)
    #ax.axis('off')
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.63,-0.19), frameon=False, columnspacing=1, handlelength=0.6, ncol=2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4)
    #ax.axis('off')
    
    ax3 = plt.subplot(gs[1:4,int(num_x*3/5+1):])
    ax4 = plt.subplot(gs[6:9,int(num_x*3/5+1):])
    cdr1_results = pandas.read_table('cdr1h_optimized.csv', sep=',', header=0)
    print('CDR1H')
    cdr1_results = cdr1_results.apply(pandas.to_numeric)
    vol_0, surf_0 = get_stats(cdr1_results.dropna())
    summary_plot(surf_0, vol_0, ax3, ax4, colors=['#000080','#8080FF'], title ='1H', ylabels=True)

    labeler = Labeler(xpad=0.09,ypad=0.01, fontsize=14)

    plt.subplot(gs[0,int(num_x*3/5+1):]).set_visible(False)
    labeler.label_subplot(plt.subplot(gs[0,int(num_x*3/5+1):]),'C')
    labeler.label_subplot(ax4,'D')

    cdr3_results = pandas.read_table('cdr3h_optimized.csv', sep=',', header=0)
    print('CDR3H')
    cdr3_results = cdr3_results.apply(pandas.to_numeric)

    vol_0, surf_0 = get_stats(cdr3_results.dropna())
    summary_plot(surf_0, vol_0, ax3, ax4, colors=['#800000','#FF8080'], title ='3H', make_colorbar=False, ylabels=True)
    leg = ax3.legend(loc='center', bbox_to_anchor=(0.5, 1.3),ncol=2, columnspacing=0.1, frameon=True, fancybox=True, scatterpoints=1, borderaxespad=0, handlelength=1., handletextpad=0.5)
    
    plt.savefig('figure_S16.pdf')
    plt.close()
