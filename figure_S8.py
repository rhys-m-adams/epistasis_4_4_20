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
from scipy.optimize import minimize

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
    clones = pandas.read_table('./data/titration_curves.csv',sep=',', header=0)
    rep1 = pandas.read_csv('data/replicate_1.csv')
    rep2 = pandas.read_csv('data/replicate_2.csv')
    rep3 = pandas.read_csv('data/replicate_3.csv')
    reps = [rep1, rep2, rep3]

    usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
    usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
    A1 = A[usethis1]
    A3 = A[usethis3]
    AA1= [AA[ii] for ii in usethis1]
    AA3= [AA[ii] for ii in usethis3]
    clone_keys = [(k1+k3).replace(' ','') for k1, k3 in zip(clones[' CDR1HAA'],clones[' CDR3HAA'])]
    clones.index = clone_keys
    Z = []
    Z_centered = []
    res = []
    
    for k in set(clone_keys):
        KDs = []
        for rep in reps:
            rep['AA'] = [aa1+aa3 for aa1,aa3 in zip(rep['CDR1H_AA'].tolist() ,rep['CDR3H_AA'].tolist())]
            KDs.extend(rep['fit_KD'].loc[rep['AA'].isin([k])])
        
        KDs = np.array(KDs)
        KDs = np.log10(KDs[np.isfinite(KDs)])
        if not med_rep['KD_exclude'].loc[k]:
            clone_KD = np.log10(clones[' KD'].loc[k])
            Y = clone_KD.mean()
            X = np.mean(KDs)
            res.append(X-Y)
    
    standard_deviations = []    
    for k in set(clone_keys):
        KDs = []
        for rep in reps:
            rep['AA'] = [aa1+aa3 for aa1,aa3 in zip(rep['CDR1H_AA'].tolist() ,rep['CDR3H_AA'].tolist())]
            KDs.extend(rep['fit_KD'].loc[rep['AA'].isin([k])])
        
        KDs = np.array(KDs)
        KDs = np.log10(KDs[np.isfinite(KDs)])
        if not med_rep['KD_exclude'].loc[k]:
            clone_KD = np.log10(clones[' KD'].loc[k])
            Y = clone_KD.mean()
            SY = clone_KD.std(ddof=1)/np.sqrt(clone_KD.shape[0])
            X = np.mean(KDs)
            SX = KDs.std(ddof=1)/np.sqrt(KDs.shape[0])
            standard_deviations.append(SX)
            Z.append((X-Y)/np.sqrt(SX+SY))
            Z_centered.append((X-Y - np.mean(res))/np.sqrt(SX+SY))

    print('residual mean: %f'%(np.mean(res)))
    print('Clonal Z std:%f, kolmogorov test for normality: %f'%(np.std(Z), kstest(Z,'norm')[1]))
    print('Average std:%f'%(np.mean(standard_deviations)))
    plt.ion()
    plt.close('all')

    figsize=(3.5,3.5)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.2,0.2,0.7,0.7])

    Z = np.sort(np.array(Z))
    freq, x = np.histogram(Z, bins = np.linspace(-4.5,4.5,20))
    dx = x[1]-x[0]
    x = (x[:-1]+x[1:])/2.
    ax.bar(x,freq, width=0.4, label='observed')
    
    x = np.linspace(-3,3,101)
    ax.plot(x, norm(0,1).pdf(x)*dx*freq.sum(), c='r', label='expected')
    
    ax.text(0.05,0.9,'KS-test: %.2f'%(kstest(Z,'norm')[1]),transform=ax.transAxes )
    ax.set_ylim([0,4])
    ax.set_yticks(list(range(5)))
    ax.set_yticklabels(list(range(5)))
    plt.legend(loc='upper right')
    ax.set_xlabel('clonal Z')
    ax.set_ylabel('frequency')
    
    plt.savefig('figure_S8.pdf')
    plt.close()

