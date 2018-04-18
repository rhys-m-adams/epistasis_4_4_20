#!/usr/bin/env python
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pandas
import pdb
import matplotlib.pyplot as plt
from helper import *
from scipy.stats import norm, levene, kstest, mannwhitneyu
from labeler import Labeler
from data_preparation_transformed import get_data, get_f1, get_null, wt_aa
from get_fit_PWM_transformation import get_transformations
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from figure_2 import plot_KD_sign_epistasis

logKD2PWM, PWM2logKD = get_transformations()
med_rep_all, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.font_manager.FontProperties(family = 'Helvetica')
mpl.rcParams['pdf.fonttype'] = 42

usethis = np.array(med_rep_all['CDR1_muts'] == 0) & np.array(med_rep_all['CDR3_muts'] == 0)
wt_val = np.array(med_rep_all['KD'])[usethis]
white_point = (PWM2logKD(wt_val)+9.5)/4.5
affinity_maturation = (-6+9.5)/4.5
cdict1 = {'red':((0.,1.,1.),
                (white_point,1.,1.),
                (affinity_maturation, 0, 0.0),
                (1.,0.25,0.25)),
            'green':((0.,0.,0.),
                (white_point,1.,1.),
                (affinity_maturation, 0, 0.0),
                (1.,0.25,0.25)),
            'blue':((0.,0.,0.),
                (white_point,1.,1.),
                (affinity_maturation, 1, 1),
                (1.,0.25,0.25))
            }
red_blue = LinearSegmentedColormap('red_blue', cdict1)


if __name__ == '__main__':
    plt.ion()
    figsize=(7.3,6)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(
        bottom = 0.04,
        top = 0.8,
        left = 0.15,
        right = 0.85,
        hspace = 0.6,
        wspace = 0.6)
    
    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=0.0,ypad=-0.04,fontsize=14)
    std1 = []
    std3 = []
    std_null = []
    seqs1 = []
    seqs3 = []
    for ii, rep_ind in enumerate([None, 0,1,2]):
        print ' '
        print ii
        med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM, replicate_use=rep_ind)
        med_rep['KD_std'] = med_rep_all['KD_std'].loc[med_rep.index]
        med_rep['KD_exclude'] = med_rep_all['KD_exclude'].loc[med_rep.index]
        usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
        usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
        A1 = A[usethis1]
        A3 = A[usethis3]
        AA1= [AA[jj] for jj in usethis1]
        AA3= [AA[jj] for jj in usethis3]
        
        usethis1 = med_rep.index[usethis1]
        usethis3 = med_rep.index[usethis3]
        KD1 = np.array((med_rep['KD'].loc[usethis1]))
        KD3 = np.array((med_rep['KD'].loc[usethis3]))
        KD1_std = np.array((med_rep['KD_std'].loc[usethis1]))
        KD3_std = np.array((med_rep['KD_std'].loc[usethis3]))
        
        if ii==0:
            CDR1_all = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'All replicates\n1H', 28, 'ALL', make_colorbar=False, epistasis='beneficial', y_offset=4)
            CDR1_ben = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'All replicates\n1H', 28, 'ALL', make_colorbar=False, epistasis='deleterious', y_offset=4)
            CDR1_all = pandas.concat([CDR1_all, CDR1_ben])
            
        else:
            CDR1_del = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Replicate %i\n1H'%ii, 28, 'ALL', make_colorbar=False, epistasis='beneficial', y_offset=4, cdf_cutoff=0.5)
            CDR1_ben = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Replicate %i\n1H'%ii, 28, 'ALL', make_colorbar=False, epistasis='deleterious', y_offset=4, cdf_cutoff=0.5)
            CDR1_del = pandas.concat([CDR1_del, CDR1_ben])
            mylist = np.intersect1d(CDR1_all.index, CDR1_del.index)
            seqs1.append(mylist)
        
        if ii==0:
            CDR3_all = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='beneficial',y_offset=0)
            CDR3_ben = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='deleterious',y_offset=0)
            CDR3_all = pandas.concat([CDR3_all, CDR3_ben])
            
        if ii!=0:
            CDR3_del = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='beneficial',y_offset=0, cdf_cutoff=0.5)
            CDR3_ben = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='deleterious',y_offset=0, cdf_cutoff=0.5)
            CDR3_del = pandas.concat([CDR3_del, CDR3_ben])
            mylist = np.intersect1d(CDR3_all.index, CDR3_del.index)
            seqs3.append(mylist)
    

    all123 = np.intersect1d(np.intersect1d(seqs1[0], seqs1[1]), seqs1[2])
    all12 = np.setdiff1d(np.intersect1d(seqs1[0], seqs1[1]), all123)
    all13 = np.setdiff1d(np.intersect1d(seqs1[0], seqs1[2]), all123)
    all23 = np.setdiff1d(np.intersect1d(seqs1[1], seqs1[2]), all123)
    all1 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[0], all12), all13), all123)
    all2 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[1], all12), all23), all123)
    all3 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[2], all13), all23), all123)
    
    all3 = np.intersect1d(np.intersect1d(seqs1[0], seqs1[1]), seqs1[2])
    all2 = np.unique(
        np.hstack((np.intersect1d(seqs1[0], seqs1[1]),
        np.intersect1d(seqs1[0], seqs1[2]),
        np.intersect1d(seqs1[1], seqs1[2]))))
    all1 = np.unique(np.hstack(seqs1))
    ax = axes[0,0]
    labeler.label_subplot(ax,'A')  
    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all1))),
    np.sqrt(len(all1)),                    # radius
    alpha=1, facecolor=[0.66,0.66,0.66], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    #ax.text(0,np.sqrt(len(all1))*2,'# single agreements: %i'%len(all1), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all2))),           # (x,y)
    np.sqrt(len(all2)),                    # radius
    alpha=1, facecolor=[0.33,0.33,0.33], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    ax.text(0,np.sqrt(len(all2))*1.5,'%i'%len(all2), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all3))),           # (x,y)
    np.sqrt(len(all3)),                    # radius
    alpha=1, facecolor=[0,0,0], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    ax.text(0,np.sqrt(len(all3))*1,'%i'%len(all3), color=[1,1,1], ha='center')

    ax.set_xlim([-7,7])
    ax.set_ylim([0,14])
    #venn3(subsets = (len(all1), len(all2), len(all3), len(all12), len(all13), len(all23), len(all123)), set_labels = ('Replicate 1', 'Replicate 2', 'Replicate 3'), ax=ax)
    ax.set_title('1H\n statistically significant\nepistasis examples: %i'%len(CDR1_all.index))
    ax.axis('off')
    ax = axes[0, 1]
    all123 = np.intersect1d(np.intersect1d(seqs3[0], seqs3[1]), seqs3[2])
    all12 = np.setdiff1d(np.intersect1d(seqs3[0], seqs3[1]), all123)
    all13 = np.setdiff1d(np.intersect1d(seqs3[0], seqs3[2]), all123)
    all23 = np.setdiff1d(np.intersect1d(seqs3[1], seqs3[2]), all123)
    all1 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[0], all12), all13), all123)
    all2 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[1], all12), all23), all123)
    all3 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[2], all13), all23), all123)
    
    all3 = np.intersect1d(np.intersect1d(seqs3[0], seqs3[1]), seqs3[2])
    all2 = np.unique(
        np.hstack((np.intersect1d(seqs3[0], seqs3[1]),
        np.intersect1d(seqs3[0], seqs3[2]),
        np.intersect1d(seqs3[1], seqs3[2]))))
    all1 = np.unique(np.hstack(seqs3))

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all1))),           # (x,y)
    np.sqrt(len(all1)),                    # radius
    alpha=1, facecolor=[0.66,0.66,0.66], edgecolor="black", linewidth=1, linestyle='solid', label='# in 1+ replicates'
    )
    )
    #ax.text(0,np.sqrt(len(all1))*2,'# single agreements: %i'%len(all1), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all2))),           # (x,y)
    np.sqrt(len(all2)),                    # radius
    alpha=1, facecolor=[0.33,0.33,0.33], edgecolor="black", linewidth=1, linestyle='solid',label='# in 2+ replicates'
    )
    )
    ax.text(0,np.sqrt(len(all2))*1.8,'%i'%len(all2), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all3))),           # (x,y)
    np.sqrt(len(all3)),                    # radius
    alpha=1, facecolor=[0,0,0], edgecolor="black", linewidth=1, linestyle='solid', label='# in 3 replicates'
    )
    )
    ax.text(0,np.sqrt(len(all3))*1.,'%i'%len(all3), color=[1,1,1], ha='center')
    ax.set_xlim([-7,7])
    ax.set_ylim([0,14])
    ax.axis('off')
    ax.legend(loc='upper left')
    #venn3(subsets = (len(all1), len(all2), len(all3), len(all12), len(all13), len(all23), len(all123)), set_labels = ('Replicate 1', 'Replicate 2', 'Replicate 3'), ax=ax)
    ax.set_title('3H\n statistically significant\nepistasis examples: %i'%len(CDR3_all.index))
    
    
    std1 = []
    std3 = []
    std_null = []
    seqs1 = []
    seqs3 = []
    for ii, rep_ind in enumerate([None, 0,1,2]):
        print ' '
        print ii
        med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM, replicate_use=rep_ind)
        med_rep['KD_std'] = med_rep_all['KD_std'].loc[med_rep.index]
        med_rep['KD_exclude'] = med_rep_all['KD_exclude'].loc[med_rep.index]
        usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
        usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
        A1 = A[usethis1]
        A3 = A[usethis3]
        AA1= [AA[jj] for jj in usethis1]
        AA3= [AA[jj] for jj in usethis3]
        
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

        #ax = axes[ii/2, ii%2]
        #labeler.label_subplot(ax,chr(ord('A')+ii))   
        if ii==0:
            CDR1_all = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'All replicates\n1H', 28, 'ALL', make_colorbar=False, epistasis='beneficial', y_offset=4)
            CDR1_ben = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'All replicates\n1H', 28, 'ALL', make_colorbar=False, epistasis='deleterious', y_offset=4)
            CDR1_all = pandas.concat([CDR1_all, CDR1_ben])
            
        else:
            CDR1_del = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Replicate %i\n1H'%ii, 28, 'ALL', make_colorbar=False, epistasis='beneficial', y_offset=4, cdf_cutoff=0.95)
            CDR1_ben = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Replicate %i\n1H'%ii, 28, 'ALL', make_colorbar=False, epistasis='deleterious', y_offset=4, cdf_cutoff=0.95)
            CDR1_del = pandas.concat([CDR1_del, CDR1_ben])
            mylist = np.intersect1d(CDR1_all.index, CDR1_del.index)
            seqs1.append(mylist)
        
        if ii==0:
            CDR3_all = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='beneficial',y_offset=0)
            CDR3_ben = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='deleterious',y_offset=0)
            CDR3_all = pandas.concat([CDR3_all, CDR3_ben])
            
        if ii!=0:
            CDR3_del = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='beneficial',y_offset=0, cdf_cutoff=0.95)
            CDR3_ben = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', make_colorbar=((ii/2)==1), epistasis='deleterious',y_offset=0, cdf_cutoff=0.95)
            CDR3_del = pandas.concat([CDR3_del, CDR3_ben])
            mylist = np.intersect1d(CDR3_all.index, CDR3_del.index)
            seqs3.append(mylist)
    

    all123 = np.intersect1d(np.intersect1d(seqs1[0], seqs1[1]), seqs1[2])
    all12 = np.setdiff1d(np.intersect1d(seqs1[0], seqs1[1]), all123)
    all13 = np.setdiff1d(np.intersect1d(seqs1[0], seqs1[2]), all123)
    all23 = np.setdiff1d(np.intersect1d(seqs1[1], seqs1[2]), all123)
    all1 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[0], all12), all13), all123)
    all2 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[1], all12), all23), all123)
    all3 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs1[2], all13), all23), all123)
    
    all3 = np.intersect1d(np.intersect1d(seqs1[0], seqs1[1]), seqs1[2])
    all2 = np.unique(
        np.hstack((np.intersect1d(seqs1[0], seqs1[1]),
        np.intersect1d(seqs1[0], seqs1[2]),
        np.intersect1d(seqs1[1], seqs1[2]))))
    all1 = np.unique(np.hstack(seqs1))
    ax = axes[1,0]
    labeler.label_subplot(ax,'B')  

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all1))),
    np.sqrt(len(all1)),                    # radius
    alpha=1, facecolor=[0.66,0.66,0.66], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    ax.text(0,np.sqrt(len(all1))*1.8,'%i'%len(all1), color=[0,0,0], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all2))),           # (x,y)
    np.sqrt(len(all2)),                    # radius
    alpha=1, facecolor=[0.33,0.33,0.33], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    ax.text(0,np.sqrt(len(all2))*1.5,'%i'%len(all2), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all3))),           # (x,y)
    np.sqrt(len(all3)),                    # radius
    alpha=1, facecolor=[0,0,0], edgecolor="black", linewidth=1, linestyle='solid'
    )
    )
    ax.text(0,np.sqrt(len(all3))*1,'%i'%len(all3), color=[1,1,1], ha='center')

    ax.set_xlim([-7,7])
    ax.set_ylim([0,14])
    #venn3(subsets = (len(all1), len(all2), len(all3), len(all12), len(all13), len(all23), len(all123)), set_labels = ('Replicate 1', 'Replicate 2', 'Replicate 3'), ax=ax)
    #ax.set_title('1H\n statistically significant\nepistasis examples: %i'%len(CDR1_all.index))
    ax.axis('off')
    ax = axes[1,1]
    all123 = np.intersect1d(np.intersect1d(seqs3[0], seqs3[1]), seqs3[2])
    all12 = np.setdiff1d(np.intersect1d(seqs3[0], seqs3[1]), all123)
    all13 = np.setdiff1d(np.intersect1d(seqs3[0], seqs3[2]), all123)
    all23 = np.setdiff1d(np.intersect1d(seqs3[1], seqs3[2]), all123)
    all1 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[0], all12), all13), all123)
    all2 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[1], all12), all23), all123)
    all3 = np.setdiff1d(np.setdiff1d(np.setdiff1d(seqs3[2], all13), all23), all123)
    
    all3 = np.intersect1d(np.intersect1d(seqs3[0], seqs3[1]), seqs3[2])
    all2 = np.unique(
        np.hstack((np.intersect1d(seqs3[0], seqs3[1]),
        np.intersect1d(seqs3[0], seqs3[2]),
        np.intersect1d(seqs3[1], seqs3[2]))))
    all1 = np.unique(np.hstack(seqs3))

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all1))),           # (x,y)
    np.sqrt(len(all1)),                    # radius
    alpha=1, facecolor=[0.66,0.66,0.66], edgecolor="black", linewidth=1, linestyle='solid', label='# significant in 1+ replicates'
    )
    )
    ax.text(0,np.sqrt(len(all1))*1.8,'%i'%len(all1), color=[0,0,0], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all2))),           # (x,y)
    np.sqrt(len(all2)),                    # radius
    alpha=1, facecolor=[0.33,0.33,0.33], edgecolor="black", linewidth=1, linestyle='solid',label='# significant in 2+ replicates'
    )
    )
    ax.text(0,np.sqrt(len(all2))*1.6,'%i'%len(all2), color=[1,1,1], ha='center')

    ax.add_patch(
    patches.Circle(
    (0, np.sqrt(len(all3))),           # (x,y)
    np.sqrt(len(all3)),                    # radius
    alpha=1, facecolor=[0,0,0], edgecolor="black", linewidth=1, linestyle='solid', label='# significant in 3 replicates'
    )
    )
    ax.text(0,np.sqrt(len(all3))*1.,'%i'%len(all3), color=[1,1,1], ha='center')
    ax.set_xlim([-7,7])
    ax.set_ylim([0,14])
    ax.axis('off')
    ax.legend(loc='upper left')
    #venn3(subsets = (len(all1), len(all2), len(all3), len(all12), len(all13), len(all23), len(all123)), set_labels = ('Replicate 1', 'Replicate 2', 'Replicate 3'), ax=ax)
    #ax.set_title('3H\n statistically significant\nepistasis examples: %i'%len(CDR3_all.index))
    
    plt.savefig('figure_replicate_epistasis.pdf')
    plt.close()
