#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import pandas
from helper import *
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import gridspec
from labeler import Labeler
from data_preparation_transformed import get_f1, get_data
from get_fit_PWM_transformation import get_transformations
import matplotlib.patheffects as PathEffects
import svgutils.transform as sg

def plot_blosum(f, f1, num_muts, lims, ylims, ax, curr_title='', make_cbar=False, plot_ytick=False, plot_xtick=True, min_freq=0, max_freq=2, logscale=False, custom_axis=[]):
    usethis = np.isfinite(f1) & np.isfinite(f)#& (num_muts>1) #& (f>lims[0]) & (f<lims[1]) & (f1>lims[0]) & (f1<lims[1])
    nbins = 20
    H, xedges, yedges = np.histogram2d(
        f[usethis], f1[usethis],
        bins=[np.linspace(lims[0], lims[1], nbins+1),
             np.linspace(ylims[0], ylims[1], nbins+1)])
    plt.sca(ax)

    xedges = np.linspace(lims[0], lims[1], nbins)
    yedges = np.linspace(ylims[0], ylims[1], nbins)
    [xx, yy] = np.meshgrid(xedges, yedges)

    lvls = np.logspace(start=0,stop=max_freq,num=(max_freq)*2 + 1, endpoint=True)
    lvls2 = np.logspace(start=0,stop=max_freq,num=max_freq+1, endpoint=True)
    H[H>lvls.max()] = lvls.max()
    f_inds = np.searchsorted(np.linspace(lims[0]-1e-10, lims[1]+1e-10, nbins+1), f[usethis])
    f1_inds = np.searchsorted(np.linspace(ylims[0]-1e-10, ylims[1]+1e-10, nbins+1), f1[usethis])
    color = H[f_inds-1, f1_inds-1]
    density_sort = np.argsort(color)
    color[color>10**max_freq] = 10**max_freq
    cax = ax.scatter(f[usethis][density_sort], f1[usethis][density_sort], c=color[density_sort],cmap='jet',  norm = LogNorm(vmin=10**min_freq, vmax=10**max_freq + 1e-2))

    if logscale:
        in_interval = lambda x: (x >= lims[0]) and (x <= lims[1])
        xticks = [ii for ii in range(int(np.floor(lims[0]))-1, int(np.ceil(lims[1]))+1) if ((ii%2)==1) and in_interval(ii)]
        ax.set_xticks(xticks)
        #yticks = [ii for ii in range(int(np.floor(ylims[0]))-1, int(np.ceil(ylims[1]))+1) if ((ii%2)==1) and in_interval(ii)]
        #ax.set_yticks(yticks)
        format_num = lambda x:int(np.round(x))
        for x in xticks:
            if np.round(x) != x:
                format_num = lambda x:np.round(x,1)

        xticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in xticks]
        ax.set_xticklabels(xticklabels)
        if plot_ytick:
            yticks = ax.get_yticks()
            yticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in yticks]
            #ax.set_yticklabels(yticklabels)
            ax.tick_params(axis='y', which='major', pad=2)

        else:
            ax.set_yticklabels([])

        if not plot_xtick:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    elif len(custom_axis)>0:
        ax.set_xticks(custom_axis)
        ax.set_yticks(custom_axis)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(4))

        if plot_ytick:
            ax.yaxis.set_major_locator(MaxNLocator(4))

        else:
            ax.set_yticklabels([])
    
    lim_range = lims[1] - lims[0]
    Rsquare = pearsonr(f[usethis],f1[usethis])[0]**2
    if Rsquare<1e-3:
        txt = ax.text(0.11,0.02, r'$R^2<0.001$', transform=ax.transAxes)
    else:
        txt = ax.text(0.11,0.02, r'$R^2=$' + str(np.round(Rsquare,3)), transform=ax.transAxes)
    #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=1)])
    plt.savefig('deleteme.pdf')
    cbar = []
    if make_cbar:

        p3 = np.array(ax.get_position().get_points())
        position = ([p3[1][0]+0.005, p3[0][1], 0.015, p3[1][1]-p3[0][1]])

        cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical')
        cbar.set_label('density',labelpad=0)
        xtl = []
        for ii in range(0, len(lvls) - 1):
            if (ii%2)==0:
                xtl.append(r'$10^{%i}$'%(int(ii)/2))
            else:
                xtl.append('')
        
        xtl.append(r'$\geq 10^{%i}$'%(max_freq))

        def cleanup(x):
            if x in lvls2:
                return r'$10^{%i}$'%(np.log10(x))
            return r'$10^{\frac{%i}{2}}$'%np.round(np.log10(x)*2)

        #level_ticks = [cleanup(level) for level in lvls]
        #level_ticks[-1] = r'$\geq$' + level_ticks[-1]
        #cbar.set_ticklabels(xtl)

    return cbar

if __name__ == '__main__':
    logKD2PWM, PWM2logKD = get_transformations() #choose log transformation
    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM) #get data
    #separate data into CDR1H, CDR3H
    usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
    usethis3 = np.where(med_rep['CDR1_muts']==0)[0]

    A1 = A[usethis1]
    A3 = A[usethis3]

    usethis1 = med_rep.index[usethis1]
    usethis3 = med_rep.index[usethis3]

    KD1 = np.array((med_rep['KD'].loc[usethis1]))
    KD3 = np.array((med_rep['KD'].loc[usethis3]))

    E1 = np.array((med_rep['E'].loc[usethis1]))
    E3 = np.array((med_rep['E'].loc[usethis3]))

    num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
    num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
    E = (np.array(10**(med_rep['E'])))
    Es = np.sort(E)
    print('percent expression decrease first quartile: %f'%np.nanpercentile(Es, 25))
    print('percent expression decrease third quartile: %f'%np.nanpercentile(Es, 50))
    print('percent expression decrease third quartile: %f'%np.nanpercentile(Es, 75))

    cdr1_wtseq = 'TFSDYWMNWV'
    cdr3_wtseq = 'GSYYGMDYWG'
    cdr1_optseq = 'TFGHYWMNWV'
    cdr3_optseq = 'GASYGMEYLG'

    wtseq = cdr1_wtseq + cdr3_wtseq
    usethis = np.array(med_rep['CDR1_muts'] == 0) & np.array(med_rep['CDR3_muts'] == 0)
    wt_val = np.array(med_rep['KD'])[usethis]

    # Create figure with subplots and specified spacing
    figsize=(7.3*0.7,3.2)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(2,2,figsize=figsize)
    
    plt.subplots_adjust(
        bottom = 0.15,
        top = 0.92,
        left = 0.05,
        right = 0.85,
        hspace = 0.8,
        wspace = 0.6)

    # Make a labeler to add labels to subplots
    labeler = Labeler(xpad=.015,ypad=.007,fontsize=14)
    
    f1, x = get_f1(A1, num_muts1, KD1, wt_val, limit=KD_lims)
    usethis = np.isfinite(E1) & np.isfinite(KD1)
    myf = stats.pearsonr(np.exp(E1[usethis]), KD1[usethis])
    print('CDR1H spearman correlation E to F:%f, p-value: %e'%(myf[0], myf[1]))

    myf = stats.linregress(np.exp(E1[usethis]), KD1[usethis])    
    residual = myf[0] * np.exp(E1[usethis]) + myf[1] - KD1[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum((KD1[usethis] - np.mean(KD1[usethis]))**2)
    print('CDR1H R^2, m*E + b = F: %f, p-value: %e'%(Rsquare, myf[3]))
    
    usethis = np.isfinite(E1) & np.isfinite(KD1-f1) & (num_muts1==1)
    myf = stats.linregress(np.exp(E1[usethis]), KD1[usethis])    
    residual = myf[0] * np.exp(E1[usethis]) + myf[1] - KD1[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum((KD1[usethis] - np.mean(KD1[usethis]))**2)
    print('CDR1H 1 mut R^2, m*E + b = F: %f, p-value: %e'%(Rsquare, myf[3]))
    
    usethis = np.isfinite(E1) & np.isfinite(KD1-f1) & (num_muts1>1)
    myf = stats.linregress(np.exp(E1[usethis]), KD1[usethis]-f1[usethis])
    residual = myf[0] * np.exp(E1[usethis]) + myf[1] - (KD1 - f1)[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum(((KD1-f1)[usethis] - np.mean((KD1-f1)[usethis]))**2)
    print('CDR1H R^2 m*E + b = (F - F1): %f, m p-value: %f'%(Rsquare, myf[3]))
    ax = axes[0,0]
    ax.set_ylabel('PWM [M]', labelpad=2)
    ax.set_title('1H')
    
    ax = axes[0,0]
    cbar = plot_blosum(KD1, E1, num_muts1, KD_lims, [-1,0.5], ax)
        
    labeler = Labeler(xpad=.12,ypad=.007,fontsize=14)
    ax.set_xlabel(r'log $_{10} K_d$')
    ax.set_ylabel('Expression', labelpad=2)
    
    ax = axes[1,0]
    cbar = plot_blosum(KD1-f1, E1, num_muts1, [-1,3.5], [-1,0.5], ax)
    ax.set_xlabel(r'log $_{10} K_d - $ log$_{10} PWM$')
    ax.set_ylabel('Expression', labelpad=2)
    f1, x = get_f1(A3, num_muts3, KD3, wt_val, limit=KD_lims)
    usethis = np.isfinite(E3) & np.isfinite(KD3)
    myf = stats.pearsonr(np.exp(E3[usethis]), KD3[usethis])
    print('CDR3H spearman correlation E to F:%f, p-value: %e'%(myf[0], myf[1]))
    
    myf = stats.linregress(np.exp(E3[usethis]), KD3[usethis])
    residual = myf[0] * np.exp(E3[usethis]) + myf[1] - KD3[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum((KD3[usethis] - np.mean(KD3[usethis]))**2)
    print('CDR3H R^2, m*E + b = F: %f, p-value: %e'%(Rsquare, myf[3]))

    usethis = np.isfinite(E3) & np.isfinite(KD3-f1) & (num_muts3==1)    
    myf = stats.linregress(np.exp(E3[usethis]), KD3[usethis])
    residual = myf[0] * np.exp(E3[usethis]) + myf[1] - KD3[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum((KD3[usethis] - np.mean(KD3[usethis]))**2)
    print('CDR3H 1 mut R^2, m*E + b = F: %f, p-value: %e'%(Rsquare, myf[3]))
    
    usethis = np.isfinite(E3) & np.isfinite(KD3-f1) & (num_muts3>1)
    myf = stats.linregress(np.exp(E3[usethis]), KD3[usethis]-f1[usethis])
    residual = myf[0] * np.exp(E3[usethis]) + myf[1] - (KD3-f1)[usethis]
    Rsquare = 1 - np.sum(residual**2)/ np.sum(((KD3-f1)[usethis] - np.mean((KD3-f1)[usethis]))**2)
    print('CDR3H R^2 m*E + b = (F - F1): %f, m p-value: %f'%(Rsquare, myf[3]))
    

    ax = axes[0,1]
    cbar = plot_blosum(KD3, E3, num_muts3, KD_lims, [-1,0.5], ax, make_cbar=True,)
    ax.set_title('3H')
    ax.set_xlabel(r'log $_{10} K_d$ [M]')
    ax.set_ylabel('Expression', labelpad=2)

    ax = axes[1,1]
    cbar = plot_blosum(KD3-f1, E3, num_muts3, [-1,3.5], [-1,0.5], ax, make_cbar=True,)
    #plot_blosum(KD3, f1, num_muts3, KD_lims, ax, make_cbar=True, plot_ytick=True, max_freq=3,min_freq=1)
    ax.set_xlabel(r'log $_{10} K_d - $ log$_{10} PWM$')
    ax.set_ylabel('Expression', labelpad=2)
    
    plt.savefig('./figure_expression.pdf')
    plt.close()
