from data_preparation_transformed import get_f1, get_data
from get_fit_PWM_transformation import get_transformations
import numpy as np
import matplotlib.pyplot as plt
from labeler import Labeler
from scipy.stats import spearmanr, linregress
import pylab
import matplotlib as mpl
import pdb
import pandas
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm

np.random.seed(0)
wt_seq = 'TFSDYWMNWVGSYYGMDYWG'
def plot_blosum(f, f1, num_muts, lims, ylims, ax, curr_title='', make_cbar=False, plot_ytick=False, plot_xtick=True, min_freq=0, max_freq=2, logscale=True, custom_axis=[]):
    usethis = np.isfinite(f1) & np.isfinite(f)
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
        format_num = lambda x:int(np.round(x))
        for x in xticks:
            if np.round(x) != x:
                format_num = lambda x:np.round(x,1)

        xticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in xticks]
        ax.set_xticklabels(xticklabels)
        if plot_ytick:
            yticks = ax.get_yticks()
            yticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in yticks]
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
    txt = ax.text(0.31,0.02, r'$R^2=$' + str(np.round(Rsquare,2)), transform=ax.transAxes)
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

    
    return cbar

logKD2PWM, PWM2logKD = get_transformations() #choose log transformation
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM) #get data
usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]

AA1 = [AA[ind] for ind in usethis1]
AA3 = [AA[ind] for ind in usethis3]

usethis1 = med_rep.index[usethis1]
usethis3 = med_rep.index[usethis3]

KD1 = np.array((med_rep['KD'].loc[usethis1]))
KD3 = np.array((med_rep['KD'].loc[usethis3]))

E1 = np.array((med_rep['E'].loc[usethis1]))
E3 = np.array((med_rep['E'].loc[usethis3]))

num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])

wt_val = med_rep['KD'].loc[wt_seq]
##############################

fig, axes = plt.subplots(1,2, figsize=(7.3,3))
plt.subplots_adjust(
    bottom = 0.2,
    top = 0.92,
    left = 0.11,
    right = 0.86,
    hspace = 0.4,
    wspace = 0.7)

blosum_matrix = pandas.read_csv('blosum62.csv', header=0, index_col=0)
blosum = np.array([np.sum([blosum_matrix[wt_seq[ii]].loc[seq[ii]] for ii in range(len(seq))]) for seq in AA1])

ax = axes[0]
plot_blosum(KD1, blosum, num_muts1, KD_lims,[blosum.min(),blosum.max()], ax, curr_title='1H', make_cbar=False, plot_ytick=True, plot_xtick=True, max_freq=2, min_freq=0, logscale=True, custom_axis=[])
ax.set_xlabel(r'$K_d$')
ax.set_ylabel(r'BLOSUM62 score')
ax.set_title('1H')

blosum_matrix = pandas.read_csv('blosum62.csv', header=0, index_col=0)
blosum = np.array([np.sum([blosum_matrix[wt_seq[ii]].loc[seq[ii]] for ii in range(len(seq))]) for seq in AA3])
ax = axes[1]
slope, intercept, _,_,_ = linregress(blosum, KD3)

plot_blosum(KD3, blosum, num_muts3, KD_lims, [blosum.min(),blosum.max()], ax, curr_title='3H', make_cbar=True, plot_ytick=True, plot_xtick=True, max_freq=2, min_freq=0, logscale=True, custom_axis=[])

ax.set_xlabel(r'$K_d$')
ax.set_ylabel(r'BLOSUM62 score')
ax.set_title('3H')

plt.savefig('figure_S1.pdf')
plt.close()