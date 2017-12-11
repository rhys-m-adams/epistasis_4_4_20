#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import pandas
from helper import *
from scipy.stats import norm, chi2, pearsonr, spearmanr
from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from labeler import Labeler
from data_preparation_transformed import get_f1, get_data
from get_fit_PWM_transformation import get_transformations
import matplotlib.patheffects as PathEffects
import svgutils.transform as sg
#import os
logKD2PWM, PWM2logKD = get_transformations()

def make_heatmap(AA, KD, num_muts, reference):
    inds = np.where(num_muts == 1)[0]
    aas = ['G','A','V','I','L','M','F','Y','W','S','T','N','Q','C','P','H','K','R','D','E']
    aa_map = {k:ii for ii, k in enumerate(aas)}
    wt_val = float(KD[np.where(num_muts==0)[0]])
    out = np.zeros((20, len(reference))) + wt_val
    for ind in inds:
        curr_AA = AA[ind]
        curr_KD = KD[ind]
        residue_pos = int(np.where([curr_AA[ii] != reference[ii] for ii in range(len(reference))])[0])
        out[aa_map[curr_AA[residue_pos]], residue_pos] = curr_KD
    out = pandas.DataFrame(out, index=aas)
    return out

# This is the function that does all of the plotting
def plot_panel(ax, heatmap, wtseq, optseq, colormap, make_cbar=True, plot_yticks=True):
    vlim = [-9.5, -5.0]
    pos = heatmap.keys()
    cax = ax.pcolor(heatmap, cmap=colormap, vmin=vlim[0], vmax=vlim[1])
    aa_index = np.array(heatmap.index)

    for ii, aa in enumerate(wtseq):
        jj = np.where(aa_index==aa)[0][0]
        plt.scatter(ii+0.5, jj+0.5, \
            marker='o', c=[0.8,0.2,0.8], linewidths=0.5, s=10)

        # Plot OPT seq mutation if any occurs at position ii
        if not (optseq[ii] == aa):
            jj = np.where(aa_index==optseq[ii])[0][0]
            plt.scatter(ii+0.5,jj+0.5, \
                marker='o', c='Lime', linewidths=0.5, s=10)

    ax.set_xlabel('VH position',labelpad=2)
    ticks = np.array([0,4,8])
    ax.set_xticks(ticks+0.5)
    ax.set_xticklabels([str(k) for k in pos[ticks]], fontsize=9)

    if plot_yticks:
        ax.set_yticks(np.arange(0.5, heatmap.shape[0] + 0.5))
        ax.set_yticklabels(heatmap.index, ha='left', fontsize=9)
        [tick.set_color(aa_colors2[ii]) for (ii,tick) in zip(aa_index,ax.yaxis.get_ticklabels())]
        ax.tick_params(axis='y', which='major', pad=6)
    else:
        ax.set_yticks([])

    if make_cbar:
        ticks = [-9.0, -8.0, -7.0, -6.0, -5.0]
        ticklabels = [r'$10^{%i}$'%t for t in ticks]
        #ticklabels[0]= r'$\leq$' + ticklabels[0]
        ticklabels[-1]= r'$\geq$' + ticklabels[-1]
        p3 = ax.get_position().get_points()
        x00, y0 = p3[0]
        x01, y1 = p3[1]

        # [left, bottom, width, height]
        position = ([x01+0.005, y0, 0.015, y1-y0])
        cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.tick_params()
        cbar.set_label(r'$K_D$ [M]',labelpad=-10)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_xlim([0,heatmap.shape[1]])
    ax.set_ylim([0,heatmap.shape[0]])

def plot_epistasis(f, f1, num_muts, lims, ax, curr_title='', make_cbar=False, plot_ytick=False, plot_xtick=True, max_freq=2, logscale=True, custom_axis=[]):
    usethis = np.isfinite(f1) & np.isfinite(f)& (num_muts>1) #& (f>lims[0]) & (f<lims[1]) & (f1>lims[0]) & (f1<lims[1])
    Rsquare = 1 - np.nansum((f1[usethis]-f[usethis])**2)/np.nansum((f[usethis]-np.nanmean(f[usethis]))**2)
    F = np.nanvar((f1[usethis]-f[usethis]))/np.nanvar((f[usethis]-np.nanmean(f[usethis])))
    F_pval = stats.f.cdf(F, usethis.sum()-1, usethis.sum()-1)
    print 'PWM R^2 for %s:%f, p-val:%e (F-test df %i, %i)'%(curr_title, Rsquare, F_pval, usethis.sum()-1,usethis.sum()-1)
    corr = np.corrcoef(f[usethis], f1[usethis])

    nbins = 20
    lims = PWM2logKD(np.array(lims))
    H, xedges, yedges = np.histogram2d(
        PWM2logKD(f[usethis]), PWM2logKD(f1[usethis]),
        bins=[np.linspace(lims[0], lims[1], nbins+1),
             np.linspace(lims[0], lims[1], nbins+1)])
    plt.sca(ax)

    xedges = np.linspace(lims[0], lims[1], nbins)
    yedges = np.linspace(lims[0], lims[1], nbins)
    [xx, yy] = np.meshgrid(xedges, yedges)

    lvls = np.logspace(start=-0.5,stop=max_freq,num=(max_freq+1)*2, endpoint=True)
    lvls2 = np.logspace(start=0,stop=max_freq,num=max_freq+1, endpoint=True)
    H[H<0.1]=0.1
    H[H>lvls.max()] = lvls.max()
    im = ax.contourf(xx, yy, H.transpose(), \
        levels=lvls, norm = LogNorm(), \
        cmap = 'Greys', zorder= 2, linestyles=None)


    if logscale:
        xticks = [-9,-7,-5]
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        format_num = lambda x:int(np.round(x))
        for x in xticks:
            if np.round(x) != x:
                format_num = lambda x:np.round(x,1)

        xticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in xticks]
        ax.set_xticklabels(xticklabels)
        if plot_ytick:
            yticks = ax.get_yticks()
            yticklabels = [r'$10^{'+str(format_num(x))+'}$' for x in yticks]
            ax.set_yticklabels(yticklabels)
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

    txt = ax.text(0.2,0.1, r'$R^2=$' + str(np.round(Rsquare,2)), transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    ax.set_aspect(1.)
    plt.savefig('deleteme.pdf')
    cbar = []
    if make_cbar:

        p3 = np.array(ax.get_position().get_points())
        position = ([p3[1][0]+0.005, p3[0][1], 0.015, p3[1][1]-p3[0][1]])

        cbar = plt.colorbar(im, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=lvls)
        cbar.set_label('frequency',labelpad=0)
        xtl = []
        for ii in range(0, (max_freq+1)*2-1):
            if (ii%2)==1:
                xtl.append(r'$10^{%i}$'%((ii-1)/2))
            else:
                xtl.append('')
            xtl.append(r'$\geq 10^{%i}$'%(max_freq))

        def cleanup(x):
            if x in lvls2:
                return r'$10^{%i}$'%(np.log10(x))
            return r'$10^{\frac{%i}{2}}$'%np.round(np.log10(x)*2)

        level_ticks = [cleanup(level) for level in lvls]
        level_ticks[-1] = r'$\geq$' + level_ticks[-1]


    return cbar

if __name__ == '__main__':
    med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

    usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
    usethis3 = np.where(med_rep['CDR1_muts']==0)[0]

    A1 = A[usethis1]
    A3 = A[usethis3]

    usethis1 = med_rep.index[usethis1]
    usethis3 = med_rep.index[usethis3]

    KD1 = np.array((med_rep['KD'].loc[usethis1]))
    KD3 = np.array((med_rep['KD'].loc[usethis3]))

    num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
    num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])

    cdr1_wtseq = 'TFSDYWMNWV'
    cdr3_wtseq = 'GSYYGMDYWG'
    cdr1_optseq = 'TFGHYWMNWV'
    cdr3_optseq = 'GASYGMEYLG'

    wtseq = cdr1_wtseq + cdr3_wtseq
    usethis = np.array(med_rep['CDR1_muts'] == 0) & np.array(med_rep['CDR3_muts'] == 0)
    wt_val = np.array(med_rep['KD'])[usethis]
    white_point = (PWM2logKD(wt_val)+9.5)/4.5
    affinity_maturation = (-6+9.5)/4.5
    #cdict1 = {'red':((0.,1.,1.),
    #                    (white_point,1.,1.),
    #                    (affinity_maturation, 0, 0.5),
    #                    (1.,0.8,0.8)),
    #                'green':((0.,0.,0.),
    #                    (white_point,1.,1.),
    #                    (1.,0.,0.)),
    #                'blue':((0.,0.,0.),
    #                    (white_point,1.,1.),
    #                    (1.,1.,1.))
    #                }
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

    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'svgfont'
    #mpl.rcParams['svg.fonttype'] = 'none'
    # Needed for proper focusing
    plt.ion()
    plt.close('all')

    # Create figure with subplots and specified spacing
    figsize=(7.3*0.7,3.)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(figsize=figsize)
    gs = gridspec.GridSpec(6, 40)
    plt.subplots_adjust(
        bottom = 0.16,
        top = 0.92,
        left = 0.05,
        right = 0.88,
        hspace = 2.3,
        wspace = 0.2)

    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=.015,ypad=.007,fontsize=14)
    A_heatmap = make_heatmap(AA, med_rep['KD'], med_rep['CDR1_muts'] + med_rep['CDR3_muts'], wtseq)
    A_heatmap = A_heatmap.rename(columns={k:v for k,v in enumerate(range(28, 38)+range(100,110))})

    #ax = plt.subplot(gs[0:6,0:4])
    #ax.set_xlabel('VH position')

    # Affinity plot, lib1
    ax = plt.subplot(gs[0:6,0:7])
    labeler.label_subplot(ax,'C')
    plot_panel(ax, A_heatmap[range(28,38)], cdr1_wtseq, cdr1_optseq, colormap=red_blue, make_cbar = False)
    ax.set_title('1H', fontsize=mpl.rcParams['font.size'])

    # Affinity plot, lib2
    ax = plt.subplot(gs[0:6,9:16])
    #labeler.label_subplot(ax,'B')
    plot_panel(ax, A_heatmap[range(100,110)], cdr3_wtseq, cdr3_optseq, colormap=red_blue, make_cbar = True, plot_yticks=False)
    ax.set_title('3H', fontsize=mpl.rcParams['font.size'])

    ax = plt.subplot(gs[0:3,30:40])

    f1, x = get_f1(A1, num_muts1, KD1, wt_val, limit=KD_lims)
    plot_epistasis(KD1, f1, num_muts1, KD_lims, ax, make_cbar=True, plot_ytick=True, plot_xtick=False, max_freq=1)
    #ax.set_xlabel(r'$K_D$ [M]')
    ax.set_ylabel('PWM [M]', labelpad=2)
    txt = ax.text(0.05, 0.85,'1H', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    labeler = Labeler(xpad=.12,ypad=.007,fontsize=14)
    #labeler.label_subplot(ax,'B')

    #p3 = np.array(ax.get_position().get_points())
    #height = p3[1][1]-p3[0][1]


    ax = plt.subplot(gs[3:6,30:40])
    f1, x = get_f1(A3, num_muts3, KD3, wt_val, limit=KD_lims)
    cbar = plot_epistasis(KD3, f1, num_muts3, KD_lims, ax, make_cbar=True, plot_ytick=True, max_freq=1)
    #position = cbar.get_position()
    #cbar.set_position([position.x,position.y,position.width, height])

    ax.set_xlabel(r'$K_D$ [M]')
    ax.set_ylabel('PWM [M]', labelpad=2)
    txt = ax.text(0.05, 0.85,'3H', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])#labeler.label_subplot(ax,'D')
    #ax.set_xlabel(r'$K_D$ [M]')
    ax = plt.subplot(gs[0:3,30:40])
    labeler.label_subplot(ax,'D')

    plt.savefig('./figure_1_lower.svg')
    plt.savefig('./figure_1_lower.pdf')
    plt.close()

    fig = sg.SVGFigure( "%.2fcm"%(7.3*0.7*2.54), "%.2fcm"%(4.1*2.54))
    #fig = sg.SVGFigure( "%.2f"%(7.3*0.7*2.54), "%.2f"%(4.1*2.54))

    # load matpotlib-generated figures
    fig1 = sg.fromfile('figure_1_top.svg')
    fig2 = sg.fromfile('figure_1_lower.svg')
    # get the plot objects
    plot1 = fig1.getroot()
    plot1.moveto(0, 0)#, scale=7.3*0.7/144.5669)

    plot2 = fig2.getroot()

    plot2.moveto(0, 80)#*7.3*0.7/144.5669, scale=7.3*0.7/144.5669)

    # add text labels

    # append plots and labels to figure
    fig.append([plot1, plot2])

    # save generated SVG files
    fig.save("figure_1.svg")
    #os.system('inkscape -A figure_1_inkscape.pdf figure_1.svg')
