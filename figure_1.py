#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import pandas
from helper import *
from scipy import stats
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import gridspec
from labeler import Labeler
from data_preparation_transformed import get_f1, get_data
from get_fit_PWM_transformation import get_transformations
import matplotlib.patheffects as PathEffects
import svgutils.transform as sg

def make_heatmap(AA, KD, num_muts, reference):
    #convert KD from amino acid sequences with mutation away from reference, and 
    #return a pandas array with columns denoting position, rows denoting amino acid
    inds = np.where(num_muts == 1)[0]
    wt_ind = np.where(num_muts == 0)[0]
    get_position = lambda seq:int(np.where([seq[ii] != reference[ii] for ii in range(len(reference))])[0])
    AA1 = [AA[ind] for ind in inds]
    positions = [get_position(aa) for aa in AA1] + list(range(len(reference)))
    AA_mut = [aa[position] for aa, position in zip(AA1, positions)] + list(reference)
    KD = KD[inds].tolist() + KD[wt_ind].tolist() * len(reference)
    out = pandas.DataFrame({'AA':AA_mut, 'residue':positions,'KD':KD})
    out = out.pivot(index='AA', columns='residue', values='KD')
    aas = ['G','A','V','I','L','M','F','Y','W','S','T','N','Q','C','P','H','K','R','D','E']
    return out.loc[aas]
    
# This is the function that does all of the plotting
def plot_panel(ax, heatmap, wtseq, optseq, colormap, make_cbar=True, plot_yticks=True):
    #plot heatmap to the ax axis. Wtseq and optimal (optseq) will be overlaid on top.
    #colormap is the colormap of the heatmap. make_cbar specifies if the colorbar should be made
    #plot_yticks denotes whether to plot the amino acids.
    vlim = [-9.5, -5.0]
    pos = np.array(heatmap.keys())
    cax = ax.pcolor(heatmap, cmap=colormap, vmin=vlim[0], vmax=vlim[1])
    aa_index = np.array(heatmap.index)
    jjs = np.array([np.where(aa_index==aa)[0][0] for aa in wtseq])
    ax.scatter(np.arange(len(jjs))+0.5, jjs+0.5, \
        marker='o', c=[0.8,0.2,0.8], linewidths=0.5, s=10, label='wt')
    iis = np.array([opt_pos for opt_pos, aa in enumerate(wtseq) if not optseq[opt_pos]==aa])
    jjs = np.array([np.where(aa_index==optseq[ii])[0][0] for ii in iis])
    ax.scatter(iis+0.5,jjs+0.5, \
        marker='o', c='Lime', linewidths=0.5, s=10, label='opt')

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
        ticklabels[-1]= r'$\geq$' + ticklabels[-1]
        p3 = ax.get_position().get_points()
        x00, y0 = p3[0]
        x01, y1 = p3[1]

        # [left, bottom, width, height]
        position = ([x01+0.005, y0, 0.015, y1-y0])
        cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels)
        #cbar.ax.tick_params()
        cbar.set_label(r'$K_d$ [M]',labelpad=-10)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_xlim([0,heatmap.shape[1]])
    ax.set_ylim([0,heatmap.shape[0]])

def plot_epistasis(f, f1, num_muts, lims, ax, curr_title='', make_cbar=False, plot_ytick=False, plot_xtick=True,min_freq=0, max_freq=3, logscale=True, custom_axis=[]):
    usethis = np.isfinite(f1) & np.isfinite(f)& (num_muts>1) #& (f>lims[0]) & (f<lims[1]) & (f1>lims[0]) & (f1<lims[1])
    Rsquare = 1 - np.nansum((f1[usethis]-f[usethis])**2)/np.nansum((f[usethis]-np.nanmean(f[usethis]))**2)
    F = np.nanvar((f1[usethis]-f[usethis]))/np.nanvar((f[usethis]-np.nanmean(f[usethis])))
    F_pval = stats.f.cdf(F, usethis.sum()-1, usethis.sum()-1)
    print('PWM R^2 for %s:%f, p-val:%e (F-test df %i, %i)'%(curr_title, Rsquare, F_pval, usethis.sum()-1,usethis.sum()-1))
    corr = np.corrcoef(f[usethis], f1[usethis])

    nbins = 20
    lims = np.array(lims)
    H, xedges, yedges = np.histogram2d(
        f[usethis], f1[usethis],
        bins=[np.linspace(lims[0], lims[1], nbins+1),
             np.linspace(lims[0], lims[1], nbins+1)])
    plt.sca(ax)

    xedges = np.linspace(lims[0], lims[1], nbins)
    yedges = np.linspace(lims[0], lims[1], nbins)
    [xx, yy] = np.meshgrid(xedges, yedges)

    lvls = np.logspace(start=0,stop=max_freq,num=(max_freq)*2 + 1, endpoint=True)
    lvls2 = np.logspace(start=0,stop=max_freq,num=max_freq+1, endpoint=True)
    #H[H<10**-0.25]=10**-0.25
    #H[H>lvls.max()] = lvls.max()
    H/=(xedges[1]-xedges[0]) * (yedges[1] - yedges[0])
    f_inds = np.searchsorted(np.linspace(lims[0]-1e-10, lims[1]+1e-10, nbins+1), f[usethis])
    f1_inds = np.searchsorted(np.linspace(lims[0]-1e-10, lims[1]+1e-10, nbins+1), f1[usethis])
    color = H[f_inds-1, f1_inds-1]
    density_sort = np.argsort(color)
    #color[color>10**max_freq] = 10**max_freq
    cax = ax.scatter(f[usethis][density_sort], f1[usethis][density_sort], s=20, c=color[density_sort],cmap='jet',  norm = LogNorm(vmin = 10**min_freq, vmax=10**max_freq))
    #cax = ax.contourf(xx, yy, H.transpose(), \
    #    levels=lvls, norm = LogNorm(), \
    #    cmap = 'gnuplot', zorder= 2, linestyles=None)


    if logscale:
        in_interval = lambda x: (x >= lims[0]) and (x <= lims[1])
        xticks = [ii for ii in range(int(np.floor(lims[0]))-1, int(np.ceil(lims[1]))+1) if ((ii%2)==1) and in_interval(ii)]
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
    
    lim_range = lims[1] - lims[0]
    ax.set_xlim(left=lims[0]-lim_range/4.5)
    ax.set_ylim(bottom=lims[0]-lim_range/4.5)
    txt = ax.text(0.31,0.02, r'$R^2=$' + str(np.round(Rsquare,2)), transform=ax.transAxes)
    #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=1)])
    ax.set_aspect(1)
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

    mpl.rcParams['font.size'] = 10
    mpl.font_manager.FontProperties(family = 'Helvetica')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'svgfont'
    #mpl.rcParams['svg.fonttype'] = 'none'
    # Needed for proper focusing
    plt.ion()
    plt.close('all')

    # Create figure with subplots and specified spacing
    figsize=(7.3*0.7,3.2)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(figsize=figsize)
    gs = gridspec.GridSpec(13, 40)
    plt.subplots_adjust(
        bottom = 0.15,
        top = 0.92,
        left = 0.05,
        right = 0.85,
        hspace = 2.3,
        wspace = 0.2)

    # Make a labeler to add labels to subplots
    labeler = Labeler(xpad=.015,ypad=.007,fontsize=14)
    A_heatmap = make_heatmap(AA, med_rep['KD'], med_rep['CDR1_muts'] + med_rep['CDR3_muts'], wtseq)
    A_heatmap = A_heatmap.rename(columns={k:v for k,v in enumerate(list(range(28, 38))+list(range(100,110)))})
    # Affinity plot, lib1
    ax = plt.subplot(gs[0:12,0:7])
    labeler.label_subplot(ax,'C')
    plot_panel(ax, A_heatmap[list(range(28,38))], cdr1_wtseq, cdr1_optseq, colormap=red_blue, make_cbar = False)
    ax.set_title('1H', fontsize=mpl.rcParams['font.size'])

    # Affinity plot, lib2
    ax = plt.subplot(gs[0:12,9:16])
    plot_panel(ax, A_heatmap[list(range(100,110))], cdr3_wtseq, cdr3_optseq, colormap=red_blue, make_cbar = True, plot_yticks=False)
    ax.set_title('3H', fontsize=mpl.rcParams['font.size'])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.15),
          fancybox=True, shadow=True, ncol=5)
    
    ax = plt.subplot(gs[0:6,30:40])

    f1, x = get_f1(A1, num_muts1, KD1, wt_val, limit=KD_lims)
    usethis = np.isfinite(E1) & np.isfinite(KD1)
    myf = stats.spearmanr(np.exp(E1[usethis]), KD1[usethis])
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

    plot_epistasis(KD1, f1, num_muts1, KD_lims, ax, make_cbar=True, plot_ytick=True, plot_xtick=False, max_freq=3,min_freq=1)
    ax.set_ylabel('PWM [M]', labelpad=2)
    #txt = ax.text(0.02, 0.77,'1H', transform=ax.transAxes)
    ax.set_title('1H')
    #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.5)])
    labeler = Labeler(xpad=.12,ypad=.007,fontsize=14)

    ax = plt.subplot(gs[7:13,30:40])
    f1, x = get_f1(A3, num_muts3, KD3, wt_val, limit=KD_lims)
    usethis = np.isfinite(E3) & np.isfinite(KD3)
    myf = stats.spearmanr(np.exp(E3[usethis]), KD3[usethis])
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
    cbar = plot_epistasis(KD3, f1, num_muts3, KD_lims, ax, make_cbar=True, plot_ytick=True, max_freq=3,min_freq=1)

    ax.set_xlabel(r'$K_d$ [M]')
    ax.set_ylabel('PWM [M]', labelpad=2)
    #txt = ax.text(0.02, 0.77,'3H', transform=ax.transAxes)
    ax.set_title('3H')
    #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.5)])#labeler.label_subplot(ax,'D')
    ax = plt.subplot(gs[0:6,30:40])
    labeler = Labeler(xpad=.09,ypad=.007,fontsize=14)
    labeler.label_subplot(ax,'D')

    plt.savefig('./figure_1_lower.svg')
    plt.savefig('./figure_1_lower.pdf')
    plt.close()

    fig = sg.SVGFigure( "%.2fcm"%(7.3*0.7*2.54), "%.2fcm"%(4.1*2.54))
    # load matplotlib-generated figures
    fig1 = sg.fromfile('figure_1_top.svg')
    fig2 = sg.fromfile('figure_1_lower.svg')
    # get the plot objects
    plot1 = fig1.getroot()
    plot1.moveto(0, 0)
    plot2 = fig2.getroot()
    plot2.moveto(0, 80)
    
    # append plots and labels to figure
    fig.append([plot1, plot2])

    # save generated SVG files
    fig.save("figure_1.svg")
