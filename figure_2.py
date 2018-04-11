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

logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.font_manager.FontProperties(family = 'Helvetica')
mpl.rcParams['pdf.fonttype'] = 42

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


def plot_node(ax, x, y, width, height, color, border_width=0):
    # plot a node (as in figure 2D) in axis ax, centered at postion x,y with width/height, and specified color
    patches = []
    white_patches = []
    patch_colors = []
    white_colors = []

    fancybox = mpatches.FancyBboxPatch(
        np.array([x, y]) - np.array([width/2., height/2.]), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.08))
    patches.append(fancybox)
    patch_colors.append(color)

    collection = PatchCollection(patches, cmap=red_blue, alpha=1, lw = border_width, zorder=15, edgecolor=[0,0,0])
    collection.set_clim(vmin=-9.5, vmax=-5)
    collection.set_array(np.array(patch_colors).flatten())
    ax.add_collection(collection)


def text_color(KD):
    #put light text on dark backgrounds, or dark text on light backgrounds
    if (KD<-9.25) or (KD>-7.5):
        return [1,1,1]
    else:
        return [0,0,0]


def plot_connected_dm(representative, ax, plot_title, make_colorbar, y_offset, wt_val, PWM2logKD):
    #plot the representative sign epistasis pairs shown in figure 2D
    #ax - matplotlib axis
    #plot_title - name of the plot
    #make_colorbar - boolean value for whether to make a colorbar
    #y_offset - offset the figure along the y-axis so you can put one on top of the other
    #wt_val - value of the wildtype, i.e. starting value
    #PWM2logKD - function transforming PWM values to log KD values, usually the same.
    
    angles = (np.array([90,180,270, 360])-45)/180. * np.pi
    shown_x = 2 * np.cos(angles)
    shown_y = 2 * np.sin(angles)+y_offset

    off_angle = np.array([10,33,10,33])
    r1 = np.array([1,1.3,1,1.3])
    angles1 = angles + off_angle * np.pi/ 180
    off_angle = np.array([33,10,33,10])
    r2 = np.array([1.3,1,1.3,1])

    angles2 = angles - off_angle * np.pi/180
    inner_shown_x1= np.cos(angles1)*r1
    inner_shown_y1= np.sin(angles1)*r1+y_offset

    inner_shown_x2= np.cos(angles2)*r2
    inner_shown_y2= np.sin(angles2)*r2+y_offset

    ax.axis('off')
    cax = ax.scatter(0, y_offset,s=1, c=-8, zorder=0, vmin=-9.5, vmax=-5, cmap=red_blue)

    plot_node(ax, 0, y_offset, 0.8, 0.3, PWM2logKD(wt_val), border_width=1)
    ax.text(0, y_offset, 'WT', zorder=21, ha='center', va='center')

    for ii, k in enumerate(representative.index):
        single_mut1 = r'%s$_{%i}$'%(representative['AA1'].loc[k],representative['pos1'].loc[k])
        single_mut2 = r'%s$_{%i}$'%(representative['AA2'].loc[k],representative['pos2'].loc[k])
        double_mut = single_mut1 + single_mut2
        logKD1 = PWM2logKD(representative['h1'].loc[k] + wt_val)
        plot_node(ax,inner_shown_x1[ii],inner_shown_y1[ii],0.95,0.3, logKD1)
        txt = ax.text(inner_shown_x1[ii],inner_shown_y1[ii], single_mut1, zorder=21, ha='center', va='center', color = text_color(logKD1))

        logKD2 = PWM2logKD(representative['h2'].loc[k] + wt_val)
        plot_node(ax,inner_shown_x2[ii],inner_shown_y2[ii],0.95,0.3,logKD2)
        txt = ax.text(inner_shown_x2[ii],inner_shown_y2[ii],single_mut2, zorder=21, ha='center', va='center', color = text_color(logKD2))

        logKD12 = PWM2logKD(representative['F'].loc[k])
        plot_node(ax,shown_x[ii],shown_y[ii],2.1,0.4, logKD12)
        txt = ax.text(shown_x[ii],shown_y[ii], double_mut, zorder=21, ha='center', va='center', color = text_color(logKD12))

        ax.plot([inner_shown_x1[ii], shown_x[ii]], [inner_shown_y1[ii], shown_y[ii]], c=[0,0,0])
        ax.plot([inner_shown_x2[ii], shown_x[ii]], [inner_shown_y2[ii], shown_y[ii]], c=[0,0,0])
        ax.plot([inner_shown_x1[ii], 0], [inner_shown_y1[ii], y_offset], c=[0,0,0])
        ax.plot([inner_shown_x2[ii], 0], [inner_shown_y2[ii], y_offset], c=[0,0,0])

    if make_colorbar:
        ticks = [-9,-7,-5]

        p3 = ax.get_position().get_points()
        x00, y0 = p3[0]
        x01, y1 = p3[1]

        # [left, bottom, width, height]
        position = ([x00+0.03, 0.11, x01-x00-0.06, 0.02])
        #cbar = plt.colorbar(im, orientation='vertical', ticks=lvls2)
        cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='horizontal', ticks=ticks)
        cbar.set_ticklabels(['$10^{%i}$'%(logKD) for logKD in ticks])


        #cbar.ax.set_yticklabels([r'$10^{%d}$'%np.log10(t) for t in lvls2])
        cbar.set_label(r'$K_d$ [M]',labelpad=2)

    ax.set_ylim([-1.8,5.8])
    ax.set_xlim([-2.6,2.6])
    ax.text(0.,1.7 + y_offset, plot_title, ha='center')#plt.colorbar(cax)


def prepare_X_Z(A, val, val_std, AA, offset):
    num_muts = np.array(A.sum(axis=1)).flatten()
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = float(val[ind_zero])
    val_var = val_std**2
    f1, x = get_f1(A, num_muts, val, wt_val)#, limit=[-9.5,-5])
    f1_var, x_var = get_f1(A, num_muts, val_var, 0)

    usethis = np.where((num_muts==2) & np.isfinite(val) & np.isfinite(val_std))[0]
    AA2 = [AA[ind] for ind in usethis]
    out = pandas.DataFrame({'F':val[usethis], 'F_PWM':f1[usethis], 'F_PWM_std': np.sqrt(f1_var[usethis]), 'F_std':val_std[usethis]}, index=AA2)
    Z1 = np.array([x[np.array(Arow).flatten()==1]/np.sqrt(x_var[np.array(Arow).flatten()==1]+1e-16) for Arow in np.array(A[usethis].todense())])
    std1 = np.sqrt(np.array([x_var[np.array(Arow).flatten()==1] for Arow in np.array(A[usethis].todense())]))
    X1 = np.array([x[np.array(Arow).flatten()==1] for Arow in np.array(A[usethis].todense())])
    #out['Z'] = (out['f'] - out['f1']) * 1./np.sqrt(out['f1_std']**2 + out['f_std']**2)
    #X2 = np.array(out[['f','f']])

    out['Z_sign1'] = (np.array(out['F']) - (X1[:,1] + wt_val)) * 1./np.sqrt(np.array(out['F_std'])**2 + std1[:,1]**2 + 1e-16)
    out['Z_sign2'] = (np.array(out['F']) - (X1[:,0] + wt_val)) * 1./np.sqrt(np.array(out['F_std'])**2 + std1[:,0]**2 + 1e-16)
    out['delta_1'] = (np.array(out['F']) - (X1[:,1] + wt_val))
    out['delta_2'] = (np.array(out['F']) - (X1[:,0] + wt_val))
    out['Z_PWM'] = (out['F'] - out['F_PWM']) * 1./np.sqrt(out['F_PWM_std']**2 + out['F_std']**2 + 1e-16)
    AA2 = [AA[ind] for ind in usethis]
    double_mut_pos = np.array([np.where([wt!=mut for wt,mut in zip(wt_aa, aa)])[0] for aa in AA2])
    out['Z1'] = Z1[:,0]
    out['Z2'] = Z1[:,1]
    out['h1'] = X1[:,0]
    out['h2'] = X1[:,1]
    out['h1std'] = std1[:,0]
    out['h2std'] = std1[:,1]
    out['pos1'] = double_mut_pos[:,0]+offset
    out['pos2'] = double_mut_pos[:,1]+offset
    out['AA1'] = [aa[ind] for aa,ind in zip(AA2, double_mut_pos[:,0])]
    out['AA2'] = [aa[ind] for aa,ind in zip(AA2, double_mut_pos[:,1])]

    return out, wt_val


def get_FDR_p(single_cutoff, double_cutoff):
    sampling = 10000000
    S = np.eye(4)*4./3 - 1./3
    val, vec = np.linalg.eigh(S)
    inv = vec.dot(np.diag(np.sqrt(np.abs(val))))
    nn = vec.dot(np.repeat(np.sqrt(np.array([(np.abs(val))])),sampling,axis=0).T * np.random.randn(4,sampling))
    double = sum(np.prod(nn[0:2]<single_cutoff, axis=0) * np.prod(nn[2:4]>double_cutoff, axis=0))/float(sampling)
    single = 2 * sum((nn[0]<single_cutoff) & (nn[1]<single_cutoff))/float(sampling) - double
    return single, double

def query_FDR_p(single_cutoff, double_cutoff):
    single_cutoff = (np.round(single_cutoff,8))
    double_cutoff = (np.round(double_cutoff,8))
    try:
        known = pandas.read_csv('FDR_table.csv',header=0, index_col=0)
    except:
        single, double = get_FDR_p(single_cutoff, double_cutoff)
        pandas.DataFrame({'cutoff1':[single_cutoff],'cutoff2':[double_cutoff],'single':[single],'double':[double]}).to_csv('FDR_table.csv')
        known = pandas.read_csv('FDR_table.csv',header=0, index_col=0)

    usethis = known['cutoff1'].isin([single_cutoff]) & known['cutoff2'].isin([double_cutoff])
    if np.sum(usethis) > 0:
        single = float(known['single'].loc[usethis])
        double = float(known['double'].loc[usethis])
    else:
        single, double = get_FDR_p(single_cutoff, double_cutoff)
        known = known.append(pandas.DataFrame({'cutoff1':[single_cutoff],'cutoff2':[double_cutoff],'single':[single],'double':[double]}))
        known.to_csv('FDR_table.csv')
    return single, double


def plot_KD_sign_epistasis(A, val, val_std, AA, title_name, AA_pos_offset, logical_operator, ax=None, make_colorbar=False, out ={}, fid=None, epistasis='beneficial', y_offset=0, PWM2logKD=lambda x:x):
    summary, wt_val = prepare_X_Z(A, val, val_std, AA, AA_pos_offset)
    first_size = summary.shape[0]
    cutoff = logKD2PWM(-6) - wt_val

    if logical_operator == 'AND':
        allowable = (summary['h1'] > cutoff) & (summary['h2'] > cutoff)
        num_catastrophic = '$2$'
    if logical_operator == 'XOR':
        allowable = (summary['h1'] > cutoff) ^ (summary['h2'] > cutoff)
        num_catastrophic = '$1$'
    if logical_operator == 'NAND':
        allowable = ~((summary['h1'] > cutoff) & (summary['h2'] > cutoff))
        num_catastrophic = '$0, 1$'
    if logical_operator == 'NOR':
        allowable = ~((summary['h1'] > cutoff) | (summary['h2'] > cutoff))
        num_catastrophic = '$0$'
    if logical_operator == 'ALL':
        allowable = np.isfinite(summary['h1']) & np.isfinite(summary['h2'])
        num_catastrophic = '$0-2$'

    cutoff = norm.ppf(0.95)
    if epistasis == 'deleterious':
        summary['Z1'] *= -1
        summary['Z2'] *= -1
        summary['Z_sign1'] *= -1
        summary['Z_sign2'] *= -1


    p_improved = (PWM2logKD(summary['F']) < wt_val).sum() / float(summary.shape[0])
    p_viable = (PWM2logKD(summary['F']) < -6).sum() / float(summary.shape[0])

    p_allowable = allowable.sum() / float(first_size)
    single, double = query_FDR_p(-cutoff, cutoff)
    expected_num_r_s_e = np.max([first_size * double * p_allowable, 0.01])
    expected_num_s_e = np.max([first_size * single * p_allowable, 0.01])

    expected_num_viable = np.max([expected_num_s_e * p_viable, 0.01])
    expected_num_viable_r_s_e = np.max([expected_num_r_s_e * p_viable, 0.01])

    expected_super_sig = np.max([expected_num_s_e * p_improved, 0.01])

    viable = allowable & ((summary['Z1'] > cutoff) | (summary['Z2'] > cutoff))
    summary = summary.loc[viable]
    total_count = summary.shape[0]

    summary = summary.loc[((summary['Z_sign1'] < -cutoff) & (summary['Z1'] > cutoff) ) | ((summary['Z_sign2'] < -cutoff) & (summary['Z2'] > cutoff))]

    num_s_e = summary.shape[0]
    num_r_s_e = summary.loc[((summary['Z_sign1'] < -cutoff) & (summary['Z1'] > cutoff)) & ((summary['Z_sign2'] < -cutoff) & (summary['Z2'] > cutoff))].shape[0]

    viable = summary.loc[PWM2logKD(summary['F']) < -6 ]

    num_viable = viable.shape[0]
    num_viable_r_s_e = viable.loc[((viable['Z_sign1'] < -cutoff) & (viable['Z1'] > cutoff)) & ((viable['Z_sign2'] < -cutoff) & (viable['Z2'] > cutoff))].shape[0]

    super_sig = (PWM2logKD(summary['F']) < wt_val).sum()
    if not(fid == None):
        fid.write('%s & %s & %s & %i & %i/%.2f  & %i/%.2f & %i/%.2f \\\\ \\hline \n'%(title_name, num_catastrophic, epistasis, total_count, num_s_e, expected_num_s_e, num_r_s_e, expected_num_r_s_e, num_viable, expected_num_viable))

    if ax != None:
        for_show = pandas.DataFrame(summary)
        usethis = []

        if for_show.shape[0]>0:
            usethis.append(for_show.index[np.argsort(for_show[['Z_sign1','Z_sign2']].min(axis=1))[0]])
            for_show = for_show.drop(usethis[0])
        if for_show.shape[0]>0:
            usethis.append(for_show.index[np.argsort(for_show['Z1']+for_show['Z2'])[-1]])
            for_show = for_show.drop(usethis[-1])
        if for_show.shape[0]>0:
            usethis.append(for_show.index[np.argsort(for_show['F'])[0]])
            for_show = for_show.drop(usethis[-1])
        if for_show.shape[0]>0:
            usethis.append(for_show.index[np.argsort(for_show['F'])[0]])
            for_show = for_show.drop(usethis[-1])
        if len(usethis)>0:
            representative = summary.loc[usethis]
            plot_connected_dm(representative, ax, title_name, make_colorbar, y_offset, wt_val, PWM2logKD)

    print '%s %s, average effect: %f'%(title_name, logical_operator, (summary['F'] - summary['F_PWM']).mean())
    return summary


def calculate_Z_epistasis_by_pos(A, num_muts, val, val_std, pos, usethis, lims, epi_range = [-np.inf, np.inf]):
    f = (np.array(val)).flatten()
    val_var = val_std**2
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = val[ind_zero]

    f1, x = get_f1(A, num_muts, f, wt_val, limit =lims)
    f1_var, x_var = get_f1(A, num_muts, val_var, 0)
    sigma_m = np.sqrt(f1_var + val_var)
    usethis = np.isfinite(f1) & np.isfinite(f)& (num_muts>=2) & usethis  #& (f>lims[0]) & (f<lims[1])# & (f1>lims[0]) & (f1<lims[1])
    deviance = [[[] for jj in range(10)] for ii in range(10)]
    n = [[[] for jj in range(10)] for ii in range(10)]
    for l, lhat, s_m, p in zip(f[usethis], f1[usethis], sigma_m[usethis], pos[usethis]):
        curr = np.where(p!=0)[0]
        for ii in curr:
            for jj in curr:
                delta = l - lhat
                if (delta>epi_range[0]) and (delta<epi_range[1]):
                    deviance[ii][jj].append(((l-lhat)/s_m)**2)

    for ii in range(10):
        for jj in range(10):
            if len(deviance[ii][jj]):
                curr = np.array(deviance[ii][jj])
                usethis = np.isfinite(curr)
                n[ii][jj] = np.sum(usethis)
                if n[ii][jj] < 1:
                    deviance[ii][jj] = np.nan
                else:
                    deviance[ii][jj] = np.sqrt(np.nanmean(curr[usethis]))
                if ii==jj:
                    deviance[ii][jj] = np.nan

            else:
                n[ii][jj] = 0
                deviance[ii][jj] = np.nan

    deviance = np.array(deviance)
    return deviance


def plot_Z_epistasis_by_pos(deviance, in_max, offset, cutoff, ax, curr_title=None, opt=[], make_colorbar=False, make_ylabel=False):
    cmap = plt.cm.OrRd
    cmap.set_bad('gray',1.)
    pop1 = []
    pop2 = []
    for ii in range(deviance.shape[0]):
        for jj in range(ii+1,deviance.shape[0]):
            if np.isfinite(deviance[ii,jj]):
                if (ii in opt) and (jj in opt):
                    pop1.append(deviance[ii,jj])
                else:
                    pop2.append(deviance[ii,jj])

    print 'epistasis enrichment in OPT (%s domain) by position p-val: %f (mann whitney test)'%(curr_title, mannwhitneyu(pop1,pop2)[1])
    print 'number of optimized mutant pairs with epistatic Z>3: %i'%np.sum(np.array(pop1)>3)
    print 'number of epistatic hotspots Z>3: %i'%int(np.nansum(np.array(deviance)>3)/2)
    deviance = np.ma.array (deviance, mask=np.isnan(deviance))
    opt = np.array(opt, dtype=float)+0.5
    for ii in range(len(opt)):
        for jj in range(len(opt)):
            if ii!=jj:
                ax.scatter(opt[ii], opt[jj], s=12, c=[0,0.8,0.], zorder=20)
    cax = ax.pcolor(deviance, facecolor=[0.6,0.6,0.6], cmap=cmap,vmin=0, vmax = in_max)
    ax.set_facecolor([0.6,0.6,0.6])

    ax.set_xlabel('VH position')
    if make_ylabel:
        ax.set_ylabel('VH position')

    ax.set_xticks(np.arange(0,deviance.shape[0],4)+0.5)
    ax.set_yticks(np.arange(0,deviance.shape[0],4)+0.5)
    ax.set_xticklabels(np.arange(0,deviance.shape[0],4)+offset)
    ax.set_yticklabels(np.arange(0,deviance.shape[0],4)+offset)
    ax.tick_params(axis='y', which='major', pad=1)

    ax.set_title(curr_title)
    ax.set_xlim([0,deviance.shape[0]])
    ax.set_ylim([0,deviance.shape[0]])
    ax.set_aspect(1.)
    plt.savefig('deleteme.pdf')

    if make_colorbar:
        p3 = ax.get_position().get_points()
        x00, y0 = p3[0]
        x01, y1 = p3[1]
        height = y1 - y0
        position = ([x01+0.01, y0, 0.015, height])
        
        delta = np.max([int(in_max/5),1])
        ticks = range(0, in_max+1, delta)
        cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=ticks)
        cbar.set_label(r'$\langle Z^2 \rangle^\frac{1}{2}$', labelpad=-10)
        
        ticklabels = [r'$%i$'%(ii) for ii in ticks]
        ticklabels[-1]=r'$\geq$'+ticklabels[-1]
        cbar.set_ticklabels(ticklabels)


def plot_epistasis_Z(A, num_muts, val, val_std, Z_null, name, title_name, lims, ax, make_ytick = False, plot_null=False):
    Z_null= np.array(Z_null)
    f = np.array(val)
    val_var = val_std**2
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = val[ind_zero]
    
    f1, x = get_f1(A, num_muts, f, wt_val, limit = lims)
    f1_var, x_var = get_f1(A, num_muts, val_var, 0)
    usethis = np.isfinite(f1) & np.isfinite(f) & (num_muts==2) & ((f1_var + val_var) != 0)
    residual = -f + f1
    Rsquare = 1 - np.sum(residual[usethis]**2)/np.sum((f[usethis] - np.mean(f[usethis]))**2)
    Z = np.zeros(residual.shape) * np.nan
    Z[usethis] = residual[usethis] * 1./np.sqrt(f1_var[usethis] + val_var[usethis])
    usethis = usethis & np.isfinite(Z)
    fraction_epistasis = ((np.var(Z[usethis]) - np.nanvar(Z_null)) / np.var(Z[usethis]))
    non_boundary_f1 = (f1[usethis] < (lims[1] - 1e-8)) & (f1[usethis] > (lims[0] + 1e-8))
    print ' '
    print '%s, %s'%(title_name, name)
    print '# usable, non-boundary sequences: %i'%(np.sum(usethis))
    print '# usable, non-boundary f1 values: %i'%(np.sum(non_boundary_f1))
    print 'PWM Z mean: %f, PWM Z std: %f, PWM Z median: %f, max |Z|: %f'%(np.mean(Z[usethis]), np.std(Z[usethis]), np.median(Z[usethis]), np.max(np.abs(Z[usethis])))
    print 'average KD error: %f, median KD error: %f'%(np.mean(val_std[usethis]), np.median(val_std[usethis]))
    print 'measurement mean: %f, measurement std: %f'%(np.nanmean(Z_null), np.nanstd(Z_null))
    print 'fraction epistatic: %f'%fraction_epistasis
    print 'Test for equal variance by Levene''s test:%e'%(levene(Z[usethis], Z_null)[1])
    print 'Rsquare: %f'%Rsquare
    print 'Repistasis: %f'%((1-Rsquare)*fraction_epistasis)
    print 'Rmeasurement: %f'%((1-Rsquare)*(1-fraction_epistasis))
    print 'percent Z >  1.64: %f '%(100 * (Z[usethis] > 1.64).sum()/float(usethis.sum()))
    print 'percent Z < -1.64: %f '%(100 * (Z[usethis] < -1.64).sum()/float(usethis.sum()))
    print 'max |Z|: %f'%np.nanmax(np.abs(Z[usethis]))
    
    if ax is not None:
        Z = Z[usethis]
        Z[Z<=-20] = -20
        Z[Z>=20] = 20
        Z_null[Z_null<=-20] = -20
        Z_null[Z_null>=20] = 20
        x = np.linspace(-20, 20, 31)
        y, xpos = np.histogram(Z, bins=x, normed=1)
        null_y, xpos = np.histogram(Z_null, bins=x, normed=1)
        x = np.linspace(-20, 20, 30)
        high_x = np.linspace(-20, 20, 10000)
        lw = 1
        if plot_null:
            ax.plot(high_x, norm.pdf(high_x), c=[0.,0.,0.], lw=lw, zorder=20, label =r'$N(0,1)$')
            ax.plot(x, null_y, c=[0.9,0.6,0.1], lw=lw, zorder=21, label ='error')

        if title_name == '1H':
            ax.plot(x, y, c=[0.,0.,0.8], lw=lw, zorder=22, label=title_name)
        elif title_name == '3H':
            ax.plot(x, y, c=[0.8,0.,0.], lw=lw, zorder=22, label=title_name)
        else:
            ax.plot(x, y, c=[0.,0.8,0.], lw=lw, zorder=22, label=title_name)

        ax.set_xlabel(name)
        ax.tick_params(axis='y', which='major', pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_yticks([0,0.15,0.3,0.45])
        if make_ytick:
            ax.set_ylabel(r'PDF', labelpad=0)
        else:
            ax.set_yticklabels([])
        leg = ax.legend(loc='upper left',frameon=False, borderaxespad=0, handlelength=1., handletextpad=0.5, labelspacing=0.2)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(lw)

        ax.set_ylim([0,0.6])
        ax.set_xlim([-20,20])

def plot_sign_epistasis_example(ax):
    ax.plot([0,1],[0, 1], c=[0.3,0.3,0.3], lw=2)
    ax.plot([0,1],[0, 0.5], c=[0.3,0.3,0.3], lw=2)
    
    ax.plot([1,2],[1, 1.5], c=[0.3,0.3,0.3], lw=2)
    ax.plot([1,2],[0.5, 1.5], c=[0.3,0.3,0.3], lw=2)
    
    ax.plot([1,2],[1, 1.1], c=[0.8,0.,0.], lw=2)
    ax.plot([1,2],[0.5, 1.1], c=[0.8,0.,0.], lw=2)
    
    ax.plot([1,2],[1, 1.9], c=[0.0,0.0,0.8], lw=2)
    ax.plot([1,2],[0.5, 1.9], c=[0.0,0.0,0.8], lw=2)

    size = 200
    ax.scatter(0,0,s=size, c=[0.7,0.7,0.7], zorder=10)
    ax.scatter(1,1, c=[0.1,0.1,0.1],s=size, zorder=10)
    ax.scatter(1,0.5, c=[0.1,0.1,0.1],s=size, zorder=10)
    ax.scatter(2,1.5, c=[0.0,0.0,0.0],s=size/2., zorder=10)
    ax.scatter(2,1.9, c=[0.,0.,0.8],s=size/2., zorder=10)
    ax.scatter(2,1.1, c=[0.8,0.,0.],s=size/2., zorder=10)
    ax.scatter(2,0.66, s=size/2., c=[1.,0,1], zorder=10)
    ax.scatter(2,-0.25, s=size/2., c=[0.9,0.6,0.1], zorder=10)
    ax.text(2.2,0.66, 'sign epistasis', zorder=10, va='center')
    ax.text(2.2,-0.25, 'reciprocal\nsign epistasis', zorder=10, va='center')
    ax.text(2.2,1.5, 'PWM', zorder=10, va='center')
    ax.text(2.2,1.9, 'deleterious epistasis', zorder=10, va='center')
    ax.text(2.2,1.1, 'beneficial epistasis', zorder=10, va='center')

    ax.text(1,0.5, 'A', zorder=10, ha='center', va='center', color=[1,1,1])
    ax.text(1,1, 'B', zorder=10, ha='center', va='center', color=[1,1,1])


    ax.plot([1,2],[1,-0.25], c=[0.9,0.6,0.1], lw=2)
    ax.plot([1,2],[0.5,-0.25], c=[0.9,0.6,0.1], lw=2)

    ax.plot([1,2],[0.5,0.66], c=[1, 0.,1], lw=2)
    ax.plot([1,2],[1,0.66], c=[1.,0,1], lw=2)
    ax.set_yticks([])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['WT', 'single\nmutant','double\nmutant'])
    ax.set_ylabel('binding\nenergy')
    ax.set_ylim([-0.6,2.1])
    ax.set_xlim([-0.16,2.16])
    def simpleaxis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    simpleaxis(ax)

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
    x = np.sort((Z-np.mean(Z))/np.std(Z))
    num_muts = np.array(med_rep[['CDR1_muts','CDR3_muts']]).sum(axis=1).flatten()
    KD = np.array(med_rep['KD']).flatten()
    KD_std = np.array(med_rep['KD_std']).flatten()
    KD_use = ~np.array(med_rep['KD_exclude'])
    
    plot_epistasis_Z(A[KD_use], num_muts[KD_use], KD[KD_use], KD_std[KD_use], Z, '', 'All', KD_lims, None)
    
    opt1 = [2, 3]
    opt3 = [1, 2, 6, 8]
    
    Z_by_pos1 = calculate_Z_epistasis_by_pos(A1, num_muts1, KD1, KD1_std, pos1, KD_use1, KD_lims)
    Z_by_pos3 = calculate_Z_epistasis_by_pos(A3, num_muts3, KD3, KD3_std, pos3, KD_use3, KD_lims)
    
    print 'kolmogorov smirnov test of normality for log KD null distribution: %f'%(kstest(x,'norm')[1])
    
    plt.ion()
    plt.close('all')
    
    figsize=(7.3*0.7,5.2)
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
    
    ax = plt.subplot(gs[6:10,0:22])
    plot_epistasis_Z(A1[KD_use1], num_muts1[KD_use1], KD1[KD_use1], KD1_std[KD_use1], Z, r'', '1H', KD_lims, ax, make_ytick=True, plot_null=True)
    labeler.label_subplot(ax,'B')
    
    plot_epistasis_Z(A3[KD_use3], num_muts3[KD_use3], KD3[KD_use3], KD3_std[KD_use3], Z, r'Z', '3H', KD_lims, ax, make_ytick=True)
    ax.set_yscale('symlog',linthreshy=1e-2, linscaley=0.2)
    
    ax = plt.subplot(gs[11:16,0:8])
    plot_Z_epistasis_by_pos(Z_by_pos1, 20, 28, 3, ax, curr_title = '1H', opt=opt1, make_ylabel=True)
    labeler = Labeler(xpad=0.04,ypad=0.01,fontsize=14)
    labeler.label_subplot(ax,'C')
    
    ax = plt.subplot(gs[11:16,12:20])
    plot_Z_epistasis_by_pos(Z_by_pos3, 20, 100, 3, ax, curr_title = '3H', opt=opt3, make_colorbar=True)
    
    ax = plt.subplot(gs[0:5,0:13])
    plot_sign_epistasis_example(ax)
    labeler.label_subplot(ax,'A')
    
    ax = plt.subplot(gs[0:15,26:])
    labeler = Labeler(xpad=-0.01,ypad=-0.0, fontsize=14)
    labeler.label_subplot(ax,'D')
    CDR1_del = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'ALL', ax=ax, make_colorbar=False, epistasis='beneficial', y_offset=4)
    CDR3_del = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', ax=ax, make_colorbar=True, epistasis='beneficial')

    plt.savefig('figure_2.pdf')
    plt.close()
