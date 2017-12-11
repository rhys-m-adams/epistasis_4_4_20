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
from get_fit_PWM_transformation import get_transformations
from bayesian_lasso_emcee import bayesian_lasso
import itertools
import errno
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.useafm'] = True
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.font_manager.FontProperties(family = 'Helvetica')

pos_cutoff = 2

def fit_ave_epistasis_by_pos(A, num_muts, val, AA, seq, lims):
    np.random.seed(1)
    f = (np.array(val)).flatten()
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = val[ind_zero]

    f1, x = get_f1(A, num_muts, f, wt_val, limit =lims)
    usable = np.isfinite(f1) & np.isfinite(f)& (num_muts>1)
    usable = np.where(usable)[0]
    my_p = np.random.permutation(usable.shape[0]).tolist()
    delta = usable.shape[0]/10
    subsets = [usable[my_p[ind:(ind+delta)]] for ind in range(0,len(my_p), delta)]
    epi_energy, average_epi, Rsquare, penalties, epi_mu, epi_sigma, epi_p = fit_epistasis(A, num_muts, val, AA, seq, lims, subsets)
    return epi_energy, average_epi, Rsquare, penalties, epi_mu, epi_sigma, epi_p


def fit_lasso(full_A, f, f1, subsets, lims):
    A = np.array(full_A)
    SSE = 0
    y = f-f1
    usethis = np.isfinite(y)
    max_penalty = np.max(np.abs(-A[usethis].T.dot(y[usethis])) / np.sum(usethis)) * 10
    penalties = np.logspace(np.log10(max_penalty), np.log10(max_penalty)-6, 100)
    for jj in range(len(subsets)):
        test = subsets.pop(0)
        train = [item for sublist in subsets for item in sublist]
        A_train = A[train]
        y_train = y[train]
        usethis = np.isfinite(y_train)
        A_train = A_train[usethis]
        empty_A = full_A.sum(axis=0)>=3

        _ , coef_path, _ = lasso_path(A_train[:, empty_A], y_train[usethis], alphas=penalties)

        A_test = A[test]
        f1_test = f1[test]
        yhat = A_test[:,empty_A].dot(coef_path) + np.array([f1_test]).T
        yhat[yhat<(lims[0])] = lims[0]
        yhat[yhat>(lims[1])] = lims[1]
        residual = (yhat-np.array([f[test]]).T)**2
        SSE += np.sum(residual, axis=0)
        subsets.append(test)


    ind = np.argsort(SSE)[0]
    best_penalty = penalties[ind]
    clf = Lasso(alpha=best_penalty)
    usethis = list(itertools.chain.from_iterable(subsets))
    A_train = A[usethis]
    y_train = y[usethis]
    usethis = np.isfinite(y_train)
    A_train = A_train[usethis]
    y_train = y_train[usethis]
    empty_A = full_A.sum(axis=0)>=3

    clf.fit(A_train[:, empty_A], y_train)
    coeff = np.zeros(full_A.shape[1])
    coeff [empty_A]= clf.coef_
    return penalties, SSE, best_penalty, coeff

def make_linear_epistasis_model(AA, seq):
    A = []
    biochemistry = {'A':0, 'C':1,'D':2,'E':2,'F':0,'G':0,'H':3,'I':0,'K':3,'L':0,'M':0,'N':1,'P':0,'Q':1,'R':3,'S':1,'T':1,'V':0,'W':0,'Y':1}
    paired = np.array([[0,1,2,3],[1,4,5,6],[2,5,7,8],[3,6,8,9]], dtype=int)
    num_atoms = {'A':1, 'C':2,'D':4,'E':5,'F':7,'G':0,'H':5,'I':4,'K':5,'L':4,'M':4,'N':4,'P':3,'Q':5,'R':7,'S':2,'T':3,'V':3,'W':10,'Y':8}
    original_size = (40,40)

    def make_A(aa):
        curr = np.where([a != ref for a, ref in zip(aa, seq)])[0]
        tempA = np.zeros(original_size)
        for ii in curr:
            for jj in curr:
                if ii < jj:
                    tempA[ii * 4 + biochemistry[aa[ii]], jj * 4 + biochemistry[aa[jj]]] = 1
        return tempA

    for aa in AA:
        tempA =make_A(aa)
        A.append(tempA.flatten().tolist())

    A = np.array(A)
    return A

def fit_epistasis(A, num_muts, val, AA, seq, lims, subsets):
    original_size = (40,40)
    f = (np.array(val)).flatten()
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = val[ind_zero]
    f1, x = get_f1(A, num_muts, f, wt_val, limit =lims)

    A = make_linear_epistasis_model(AA, seq)
    penalties, SSE, best_penalty, coeff = fit_lasso(A, f, f1, subsets, lims)
    Rsquare = 1 - SSE/np.nansum((f[list(itertools.chain.from_iterable(subsets))]-np.nanmean(f[list(itertools.chain.from_iterable(subsets))]))**2)
    print Rsquare
    #pdb.set_trace()
    epi_energy = A.dot(coeff)

    def coeff_to_matrix(x, usethis, offset=0):
        out = np.zeros((A.shape[1])) + offset
        out[usethis] = x
        out = np.reshape(out, original_size)
        return out

    nonzero = coeff != 0
    average_epi = coeff_to_matrix(coeff[nonzero], nonzero)
    usethis = list(itertools.chain.from_iterable(subsets))
    A_bayesian = A[usethis]
    y_bayesian = (f-f1)[usethis]
    A_bayesian = A_bayesian[:, nonzero]
    epi_mu, epi_sigma, epi_p, sample_rate = bayesian_lasso(A_bayesian, y_bayesian, best_penalty, coeff[nonzero], burn_in=5000,num_iterations=50000, nwalkers = int(np.sum(nonzero)*2+2))
    epi_mu = coeff_to_matrix(epi_mu, nonzero)
    epi_sigma = coeff_to_matrix(epi_sigma, nonzero)
    epi_p = coeff_to_matrix(epi_p, nonzero, offset=1.)

    Rsquare = 1 - SSE/np.nansum((f[usethis] - np.nanmean(f[usethis]))**2)
    return epi_energy, average_epi, Rsquare, penalties, epi_mu, epi_sigma, epi_p


def plot_connections(average_epi, seq, offset, cutoff, ax, x_offset, visible=True, label_on=False):
    pi = np.pi
    colors = [[1,1,0],[0.5,0.5,0],[1,0.6,0.6],[0.6,0.6,1],[0.,0.8,0.0],[0.5,1,0.5],[0,1,1],[0.8,0.,0.],[0.7,0,0.7],[0.,0.,0.8]]
    colors1 = np.array([[0.8,0.6,0],[0,0.7,0],[1.,0.3,0.3],[0.3,0.3,1.]])
    paired = np.array([[0,1,2,3],[1,4,5,6],[2,5,7,8],[3,6,8,9]], dtype=int)
    theta = np.cumsum(np.linspace(-4.5,4.5,10))
    theta = np.linspace(5* np.pi/4., -1* np.pi/4., 10 )
    theta -= np.tan(theta)/90
    coords = np.array([np.cos(theta)+x_offset, np.sin(theta)]).T

    x = np.linspace(-1,1,100)
    y = x**2 - 1
    if label_on:
        ax.plot(coords[0,0],coords[0,1], c=colors1[2], zorder=0, label='acidic (DE)')
        ax.plot(coords[0,0],coords[0,1], c=colors1[3], zorder=0, label='basic (HKR)')
        ax.plot(coords[0,0],coords[0,1], c=colors1[0], zorder=0, label='nonpolar (AFGILMPVW)')
        ax.plot(coords[0,0],coords[0,1], c=colors1[1], zorder=0, label='polar (CNQSTY)')

    coeff = 3./np.max(np.abs(average_epi))
    out = np.zeros(10)
    rotate = lambda angle:np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    for ii in range(10):
        if visible:
            ax.scatter(coords[ii,0],coords[ii,1], s=350, c=[0.9,0.9,0.9], zorder=20, lw=0)
            ax.text(coords[ii,0],coords[ii,1], r'%i'%(offset+ii), ha='center', va='center', zorder=21)

        for jj in range(10):
            count = 1
            for kk in range(4):
                for ll in range(4):
                    if (np.abs(average_epi[ii*4+kk,jj*4+ll]) > cutoff) and visible:
                        radius = np.sqrt(np.sum(np.diff(coords[[ii,jj]], axis=0)**2))
                        center = np.array([np.mean(coords[[ii,jj],0]), np.mean(coords[[ii,jj],1])])
                        angle = float(np.arctan(np.diff(coords[[ii,jj],1]) / (np.diff(coords[[ii,jj],0])+1e-10)))

                        too_close = True
                        while too_close:
                            curvature = ((count/2)%6.5)/4. * (-1)**count
                            new_coords = np.array([x,y*curvature])*radius/2.
                            new_coords = rotate(angle).dot(new_coords)
                            new_coords += np.array([center]).T
                            count+=1
                            def get_dist(trajectory, compare):
                                dists = np.sqrt(np.sum((trajectory.T - np.array([compare]))**2, axis=1))

                                if np.max(np.sqrt(np.sum((trajectory.T - np.array([x_offset, 0]))**2, axis=1)))>1.05:
                                    return 0

                                return np.min(dists)

                            min_dist = np.min([get_dist(new_coords, coords[scan]) for scan in range(10) if (ii != scan) and (jj != scan)])
                            if min_dist > 0.35:
                                too_close = False

                        mid_color = (colors1[kk] + colors1[ll])/2.

                        ax.plot(new_coords[0][:45],new_coords[1][:45], c=colors1[kk], lw=coeff*np.abs(average_epi[ii*4+kk,jj*4+ll]))
                        ax.plot(new_coords[0][55:],new_coords[1][55:], c=colors1[ll], lw=coeff*np.abs(average_epi[ii*4+kk,jj*4+ll]))

                        for mm in range(40,60):
                            p = (59 - mm)/19.
                            curr_color = colors1[kk]*p + colors1[ll]*(1-p)
                            ax.plot(new_coords[0][mm:(mm+2)],new_coords[1][mm:(mm+2)], c=curr_color, lw=coeff*np.abs(average_epi[ii*4+kk,jj*4+ll]))

                        out[paired[kk,ll]] += 1

    ax.set_ylim([-1.2,1.3])
    return out

def get_stats(in_pd):
    possible_muts = np.cumsum(np.unique(np.array(in_pd['predicted_total'])))
    vol_0 = {'mu1':[], 'sigma1':[],'mu2':[], 'sigma2':[],'boundary':[],'num_muts':[],'dm2_dJ':[],'sigma_dm2_dJ':[]}
    surf_0 = {'mu1':[], 'sigma1':[],'mu2':[], 'sigma2':[],'boundary':[],'num_muts':[],'dm2_dJ':[],'sigma_dm2_dJ':[]}
    ind = np.argsort(in_pd['seed'])
    in_pd = in_pd.loc[ind]
    for boundary in np.unique(in_pd['boundary'].tolist())[::-1]:
        for num_muts in [10]:
            if boundary <= -9.5:
                continue
            try:
                usethis = (in_pd['num_muts'] == num_muts) & (in_pd['boundary']==boundary)
                in_csv = in_pd.loc[usethis].drop_duplicates(subset=['seed','num_J','boundary','num_muts'])
            except:
                continue


            fraction_in = np.array(in_csv['num_viable']).flatten()
            fraction_boundary = np.array(in_csv['exit_percent'], dtype =float).flatten()
            num_Js = np.array(in_csv['num_J']).flatten()
            usethis = num_Js==0
            bootstrap_J = np.unique(num_Js)[-1]
            usethis2 = num_Js==bootstrap_J
            common_seed = np.intersect1d(in_csv['seed'].loc[usethis], in_csv['seed'].loc[usethis2])
            usethis &= in_csv['seed'].isin(common_seed)
            usethis2 &= in_csv['seed'].isin(common_seed)

            if np.nanmean(fraction_in[usethis]) == 0:
                continue

            def bootstrap_var(x):
                n = x.shape[0]
                inds = np.arange(x.shape[0])
                xhat = np.array([np.sum(x[inds!=ii])/(n-1) for ii in inds])
                mu = np.mean(x)

                s_var = (n - 1.) / n * np.sum((xhat-mu) ** 2)
                return s_var

            mu1 = np.nanmean((fraction_in[usethis]))
            SE1 = np.sqrt(bootstrap_var(fraction_in[usethis]))

            vol_0['mu1'].append( mu1 )
            vol_0['sigma1'].append( SE1 )

            mu2 = np.nanmean(fraction_in[usethis2])
            SE2 = np.sqrt(bootstrap_var(fraction_in[usethis2]))

            vol_0['mu2'].append( mu2 )
            vol_0['sigma2'].append( SE2 )

            dmu2_dJ = np.nanmean(((fraction_in[usethis2]) - (fraction_in[usethis]))/bootstrap_J)
            SEdmu2_dJ = np.nanstd(((fraction_in[usethis2]) - (fraction_in[usethis]))/bootstrap_J, ddof=-1) / np.sqrt(np.sum(usethis2))

            vol_0['dm2_dJ'].append(dmu2_dJ)
            vol_0['sigma_dm2_dJ'].append(SEdmu2_dJ)

            vol_0['num_muts'].append(num_muts)
            vol_0['boundary'].append(boundary)
            ############################surface now
            mu1 = np.nanmean((fraction_boundary[usethis]))
            SE1 = np.sqrt(bootstrap_var(fraction_boundary[usethis]))

            surf_0['mu1'].append( (mu1) )
            surf_0['sigma1'].append( (SE1) )

            mu2 = np.nanmean(fraction_boundary[usethis2])
            SE2 = np.sqrt(bootstrap_var(fraction_boundary[usethis2]))
            surf_0['mu2'].append( mu2 )
            surf_0['sigma2'].append( SE2 )

            dmu2_dJ =  np.nanmean(((fraction_boundary[usethis2]) - (fraction_boundary[usethis]))/bootstrap_J)
            SEdmu2_dJ = np.nanstd(((fraction_boundary[usethis2]) - (fraction_boundary[usethis]))/bootstrap_J, ddof=-1) / np.sqrt(np.sum(usethis2))

            surf_0['dm2_dJ'].append(dmu2_dJ)
            surf_0['sigma_dm2_dJ'].append(SEdmu2_dJ)
            surf_0['num_muts'].append(num_muts)
            surf_0['boundary'].append(boundary)



    vol_0 = pandas.DataFrame(vol_0)
    surf_0 = pandas.DataFrame(surf_0)
    return vol_0, surf_0

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def summary_plot(surf, vol, ax1, ax2, colors, title=[''], make_colorbar=False, ylabels=False, make_legend=False):
    V1 = vol.pivot(index='num_muts', columns='boundary', values='mu1')
    V1_sigma = vol.pivot(index='num_muts', columns='boundary', values='sigma1')

    V2 = vol.pivot(index='num_muts', columns='boundary', values='mu2')
    V2_sigma = vol.pivot(index='num_muts', columns='boundary', values='sigma2')

    #dV_dJ = vol.pivot(index='num_muts', columns='boundary', values='dm2_dJ')
    #dV_dJ_sigma = vol.pivot(index='num_muts', columns='boundary', values='sigma_dm2_dJ')

    S1 = surf.pivot(index='num_muts', columns='boundary', values='mu1')
    S1_sigma = surf.pivot(index='num_muts', columns='boundary', values='sigma1')

    S2 = surf.pivot(index='num_muts', columns='boundary', values='mu2')
    S2_sigma = surf.pivot(index='num_muts', columns='boundary', values='sigma2')

    #dS_dJ = surf.pivot(index='num_muts', columns='boundary', values='dm2_dJ')
    #dS_dJ_sigma = surf.pivot(index='num_muts', columns='boundary', values='sigma_dm2_dJ')

    for num_muts in [10]:
        ax = ax1

        x = np.array(V1.loc[num_muts])
        y = np.array(V2.loc[num_muts])
        xs = np.array(V1_sigma.loc[num_muts])
        ys = np.array(V2_sigma.loc[num_muts])
        boundary = np.array(V1.keys())
        cax = ax.plot(10**boundary, x, lw=1, c=colors[0], label=title + r', PWM')
        cax = ax.plot(10**boundary, y, lw=1, c=colors[1], label=title + r', pair')

        cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=colors[0],markersize=0, capsize=0)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=colors[1],markersize=0, capsize=0)
        #cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=colors[0],markersize=0, capsize=0)
        #cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=colors[1],markersize=0, capsize=0)
        cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)

        #ax.set_xlabel(r'$K_D$ [M]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        scale_min = np.min(np.hstack((x,y)))
        scale_max = np.max(np.hstack((x,y)))
        scale_min = 10**(np.floor(np.log10(scale_min)))
        scale_max = 10**(np.ceil(np.log10(scale_max)))

        ax.set_xlim([1e-9, 1e-6])
        ax.set_ylim([1e4,1e8])
        #ax.plot([0,1],[0,1],'--', c=[0.3,0.3,0.3], transform=ax.transAxes)
        if make_legend:
            #leg = ax.legend(loc='upper left',frameon=False, scatterpoints=1, borderaxespad=0, handlelength=1., handletextpad=0)
            print 'nevermore'
        '''if make_colorbar:
            ticks = [-9,-8,-7,-6]

            p3 = ax.get_position().get_points()
            x00, y0 = p3[0]
            x01, y1 = p3[1]

            midpoint = (y1+y0)/2.
            height = y1-y0
            # [left, bottom, width, height]
            position = ([x01+0.02, midpoint - height/3, 0.01, 2 * height/3.])
            #cbar = plt.colorbar(im, orientation='vertical', ticks=lvls2)
            cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(['$10^{%i}$'%(logKD) for logKD in ticks])


            #cbar.ax.set_yticklabels([r'$10^{%d}$'%np.log10(t) for t in lvls2])
            cbar.set_label(r'boundary $K_D$ [M]',labelpad=2)
        '''
        ax.set_yticks([1e4,1e6,1e8])
        ax.set_xticks([1e-9,1e-8,1e-7,1e-6])
        #ax.set_title(title)
        #ax.set_aspect(1)
        if not ylabels:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$V$')

        ax = ax2
        x = np.array(S1.loc[num_muts])
        y = np.array(S2.loc[num_muts])
        xs = np.array(S1_sigma.loc[num_muts])
        ys = np.array(S2_sigma.loc[num_muts])
        boundary = np.array(S1.keys())

        cax = ax.plot(10**boundary, x, lw=1, c=colors[0], label=title + r', PWM')
        cax = ax.plot(10**boundary, y, lw=1, c=colors[1], label=title + r', pair')

        #cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=colors[0],markersize=0, capsize=0)
        #cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=colors[1],markersize=0, capsize=0)
        cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        ax.set_xlabel(r'$K_D$ [M]')
        ax.set_xscale('log')
        ax.set_yscale('linear')

        ax.xaxis.set_major_locator(MaxNLocator(4))
        #ax.yaxis.set_major_locator(MaxNLocator(4))
        scale_min = np.min(np.hstack((x,y)))
        scale_max = np.max(np.hstack((x,y)))
        scale_min = np.floor(scale_min*20)/20.
        scale_max = np.ceil(scale_max*20)/20.
        #ax.set_aspect(1)
        ax.set_yticks([0.7,0.8,0.9])
        ax.set_xticks([1e-9,1e-8,1e-7,1e-6])

        if not ylabels:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$A/V$')

        #ax.set_xlim([0.65,0.95])
        ax.set_ylim([0.65,0.95])
        ax.set_xlim([1e-9, 1e-6])


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
KD1_err = np.array((med_rep['KD_err'].loc[usethis1]))
KD3_err = np.array((med_rep['KD_err'].loc[usethis3]))
E1 = np.array((med_rep['E'].loc[usethis1]))
E3 = np.array((med_rep['E'].loc[usethis3]))
E1_err = np.array((med_rep['E_err'].loc[usethis1]))
E3_err = np.array((med_rep['E_err'].loc[usethis3]))
num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
KD_use1 = ~np.array(med_rep['KD_exclude'].loc[usethis1])
KD_use3 = ~np.array(med_rep['KD_exclude'].loc[usethis3])

print '# of triple mutants, CDR1: %i, CDR3:%i'%(np.sum(num_muts1==3),np.sum(num_muts3==3))
print '# of triple mutants within boundary, CDR1: %i, CDR3:%i'%(np.sum(KD_use1[num_muts1==3]),np.sum(KD_use3[num_muts3==3]))
mkdir('./biochemical_fit')
try:
    KD_epi_1 = np.array(pandas.read_csv('./biochemical_fit/KD_epi_1.csv',index_col=0))
    KD_average_epi_1= np.array(pandas.read_csv('./biochemical_fit/KD_average_epi_1.csv',index_col=0))
    Rsquare1= np.array(pandas.read_csv('./biochemical_fit/Rsquare1.csv',index_col=0))
    penalties1= np.array(pandas.read_csv('./biochemical_fit/penalties1.csv',index_col=0))
    epi_mu1= np.array(pandas.read_csv('./biochemical_fit/epi_mu1.csv',index_col=0))
    epi_sigma1= np.array(pandas.read_csv('./biochemical_fit/epi_sigma1.csv',index_col=0))
    epi_p1 = np.array(pandas.read_csv('./biochemical_fit/epi_p1.csv',index_col=0))

except:
    KD_epi_1, KD_average_epi_1, Rsquare1, penalties1, epi_mu1, epi_sigma1, epi_p1 = fit_ave_epistasis_by_pos(A1, num_muts1, KD1, AA1, cdr1_list, KD_lims)
    pandas.DataFrame(KD_epi_1).to_csv('./biochemical_fit/KD_epi_1.csv')
    pandas.DataFrame(KD_average_epi_1).to_csv('./biochemical_fit/KD_average_epi_1.csv')
    pandas.DataFrame(Rsquare1).to_csv('./biochemical_fit/Rsquare1.csv')
    pandas.DataFrame(penalties1).to_csv('./biochemical_fit/penalties1.csv')
    pandas.DataFrame(epi_mu1).to_csv('./biochemical_fit/epi_mu1.csv')
    pandas.DataFrame(epi_sigma1).to_csv('./biochemical_fit/epi_sigma1.csv')
    pandas.DataFrame(epi_p1).to_csv('./biochemical_fit/epi_p1.csv')

try:
    KD_epi_3 = np.array(pandas.read_csv('./biochemical_fit/KD_epi_3.csv',index_col=0))
    KD_average_epi_3= np.array(pandas.read_csv('./biochemical_fit/KD_average_epi_3.csv',index_col=0))
    Rsquare3= np.array(pandas.read_csv('./biochemical_fit/Rsquare3.csv',index_col=0))
    penalties3= np.array(pandas.read_csv('./biochemical_fit/penalties3.csv',index_col=0))
    epi_mu3= np.array(pandas.read_csv('./biochemical_fit/epi_mu3.csv',index_col=0))
    epi_sigma3= np.array(pandas.read_csv('./biochemical_fit/epi_sigma3.csv',index_col=0))
    epi_p3 = np.array(pandas.read_csv('./biochemical_fit/epi_p3.csv',index_col=0))

except:
    KD_epi_3, KD_average_epi_3, Rsquare3, penalties3, epi_mu3, epi_sigma3, epi_p3 = fit_ave_epistasis_by_pos(A3, num_muts3, KD3, AA3, cdr3_list, KD_lims)
    pandas.DataFrame(KD_epi_3).to_csv('./biochemical_fit/KD_epi_3.csv')
    pandas.DataFrame(KD_average_epi_3).to_csv('./biochemical_fit/KD_average_epi_3.csv')
    pandas.DataFrame(Rsquare3).to_csv('./biochemical_fit/Rsquare3.csv')
    pandas.DataFrame(penalties3).to_csv('./biochemical_fit/penalties3.csv')
    pandas.DataFrame(epi_mu3).to_csv('./biochemical_fit/epi_mu3.csv')
    pandas.DataFrame(epi_sigma3).to_csv('./biochemical_fit/epi_sigma3.csv')
    pandas.DataFrame(epi_p3).to_csv('./biochemical_fit/epi_p3.csv')

if __name__ == '__main__':
    print '# of found epistatic terms from biochemical model, CDR1H: %i, CDR3H: %i'%(np.sum(KD_average_epi_1!=0),np.sum(KD_average_epi_3!=0))
    A_opt1 = make_linear_epistasis_model(['TFGHYWMNWV'], cdr1_list)
    A_opt3 = make_linear_epistasis_model(['GASYGMEYLG'], cdr3_list)
    epi_contributions1 = (A_opt1 * (KD_average_epi_1.flatten())).flatten()
    epi_contributions3 = (A_opt3 * (KD_average_epi_3.flatten())).flatten()
    usethis1 = np.where(epi_contributions1)[0]
    usethis3 = np.where(epi_contributions3)[0]
    print 'Epistatic contribution to OPT CDR1H domain: '+ str(epi_contributions1[usethis1])
    print 'Epistatic contribution to OPT CDR3H domain: '+ str(epi_contributions3[usethis3])
    print 'Epistatic contribution to OPT CDR1H domain: '+ str(np.sum(epi_contributions1[usethis1]))
    print 'Epistatic contribution to OPT CDR3H domain: '+ str(np.sum(epi_contributions3[usethis3]))

    print 'sum |epistatic contribution| CDR1: %f'%(np.sum(np.abs(KD_average_epi_1)))
    print 'sum |epistatic contribution| CDR3: %f'%(np.sum(np.abs(KD_average_epi_3)))
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


    ax = plt.subplot(gs[0:5,0:(num_x/2)])
    CDR1_pos_connections = plot_connections(get_sign_model(KD_average_epi_1,1) * (epi_p1<(5e-2)), cdr1_list, 28,0., ax,0)
    ax.axis('off')
    # Make a labler to add labels to subplots
    labeler = Labeler(xpad=-0.03,ypad=0.01,fontsize=14)
    labeler.label_subplot(ax,'A')
    ax.set_title('beneficial')
    CDR3_pos_connections = plot_connections(get_sign_model(KD_average_epi_3,1) * (epi_p3<(5e-2)), cdr3_list, 100,0., ax,2.5)
    ax.set_xlim([-1.3,3.8])
    ax.axis('off')

    ax = plt.subplot(gs[6:,0:(num_x/2)])
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

    ax3 = plt.subplot(gs[1:4,(num_x*3/5+1):])
    ax4 = plt.subplot(gs[6:9,(num_x*3/5+1):])
    cdr1_results = pandas.read_csv('cdr1h.csv', header=0)
    vol_0, surf_0 = get_stats(cdr1_results)
    summary_plot(surf_0, vol_0, ax3, ax4, colors=['#000080','#8080FF'], title ='1H', ylabels=True)

    labeler = Labeler(xpad=0.09,ypad=0.01, fontsize=14)

    plt.subplot(gs[0,(num_x*3/5+1):]).set_visible(False)
    labeler.label_subplot(plt.subplot(gs[0,(num_x*3/5+1):]),'C')
    labeler.label_subplot(ax4,'D')

    cdr3_results = pandas.read_csv('cdr3h.csv', header=0)
    vol_0, surf_0 = get_stats(cdr3_results)
    summary_plot(surf_0, vol_0, ax3, ax4, colors=['#800000','#FF8080'], title ='3H', make_colorbar=False, ylabels=True, make_legend=True)
    leg = ax3.legend(loc='center', bbox_to_anchor=(0.5, 1.3),ncol=2, columnspacing=0.1, frameon=True, fancybox=True, scatterpoints=1, borderaxespad=0, handlelength=1., handletextpad=0.5)


    plt.savefig('figure_3.pdf')
    plt.close()
