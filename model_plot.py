#!/usr/bin/env python
from matplotlib.ticker import MaxNLocator
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas
from helper import *
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
from scipy.stats import norm


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
mpl.font_manager.FontProperties(family = 'Helvetica')

def fit_ave_epistasis_by_pos(A, num_muts, val, AA, seq, lims):
    np.random.seed(1)
    f = (np.array(val)).flatten()
    ind_zero = np.where(np.array(num_muts==0))[0]
    wt_val = val[ind_zero]

    f1, x = get_f1(A, num_muts, f, wt_val, limit =lims)
    usable = np.isfinite(f1) & np.isfinite(f)& (num_muts>1)
    usable = np.where(usable)[0]
    my_p = np.random.permutation(usable.shape[0]).tolist()
    delta = int(usable.shape[0]/10)
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
    print(Rsquare)
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
    for boundary in np.unique(in_pd['boundary'].dropna().tolist())[::-1]:
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
            
            delta_surf = ((fraction_boundary[usethis2]) - (fraction_boundary[usethis]))/bootstrap_J
            dmu2_dJ =  np.nanmean(delta_surf)
            SEdmu2_dJ = np.nanstd(delta_surf, ddof=-1) / np.sqrt(np.sum(usethis2))
            surf_0['dm2_dJ'].append(dmu2_dJ)
            surf_0['sigma_dm2_dJ'].append(SEdmu2_dJ)
            surf_0['num_muts'].append(num_muts)
            surf_0['boundary'].append(boundary)

    vol_0 = pandas.DataFrame(vol_0)
    surf_0 = pandas.DataFrame(surf_0)
    print(norm.cdf(-vol_0['dm2_dJ']/vol_0['sigma_dm2_dJ']))
    print(norm.cdf(-surf_0['dm2_dJ']/surf_0['sigma_dm2_dJ']))
    
    return vol_0, surf_0

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def summary_plot(surf, vol, ax1, ax2, colors, title=[''], make_colorbar=False, ylabels=False):
    V1 = vol.pivot(index='num_muts', columns='boundary', values='mu1')
    V1_sigma = vol.pivot(index='num_muts', columns='boundary', values='sigma1')

    V2 = vol.pivot(index='num_muts', columns='boundary', values='mu2')
    V2_sigma = vol.pivot(index='num_muts', columns='boundary', values='sigma2')

    S1 = surf.pivot(index='num_muts', columns='boundary', values='mu1')
    S1_sigma = surf.pivot(index='num_muts', columns='boundary', values='sigma1')

    S2 = surf.pivot(index='num_muts', columns='boundary', values='mu2')
    S2_sigma = surf.pivot(index='num_muts', columns='boundary', values='sigma2')

    for num_muts in [10]:
        ax = ax1

        x = np.array(V1.loc[num_muts])
        y = np.array(V2.loc[num_muts])
        xs = np.array(V1_sigma.loc[num_muts])
        ys = np.array(V2_sigma.loc[num_muts])
        boundary = np.array(list(V1.keys()))
        cax = ax.plot(10**boundary, x, lw=1, c=colors[0], label=title + r', PWM')
        cax = ax.plot(10**boundary, y, lw=1, c=colors[1], label=title + r', pair')

        cax = ax.errorbar(10**boundary, x, yerr=np.sqrt(xs**2+ys**2), fmt='.', c=colors[0],markersize=0, capsize=0)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=colors[1],markersize=0, capsize=0)
        cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)

        ax.set_xscale('log')
        ax.set_yscale('log')
        scale_min = np.min(np.hstack((x,y)))
        scale_max = np.max(np.hstack((x,y)))
        scale_min = 10**(np.floor(np.log10(scale_min)))
        scale_max = 10**(np.ceil(np.log10(scale_max)))

        ax.set_xlim([1e-9, 1e-6])
        #ax.set_ylim([1e4,1e8])

        ax.set_yticks([1e4,1e6,1e8])
        ax.set_xticks([1e-9,1e-8,1e-7,1e-6])
        if not ylabels:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$V$')

        ax = ax2
        x = np.array(S1.loc[num_muts])
        y = np.array(S2.loc[num_muts])
        xs = np.array(S1_sigma.loc[num_muts])
        ys = np.array(S2_sigma.loc[num_muts])
        boundary = np.array(list(S1.keys()))

        cax = ax.plot(10**boundary, x, lw=1, c=colors[0], label=title + r', PWM')
        cax = ax.plot(10**boundary, y, lw=1, c=colors[1], label=title + r', pair')

        cax = ax.errorbar(10**boundary, x, yerr=xs, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        cax = ax.errorbar(10**boundary, y, yerr=ys, fmt='.', c=[0,0,0],markersize=0, capsize=0, zorder=21)
        ax.set_xlabel(r'$K_d$ [M]')
        ax.set_xscale('log')
        ax.set_yscale('linear')

        ax.xaxis.set_major_locator(MaxNLocator(4))
        scale_min = np.min(np.hstack((x,y)))
        scale_max = np.max(np.hstack((x,y)))
        scale_min = np.floor(scale_min*20)/20.
        scale_max = np.ceil(scale_max*20)/20.
        ax.set_yticks([0.7,0.8,0.9])
        ax.set_xticks([1e-9,1e-8,1e-7,1e-6])

        if not ylabels:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$A/V$')

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
num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
KD_use1 = ~np.array(med_rep['KD_exclude'].loc[usethis1])
KD_use3 = ~np.array(med_rep['KD_exclude'].loc[usethis3])

print('# of triple mutants, CDR1: %i, CDR3:%i'%(np.sum(num_muts1==3),np.sum(num_muts3==3)))
print('# of triple mutants within boundary, CDR1: %i, CDR3:%i'%(np.sum(KD_use1[num_muts1==3]),np.sum(KD_use3[num_muts3==3])))
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

