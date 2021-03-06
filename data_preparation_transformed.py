#!/usr/bin/env python
import pdb
from make_A import makeSparse
import pandas
from helper import *
#this function provides a uniform set of data across the different analyses,
#ensuring consistency

cdr1_wtseq = 'TFSDYWMNWV'
cdr3_wtseq = 'GSYYGMDYWG'
cdr1_list = [s for s in cdr1_wtseq]
cdr3_list = [s for s in cdr3_wtseq]
wt_aa = cdr1_wtseq + cdr3_wtseq


def get_data(transform = lambda x:x, exclude_boundary=True, KD_lims=[-9.5,-5], prefix='data/', replicate_use=None):
    #get_data(transform = lambda x:x, exclude_boundary=True, KD_lims=[-9.5,-5], prefix='data/')
    #transform - a function, typically like should it be log transformed (default)
    #or in KD (lambda x:10**x)
    #exclude_boundary - return a list of whethter the value is at the boundary_exceeded
    #KD_lims- define where the boundaries are
    #prefix - where the data is stored
    
    #get triplicate measurements
    rep1 = pandas.read_csv(prefix + 'replicate_1.csv')
    rep2 = pandas.read_csv(prefix + 'replicate_2.csv')
    rep3 = pandas.read_csv(prefix + 'replicate_3.csv')

    reps = [rep1, rep2, rep3]
    if not (replicate_use is None):
        reps = [reps[replicate_use]]
    #transform the limits to whatever unit we're working in
    KD_lims = [transform(KD_lims[0]), transform(KD_lims[1])]
    exp_lims = [-1, 0.5]
    
    #start by calculating WT expression to normalize the expression, since this can 
    #vary between triplicates
    wt_int = aa2int(wt_aa)
    hamming = lambda x, y: np.sum([a1!=a2 for a1, a2 in zip(x,y)])
    keys = []
    wt_mEs = []
    for rep in reps:
        rep.index = rep['CDR1H_AA'] + rep['CDR3H_AA']
        keys.extend(rep.index)
        wt_mEs.append(np.nanmedian(rep['expression'].loc[(rep['CDR1H_AA'] == cdr1_wtseq) & (rep['CDR3H_AA'] == cdr3_wtseq)]))

    wt_mEs = np.array(wt_mEs)
    keys = sorted(set(keys))
    KDs = []
    exp = []
    KD_std = []
    exp_std = []
    CDR1_muts = []
    CDR3_muts = []
    boundary = []
    fraction = []
    for key in keys:
        #for each amino acid sequence get all of its KD and expression measurements
        # over synonymous mutants and triplicates
        temp = []
        tempE = []
        temp_fraction = []
        for ii, rep in enumerate(reps):
            if key in rep.index:
                temp.extend(rep['fit_KD'].loc[rep.index.isin([key])].tolist())
                tempE.extend((rep['expression'].loc[rep.index.isin([key])]/wt_mEs[ii]).tolist())
                temp_fraction.append( np.sum(rep['fit_fraction'].loc[rep.index.isin([key])]) )

        
        #log transform data, note if its median is at the boundary
        temp = np.log10(np.array(temp))
        tempE = np.log10(np.array(tempE))
        usethis = np.isfinite(temp)
        temp = transform(temp[usethis])
        med_boundary = (np.nanmedian(temp) <= KD_lims[0]) or (np.nanmedian(temp) >= KD_lims[1]) and exclude_boundary
        nKD = np.sum(np.isfinite(temp))
        nE = np.sum(np.isfinite(tempE))
        out_KD = np.mean(temp)
        out_E = np.mean(tempE)
        KDs.append(out_KD)
        exp.append(out_E)
        if nKD>1:
            KD_std.append(np.nanstd(temp, ddof = 1)/np.sqrt(nKD))#record KD sample error as standard deviation
            boundary.append(med_boundary)
        else:
            KD_std.append(np.nan)
            boundary.append(med_boundary)
        if nE>1:
            exp_std.append(np.nanstd(tempE, ddof = 1)/np.sqrt(nE))#record Expression sample error as standard deviation
        else:
            exp_std.append(np.nan)
        
        
        CDR1_muts.append(hamming(key[:10], wt_aa[:10]))
        CDR3_muts.append(hamming(key[10:], wt_aa[10:]))
        fraction.append(np.mean(temp_fraction))

    out = pandas.DataFrame({'KD':KDs, 'E':exp,'KD_std':KD_std, 'E_std':exp_std, 'KD_exclude':boundary, 'CDR1_muts':CDR1_muts, 'CDR3_muts':CDR3_muts, 'fraction':fraction}, index=keys)
    lib_seq = np.array([aa2int(s) for s in keys])
    pos = np.array([aa2int(s) - wt_int for s in keys])
    A2, A = makeSparse(lib_seq, wt_int, 20)
    return out, pos, A, list(keys), A2, KD_lims, exp_lims


def get_data_ind(transform = lambda x:x, replicate_use=None):
    rep1 = pandas.read_csv('data/replicate_1.csv')
    rep2 = pandas.read_csv('data/replicate_2.csv')
    rep3 = pandas.read_csv('data/replicate_3.csv')

    reps = [rep1, rep2, rep3]
    if not (replicate_use is None):
        reps = [reps[replicate_use]]
    
    KD_lims = [transform(-9.5), transform(-5)]
    exp_lims = [-1, 0.5]

    wt_int = aa2int(wt_aa)
    hamming = lambda x, y: np.sum([a1!=a2 for a1, a2 in zip(x,y)])
    keys = []
    wt_mEs = []
    for rep in reps:
        rep.index = rep['CDR1H_AA'] + rep['CDR3H_AA']
        keys.extend(rep.index)
        wt_mEs.append(np.nanmedian(rep['expression'].loc[(rep['CDR1H_AA'] == cdr1_wtseq) & (rep['CDR3H_AA'] == cdr3_wtseq)]))

    wt_mEs = np.array(wt_mEs)
    keys = sorted(set(keys))
    KDs = []
    exp = []
    CDR1_muts = []
    CDR3_muts = []
    boundary = []
    AA = []
    for key in keys:
        for ii, rep in enumerate(reps):
            temp = []
            tempE = []
            if key in rep.index:
                temp.extend(rep['fit_KD'].loc[rep.index.isin([key])].tolist())
                tempE.extend((rep['expression'].loc[rep.index.isin([key])]/wt_mEs[ii]).tolist())

            temp = np.log10(np.array(temp))
            temp = transform(temp)

            tempE = np.log10(np.array(tempE))
            out_KD = (temp).tolist()
            out_E = (tempE).tolist()

            KDs.extend(out_KD)
            exp.extend(out_E)

            CDR1_muts.extend([hamming(key[:10], wt_aa[:10])]*temp.shape[0])
            CDR3_muts.extend([hamming(key[10:], wt_aa[10:])]*temp.shape[0])
            AA.extend([key]*temp.shape[0])

    out = pandas.DataFrame({'KD':KDs, 'E':exp, 'CDR1_muts':CDR1_muts, 'CDR3_muts':CDR3_muts}, index=AA)
    lib_seq = np.array([aa2int(s) for s in AA])
    pos = np.array([aa2int(s) - wt_int for s in AA])
    A2, A = makeSparse(lib_seq, wt_int, 20)
    return out, pos, A, AA, A2, KD_lims, exp_lims


def get_f1(A, num_muts, val, wt_val, limit=[]):
    ind = np.where(np.array(num_muts==1))[0]
    x = np.zeros(A.shape[1])*np.nan

    for ii in ind:
        curr = np.where(np.array(A[ii].todense()))[1]
        x[curr] = val[ii] - wt_val

    f1 = A.dot(x) + wt_val
    if len(limit)>0:
        usethis = np.isfinite(f1)
        sub_f1 = f1[usethis]
        sub_f1[sub_f1<=limit[0]] = limit[0]
        sub_f1[sub_f1>=limit[1]] = limit[1]
        f1[usethis] = sub_f1
    return f1, x


def get_null(transform = lambda x:x, exclude_boundary=True, KD_lims=[-9.5,-5], replicate_use=None, prefix='data/'):
    rep1 = pandas.read_csv(prefix + 'replicate_1.csv')
    rep2 = pandas.read_csv(prefix + 'replicate_2.csv')
    rep3 = pandas.read_csv(prefix + 'replicate_3.csv')

    all_reps = [rep1, rep2, rep3]
    if not (replicate_use is None):
        all_reps = [all_reps[replicate_use]]
    
    wt_int = aa2int(wt_aa)
    hamming = lambda x, y: np.sum([a1!=a2 for a1, a2 in zip(x,y)])
    keys = []
    wt_mEs = []
    for rep in all_reps:
        rep.index = rep['CDR1H_AA'] + rep['CDR3H_AA']
        keys.extend(rep.index)
        wt_mEs.append(np.nanmedian(rep['expression'].loc[(rep['CDR1H_AA'] == cdr1_wtseq) & (rep['CDR3H_AA'] == cdr3_wtseq)]))

    wt_mEs = np.array(wt_mEs)
    keys = sorted(set(keys))

    Z = []
    ZE = []
    for key in keys:
        temp = []
        tempE = []
        for ii, rep in enumerate(all_reps):
            curr_rep = rep.loc[rep.index.isin([key])]
            if curr_rep.shape[0]>1:
                temp.append(curr_rep['fit_KD'].tolist())
                tempE.append((curr_rep['expression']/wt_mEs[ii]).tolist())
        if (len(temp)==0) or (key == cdr1_wtseq+cdr3_wtseq):
            continue

        temp = np.log10(np.array(temp))
        if np.sum(np.isfinite(temp)) == 0:
            continue

        usethis = np.where(np.sum(np.isfinite(temp),axis=0) > 0)[0]
        for ind in usethis:
            if (np.nanmedian(temp[:,ind], axis=0) <= KD_lims[0]) or (np.nanmedian(temp[:,ind], axis=0) >= KD_lims[1]) and exclude_boundary:
                temp[:, ind] = np.nan

        usethis = np.isfinite(temp)
        temp[usethis] = transform(temp[usethis])
        tempE = np.log10(np.array(tempE))
        num_syn = temp.shape[1]

        nKD = np.isfinite(temp).sum(axis=0)
        nE = np.isfinite(tempE).sum(axis=0)
        dfKD = nKD-1
        dfE = nE-1
        out_KD = np.mean(temp, axis=0)
        out_E = np.mean(tempE, axis=0)
        KD_S = np.std(temp, axis=0, ddof = 1)
        E_S = np.std(tempE, axis=0, ddof = 1)
        for ii in range(0, num_syn-1):
            for jj in range(ii+1, num_syn):
                pooled = (KD_S[ii]**2/nKD[ii] + KD_S[jj]**2/nKD[jj])
                if pooled == 0:
                    continue
                
                Z.append( (out_KD[ii]-out_KD[jj])/np.sqrt(pooled))
                Z.append( -(out_KD[ii]-out_KD[jj])/np.sqrt(pooled))
                pooled = (E_S[ii]**2/nE[ii] + E_S[jj]**2/nE[jj])
                ZE.append((out_E[ii] -out_E[jj] )/np.sqrt(pooled))
                ZE.append(-(out_E[ii] -out_E[jj] )/np.sqrt(pooled))

    x = np.linspace(-30, 30, 61)
    Z = np.array(Z)
    Z = Z[np.isfinite(Z)]

    hist_Z = np.array(Z)
    hist_Z[hist_Z<=-30] = -30
    hist_Z[hist_Z>=30] = 30
    yK, xpos = np.histogram(Z, bins=x)
    yK = np.array(yK, dtype=float) / np.trapz(yK,(xpos[1:]+xpos[:-1])/2.)

    xK = np.linspace(-30, 30, 60)

    xE = np.linspace(-30, 30, 61)
    ZE = np.array(ZE)
    ZE = ZE[np.isfinite(ZE)]

    hist_Z = np.array(ZE)
    hist_Z[hist_Z<=-30] = -30
    hist_Z[hist_Z>=30] = 30

    yE, xpos = np.histogram(hist_Z, bins=x)
    yE = np.array(yE, dtype=float) / np.trapz(yE,(xpos[1:]+xpos[:-1])/2.)

    xE = np.linspace(-30, 30, 60)
    return xK, yK, xE, yE, Z, ZE
