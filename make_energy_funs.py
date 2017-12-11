import numpy as np
import pandas
import pdb
from data_preparation_transformed import get_data, get_f1, get_null
from get_fit_PWM_transformation import get_transformations
logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=logKD2PWM)

usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]
A2_1 = A2[usethis1]
A2_3 = A2[usethis3]
pos1 = pos[usethis1]%20
pos3 = pos[usethis3]%20
pos1 = pos1[:,:10]
pos3 = pos3[:,10:]
usethis1 = med_rep.index[usethis1]
usethis3 = med_rep.index[usethis3]
KD1 = np.array((med_rep['KD'].loc[usethis1]))
KD3 = np.array((med_rep['KD'].loc[usethis3]))
E1 = np.array((med_rep['E'].loc[usethis1]))
E3 = np.array((med_rep['E'].loc[usethis3]))
num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])
KD_use1 = ~np.array(med_rep['KD_exclude'].loc[usethis1])
KD_use3 = ~np.array(med_rep['KD_exclude'].loc[usethis3])

def make_heatmap(position, num_muts, value):
    usethis = num_muts == 1
    out = np.zeros((position.shape[1], 20))
    for pos, val in zip(position[usethis], value[usethis]):
        ind = np.where(pos)[0]
        out[ind, pos[ind]] = val
    return out

def make_heatmap2(position, num_muts, value):
    usethis = num_muts == 2
    out = np.zeros((position.shape[1], position.shape[1], 20, 20))*np.nan
    for pos, val in zip(position[usethis], value[usethis]):
        ind = np.where(pos)[0]
        out[ind[0], ind[1], pos[ind[0]], pos[ind[1]]] = val
        out[ind[1], ind[0], pos[ind[1]], pos[ind[0]]] = val
    return out

A_wt = KD1[num_muts1==0]
E_wt = E1[num_muts1==0]

A_heatmap1 = make_heatmap(pos1, num_muts1, KD1-A_wt)
A_heatmap3 = make_heatmap(pos3, num_muts3, KD3-A_wt)
E_heatmap1 = make_heatmap(pos1, num_muts1, E1-E_wt)
E_heatmap3 = make_heatmap(pos3, num_muts3, E3-E_wt)

def sub_heatmap2(pos, num_muts, delta_E1, delta_E2, dropout_ind):
    double_muts = np.where(num_muts==2)[0].tolist()
    double_muts.pop(dropout_ind)
    ind = np.array(double_muts)
    heatmap_2_1 = make_heatmap2(pos[ind], num_muts[ind], delta_E1[ind])
    heatmap_2_2 = make_heatmap2(pos[ind], num_muts[ind], delta_E2[ind])
    return heatmap_2_1, heatmap_2_2
    
    
f1 = get_f1(A1, num_muts1, KD1, A_wt, limit=[-9.5,-5])[0]
A_heatmap1_2 = make_heatmap2(pos1, num_muts1, KD1-f1)

f1 = get_f1(A3, num_muts3, KD3, A_wt, limit=[-9.5,-5])[0]
A_heatmap3_2 = make_heatmap2(pos3, num_muts3, KD3-f1)

f1 = get_f1(A1, num_muts1, E1, E_wt, limit=[-1,0.5])[0]
E_heatmap1_2 = make_heatmap2(pos1, num_muts1, E1-f1)

f1 = get_f1(A3, num_muts3, E3, E_wt, limit=[-1,0.5])[0]
E_heatmap3_2 = make_heatmap2(pos3, num_muts3, E3-f1)


