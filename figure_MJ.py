from structure_connections import distances_ave
from data_preparation_transformed import get_f1, get_data
from get_fit_PWM_transformation import get_transformations
import itertools
import numpy as np
import matplotlib.pyplot as plt
from labeler import Labeler
from scipy.stats import spearmanr, linregress
import pylab
import matplotlib as mpl
import pdb
import pandas
from sklearn.linear_model import lasso_path, Lasso
from scipy.stats import pearsonr
from figure_1 import plot_epistasis

np.random.seed(0)
wt_seq = 'TFSDYWMNWVGSYYGMDYWG'



def make_A_matrix(distances, seq2A, seq, ref, max_size):
    out = np.zeros(max_size)

    for ii in range(10):
        for jj in range(10):
            if (seq[ii]!=ref[ii]) and (seq[jj]!=ref[jj]):
                if distances[ii,jj]>0:
                    if (seq[ii] + seq[jj]) in seq2A:
                        out[seq2A[seq[ii] + seq[jj]]] += 1
    return out

distances_ave = (distances_ave<6.5)*1.
distances_ave -= np.diag(np.diag(distances_ave, k=-1), k=-1)
distances_ave -= np.diag(np.diag(distances_ave, k=0), k=0)
distances_ave -= np.diag(np.diag(distances_ave, k=1), k=1)

AAs = 'ACDEFGHIKLMNPQRSTVWY'
count = 0
A2seq = {}
for ii, s1 in enumerate(AAs):
    for s2 in AAs[ii:]:
        A2seq[count] = s1+s2
        count+=1

seq2A = {A2seq[k]:k for k in A2seq}
logKD2PWM, PWM2logKD = get_transformations() #choose log transformation
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM) #get data
usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]

AA1 = [AA[ind] for ind in usethis1]
AA3 = [AA[ind] for ind in usethis3]


fit_A1 = np.array([make_A_matrix(distances_ave[:10,:10], seq2A, AA[ind], wt_seq[:10], count) for ind in usethis1])
fit_A3 = np.array([make_A_matrix(distances_ave[10:,10:], seq2A, AA[ind][10:], wt_seq[10:], count) for ind in usethis3])
fit_Blosum_A1 = np.array([make_A_matrix(np.ones((10,10)), seq2A, AA[ind], wt_seq[:10], count) for ind in usethis1])
fit_Blosum_A3 = np.array([make_A_matrix(np.ones((10,10)), seq2A, AA[ind][10:], wt_seq[10:], count) for ind in usethis3])

fit_A1[:, np.std(fit_A1, axis=0)==0] = 0
fit_A3[:, np.std(fit_A3, axis=0)==0] = 0

usethis1 = med_rep.index[usethis1]
usethis3 = med_rep.index[usethis3]

KD1 = np.array((med_rep['KD'].loc[usethis1]))
KD3 = np.array((med_rep['KD'].loc[usethis3]))

num_muts1 = np.array(med_rep['CDR1_muts'].loc[usethis1])
num_muts3 = np.array(med_rep['CDR3_muts'].loc[usethis3])

wt_val = med_rep['KD'].loc[wt_seq]

f1, x = get_f1(A1, num_muts1, KD1, wt_val, limit=KD_lims)
usethis = np.isfinite(KD1) & np.isfinite(f1) & (num_muts1 > 1)

fig, axes = plt.subplots(1,2, figsize=(7.3,3))

plt.subplots_adjust(
    bottom = 0.16,
    top = 0.93,
    left = 0.12,
    right = 0.88,
    hspace = 0.4,
    wspace = 0.7)


MJ_matrix = pandas.read_csv('miyazawa_jeringan.csv', header=0, index_col=0)
MJ_mat = np.zeros(count)
for ii in A2seq:
    s = A2seq[ii]
    MJ_mat[ii] = MJ_matrix[s[0]].loc[s[1]]

Alit1 = np.vstack((fit_A1.dot(np.ones(fit_A1.shape[1])),fit_A1.dot(MJ_mat))).T
myf = np.linalg.lstsq(Alit1[usethis], KD1[usethis] - f1[usethis], rcond=None)[0]
ax = axes[0]
f2 = Alit1.dot(myf) + f1
f2[f2>KD_lims[1]] = KD_lims[1]
f2[f2<KD_lims[0]] = KD_lims[0]
plot_epistasis(KD1, f2, num_muts1, KD_lims, ax, curr_title='1H', make_cbar=False, plot_ytick=True, plot_xtick=True, max_freq=3, min_freq=1)
ax.set_title('1H')
ax.set_ylabel(r'fit $K_d$ [M] (Miyazawa Jeringan)')
ax.set_xlabel(r'fit $K_d$, [M]')

f1, x = get_f1(A3, num_muts3, KD3, wt_val, limit=KD_lims)
usethis = np.isfinite(KD3) & np.isfinite(f1) & (num_muts3 > 1)

MJ_matrix = pandas.read_csv('miyazawa_jeringan.csv', header=0, index_col=0)
MJ_mat = np.zeros(count)
for ii in A2seq:
    s = A2seq[ii]
    MJ_mat[ii] = MJ_matrix[s[0]].loc[s[1]]

Alit3 = np.vstack((fit_A3.dot(np.ones(fit_A3.shape[1])),fit_A3.dot(MJ_mat))).T
myf = np.linalg.lstsq(Alit3[usethis], (KD3 - f1)[usethis], rcond=None)[0]
ax = axes[1]
f2 = Alit3.dot(myf) + f1
f2[f2>KD_lims[1]] = KD_lims[1]
f2[f2<KD_lims[0]] = KD_lims[0]
plot_epistasis(KD3, f2, num_muts3, KD_lims, ax, curr_title='3H', make_cbar=True, plot_ytick=True, plot_xtick=True, max_freq=3, min_freq=1)
ax.set_title('3H')
ax.set_xlabel(r'fit $K_d$, [M]')
plt.savefig('figure_MJ.pdf')
plt.close()

