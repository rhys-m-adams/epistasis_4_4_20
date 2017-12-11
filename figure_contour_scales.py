from data_preparation_transformed import get_data, get_f1
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import numpy as np
from labeler import Labeler
import pylab
from figure_1 import plot_epistasis

pylab.rcParams['font.size'] = 11
pos_cutoff = 2

plt.ion()
plt.close('all')
figsize=(7.3,3.2)
rows = 1
cols = 2
fig, axes = plt.subplots(rows,cols,figsize=figsize)
plt.subplots_adjust(
    bottom = 0.15,
    top = 0.9,
    left = 0.1,
    right = 0.85,
    hspace = 0.4,
    wspace = 0.6)

# Make a labler to add labels to subplots
labeler = Labeler(xpad=0.04,ypad=0.01,fontsize=17)

ax = axes[0]
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=lambda x:10**x)
num_muts = np.array(med_rep['CDR1_muts']) + np.array(med_rep['CDR3_muts'])
KD = np.array(med_rep['KD'])

wt_val = KD[num_muts==0]
f1, x = get_f1(A, num_muts, KD, wt_val, limit=KD_lims)
plot_epistasis(KD*1e6, f1*1e6, num_muts, np.array(KD_lims)*1e6, ax, plot_ytick=True, logscale=False, max_freq=1, make_cbar=False)
ax.set_xlabel(r'$K_D$, [M]')
ax.set_ylabel(r'PWM [M]')
ax.set_xticks([0,2,4,6,8,10])
ax.set_yticks([0,2,4,6,8,10])
ax.text(1.05,-0.15,r'$\times 10^{-6}$', transform=ax.transAxes)
ax.text(-0.05,1.03,r'$\times 10^{-6}$', transform=ax.transAxes)
labeler.label_subplot(ax,'A')

ax = axes[1]
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(transform=lambda x:x)
KD = np.array(med_rep['KD'])

wt_val = KD[num_muts==0]
f1, x = get_f1(A, num_muts, KD, wt_val, limit=KD_lims)
plot_epistasis(KD, f1, num_muts, KD_lims, ax, plot_ytick=True, max_freq=1, logscale=True, make_cbar=True)
ax.set_xlabel(r'$K_D$')
ax.set_ylabel(r'PWM [M]')
labeler.label_subplot(ax,'B')

plt.savefig('contour_scales.pdf')
plt.close()
