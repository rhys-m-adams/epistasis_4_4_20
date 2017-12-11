from structure_connections import distances
from figure_z_epistasis_pos import Z_by_pos1, Z_by_pos3
import numpy as np
import matplotlib.pyplot as plt
from labeler import Labeler
from scipy.stats import spearmanr
import pylab
import matplotlib as mpl

distances[distances==0] = np.nan

plt.ion()
plt.close('all')
mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
mpl.font_manager.FontProperties(family = 'Helvetica')

figsize=(7.3,3.2)
rows = 1
cols = 2
fig, axes = plt.subplots(rows,cols,figsize=figsize)
plt.subplots_adjust(
    bottom = 0.15,
    top = 0.9,
    left = 0.1,
    right = 0.95,
    hspace = 0.4,
    wspace = 0.3)

# Make a labler to add labels to subplots
labeler = Labeler(xpad=0.05,ypad=0.02,fontsize=14)
ax = axes[0]
labeler.label_subplot(ax,'A')

x = distances[:10,:10].flatten()
y = Z_by_pos1.flatten()
usethis = np.isfinite(x) & np.isfinite(y)
ax.scatter(x,y,c=[0.5,0.5,1])
ax.text(0.2,0.85,r'$r=%.2f$'%(spearmanr(np.log(x[usethis]), y[usethis])[0]), transform=ax.transAxes)
ax.text(0.2,0.7,r'$p=%.3f$'%(spearmanr(np.log(x[usethis]), y[usethis])[1]), transform=ax.transAxes)
ax.set_xscale('log')
ax.set_ylim([0,20])
ax.set_xlim([1,100])
ax.set_ylabel(r'$\langle Z_{\rm{epi}}^2 \rangle^{\frac{1}{2}}$')
ax.set_xlabel(r'distance ($\AA$)')

ax.set_title('1H')

ax = axes[1]
labeler.label_subplot(ax,'B')

x = distances[10:,10:].flatten()
y = Z_by_pos3.flatten()
usethis = np.isfinite(x) & np.isfinite(y)
ax.scatter(x,y,c=[0.5,0.5,1])
ax.set_ylim([0,35])
ax.set_xlim([1,100])
ax.set_xlabel(r'distance ($\AA$)')
ax.text(0.2,0.85,r'$r=%.2f$'%(spearmanr(np.log(x[usethis]), y[usethis])[0]), transform=ax.transAxes)
ax.text(0.2,0.7,r'$p=%.3f$'%(spearmanr(np.log(x[usethis]), y[usethis])[1]), transform=ax.transAxes)
ax.set_xscale('log')
ax.set_title('3H')

plt.savefig('distance_epistasis.pdf')
plt.close()

plt.ion()
plt.close('all')
