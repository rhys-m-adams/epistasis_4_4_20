from structure_connections import distances, distances_a, distances_b, fl_d
from figure_S11 import Z_by_pos1, Z_by_pos3, Z_by_pos1_neg, Z_by_pos3_neg, Z_by_pos1_pos, Z_by_pos3_pos
import numpy as np
import matplotlib.pyplot as plt
from labeler import Labeler
from scipy.stats import spearmanr
import pylab
import matplotlib as mpl
import pdb
distances[distances==0] = np.nan

def plot_distance(x, y, ax):
    usethis = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x,y,c=[0.5,0.5,1])
    ax.text(0.7,0.85,r'$r=%.2f$'%(spearmanr(np.log(x[usethis]), y[usethis])[0]), transform=ax.transAxes)
    ax.text(0.7,0.7,r'$p=%.3f$'%(spearmanr(np.log(x[usethis]), y[usethis])[1]), transform=ax.transAxes)
    ax.set_xscale('log')
    ax.set_ylim([0,20])
    ax.set_xlim([1,100])

plt.ion()
plt.close('all')
mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
mpl.font_manager.FontProperties(family = 'Helvetica')

figsize=(7.3,10)
rows = 6
cols = 2

fig, axes = plt.subplots(rows,cols,figsize=figsize)
plt.subplots_adjust(
    bottom = 0.05,
    top = 0.95,
    left = 0.1,
    right = 0.95,
    hspace = 0.8,
    wspace = 0.3)

labeler = Labeler(xpad=0.05,ypad=0.02,fontsize=14)
xnames = ['min',r'$C_{\alpha}$',r'$C_{\beta}$','min','min']
ylabel_base = r'$\langle Z_{\rm{epi}}^2 \rangle^{\frac{1}{2}}$'
ynames = [ylabel_base,ylabel_base,ylabel_base, ylabel_base+'(positive)', ylabel_base+'(negative)']

distances = [distances, distances_a, distances_b, distances, distances]
epistasis = [(Z_by_pos1, Z_by_pos3),(Z_by_pos1, Z_by_pos3),(Z_by_pos1, Z_by_pos3), (Z_by_pos1_neg, Z_by_pos3_neg), (Z_by_pos1_pos, Z_by_pos3_pos)]

for ii, distance in enumerate(distances):
# Make a labler to add labels to subplots
    ax = axes[ii,0]
    labeler.label_subplot(ax, chr(ord('A')+ii))

    x = distance[:10,:10].flatten()
    y = epistasis[ii][0].flatten()
    plot_distance(x, y, ax)
    
    ax.set_ylabel(ynames[ii])
    ax.set_xlabel(r'%s distance ($\AA$)'%(xnames[ii]))
    if ii==0:
        ax.set_title('1H')

    ax = axes[ii,1]
    #labeler.label_subplot(ax,'B')

    x = distance[10:,10:].flatten()
    y = epistasis[ii][1].flatten()
    plot_distance(x, y, ax)

    ax.set_xlabel(r'%s distance ($\AA$)'%(xnames[ii]))
    if ii==0:
        ax.set_title('3H')

ax = axes[ii+1, 0]
labeler.label_subplot(ax, chr(ord('A')+ii+1))

plot_distance(fl_d[:10], np.sqrt(np.nanmean(Z_by_pos1**2, axis=0)), ax)
ax.set_xlabel('min distance to fluorescein')
ax.set_ylabel(r'$\langle Z_{\rm{epi}}^2 \rangle^{\frac{1}{2}}$')
ax = axes[ii+1, 1]
plot_distance(fl_d[10:], np.sqrt(np.nanmean(Z_by_pos3**2, axis=0)), ax)
ax.set_xlabel('distance to fluorescein')
ax.set_ylabel(r'$\langle Z_{\rm{epi}}^2 \rangle^{\frac{1}{2}}$')
#ax.scatter()
#print(spearmanr(fl_d[:10], np.nansum(Z_by_pos1, axis=0)))
#ax = axes[ii+1,1]
#ax.scatter(fl_d[10:], np.nansum(Z_by_pos3, axis=0))
#print(spearmanr(fl_d[10:], np.nansum(Z_by_pos3, axis=0)))

plt.savefig('figure_S12.pdf')
plt.close()
