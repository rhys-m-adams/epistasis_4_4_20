#!/usr/bin/env python
import pylab
from matplotlib.patches import Rectangle
import pdb
import matplotlib.pyplot as plt
from helper import *
from labeler import Labeler
from structure_connections import distances
import matplotlib.image as mpimg
from matplotlib import gridspec
from data_preparation_transformed import get_data, get_f1
from get_fit_PWM_transformation import get_transformations
import subprocess
import os
import matplotlib.patches as patches
from figure_2 import calculate_Z_epistasis_by_pos

logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

pylab.rcParams['font.size'] = 11
pos_cutoff = 2


def write_pymol_commands(deviance, cutoff, offset, in_max, fid):
    count = 0
    for ii in range(10):
        for jj in range(10):
            if (ii>jj) and (deviance[ii,jj] > cutoff):
                fid.write('select pk1, resi '+str(ii+offset)+' and n. CA and chain H\n')
                fid.write('select pk2, resi '+str(jj+offset)+' and n. CA and chain H\n')
                curr = deviance[ii,jj]
                if curr>in_max:
                    curr = in_max
                color = plt.cm.OrRd(int(curr*255./in_max))
                color = (color[0],color[1],color[2])
                #colormap(nameOfTheColormapYouWant)
                fid.write('draw_links pk1, pk2, color='+str(color)+', radius=0.2\n')
                count += 1
    print('domain at %i, num_sigificant at cutoff %f: %i'%(offset, cutoff, count))


def fuzzy_patch(left, bottom, width, height):
    for dx,dy in zip(np.linspace(0,width/8.,10),np.linspace(0,height/8,10)):
        p= patches.Rectangle(
            (left+dx, bottom+dy), width-dx*2, height-dy*2,
            alpha=0.15,
            facecolor="#ffffff",
            lw=0,
            transform=ax.transAxes,
            zorder=20)

        ax.add_patch(p)

def plot_Z_structure(cutoff, offsets, deviances, in_max, ax):
    fid=open('./pymol_Z_score.pml', 'w')
    #fid.write('view v1, store\n')
    fid.write('set line_width, 10\n')

    for deviance, offset in zip(deviances, offsets):
        write_pymol_commands(deviance, cutoff, offset, in_max, fid)

    fid.write('''set_view (\
        -0.769555032,    0.321604371,   -0.551684856,\
        -0.630766332,   -0.248065799,    0.735252023,\
         0.099607073,    0.913805008,    0.393757612,\
        -0.000041109,    0.000206268,  -68.325546265,\
        23.536064148,   21.614181519,   33.773040771,\
      -568286.937500000, 568423.000000000,  -20.000000000 )\n''')
    fid.write('ray 800,800\n')
    fid.write('png ./epistasis_Z_structure.png\n')
    fid.write('quit\n')
    fid.close()

    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(['pymol','-r','./draw_links.py','./Structure/epistasis3.pse', './pymol_Z_score.pml'], stdout=devnull, stderr=subprocess.STDOUT)

    img = mpimg.imread('epistasis_Z_structure.png')
    ax.imshow(img)
    cax = ax.scatter([0,0],[0,0], c=[0,20], cmap=plt.cm.OrRd, zorder=1, visible=False)

    p3 = ax.get_position().get_points()
    x00, y0 = p3[0]
    x01, y1 = p3[1]
    height = y1 - y0
    # [left, bottom, width, height]
    #position = ([x01+0.01, y0+0.08, 0.01, 0.25])
    position = ([0.01, 0.7, 0.04, 0.2])


    delta = np.max([int(in_max/5),1])
    ticks = list(range(0, 21, 4))
    cbar = plt.colorbar(cax, cax=plt.gcf().add_axes(position), orientation='vertical', ticks=ticks)
    cbar.set_label(r'$\langle Z^2 \rangle^\frac{1}{2}$', labelpad=-10)

    ticklabels = [r'$%i$'%(ii) for ii in ticks]
    ticklabels[-1]=r'$\geq$'+ticklabels[-1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)

    ax.text(0.03,0.55,'1H',transform=ax.transAxes, zorder=21,fontsize=14, color=[0,0,1])
    ax.text(0.8,0.75,'3H',transform=ax.transAxes, zorder=21,fontsize=14, color=[1,0,0])
    ax.text(0.6,0.9,'Fluorescein',transform=ax.transAxes, zorder=21,fontsize=14, color=[0,0.8,0])

    ax.text(0.3,0.19,r'T$_{28}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.17,0.22,r'F$_{29}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.05,0.35,r'S$_{30}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.18,0.53,r'D$_{31}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.33,0.52,r'Y$_{32}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.38,0.44,r'W$_{33}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.41,0.4,r'M$_{34}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.42 ,0.34,r'N$_{35}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.44,0.15,r'W$_{36}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.54,0.18,r'V$_{37}$', transform=ax.transAxes, zorder=21,fontsize=10)

    ax.text(0.51 ,0.36,r'G$_{100}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.44,0.46,r'S$_{101}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.45,0.57,r'Y$_{102}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.53,0.80,r'Y$_{103}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.7,0.79,r'G$_{104}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.81,0.59,r'M$_{105}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.81,0.44,r'D$_{106}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.81,0.3,r'Y$_{107}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.81,0.17,r'W$_{108}$', transform=ax.transAxes, zorder=21,fontsize=10)
    ax.text(0.62,0.01,r'G$_{109}$', transform=ax.transAxes, zorder=21,fontsize=10)

    fuzzy_patch(0.49, 0.77, 0.11, 0.1)
    fuzzy_patch(0.41, 0.32, 0.12, 0.1)#N35
    fuzzy_patch(0.50, 0.33, 0.13, 0.1)#G100
    #fuzzy_patch(0.39, 0.48, 0.11, 0.1)


usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]
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

Z_by_pos1 = calculate_Z_epistasis_by_pos(A1, num_muts1, KD1, KD1_std, pos1, KD_use1, KD_lims)
Z_by_pos3 = calculate_Z_epistasis_by_pos(A3, num_muts3, KD3, KD3_std, pos3, KD_use3, KD_lims)

Z_by_pos1_pos = calculate_Z_epistasis_by_pos(A1, num_muts1, KD1, KD1_std, pos1, KD_use1, KD_lims, epi_range=[0,np.inf])
Z_by_pos3_pos = calculate_Z_epistasis_by_pos(A3, num_muts3, KD3, KD3_std, pos3, KD_use3, KD_lims, epi_range=[0,np.inf])

Z_by_pos1_neg = calculate_Z_epistasis_by_pos(A1, num_muts1, KD1, KD1_std, pos1, KD_use1, KD_lims, epi_range=[-np.inf,0])
Z_by_pos3_neg = calculate_Z_epistasis_by_pos(A3, num_muts3, KD3, KD3_std, pos3, KD_use3, KD_lims, epi_range=[-np.inf,0])
if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    figsize=(3.5,3.5)
    rows = 2
    cols = 2
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02,0.02,1.03,1.03])
    cutoff = 3
    plot_Z_structure(cutoff, [28,100], [Z_by_pos1, Z_by_pos3], 20, ax)
    ax.axis('off')
    plt.savefig('figure_S11.pdf', dpi = 250)
    plt.close()
