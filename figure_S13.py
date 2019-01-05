#!/usr/bin/env python
import pylab
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import pdb
from make_A import makeSparse
import matplotlib.pyplot as plt
import pandas
from helper import *
from scipy.stats import norm
from labeler import Labeler
from matplotlib.colors import LinearSegmentedColormap
from data_preparation_transformed import get_data, get_f1, cdr1_list, cdr3_list, wt_aa
from get_fit_PWM_transformation import get_transformations
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as PathEffects
from figure_2 import plot_KD_sign_epistasis

logKD2PWM, PWM2logKD = get_transformations()
med_rep, pos, A, AA, A2, KD_lims, exp_lims = get_data(logKD2PWM)

pylab.rcParams['font.size'] = 11

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

usethis1 = np.where(med_rep['CDR3_muts']==0)[0]
usethis3 = np.where(med_rep['CDR1_muts']==0)[0]
A1 = A[usethis1]
A3 = A[usethis3]
AA1= [AA[ii] for ii in usethis1]
AA3= [AA[ii] for ii in usethis3]
usethis1 = med_rep.index[usethis1]
usethis3 = med_rep.index[usethis3]
KD1 = np.array((med_rep['KD'].loc[usethis1]))
KD3 = np.array((med_rep['KD'].loc[usethis3]))
KD1_std = np.array((med_rep['KD_std'].loc[usethis1])) + 1e-10
KD3_std = np.array((med_rep['KD_std'].loc[usethis3])) + 1e-10

plt.ion()
plt.close('all')
figsize=(7.3,3.8)

fig, axes = plt.subplots(1,2,figsize=figsize)
plt.subplots_adjust(
    bottom = 0.2,
    top = 0.95,
    left = 0.1,
    right = 0.88,
    hspace = 0.02,
    wspace = 0.02)
ax = axes[0]

CDR1 = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Beneficial', 28, 'ALL', ax=ax, make_colorbar=False, epistasis='beneficial', y_offset=4)
CDR3 = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '', 90, 'ALL', ax=ax, make_colorbar=True, epistasis='beneficial')
ax.text(-3, 0, '3H')
ax.text(-3, 4, '1H')
ax = axes[1]

CDR1_del = plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, 'Deleterious', 28, 'ALL', ax=ax, make_colorbar=False, epistasis='deleterious', y_offset=4)
CDR3_del = plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '', 90, 'ALL', ax=ax, make_colorbar=True, epistasis='deleterious')
pandas.concat([CDR1, CDR3, CDR1_del, CDR3_del]).to_csv('S1_table_sign_epistasis.csv')

plt.savefig('figure_S13.pdf')
plt.close()

KD_table = open('./CDR1H_sign_epistasis.tex', 'w')
header = '''\\begin{center}
\\begin{tabular}{ | l | l | l | l | l | l | l |}
\\hline\n
domain &
\multicolumn{1}{|p{2cm}|}{\centering \# of corresponding, catastraphic single mutants} &
\multicolumn{1}{|p{2cm}|}{\centering epistasis type } &
\multicolumn{1}{|p{2cm}|}{\centering \# candidate mutants } &
\multicolumn{1}{|p{2cm}|}{\centering \# of mutants with sign epistasis (obs/exp)} &
\multicolumn{1}{|p{2cm}|}{\centering \# of mutants with reciprocal sign epistasis (obs/exp)} &
\multicolumn{1}{|p{2cm}|}{\centering \# of viable mutants with sign epistasis (obs/exp)} 
  \\\\ \\hline \n'''
KD_table.write(header)

KD_table.write('''$^A$''')
plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'NAND', fid=KD_table)
KD_table.write('''$^B$''')
plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'AND', fid=KD_table)
KD_table.write('''$^C$''')
plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'ALL', fid=KD_table)
KD_table.write('''$^D$''')
plot_KD_sign_epistasis(A1, KD1, KD1_std, AA1, '1H', 28, 'ALL', epistasis='deleterious', fid=KD_table)

footer = '''    \\end{tabular} \n
\\end{center}'''
KD_table.write(footer)
KD_table.close()

KD_table = open('./CDR3H_sign_epistasis.tex', 'w')
header = '''\\begin{center}
\\begin{tabular}{ | l | l | l | l | l | l | l |}
\\hline\n
domain &
\multicolumn{1}{|p{2cm}|}{\centering \# of catastrophic mutations} &
\multicolumn{1}{|p{2cm}|}{\centering epistasis type } &
\multicolumn{1}{|p{2cm}|}{\centering \# candidate mutants } &
\multicolumn{1}{|p{2cm}|}{\centering \# of mutants with sign epistasis (obs/exp)} &
\multicolumn{1}{|p{2cm}|}{\centering \# of mutants with reciprocal sign epistasis (obs/exp)} &
\multicolumn{1}{|p{2cm}|}{\centering \# of viable mutants with sign epistasis (obs/exp)} 
  \\\\ \\hline \n'''

KD_table.write(header)

KD_table.write('''$^A$''')
plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'NAND', fid=KD_table)
KD_table.write('''$^B$''')
plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'AND', fid=KD_table)
KD_table.write('''$^C$''')
plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', fid=KD_table)
KD_table.write('''$^D$''')
plot_KD_sign_epistasis(A3, KD3, KD3_std, AA3, '3H', 90, 'ALL', epistasis='deleterious', fid=KD_table)
footer = '''    \\end{tabular} \n
\\end{center}'''
KD_table.write(footer)
KD_table.close()
