########################################
#   plot_fig07d_anaa_lower.py
#
#   Description. Script used to actually plot Fig. 07 (c) of the paper.
#
#   Author. @victorcroisfelt
#
#   Date. December 29, 2021
#
#   This code is part of the code package used to generate the numeric results
#   of the paper:
#
#   Croisfelt, V., Abrão, T., and Marinello, J. C., “User-Centric Perspective in
#   Random Access Cell-Free Aided by Spatial Separability”, arXiv e-prints, 2021.
#
#   Available on:
#
#                   https://arxiv.org/abs/2107.10294
#
#   Comment. Please, make sure that you have the required data files. They are 
#	obtained by running the scripts:
#
#	- data_fig07_08_bcf.py
#	- data_fig07_08_cellular.py
#	- data_fig07_08_cellfree.py
#
########################################
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

axis_font = {'size':'12'}

plt.rcParams.update({'font.size': 12})

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

########################################
# Loading data
########################################

data_cellular = np.load('data/fig07_08_cellular.npz')
data_bcf = np.load('data/fig07_08_bcf.npz')

data_cellfree_est1_fixed = np.load('data/fig07_08_cellfree_est1_fixed_lower.npz')
data_cellfree_est2_fixed = np.load('data/fig07_08_cellfree_est2_fixed_lower.npz')
data_cellfree_est3_fixed = np.load('data/fig07_08_cellfree_est3_fixed_lower.npz')

data_cellfree_est1_flexible = np.load('data/fig07_08_cellfree_est1_flexible_lower.npz')
data_cellfree_est2_flexible = np.load('data/fig07_08_cellfree_est2_flexible_lower.npz')
data_cellfree_est3_flexible = np.load('data/fig07_08_cellfree_est3_flexible_lower.npz')

# Extract x-axis
K0values = data_cellular["K0values"]

# Extract ANAA
anaa_cellular = data_cellular["anaa"]
anaa_bcf = data_bcf["anaa"]

anaa_cellfree_est1_fixed = data_cellfree_est1_fixed["anaa"]
anaa_cellfree_est2_fixed = data_cellfree_est2_fixed["anaa"]
anaa_cellfree_est3_fixed = data_cellfree_est3_fixed["anaa"]

anaa_cellfree_est1_flexible = data_cellfree_est1_flexible["anaa"]
anaa_cellfree_est2_flexible = data_cellfree_est2_flexible["anaa"]
anaa_cellfree_est3_flexible = data_cellfree_est3_flexible["anaa"]

########################################
# Plot 
########################################

# Pre-processing
marker_mask_fixed = np.arange(K0values.size)[np.where(np.arange(K0values.size) % 2)]

marker_mask_flexible = set(np.arange(K0values.size)) - set.intersection(set(np.arange(K0values.size)), set(marker_mask_fixed))
marker_mask_flexible = list(marker_mask_flexible)

# Fig. 07c
fig, ax = plt.subplots(figsize=(4/3 * 3.15, 2))

ax.plot(K0values, anaa_bcf, linewidth=1.5, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='black', label='BCF: $N=8$')
ax.plot(K0values, anaa_cellular, linewidth=1.5, linestyle='-', color='black', label='Ce-SUCRe [3]: $M=64$')

ax.plot(K0values, anaa_cellfree_est1_fixed, linewidth=1.5, linestyle='--', label='CF-SUCRe: Est. 1, $N=8$')
ax.plot(K0values, anaa_cellfree_est2_fixed, linewidth=1.5, linestyle='-.', label='CF-SUCRe: Est. 2, $N=8$')
ax.plot(K0values, anaa_cellfree_est3_fixed, linewidth=1.5, linestyle=':', label='CF-SUCRe: Est. 3, $N=8$')

plt.gca().set_prop_cycle(None)

ax.plot(K0values, anaa_cellfree_est1_flexible, linewidth=1.5, linestyle='--')
ax.plot(K0values, anaa_cellfree_est2_flexible, linewidth=1.5, linestyle='-.')
ax.plot(K0values, anaa_cellfree_est3_flexible, linewidth=1.5, linestyle=':')

plt.gca().set_prop_cycle(None)

ax.plot(K0values[marker_mask_fixed], anaa_cellfree_est1_fixed[marker_mask_fixed], linewidth=0, marker='d', markersize=7, fillstyle='none', markevery=5, color='black', label='Fixed')

ax.plot(K0values[marker_mask_fixed], anaa_cellfree_est1_fixed[marker_mask_fixed], linewidth=0, marker='d', markersize=7, fillstyle='none', markevery=5)
ax.plot(K0values[marker_mask_fixed], anaa_cellfree_est2_fixed[marker_mask_fixed], linewidth=0, marker='d', markersize=7, fillstyle='none', markevery=5)
ax.plot(K0values[marker_mask_fixed], anaa_cellfree_est3_fixed[marker_mask_fixed], linewidth=0, marker='d', markersize=7, fillstyle='none', markevery=5)

plt.gca().set_prop_cycle(None)

ax.plot(K0values[marker_mask_flexible], anaa_cellfree_est1_flexible[marker_mask_flexible], linewidth=0, marker='x', markersize=7, fillstyle='none', markevery=5, color='black', label='Greedy flexible')

ax.plot(K0values[marker_mask_flexible], anaa_cellfree_est1_flexible[marker_mask_flexible], linewidth=0, marker='x', markersize=7, fillstyle='none', markevery=5)
ax.plot(K0values[marker_mask_flexible], anaa_cellfree_est2_flexible[marker_mask_flexible], linewidth=0, marker='x', markersize=7, fillstyle='none', markevery=5)
ax.plot(K0values[marker_mask_flexible], anaa_cellfree_est3_flexible[marker_mask_flexible], linewidth=0, marker='x', markersize=7, fillstyle='none', markevery=5)

ax.set_xscale('log', base=10)

ax.set_yticks(np.array([1, 3, 5, 7, 9, 10]))

ax.set_xlim([900, 52000])

ax.set_xlabel(r'number of inactive users $|\mathcal{U}|$')
ax.set_ylabel('ANAA')

ax.legend(fontsize='x-small',markerscale=0.75)

ax.grid(linestyle='--', visible=True, alpha=0.25)

plt.show()

