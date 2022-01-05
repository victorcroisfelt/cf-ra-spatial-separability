########################################
#   plot_fig08_tcp.py
#
#   Description. Script used to actually plot Fig. 08 of the paper.
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

data_cellfree_est1_lower = np.load('data/fig07_08_cellfree_est1_flexible_lower.npz')
data_cellfree_est2_lower = np.load('data/fig07_08_cellfree_est2_flexible_lower.npz')
data_cellfree_est3_lower = np.load('data/fig07_08_cellfree_est3_flexible_lower.npz')

data_cellfree_est1_practical = np.load('data/fig07_08_cellfree_est1_flexible_practical.npz')
data_cellfree_est2_practical = np.load('data/fig07_08_cellfree_est2_flexible_practical.npz')
data_cellfree_est3_practical = np.load('data/fig07_08_cellfree_est3_flexible_practical.npz')

# Extract x-axis
K0values = data_cellular["K0values"]

# Extract data
tcp_cellular = data_cellular["tcp"]
tcp_bcf = data_bcf["tcp"]

tcp_cellfree_est1_lower = data_cellfree_est1_lower["tcp"]
tcp_cellfree_est2_lower = data_cellfree_est2_lower["tcp"]
tcp_cellfree_est3_lower = data_cellfree_est3_lower["tcp"]

tcp_cellfree_est1_practical = data_cellfree_est1_practical["tcp"]
tcp_cellfree_est2_practical = data_cellfree_est2_practical["tcp"]
tcp_cellfree_est3_practical = data_cellfree_est3_practical["tcp"]

########################################
# Plot
########################################

# Pre-processing
list_K0values = K0values.tolist()

#marker_mask = [list_K0values.index(100), list_K0values.index(1000), list_K0values.index(10000), -1]

marker_mask_lower = np.arange(K0values.size)[np.where(np.arange(K0values.size) % 2)]

marker_mask_practical = set(np.arange(K0values.size)) - set.intersection(set(np.arange(K0values.size)), set(marker_mask_lower))
marker_mask_practical = list(marker_mask_practical)


# Fig. 07d
#fig, ax = plt.subplots(figsize=(3.15, 3/2))
fig, ax = plt.subplots(figsize=(4/3 * 3.15, 2))

ax.plot(K0values, tcp_cellular, linewidth=1.5, linestyle='-', color='black')
ax.plot(K0values, tcp_bcf, linewidth=1.5, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='black')

ax.plot(K0values, tcp_cellfree_est1_lower, linewidth=1.5, linestyle='--')
ax.plot(K0values, tcp_cellfree_est2_lower, linewidth=1.5, linestyle='-.')
ax.plot(K0values, tcp_cellfree_est3_lower, linewidth=1.5, linestyle=':')

plt.gca().set_prop_cycle(None)

ax.plot(K0values, tcp_cellfree_est1_practical, linewidth=1.5, linestyle='--')
ax.plot(K0values, tcp_cellfree_est2_practical, linewidth=1.5, linestyle='-.')
ax.plot(K0values, tcp_cellfree_est3_practical, linewidth=1.5, linestyle=':')

plt.gca().set_prop_cycle(None)

ax.plot(K0values[marker_mask_lower], tcp_cellfree_est1_lower[marker_mask_lower], linewidth=0, markersize='9', marker='1', fillstyle='left', markevery=5, color='black', label='Lower')

ax.plot(K0values[marker_mask_lower], tcp_cellfree_est1_lower[marker_mask_lower], linewidth=0, markersize='9', marker='1', fillstyle='left', markevery=5)
ax.plot(K0values[marker_mask_lower], tcp_cellfree_est2_lower[marker_mask_lower], linewidth=0, markersize='9', marker='1', fillstyle='left', markevery=5)
ax.plot(K0values[marker_mask_lower], tcp_cellfree_est3_lower[marker_mask_lower], linewidth=0, markersize='9', marker='1', fillstyle='left', markevery=5)

plt.gca().set_prop_cycle(None)

ax.plot(K0values[marker_mask_practical], tcp_cellfree_est1_practical[marker_mask_practical], linewidth=0, markersize='9', marker='+', fillstyle='right', markevery=5, color='black', label='Practical')

ax.plot(K0values[marker_mask_practical], tcp_cellfree_est1_practical[marker_mask_practical], linewidth=0, markersize='9', marker='+', fillstyle='right', markevery=5)
ax.plot(K0values[marker_mask_practical], tcp_cellfree_est2_practical[marker_mask_practical], linewidth=0, markersize='9', marker='+', fillstyle='right', markevery=5)
ax.plot(K0values[marker_mask_practical], tcp_cellfree_est3_practical[marker_mask_practical], linewidth=0, markersize='9', marker='+', fillstyle='right', markevery=5)

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.set_xlabel(r'number of inactive users $|\mathcal{U}|$')
ax.set_ylabel('TCP [mW]')

ax.grid(visible=True, alpha=0.5)

ax.text(9e3, 15, "CF-SUCRe Est. 1 and Est. 2\n curves are coinciding", fontsize='x-small')

ax.set_ylim([9.998101209448855, 89548.48326256676])

ax.fill_between(K0values, np.zeros(K0values.size), tcp_bcf, alpha=0.25, color="#661D98", label="fitting EE region")

ax.legend(fontsize='x-small')

ax.set_xlim([900, 52000])

plt.show()
