########################################
#   plot_fig07d_anaa_practical.py
#
#   Description. Script used to actually plot Fig. 07 (d) of the paper.
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
data_bcf = np.load('data/fig07e_bcf.npz')

data_cellfree_est1 = np.load('data/fig07e_cellfree_est1.npz')
data_cellfree_est2 = np.load('data/fig07e_cellfree_est2.npz')
data_cellfree_est3 = np.load('data/fig07e_cellfree_est3.npz')

# Extract x-axis
L_range = data_cellfree_est1["L_range"]
N_range = data_cellfree_est1["N_range"]


# Extract ANAA
anaa_bcf = data_bcf["anaa"]

anaa_cellfree_est1 = data_cellfree_est1["anaa"]
anaa_cellfree_est2 = data_cellfree_est2["anaa"]
anaa_cellfree_est3 = data_cellfree_est3["anaa"]

########################################
# Plot
########################################

# Fig. 07e
fig, ax = plt.subplots(figsize=(4/3 * 3.15, 2))
#fig, ax = plt.subplots(figsize=(1/3 * (6.30), 3))

# Go through all values of N
for nn, N in enumerate(N_range):

	plt.gca().set_prop_cycle(None)

	if N == 1:

		# BCF
		ax.plot(L_range[:-2], anaa_bcf[:-2], linewidth=1.5, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='black', label='BCF')

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=1.5, linestyle='--', color='black', label='CF-SUCRe: Est. 1')
		ax.plot(L_range[:-2], anaa_cellfree_est2[:-2, nn], linewidth=1.5, linestyle='-.', color='black', label='CF-SUCRe: Est. 2')
		ax.plot(L_range[:-2], anaa_cellfree_est3[:-2, nn], linewidth=1.5, linestyle=':', color='black', label='CF-SUCRe: Est. 3')

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=1.5, linestyle='--')
		ax.plot(L_range[:-2], anaa_cellfree_est2[:-2, nn], linewidth=1.5, linestyle='-.')
		ax.plot(L_range[:-2], anaa_cellfree_est3[:-2, nn], linewidth=1.5, linestyle=':')

	elif N == 8:

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=1.5, linestyle='--')
		ax.plot(L_range[:-2], anaa_cellfree_est2[:-2, nn], linewidth=1.5, linestyle='-.')
		ax.plot(L_range[:-2], anaa_cellfree_est3[:-2, nn], linewidth=1.5, linestyle=':')

	plt.gca().set_prop_cycle(None)

	if N == 1:

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=0.0, marker='^', color='black', label='$N=1$')

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=0.0, marker='^')
		ax.plot(L_range[:-2], anaa_cellfree_est2[:-2, nn], linewidth=0.0, marker='^')
		ax.plot(L_range[:-2], anaa_cellfree_est3[:-2, nn], linewidth=0.0, marker='^')

	elif N == 8:

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=0.0, marker='v', color='black', label='$N=8$')

		ax.plot(L_range[:-2], anaa_cellfree_est1[:-2, nn], linewidth=0.0, marker='v')
		ax.plot(L_range[:-2], anaa_cellfree_est2[:-2, nn], linewidth=0.0, marker='v')
		ax.plot(L_range[:-2], anaa_cellfree_est3[:-2, nn], linewidth=0.0, marker='v')


def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2

ax.set_xscale('function', functions=(forward, inverse))

ax.set_xticks(L_range[:-2])
ax.set_yticks(np.array([1, 3, 5, 7, 9, 10]))

ax.grid(visible=True, alpha=0.25, linestyle='--')

ax.set_xlabel(r'number of APs $L$')
ax.set_ylabel('ANAA')

ax.legend(fontsize='xx-small', markerscale=.5)

plt.show()
