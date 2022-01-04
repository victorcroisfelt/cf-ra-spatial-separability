########################################
#   plot_fig05_barplot.py
#
#   Description. Script used to actually plot Fig. 5 of the paper.
#
#   Author. @victorcroisfelt
#
#   Date. December 31, 2021
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
#	- data_fig05_barplot_cellfree.py
#	- data_fig05_barplot_cellular.py
#
########################################
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

axis_font = {'size':'8'}

plt.rcParams.update({'font.size': 8})

matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

########################################
# Lookup table
########################################

# Load best pair look up table
load = np.load("lookup/lookup_fig05_best_pair_est1.npz", allow_pickle=True)
best_pair_lookup_est1 = load["best_pair"]
best_pair_lookup_est1 = best_pair_lookup_est1.item()

load = np.load("lookup/lookup_fig05_best_pair_est2.npz", allow_pickle=True)
best_pair_lookup_est2 = load["best_pair"]
best_pair_lookup_est2 = best_pair_lookup_est2.item()

load = np.load("lookup/lookup_fig05_best_pair_est3.npz", allow_pickle=True)
best_pair_lookup_est3 = load["best_pair"]
best_pair_lookup_est3 = best_pair_lookup_est3.item()

# Load possible values of delta for Estimator 3
load = np.load("lookup/lookup_fig05_06_delta.npz", allow_pickle=True)
delta_lookup = load["delta"]
delta_lookup = delta_lookup.item()

# Range of collision sizes
collisions = np.arange(1, 11)

best_delta = np.zeros((collisions.size))

# Go throguh all collisions
for cs, collisionSize in enumerate(collisions):
	best_delta[cs] = delta_lookup[(collisionSize, 8, (best_pair_lookup_est3[(collisionSize, 8)])[1])]

########################################
# Loading data
########################################
print('--------------------------------------------------')
print('Fig 05: barplot')
print('--------------------------------------------------\n')

# Load data
data_cellular = np.load("data/fig05_barplot_cellular.npz")

data_cellfree_est1 = np.load("data/fig05_barplot_cellfree_est1.npz", allow_pickle=True)
data_cellfree_est2 = np.load("data/fig05_barplot_cellfree_est2.npz", allow_pickle=True)
data_cellfree_est3 = np.load("data/fig05_barplot_cellfree_est3.npz", allow_pickle=True)

# Print Table II
print("*** TABLE II ****")
print("\n")
print("Estimator 1: " + str(best_pair_lookup_est1.values()))
print("\n")
print("Estimator 2: " + str(best_pair_lookup_est2))
print("\n")
print("Estimator 3: " + str(best_pair_lookup_est3))
print("             " + str())
print("\n")
print("*****************\n")

print("wait for the plot...\n")

# Extract NMSEs
nmse_cellular = data_cellular["nmse"]

nmse_cellfree_est1 = data_cellfree_est1["nmse"]
nmse_cellfree_est2 = data_cellfree_est2["nmse"]
nmse_cellfree_est3 = data_cellfree_est3["nmse"]

########################################
# Plot
########################################

fig, ax = plt.subplots(figsize=(3.15, 1.5))

width = 0.2
error_kw = dict(lw=1, capsize=1, capthick=1)

dy_cellular = [-(nmse_cellular[0] - nmse_cellular[1]), nmse_cellular[-1] - nmse_cellular[1]]
dy_cellfree_est1 = [-(nmse_cellfree_est1[0] - nmse_cellfree_est1[1]), nmse_cellfree_est1[-1] - nmse_cellfree_est1[1]]
dy_cellfree_est2 = [-(nmse_cellfree_est2[0] - nmse_cellfree_est2[1]), nmse_cellfree_est2[-1] - nmse_cellfree_est2[1]]
dy_cellfree_est3 = [-(nmse_cellfree_est3[0] - nmse_cellfree_est3[1]), nmse_cellfree_est3[-1] - nmse_cellfree_est3[1]]

ax.bar(collisions - 3/2*width, nmse_cellular[1], yerr=dy_cellular, width=width, linewidth=2.0, color='black', label='Cellular', align='center', alpha=0.5, log=True, error_kw=error_kw)
ax.bar(collisions - 1/2*width, nmse_cellfree_est1[1], yerr=dy_cellfree_est1, width=width, linewidth=2.0, label='Cell-free: Est. 1, $N=8$', align='center', alpha=0.5, log=True, error_kw=error_kw)
ax.bar(collisions + 1/2*width, nmse_cellfree_est2[1], yerr=dy_cellfree_est2, width=width, linewidth=2.0, label='Cell-free: Est. 2, $N=8$', align='center', alpha=0.5, log=True, error_kw=error_kw)
ax.bar(collisions + 3/2*width, nmse_cellfree_est3[1], yerr=dy_cellfree_est3, width=width, linewidth=2.0, label='Cell-free: Est. 3, $N=8$', align='center', alpha=0.5, log=True, error_kw=error_kw)

ax.set_xlabel("collision size $|\mathcal{S}_t|$")
ax.set_ylabel("${\mathrm{NMSE}}$")

ax.set_xticks(collisions)

ax.legend(fontsize='xx-small')

plt.show()

print("------------------- all done :) ------------------")
