########################################
#   plot_fig06_neb_nmse.py
#
#   Description. Script used to plot Fig. 6 of the paper based on Table II.
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
########################################
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

import time

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

np.random.seed(42)

axis_font = {'size':'8'}

plt.rcParams.update({'font.size': 8})

matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

########################################
# System parameters
########################################

# Define number of BS antennas
M = 64

# Define number of APs
L = 64

# UL transmit power
p = 100

# DL transmit power
q = 200
ql = q/L # per AP

# Noise variance
sigma2 = 1

# Number of RA pilot signals
taup = 5

########################################
# Geometry
########################################

# Define square length
squareLength = 400

# Create square grid of APs
APperdim = int(np.sqrt(L))
APpositions = np.linspace(squareLength/APperdim, squareLength, APperdim) - squareLength/APperdim/2
APpositions = APpositions + 1j*APpositions[:, None]
APpositions = APpositions.reshape(L)

# Define BS position
BSposition = (squareLength/2)*(1 + 1j)

########################################
# Lookup table
########################################

# Load best pair look up table
load = np.load("lookup/lookup_fig06_best_pair_est1.npz", allow_pickle=True)
best_pair_lookup_est1 = load["best_pair"]
best_pair_lookup_est1 = best_pair_lookup_est1.item()

load = np.load("lookup/lookup_fig06_best_pair_est2.npz", allow_pickle=True)
best_pair_lookup_est2 = load["best_pair"]
best_pair_lookup_est2 = best_pair_lookup_est2.item()

load = np.load("lookup/lookup_fig06_best_pair_est3.npz", allow_pickle=True)
best_pair_lookup_est3 = load["best_pair"]
best_pair_lookup_est3 = best_pair_lookup_est3.item()

# Load possible values of delta for Estimator 3
load = np.load("lookup/lookup_fig05_06_delta.npz", allow_pickle=True)
delta_lookup = load["delta"]
delta_lookup = delta_lookup.item()

########################################
# Simulation parameters
########################################

# Set number of setups
numsetups = 100

# Set number of channel realizations
numchannel = 100

# Range of number of antennas per AP
Nrange = np.arange(1, 11)

# Extract maximum number of antennas per AP
Nmax = np.max(Nrange)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Fig. 06: NEB and NMSE")
print("--------------------------------------------------\n")

# Store total time
total_time = time.time()

# Generate noise realization at UEs
eta = np.sqrt(sigma2/2)*(np.random.randn(numsetups, 2, numchannel) + 1j*np.random.randn(numsetups, 2, numchannel))

# Generate UEs locations
UElocations = squareLength*(np.random.rand(numsetups, 2) + 1j*np.random.rand(numsetups, 2))

#####
# Cellular
#####
print("*** Cellular ***")
print("\n")

# Prepare to save cellular results
bias_cellular = np.zeros((numsetups, 2), dtype=np.float_)
nmse_cellular = np.zeros((numsetups, 2), dtype=np.float_)


#####


# Generate noise realizations at the BS
n_ = np.sqrt(sigma2/2)*(np.random.randn(numsetups, M, numchannel) + 1j*np.random.randn(numsetups, M, numchannel))

# Compute UEs distances to the BS
UEdistances = np.abs(UElocations - BSposition)

# Compute average channel gains according to Eq. (1)
channel_gains = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistances**2 + 10**2)))/10)

# Compute true value of alpha
alpha_cellular = p * taup * channel_gains.sum(axis=1)

# Generate channel matrix for the BS equipped with M antennas
H_ = np.sqrt(channel_gains[:, None, :, None]/2)*(np.random.randn(numsetups, M, 2, numchannel) + 1j*np.random.randn(numsetups, M, 2, numchannel))

# Compute received signal (equivalent to Eq. (4))
yt_ = np.sqrt(p * taup) * H_.sum(axis=2) + n_

# Compute precoded DL signal (equiavalent to Eq. (10))
vt_ = np.sqrt(q) * (yt_ / np.linalg.norm(yt_, axis=1)[:, None, :])


# Go through all colliding UEs
for k in range(2):

    # Compute received DL signal at UE k (equivalent to Eq. (12))
    z_k = np.sqrt(taup) * (H_[:, :, k, :].conj() * vt_).sum(axis=1) + eta[:, k, :]


    #####
    # Estimation
    #####


    # Compute constants
    den = z_k.real/np.sqrt(M)
    num = np.sqrt(q * p) * taup * channel_gains[:, k]

    # Compute estimate
    alphahat = ((num[:, None]/den)**2) - sigma2

    # Compute own total UL signal power (equivalent to Eq. (15))
    gamma = p * taup * channel_gains[:, k]

    # Avoiding underestimation
    for ch in range(numchannel):

        mask = alphahat[:, ch] <= gamma
        alphahat[mask, ch] = gamma[mask]

    # Compute stats
    bias_cellular[:, k] = (alphahat.mean(axis=-1) - alpha_cellular)/alpha_cellular
    nmse_cellular[:, k] = np.mean((np.abs(alphahat - alpha_cellular[:, None])**2), axis=-1)/(alpha_cellular**2)

print("cellular simulation part is done.\n")


#####
# Cell-free
#####
print("*** Cell-free ***\n")

# Prepare to save cell-free results
bias1_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)
bias2_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)
bias3_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)

nmse1_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)
nmse2_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)
nmse3_cellfree = np.zeros((Nrange.size, numsetups, 2, numchannel), dtype=np.float_)


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numsetups, Nmax, L, numchannel) + 1j*np.random.randn(numsetups, Nmax, L, numchannel))

# Compute UEs distances to each AP
UEdistances = np.abs(UElocations[:, :, np.newaxis] - APpositions)

# Compute average channel gains according to Eq. (1)
channel_gains = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistances**2 + 10**2)))/10)


# Go through all setups
for ss in range(numsetups):

    # Storing time
    timer_start = time.time()

    # Print current setup
    print(f"\tsetup: {ss}/{numsetups - 1}")

    # Extract current average channel gains
    channel_gains_current = channel_gains[ss, :, :]

    # Generate channel matrix for each AP equipped with N antennas
    G_ = np.sqrt(channel_gains_current[None, :, :, None]/2)*(np.random.randn(Nmax, 2, L, numchannel) + 1j*np.random.randn(Nmax, 2, L, numchannel))

    # Go through all values of N
    for nn, N in enumerate(Nrange):

        # Extract current channel matrix
        Gn = G_[:N, :, :, :]

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * Gn.sum(axis=1) + n_[ss, :N, :, :]

        # Obtain pilot activity vector according to Eq. (8)
        atilde_t = (1/N) * np.linalg.norm(Yt_, axis=0)**2
        atilde_t[atilde_t < sigma2] = 0.0

        # Extract current best pair 
        (Ccal_size_est1, Lmax_est1) = best_pair_lookup_est1[(2, N)]
        (Ccal_size_est2, Lmax_est2) = best_pair_lookup_est2[(2, N)]
        (Ccal_size_est3, Lmax_est3) = best_pair_lookup_est3[(2, N)]

        # Obtain sets of pilot-serving APs (Definition 2)
        Pcal_est1 = np.argsort(atilde_t, axis=0)[-Lmax_est1:, :]
        Pcal_est2 = np.argsort(atilde_t, axis=0)[-Lmax_est2:, :]
        Pcal_est3 = np.argsort(atilde_t, axis=0)[-Lmax_est3:, :]

        # Go thorugh all realizations
        for rr in range(numchannel):

            # Extract Pcals
            Pcal_est1_current = Pcal_est1[:, rr]
            Pcal_est2_current = Pcal_est2[:, rr]
            Pcal_est3_current = Pcal_est3[:, rr]

            # Compute precoded DL signal according to Eqs. (10) and (35)
            Vt_est1 = np.sqrt(ql) * (Yt_[:, Pcal_est1_current, rr] / np.linalg.norm(Yt_[:, Pcal_est1_current, rr], axis=0))
            Vt_est2 = np.sqrt(ql) * (Yt_[:, Pcal_est2_current, rr] / np.linalg.norm(Yt_[:, Pcal_est2_current, rr], axis=0))
            Vt_est3 = np.sqrt(ql) * (Yt_[:, Pcal_est3_current, rr] / np.sqrt(N * (np.maximum(atilde_t[:, rr] - sigma2, np.zeros(atilde_t[:, rr].size))).sum()))

            # Compute true total UL signal power of colliding UEs according to 
            # Eq. (16)
            alpha_est1 = (p * taup * channel_gains_current[:, Pcal_est1_current]).sum()
            alpha_est2 = (p * taup * channel_gains_current[:, Pcal_est2_current]).sum()
            alpha_est3 = (p * taup * channel_gains_current[:, Pcal_est3_current]).sum()

            # Go through all colliding users
            for k in range(2):

                # Compute received DL signal at UE k according to Eq. (12)
                z_k_est1 = np.sqrt(taup) * (Gn[:, k, Pcal_est1_current, rr].conj() * Vt_est1).sum() + eta[ss, k, rr]
                z_k_est2 = np.sqrt(taup) * (Gn[:, k, Pcal_est2_current, rr].conj() * Vt_est2).sum() + eta[ss, k, rr]
                z_k_est3 = np.sqrt(taup) * (Gn[:, k, Pcal_est3_current, rr].conj() * Vt_est3).sum() + eta[ss, k, rr]

                # Obtain set of nearby APs of UE k (Definition 1)
                Ccal_est1 = np.argsort(ql * channel_gains_current[k])[-Ccal_size_est1:]
                Ccal_est2 = np.argsort(ql * channel_gains_current[k])[-Ccal_size_est2:]
                Ccal_est3 = np.argsort(ql * channel_gains_current[k])[-Ccal_size_est3:]

                # Obtain natural set of nearby APs of UE k (Definition 1)
                checkCcal = np.arange(L)[ql * channel_gains_current[k] > sigma2]

                if len(checkCcal) == 0:
                    checkCcal = np.array([np.argmax(ql * channel_gains_current[k, :])])

                if len(Ccal_est1) > len(checkCcal):
                    Ccal_est1 = checkCcal

                if len(Ccal_est2) > len(checkCcal):
                    Ccal_est2 = checkCcal

                if len(Ccal_est3) > len(checkCcal):
                    Ccal_est3 = checkCcal


                #####
                # Estimator 1
                #####


                # Compute constants
                cte = z_k_est1.real/np.sqrt(N)
                num = np.sqrt(ql * p) * taup * channel_gains_current[k, Ccal_est1]

                # Compute estimate according to Eq. (28)
                alphahat_est1 = ((num.sum()/cte)**2) - sigma2

                # Compute own total UL signal power in Eq. (15)
                gamma_est1 = p * taup * channel_gains_current[k, Ccal_est1].sum()

                # Avoiding underestimation
                if alphahat_est1 < gamma_est1:
                    alphahat_est1 = gamma_est1


                #####
                # Estimator 2
                #####


                # Compute constants
                cte = z_k_est2.real/np.sqrt(N)
                num = np.sqrt(ql * p) * taup * channel_gains_current[k, Ccal_est2]
                num23 = num**(2/3)
                cte2 = (num23.sum()/cte)**2

                # Compute estimate according to Eq. (32)
                alphahat_est2 = (cte2 * num23 - sigma2).sum()

                # Compute own total UL signal power in Eq. (15)
                gamma_est2 = p * taup * channel_gains_current[k, Ccal_est2].sum()

                # Avoiding underestimation
                if alphahat_est2 < gamma_est2:
                    alphahat_est2 = gamma_est2


                #####
                # Estimator 3
                #####


                # Compute new constant according to Eq. (38)
                delta = delta_lookup[(2, N, Lmax_est3)]
                underline_cte = delta * (z_k_est3.real - sigma2)/np.sqrt(N)
                num = np.sqrt(ql * p) * taup * channel_gains_current[k, Ccal_est3]

                # Compute estimate according to Eq. (40)
                alphahat_est3 = (num.sum()/cte)**2

                # Compute own total UL signal power in Eq. (15)
                gamma_est3 = p * taup * channel_gains_current[k, Ccal_est3].sum()

                # Avoiding underestimation
                if alphahat_est3 < gamma_est3:
                    alphahat_est3 = gamma_est3



                # Store stats
                bias1_cellfree[nn, ss, k, rr] = (alphahat_est1 - alpha_est1)/alpha_est1
                bias2_cellfree[nn, ss, k, rr] = (alphahat_est2 - alpha_est2)/alpha_est2
                bias3_cellfree[nn, ss, k, rr] = (alphahat_est3 - alpha_est3)/alpha_est3

                nmse1_cellfree[nn, ss, k, rr] = (np.abs(alphahat_est1 - alpha_est1)**2)/alpha_est1**2
                nmse2_cellfree[nn, ss, k, rr] = (np.abs(alphahat_est2 - alpha_est2)**2)/alpha_est2**2
                nmse3_cellfree[nn, ss, k, rr] = (np.abs(alphahat_est3 - alpha_est3)**2)/alpha_est3**2

    print("\t[setup] elapsed " + str(np.round(time.time() - timer_start, 4)) + " seconds.\n")

print("cell-free simulation part is done.\n")


print("total simulation time was " + str(np.round(time.time() - total_time, 4)) + " seconds.\n")
print("wait for the plots...\n")

# Processing
bias1_cellfree = bias1_cellfree.mean(axis=-1)
bias2_cellfree = bias2_cellfree.mean(axis=-1)
bias3_cellfree = bias3_cellfree.mean(axis=-1)

nmse1_cellfree = nmse1_cellfree.mean(axis=-1)
nmse2_cellfree = nmse2_cellfree.mean(axis=-1)
nmse3_cellfree = nmse3_cellfree.mean(axis=-1)

########################################
# Plot
########################################

# Plot 6a
fig, ax = plt.subplots(figsize=(3.15/2, 3/2))

ax.plot(Nrange, np.median(bias_cellular)*np.ones(Nrange.size), linewidth=1.5, linestyle='-', color='black', label='Cellular')
ax.plot(Nrange, np.median(bias1_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle='--', label='Cell-free: Est. 1')
ax.plot(Nrange, np.median(bias2_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle='-.', label='Cell-free: Est. 2')
ax.plot(Nrange, np.median(bias3_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle=':', label='Cell-free: Est. 3')

plt.gca().set_prop_cycle(None)

ax.fill_between(Nrange, np.percentile(bias_cellular, 25)*np.ones(Nrange.size), np.percentile(bias_cellular, 75)*np.ones(Nrange.size), linewidth=0, alpha=0.25, color='black')
ax.fill_between(Nrange, np.percentile(bias1_cellfree, 25, axis=(-1, -2)), np.percentile(bias1_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)
ax.fill_between(Nrange, np.percentile(bias2_cellfree, 25, axis=(-1, -2)), np.percentile(bias2_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)
ax.fill_between(Nrange, np.percentile(bias3_cellfree, 25, axis=(-1, -2)), np.percentile(bias3_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)

ax.set_xlabel('$N$')
ax.set_ylabel('$\mathrm{NEB}$')

ax.set_ylim([-0.7, 0.3])

ax.set_xticks(np.arange(1,11))

ax.legend(fontsize='xx-small', loc='upper right')

plt.show()


# Plot 6b
fig, ax = plt.subplots(figsize=(3.15/2, 3/2))

ax.plot(Nrange, np.median(nmse_cellular)*np.ones(Nrange.size), linewidth=1.5, linestyle='-', color='black', label='Cellular')
ax.plot(Nrange, np.median(nmse1_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle='--', label='Cell-free: Est. 1')
ax.plot(Nrange, np.median(nmse2_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle='-.', label='Cell-free: Est. 2')
ax.plot(Nrange, np.median(nmse3_cellfree, axis=(-1, -2)), linewidth=1.5, linestyle=':', label='Cell-free: Est. 3')

plt.gca().set_prop_cycle(None)

ax.fill_between(Nrange, np.percentile(nmse_cellular, 25)*np.ones(Nrange.size), np.percentile(nmse_cellular, 75)*np.ones(Nrange.size), linewidth=0, alpha=0.25, color='black')
ax.fill_between(Nrange, np.percentile(nmse1_cellfree, 25, axis=(-1, -2)), np.percentile(nmse1_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)
ax.fill_between(Nrange, np.percentile(nmse2_cellfree, 25, axis=(-1, -2)), np.percentile(nmse2_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)
ax.fill_between(Nrange, np.percentile(nmse3_cellfree, 25, axis=(-1, -2)), np.percentile(nmse3_cellfree, 75, axis=(-1, -2)), linewidth=0, alpha=0.25)

ax.set_yscale('log', base=10)

ax.set_xlabel('$N$')
ax.set_ylabel('$\mathrm{NMSE}$')

ax.set_ylim([2e-2, 1])

ax.set_xticks(np.arange(1,11))

plt.show()

print("------------------- all done :) ------------------")