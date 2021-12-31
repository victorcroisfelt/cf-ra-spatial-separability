########################################
#   plot_fig04a_nmd.py
#
#   Description. Script used to plot Fig. 4 of the paper.
#
#   Author. @victorcroisfelt
#
#   Date. December 27, 2021
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

import time

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

np.random.seed(42)

axis_font = {'size':'12'}

plt.rcParams.update({'font.size': 12})

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

########################################
# System parameters
########################################

# Define number of APs
L = 64

# Define number of antennas per AP
N = 8

# UL transmit power
p = 100

# DL transmit power per AP
ql = 200/L

# Define noise power
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

########################################
# SUCRe parameters
########################################

# Probability of access
pA = 0.001

########################################
# Simulation parameters
########################################

# Set the number of RA blocks
numRAblocks = 100

# Range of collision sizes
collisions = np.arange(1, 11)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, L+1)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Fig. 04: NMD and average probability")
print("--------------------------------------------------\n")

# Store total time
total_time = time.time()

# Store enumeration of L
enumerationL = np.arange(L)

# Prepare to save simulation results
nmd = np.zeros((collisions.size, Lmax_range.size, numRAblocks))
probability = np.zeros((collisions.size, Lmax_range.size, numRAblocks))


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numRAblocks, N, L) + 1j*np.random.randn(numRAblocks, N, L))


# Go through all collision sizes
for cs, collisionSize in enumerate(collisions):

    # Storing time
    timer_start = time.time()

    # Print out current data point
    print(f"\tcollision: {cs}/{collisions.size-1}")

    # Generate normalized channel matrix for each AP equipped with N antennas
    Gnorm_ = np.sqrt(1/2)*(np.random.randn(numRAblocks, N, collisionSize, L) + 1j*np.random.randn(numRAblocks, N, collisionSize, L))


    # Go through all RA blocks
    for rr in range(numRAblocks):


        #####
        # Generating UEs
        #####


        # Generate UEs locations
        UElocations = squareLength*(np.random.rand(collisionSize) + 1j*np.random.rand(collisionSize))

        # Compute UEs distances to each AP
        UEdistance = abs(APpositions - UElocations[:, None])

        # Compute average channel gains according to Eq. (1)
        betas = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistance**2 + 10**2)))/10)

        # Randomize which pilot each UE chose
        pilotSelections = np.random.randint(1, taup+1, size=collisionSize);
        pilotSelections += -1

        # Generate channel matrix for each AP equipped with N antennas
        G_ = np.sqrt(betas[None, :, :]) * Gnorm_[rr, :, :, :] 

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * np.sum(G_, axis=1) + n_[rr, :, :]

        # Store l2-norms of Yt
        Yt_norms = np.linalg.norm(Yt_, axis=0)

        # Obtain pilot activity vector according to Eq. (8)
        atilde_t = (1/N) * Yt_norms**2
        atilde_t[atilde_t < sigma2] = 0.0
        
        # Prepare to store sum over the sets of nearby APs
        sum_checkCcal = np.zeros(collisionSize)


        # Go through all colliding UEs
        for k in range(collisionSize):

            # Obtain natural set of nearby APs of UE k (Definition 1)
            checkCcal = enumerationL[ql * betas[k, :] > sigma2]

            if len(checkCcal) == 0:
                checkCcal = np.array([np.argmax(ql * betas[k, :])])

            # Calculate sum of betas over natural set of nearby APs of UE k
            sum_checkCcal[k] = np.sum(betas[k, checkCcal])


        # Go through all different values of Lmax
        for ll, Lmax in enumerate(Lmax_range):

            # Obtain set of pilot-serving APs (Definition 2)
            Pcal = np.argsort(atilde_t)[-Lmax:]
            Pcal = np.delete(Pcal, atilde_t[Pcal] == 0)

            # Calculate sum of betas over set of pilot-serving APs
            sum_Pcal = np.sum(betas[k, Pcal])

            # Store results
            nmd[cs, ll, rr] = np.mean(sum_checkCcal - sum_Pcal)/sum_Pcal
            probability[cs, ll, rr] = np.mean(sum_Pcal > sum_checkCcal)

    print('\t[collision] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')


# Calculate averages
avg_nmd = np.mean(nmd, axis=-1)
avg_probability = np.mean(probability, axis=-1)

print("total simulation time was " + str(np.round(time.time() - total_time, 4)) + " seconds.\n")
print("wait for the plots...\n")

########################################
# Plot
########################################

# NMD
fig = plt.figure(figsize=(3.15,3))
ax = fig.gca(projection="3d")

X_, Y_ = np.meshgrid(Lmax_range, collisions)

ax.plot_surface(X_, Y_, np.sign(avg_nmd)*np.log10(1+np.abs(avg_nmd)))

ax.set_xlabel("$L^{\max}$")
ax.set_ylabel("$|\mathcal{S}_t|$")
ax.set_zlabel(r"transformed $\overline{\mathrm{NMD}}$")

ax.set_xticks([0, 16, 32, 48, 64])
ax.set_yticks([1, 2, 4, 6, 8, 10])

ax.view_init(elev=20, azim=-45)

plt.show()

# Probability
fig = plt.figure(figsize=(3.15,3))
ax = fig.gca(projection="3d")

X_, Y_ = np.meshgrid(Lmax_range, collisions)

ax.plot_surface(X_, Y_, avg_probability)

ax.set_xlabel("$L^{\max}$")
ax.set_ylabel("$|\mathcal{S}_t|$")
ax.set_zlabel(r"average probability")

ax.set_xticks([0, 16, 32, 48, 64])
ax.set_yticks([1, 2, 4, 6, 8, 10])

ax.view_init(elev=30, azim=-135)

plt.show()

print("------------------- all done :) ------------------")