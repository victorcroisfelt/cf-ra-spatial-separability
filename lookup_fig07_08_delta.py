########################################
#   lookup_fig07_08_delta.py
#
#   Description. Script used to obtain the average values of delta for Estimator 3
#   when considering diffent values of Lmax.
#
#   Author. @victorcroisfelt
#
#   Date. January 01, 2022
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

import time

from settings_fig07_08 import *

########################################
# Simulation parameters
########################################

# Set number of setups
numsetups = 100

# Set number of channel realizations
numchannel = 100

# Range of collision sizes
collisions = np.arange(1, 51).astype(int)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, (L+1)).astype(int)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Lookup Fig 07 & 08: Est. 3 -- delta")
print("\tN = " + str(N))
print("--------------------------------------------------\n")

# Store total time 
total_time = time.time()

# Prepare to save simulation results
avg_effective_power = np.zeros((collisions.size, Lmax_range.size, numsetups, numchannel))


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numsetups, N, L, numchannel) + 1j*np.random.randn(numsetups, N, L, numchannel))


# Go through all collision sizes
for cs, collisionSize in enumerate(collisions):

    # Storing time
    timer_start = time.time()

    # Print current data point
    print(f"\tcollision: {cs}/{collisions.size-1}")


    #####
    # Generating UEs
    #####


    # Generate UEs locations
    UElocations = squareLength*(np.random.rand(numsetups, collisionSize) + 1j*np.random.rand(numsetups, collisionSize))

    # Compute UEs distances to each AP
    UEdistances = np.abs(UElocations[:, :, np.newaxis] - APpositions)

    # Compute average channel gains according to Eq. (1)
    channel_gains = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistances**2 + 10**2)))/10)


    # Go through all setups
    for ss in range(numsetups):

        # Generate normalized channel matrix for each AP equipped with N antennas
        Gnorm_ = np.sqrt(1/2)*(np.random.randn(N, collisionSize, L, numchannel) + 1j*np.random.randn(N, collisionSize, L, numchannel))

        # Compute channel matrix
        G_ = np.sqrt(channel_gains[ss, None, :, :, None]) * Gnorm_

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * G_.sum(axis=1) + n_[ss, :, :, :]

        # Store l2-norms of Yt
        Yt_norms = np.linalg.norm(Yt_, axis=0)

        # Obtain pilot activity vector according to Eq. (8)
        atilde_t = (1/N) * Yt_norms**2
        atilde_t[atilde_t < sigma2] = 0.0

        # Compute alphawidehat according to Eq. (34)
        alphawidehat = (atilde_t - sigma2).sum(axis=-2)


        # Go thorugh all channel realizationsN
        for ch in range(numchannel):

            # Go through all Lmax values
            for lm, Lmax in enumerate(Lmax_range):

                # Obtain set of pilot-serving APs (Definition 2)
                Pcal = np.argsort(atilde_t[:, ch])[-Lmax:]
                Pcal = np.delete(Pcal, atilde_t[Pcal, ch] == 0)

                # Compute effective according to Eq. (36)
                effective_power = (ql / N) * (Yt_norms[Pcal, ch]**2) / alphawidehat[ch]
            
                # Store average effective power
                avg_effective_power[cs, lm, ss, ch] = effective_power.mean()


    print("\t[collision] elapsed " + str(np.round(time.time() - timer_start, 4)) + " seconds.\n")


# Compute average effective power
avg_effective_power = np.nanmean(avg_effective_power, axis=(-1, -2))

# Compute average delta 
avg_delta = np.sqrt(ql / avg_effective_power)

print("total simulation time was " + str(np.round(time.time() - total_time, 4)) + " seconds.\n")
print("wait for lookup saving...\n")

# Store it as a dictionary
dict = {}

# Go through all collision sizes
for cs, collisionSize in enumerate(collisions):

    # Go through all Lmax values
    for lm, Lmax in enumerate(Lmax_range):

        dict[(collisionSize, Lmax)] = avg_delta[cs, lm]

np.savez("lookup/lookup_fig07_08_delta.npz",
    delta=dict
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")


