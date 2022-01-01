########################################
#   lookup_fig05_06_delta.py
#
#   Description. Script used to obtain the average values of delta for Estimator 3
#   when considering diffent values of Lmax and N.
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
#   Comment. The result of this script is stored in the lookup folder. It is 
#   required to run the following codes:
#
#       - lookup_fig05_06_best_pair.py
#       - data_fig05_barplot_cellfree.py
#       - plot_fig06_neb_nmse.py
#
########################################
import numpy as np

import time

########################################
# System parameters
########################################

# Define number of APs
L = 64

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
# Simulation parameters
########################################

# Set number of setups
numsetups = 100

# Set number of channel realizations
numchannel = 100

# Range of collision sizes
collisions = np.arange(1, 11).astype(int)

# Range of number of antennas per AP
N_range = np.arange(1, 11).astype(int)

# Extract maximum number of antennas per AP
N_max = np.max(N_range)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, L+1).astype(int)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Lookup Fig 05 & 06: Est. 3 -- delta")
print("--------------------------------------------------\n")

# Store total time 
total_time = time.time()

# Prepare to save simulation results
avg_effective_power = np.zeros((collisions.size, N_range.size, Lmax_range.size, numsetups, numchannel))


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numsetups, N_max, L, numchannel) + 1j*np.random.randn(numsetups, N_max, L, numchannel))


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

    # Generate normalized channel matrix for each AP equipped with N antennas
    Gnorm_ = np.sqrt(1/2)*(np.random.randn(numsetups, N_max, collisionSize, L, numchannel) + 1j*np.random.randn(numsetups, N_max, collisionSize, L, numchannel))

    # Compute channel matrix
    G_ = np.sqrt(channel_gains[:, None, :, :, None]) * Gnorm_


    # Go through all values of N
    for nn, N in enumerate(N_range):

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * G_[:, :N, :, :, :].sum(axis=2) + n_[:, :N, :, :]

        # Store l2-norms of Yt
        Yt_norms = np.linalg.norm(Yt_, axis=1)

        # Obtain pilot activity vector according to Eq. (8)
        atilde_t = (1/N) * Yt_norms**2
        atilde_t[atilde_t < sigma2] = 0.0

        # Compute alphawidehat according to Eq. (34)
        alphawidehat = (atilde_t - sigma2).sum(axis=-2)


        # Go through all setups
        for ss in range(numsetups):


            # Go thorugh all channel realizations
            for ch in range(numchannel):


                # Go through all Lmax values
                for lm, Lmax in enumerate(Lmax_range):

                    # Obtain set of pilot-serving APs (Definition 2)
                    Pcal = np.argsort(atilde_t[ss, :, ch])[-Lmax:]
                    Pcal = np.delete(Pcal, atilde_t[ss, Pcal, ch] == 0)

                    # Compute effective according to Eq. (36)
                    effective_power = (ql / N) * (Yt_norms[ss, Pcal, ch]**2) / alphawidehat[ss, ch]
                
                    # Store average effective power
                    avg_effective_power[cs, nn, lm, ss, ch] = effective_power.mean()

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

        # Go through all values of N
        for nn, N in enumerate(N_range):

                # Go through all Lmax values
                for lm, Lmax in enumerate(Lmax_range):

                    dict[(collisionSize, N, Lmax)] = avg_delta[cs, nn, lm]

np.savez("lookup/lookup_fig05_06_delta.npz",
    delta=dict
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")
