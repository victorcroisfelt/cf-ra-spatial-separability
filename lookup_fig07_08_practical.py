########################################
#   lookup_fig07_08_practical.py
#
#   Description. Script used to obtain a lookup table for Lmax and delta
#   related to the practical bound of performance obtained using Algorithm 1. 
#   This script is independent of the estimator choice.
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
#   Comment. The result of this script is stored in the lookup folder. It is 
#   required to run the following code:
#
#       - data_fig07_08_cellfree_practical.py
#
########################################
import numpy as np

import time

from settings_fig07_08 import *

########################################
# Simulation parameters
########################################

# Set number of random transmission rounds R
numrounds = 100

# Set number of transmission repetitions E
numrepetitions = 100

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Lookup Fig 07 & 08: cell-free -- practical")
print("\tN = " + str(N))
print("--------------------------------------------------\n")

# Store total time 
total_time = time.time()

# Prepare to save simulation results
Lmax_practical = np.zeros((K0values.size))
delta_practical = np.zeros((K0values.size))


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numrounds, taup, N, L, numrepetitions) + 1j*np.random.randn(numrounds, taup, N, L, numrepetitions))


# Go through all different number of inactive UEs
for kk, K0 in enumerate(K0values):

    # Storing time
    timer_start = time.time()

    # Print current data point
    print(f"\tinactive UEs: {kk}/{K0values.size-1}")

    # Generate the number of UEs that wish to access the network (for the first
    # time) in each of the RA blocks
    newUEs = np.random.binomial(K0, pA, size=numrounds)

    # Store temp results
    alphawidehat = np.zeros((numrounds, taup, numrepetitions))
    Yt_norms = np.zeros((numrounds, taup, L, numrepetitions))
    Atilde_ = np.zeros((numrounds, taup, L, numrepetitions))

    # Go through all rounds
    for ss in range(numrounds):

        # Extract number of acessing UEs
        numUEs = newUEs[ss]

        # Check existence of transmission
        if numUEs != 0:


            #####
            # Generating UEs
            #####


            # Generate UEs locations
            UElocations = squareLength*(np.random.rand(numUEs) + 1j*np.random.rand(numUEs))

            # Compute UEs distances to each AP
            UEdistances = np.abs(UElocations[:, np.newaxis] - APpositions)

            # Compute average channel gains according to Eq. (1)
            channel_gains = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistances**2 + 10**2)))/10)


            #####
            # Pilot transmission
            #####


            # Randomize which pilot each UE chose
            pilotSelections = np.random.randint(1, taup+1, size=numUEs)
            pilotSelections += -1

            # Generate channel matrix at each AP equipped with N antennas
            G_ = np.sqrt(channel_gains[None, :, :, None]/2)*(np.random.randn(N, numUEs, L, numrepetitions) + 1j*np.random.randn(N, numUEs, L, numrepetitions))

            # Get list of active pilots
            activePilots = np.unique(pilotSelections)


            # Go through all active RA pilots
            for tt in activePilots:

                # Extract UEs that transmit pilot t
                UEindices = np.where(pilotSelections == tt)[0]

                # Compute received signal according to Eq. (4)
                Yt_ = np.sqrt(p * taup)*(G_[:, UEindices, :, :]).sum(axis=1) + n_[ss, tt, :, :, :]

                # Store l2-norms
                Yt_norms[ss, tt, :] = np.linalg.norm(Yt_, axis=0)

                # Obtain pilot activity vector according to Eq. (8)
                atilde_t = (1/N) * Yt_norms[ss, tt]**2
                atilde_t[atilde_t < sigma2] = 0.0

                # Compute alphawidehat according to Eq. (34)
                alphawidehat[ss, tt, :] = ((atilde_t - sigma2).sum(axis=0))

                # Obtain the t-th row of the pilot activity matrix according 
                # to Eq. (8)
                Atilde_[ss, tt, :, :] = atilde_t


            #####
            # Lmax practical
            #####


            # Average out pilot activity matrix
            bar_Atilde_  = Atilde_[ss][:, :, :].mean(axis=-1)

            # Comparison average threshold (Algorithm 1 - Step 16) 
            epsilon = bar_Atilde_.mean(axis=-1)
            
            # Perform element-wise comparison (Algorithm 1 - Step 17) 
            Lt = np.array([(bar_Atilde_[tt, :] >= epsilon[tt]).sum() for tt in activePilots]).astype(np.float_)
            Lt[Lt == 0.0] = np.nan

            # Get and store Lmax_r according to Algorithm 1 - Step 21 disregarding
            # null pilots
            Lmax_practical[kk] += np.nanmean(Lt) / numrounds

    # Get final Lmax according to Algorithm 1 - Step 23
    Lmax_practical[kk] = np.ceil(Lmax_practical[kk])


    #####
    # Average effective power
    #####

    # Extract current Lmax
    Lmax = int(Lmax_practical[kk])

    effective_power = np.zeros((numrounds, taup, numrepetitions))

    # Go through all rounds
    for ss in range(numrounds):

        # Go thorugh all repetitions
        for rr in range(numrepetitions):

            # Go through all RA pilots
            for tt in range(taup):

                # Obtain set of pilot-serving APs (Definition 2)
                Pcal = np.argsort(Atilde_[ss][tt, :, rr])[-Lmax:]
                Pcal = np.delete(Pcal, Atilde_[ss][tt, Pcal, rr] == 0.0)

                # Compute effective according to Eq. (36)
                if len(Pcal):

                    
                    effective_power[ss, tt, rr] = ((ql / N) * (Yt_norms[ss, tt, Pcal, rr]**2) / alphawidehat[ss, tt, rr]).mean()

                else: 

                    effective_power[ss, tt, rr] = np.nan

    # Compute average effective power
    avg_effective_power = np.nanmean(effective_power)

    # Compute average delta 
    delta_practical[kk] = np.sqrt(ql / avg_effective_power)

    print('\t[|U|] elapsed ' + str(np.round(time.time() - timer_start)) + ' seconds.\n')

print("total simulation time was " + str(np.round(total_time, 4)) + " seconds.\n")
print("wait for Lookup saving...\n")

# Change data type of Lmax
Lmax_practical = Lmax_practical.astype(int)

np.savez("lookup/lookup_fig07_08_practical.npz",
    Lmax_practical=Lmax_practical,
    delta_practical=delta_practical
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")
