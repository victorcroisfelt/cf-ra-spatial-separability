########################################
#   lookup_fig07e_practical.py
#
#   Description. Script used to obtain a lookup table for Lmax related to the 
#   practical bound of performance obtained using Algorithm 1. This script is
#   independent of the estimator choice.
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
#       - data_fig07e_cellfree.py
#
########################################
import numpy as np

import time

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

########################################
# System parameters
########################################

# UL transmit power
p = 100

# Define noise power
sigma2 = 1

# Number of RA pilot signals
taup = 5

# Number of inactive users
K0 = 10000

########################################
# Geometry
########################################

# Define square length
squareLength = 400

########################################
# SUCRe Parameters
########################################

# Probability that an inactive UE wants to become active in a given block
pA = 0.001

########################################
# Simulation Parameters
########################################

# Set number of random transmission rounds R
numrounds = 100

# Set number of transmission repetitions E
numrepetitions = 100

# Range of number of APs
L_range = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144])

# Range of number of antennas per AP
N_range = np.array([1, 2, 4, 8])

# Get maximum of ranges
L_max = L_range.max()
N_max = N_range.max()

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("\t Lookup Fig 07e: cell-free -- practical")
print("--------------------------------------------------\n")

# Store total time 
total_time = time.time()

# Prepare to save simulation results
Lmax_practical = np.zeros((L_range.size, N_range.size))
delta_practical = np.zeros((L_range.size, N_range.size))


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numrounds, taup, N_max, L_max, numrepetitions) + 1j*np.random.randn(numrounds, taup, N_max, L_max, numrepetitions))

# Generate the number of UEs that wish to access the network (for the first
# time) in each of the RA blocks
newUEs = np.random.binomial(K0, pA, size=numrounds)


# Go through all different number of APs
for ll, L in enumerate(L_range):

    # Storing time
    timer_start = time.time()

    # Print current data point
    print(f"\tL: {ll}/{len(L_range)-1}")

    # Compute DL transmit power per AP
    ql = 200/L

    # Create square grid of APs
    APperdim = int(np.sqrt(L))
    APpositions = np.linspace(squareLength/APperdim, squareLength, APperdim) - squareLength/APperdim/2
    APpositions = APpositions + 1j*APpositions[:, None]
    APpositions = APpositions.reshape(L)

    # Go through all different number of antennas per APs
    for nn, N in enumerate(N_range):

        # Print current data point
        print(f"\t\tN: {nn}/{len(N_range)-1}")

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
                    Yt_ = np.sqrt(p * taup)*(G_[:, UEindices, :, :]).sum(axis=1) + n_[ss, tt, :N, :L, :]

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
                Lmax_practical[ll, nn] += np.nanmean(Lt) / numrounds

        # Get final Lmax according to Algorithm 1 - Step 23
        Lmax_practical[ll, nn] = np.ceil(Lmax_practical[ll, nn])


        #####
        # Average effective power
        #####

        # Extract current Lmax
        Lmax = int(Lmax_practical[ll, nn])

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
        delta_practical[ll, nn] = np.sqrt(ql / avg_effective_power)

    print('\t[L] elapsed ' + str(np.round(time.time() - timer_start)) + ' seconds.\n')

print("total simulation time was " + str(np.round(total_time, 4)) + " seconds.\n")
print("wait for Lookup saving...\n")

# Change data type of Lmax
Lmax_practical = Lmax_practical.astype(int)

dict = {}

# Go through all different number of APs
for ll, L in enumerate(L_range):

    # Go through all different number of antennas per APs
    for nn, N in enumerate(N_range):

        dict[(L, N)] = (Lmax_practical[ll, nn], delta_practical[ll, nn])


np.savez("lookup/lookup_fig07e_practical.npz",
    practical=dict
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")
