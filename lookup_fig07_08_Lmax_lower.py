########################################
#   lookup_fig07_08_Lmax_lower.py
#
#   Description. Script used to obtain a lookup table for Lmax related to the 
#   lower bound of performance obtained when the APs known the best choice of 
#   Lmax given a collision size. Best here is measured in terms of the median 
#   NMSE. You should choose the estimator.
#
#   Author. @victorcroisfelt
#
#   Date. December 01, 2022
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
#       - data_fig07_08_cellfree_lower.py
#
#   Please, make sure that you have the files produced by:
#       
#       - lookup_fig07_08_delta.py
#
########################################
import numpy as np

import time

from settings_fig07_08 import *

from tempfile import TemporaryFile

########################################
# SELECTION
########################################

# Choose the estimator
estimator = "est1"
#estimator = "est2"
estimator = "est3"

########################################
# Lookup table
########################################

# Load possible values of delta for Estimator 3
if estimator == "est3":

    load = np.load("lookup/lookup_fig07_08_delta.npz", allow_pickle=True)
    delta_lookup = load["delta"]
    delta_lookup = delta_lookup.item()

########################################
# Simulation parameters
########################################

# Set the number of setups
numsetups = 100

# Set the number of channel realizations
numchannel = 100

# Range of collision sizes
collisions = np.arange(1, 51)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, L+1)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Lookup Fig 07 & 08: cell-free -- lower bound")
print("\testimator: " + estimator)
print("\tN = " + str(N))
print("--------------------------------------------------\n")

# Store total time 
total_time = 0.0

# Store enumeration of L
enumerationL = np.arange(L)

# Prepare to save simulation results
best_Lmax = np.zeros(collisions.size, dtype=int)


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(N, L, numchannel) + 1j*np.random.randn(N, L, numchannel))

# Generate noise realization at UEs
eta = np.sqrt(sigma2/2)*(np.random.randn(numsetups, collisions.max(), numchannel) + 1j*np.random.randn(numsetups, collisions.max(), numchannel))


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


    #####
    # Common signals
    #####
    

    # Go through all setups
    for ss in range(numsetups):

        # Extract current average channel gains
        channel_gains_current = channel_gains[ss, :, :]

        # Generate channel matrix for each AP equipped with N antennas
        G_ = np.sqrt(channel_gains_current[None, :, :, None]/2) * (np.random.randn(N, collisionSize, L, numchannel) + 1j*np.random.randn(N, collisionSize, L, numchannel))

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * G_.sum(axis=1) + n_

        # Obtain pilot activity matrix according to Eq. (8)
        atilde_t = (1/N) * np.linalg.norm(Yt_, axis=0)**2
        atilde_t[atilde_t < sigma2] = 0.0

        np.savez("temp/temp_" + str(ss) + ".npz", G_=G_, Yt_=Yt_, atilde_t=atilde_t, allow_pickle=True)


    #####
    # Binary search
    #####


    # Prepare to save median NMSE
    median_nmse = np.zeros((3))

    # Initialize pointers
    left = 0
    right = (Lmax_range.size - 1)

    # Initialize memory
    memory = {}


    while True:

        # Get middle element
        middle = int(np.floor((left + right)/2))

        # Create an array with three evaluation points 
        indexes = np.array([left, middle, right], dtype=int)


        # Go through each evaluation point
        for lnew, lold in enumerate(indexes):

            # Check memory
            if lold in memory.keys():
                median_nmse[lnew] = memory[lold]

            else:

                # Prepare to save nmse
                nmse = np.zeros((numsetups, collisionSize, numchannel))


                # Go through all setups
                for ss in range(numsetups):

                    # Load
                    load = np.load("temp/temp_" + str(ss) + ".npz", allow_pickle=True)

                    G_ = load["G_"]
                    Yt_ = load["Yt_"]
                    atilde_t = load["atilde_t"]

                    # Obtain set of pilot-serving APs (Definition 2)
                    Pcal = np.argsort(atilde_t, axis=0)[-Lmax_range[lold]:]
                    

                    # Go through all channel realizations
                    for ch in range(numchannel):

                        # Extract current Pcal
                        Pcal_current = Pcal[:, ch]

                        # Check if all APs in Pcal are really valid ones
                        Pcal_current = np.delete(Pcal_current, atilde_t[Pcal_current, ch] == 0)


                        #####
                        # SUCRe: step 3
                        #####


                        if estimator == 'est3':

                            # Denominator according to Eqs. (34) and (35)
                            den = np.sqrt(N * np.maximum(atilde_t[Pcal_current, ch] - sigma2, np.zeros(atilde_t[Pcal_current, ch].shape)).sum())

                            # Compute precoded DL signal according to Eq. (35)
                            Vt_ = np.sqrt(ql) * (Yt_[:, Pcal_current, ch] / den)

                        else:

                            # Compute precoded DL signal according to Eq. (10)
                            Vt_ = np.sqrt(ql) * (Yt_[:, Pcal_current, ch] / np.linalg.norm(Yt_[:, Pcal_current, ch], axis=0))

                        # Compute true total UL signal power of colliding UEs 
                        # according to Eq. (16)
                        alpha_true = p * taup * channel_gains[ss, :, Pcal_current].sum()


                        # Go through all colliding UEs
                        for k in range(collisionSize):

                            # Compute received DL signal at UE k according to Eq. 
                            # (12)
                            z_k = np.sqrt(taup) * (G_[:, k, Pcal_current, ch].conj() * Vt_).sum() + eta[ss, k, ch]

                            # Obtain natural set of nearby APs of UE k (Definition 1)
                            checkCcal_k = enumerationL[ql * channel_gains[ss, k, :] > sigma2]

                            if len(checkCcal_k) == 0:
                                checkCcal_k = np.array([np.argmax(ql * channel_gains[ss, k, :])])


                            #####
                            # Estimation
                            #####


                            # Compute constants
                            cte = z_k.real/np.sqrt(N)
                            num = np.sqrt(ql * p) * taup * channel_gains[ss, k, checkCcal_k]

                            
                            if estimator == 'est1':

                                # Compute estimate according to Eq. (28)
                                alphahat = ((num.sum()/cte)**2) - sigma2

                            elif estimator == 'est2':

                                num23 = num**(2/3)
                                cte2 = (num23.sum()/cte)**2

                                # Compute estimate according to Eq. (32)
                                alphahat = (cte2 * num23 - sigma2).sum()

                            elif estimator == 'est3':

                                # Define compensation factor in Eq. (39)
                                delta = delta_lookup[(collisionSize, Lmax_range[lold])]

                                # Compute new constant according to Eq. (38)
                                underline_cte = delta * (z_k.real - sigma2)/np.sqrt(N)

                                # Compute estimate according to Eq. (40)
                                alphahat = (num.sum() / underline_cte)**2

                            # Compute own total UL signal power in Eq. (15)
                            gamma = p * taup * channel_gains[ss, k, checkCcal_k].sum()

                            # Avoiding underestimation
                            if alphahat < gamma:
                                alphahat = gamma

                            # Get and store inner loop stats
                            nmse[ss, k, ch] = (np.abs(alphahat - alpha_true)**2)/(alpha_true**2)


                # Compute median nmse 
                median_nmse[lnew] = np.median(nmse.mean(axis=-1), axis=(-1, -2))

                # Store on memory
                memory[lold] = median_nmse[lnew]

        # Check binary search step
        if median_nmse[1] <= median_nmse[2]:
            right = middle
                    
        elif median_nmse[0] > median_nmse[1]:
            left = middle 

        else: 
            best_Lmax[cs] = Lmax_range[indexes[median_nmse.argmin()]]
            break

    total_time += (time.time() - timer_start)
    print('\t[collision] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')

print("total simulation time was " + str(np.round(total_time, 4)) + " seconds.\n")
print("wait for Lookup saving...\n")

# Save as a dictionary 
dict = {}

# Go through all collision sizes
for ss, collisionSize in enumerate(collisions):
    dict[collisionSize] = best_Lmax[ss]

# Save simulation results
np.savez('Lookup/lookup_fig07_08_Lmax_' + estimator + '_lower.npz',
    best_Lmax=dict
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")