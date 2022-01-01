########################################
#   lookup_fig05_06_best_pair.py
#
#   Description. Script used to obtain a lookup table for the best parameter 
#   pair (Csize, Lmax) given system parameters and for different values of 
#   number of antennas per AP, N.
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
#       - data_fig05_barplot_cellfree.py
#       - plot_fig05_barplot.py
#       - plot_fig06_neb_nmse.py
#
#   Please, make sure that you have the files produced by:
#       
#       - lookup_fig05_06_delta.py
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
# SELECTION
########################################

# Choose the estimator
estimator = "est1"
estimator = "est2"
estimator = "est3"

# Choose the figure
figure = "fig05"
#figure = "fig06"

########################################
# Lookup table
########################################

# Load possible values of delta for Estimator 3
if estimator == "est3":

    load = np.load("lookup/lookup_fig05_06_delta.npz", allow_pickle=True)
    delta_lookup = load["delta"]
    delta_lookup = delta_lookup.item()

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

# Set the number of setups
numsetups = 100

# Set the number of channel realizations
numchannel = 100

# Range of number of nearby APs
Csize_range = np.arange(1, 8)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, L+1)

# Check figure
if figure == "fig05":

    # Range of collision sizes
    collisions = np.arange(1, 11)

    # Range of number of antennas per AP
    N_range = np.array([8])

    # Extract maximum number of antennas per AP
    N_max = np.max(N_range)

elif figure == "fig06":

    # Range of collision sizes
    collisions = np.array([2])

    # Range of number of antennas per AP
    N_range = np.arange(1, 11)

    # Extract maximum number of antennas per AP
    N_max = np.max(N_range)

########################################
# Simulation
########################################
print("--------------------------------------------------")
print("Lookup Fig 05 & 06: best pair")
print("\testimator: " + estimator)
print("\tfigure: " + figure)
print("--------------------------------------------------\n")

# Store total time 
total_time = time.time()

# Store enumeration of L
enumerationL = np.arange(L)

# Prepare to save simulation results
best_pair = np.zeros((collisions.size, N_range.size), dtype=tuple)


#####


# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(numsetups, N_max, L, numchannel) + 1j*np.random.randn(numsetups, N_max, L, numchannel))

# Generate noise realization at UEs
eta = np.sqrt(sigma2/2)*(np.random.randn(numsetups, collisions.max(), numchannel) + 1j*np.random.randn(numsetups, collisions.max(), numchannel))


# Go through all collision sizes
for cc, collisionSize in enumerate(collisions):

    # Storing time
    timer_start = time.time()

    # Print current data point
    print(f"\tcollision: {cc}/{collisions.size-1}")


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

        # Storing time
        timer_nn = time.time()

        # Print current data point
        print(f"\t\tN: {nn}/{N_range.size-1}")

        # Compute received signal according to Eq. (4)
        Yt_ = np.sqrt(p * taup) * G_[:, :N, :, :, :].sum(axis=2) +  n_[:, :N, :, :]

        # Store l2-norms of Yt
        Yt_norms = np.linalg.norm(Yt_, axis=1)

        # Obtain pilot activity vector according to Eq. (8)
        atilde_t = (1/N) * Yt_norms**2
        atilde_t[atilde_t < sigma2] = 0.0

        # Prepare to save nmse
        nmse = np.zeros((Csize_range.size, Lmax_range.size, numsetups, collisionSize, numchannel))


        # Go through all setups
        for ss in range(numsetups):


            # Go through all channel realizations
            for ch in range(numchannel):


                # Go through all Lmax values
                for lm, Lmax in enumerate(Lmax_range):

                    # Obtain set of pilot-serving APs (Definition 2)
                    Pcal = np.argsort(atilde_t[ss, :, ch])[-Lmax:]
                    Pcal = np.delete(Pcal, atilde_t[ss, Pcal, ch] == 0)

    
                    #####
                    # SUCRe - step 2
                    #####


                    if estimator == 'est3':

                        # Denominator according to Eqs. (34) and (35)
                        den = np.sqrt(N * (atilde_t[ss, :, ch] - sigma2).sum())

                        # Compute precoded DL signal according to Eq. (35)
                        Vt_ = np.sqrt(ql) * (Yt_[ss][:, Pcal, ch] / den)

                    else:

                        # Compute precoded DL signal according to Eq. (10)
                        Vt_ = np.sqrt(ql) * (Yt_[ss][:, Pcal, ch] / Yt_norms[ss, Pcal, ch])

                    # Compute true total UL signal power of colliding UEs 
                    # according to Eq. (16)
                    alpha_true = p * taup * channel_gains[ss, :, Pcal].sum()


                    # Go through all colliding UEs
                    for k in range(collisionSize):

                        # Compute received DL signal at UE k according to Eq. 
                        # (12)
                        z_k = np.sqrt(taup) * ((G_[ss][:N, k, Pcal, ch].conj() * Vt_).sum()) + eta[ss, k, ch]

                        # Compute approximation
                        cte = z_k.real/np.sqrt(N)

                        # Obtain natural set of nearby APs of UE k (Definition 1)
                        checkCcal_k = enumerationL[ql * channel_gains[ss, k, :] > sigma2]

                        if len(checkCcal_k) == 0:
                            checkCcal_k = np.array([np.argmax(ql * channel_gains[ss, k, :])])

                        # Sort importance of each AP for UE k 
                        sort = np.argsort(ql * channel_gains[ss, k, :])

                        # Go through all different values of Csize
                        for cs, Csize in enumerate(Csize_range):        

                            # Obtain set of nearby APs of UE k (Definition 1)
                            Ccal_k = sort[-Csize:]

                            if len(Ccal_k) > len(checkCcal_k):
                                Ccal_k = checkCcal_k


                            #####
                            # Estimation
                            #####


                            # Compute constant 
                            num = np.sqrt(ql * p) * taup * channel_gains[ss, k, Ccal_k]
                    
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
                                delta = delta_lookup[(collisionSize, N, Lmax)]

                                # Compute new constant according to Eq. (38)
                                underline_cte = delta * (z_k.real - sigma2)/np.sqrt(N)

                                # Compute estimate according to Eq. (40)
                                alphahat = (num.sum() / underline_cte)**2

                            # Compute own total UL signal power in Eq. (15)
                            gamma = p * taup * channel_gains[ss, k, Ccal_k].sum()

                            # Avoiding underestimation
                            if alphahat < gamma:
                                alphahat = gamma

                            # Get and store inner loop stats
                            nmse[cs, lm, ss, k, ch] = (np.abs(alphahat - alpha_true)**2)/(alpha_true**2)

        # Compute median nmse 
        median_nmse = np.median(nmse.mean(axis=-1), axis=(-1, -2))

        # Get and store best pair
        best_index = np.unravel_index(median_nmse.argmin(), median_nmse.shape)
        best_pair[cc, nn] = tuple(x + 1 for x in best_index)

        print('\t\t[N] elapsed ' + str(np.round(time.time() - timer_nn, 4)) + ' seconds.\n')

    total_time += (time.time() - timer_start)
    print('\t[collision] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')

print("total simulation time was " + str(np.round(total_time, 4)) + " seconds.\n")
print("wait for Lookup saving...\n")

# Save as a dictionary 
dict = {}

# Go through all collision sizes
for cs, collisionSize in enumerate(collisions):

    # Go through all values of N
    for nn, N in enumerate(N_range):

        dict[(collisionSize, N)] = best_pair[cs, nn]

# Save simulation results
np.savez("lookup/lookup_" + figure + "_best_pair_" + estimator + ".npz",
    best_pair=dict
)

print("the lookup table has been saved in the /lookup folder.\n")

print("------------------- all done :) ------------------")