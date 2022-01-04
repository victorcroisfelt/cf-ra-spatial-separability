########################################
#   analysis_delta_eff_power_est3.py
#
#   Description. Script used to perform the analysis about the effective power 
#   of Estimator 3 reported in Section V.E-2) of the paper.
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
########################################
import numpy as np

import time

########################################
# System parameters
########################################

# Define number of APs
L = 64

# Define number of antennas per AP
N = 1

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
numRAblocks = 10000

# Range of number of inactive users
K0values = np.concatenate((np.arange(1e2, 1e3, 1e2), np.arange(1e3, 1e4, 2.5e2), np.arange(1e4, 52500, 2.5e3))).astype(int)

# Range of maximum number of pilot-serving APs
Lmax_range = np.arange(1, 64)

########################################
# Simulation
########################################

print("--------------------------------------------------")
print("\t Analysis: Est. 3 -- delta")
print("--------------------------------------------------\n")

# Generate noise realizations at APs
n_ = np.sqrt(sigma2/2)*(np.random.randn(N, L, taup, numRAblocks) + 1j*np.random.randn(N, L, taup, numRAblocks))

# Prepare to save simulation results
avg_effective_power = np.empty((K0values.size, Lmax_range.size, numRAblocks, taup))
avg_effective_power[:] = np.nan

# Go through all different number of inactive UEs
for kk, K0 in enumerate(K0values):

    # Storing time
    timer_kk = time.time()

    # Print current data point
    print(f"\tinactive UEs: {kk}/{len(K0values)-1}")

    # Generate the number of UEs that wish to access the network (for the first
    # time) in each of the RA blocks
    newUEs = np.random.binomial(K0, pA, size=numRAblocks)

    # Go through all RA blocks
    for rr in range(numRAblocks):

        # Generate UEs locations
        newUElocations = squareLength*(np.random.rand(newUEs[rr]) + 1j*np.random.rand(newUEs[rr]))

        # Compute UEs distances to each AP
        newUEdistances = abs(APpositions - newUElocations[:, None])

        # Compute average channel gains according to Eq. (1)
        newBetas = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(newUEdistances**2 + 10**2)))/10)

        # Randomize which pilot each UE chose
        pilotSelections = np.random.randint(1, taup+1, size=newUEs[rr])
        pilotSelections += -1

        # Check existence of transmission
        if newUEs[rr] != 0:

                # Generate channel matrix at each AP equipped with N antennas
                G = np.sqrt(newBetas[None, :, :]/2)*(np.random.randn(N, newUEs[rr], L) + 1j*np.random.randn(N, newUEs[rr], L))

                # Get list of chosen pilots
                chosenPilots = np.unique(pilotSelections)

                # Go through all chosen RA pilots
                for tt in chosenPilots:

                    # Extract UEs that transmit pilot t
                    UEindices = np.where(pilotSelections == tt)[0]

                    # Compute received signal according to Eq. (4)
                    Yt = np.sqrt(p * taup)*(G[:, UEindices, :]).sum(axis=1) + n_[:, :, tt, rr]

                    # Store l2-norms of Yt
                    Yt_norms = np.linalg.norm(Yt, axis=0)

                    # Obtain pilot activity matrix according to Eq. (8)
                    Atilde = (1/N) * Yt_norms**2
                    Atilde[Atilde < sigma2] = 0.0

                    # Go through all Lmax values
                    for ll, Lmax in enumerate(Lmax_range):

                        # Obtain set of pilot-serving APs (Definition 2)
                        Pcal = np.argsort(Atilde)[-Lmax:]
                        Pcal = np.delete(Pcal, Atilde[Pcal] == 0)

                        # Compute alphawidehat according to Eq. (34)
                        alphawidehat = np.maximum(Atilde - sigma2, np.zeros((Atilde.shape))).sum()

                        # Compute effective power per AP l in Pcal according to 
                        # Eq. (36)
                        effective_power_per_ap = (ql / N) * (Yt_norms[Pcal]**2) / alphawidehat
                        
                        # Store average effective power
                        avg_effective_power[kk, ll, rr, tt] = effective_power_per_ap.mean()

    print("\t[|U|] elapsed " + str(np.round(time.time() - timer_kk, 4)) + " seconds.\n")

# Compute average effective power
avg_effective_power = np.nanmean(avg_effective_power, axis=(-1, -2))

print("Average delta in Eq. (39) as a functin of Lmax and |U|:")

# Go through all different number of inactive UEs
for kk, K0 in enumerate(K0values):

    print("K0 = " + str(K0) + ": ", np.round(np.sqrt(ql / avg_effective_power[kk]), 2))

