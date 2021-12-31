########################################
#   plot_fig02_illustration.py
#
#   Description. Script used to plot Fig. 2 of the paper.
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

axis_font = {'size':'8'}

plt.rcParams.update({'font.size': 8})

matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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
taup = 3

# Define maximum number of pilot-serving APs
Lmax = 10

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
# Simulation
########################################
print("--------------------------------------------------")
print("Fig. 02: illustration")
print("--------------------------------------------------\n")

print("wait for the plot...\n")

# Define UEs locations
UEpositions = np.array(
    [150+1j*150, 250+1j*150, 100+1j*300, 300+1j*300, 50+1j*50, 350+1j*50]
    )

# Compute UEs distances to each AP
UEdistances = abs(APpositions - UEpositions[:, None])

# Compute average channel gains according to Eq. (1)
betas = 10**((94.0 - 30.5 - 36.7 * np.log10(np.sqrt(UEdistances**2 + 10**2)))/10)

# Assign a pilot to each UE
pilotSelection = np.array([0, 0, 1, 1, 2, 2])


#####


# Prepare to store set of pilot-serving APs
Pcal_store = np.zeros((3, Lmax), dtype=np.uint)

# Go through all RA pilots
for tt in range(3):

    # Extract UEs that transmit pilot t
    UEindices = np.where(pilotSelection == tt)[0]

    # Obtain collision size
    collisionSize = len(UEindices)

    # Compute asymptotic pilot activity vector according to Eq. (6)
    atilde_t = p * taup * betas[UEindices, :].sum(axis=0) + sigma2

    # Obtain set of pilot-serving APs (Definition 2)
    Pcal = np.argsort(atilde_t)[-Lmax:]
    Pcal = np.delete(Pcal, atilde_t[Pcal] == 0)

    # Store it
    Pcal_store[tt, :] = Pcal

########################################
# Plot
########################################

# Compute limit distance according to Eq. (3)
limit_distance = (ql * 10**((94.0 - 30.5)/10))**(1/3.67)

# Draw circles defining nearby region of each UE
circle1 = plt.Circle((UEpositions[0].real, UEpositions[0].imag), radius=limit_distance, color='#E95C20FF', alpha=0.25)
circle2 = plt.Circle((UEpositions[1].real, UEpositions[1].imag), radius=limit_distance, color='#E95C20FF', alpha=0.25)

circle3 = plt.Circle((UEpositions[2].real, UEpositions[2].imag), radius=limit_distance, color='#006747FF', alpha=0.25)
circle4 = plt.Circle((UEpositions[3].real, UEpositions[3].imag), radius=limit_distance, color='#006747FF', alpha=0.25)

circle5 = plt.Circle((UEpositions[4].real, UEpositions[4].imag), radius=limit_distance, color='#4F2C1DFF', alpha=0.25)
circle6 = plt.Circle((UEpositions[5].real, UEpositions[5].imag), radius=limit_distance, color='#4F2C1DFF', alpha=0.25)

# Actual plot
fig, ax = plt.subplots(figsize=(3.15, 3))

ax.plot(APpositions.real, APpositions.imag, '.', markersize=4, color='black', label="AP")

ax.plot(UEpositions[0:2].real, UEpositions[0:2].imag, linewidth=0, marker='s', markersize=5, color='#E95C20FF', label="UEs in $\mathcal{S}_1$")
ax.plot(UEpositions[2:4].real, UEpositions[2:4].imag, linewidth=0, marker='<', markersize=5, color='#006747FF', label="UEs in $\mathcal{S}_2$")
ax.plot(UEpositions[4:6].real, UEpositions[4:6].imag, linewidth=0, marker='>', markersize=5, color='#4F2C1DFF', label="UEs in $\mathcal{S}_3$")

ax.add_patch(circle1)
ax.add_patch(circle2)

ax.add_patch(circle3)
ax.add_patch(circle4)

ax.add_patch(circle5)
ax.add_patch(circle6)

ax.plot(APpositions[Pcal_store[0]].real, APpositions[Pcal_store[0]].imag, 'x', markersize=8, color='#E95C20FF', label="APs in $\mathcal{P}_1$")
ax.plot(APpositions[Pcal_store[1]].real, APpositions[Pcal_store[1]].imag, '+', markersize=8, color='#006747FF', label="APs in $\mathcal{P}_2$")
ax.plot(APpositions[Pcal_store[2]].real, APpositions[Pcal_store[2]].imag, '1', markersize=8, color='#4F2C1DFF', label="APs in $\mathcal{P}_3$")

ax.plot([UEpositions[2].real, UEpositions[2].real + limit_distance], [UEpositions[2].imag, UEpositions[2].imag], linewidth=1, linestyle='-', color='black')
ax.text(UEpositions[2].real+35, UEpositions[2].imag+5, r"$d^{\text{lim}}$")

ax.set_xlabel('$x$ spatial dimension [m]')
ax.set_ylabel('$y$ spatial dimension [m]')

ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 100, 200, 300, 400])

ax.legend(fontsize='x-small', loc='lower center', markerscale=.75, ncol=3)

plt.show()

print("------------------- all done :) ------------------")