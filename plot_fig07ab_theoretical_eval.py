########################################
#   plot_fig07ab_theoretical_eval.py
#
#   Description. Script used to plot Figs. 7a and 7b of the paper.
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

from scipy import integrate

import matplotlib
import matplotlib.pyplot as plt

import warnings

########################################
# Preamble
########################################

# Comment the line below to see possible warnings related to python version 
# issues
warnings.filterwarnings("ignore")

axis_font = {'size':'12'}

plt.rcParams.update({'font.size': 12})

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

########################################
# Fixed parameters
########################################

# Square length
ell = 400

# Noise variance
sigma2 = 10**(-94/10)

# Multiplicative pathloss constant -- Eq. (1)
Omega = 10**(-30.5/10)

# Pathloss exponent -- Eq. (1)
gamma = 3.67

# Probability of access
Pa = 0.001

# Number of RA pilot signals
taup = 5

########################################
# Variable parameters
########################################

# Range of number of APs
Lrange = np.array([4, 16, 64, 100, 256])

# Range of number of inactive users
K0values = np.concatenate((np.array([100, 250]), np.arange(500, 1000, 100), np.arange(1000, 50100, 100, dtype=np.uint), np.arange(16000, 51000, 1000, dtype=np.uint)))

########################################
# Auxiliar functions
########################################

def distance_cdf(ell, d):
    """ 
    Compute CDF of the distance between two winning UEs according to Eq. (20). 
    We always consider that 0 <= d <= ell. 

    Parameters
    ----------
    ell : float
        Square length in meters.

    d : float
        Distance point being evaluated.

    Returns
    -------

    Fd : float between 0 and 1
        CDF of the distance.

    """

    g0 = (2/3) * ell * d**3
    gd = ell**2 * d**2 * np.pi/2 - ell*d**3 + (ell/3)*d**3 + (ell**2) * (d**2) / 2 - (d**4)/4

    Fd = 2 * (gd - g0) / (ell**4)

    return Fd

########################################
# Simulation
########################################

print('--------------------------------------------------')
print(f"Fig. 07 (a) and (b): theoretical evaluation")
print('--------------------------------------------------\n')

print("wait for the plots...\n")

# Prepare simulation results
rhoAdom_k = np.zeros(shape=(Lrange.size, K0values.size))
Psi_k = np.zeros(shape=(Lrange.size, K0values.size))

# Go through all values of L
for ll, L in enumerate(Lrange):

    # Compute current DL transmit power per AP
    ql = 200/L

    # Compute limit distance according to Eq. (3)
    dlim = (Omega * ql / sigma2)**(1/gamma)

    # Compute total area of UE k
    A_k = np.pi * dlim**2

    # Compute average overlapping area according to Eqs. (22), (24), and (25)
    integrand1 = lambda x: np.real(2*dlim**2 * np.arccos(((x/2/dlim) if x <= 2*dlim else 1.0)) * (2/(ell**4)) * ( (1 + np.pi)*ell**2*x - 4*ell*x**2 - x**3))
    integrand2 = lambda x: np.real(x/2 * np.sqrt((4 * dlim**2 - x**2) if 4 * dlim**2 >= x**2 else 0.0) * (2/(ell**4)) * ( (1 + np.pi)*ell**2*x - 4*ell*x**2 - x**3))

    result1 = integrate.quad(integrand1, 0.0, ell)
    result2 = integrate.quad(integrand2, 0.0, 4 * dlim**2)

    Aovlp_k = result1[0] - result2[0]

    # Compute probability that the distance of two UEs is less than 2dlim
    F2dlim = distance_cdf(ell, 2*dlim)

    # Go through all different number of inactive UEs
    for ss, K0 in enumerate(K0values):

        # Compute average collision size
        avg_collisionSize = (Pa / taup) * K0
        bar_collisionSize = np.max([avg_collisionSize - 1, 0.0])

        # Compute average dominant area according to Eq. (21)
        Adom_k = A_k - F2dlim * bar_collisionSize * Aovlp_k

        # Compute and store Psi according to Eq. (26)
        Psi_k[ll, ss] = Adom_k/A_k

        # Compute and store number of exclusive pilot-serving APs
        rhoAdom_k[ll, ss] = (L/ell**2) * Adom_k

# Get special indexes
special_index = np.empty(Lrange.size)
special_index[:] = np.nan

# Go through all values of L
for ll, L in enumerate(Lrange):

    try: 
        # Treat negative values
        index = np.where(Psi_k[ll, :] < 0.0)[0][0]
        Psi_k[ll, index:] = np.nan
        rhoAdom_k[ll, index:] = np.nan
    except:
        pass

    index = 0

    while True:

        if index == (len(K0values) - 1):
            break

        if Psi_k[ll, index + 1] > Psi_k[ll, index]:
            special_index[ll] = index
            break 

        index += 1

########################################
# Plot
########################################

# Fig. 07a
fig, ax = plt.subplots(figsize=(4/3 * 3.15, 2))

ax.plot(np.logspace(2, np.log10(50000), num=6), np.ones(6), color='black', linewidth=0.0, marker='x', label='Reference line: at least one AP')

ax.plot(K0values, rhoAdom_k[0], linewidth=1.5, linestyle='-', label='$L=4$ APs')
ax.plot(K0values, rhoAdom_k[1], linewidth=1.5, linestyle='--', label='$L=16$ APs')
ax.plot(K0values, rhoAdom_k[2], linewidth=1.5, linestyle='-.', label='$L=64$ APs')
ax.plot(K0values, rhoAdom_k[3], linewidth=1.5, linestyle=':', label='$L=100$ APs')

ax.set_xscale('log', base=10)

ax.set_xlim([900, 52000])

ax.set_xlabel('number of inactive users $|\mathcal{U}|$')
ax.set_ylabel(r'$\rho A^{\text{dom}}_{k}$')

ax.legend(fontsize='x-small')

ax.grid(linestyle='--', visible=True, alpha=0.25)

plt.show()

# Fig. 07b
fig, ax = plt.subplots(figsize=(4/3 * 3.15, 2))

ax.plot(K0values, Psi_k[0], linewidth=1.5, linestyle='-', label='$\Psi_k$: $L=4$ APs')
ax.plot(K0values, Psi_k[1], linewidth=1.5, linestyle='--', label='$\Psi_k$: $L=16$ APs')
ax.plot(K0values, Psi_k[2], linewidth=1.5, linestyle='-.', label='$\Psi_k$: $L=64$ APs')
ax.plot(K0values, Psi_k[3], linewidth=1.5, linestyle=':', label='$\Psi_k$: $L=100$ APs')

ax.set_xscale('log', base=10)

ax.set_xlim([900, 52000])

ax.set_xlabel('number of inactive users $|\mathcal{U}|$')
ax.set_ylabel('$\Psi_k$')

ax.legend(fontsize='x-small')

ax.grid(linestyle='--', visible=True, alpha=0.25)

plt.show()

print("------------------- all done :) ------------------")

