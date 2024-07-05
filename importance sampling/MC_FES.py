import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def MCFreeEnergy(colvar,V,n_samples = 1000000,L= 2 ,beta=1):

    # Importance sampling parameters
    x_min, x_max = -L, L
    y_min, y_max = -L, L

    # Generate uniform samples
    x_samples = np.random.uniform(x_min, x_max, n_samples)
    y_samples = np.random.uniform(y_min, y_max, n_samples)
    input = np.transpose(np.array([x_samples, y_samples]))

    # Compute Boltzmann weights
    weights = np.exp(-beta * V(input))
    # Compute the collective variable values
    q_values = colvar(x_samples, y_samples,only_proj=True)

    # Weighted histogram the q values to estimate P(q)
    q_bins = np.linspace(-1.5, 1.5, 200)

    # Compute the KDE of the q values with weights
    kde = gaussian_kde(q_values, weights=weights, bw_method='scott')
    q_bins = np.linspace(-3, 3, 100)
    P_q = kde(q_bins)

    # Compute the free energy A(q) = -k_B T ln P(q)
    A_q = -1 / beta * np.log(P_q + 1e-20)

    return q_bins, A_q





