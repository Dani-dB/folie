import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def V(x, y):
    A = 1
    B = 1
    C = 2
    V = A * (x**2 - B)**2 + C * y**2
    return V

L = 2
max_V = 10 #limit for plotting
max_A = 5

# Create figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

plot_PES = True
if plot_PES:
    x = np.linspace(-L, L, 100)
    y = np.linspace(-L, L, 100)
    X, Y = np.meshgrid(x, y)

    Z = V(X, Y)

    c = ax[0].contourf(X, Y, Z, levels=np.linspace(0,max_V,30), cmap='viridis')
    fig.colorbar(c, ax=ax[0], label='V(x, y)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('2D Potential Energy Surface')

ax[0].set_xlim([-L,L])
ax[0].set_ylim([-L,L])


# Number of colvars to use between theta=0 (x) and pi/2 (y)
n_theta = 5

# Illustrate different colvars
for theta in np.linspace(0, np.pi / 2, n_theta):
    dx = L * np.cos(theta) * 1.5
    dy = L * np.sin(theta) * 1.5
    ax[0].plot([-dx, dx], [-dy, dy], linewidth=2)


# Define the collective variable q(x, y)
def q(theta, x, y):
    return np.cos(theta) * x + np.sin(theta) * y

# Importance sampling parameters
n_samples = 1000000  # Number of samples
beta = 1.0         # Inverse temperature (1/k_B T)
x_min, x_max = -L, L
y_min, y_max = -L, L

theta = 0
#  theta = np.pi / 4
# theta = np.pi / 2

# Generate uniform samples
x_samples = np.random.uniform(x_min, x_max, n_samples)
y_samples = np.random.uniform(y_min, y_max, n_samples)

# Compute Boltzmann weights
weights = np.exp(-beta * V(x_samples, y_samples))

def plot_FE(theta):
    # Compute the collective variable values
    q_values = q(theta, x_samples, y_samples)

    # Weighted histogram the q values to estimate P(q)
    q_bins = np.linspace(-3, 3, 100)
    # hist, bin_edges = np.histogram(q_values, bins=q_bins, weights=weights, density=True)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute the KDE of the q values with weights
    kde = gaussian_kde(q_values, weights=weights, bw_method='scott')
    q_bins = np.linspace(-3, 3, 100)
    P_q = kde(q_bins)

    # Compute the free energy A(q) = -k_B T ln P(q)
    A_q = -1 / beta * np.log(P_q + 1e-20)

    cx = np.cos(theta)
    cy = np.sin(theta)
    # Plot the free energy profile -- Shift so that the minimum A(q) is zero
    ax[1].plot(q_bins, A_q - np.min(A_q), label=f'{cx:.2f} x + {cy:.2f} y')



for theta in np.linspace(0, np.pi / 2, n_theta):
    plot_FE(theta)


ax[1].set_xlim(-2,2)
ax[1].set_ylim(0, max_A)
ax[1].set_xlabel('q')
ax[1].set_ylabel('Free Energy A(q)')

ax[1].legend()
ax[1].set_title('Free Energy Profile along different variables')


plt.tight_layout()
plt.show()
