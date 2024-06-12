import numpy as np
import matplotlib.pyplot as plt

def V(x, y):
    A = 2
    B = 1
    C = 5
    V = A * (x**2 - B)**2 + C * y**2
    return V

L = 2

x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
X, Y = np.meshgrid(x, y)

Z = V(X, Y)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('V(x, y)')
ax.set_title('3D View of the 2D Potential Energy Surface')

plt.show()


# Define the collective variable q(x, y)
def q(x, y):
    # theta = 0
    # theta = np.pi / 4
    theta = np.pi / 2
    return np.cos(theta) * x + np.sin(theta) * y

# Importance sampling parameters
n_samples = 100000  # Number of samples
beta = 1.0          # Inverse temperature (1/k_B T)
x_min, x_max = -L, L
y_min, y_max = -L, L

# Generate uniform samples
x_samples = np.random.uniform(x_min, x_max, n_samples)
y_samples = np.random.uniform(y_min, y_max, n_samples)

# Compute Boltzmann weights
weights = np.exp(-beta * V(x_samples, y_samples))

# Compute the collective variable values
q_values = q(x_samples, y_samples)

# Weighted histogram the q values to estimate P(q)
q_bins = np.linspace(-3, 3, 100)
hist, bin_edges = np.histogram(q_values, bins=q_bins, weights=weights, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute the free energy A(q) = -k_B T ln P(q)
P_q = hist + 1e-10  # Avoid log(0)
A_q = -1 / beta * np.log(P_q)

# Plot the free energy profile
plt.plot(bin_centers, A_q - np.min(A_q))  # Shift so that the minimum A(q) is zero
plt.xlabel('q')
plt.ylabel('Free Energy A(q)')
plt.title('Free Energy Profile')
plt.show()