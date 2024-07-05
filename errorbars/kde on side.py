import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Sample data for the initial plot
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Create the initial plot
fig, ax1 = plt.subplots()
ax1.plot(x, y, label='Original Plot')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')

# Extract the y-values (ordinate) from the initial plot
y_values = y

# Compute the KDE
kde = gaussian_kde(y_values)
y_kde = np.linspace(min(y_values), max(y_values), 100)
kde_values = kde(y_kde)

# Create a new subplot for the KDE plot on the right side
ax2 = fig.add_axes([0.75, 0.1, 0.2, 0.7])
ax2.plot(kde_values, y_kde)
ax2.set_xlabel('Density')
ax2.set_ylabel('Y-axis (from original plot)')
ax2.set_ylim(ax1.get_ylim())

# Adjust layout to make space for the KDE plot
fig.tight_layout(rect=[0, 0, 0.7, 0.8])

plt.show()
