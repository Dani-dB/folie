import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from  MC_FES import MCFreeEnergy

def V(x, y):
    A = 1
    B = 1
    C = 2
    V = A * (x**2 - B)**2 + C * y**2
    return V

# Define the collective variable q(x, y)
def q(theta, x, y):
    return np.cos(theta) * x + np.sin(theta) * y

theta = np.pi/3
abs, ord =MCFreeEnergy(q=q,V=V,theta=theta)

max_A= 5
fid, ax = plt.subplots()
ax.set_xlim(-2,2)
ax.set_ylim(0, max_A)
ax.set_xlabel('q')
ax.set_ylabel('Free Energy A(q)')
ax.plot(abs,ord)
ax.legend()
ax.set_title('Free Energy Profile along different variables')

plt.show()