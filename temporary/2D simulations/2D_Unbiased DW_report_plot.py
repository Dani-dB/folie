"""
================================
2D Double Well
================================

Estimation of an overdamped Langevin.
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import scipy as sc

""" Script for simulation of 2D double well and projection along user provided direction, No fitting is carried out   """
x = np.linspace(-1.8, 1.8, 12)
y = np.linspace(-1.8, 1.8, 12)
input = np.transpose(np.array([x, y]))

D= np.array([0.5])
diff_function = fl.functions.Polynomial(deg=0, coefficients=D * np.eye(2, 2))
a, b = D*5, D*10.0

quartic2d = fl.functions.Quartic2D(a=a, b=b)

X, Y = np.meshgrid(x, y)
print(X.shape, Y.shape)

# Plot potential surface
pot = quartic2d.potential_plot(X, Y)
print(pot.shape)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, pot, rstride=1, cstride=1, cmap="jet", edgecolor="none")
ax.set_xlabel('$X$', fontsize=15)
ax.set_ylabel('$Y$',fontsize=15)
ax.zaxis.set_rotate_label(False) 
ax.set_yticks([-1.5,0, 1.5])
ax.set_xticks([-1.5,0, 1.5])
ax.set_zticks([0,10, 35])
ax.set_zlabel('$A(x,y)$', fontsize=18,rotation = 0)


# Plot Force function
ff = quartic2d.force(input)  # returns x and y components of the force : x_comp =ff[:,0] , y_comp =ff[:,1]
U, V = np.meshgrid(ff[:, 0], ff[:, 1])
fig, ax = plt.subplots()
ax.quiver(x, y, U, V)
ax.set_title("Force",fontsize= 23)

ax.tick_params(axis='both', which='major', labelsize=15)
# ax.set_zlabel('$A(x,y)$', fontsize=18,rotation = 0)
dt=5e-4
model_simu = fl.models.overdamped.Overdamped(force=quartic2d, diffusion=diff_function)
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt)

# initialize positions
ntraj = 50
q0 = np.empty(shape=[ntraj, 2])
for i in range(ntraj):
    for j in range(2):
        q0[i][j] = 0.000

# Calculate Trajectory
time_steps = 3000
data = simulator.run(time_steps, q0, save_every=1)

# Plot the resulting trajectories
fig, axs = plt.subplots()
for n, trj in enumerate(data):
    axs.plot(trj["x"][:, 0], trj["x"][:, 1])
    axs.spines["left"].set_position("center")
    axs.spines["right"].set_color("none")
    axs.spines["bottom"].set_position("center")
    axs.spines["top"].set_color("none")
    axs.xaxis.set_ticks_position("bottom")
    axs.yaxis.set_ticks_position("left")
    axs.set_xlabel("$X(t)$")
    axs.set_ylabel("$Y(t)$")
    axs.set_title("X-Y Trajectory")
    axs.grid()

# plot Trajectories
fig, bb = plt.subplots(1, 2)
for n, trj in enumerate(data):
    bb[0].plot(trj["x"][:, 0])
    bb[1].plot(trj["x"][:, 1])

    # Set visible  axis
    bb[0].spines["right"].set_color("none")
    bb[0].spines["bottom"].set_position("center")
    bb[0].spines["top"].set_color("none")
    bb[0].xaxis.set_ticks_position("bottom")
    bb[0].yaxis.set_ticks_position("left")
    bb[0].set_xlabel("$timestep$")
    bb[0].set_ylabel("$X(t)$")

    # Set visible axis
    bb[1].spines["right"].set_color("none")
    bb[1].spines["bottom"].set_position("center")
    bb[1].spines["top"].set_color("none")
    bb[1].xaxis.set_ticks_position("bottom")
    bb[1].yaxis.set_ticks_position("left")
    bb[1].set_xlabel("$timestep$")
    bb[1].set_ylabel("$Y(t)$")

    bb[0].set_title("X Dynamics")
    bb[1].set_title("Y Dynamics")

#########################################################################
# function to project along a given orthonormal specified by the angle
########################################################################
def project(trajectory,theta):
    x_pt = trajectory[0]*np.cos(theta) + trajectory[1]*np.sin(theta)
    y_pt = -trajectory[0]*np.sin(theta) + trajectory[1]*np.cos(theta)
    return x_pt,y_pt

#########################################
#  PROJECTION ALONG CHOSEN COORDINATE  #
#########################################

# Choose unit versor of direction
u = np.array([1, 1])
u_norm = (1 / np.linalg.norm(u, 2)) * u
w = np.empty_like(trj["x"][:, 0])
s = np.empty_like(trj["x"][:, 0])
proj_data = fl.Trajectories(dt=1e-3)
fig, axs = plt.subplots()
for n, trj in enumerate(data):
    for i in range(len(trj["x"])):
        w[i], s[i]= project(trj["x"][i],np.pi/4)
    proj_data.append(fl.Trajectory(1e-3, deepcopy(w.reshape(len(trj["x"][:, 0]), 1))))
    axs.plot(proj_data[n]["x"])
    axs.set_xlabel("$timesteps$")
    axs.set_ylabel("$w(t)$")
    axs.set_title("trajectory projected along $u =$" + str(u) + " direction")
    axs.grid()

#############################################################
# CREATE REFERENCE FOR FREE ENERGY USING IMPORTANCE SAMPLING #
#############################################################
def Pot(x, y):
    a = 2.5
    b = 5
    return a * (x**2 - 1)**2 + b * y**2

L = 3
x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
X, Y = np.meshgrid(x, y)

def q(x, y):
    # theta = 0
    # theta = np.pi / 4
    theta = np.pi / 4
    return np.cos(theta) * x + np.sin(theta) * y

# Importance sampling parameters
n_samples = 100000  # Number of samples
beta = 1         # Inverse temperature (1/k_B T)
x_min, x_max = -L, L
y_min, y_max = -L, L

# Generate uniform samples
x_samples = np.random.uniform(x_min, x_max, n_samples)
y_samples = np.random.uniform(y_min, y_max, n_samples)

# Compute Boltzmann weights
weights = np.exp(-beta * Pot(x_samples, y_samples))

# Compute the collective variable values
q_values = q(x_samples, y_samples)


# Weighted histogram the q values to estimate P(q)
q_bins = np.linspace(-3, 3, 201)
hist, bin_edges = np.histogram(q_values, bins=q_bins, weights=weights, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

print('n of bin centers', len(bin_centers))
# Compute the free energy A(q) = -k_B T ln P(q)
P_q = hist + 1e-10  # Avoid log(0)
A_q = -1 / beta * np.log(P_q)

# Plot the free energy profile
fig, ip = plt.subplots()
ip.plot(bin_centers, A_q - np.min(A_q))  # Shift so that the minimum A(q) is zero
ip.set_xlabel('q')
ip.set_ylabel('Free Energy A(q) ')
ip.set_title('Free Energy Profile with beta = '+str(beta))


############################################
             #  TRAINING  #
############################################

domain = fl.MeshedDomain.create_from_range(np.linspace(proj_data.stats.min, proj_data.stats.max, 4).ravel())
trainmodel = fl.models.Overdamped(force = fl.functions.BSplinesFunction(domain),has_bias=None)
xfa = np.linspace(proj_data.stats.min, proj_data.stats.max, 75)
xfa =np.linspace(-1.6,1.6,75)



fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()

fig,axb = plt.subplots()
axb.set_title("Free Energy (MLE)")
axb.set_xlabel("$X$")
axb.set_ylabel("$A_{MLE}(X)$")
axb.grid()

KM_Estimator = fl.KramersMoyalEstimator(deepcopy(trainmodel))
res_KM = KM_Estimator.fit_fetch(proj_data)

axs[0].plot(xfa, res_KM.force(xfa.reshape(-1, 1)),  marker="x",label="KramersMoyal")
axs[1].plot(xfa, res_KM.diffusion(xfa.reshape(-1, 1)), marker="x",label="KramersMoyal")
print("KramersMoyal ", res_KM.coefficients)
for name,marker,color, transitioncls in zip(
    ["Euler", "Elerian", "Kessler", "Drozdov"],
        ["|","1","2","3"],
        ["#1f77b4ff","#9e4de6ff","#2ca02cff","#ff7f0eff"],
    [
        fl.EulerDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    estimator = fl.LikelihoodEstimator(transitioncls(deepcopy(trainmodel)),n_jobs=4)
    res = estimator.fit_fetch(deepcopy(proj_data))
    res.remove_bias()
    print(name, res.coefficients)
    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)),marker=marker, label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)),marker=marker, label=name)
    fes = fl.analysis.free_energy_profile_1d(res,xfa)
    axb.plot(xfa, fes-fes[37],marker,color=color, label=name)
axb.plot(bin_centers, A_q - A_q[100],color ="#bd041cff",label ="MC sampling")  # Shift so that the minimum A(q) is zero

# axb.plot(q,A-A[37],color="#bd041cff",label='Numerically integrated')

axs[0].legend()
axs[1].legend()
axb.legend()

plt.show()
