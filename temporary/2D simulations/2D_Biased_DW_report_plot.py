"""
================================
2D Biased Double Well
================================

Estimation of an overdamped Langevin in presence of biased dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from scipy import integrate
import scipy as sc
import time

checkpoint1 = time.time()
x = np.linspace(-1.8, 1.8, 36)
y = np.linspace(-1.8, 1.8, 36)
input = np.transpose(np.array([x, y]))
D= 0.5
diff_function = fl.functions.Polynomial(deg=0, coefficients=D * np.eye(2, 2))
a, b = D*5.0, D*10.0
quartic2d = fl.functions.Quartic2D(a=a, b=b)
X, Y = np.meshgrid(x, y)

# Plot potential surface
pot = quartic2d.potential_plot(X, Y)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, pot, rstride=1, cstride=1, cmap="jet", edgecolor="none")

# Plot Force function
ff = quartic2d.force(input)  # returns x and y components of the force : x_comp =ff[:,0] , y_comp =ff[:,1]
U, V = np.meshgrid(ff[:, 0], ff[:, 1])
fig, ax = plt.subplots()
ax.quiver(x, y, U, V)
ax.set_title("Force")


##Definition of the Collective variable function of old coordinates
def colvar(x, y):
    gradient = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    q= (x + y)/np.sqrt(2)
    return q, gradient  # need to return both colvar function q=q(x,y) and gradient (dq/dx,dq/dy)


dt = 5e-4
model_simu = fl.models.overdamped.Overdamped(force=quartic2d, diffusion=diff_function)
simulator = fl.simulations.ABMD_2D_to_1DColvar_Simulator(fl.simulations.EulerStepper(model_simu), dt, colvar=colvar, k=25.0, qstop=1.7)

# Choose number of trajectories and initialize positions
ntraj = 50
q0 = np.empty(shape=[ntraj, 2])
for i in range(ntraj):
    for j in range(2):
        q0[i][j] = -1.2

####################################
##       CALCULATE TRAJECTORY     ##
####################################

time_steps = 5000
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
    axs.set_xlim(-1.8, 1.8)
    axs.set_ylim(-1.8, 1.8)
    axs.grid()

# plot x,y Trajectories in separate subplots
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

#########################################
#  PROJECTION ALONG CHOSEN COORDINATE   #
#########################################

# Choose unit versor of direction
u = np.array([1, 1])
u_norm = (1 / np.linalg.norm(u, 2)) * u
w = np.empty_like(trj["x"][:, 0])
b = np.empty_like(trj["x"][:, 0])
proj_data = fl.data.trajectories.Trajectories(dt=dt)  # create new Trajectory object in which to store the projected trajectory dictionaries
fig, axs = plt.subplots()
for n, trj in enumerate(data):
    for i in range(len(trj["x"])):
        w[i] = np.dot(trj["x"][i], u_norm)
        b[i] = np.dot(trj["bias"][i], u_norm)
    proj_data.append(fl.Trajectory(dt, deepcopy(w.reshape(len(trj["x"][:, 0]), 1)), bias=deepcopy(b.reshape(len(trj["bias"][:, 0]), 1))))
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

L = 2.0
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
trainmodel = fl.models.Overdamped(force = fl.functions.BSplinesFunction(domain),has_bias=True)
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
