"""
================================
1D Double Well to generate plots to put in the report
================================

Estimation of an overdamped Langevin.
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from copy import deepcopy

free_energy_coeff = 0.2 * np.array([0, 0, -4.5, 0, 0.1])
free_energy = np.polynomial.Polynomial(free_energy_coeff)
D = np.array([0.5])

force_coeff = D*np.array([-free_energy_coeff[1], -2 * free_energy_coeff[2], -3 * free_energy_coeff[3], -4 * free_energy_coeff[4]])

force_function = fl.functions.Polynomial(deg=3, coefficients=force_coeff)
diff_function = fl.functions.Polynomial(deg=0, coefficients=D)

# Plot of Free Energy and Force
x_values = np.linspace(-7, 7, 100)
fig, axs = plt.subplots(1, 2)
axs[0].plot(x_values, free_energy(x_values))
axs[1].plot(x_values, force_function(x_values.reshape(len(x_values), 1)))
axs[0].set_title("Free Energy A(x)")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$A(x)$")
axs[0].grid()
axs[1].set_title("SDE Force $F(x) = -D \; \dfrac{\mathrm{d}A}{\mathrm{d}x} $")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$ F(x) $ ")
axs[1].grid()

# Define model to simulate and type of simulator to use
dt = 1e-3
model_simu = fl.models.overdamped.Overdamped(force_function, diffusion=diff_function)
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt)


# initialize positions
ntraj = 50
q0 = np.empty(ntraj)
for i in range(len(q0)):
    q0[i] = 0
# Calculate Trajectory
time_steps = 20000
data = simulator.run(time_steps, q0, save_every=1)

# Plot resulting Trajectories
fig, axs = plt.subplots()
for n, trj in enumerate(data):
    axs.plot(trj["x"],linewidth = 1)
    axs.set_title("Trajectories")
    axs.set_xlabel("$timestep$")
    axs.set_ylabel("$x(t)$")
    axs.grid()


fig, axs = plt.subplots(1, 2)
fig, axb = plt.subplots()
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()

axs[1].set_title("Diffusion Coefficient")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("D")
axs[1].grid()

xfa = np.linspace(-7.0, 7.0, 31)
axs[0].plot(xfa, model_simu.force(xfa.reshape(-1, 1)), label="Exact")
axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")

axb.set_title("Free Energy")
axb.set_xlabel("$x$")
axb.set_ylabel("$A(x)$")
axb.grid()
axb.plot(xfa, free_energy(xfa.reshape(-1, 1)), label="Exact")

n_knots= 5
domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min , data.stats.max , n_knots).ravel())
trainmodel = fl.models.Overdamped(force=fl.functions.BSplinesFunction(domain), has_bias=None)
for name,marker, transitioncls in zip(
    ["Euler", "Elerian", "Kessler", "Drozdov"],
    ["x", "1","2","3","|"],
    [
        fl.EulerDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    estimator = fl.LikelihoodEstimator(transitioncls(deepcopy(trainmodel)),n_jobs=4)
    res = estimator.fit_fetch(data)
    print(res.coefficients)
    fes = fl.analysis.free_energy_profile_1d(res,xfa)
    axs[0].plot(xfa,res.force(xfa.reshape(-1,1)),marker,label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)),marker, label=name)
    axb.plot(xfa, fes-fes[16],marker, label=name)
axb.legend()  
axs[0].legend()
axs[1].legend()
plt.show()