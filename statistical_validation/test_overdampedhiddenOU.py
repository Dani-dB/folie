import numpy as np
import folie as fl
import matplotlib.pyplot as plt


model_simu = model_simu = fl.models.OrnsteinUhlenbeck(dim=3)
data_dts = []
list_dts = [1e-3, 5e-3, 1e-2, 5e-2]
for dt in list_dts:
    simulator = fl.Simulator(fl.ExactDensity(model_simu), dt, keep_dim=1)
    data_dts.append(simulator.run(5000, np.random.normal(loc=0.0, scale=1.0, size=(25, 3)), 25))
    # Only append subset of the trajectory
fig, axs = plt.subplots(1, len(model_simu.coefficients))

for name, transitioncls in zip(
    ["Euler"],
    [
        fl.EulerDensity,
    ],
):
    fun_lin = fl.functions.Linear().fit(data_dts[0])
    fun_frct = fl.functions.Constant().fit(data_dts[0])
    fun_cst = fl.functions.Constant().fit(data_dts[0])
    model = fl.models.OverdampedHidden(fun_lin, fun_frct, fun_cst, dim=1, dim_h=2)
    estimator = fl.EMEstimator(transitioncls(model), max_iter=15, verbose=2, verbose_interval=1)
    coeffs_vals = np.empty((len(data_dts), len(model.coefficients)))
    for n, data in enumerate(data_dts):
        res = estimator.fit_fetch(
            data[
                :,
            ]
        )
        coeffs_vals[n, :] = res.coefficients
    for n in range(len(axs)):
        axs[n].plot(list_dts, np.abs(coeffs_vals[:, n] - model_simu.coefficients[n]), "-+", label=name)
for n in range(len(axs)):
    axs[n].legend()
    axs[n].set_yscale("log")
    axs[n].grid()
    axs[n].set_xlabel("$\\Delta t")
    axs[n].set_ylabel("$|c-c_{real}|$")
plt.show()
