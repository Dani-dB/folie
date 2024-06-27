import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from copy import deepcopy

def Generate_Plot_Trajectories_Data(simulator, q0, time_steps,plot=True):

    # Calculate Trajectory
    data = simulator.run(time_steps, q0, save_every=1)
    # Plot Trajectories
    if plot:
        fig, axs = plt.subplots()
        for n, trj in enumerate(data):
            axs.plot(trj["x"])
            axs.set_xlabel("$timestep$")
            axs.set_ylabel("$x(t)$")
            axs.grid()
        return  data ,axs
    else:
        return data, None
    

def Train_all_loop(model_simu,data,trainmodel):
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Drift")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$F(x)$")
    axs[0].grid()

    axs[1].set_title("Diffusion Coefficient")  # i think should be diffusion coefficient
    axs[1].set_xlabel("$x$")
    axs[1].set_ylabel("$D(x)$")
    axs[1].grid()

    xfa = np.linspace(-7.0, 7.0, 75)
    model_simu.remove_bias()
    axs[0].plot(xfa, model_simu.drift(xfa.reshape(-1, 1)), label="Exact")
    axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")


    name = "KramersMoyal"
    estimator = fl.KramersMoyalEstimator(deepcopy(trainmodel))
    res = estimator.fit_fetch(data)
    res.remove_bias()
    axs[0].plot(xfa, res.drift(xfa.reshape(-1, 1)), "--", label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), "--", label=name)

    for name, marker, transitioncls in zip(
        ["Euler", "Elerian", "Kessler", "Drozdov"],
        ["x", "1", "2", "3"],
        [
            fl.EulerDensity,
            fl.ElerianDensity,
            fl.KesslerDensity,
            fl.DrozdovDensity,
        ],
    ):
        estimator = fl.LikelihoodEstimator(transitioncls(deepcopy(trainmodel)), n_jobs=4)
        res = estimator.fit_fetch(deepcopy(data))
        print(name, res.coefficients)
        res.remove_bias()
        axs[0].plot(xfa, res.drift(xfa.reshape(-1, 1)), marker=marker, label=name)
        axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), marker=marker, label=name)
    axs[0].legend()
    axs[1].legend()
    return axs

def train_single_estimator(density,data,trainmodel):
    estimator = fl.LikelihoodEstimator(density(deepcopy(trainmodel)), n_jobs=4)
    res = estimator.fit_fetch(deepcopy(data))
    res.remove_bias()

    return res


def mean_Fes(estimator_fes,x):
    meanfes = np.empty_like(x)
    for i in range(len(x)): #sum over xfa points
        sum=0
        for replica_index in range(len(estimator_fes)):
            sum += estimator_fes[replica_index][i] # sum over the replicas for the Euler estimator for the i-th point
        meanfes[i]= sum/len(estimator_fes)
    return meanfes

def variance_Fes(estimator_fes,x, estimator_mean_fes=None):
    variancefes = np.empty_like(x)
    if estimator_mean_fes is None: 
        estimator_mean_fes= mean_Fes(estimator_fes,x)
    for i in range(len(x)): #sum over xfa points
        sum=0
        for replica_index in range(len(estimator_fes)):
            sum += (estimator_fes[replica_index][i]- estimator_mean_fes[i])**2 # sum over the replicas for the Euler estimator for the i-th point
        variancefes[i]= sum/(len(estimator_fes)-1)
    return variance_Fes


# err =np.empty_like(var_fes)
# for i in range(len(xfa)):
#      sum = ((fes1[0][i]-mean_fes[i])**2+(fes2[0][i]-mean_fes[i])**2 + (fes3[0][i]-mean_fes[i])**2 + (fes4[0][i]-mean_fes[i])**2 ) # sum over the replicas for the jth estimator 
#      var_fes[i]= sum/3