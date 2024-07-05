"""
================================
1D Double Well
================================

Definition of functions to generate data and a train models, meant to work more efficiently on a script who does this several times   
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from copy import deepcopy

def Generate_Plot_Trajectories_Data(simulator, q0, time_steps,savevery=1,plot=True):
    """
    Performs a simulation of trajectories when given the properly defined simulator object
    if flag plot = True it also returns the trajectory plot 
    Parameters
    ------------
        simulator: simulator class instance
            simulator whose parameter are already defined 

        q0: array
            array of initial conditions

        time_steps: int
            number of timesteps 
        saveevery: int
            save_every paramter of simulator.run method 
        plot: boolean
            if True it also returns the trajectory plot, otherwise it returns a `None` object

    """

    # Calculate Trajectory
    data = simulator.run(time_steps, q0, save_every=savevery)
    # Plot Trajectories
    if plot:
        fig, axs = plt.subplots()
        abscissa = np.linspace(0,time_steps,len(data[0]["x"]))
        for n, trj in enumerate(data):
            axs.plot(abscissa,trj["x"])
            axs.set_xlabel("$timestep$")
            axs.set_ylabel("$x(t)$")
            axs.grid()
        return  data ,axs
    else:
        return data, None
    

def Train_all_loop(model_simu,data,trainmodel):
    """
    Run the same analysis as foli/examples/toy_models/plot_1D_Double_Well.py
    It returns only the plots

    Parameters
------------

    model_simu: Model class instance
        object who genearetd the data, used onl to plot the refernce values of drift and diffusion

    data: folie.Trajectories() class instance 
        The folie.Trajectories object storing the trajectory to analyze 

    trainmodel: Model class instance
        Object describing the model whose parameter are to train    

    """
    fig, axs = plt.subplots(1, 2)
    fig, axf = plt.subplots()
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
        fes = fl.analysis.free_energy_profile_1d(res,xfa)
        axf.plot(xfa,fes,marker=marker,label =name)
    axf.legend()
    axs[0].legend()
    axs[1].legend()
    return axs , axf

def train_single_estimator(density,data,trainmodel):
    """
    Performs a training of the input `trainmodel` with the specified transition density

    Parameters
    ------------

        density: TransitionDensity class instance
            The transition density to use during the estimation 

        data: folie.Trajectories() class instance 
            The folie.Trajectories object storing the trajectory to analyze 

        trainmodel: Model class instance
            Object describing the model whose parameter are to train    

    """
    estimator = fl.LikelihoodEstimator(density(deepcopy(trainmodel)), n_jobs=4)
    res = estimator.fit_fetch(deepcopy(data))
    res.remove_bias()
    return deepcopy(res)


def mean_Fes(estimator_fes,x):
    """
    Return an array containing the mean free energy given a list of arrays of different free energy calculated over different replicas of the system 

    Parameters
    ------------
        estimator_fes: list
            List with different free energy arrays, one per each replica of the system. Each entry has to be of same lenght as `x`

        x: array
            array of the abscissa value, must be of the same length as any entry of `estimator_fes`
    """
    meanfes = np.empty_like(x)
    for i in range(len(x)): #sum over xfa points
        sum=0
        for replica_index in range(len(estimator_fes)):
            sum += estimator_fes[replica_index][i] # sum over the replicas for the given estimator for the i-th point
        meanfes[i]= sum/len(estimator_fes)
    return meanfes

def variance_Fes(estimator_fes,x, estimator_mean_fes=None):
    """
    Return an array containing the sample variance of free energy given a list of arrays of different free energy calculated over different replicas of the system 

    Parameters
    ------------
        estimator_fes: list
            List with different free energy arrays, one per each replica of the system. Each entry has to be of same lenght as `x`

        x: array
            array of the abscissa value, must be of the same length as any entry of `estimator_fes`

        estimator_mean_fes: array
        mean free energy around which to compute the sample variance, default value is None
        must be of same length as `x`
    """
    variancefes = np.empty_like(x)
    if estimator_mean_fes is None: 
        estimator_mean_fes= mean_Fes(estimator_fes,x)
    for i in range(len(x)): #sum over x points
        sum=0
        for replica_index in range(len(estimator_fes)):
            sum += (estimator_fes[replica_index][i]- estimator_mean_fes[i])**2 # sum over the replicas for the given estimator for the i-th point
        variancefes[i]= sum/(len(estimator_fes)-1)
    return variancefes


# err =np.empty_like(var_fes)
# for i in range(len(xfa)):
#      sum = ((fes1[0][i]-mean_fes[i])**2+(fes2[0][i]-mean_fes[i])**2 + (fes3[0][i]-mean_fes[i])**2 + (fes4[0][i]-mean_fes[i])**2 ) # sum over the replicas for the jth estimator 
#      var_fes[i]= sum/3