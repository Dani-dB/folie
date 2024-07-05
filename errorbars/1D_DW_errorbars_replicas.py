"""
===================================================
Errorbar Estimation for 1D Double Well Potential 
===================================================

Computes the free energy of a 1D Double Well n_replicas times with as many datasets,thus retaining the average free energy.
Errorbars are computed as the sample standard deviation from the mean.
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from copy import deepcopy
import utilsDWbias as utl
import scipy as sc

#define system to simulate 
coeff = 0.2 * np.array([0, 0, -4.5, 0, 0.1])
free_energy = np.polynomial.Polynomial(coeff)
D = np.array([0.5])

drift_coeff = D * np.array([-coeff[1], -2 * coeff[2], -3 * coeff[3], -4 * coeff[4]])
drift_function = fl.functions.Polynomial(deg=3, coefficients=drift_coeff)
diff_function = fl.functions.Polynomial(deg=0, coefficients=D)

# initialize positions
ntraj = 50
time_steps= 35000
q0 = np.empty(ntraj)
for i in range(len(q0)):
    q0[i] = -6

# derfine number of replicas used to estimate the average and store the trained object and free energy arrays in lists
n_replicas = 5
Eulres, Elnres, Kslres, Drzres = [], [], [],[]
Eulfes, Elnfes,Kslfes,Drzfes = [],[],[], []

xfa = np.linspace(-7,7,100)

for i in range(n_replicas): # Create different datasets (replicas), for each one of them run an estimation with all the chosen estimators
    
    # Definition of model to simulate and simulator object 
    dt=5e-4
    model_simu = fl.models.overdamped.Overdamped(drift_function, diffusion=diff_function)
    simulator = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(model_simu), dt, k=10.0, xstop=6.0)

    # Generate the dataset, possibly plotting the trajectory
    data, discard= utl.Generate_Plot_Trajectories_Data(simulator,q0,time_steps,savevery=2,plot=True) 
    
    #Create model to train upon
    domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min, data.stats.max, 4).ravel())
    trainmodel = fl.models.Overdamped(fl.functions.BSplinesFunction(domain), has_bias=True)
    
    # Fit the current dataset retrieving the trained object and associated inferred free energy for Euler, Elerian, Kessler and Drozdov estimators
    Eulres.append(utl.train_single_estimator(fl.EulerDensity,data,trainmodel))
    Eulfes.append(fl.analysis.free_energy_profile_1d(Eulres[i],xfa))

    Elnres.append(utl.train_single_estimator(fl.ElerianDensity,data,trainmodel))
    Elnfes.append(fl.analysis.free_energy_profile_1d(Elnres[i],xfa))

    Kslres.append(utl.train_single_estimator(fl.KesslerDensity,data,trainmodel))
    Kslfes.append(fl.analysis.free_energy_profile_1d(Kslres[i],xfa))

    Drzres.append(utl.train_single_estimator(fl.DrozdovDensity,data,trainmodel))
    Drzfes.append(fl.analysis.free_energy_profile_1d(Drzres[i],xfa))


###########################################################################################################################
##                                                  COMPUTE ERRORBARS
###########################################################################################################################

# calculate the mean free energy surfaces 
Eul_mean_fes = utl.mean_Fes(Eulfes,xfa)
Eln_mean_fes = utl.mean_Fes(Elnfes,xfa)
Ksl_mean_fes = utl.mean_Fes(Kslfes,xfa)
Drz_mean_fes = utl.mean_Fes(Drzfes,xfa)

# calculate the mean fre energy surfaces 
Eul_variance_fes = utl.variance_Fes(Eulfes,xfa,estimator_mean_fes=Eul_mean_fes)
Eln_variance_fes = utl.variance_Fes(Elnfes,xfa)
Ksl_variance_fes = utl.variance_Fes(Kslfes,xfa)
Drz_variance_fes = utl.variance_Fes(Drzfes,xfa,Drz_mean_fes)

# calculate standard deviation and emplot it or errorbars 
Eul_std_fes = np.sqrt(Eul_variance_fes)
Eln_std_fes = np.sqrt(Eln_variance_fes)
Ksl_std_fes = np.sqrt(Ksl_variance_fes)
Drz_std_fes = np.sqrt(Drz_variance_fes)

fig, ax =plt.subplots(2,2,figsize=(10,7))
fig.suptitle("MLE for $\\langle A (q)\\rangle$ including errorbars")
ax.set_xlabel('q')
ax.set_ylabel('$\\langle A \\rangle$')

# Plot the average free energies for each estimators with the associated errorbars 
for index in range(n_replicas):
    ax[0][0].plot(xfa,Eulfes[index],label='dataset '+str(index + 1))
ax[0][0].errorbar(xfa,Eul_mean_fes,yerr=Eul_std_fes, errorevery=(0,5),fmt='o',color ='red', ecolor='C5',alpha=1, label="Mean free energy")
ax[0][0].set_title('Euler')
ax[0][0].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[0][0].legend()

for index in range(n_replicas):
    ax[0][1].plot(xfa,Elnfes[index],label='dataset '+str(index + 1))
ax[0][1].errorbar(xfa,Eln_mean_fes,yerr=Eln_std_fes, errorevery=(1,5),fmt='o',color ='red', ecolor='C5',alpha=1, label="Mean free energy")
ax[0][1].set_title('Elerian')
ax[0][1].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[0][1].legend()

for index in range(n_replicas):
    ax[1][0].plot(xfa,Kslfes[index],label='dataset '+str(index + 1))
ax[1][0].errorbar(xfa,Ksl_mean_fes,yerr=Ksl_std_fes, errorevery=(2,5),fmt='o',color ='red', ecolor='C5',alpha=1, label="Mean free energy")
ax[1][0].set_title('Kessler')
ax[1][0].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[1][0].legend()

for index in range(n_replicas):
    ax[1][1].plot(xfa,Drzfes[index],label='dataset '+str(index + 1))
ax[1][1].errorbar(xfa,Drz_mean_fes,yerr=Drz_std_fes, errorevery=(3,5),fmt='o',color ='red', ecolor='C5',alpha=1, label="Mean free energy")
ax[1][1].set_title('Drozdov')
ax[1][1].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[1][1].legend()

fig, P = plt.subplots()
P.errorbar(xfa[0::4],Eul_mean_fes[0::4],yerr=Eul_std_fes[0::4],fmt='1',color ='C1', ecolor='C1',alpha=1, label="Euler Fes")
P.errorbar(xfa[1::4],Eln_mean_fes[1::4],yerr=Eln_std_fes[1::4],fmt='2',color ='C2', ecolor='C2',alpha=1, label="Elerian Fes")
P.errorbar(xfa[2::4],Ksl_mean_fes[2::4],yerr=Ksl_std_fes[2::4],fmt='3',color ='C3', ecolor='C3',alpha=1, label="Kessler Fes")
P.errorbar(xfa[3::4],Drz_mean_fes[3::4],yerr=Drz_std_fes[3::4],fmt='4',color ='C4', ecolor='C4',alpha=1, label="Drodov Fes")
P.plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))),color='C0' ,label="Exact")

P.legend()
P.set_title("MLE for $\\langle A (q)\\rangle$ including errorbars")
P.set_xlabel('q')
P.set_ylabel('$\\langle A \\rangle$')
plt.show()