import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from copy import deepcopy
import utilsDWbias as utl

coeff = 0.2 * np.array([0, 0, -4.5, 0, 0.1])
free_energy = np.polynomial.Polynomial(coeff)
D = np.array([0.5])

drift_coeff = D * np.array([-coeff[1], -2 * coeff[2], -3 * coeff[3], -4 * coeff[4]])
drift_function = fl.functions.Polynomial(deg=3, coefficients=drift_coeff)
diff_function = fl.functions.Polynomial(deg=0, coefficients=D)

# Define model to simulate and type of simulator to use
model_simu = fl.models.overdamped.Overdamped(drift_function, diffusion=diff_function)
simulator = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(model_simu), 1e-3, k=10.0, xstop=6.0)

# initialize positions
ntraj = 30
time_steps= 25000
q0 = np.empty(ntraj)
for i in range(len(q0)):
    q0[i] = -6

n_replicas = 4
Eulres, Elnres, Kslres, Drzres = [], [], [],[]
Eulfes, Elnfes,Kslfes,Drzfes = [],[],[], []

xfa = np.linspace(-7,7,50)

# data, discard= utl.Generate_Plot_Trajectories_Data(deepcopy(simulator),q0,time_steps,plot=False) 
    
# domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min, data.stats.max, 4).ravel())
# trainmodel = fl.models.Overdamped(fl.functions.BSplinesFunction(domain), has_bias=True)
# im1, im2 = utl.Train_all_loop(model_simu,data,trainmodel)
# im2.plot(xfa,free_energy(xfa))
plt.show()
for i in range(n_replicas):

    data, discard= utl.Generate_Plot_Trajectories_Data(deepcopy(simulator),q0,time_steps,plot=False) 
    
    domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min, data.stats.max, 4).ravel())
    trainmodel = fl.models.Overdamped(fl.functions.BSplinesFunction(domain), has_bias=True)

    Eulres.append(utl.train_single_estimator(fl.EulerDensity,data,trainmodel))
    Eulfes.append(fl.analysis.free_energy_profile_1d(Eulres[i],xfa))

    Elnres.append(utl.train_single_estimator(fl.ElerianDensity,data,trainmodel))
    Elnfes.append(fl.analysis.free_energy_profile_1d(Elnres[i],xfa))

    Kslres.append(utl.train_single_estimator(fl.KesslerDensity,data,trainmodel))
    Kslfes.append(fl.analysis.free_energy_profile_1d(Kslres[i],xfa))

    Drzres.append(utl.train_single_estimator(fl.DrozdovDensity,data,trainmodel))
    Drzfes.append(fl.analysis.free_energy_profile_1d(Drzres[i],xfa))

# eulres= utl.train_single_estimator(fl.EulerDensity,data,trainmodel)
# for i in range(n_replicas):
#     print(Eulres[i].coefficients)

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


# Eul_var_fes=np.empty_like(xfa)
# err =np.empty_like(var_fes)
# for i in range(len(xfa)):
#      sum = ((fes1[0][i]-mean_fes[i])**2+(fes2[0][i]-mean_fes[i])**2 + (fes3[0][i]-mean_fes[i])**2 + (fes4[0][i]-mean_fes[i])**2 ) # sum over the replicas for the jth estimator 
#      var_fes[i]= sum/3

# err = np.sqrt(var_fes)
fig, ax =plt.subplots(2,2,figsize=(10,7))
fig.suptitle("MLE for $\\langle A (q)\\rangle$ including errorbars")
# ax.set_xlabel('q')
# ax.set_ylabel('$\\langle A \\rangle$')


for index in range(n_replicas):
    ax[0][0].plot(xfa,Eulfes[index],label='dataset '+str(index + 1))
ax[0][0].errorbar(xfa,Eul_mean_fes,yerr=Eul_std_fes, errorevery=(0,5),fmt='o',color ='red', ecolor='C4',alpha=1, label="Mean free energy")
ax[0][0].set_title('Euler')
ax[0][0].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[0][0].legend()

for index in range(n_replicas):
    ax[0][1].plot(xfa,Elnfes[index],label='dataset '+str(index + 1))
ax[0][1].errorbar(xfa,Eln_mean_fes,yerr=Eln_std_fes, errorevery=(1,5),fmt='o',color ='red', ecolor='C4',alpha=1, label="Mean free energy")
ax[0][1].set_title('Elerian')
ax[0][1].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[0][1].legend()

for index in range(n_replicas):
    ax[1][0].plot(xfa,Kslfes[index],label='dataset '+str(index + 1))
ax[1][0].errorbar(xfa,Ksl_mean_fes,yerr=Ksl_std_fes, errorevery=(2,5),fmt='o',color ='red', ecolor='C4',alpha=1, label="Mean free energy")
ax[1][0].set_title('Kessler')
ax[1][0].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[1][0].legend()

for index in range(n_replicas):
    ax[1][1].plot(xfa,Drzfes[index],label='dataset '+str(index + 1))
ax[1][1].errorbar(xfa,Drz_mean_fes,yerr=Drz_std_fes, errorevery=(3,5),fmt='o',color ='red', ecolor='C4',alpha=1, label="Mean free energy")
ax[1][1].set_title('Drozdov')
ax[1][1].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
ax[1][1].legend()

# for estimator_index in range(4):
#     for index in range(n_replicas):
#         ax[estimator_index].plot(xfa,Drzfes[index],label='dataset '+str(index + 1))
#     ax[estimator_index].errorbar(xfa,Drz_mean_fes,yerr=Drz_std_fes, errorevery=(0,5),fmt='o',color ='red', ecolor='C4',alpha=1, label="Mean free energy")
#     ax[estimator_index].set_title(names[estimator_index])
#     ax[estimator_index].plot(xfa, free_energy(xfa.reshape(-1, 1))-np.min(free_energy(xfa.reshape(-1, 1))), label="Exact")
# ax[estimator_index].legend()


# fig, P = plt.subplots()
# P.errorbar(xfa,Eln_mean_fes,yerr=Eln_std_fes, errorevery=(1,5),fmt='2',color ='red', ecolor='C4',alpha=1, label="Mean free energy")



plt.show()