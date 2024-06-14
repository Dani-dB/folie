import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

x = np.linspace(-1.8,1.8,36)
y = np.linspace(-1.8,1.8,36)
input=np.transpose(np.array([x,y]))

D=0.5
diff_function= fl.functions.Polynomial(deg=0,coefficients=D * np.eye(2,2))
a,b = 5, 10
drift_quartic2d= fl.functions.Quartic2D(a=D*a,b=D*b)  # simple way to multiply D*Potential here force is the SDE force (meandispl)  ## use this when you need the drift ###
quartic2d= fl.functions.Quartic2D(a=a,b=b)            # Real potential , here force is just -grad pot ## use this when you need the potential energy ###
X,Y =np.meshgrid(x,y)

# Plot potential surface 
pot = quartic2d.potential_plot(X,Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,pot, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')


dt = 5e-4
model_simu=fl.models.overdamped.Overdamped(force=drift_quartic2d,diffusion=diff_function)
simulator=fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt)

# initialize positions 
ntraj=50
q0= np.empty(shape=[ntraj,2])
for i in range(ntraj):
    for j in range(2):
        q0[i][j]=0.0000
time_steps=20000
data_2d_unbias = simulator.run(time_steps, q0,save_every=1) 

# Projection on x 
xdata = fl.data.trajectories.Trajectories(dt=dt) 
for n, trj in enumerate(data_2d_unbias):
    xdata.append(fl.data.trajectories.Trajectory(dt, trj["x"][:,0].reshape(len(trj["x"][:,0]),1)))
xfa = np.linspace(-1.3, 1.3, 75)
xforce = -4*a*(xfa** 3 - xfa)

# Traning on x 

n_knots=4
xfa = np.linspace(-1.3, 1.3, 75)
domain = fl.MeshedDomain.create_from_range(np.linspace(xdata.stats.min,xdata.stats.max , n_knots).ravel())
splines_trainmodelx = fl.models.Overdamped(force = fl.functions.BSplinesFunction(domain), diffusion= fl.functions.BSplinesFunction(domain), has_bias = None)

KM_estimator = fl.KramersMoyalEstimator(deepcopy(splines_trainmodelx), n_jobs=4)
Eul_estimator = fl.LikelihoodEstimator(fl.EulerDensity(deepcopy(splines_trainmodelx)),n_jobs=4)
Eln_estimator = fl.LikelihoodEstimator(fl.ElerianDensity(deepcopy(splines_trainmodelx)),n_jobs=4)
Ksl_estimator = fl.LikelihoodEstimator(fl.KesslerDensity(deepcopy(splines_trainmodelx)),n_jobs=4)
Drz_estimator = fl.LikelihoodEstimator(fl.DrozdovDensity(deepcopy(splines_trainmodelx)),n_jobs=4)

KM_res=KM_estimator.fit_fetch(xdata)
Eul_res=Eul_estimator.fit_fetch(xdata)
Eln_res=Eln_estimator.fit_fetch(xdata)
Ksl_res=Ksl_estimator.fit_fetch(xdata)
Drz_res=Drz_estimator.fit_fetch(xdata)

res_vec = [KM_res,Eul_res,Eln_res,Ksl_res,Drz_res] # made a list of all the trained estimators 

# PLOT OF THE RESULTS 

fig, axs = plt.subplots(1, 2, figsize=(14, 8))
fig, axb = plt.subplots(figsize=(14, 8))

axs[0].set_title("Force (MLE) ")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[1].set_title("Diffusion D (MLE)")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$") 
axs[1].grid()

#Plot exact quantities 
xy=np.transpose(np.array([xfa,np.zeros(len(xfa))]))

axb.plot(xfa,  quartic2d.potential(xy)-quartic2d.potential(xy)[37], label='Exact')
axs[0].plot(xfa,  drift_quartic2d.force(xy)[:,0], label="Exact")

# #Plot inferred quantities 
names = ["KM","Euler", "Elerian", "Kessler", "Drozdov"]
markers = ["x", "1","2","3","|"]
for i in range(len(names)):

    print(names[i],res_vec[i].coefficients)
    fes = fl.analysis.free_energy_profile_1d(res_vec[i],xfa)
    axs[0].plot(xfa,res_vec[i].force(xfa.reshape(-1,1)),markers[i],label=names[i])
    axs[1].plot(xfa, res_vec[i].diffusion(xfa.reshape(-1, 1)),markers[i], label=names[i])

    axb.plot(xfa, fes-fes[37],markers[i], label=names[i])

axs[0].legend()
axs[1].legend()
axb.legend()
fig.suptitle('Proj on x axis')

plt.show()