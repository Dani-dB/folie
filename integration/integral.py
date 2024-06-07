
import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from scipy import integrate

x =np.linspace(-10,10,101)
a,b= 5,10
q= np.linspace(-1.2,1.2,101)
I=[]
Z=[]
A=[]
for i in range(len(q)):
    y = np.exp(-(a*(x**2-1)**2 - 0.5*b*x**2 -np.sqrt(2)*b*q[i]*x))
    I.append(integrate.simpson(y, x=x))
    Z.append(np.exp(-0.5*b*q[i]**2)*I[i])
    A.append(-np.log(Z[i]))


fig, ax = plt.subplots()
# ax.plot(x,y,label='Function to integrate')
ax.plot(q,A,label='integrated')
ax.legend()
plt.show()
