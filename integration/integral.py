
import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from scipy import integrate

x =np.linspace(-100,100,1001)
a,b= 2.5,5
q= np.linspace(-2,2,101)
I=[]
Z=[]
A=[]
for i in range(len(q)):
    y = np.exp(-(a*(x**2-1)**2 - 0.5*b*x**2 -b*q[i]*x))
    I.append(integrate.simpson(y, x=x))
    Z.append(np.exp(-0.5*b*q[i]**2)*I[i])
    A.append(-np.log(Z[i]))


fig, ax = plt.subplots()
# ax.plot(x,y,label='Function to integrate')
ax.plot(q,A,label='integrated')
ax.legend()
plt.show()
