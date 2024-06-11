

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)

N = 100
x = np.random.randint(0, 9, N)
bins = np.arange(10)

kde = stats.gaussian_kde(x)
xx = np.linspace(0, 9, 1000)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(x, density=True, bins=bins, alpha=0.3)
ax.plot(xx, kde(xx))
plt.show()
a=np.array([1,2,3,4])
b=np.array([5,6,7,8,9])
c= np.concatenate((a,b))
d=np.array([0,627,48,69])
c= np.concatenate((c,d))
c= np.concatenate((c,d))
c= np.concatenate((c,a))
print(np.arange(0,9))