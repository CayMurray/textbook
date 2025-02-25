## MODULES ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## SETTINGS ##

w,d = 2*np.pi, 0.25
start,stop = 0,10
dt = 0.01
x0,v0 = 1,1


## SIM ##

num_t = int(stop/dt)
t = np.linspace(start,stop,num_t)

A = np.array([[0,1],[-w**2,-2*d*w]])
X = np.zeros((2,len(t)))

X[:,0] = (x0,v0)

for t_point in range(num_t-1):
    X[:,t_point+1] = X[:,t_point] + A@X[:,t_point]*dt    


## VISUALIZE ##

fig,ax = plt.subplots()
sns.lineplot(ax=ax,x=t,y=X[0,:])
plt.show()