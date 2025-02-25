## MODULES ##

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## SETTINGS ##

gamma = 0.1
omega = 1.0

u0 = 1.0
v0 = 0.0

t_start,t_end = 0,100
dt = 0.01
n_steps = int((t_end-t_start)/dt)
time_points = np.linspace(t_start,t_end,n_steps)

u = np.zeros(n_steps)
v = np.zeros(n_steps)

u[0] = u0
v[0] = v0

for t in range(1,n_steps):
    u[t] = u[t-1] + dt*v[t-1]
    v[t] = v[t-1] + dt*(-(gamma*v[t-1] + (omega**2)*u[t-1]))

fig,ax = plt.subplots()
for (label,data) in [('position',u),('velocity',v)]:
    sns.lineplot(ax=ax,x=time_points,y=data,label=label)

plt.show()