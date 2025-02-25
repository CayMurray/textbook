import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,10,1000)
y = np.cos(t)

fig,ax = plt.subplots()
ax.plot(t,y)

fig.savefig('/workspaces/textbook/figs/test.png',dpi=300,transparent=True)