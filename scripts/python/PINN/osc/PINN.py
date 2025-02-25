## MODULES ## 

import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

torch.manual_seed(42)

## CLASSES AND FUNCS ##


def to_numpy(x):
    return x.detach().cpu().numpy().squeeze()

def oscillator(d,w0,t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi + w*t)
    exp = torch.exp(-d*t)
    return exp*2*A*cos


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1,64),nn.Tanh(),
                                    nn.Linear(64,32),nn.Tanh(),
                                    nn.Linear(32,16),nn.Tanh(),
                                    nn.Linear(16,1))
        
    def forward(self,x):
        x = self.layers(x)
        return x


## SETTINGS ##

d,w0 = 0,20 
t_start,t_end = 0,1
t_step = 500
use_PINN = True

epochs = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## RUN OSCILLATOR 

t = torch.linspace(t_start,t_end,t_step).view(-1,1) 
y_analytical = oscillator(d,w0,t)
t_slice = t[0:200:20]
y_slice = y_analytical[0:200:20]
t_p = torch.linspace(t_start,t_end,30,requires_grad=True).view(-1,1).to(device)


## TRAIN LOOP ##

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
criterion = nn.MSELoss()
t_input,y = t_slice.to(device),y_slice.to(device)

mu,k = 2*d, w0**2
for epoch in range(epochs):
    print(f'Epoch {epoch}') if epoch%2500 == 0 else None 
    optimizer.zero_grad()
    y_h = model(t_input)
    loss_1 = criterion(y_h,y)

    y_p = model(t_p)
    dt = torch.autograd.grad(y_p,t_p,torch.ones_like(y_p),create_graph=True)[0]
    dt2 = torch.autograd.grad(dt,t_p,torch.ones_like(y_p),create_graph=True)[0]
    physics = dt2 + mu*dt + k*y_p
    loss_2 = (1e-4) * torch.mean(physics**2)

    loss = loss_1 + loss_2 if use_PINN else loss_1
    loss.backward()
    optimizer.step()

predicted = model(t.to(device))
 

## VISUALIZE ##

fig,ax = plt.subplots()
sns.scatterplot(ax=ax,x=t_slice.squeeze(),y=y_slice.squeeze(),label='slice',color='orange')
sns.lineplot(ax=ax,x=t.squeeze(),y=to_numpy(predicted), label='predicted')
sns.lineplot(ax=ax,x=t.squeeze(),y=y_analytical.squeeze(),label='analytical')
plt.show()