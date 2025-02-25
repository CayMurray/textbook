## MODULES ##

import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import torch
import torch.nn as nn

torch.manual_seed(42)

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from models import PINN

## FUNCS ##

def exp(k,t):
    return torch.exp(-k*t)

def to_numpy(x):
    return x.detach().cpu().numpy().squeeze()

def visualize(*args):

    fig,ax = plt.subplots()
    for (type,x,y,label,color) in args:
        if type == 'scatter':
            sns.scatterplot(ax=ax,x=to_numpy(x),y=to_numpy(y),label=label,color=color)
    
        elif type == 'line':
            sns.lineplot(ax=ax,x=to_numpy(x),y=to_numpy(y),label=label,color=color)
            
    plt.setp(ax,xlabel='',ylabel='',xticks=[],yticks=[])
    plt.show()

def train_loop(model,criterion,optimizer,data,epochs):

    for epoch in range(epochs):
        optimizer.zero_grad()
        t_input,y_input,t_p = data
        y_hat = model(t_input)
        loss_1 = criterion(y_hat,y_input)

        y_p = model(t_p)
        dy = torch.autograd.grad(y_p,t_p,torch.ones_like(y_p),create_graph=True)[0]
        physics = dy + k*y_p
        loss_2 = (1e-1)*torch.mean(physics**2)

        loss = loss_1 + loss_2 if use_PINN else loss_1
        loss.backward(retain_graph=True)
        optimizer.step()

    return model


## SETTINGS ##

k = 5
y0 = 1
start = 0
stop = 1
n_steps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10000
use_PINN = True


## SIM ##

t = torch.linspace(start,stop,n_steps,requires_grad=True).view(-1,1).to(device)
y = exp(k,t)

t_input, y_input = t[start:500:10], y[start:500:10]

y_noise = y_input + torch.normal(0,0.05,y_input.shape).to(device)
t_p = torch.linspace(start,stop,30,requires_grad=True).view(-1,1).to(device)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
criterion = nn.MSELoss()
model = train_loop(model,criterion,optimizer,(t_input,y_noise,t_p),epochs)
print(device)

## VISUALIZE ##

y_predict = model(t)
visualize(('line',t,y_predict,'Predicted','#DD8452'),
          ('line',t,y,'Actual','#4C72B0'),
          ('scatter',t_input,y_noise,'Train','#4C72B0'))