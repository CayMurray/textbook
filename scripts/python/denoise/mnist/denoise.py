## MODULES ## 

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from models import MLP

import matplotlib.pyplot as plt
import seaborn as sns


## FUNCS ##

def model_performance(*args):
    fig,ax = plt.subplots()
    for (label,data) in args:
        sns.lineplot(ax=ax,x=[i for i in range(len(data))],y=data,label='label')
    ax.set_xticks(np.arange(0,len(data),1))
    plt.show()
    
def visualize(model,test_iter,corrupt):
    recon = model(corrupt)

    fig,axes = plt.subplots(nrows=len(test_iter),ncols=3,layout='compressed')
    for i,(n,t,r) in enumerate(zip(noise_iter,test_iter,recon)):
        axes[i,0].imshow(n.detach().cpu().squeeze())
        axes[i,1].imshow(t.detach().cpu().squeeze())
        axes[i,2].imshow(r.detach().cpu().reshape(28,28))

    plt.setp(axes,xticks=[],yticks=[])
    plt.show()


def train_loop(model,device='cpu',epochs=5):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    criteron = nn.MSELoss()

    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        print(f'Epoch {epoch}...')
        epoch_loss = 0

        for batch,_ in train_loader:
            optimizer.zero_grad()
            input = batch.to(device)
            recon = model(input)
            loss = criteron(input.view(-1,784),recon)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()*batch.size(0)
        train_loss.append(epoch_loss/len(train_loader.sampler))

        t_loss = 0
        with torch.no_grad():
            for batch,_  in test_loader:
                input = batch.to(device)
                recon = model(input)
                loss = criteron(input.view(-1,784),recon)
                t_loss += loss.item()*batch.size(0)
            test_loss.append(t_loss/len(test_loader.sampler))

    #model_performance(('train',train_loss),('test',test_loss))
    return model


## LOAD DATA ##

batch_size = 32
transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

train_data = datasets.MNIST(root='data',transform=transformations,train=True,download=False)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_data = datasets.MNIST(root='data',transform=transformations,train=False,download=False)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)


## TRAIN LOOP ##

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)
model = train_loop(model, device=device, epochs=10)


## INFERENCE ##

test_iter = next(iter(test_loader))[0][:5].to(device)
noise = torch.normal(0,1,test_iter.shape).to(device)
noise_iter = test_iter + noise
blur = transforms.GaussianBlur(5,sigma=1.5)
blur_iter = blur(test_iter)

visualize(model,test_iter,noise_iter)

