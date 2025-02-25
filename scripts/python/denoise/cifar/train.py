## MODULES ##

import os
import numpy as np

import torch.utils
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import torch 
from torchvision import transforms as T, datasets
from torch.utils.data import Dataset,DataLoader

from models import CNN

import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
    

## FUNCS ##

def add_noise(data,noise_factor,device):
    noisy_data = data + noise_factor*torch.normal(0,1,data.shape)
    noisy_data = torch.clamp(noisy_data,0.,1.)
    return data.to(device), noisy_data.to(device)

def visualize(*args,num_imgs=5):
    fig,ax = plt.subplots(nrows=num_imgs,ncols=len(args),layout='compressed')
    for i in range(num_imgs):
        for (d,img) in enumerate(args):
            ax[i,d].imshow(img[i].permute(1,2,0).detach().cpu())
            
    plt.setp(ax,xticks=[],yticks=[])
    plt.show()


def model_performance(*args):
    fig,ax = plt.subplots()
    for (label,data) in args:
        sns.lineplot(ax=ax,data=data,label=label)

    ax.set_xticks(np.arange(0,len(data),1))
    plt.show()

## SETTINGS ##

epochs = 10
batch_size = 32
noise_factor = 0.2
imgs_to_visualize = 5


## LOAD DATA ##

trans = T.Compose([T.ToTensor()])
train_data = datasets.CIFAR10(root='./data',transform=trans,train=True,download=False)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_data = datasets.CIFAR10(root='./data',transform=trans,train=False,download=False)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)


## TRAIN ##

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) 
criterion = torch.nn.MSELoss()

train_loss = []
test_loss = []
for epoch in range(epochs):
    print(f'Epoch {epoch}...')
    tr = 0
    te = 0

    for (orig,_) in train_loader:
        optimizer.zero_grad()
        orig,noisy_data = add_noise(orig,noise_factor,device)
        recon = model(noisy_data)
        loss = criterion(recon,noisy_data - orig)
        loss.backward()
        optimizer.step()
        tr += loss.item()*orig.size(0)

    train_loss.append(tr/len(train_loader.sampler))

    with torch.no_grad():
        for (orig,_) in test_loader:
            orig,noisy_data = add_noise(orig,noise_factor,device)
            recon = model(noisy_data)
            loss = criterion(recon,noisy_data-orig)
            te += loss.item()*orig.size(0)

        test_loss.append(te/len(test_loader.sampler))

model_performance(('train',train_loss),('test',test_loss))


## VISUALIZE IMGS ##

orig,_ = next(iter(test_loader))
orig, noisy_data = add_noise(orig,noise_factor,device) 
recon = model(noisy_data)
visualize(orig,noisy_data,noisy_data-recon)