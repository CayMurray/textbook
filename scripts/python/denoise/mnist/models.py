## MODULES ##

import torch
import torch.nn as nn


## MODELS ##

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784,512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.ReLU(),
                                nn.Linear(64,32),
                                nn.ReLU())
        
        self.decoder = nn.Sequential(nn.Linear(32,64),
                                nn.ReLU(),
                                nn.Linear(64,128),
                                nn.ReLU(),
                                nn.Linear(128,256),
                                nn.ReLU(),
                                nn.Linear(256,512),
                                nn.ReLU(),
                                nn.Linear(512,784),
                                nn.Tanh())
        
    def forward(self,x):
        x = x.view(-1,784)
        latents = self.encoder(x)
        reconstruction = self.decoder(latents)
        return reconstruction
    