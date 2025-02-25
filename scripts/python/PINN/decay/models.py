## MODULES ##

import torch
import torch.nn as nn


## PINN ##

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1,32),nn.ReLU(),
                                    nn.Linear(32,32),nn.ReLU(),
                                    nn.Linear(32,32),nn.Sigmoid(),
                                    nn.Linear(32,1))
        
    def forward(self,x):
        x = self.layers(x)
        return x