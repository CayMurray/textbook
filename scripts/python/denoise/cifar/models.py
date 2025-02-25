## MODULES ##

import torch
import torch.nn as nn
from torch.nn import functional as F


## CNN ##

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layer_list = [(3,32),(32,16)]
        
        self.encoder = nn.Sequential()
        for (input,output) in layer_list:
            self.encoder.append(nn.Sequential(nn.Conv2d(input,output,kernel_size=3,stride=1,padding=1),
                               nn.ReLU(),
                               nn.BatchNorm2d(output)))

        self.decoder = nn.Sequential()
        for i,(output,input) in enumerate(layer_list[::-1]):
            self.decoder.append(nn.Sequential(nn.ConvTranspose2d(input,output,kernel_size=3,stride=1,padding=1,output_padding=0),
                                              nn.ReLU() if len(layer_list)-i !=1 else nn.Tanh()))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
                

## TEST ##

if __name__ == '__main__':
    tensor = torch.normal(0,1,(10,3,32,32))
    model = CNN()
    print(model)
    x = model(tensor)
    

