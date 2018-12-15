import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    
    def __init__(self, inSize, outSize, layers=[]):
        self.outSize = outSize
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
            print("************************************************", self.outSize, x.shape)
            
        return x