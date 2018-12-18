#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:56:58 2018

@author: 3673760
"""
import torch
from torch import nn



class u (nn.Module):
    def __init__(self, inSize, layers=[]):
        super(Pi, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, 1))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)  
            x = self.layers[i](x)            
   
        return x

class Q(nn.Module):
    def __init__(self, inSize, outSize):
        super(V, self).__init__()
        linear1 = nn.Linear(inSize, 20)
        linear1 = nn.Linear(inSize, 20)
        
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, s, a):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)
            x = self.layers[i](x)
        return x

