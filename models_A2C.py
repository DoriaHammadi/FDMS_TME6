#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:56:58 2018

@author: 3673760
"""
import torch
from torch import nn



class Pi(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(Pi, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)  
            x = self.layers[i](x)            
            
        x = nn.functional.softmax(x)
        
        return x

class V(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(V, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)
            x = self.layers[i](x)
        return x

