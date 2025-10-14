import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,16),
            nn.ReLU(),
            nn.Linear(16,10)
    )    
    return model

def count_parameters(model):
    return sun(p.numel()for p in
modle.parameters())               
   # return integer number (None is just a placeholder)
