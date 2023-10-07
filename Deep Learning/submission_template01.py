import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(nn.Linear(784, 256),
                           nn.ReLU(),
                           nn.Linear(256, 16),
                           nn.ReLU(), 
                           nn.Linear(16, 10))

    return model

def count_parameters(model):
    cnt = 0
    for i in model.parameters(): 
        if i.requires_grad == False: 
            continue
        cnt += i.numel()
    # верните количество параметров модели model
    return cnt
