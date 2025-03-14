import numpy as np
import torch
from torch import nn

def create_model():
    # submission_template01.py
import torch.nn as nn

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    return model
    # your code here
    # return model instance (None is just a placeholder)

    return None

def count_parameters(model):
    # submission_template01.py
import torch

def create_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 10)
    )
    return model

def count_parameters(model):
    """
    Подсчет общего количества параметров в модели.
    
    :param model:torch.nn.Module - модель, для которой нужно подсчитать параметры
    :return: int - общее количество параметров в модели
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # your code here
    # return integer number (None is just a placeholder)
    
    return None
