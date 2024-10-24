import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)  # 3 фильтра размера (5, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))  # Ядро размера 2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)  # 5 фильтров размера (3, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Ядро размера 2

        self.flatten = nn.Flatten()  # Преобразование в одномерный вектор

        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)  # 100 нейронов на выходе
        self.fc2 = nn.Linear(in_features=100, out_features=10)  # 10 нейронов на выходе


    
    def forward(self, x):
        x = self.conv1(x)  # Применяем первый сверточный слой
        x = F.relu(x)  # Применяем функцию активации ReLU
        x = self.pool1(x)  # Применяем первый слой подвыборки

        x = self.conv2(x)  # Применяем второй сверточный слой
        x = F.relu(x)  # Применяем функцию активации ReLU
        x = self.pool2(x)  # Применяем второй слой подвыборки

        x = self.flatten(x)  # Преобразуем в одномерный вектор

        x = self.fc1(x)  # Применяем первый полносвязный слой
        x = F.relu(x)  # Применяем функцию активации ReLU
        x = self.fc2(x)  # Применяем второй полносвязный слой
        return x
        
def create_model():
    return ConvNet()
