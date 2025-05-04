import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 10)

    
    def forward(self, x):
x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def create_model():
    # Эта ячейка не должна выдавать ошибку.
# Если при исполнении ячейки возникает ошибка, то в вашей реализации нейросети есть баги.
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)
# conv1
assert model.conv1.kernel_size == (5, 5), "НЕВЕРНЫЙ РАЗМЕР ЯДРА У conv1"
assert model.conv1.in_channels == 3, "НЕВЕРНЫЙ РАЗМЕР in_channels У conv1"
assert model.conv1.out_channels == 32, "НЕВЕРНЫЙ РАЗМЕР out_channels У conv1"

# pool1
assert model.pool1.kernel_size == (2), "НЕВЕРНЫЙ РАЗМЕР ЯДРА У pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "НЕВЕРНЫЙ РАЗМЕР ЯДРА У conv2"
assert model.conv2.in_channels == 32, "НЕВЕРНЫЙ РАЗМЕР in_channels У conv2"
assert model.conv2.out_channels == 64, "НЕВЕРНЫЙ РАЗМЕР out_channels У conv2"

# pool2
assert model.pool2.kernel_size == (2), "НЕВЕРНЫЙ РАЗМЕР ЯДРА У pool2"

# fc1
assert model.fc1.out_features == 1024, "НЕВЕРНЫЙ РАЗМЕР out_features У fc1"

# fc2
assert model.fc2.out_features == 10, "НЕВЕРНЫЙ РАЗМЕР out_features У fc2"
    return ConvNet()
