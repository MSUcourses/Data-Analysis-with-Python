import torch
from torch import nn
from torch.nn import functional as F  # 修正拼写错误

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()  # 正确缩进
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 输入通道3，输出通道16
        self.pool = nn.MaxPool2d(2, 2)                           # 池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)                    # 修正变量名：fcl → fc1
        self.fc2 = nn.Linear(256, 10)                            # 输出10个类别

    def forward(self, x):
        # 前向传播流程
        x = self.pool(F.relu(self.conv1(x)))   # 输出形状: [batch, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # 输出形状: [batch, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)            # 展平为 [batch, 2048]
        x = F.relu(self.fc1(x))               # 全连接层1 → [batch, 256]
        x = self.fc2(x)                       # 全连接层2 → [batch, 10]
        return x

def create_model():
    return ConvNet()
