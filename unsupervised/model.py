import torch
import torch.nn as nn

# 定义自编码器网络
class AutoEncoder_Net(nn.Module):
    def __init__(self, in_channel, embeded_channel=2):
        super(AutoEncoder_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, embeded_channel),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embeded_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, in_channel),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
    
    # 构建模型
    def forward(self, x):
        x = self.flatten(x)
        lowData = self.encoder(x)
        reconData = self.decoder(lowData)
        return x, lowData, reconData