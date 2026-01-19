import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, use_dropout: bool = False, use_bn: bool = False) -> None:
        super().__init__()
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.num_classes = 10

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = Model(True, True)
    x = torch.randn(16, 1, 28, 28)
    y = model(x)
    print(y.shape)
