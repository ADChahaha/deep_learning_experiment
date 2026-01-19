import torch.nn as nn


# 定义CNN图像识别网络
class OriginNet(nn.Module):
    def __init__(
        self, num_class=5, channel=3 
    ):  # 一共分五类，图片通道数是3
        super(OriginNet, self).__init__()
        self.num_class = num_class
        self.channel = channel
        # 设置卷积层
        self.conv1 = nn.Conv2d(
            self.channel,
            32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        # 设置ReLU激活函数
        self.relu = nn.ReLU()
        # 设置最大池化层
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 6 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_class)
        self.cell_out = []

    # 构建模型
    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv3(x)
        x = self.max_pool2d(x)

        x = self.conv4(x)
        x = self.max_pool2d(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# 定义CNN图像识别网络
class OriginNetWithRes(nn.Module):
    def __init__(
        self, num_class=5, channel=3 
    ):  # 一共分五类，图片通道数是3
        super(OriginNetWithRes, self).__init__()
        self.num_class = num_class
        self.channel = channel
        # 设置卷积层
        self.conv1 = nn.Conv2d(
            self.channel,
            32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.convshortcut1 = nn.Conv2d(
            self.channel,
            32,
            1
        )
        # 设置ReLU激活函数
        self.relu = nn.ReLU()
        # 设置最大池化层
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.convshortcut2 = nn.Conv2d(
            32,
            64,
            1
        )
        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.convshortcut3 = nn.Conv2d(
            64,
            128,
            1
        )
        self.conv4 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.convshortcut4 = nn.Conv2d(
            128,
            128,
            1
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 6 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_class)
        self.cell_out = []

    # 构建模型
    def forward(self, x):
        # Block 1: conv1 + relu，残差连接后再池化
        identity = self.convshortcut1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = x + identity
        x = self.max_pool2d(x)

        # Block 2: conv2 + relu，残差连接后再池化
        identity = self.convshortcut2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + identity
        x = self.max_pool2d(x)

        # Block 3: conv3，残差连接后再池化
        identity = self.convshortcut3(x)
        x = self.conv3(x)
        x = x + identity
        x = self.max_pool2d(x)

        # Block 4: conv4，残差连接后再池化
        identity = self.convshortcut4(x)
        x = self.conv4(x)
        x = x + identity
        x = self.max_pool2d(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x