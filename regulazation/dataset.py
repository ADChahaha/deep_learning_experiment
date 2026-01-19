import torch
import numpy as np
from utils import get_data, cfg


class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, path: str, mode: str = "train") -> None:
        self.mode = mode
        self.train_x, self.train_y, self.test_x, self.test_y = get_data(path)
        self.train_x = self.train_x.reshape(-1, 1, cfg.image_height, cfg.image_width)
        self.test_x = self.test_x.reshape(-1, 1, cfg.image_height, cfg.image_width)
        self.train_x = self.train_x / 255.0
        self.test_x = self.test_x / 255.0
        self.train_x = torch.tensor(self.train_x)
        self.train_y = torch.tensor(self.train_y)
        self.test_x = torch.tensor(self.test_x)
        self.test_y = torch.tensor(self.test_y)
        self.len = len(self.train_x)

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.train_x[idx], self.train_y[idx]
        else:
            return self.test_x[idx], self.test_y[idx]

    def __len__(self):
        if self.mode == "train":
            return len(self.train_x)
        else:
            return len(self.test_x)


if __name__ == "__main__":
    dataset = MnistDataset("data")
