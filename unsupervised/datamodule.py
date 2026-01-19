from torch.utils.data import DataLoader
import lightning
import torch
from dataset import MnistDataset


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        path: str = "data/MNIST",
        ratio: float = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = path
        self.ratio = ratio
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        # 使用自定义 MnistDataset
        train_dataset = MnistDataset(self.path, mode="train")
        test_dataset = MnistDataset(self.path, mode="test")

        # 划分训练和验证集
        train_size = int(len(train_dataset) * self.ratio)
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
