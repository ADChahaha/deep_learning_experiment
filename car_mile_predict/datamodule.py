from torch.utils.data import DataLoader
import torch
import lightning
from dataset import CarDataset


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        path: str = "",
        ratio: float = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = path
        self.ratio = ratio
        origin_dataset = CarDataset(self.path)
        origin_dataset_len = len(origin_dataset)
        train_dataset_len = int(self.ratio * origin_dataset_len)
        val_dataset_len = origin_dataset_len - train_dataset_len
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            origin_dataset,
            (train_dataset_len, val_dataset_len)
        )

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
