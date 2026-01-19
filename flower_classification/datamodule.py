from torch.utils.data import DataLoader
import torchvision
import lightning
import torch
import torchvision.transforms as transforms


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        path: str = "data/flower_photos",
        ratio: float = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = path
        self.ratio = ratio

    def setup(self, stage):
        origin_dataset = torchvision.datasets.ImageFolder(self.path, transform=transforms.Compose(
            [transforms.Resize((96, 96)),
             transforms.ToTensor()]
        ))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            origin_dataset, (self.ratio, 1 - self.ratio)
        )

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
