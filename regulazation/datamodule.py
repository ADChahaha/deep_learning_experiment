from torch.utils.data import DataLoader
import lightning
from dataset import MnistDataset


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        data_path: str = "data",
    ):
        super().__init__()
        # 训练参数初始化
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.train_dataset = MnistDataset(self.data_path, "train")
        # 随机分成ratio / 1 - ratio 比例作为训练集和测试集
        self.val_dataset = MnistDataset(self.data_path, "test")

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True
        )
    



if __name__ == "__main__":
    datamodule = MyDataModule()
    print(datamodule.train_dataset.dataset[1])
