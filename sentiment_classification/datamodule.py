from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch
import lightning
from dataset import MovieDataset
from torch.utils.data import random_split
from utils import load_glove


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        embedding_path: str,
        batch_size: int = 32,
        num_workers: int = 1,
        max_seq_len: int = 500,
        ratio: float = 0.8,
        data_path: str = "data/aclImdb_v1.tar.gz",
    ):
        super().__init__()
        # 训练参数初始化
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.data_path = data_path
        # vocab初始化
        self.vocab, _ = load_glove(embedding_path)
        self.unk_idx = self.vocab["<unk>"]
        self.pad_idx = self.vocab["<pad>"]
        # 读取dataset
        origin_dataset = MovieDataset(self.data_path, self.vocab, mode="train")
        # 随机分成ratio / 1 - ratio 比例作为训练集和测试集
        self.train_dataset, self.val_dataset = random_split(
            origin_dataset, [ratio, 1 - ratio]
        )

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        """
        batch: list of (seq_tensor, label_tensor)
        """
        seqs, labels = zip(*batch)

        truncated_seqs = [seq[: self.max_seq_len] for seq in seqs]

        lengths = torch.tensor([len(seq) for seq in truncated_seqs], dtype=torch.long)

        padded_seqs = pad_sequence(
            truncated_seqs, batch_first=True, padding_value=self.pad_idx
        )

        labels = torch.stack(labels).float()

        return padded_seqs, lengths, labels


if __name__ == "__main__":
    datamodule = MyDataModule()
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        data, length, label = batch
        print(data)
