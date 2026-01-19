from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import lightning
from dataset import Multi30K
from utils import LoadVocab

class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 1,
        max_seq_len: int = 32,
        train_path: str = "assets/datasets/train/",
        val_path: str = "assets/datasets/valid/",
        de_vocab_path: str = "assets/vocabs/de_vocab.json",
        en_vocab_path: str = "assets/vocabs/en_vocab.json",
    ):
        super().__init__()
        # 训练参数初始化
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

        # 读取vocab
        de_vocab = LoadVocab(de_vocab_path, "")
        en_vocab = LoadVocab(en_vocab_path, "")
        self.de_pad_idx = de_vocab.pad_idx
        self.en_pad_idx = en_vocab.pad_idx
        # 读取dataset
        self.train_dataset, self.val_dataset = Multi30K(train_path, de_vocab, en_vocab), Multi30K(val_path, de_vocab, en_vocab)

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True
        )

    def collate_fn(self, batch):
        """
        batch: list of (seq_tensor, seq_tensor)
        """
        de_seqs, en_seqs = zip(*batch)

        de_truncated_seqs = [seq[: self.max_seq_len] for seq in de_seqs]

        de_lengths = torch.tensor(
            [len(seq) for seq in de_truncated_seqs], dtype=torch.long
        )

        de_padded_seqs = pad_sequence(
            de_truncated_seqs, batch_first=True, padding_value=self.de_pad_idx
        )

        en_truncated_seqs = [seq[: self.max_seq_len] for seq in en_seqs]

        en_lengths = torch.tensor(
            [len(seq) for seq in en_truncated_seqs], dtype=torch.long
        )

        en_padded_seqs = pad_sequence(
            en_truncated_seqs, batch_first=True, padding_value=self.en_pad_idx
        )

        return de_padded_seqs, de_lengths, en_padded_seqs, en_lengths


if __name__ == "__main__":
    datamodule = MyDataModule()
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        de, de_length, en, en_length = batch
        print(de.shape)
        print(de_length.shape)
        print(en.shape)
        print(en_length.shape)
