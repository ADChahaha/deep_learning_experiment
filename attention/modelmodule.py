import lightning as pl
import torch
import torch.optim as optim
import torchmetrics
from model import Model
from utils import LoadVocab


class MyModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        nhead: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        lr: float = 1e-3,
        de_vocab_path: str = "assets/vocabs/de_vocab.json",
        de_idx_vocab_path: str = "assets/vocabs/de_idx_vocab.json",
        en_vocab_path: str = "assets/vocabs/en_vocab.json",
        en_idx_vocab_path: str = "assets/vocabs/en_idx_vocab.json",
    ):
        super().__init__()
        # 加载词汇表
        self.de_vocab = LoadVocab(de_vocab_path, de_idx_vocab_path)
        self.en_vocab = LoadVocab(en_vocab_path, en_idx_vocab_path)

        # 创建模型
        self.model = Model(
            self.de_vocab,
            self.en_vocab,
            embedding_dim,
            nhead,
            encoder_num_layers,
            decoder_num_layers,
        )
        # 损失函数使用 CrossEntropy
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.en_vocab.pad_idx)
        # 学习率
        self.lr = lr

    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx):

        src, src_lengths, tgt, tgt_lengths = batch
        batch_size = len(src_lengths)
        dtype = src.dtype
        device = src.device

        # 增加<bos>
        bos_column = torch.full(
            (batch_size, 1), self.en_vocab.bos_idx, dtype=dtype, device=device
        )
        tgt = torch.cat([bos_column, tgt], dim=1)  # 在列维度拼接
        tgt_lengths = tgt_lengths + 1

        input_seq_len = torch.max(src_lengths).item()
        output_seq_len = torch.max(tgt_lengths).item()

        # 生成 padding mask
        src_idx = (
            torch.arange(0, input_seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, input_seq_len)
        )
        src_key_padding_mask = src_idx >= src_lengths.unsqueeze(1).to(device)
        tgt_idx = (
            torch.arange(0, output_seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, output_seq_len)
        )
        tgt_key_padding_mask = tgt_idx >= tgt_lengths.unsqueeze(1).to(device)

        # 模型前向传播
        logits = self.model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        logits = logits.view(-1, logits.shape[-1])

        # 生成labels (取不含<bos>的部分再加上<eos>)
        eos_column = torch.full(
            (batch_size, 1), self.en_vocab.eos_idx, dtype=dtype, device=device
        )
        labels = torch.cat([tgt[:, 1:], eos_column], dim=1)
        labels = labels.view(-1)

        # 生成loss
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, src_lengths, tgt, tgt_lengths = batch
        batch_size = len(src_lengths)
        dtype = src.dtype
        device = src.device

        # 增加<bos>
        bos_column = torch.full(
            (batch_size, 1), self.en_vocab.bos_idx, dtype=dtype, device=device
        )
        tgt = torch.cat([bos_column, tgt], dim=1)  # 在列维度拼接
        tgt_lengths = tgt_lengths + 1

        input_seq_len = torch.max(src_lengths).item()
        output_seq_len = torch.max(tgt_lengths).item()

        # 生成 padding mask
        src_idx = (
            torch.arange(0, input_seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, input_seq_len)
        )
        src_key_padding_mask = src_idx >= src_lengths.unsqueeze(1).to(device)
        tgt_idx = (
            torch.arange(0, output_seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, output_seq_len)
        )
        tgt_key_padding_mask = tgt_idx >= tgt_lengths.unsqueeze(1).to(device)

        # 模型前向传播
        logits = self.model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        logits = logits.view(-1, logits.shape[-1])

        # 生成labels (取不含<bos>的部分再加上<eos>)
        eos_column = torch.full(
            (batch_size, 1), self.en_vocab.eos_idx, dtype=dtype, device=device
        )
        labels = torch.cat([tgt[:, 1:], eos_column], dim=1)
        labels = labels.view(-1)

        # 生成loss
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    pass
