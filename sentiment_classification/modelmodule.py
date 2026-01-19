import lightning as pl
import torch
import torch.optim as optim
import torchmetrics
from model import Model
from utils import load_glove

class MyModel(pl.LightningModule):
    def __init__(
        self,
        embedding_path:str,
        lr: float = 1e-3,
        embed_path: str = "data/glove.6B.100d.txt",
        input_size: int = 100,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        # 损失函数使用 BCELossWithLogitsLoss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.embed_path = embed_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        vocab, embedding_tensor = load_glove(embedding_path)
        # 学习率
        self.lr = lr
        # 数据转换
        self.model = Model(embedding_tensor, vocab["<pad>"], self.input_size, self.hidden_size, self.num_layers)

    def setup(self, stage=None):
        pass
        

    def training_step(self, batch, batch_idx):
        x, lengths, labels = batch

        y = self.model(x, lengths)
        loss = self.criterion(y, labels)

        # 将每个 batch 的损失保存
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, labels = batch

        y = self.model(x, lengths)
        loss = self.criterion(y, labels)

        preds = torch.sigmoid(y) > 0.5
        self.log("val_acc", self.val_acc(preds, labels), prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
   pass