import lightning as pl
import torch
import torch.optim as optim
import torchmetrics
from model import Model


class MyModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        use_dropout: bool = False,
        use_bn: bool = True,
        weight_decay: float = 0,
    ):
        super().__init__()
        # 损失函数使用 CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        # 学习率
        self.lr = lr
        # 模型
        self.model = Model(use_dropout=use_dropout, use_bn=use_bn)
        # 参数
        self.weight_decay = weight_decay

    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx):
        x, labels = batch

        y = self.model(x)
        loss = self.criterion(y, labels)

        # 将每个 batch 的损失保存
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.val_acc(y, labels), prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        y = self.model(x)
        loss = self.criterion(y, labels)

        self.log("val_acc", self.val_acc(y, labels), prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


if __name__ == "__main__":
    pass
