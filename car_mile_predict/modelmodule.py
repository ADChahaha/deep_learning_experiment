import lightning as pl
import torch
import torch.optim as optim
from model import Model 

class MyModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = Model()
        # 损失函数使用 MSELoss 进行回归
        self.criterion = torch.nn.MSELoss()

        # 学习率
        self.lr = lr

        # 用于存储每个 epoch 的损失值
        self.train_losses = []
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        data, label = batch
        data = data.float()  # 确保数据类型为 float32
        label = label.float()  # 确保标签为 float32

        y = self.model(data)
        loss = self.criterion(y, label)

        # 将每个 batch 的损失保存
        self.train_losses.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data = data.float()  # 确保数据类型为 float32
        label = label.float()  # 确保标签为 float32

        y = self.model(data)
        loss = self.criterion(y, label)

        # 保存每个 batch 的损失
        self.val_losses.append(loss)

        return loss

    def on_train_epoch_end(self):
        # 检查训练损失列表是否为空
        if len(self.train_losses) > 0:
            avg_train_loss = torch.stack(self.train_losses).mean()
            # 记录训练损失（只在 epoch 结束时记录）
            self.log("train_loss", avg_train_loss, prog_bar=True, on_epoch=True)

        # 清空存储的损失值
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        # 计算一个 epoch 的验证损失
        if len(self.val_losses) > 0:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.log("val_loss", avg_val_loss, prog_bar=True, on_epoch=True)

        # 清空存储的损失值
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
