import lightning
import torchmetrics
import torchvision
import torch
from model import OriginNet, OriginNetWithRes


class MyModel(lightning.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = OriginNetWithRes()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

    def training_step(self, batch):
        img, label = batch
        y = self.model(img)
        loss = self.criterion(y, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def validation_step(self, batch):
        img, label = batch
        y = self.model(img)
        loss = self.criterion(y, label)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y, label), prog_bar=True)
